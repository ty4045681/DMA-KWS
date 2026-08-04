"""Step A training runner: phoneme CTC adapter on a frozen encoder."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dma_kws.pathing import resolve_dict_path
from dma_kws.training.loaders import build_loader_kwargs

BLANK_ID = 0


@dataclass
class PhonemeAdapterTrainArgs:
    """Runtime options for Step A training."""

    train_manifest: str = ""
    dev_manifest: str = ""
    init_checkpoint: str = ""
    device: str = "cuda"
    devices: int = 1
    limit_steps: int = 0
    resume_from: str = ""


def _resolve_manifest(override: str, config_value: str, default: Path) -> Path:
    if override:
        return Path(override)
    if config_value:
        return Path(config_value)
    return default


def resolve_val_check_interval(adapter_cfg: dict[str, Any]) -> int:
    """Interval ``build_trainer_kwargs`` will actually use (``validation`` wins)."""
    validation_cfg = adapter_cfg.get("validation", {}) or {}
    return int(
        validation_cfg.get("val_check_interval", adapter_cfg.get("val_check_interval", 1000))
    )


def build_adapter_callbacks(checkpoint_dir: Path, adapter_cfg: dict[str, Any]) -> tuple[list, Any]:
    from dma_kws.training.checkpoint_callback import FreshValidationModelCheckpoint

    checkpoint_cfg = adapter_cfg.get("checkpoint", {}) or {}
    every_n_train_steps = int(checkpoint_cfg.get("every_n_train_steps", 0))
    val_check_interval = resolve_val_check_interval(adapter_cfg)
    monitor = str(checkpoint_cfg.get("monitor", "val/per"))
    mode = str(checkpoint_cfg.get("mode", "min"))

    if monitor != "val/per" or mode != "min":
        raise SystemExit(
            "phoneme-adapter checkpoint selection currently requires "
            "checkpoint.monitor='val/per' and checkpoint.mode='min'."
        )

    if val_check_interval <= 0:
        raise SystemExit(
            "phoneme_adapter.validation.val_check_interval must be positive when "
            "checkpointing monitors val/per."
        )

    # Saving from a train-batch hook can see ``val/per`` left over from the
    # previous validation.  Save only in on_validation_end, at the first fresh
    # validation that satisfies the requested optimizer-step spacing.
    checkpoint_callback = FreshValidationModelCheckpoint(
        fresh_every_n_train_steps=every_n_train_steps,
        dirpath=str(checkpoint_dir),
        monitor=monitor,
        mode=mode,
        save_top_k=int(checkpoint_cfg.get("save_top_k", 3)),
        filename="adapter_{step:07d}_{val_per:.4f}",
        save_last=True,
    )
    return [checkpoint_callback], checkpoint_callback


def run_phoneme_adapter_training(config: dict[str, Any], args: PhonemeAdapterTrainArgs) -> None:
    """Train the phoneme CTC adapter from a loaded config dict."""
    try:
        import pytorch_lightning as pl
        import torch
        from torch.utils.data import DataLoader
    except ImportError as exc:
        raise SystemExit(
            "Missing torch/pytorch-lightning. Install CUDA PyTorch on the training machine first."
        ) from exc

    from dma_kws.config import (
        fbank_kwargs,
        get_fbank_config,
        get_tokenizer_config,
        require_sections,
    )
    from dma_kws.phoneme_adapter.lightning import PhonemeAdapterCtcModule
    from dma_kws.runlog import build_loggers, logger_backend_names
    from dma_kws.stage1.dataset import Stage1Dataset, stage1_collate_fn
    from dma_kws.stage2.fbank import FbankExtractor
    from dma_kws.tokenizer import load_char_tokenizer
    from dma_kws.training.checkpoint_io import export_model_pt
    from dma_kws.training.callbacks import (
        build_console_callbacks,
        print_run_summary,
        print_training_result_summary,
    )
    from dma_kws.training.ddp import apply_step_based_validation, build_trainer_kwargs
    from dma_kws.training.device import resolve_accelerator_and_devices
    from dma_kws.training.resume import resolve_versioned_resume_path
    from dma_kws.training.metrics_history import (
        append_wide_row,
        build_metrics_history_callback,
        build_run_record,
        collect_hparams,
        numeric_callback_metrics,
    )
    from dma_kws.training.run_context import RUN_CONTEXT_KEY, build_run_context
    from dma_kws.training.run_context_callback import RunContextCheckpointCallback

    require_sections(config, ["paths", "stage1", "phoneme_adapter", "tokenizer", "training"])
    paths = config["paths"]
    stage1 = config["stage1"]
    adapter_cfg = config["phoneme_adapter"]
    training = config["training"]

    processed_dir = Path(paths["processed_root"]) / "stage1_phoneme_ctc"
    train_manifest = _resolve_manifest(
        args.train_manifest,
        str(adapter_cfg.get("train_manifest", "")).strip(),
        processed_dir / "train.jsonl",
    )
    dev_manifest = _resolve_manifest(
        args.dev_manifest,
        str(adapter_cfg.get("dev_manifest", "")).strip(),
        processed_dir / "dev.jsonl",
    )
    if not train_manifest.exists():
        raise SystemExit(f"Step A train manifest not found: {train_manifest}")

    dict_path = resolve_dict_path(config)
    tokenizer_cfg = get_tokenizer_config(config)
    tokenizer = load_char_tokenizer(
        dict_path, split_with_space=tokenizer_cfg.get("split_with_space", " ")
    )
    vocab_size = len(tokenizer._symbol_table)
    blank_id = int(tokenizer.symbol_table.get("<blank>", BLANK_ID))

    # The trunk is trained to feed Stage II, so it must see Stage II's feature
    # space. dma_kws.audio.extract_fbank is not that space: it skips the 1<<15
    # scaling FbankExtractor applies, on top of differing dither/snip_edges.
    extractor = FbankExtractor(**fbank_kwargs(get_fbank_config(config)))
    sample_rate = int(stage1.get("sample_rate", 16000))
    num_mel_bins = int(stage1.get("input_dim", 80))

    pl.seed_everything(int(training.get("seed", 2025)), workers=True)

    def _dataset(manifest: Path) -> Stage1Dataset:
        return Stage1Dataset(
            manifest,
            tokenizer=tokenizer,
            sample_rate=sample_rate,
            num_mel_bins=num_mel_bins,
            feature_extractor=extractor,
        )

    train_dataset = _dataset(train_manifest)
    if len(train_dataset) == 0:
        raise SystemExit(f"No records found in {train_manifest}")

    num_workers = int(adapter_cfg.get("num_workers", 4))
    loader_kwargs = build_loader_kwargs(num_workers, adapter_cfg.get("dataloader", {}) or {})
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=int(adapter_cfg.get("batch_size_per_gpu", 32)),
        shuffle=True,
        num_workers=num_workers,
        collate_fn=stage1_collate_fn,
        drop_last=True,
        **loader_kwargs,
    )

    validation_cfg = adapter_cfg.get("validation", {}) or {}
    dev_dataloader = None
    if dev_manifest.exists():
        dev_dataset = _dataset(dev_manifest)
        if len(dev_dataset) > 0:
            dev_dataloader = DataLoader(
                dev_dataset,
                batch_size=int(
                    validation_cfg.get("batch_size", adapter_cfg.get("batch_size_per_gpu", 32))
                ),
                shuffle=False,
                num_workers=num_workers,
                collate_fn=stage1_collate_fn,
                **loader_kwargs,
            )
    if dev_dataloader is None:
        raise SystemExit(
            f"Step A dev manifest not found or empty: {dev_manifest}. Dev PER is the whole "
            "point of this stage (it decides whether the frozen representation can carry "
            "phonemes at all), so it is required rather than optional."
        )

    init_checkpoint = args.init_checkpoint or str(adapter_cfg.get("init_checkpoint", ""))
    model = PhonemeAdapterCtcModule(
        config,
        vocab_size=vocab_size,
        blank_id=blank_id,
        init_checkpoint=init_checkpoint or None,
    )

    accelerator, devices = resolve_accelerator_and_devices(args.device, args.devices)
    if accelerator == "gpu":
        torch.set_float32_matmul_precision("high")

    checkpoint_root = Path(
        adapter_cfg.get("checkpoint_dir", Path(paths["exp_root"]) / "phoneme_adapter" / "checkpoints")
    )
    log_dir = adapter_cfg.get(
        "log_dir", Path(paths["exp_root"]) / "phoneme_adapter" / "logs"
    )
    run_name = str(adapter_cfg.get("run_name", "phoneme_adapter_ctc"))
    resume_path = resolve_versioned_resume_path(
        args.resume_from,
        checkpoint_root,
        run_name,
    )
    resume_payload = (
        torch.load(resume_path, map_location="cpu")
        if resume_path is not None
        else None
    )

    run_context = build_run_context(
        config,
        section="phoneme_adapter",
        log_dir=log_dir,
        run_name=run_name,
        limit_steps=args.limit_steps or None,
        resume_from=resume_path,
        resume_checkpoint=resume_payload,
    )
    checkpoint_dir = checkpoint_root / run_context.run_id
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    callbacks, checkpoint_callback = build_adapter_callbacks(
        checkpoint_dir,
        adapter_cfg,
    )
    callbacks.extend(build_console_callbacks(config, section="phoneme_adapter"))
    loggers = build_loggers(
        log_dir,
        run_name,
        config=config,
        section="phoneme_adapter",
        version=run_context.version,
    )
    callbacks.append(RunContextCheckpointCallback(run_context))
    hparams = collect_hparams(
        config,
        section="phoneme_adapter",
        effective_max_steps=run_context.effective_max_steps,
        extra={**run_context.identity(), "checkpoint_dir": str(checkpoint_dir)},
    )
    for train_logger in loggers:
        train_logger.log_hyperparams(hparams)
    history_callback = build_metrics_history_callback(
        run_name=run_name,
        run_id=run_context.run_id,
        default_dir=run_context.run_dir,
    )
    callbacks.insert(0, history_callback)

    trainer_kwargs = build_trainer_kwargs(
        config,
        devices,
        section="phoneme_adapter",
        limit_steps=args.limit_steps or None,
        accelerator=accelerator,
    )
    # Lightning reads an integer val_check_interval as a batch index inside one
    # epoch and raises when it exceeds the epoch length, which a small manifest
    # or a limit_steps smoke run hits immediately.
    apply_step_based_validation(trainer_kwargs, len(train_dataloader), force=True)
    print_run_summary(
        config=config,
        devices=devices,
        accelerator=accelerator,
        section="phoneme_adapter",
        train_samples=len(train_dataset),
        val_samples=len(dev_dataset),
        param_counts={
            "frozen_encoder": sum(
                param.numel() for param in model.encoder.parameters()
            ),
            "adapter": sum(param.numel() for param in model.adapter.parameters()),
            "trainable": sum(
                param.numel() for param in model.parameters() if param.requires_grad
            ),
            "frozen": sum(
                param.numel() for param in model.parameters() if not param.requires_grad
            ),
            "total": sum(param.numel() for param in model.parameters()),
        },
        paths={
            "run_id": run_context.run_id,
            "train_manifest": train_manifest,
            "dev_manifest": dev_manifest,
            "checkpoint_dir": checkpoint_dir,
            "log_dir": log_dir,
            "run_dir": run_context.run_dir,
            **(
                {"resume_from": run_context.resume_from}
                if run_context.resume_from
                else {}
            ),
            **(
                {"parent_run_id": run_context.parent_run_id}
                if run_context.parent_run_id
                else {}
            ),
        },
        effective_max_steps=run_context.effective_max_steps,
        effective_logging_backends=logger_backend_names(loggers),
    )
    trainer = pl.Trainer(
        accelerator=accelerator,
        logger=loggers,
        callbacks=callbacks,
        **trainer_kwargs,
    )

    trainer.fit(model, train_dataloader, dev_dataloader, ckpt_path=resume_path)

    global_step = int(trainer.global_step)
    if trainer.is_global_zero:
        artifact_step = global_step
        artifact_source = f"final_weights@step={global_step}"
        best_path_value = (
            checkpoint_callback.best_model_path
            if checkpoint_callback is not None
            else ""
        )
        best_path = Path(best_path_value) if best_path_value else None
        if best_path is not None and best_path.is_file():
            best_state = torch.load(best_path, map_location="cpu")
            model.load_state_dict(best_state["state_dict"])
            artifact_step = int(best_state.get("global_step", global_step))
            artifact_source = f"best_checkpoint@step={artifact_step}:{best_path}"
            print(f"Loaded best adapter weights from {best_path}")
        elif best_path is not None:
            print(
                f"Best adapter checkpoint is unavailable at {best_path}; "
                "exporting the final in-memory weights instead."
            )

        # Only the adapter is exported: Stage II rebuilds the encoder from its
        # own config, and the payload carries the config so
        # assert_stream_policy_matches can reject a Stage II run at a different
        # operating point.
        output_path = export_model_pt(
            model.adapter,
            checkpoint_dir
            / (
                f"adapter_best_step{artifact_step:06d}.pt"
                if artifact_source.startswith("best_checkpoint")
                else f"adapter_final_step{artifact_step:06d}.pt"
            ),
            config=config,
            dict_path=dict_path,
            vocab_size=vocab_size,
            step=artifact_step,
            blank_id=blank_id,
            extra={RUN_CONTEXT_KEY: run_context.as_dict()},
        )
        runs_csv = Path(paths["exp_root"]) / "phoneme_adapter" / "runs.csv"
        final_metrics = numeric_callback_metrics(dict(trainer.callback_metrics))
        append_wide_row(
            runs_csv,
            build_run_record(
                run_name=run_name,
                hparams=hparams,
                final_metrics=final_metrics,
                best_metrics=history_callback.best,
                global_step=global_step,
                duration_seconds=history_callback.duration_seconds,
                metric_step=history_callback.last_validation_step,
                best_steps=history_callback.best_steps,
                identity=run_context.identity(),
                provenance={
                    "metrics_source": "last_trainer_state",
                    "primary_artifact_source": artifact_source,
                    "primary_artifact_step": artifact_step,
                    "primary_artifact_path": str(output_path),
                },
            ),
        )
        print_training_result_summary(
            run_context=run_context,
            global_step=global_step,
            last_validation_step=history_callback.last_validation_step,
            best_checkpoint_monitor=(
                checkpoint_callback.monitor
                if checkpoint_callback is not None
                else None
            ),
            best_checkpoint_path=(
                checkpoint_callback.best_model_path
                if checkpoint_callback is not None
                else None
            ),
            best_checkpoint_score=(
                checkpoint_callback.best_model_score
                if checkpoint_callback is not None
                else None
            ),
            final_metrics=final_metrics,
            artifact_paths={
                "adapter": output_path,
                "eval_history": history_callback.csv_path,
                "runs_csv": runs_csv,
            },
            artifact_sources={"adapter": artifact_source},
            title="Phoneme Adapter Training Result",
            rich=bool((adapter_cfg.get("console", {}) or {}).get("rich", True)),
        )

    trainer.strategy.barrier("phoneme_adapter_artifacts_saved")
