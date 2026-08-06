"""Shared Stage II QbyT training entry point."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dma_kws.pathing import PROJECT_ROOT, resolve_dict_path
from dma_kws.training.device import resolve_accelerator_and_devices
from dma_kws.training.loaders import build_loader_kwargs

_EVAL_MISSING_MSG = (
    "LibriPhrase eval data is required for Stage II validation. "
    "Download LibriPhrase eval CSVs and set stage2.eval.test_dir in config, "
    "or set paths.libriphrase460_root (or libriphrase100_root / libriphrase_root)."
)


@dataclass
class Stage2TrainArgs:
    """Runtime options for Stage II training."""

    init_checkpoint: str = ""
    resume_checkpoint: str = ""
    resume_from: str = ""
    device: str = "cuda"
    devices: int = 1
    limit_steps: int = 0


def _resolve_path(stage2: dict[str, Any], key: str, default: Path) -> Path:
    raw = stage2.get(key, "")
    if raw:
        return Path(raw)
    return default


def _build_val_dataloader(config: dict[str, Any], tokenizer: Any) -> Any:
    """Build LibriPhrase hard/easy eval dataloader for training validation."""
    from torch.utils.data import DataLoader

    from dma_kws.config import get_tokenizer_config
    from dma_kws.stage2.collate import test_collate_fn
    from dma_kws.stage2.dataset import LibriPhraseEvalDataset, resolve_stage2_eval_paths

    stage2 = config["stage2"]
    eval_cfg = stage2.get("eval", {}) or {}
    split = eval_cfg.get("split", "hard")

    try:
        eval_paths = resolve_stage2_eval_paths(config)
    except ValueError as exc:
        raise SystemExit(f"{_EVAL_MISSING_MSG} ({exc})") from exc

    test_dir = eval_paths["test_dir"]
    if not test_dir.exists():
        raise SystemExit(f"{_EVAL_MISSING_MSG} Eval directory not found: {test_dir}")

    tokenizer_cfg = get_tokenizer_config(config)
    val_dataset = LibriPhraseEvalDataset(
        test_dir=test_dir,
        fbank_dir=eval_paths["fbank_dir"],
        split=split,
        csv_files=eval_paths["csv_files"],
        aggregate_csv=eval_paths["aggregate_csv"],
        tokenizer=tokenizer,
        split_with_space=tokenizer_cfg.get("split_with_space", " "),
    )
    if len(val_dataset) == 0:
        raise SystemExit(
            f"{_EVAL_MISSING_MSG} Eval split {split!r} is empty under {test_dir}."
        )

    num_workers = eval_paths["num_workers"]
    loader_kwargs = build_loader_kwargs(num_workers, stage2.get("dataloader", {}) or {})

    return DataLoader(
        val_dataset,
        batch_size=eval_paths["batch_size"],
        shuffle=False,
        num_workers=num_workers,
        collate_fn=test_collate_fn,
        # AUC/EER must cover the complete validation split. Dropping the final
        # partial batch silently changes the evaluated population (and can
        # remove an entire class on small smoke sets).
        drop_last=False,
        **loader_kwargs,
    )


def run_stage2_training(config: dict[str, Any], args: Stage2TrainArgs) -> None:
    """Train Stage II QbyT from a loaded config dict."""
    try:
        import torch
        import pytorch_lightning as pl
        from torch.utils.data import DataLoader
    except ImportError as exc:
        raise SystemExit(
            "Missing torch/pytorch-lightning. Install CUDA PyTorch on the training machine first."
        ) from exc

    from dma_kws.config import get_tokenizer_config, require_sections
    from dma_kws.runlog import build_loggers, logger_backend_names
    from dma_kws.stage2.collate import train_collate_fn
    from dma_kws.stage2.dataset import LibriPhraseTrainDataset, stage2_worker_init_fn
    from dma_kws.stage2.module import Stage2LightningModule
    from dma_kws.stage2.objective import (
        assert_sequence_objective_matches,
        resolve_sequence_objective,
    )
    from dma_kws.tokenizer import load_char_tokenizer
    from dma_kws.training.callbacks import (
        build_stage2_callbacks,
        print_run_summary,
        print_training_result_summary,
    )
    from dma_kws.training.checkpoint_io import (
        restore_best_checkpoint_weights,
        stamp_qbyt_readout_version,
    )
    from dma_kws.training.ddp import apply_step_based_validation, build_trainer_kwargs
    from dma_kws.training.ddp import rank_zero_print
    from dma_kws.training.metrics_history import (
        append_wide_row,
        build_metrics_history_callback,
        build_run_record,
        collect_hparams,
        numeric_callback_metrics,
    )
    from dma_kws.training.run_context import build_run_context, stamp_run_context
    from dma_kws.training.run_context_callback import RunContextCheckpointCallback
    from dma_kws.training.resume import resolve_versioned_resume_path

    require_sections(config, ["paths", "stage1", "stage2", "tokenizer", "training"])
    paths = config["paths"]
    stage1 = config["stage1"]
    stage2 = config["stage2"]
    training = config["training"]
    tokenizer_cfg = get_tokenizer_config(config)

    processed_root = Path(paths["processed_root"])
    feature_root = Path(paths.get("feature_root", processed_root))

    parquet_file = _resolve_path(
        stage2,
        "parquet_file",
        processed_root / "stage2_qbyt" / "aggregated_segments_with_g2p_distance.parquet",
    )
    wav_dir = _resolve_path(stage2, "wav_dir", feature_root / "fbank")

    if not parquet_file.exists():
        raise SystemExit(f"Stage II parquet not found: {parquet_file}")

    dict_path = resolve_dict_path(config)
    tokenizer = load_char_tokenizer(dict_path, split_with_space=tokenizer_cfg.get("split_with_space", " "))
    vocab_size = len(tokenizer._symbol_table)

    seed = int(training.get("seed", 2025))
    pl.seed_everything(seed, workers=True)
    sequence_objective = resolve_sequence_objective(stage2)

    train_dataset = LibriPhraseTrainDataset(
        parquet_file=parquet_file,
        wav_dir=wav_dir,
        tokenizer=tokenizer,
        negative_ratio=int(stage2.get("negative_ratio", 1)),
        hard_negative_ratio=int(stage2.get("hard_negative_ratio", 1)),
        sample_lens=int(stage2.get("sample_lens", 5000)),
        seed=seed,
        seq_label_mode=sequence_objective.target_mode,
    )

    batch_size = int(stage2.get("batch_size_per_gpu", 64))
    num_workers = int(stage2.get("num_workers", stage1.get("num_workers", 2)))
    loader_kwargs = build_loader_kwargs(num_workers, stage2.get("dataloader", {}) or {})
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=train_collate_fn,
        drop_last=True,
        # Without this the dataset's RNG is forked in an identical state into every
        # worker, so all of them draw the same negatives and clips.
        worker_init_fn=stage2_worker_init_fn,
        **loader_kwargs,
    )

    val_dataloader = _build_val_dataloader(config, tokenizer)

    resume_checkpoint = args.resume_checkpoint or stage2.get("resume_checkpoint", "")
    init_checkpoint = args.init_checkpoint or stage2.get("init_checkpoint", "")
    freeze_encoder = bool(stage2.get("freeze_encoder", False))

    limit_steps = args.limit_steps or None
    checkpoint_root = Path(
        stage2.get(
            "checkpoint_dir",
            Path(paths["exp_root"]) / "stage2_qbyt" / "checkpoints",
        )
    )
    log_dir = stage2.get(
        "log_dir", Path(paths["exp_root"]) / "stage2_qbyt" / "logs"
    )
    run_name = str(stage2.get("run_name", "stage2_qbyt"))
    resume_path = resolve_versioned_resume_path(
        args.resume_from,
        checkpoint_root,
        run_name,
    )

    resume_payload = None
    if resume_path is not None:
        resume_payload = torch.load(resume_path, map_location="cpu")
        assert_sequence_objective_matches(
            resume_payload,
            stage2,
            source=resume_path,
        )

    run_context = build_run_context(
        config,
        section="stage2",
        log_dir=log_dir,
        run_name=run_name,
        limit_steps=limit_steps,
        resume_from=resume_path,
        resume_checkpoint=resume_payload,
    )
    checkpoint_dir = checkpoint_root / run_context.run_id
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    if resume_path is not None:
        if resume_checkpoint or init_checkpoint:
            rank_zero_print(
                "Resuming full training state via ckpt_path; ignoring "
                "init_checkpoint/resume_checkpoint weight init (ckpt_path restores full state)."
            )
        model = Stage2LightningModule(
            config,
            vocab_size=vocab_size,
            freeze_encoder=freeze_encoder,
        )
    elif resume_checkpoint:
        model = Stage2LightningModule.load_from_checkpoint(
            resume_checkpoint,
            config=config,
            vocab_size=vocab_size,
            freeze_encoder=freeze_encoder,
        )
    else:
        model = Stage2LightningModule(
            config,
            vocab_size=vocab_size,
            freeze_encoder=freeze_encoder,
            init_checkpoint=init_checkpoint or None,
        )

    accelerator, devices = resolve_accelerator_and_devices(args.device, args.devices)

    if accelerator == "gpu":
        torch.set_float32_matmul_precision("high")

    loggers = build_loggers(
        log_dir,
        run_name,
        config=config,
        section="stage2",
        version=run_context.version,
    )

    hparams = collect_hparams(
        config,
        effective_max_steps=run_context.effective_max_steps,
        extra={**run_context.identity(), "checkpoint_dir": str(checkpoint_dir)},
    )
    for train_logger in loggers:
        train_logger.log_hyperparams(hparams)

    recipe = str(training.get("recipe", ""))
    callbacks = build_stage2_callbacks(
        config,
        recipe,
        checkpoint_dir=checkpoint_dir,
    )
    checkpoint_callback = callbacks[0]
    callbacks.append(RunContextCheckpointCallback(run_context))
    history_callback = build_metrics_history_callback(
        run_name=run_name,
        run_id=run_context.run_id,
        default_dir=run_context.run_dir,
    )
    # History must update before ModelCheckpoint serializes callback state at
    # validation end, otherwise a resume starts one validation behind.
    callbacks.insert(0, history_callback)

    trainer_kwargs = build_trainer_kwargs(
        config,
        devices,
        limit_steps=limit_steps,
        accelerator=accelerator,
    )
    # Keep Lightning's integer interval on one cross-epoch train-batch clock.
    # This also avoids validating it against the unsharded DataLoader length
    # before DDP replaces the sampler.
    apply_step_based_validation(trainer_kwargs, len(train_dataloader), force=True)

    param_counts = {
        "encoder": sum(p.numel() for p in model.encoder.parameters()),
        "adapter": (
            sum(p.numel() for p in model.adapter.parameters())
            if model.adapter is not None
            else 0
        ),
        "qbyt": sum(p.numel() for p in model.qbyt.parameters()),
        "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "frozen": sum(p.numel() for p in model.parameters() if not p.requires_grad),
        "total": sum(p.numel() for p in model.parameters()),
    }
    print_run_summary(
        config=config,
        devices=devices,
        accelerator=accelerator,
        train_samples=len(train_dataset),
        val_samples=len(val_dataloader.dataset),
        param_counts=param_counts,
        paths={
            "run_id": run_context.run_id,
            "parquet": parquet_file,
            "wav_dir": wav_dir,
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
        callbacks=callbacks,
        logger=loggers,
        **trainer_kwargs,
    )
    trainer.fit(
        model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
        ckpt_path=resume_path,
    )

    global_step = int(trainer.global_step)

    # With externally launched DDP every process returns from ``fit`` and keeps
    # executing this function. Keep the entire artifact transaction on rank 0:
    # concurrent CSV appends and torch.save calls can otherwise duplicate rows
    # or corrupt a checkpoint while ranks overwrite the same path.
    runs_csv = Path(paths["exp_root"]) / "stage2_qbyt" / "runs.csv"
    if trainer.is_global_zero:
        artifact_step, artifact_source = restore_best_checkpoint_weights(
            model,
            checkpoint_callback,
            final_step=global_step,
        )
        ckpt_path = checkpoint_dir / f"stage2_step{artifact_step:06d}.pt"
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
                    "primary_artifact_path": str(ckpt_path),
                },
            ),
        )
        torch.save(
            stamp_run_context(
                stamp_qbyt_readout_version(
                    {
                        "model_state_dict": model.state_dict(),
                        "config": model._checkpoint_config,
                        "step": artifact_step,
                        "tokenizer_dict_path": str(dict_path),
                        "vocab_size": vocab_size,
                    }
                ),
                run_context,
            ),
            ckpt_path,
        )
        print_training_result_summary(
            run_context=run_context,
            global_step=global_step,
            last_validation_step=history_callback.last_validation_step,
            best_checkpoint_monitor=getattr(checkpoint_callback, "monitor", None),
            best_checkpoint_path=getattr(
                checkpoint_callback, "best_model_path", None
            ),
            best_checkpoint_score=getattr(
                checkpoint_callback, "best_model_score", None
            ),
            final_metrics=final_metrics,
            artifact_paths={
                "final_checkpoint": ckpt_path,
                "eval_history": history_callback.csv_path,
                "runs_csv": runs_csv,
            },
            artifact_sources={
                "final_checkpoint": artifact_source
            },
            title="Stage II QbyT Training Result",
            rich=bool((stage2.get("console", {}) or {}).get("rich", True)),
        )

    # Do not let a non-zero rank return (or a following phase start) while rank
    # 0 is still writing the shared run record and final checkpoint.
    trainer.strategy.barrier("stage2_training_artifacts_saved")
