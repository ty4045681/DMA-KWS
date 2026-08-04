"""Stage I CTC training runner."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from dma_kws.config import get_tokenizer_config, require_sections
from dma_kws.pathing import resolve_dict_path
from dma_kws.runlog import build_loggers, logger_backend_names
from dma_kws.stage1.dataset import Stage1Dataset, stage1_collate_fn
from dma_kws.stage1.module import Stage1LightningModule
from dma_kws.tokenizer import load_char_tokenizer
from dma_kws.training.callbacks import (
    build_stage1_callbacks,
    print_run_summary,
    print_training_result_summary,
)
from dma_kws.training.checkpoint_io import export_model_pt, select_and_average_checkpoints
from dma_kws.training.ddp import build_trainer_kwargs
from dma_kws.training.device import resolve_accelerator_and_devices
from dma_kws.training.metrics_history import (
    append_wide_row,
    build_metrics_history_callback,
    build_run_record,
    collect_hparams,
    numeric_callback_metrics,
)
from dma_kws.training.resume import resolve_versioned_resume_path
from dma_kws.training.run_context import RUN_CONTEXT_KEY, build_run_context
from dma_kws.training.run_context_callback import RunContextCheckpointCallback

BLANK_ID = 0


@dataclass
class Stage1TrainArgs:
    """Runtime options for Stage I training."""

    train_manifest: str = ""
    dev_manifest: str = ""
    device: str = "cuda"
    devices: int = 1
    limit_steps: int = 0
    resume_from: str = ""


def _resolve_manifest(
    override: str,
    default: Path,
    *,
    config_value: str = "",
) -> Path:
    if override:
        return Path(override)
    if config_value:
        return Path(config_value)
    return default


def export_stage1_encoder_pt(
    model: Stage1LightningModule,
    output_path: Path,
    *,
    config: dict[str, Any],
    dict_path: Path,
    vocab_size: int,
    blank_id: int,
    step: int,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Save Stage I weights in a format loadable by Stage II ``init_checkpoint``."""
    return export_model_pt(
        model,
        output_path,
        config=config,
        dict_path=dict_path,
        vocab_size=vocab_size,
        step=step,
        blank_id=blank_id,
        extra=extra,
    )


def run_stage1_training(config: dict[str, Any], args: Stage1TrainArgs) -> None:
    """Train Stage I CTC from a loaded config dict."""
    require_sections(config, ["paths", "stage1", "tokenizer", "training"])

    paths = config["paths"]
    stage1 = config["stage1"]
    training = config["training"]

    processed_dir = Path(paths["processed_root"]) / "stage1_phoneme_ctc"
    validation_cfg = stage1.get("validation", {}) or {}

    train_manifest = _resolve_manifest(
        args.train_manifest,
        processed_dir / "train.jsonl",
    )
    dev_manifest = _resolve_manifest(
        args.dev_manifest,
        processed_dir / "dev.jsonl",
        config_value=str(validation_cfg.get("dev_manifest", "")).strip(),
    )

    dict_path = resolve_dict_path(config)
    split_with_space = get_tokenizer_config(config).get("split_with_space", " ")
    tokenizer = load_char_tokenizer(dict_path, split_with_space=split_with_space)
    vocab_size = len(tokenizer._symbol_table)
    blank_id = int(tokenizer.symbol_table.get("<blank>", BLANK_ID))

    sample_rate = int(stage1.get("sample_rate", 16000))
    num_mel_bins = int(stage1.get("input_dim", 80))
    num_decode_batches = int(validation_cfg.get("num_decode_batches", 0))
    num_workers = int(stage1.get("num_workers", 2))
    fbank_root = stage1.get("fbank_root", "")
    if not fbank_root:
        fbank_root = str(Path(paths.get("feature_root", "")) / "stage1_fbank")
    audio_root = stage1.get("audio_root", paths.get("librispeech_root", ""))

    pl.seed_everything(int(training.get("seed", 2025)), workers=True)

    dataset = Stage1Dataset(
        train_manifest,
        tokenizer=tokenizer,
        sample_rate=sample_rate,
        num_mel_bins=num_mel_bins,
        fbank_root=fbank_root or None,
        audio_root=audio_root or None,
    )
    if len(dataset) == 0:
        raise SystemExit(f"No records found in {train_manifest}")

    accelerator, devices = resolve_accelerator_and_devices(args.device, args.devices)
    # Lightning's DDP sampler shards samples, not the DataLoader batch itself.
    # Each rank must therefore receive the configured per-device batch unchanged.
    batch_size = int(stage1.get("batch_size_per_gpu", 16))
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=stage1_collate_fn,
    )

    dev_dataset = None
    dev_dataloader = None
    if dev_manifest.exists():
        dev_dataset = Stage1Dataset(
            dev_manifest,
            tokenizer=tokenizer,
            sample_rate=sample_rate,
            num_mel_bins=num_mel_bins,
            fbank_root=fbank_root or None,
            audio_root=audio_root or None,
        )
        if len(dev_dataset) > 0:
            dev_dataloader = DataLoader(
                dev_dataset,
                batch_size=int(validation_cfg.get("batch_size", stage1.get("batch_size_per_gpu", 16))),
                shuffle=False,
                num_workers=num_workers,
                collate_fn=stage1_collate_fn,
            )

    model = Stage1LightningModule(
        config,
        vocab_size=vocab_size,
        blank_id=blank_id,
        num_decode_batches=num_decode_batches,
    )

    checkpoint_root = Path(
        stage1.get("checkpoint_dir", Path(paths["exp_root"]) / "stage1_phoneme_ctc" / "checkpoints")
    )
    log_dir = stage1.get(
        "log_dir", Path(paths["exp_root"]) / "stage1_phoneme_ctc" / "logs"
    )
    run_name = str(stage1.get("run_name", "stage1_phoneme_ctc"))
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
        section="stage1",
        log_dir=log_dir,
        run_name=run_name,
        limit_steps=args.limit_steps or None,
        resume_from=resume_path,
        resume_checkpoint=resume_payload,
    )
    checkpoint_dir = checkpoint_root / run_context.run_id
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    callbacks, checkpoint_callback = build_stage1_callbacks(
        config,
        checkpoint_dir=checkpoint_dir,
        has_validation=dev_dataloader is not None,
    )
    loggers = build_loggers(
        log_dir,
        run_name,
        config=config,
        section="stage1",
        version=run_context.version,
    )
    callbacks.append(RunContextCheckpointCallback(run_context))
    hparams = collect_hparams(
        config,
        section="stage1",
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
        section="stage1",
        limit_steps=args.limit_steps or None,
        accelerator=accelerator,
    )

    print_run_summary(
        config=config,
        devices=devices,
        accelerator=accelerator,
        section="stage1",
        train_samples=len(dataset),
        val_samples=len(dev_dataset) if dev_dataset is not None else 0,
        param_counts={
            "encoder": sum(param.numel() for param in model.encoder.parameters()),
            "ctc": sum(param.numel() for param in model.ctc.parameters()),
            "trainable": sum(
                param.numel() for param in model.parameters() if param.requires_grad
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

    if dev_dataloader is not None:
        trainer.fit(model, dataloader, dev_dataloader, ckpt_path=resume_path)
    else:
        trainer.fit(model, dataloader, ckpt_path=resume_path)

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
            print(f"Loaded best Stage I weights from {best_path}")
        elif best_path is not None:
            print(
                f"Best Stage I checkpoint is unavailable at {best_path}; "
                "exporting the final in-memory weights instead."
            )

        avg_cfg = stage1.get("checkpoint_avg", {}) or {}
        avg_path = None
        if avg_cfg.get("enabled", False):
            avg_path = select_and_average_checkpoints(
                checkpoint_dir,
                last_k=int(avg_cfg.get("last_k", 10)),
                pattern=str(avg_cfg.get("pattern", "*.ckpt")),
                output_name=str(avg_cfg.get("output_name", "avg_10.ckpt")),
            )

        ckpt_path = export_stage1_encoder_pt(
            model,
            checkpoint_dir
            / (
                f"stage1_best_step{artifact_step:06d}.pt"
                if artifact_source.startswith("best_checkpoint")
                else f"stage1_final_step{artifact_step:06d}.pt"
            ),
            config=config,
            dict_path=dict_path,
            vocab_size=vocab_size,
            blank_id=blank_id,
            step=artifact_step,
            extra={RUN_CONTEXT_KEY: run_context.as_dict()},
        )
        avg_pt_path = None
        if avg_path is not None:
            avg_state = torch.load(avg_path, map_location="cpu")
            model.load_state_dict(avg_state["state_dict"])
            avg_pt_path = export_stage1_encoder_pt(
                model,
                checkpoint_dir / "stage1_avg.pt",
                config=config,
                dict_path=dict_path,
                vocab_size=vocab_size,
                blank_id=blank_id,
                step=global_step,
                extra={RUN_CONTEXT_KEY: run_context.as_dict()},
            )
        runs_csv = Path(paths["exp_root"]) / "stage1_phoneme_ctc" / "runs.csv"
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
                "encoder": ckpt_path,
                "averaged_encoder": avg_pt_path,
                "averaged_checkpoint": avg_path,
                "eval_history": history_callback.csv_path,
                "runs_csv": runs_csv,
            },
            artifact_sources={
                "encoder": artifact_source,
                **(
                    {"averaged_encoder": f"averaged_checkpoint:{avg_path}"}
                    if avg_pt_path is not None
                    else {}
                ),
                **(
                    {"averaged_checkpoint": "last_k_checkpoint_average"}
                    if avg_path is not None
                    else {}
                ),
            },
            title="Stage I Phoneme CTC Training Result",
            rich=bool((stage1.get("console", {}) or {}).get("rich", True)),
        )

    trainer.strategy.barrier("stage1_training_artifacts_saved")
