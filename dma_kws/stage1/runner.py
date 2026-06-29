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
from dma_kws.runlog import build_loggers
from dma_kws.stage1.dataset import Stage1Dataset, stage1_collate_fn
from dma_kws.stage1.module import Stage1LightningModule
from dma_kws.tokenizer import load_char_tokenizer
from dma_kws.training.callbacks import build_stage1_callbacks
from dma_kws.training.checkpoint_io import export_model_pt, select_and_average_checkpoints
from dma_kws.training.ddp import build_trainer_kwargs
from dma_kws.training.device import resolve_accelerator_and_devices
from dma_kws.training.resume import resolve_resume_path

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
    batch_size = int(stage1.get("batch_size_per_gpu", 16)) * max(1, devices)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=stage1_collate_fn,
    )

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

    checkpoint_dir = Path(
        stage1.get("checkpoint_dir", Path(paths["exp_root"]) / "stage1_phoneme_ctc" / "checkpoints")
    )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    callbacks, checkpoint_callback = build_stage1_callbacks(
        config,
        checkpoint_dir=checkpoint_dir,
        has_validation=dev_dataloader is not None,
    )

    log_dir = stage1.get("log_dir", Path(paths["exp_root"]) / "stage1_phoneme_ctc" / "logs")
    loggers = build_loggers(log_dir, str(stage1.get("run_name", "stage1_phoneme_ctc")))

    trainer_kwargs = build_trainer_kwargs(
        config,
        devices,
        section="stage1",
        limit_steps=args.limit_steps or None,
        accelerator=accelerator,
    )

    trainer = pl.Trainer(
        accelerator=accelerator,
        logger=loggers,
        callbacks=callbacks,
        **trainer_kwargs,
    )

    resume_path = resolve_resume_path(args.resume_from, checkpoint_dir)

    if dev_dataloader is not None:
        trainer.fit(model, dataloader, dev_dataloader, ckpt_path=resume_path)
    else:
        trainer.fit(model, dataloader, ckpt_path=resume_path)

    if checkpoint_callback is not None and checkpoint_callback.best_model_path:
        best_state = torch.load(checkpoint_callback.best_model_path, map_location="cpu")
        model.load_state_dict(best_state["state_dict"])
        print(f"Loaded best Stage I weights from {checkpoint_callback.best_model_path}")

    avg_cfg = stage1.get("checkpoint_avg", {}) or {}
    avg_path = None
    if avg_cfg.get("enabled", False):
        avg_path = select_and_average_checkpoints(
            checkpoint_dir,
            last_k=int(avg_cfg.get("last_k", 10)),
            pattern=str(avg_cfg.get("pattern", "*.ckpt")),
            output_name=str(avg_cfg.get("output_name", "avg_10.ckpt")),
        )

    global_step = int(trainer.global_step)
    ckpt_path = export_stage1_encoder_pt(
        model,
        checkpoint_dir / f"stage1_step{global_step:06d}.pt",
        config=config,
        dict_path=dict_path,
        vocab_size=vocab_size,
        blank_id=blank_id,
        step=global_step,
    )
    print(f"Saved {ckpt_path}")

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
        )
        print(f"Saved averaged Stage I encoder weights to {avg_pt_path}")
        print(f"Use {avg_path} or {avg_pt_path} as Stage II init checkpoint")
