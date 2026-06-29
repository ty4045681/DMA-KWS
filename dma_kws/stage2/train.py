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
        drop_last=True,
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
    from dma_kws.runlog import build_loggers
    from dma_kws.stage2.collate import train_collate_fn
    from dma_kws.stage2.dataset import LibriPhraseTrainDataset
    from dma_kws.stage2.module import Stage2LightningModule
    from dma_kws.tokenizer import load_char_tokenizer
    from dma_kws.training import resolve_resume_path
    from dma_kws.training.callbacks import build_stage2_callbacks, print_run_summary
    from dma_kws.training.ddp import build_trainer_kwargs

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

    train_dataset = LibriPhraseTrainDataset(
        parquet_file=parquet_file,
        wav_dir=wav_dir,
        tokenizer=tokenizer,
        negative_ratio=int(stage2.get("negative_ratio", 1)),
        hard_negative_ratio=int(stage2.get("hard_negative_ratio", 1)),
        sample_lens=int(stage2.get("sample_lens", 5000)),
        seed=seed,
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
        **loader_kwargs,
    )

    val_dataloader = _build_val_dataloader(config, tokenizer)

    resume_checkpoint = args.resume_checkpoint or stage2.get("resume_checkpoint", "")
    init_checkpoint = args.init_checkpoint or stage2.get("init_checkpoint", "")
    freeze_encoder = bool(stage2.get("freeze_encoder", False))

    limit_steps = args.limit_steps or None
    checkpoint_dir = Path(stage2.get("checkpoint_dir", Path(paths["exp_root"]) / "stage2_qbyt" / "checkpoints"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    resume_path = resolve_resume_path(args.resume_from, checkpoint_dir)

    if resume_path is not None:
        if resume_checkpoint or init_checkpoint:
            print(
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

    log_dir = stage2.get("log_dir", Path(paths["exp_root"]) / "stage2_qbyt" / "logs")
    run_name = str(stage2.get("run_name", "stage2_qbyt"))
    loggers = build_loggers(log_dir, run_name, config=config)

    recipe = str(training.get("recipe", ""))
    callbacks = build_stage2_callbacks(config, recipe)

    trainer_kwargs = build_trainer_kwargs(
        config,
        devices,
        limit_steps=limit_steps,
        accelerator=accelerator,
    )

    param_counts = {
        "encoder": sum(p.numel() for p in model.encoder.parameters()),
        "qbyt": sum(p.numel() for p in model.qbyt.parameters()),
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
            "parquet": parquet_file,
            "wav_dir": wav_dir,
            "checkpoint_dir": checkpoint_dir,
            "log_dir": log_dir,
        },
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
    ckpt_path = checkpoint_dir / f"stage2_step{global_step:06d}.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config,
            "step": global_step,
            "tokenizer_dict_path": str(dict_path),
            "vocab_size": vocab_size,
        },
        ckpt_path,
    )
    print(f"Saved {ckpt_path}")
