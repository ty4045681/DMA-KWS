"""Shared Stage II QbyT training entry point."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]

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

    return DataLoader(
        val_dataset,
        batch_size=eval_paths["batch_size"],
        shuffle=False,
        num_workers=eval_paths["num_workers"],
        collate_fn=test_collate_fn,
        drop_last=True,
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
    from dma_kws.training.checkpoint_callback import build_stage2_checkpoint_callback
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

    dict_path = Path(tokenizer_cfg["dict_path"])
    if not dict_path.is_absolute():
        dict_path = PROJECT_ROOT / dict_path
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
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=train_collate_fn,
        drop_last=True,
    )

    val_dataloader = _build_val_dataloader(config, tokenizer)

    resume_checkpoint = args.resume_checkpoint or stage2.get("resume_checkpoint", "")
    init_checkpoint = args.init_checkpoint or stage2.get("init_checkpoint", "")
    freeze_encoder = bool(stage2.get("freeze_encoder", False))

    if resume_checkpoint:
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

    limit_steps = args.limit_steps or None
    checkpoint_dir = Path(stage2.get("checkpoint_dir", Path(paths["exp_root"]) / "stage2_qbyt" / "checkpoints"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    accelerator = "gpu" if args.device != "cpu" and torch.cuda.is_available() else "cpu"
    devices = max(1, int(args.devices)) if accelerator == "gpu" else 1

    log_dir = stage2.get("log_dir", Path(paths["exp_root"]) / "stage2_qbyt" / "logs")
    loggers = build_loggers(log_dir, str(stage2.get("run_name", "stage2_qbyt")))

    recipe = str(training.get("recipe", ""))
    checkpoint_callback = build_stage2_checkpoint_callback(config, recipe)

    trainer_kwargs = build_trainer_kwargs(config, devices, limit_steps=limit_steps)
    trainer = pl.Trainer(
        accelerator=accelerator,
        callbacks=[checkpoint_callback],
        logger=loggers,
        **trainer_kwargs,
    )
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

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
