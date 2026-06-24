#!/usr/bin/env python3
"""Evaluate Stage II QbyT on LibriPhrase hard/easy splits (AUC/EER)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dma_kws.config import get_tokenizer_config, load_config, require_sections
from dma_kws.stage2.collate import test_collate_fn
from dma_kws.stage2.dataset import LibriPhraseEvalDataset, resolve_stage2_eval_paths
from dma_kws.stage2.module import Stage2LightningModule, _extract_state_dict
from dma_kws.tokenizer import load_char_tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--checkpoint", required=True, help="Lightning .ckpt or .pt checkpoint")
    parser.add_argument(
        "--split",
        choices=["easy", "hard", "all"],
        default="hard",
        help="LibriPhrase eval split (default: hard)",
    )
    parser.add_argument("--device", default="cuda", help="cuda or cpu")
    parser.add_argument("--batch-size", type=int, default=0, help="Override eval batch size")
    return parser.parse_args()


def _resolve_dict_path(tokenizer_cfg: dict) -> Path:
    dict_path = Path(tokenizer_cfg["dict_path"])
    if not dict_path.is_absolute():
        dict_path = PROJECT_ROOT / dict_path
    return dict_path


def _load_model(config: dict, checkpoint_path: Path, vocab_size: int) -> Stage2LightningModule:
    import torch

    model = Stage2LightningModule(config, vocab_size=vocab_size)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if checkpoint_path.suffix == ".pt":
        state = _extract_state_dict(checkpoint)
    else:
        state = checkpoint.get("state_dict", checkpoint)

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"Warning: missing keys when loading checkpoint: {len(missing)}")
    if unexpected:
        print(f"Warning: unexpected keys when loading checkpoint: {len(unexpected)}")
    return model


def evaluate(args: argparse.Namespace) -> dict[str, float | str]:
    try:
        import torch
        import pytorch_lightning as pl
        from torch.utils.data import DataLoader
    except ImportError as exc:
        raise SystemExit(
            "Missing torch/pytorch-lightning. Install CUDA PyTorch on the eval machine first."
        ) from exc

    config = load_config(args.config)
    require_sections(config, ["paths", "stage1", "stage2", "tokenizer"])

    tokenizer_cfg = get_tokenizer_config(config)
    dict_path = _resolve_dict_path(tokenizer_cfg)
    tokenizer = load_char_tokenizer(dict_path, split_with_space=tokenizer_cfg.get("split_with_space", " "))
    vocab_size = len(tokenizer._symbol_table)

    eval_paths = resolve_stage2_eval_paths(config)
    batch_size = args.batch_size or eval_paths["batch_size"]

    dataset = LibriPhraseEvalDataset(
        test_dir=eval_paths["test_dir"],
        split=args.split,
        csv_files=eval_paths["csv_files"],
        aggregate_csv=eval_paths["aggregate_csv"],
        tokenizer=tokenizer,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=eval_paths["num_workers"],
        collate_fn=test_collate_fn,
    )

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise SystemExit(f"Checkpoint not found: {checkpoint_path}")

    model = _load_model(config, checkpoint_path, vocab_size)

    accelerator = "gpu" if args.device != "cpu" and torch.cuda.is_available() else "cpu"
    trainer = pl.Trainer(accelerator=accelerator, devices=1, logger=False, enable_checkpointing=False)
    results = trainer.test(model, dataloaders=dataloader, verbose=False)

    metrics = results[0] if results else {}
    auc = float(metrics.get("test/auc", 0.0))
    eer = float(metrics.get("test/eer", 0.0))

    output = {"split": args.split, "auc": auc, "eer": eer}
    print(json.dumps(output))
    return output


def main() -> None:
    evaluate(parse_args())


if __name__ == "__main__":
    main()
