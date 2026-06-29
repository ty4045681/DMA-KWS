"""Checkpoint I/O helpers shared across training stages."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from dma_kws.training.checkpoint_avg import average_lightning_checkpoints


def extract_state_dict(checkpoint: dict[str, Any]) -> dict[str, torch.Tensor]:
    """Extract a model state dict from a Lightning or custom checkpoint."""
    if "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    if "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    if "model" in checkpoint and isinstance(checkpoint["model"], dict):
        return checkpoint["model"]
    return checkpoint


def export_model_pt(
    model: torch.nn.Module,
    output_path: Path,
    *,
    config: dict[str, Any],
    dict_path: Path,
    vocab_size: int,
    step: int,
    blank_id: int | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Save model weights in the Stage I/II ``.pt`` checkpoint format."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "config": config,
        "step": step,
        "tokenizer_dict_path": str(dict_path),
        "vocab_size": vocab_size,
    }
    if blank_id is not None:
        payload["blank_id"] = blank_id
    if extra:
        payload.update(extra)
    torch.save(payload, output_path)
    return output_path


def select_and_average_checkpoints(
    checkpoint_dir: Path,
    *,
    last_k: int,
    pattern: str = "*.ckpt",
    output_name: str = "avg_10.ckpt",
) -> Path | None:
    """Average the last ``last_k`` checkpoints matching ``pattern``."""
    candidates = sorted(checkpoint_dir.glob(pattern))
    if not candidates:
        print(f"No checkpoints matched {pattern!r} in {checkpoint_dir}; skipping average.")
        return None
    selected = candidates[-last_k:]
    output_path = checkpoint_dir / output_name
    average_lightning_checkpoints(selected, output_path)
    print(f"Averaged {len(selected)} checkpoints -> {output_path}")
    return output_path
