"""Optimizer and learning-rate scheduler factories."""

from __future__ import annotations

from typing import Any


def build_optimizer_config(stage2: dict[str, Any]) -> dict[str, Any]:
    """Return plain optimizer/scheduler settings from a ``stage2`` config section."""
    return {
        "optimizer": str(stage2.get("optimizer", "adam")).lower(),
        "lr": float(stage2.get("learning_rate", 1e-3)),
        "weight_decay": float(stage2.get("weight_decay", 0.0)),
        "warmup_steps": int(stage2.get("warmup_steps", 2500)),
        "total_steps": int(stage2.get("total_scheduler_steps", stage2.get("max_steps", 50000))),
    }


def build_cosine_warmup_optimizer(
    module: Any,
    lr_or_stage2: float | dict[str, Any],
    warmup_steps: int | None = None,
    total_steps: int | None = None,
) -> dict:
    """Build Adam/AdamW + cosine warmup scheduler config for Lightning.

    Supports the legacy signature ``(module, lr, warmup_steps, total_steps)`` used
    by Stage I, and the Stage II form ``(module, stage2_dict)``.
    """
    import torch
    from transformers import get_cosine_schedule_with_warmup

    if isinstance(lr_or_stage2, dict):
        cfg = build_optimizer_config(lr_or_stage2)
    else:
        cfg = {
            "optimizer": "adam",
            "lr": float(lr_or_stage2),
            "weight_decay": 0.0,
            "warmup_steps": int(warmup_steps or 0),
            "total_steps": int(total_steps or 0),
        }

    optimizer_name = cfg["optimizer"]
    if optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(
            module.parameters(),
            lr=cfg["lr"],
            weight_decay=cfg["weight_decay"],
        )
    elif optimizer_name == "adam":
        optimizer = torch.optim.Adam(module.parameters(), lr=cfg["lr"])
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name!r} (expected adam or adamw)")

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=cfg["warmup_steps"],
        num_training_steps=cfg["total_steps"],
    )
    return {
        "optimizer": optimizer,
        "lr_scheduler": {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1,
        },
    }
