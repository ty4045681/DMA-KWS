"""PyTorch Lightning Trainer kwargs factory for Stage II DDP training."""

from __future__ import annotations

from typing import Any


def build_trainer_kwargs(
    config: dict[str, Any],
    devices: int,
    limit_steps: int | None = None,
) -> dict[str, Any]:
    """Build keyword arguments for ``pytorch_lightning.Trainer`` from config.

    Reads ``stage2`` fields: strategy, accumulate_grad_batches, gradient_clip_val,
    val_check_interval, max_steps, and log_interval. Uses DDP when ``strategy`` is
    ``ddp`` and ``devices`` > 1; otherwise ``auto``.
    """
    stage2 = config.get("stage2", {})
    validation = stage2.get("validation", {}) or {}

    strategy = stage2.get("strategy", "auto")
    if strategy == "ddp" and devices > 1:
        trainer_strategy: str | Any = "ddp"
    else:
        trainer_strategy = "auto"

    max_steps = limit_steps if limit_steps else int(stage2.get("max_steps", 50000))
    val_check_interval = int(
        validation.get("val_check_interval", stage2.get("val_check_interval", 1000))
    )

    return {
        "devices": devices,
        "strategy": trainer_strategy,
        "max_steps": max_steps,
        "accumulate_grad_batches": int(stage2.get("accumulate_grad_batches", 1)),
        "gradient_clip_val": float(stage2.get("gradient_clip_val", 1.0)),
        "val_check_interval": val_check_interval,
        "log_every_n_steps": int(stage2.get("log_interval", 10)),
    }
