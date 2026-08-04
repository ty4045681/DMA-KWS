"""PyTorch Lightning Trainer kwargs factory for Stage II DDP training."""

from __future__ import annotations

import os
from typing import Any


def process_rank() -> int:
    """Return the initialized or launcher-provided global process rank."""
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            return int(dist.get_rank())
    except (ImportError, RuntimeError):
        pass

    for name in ("RANK", "SLURM_PROCID", "LOCAL_RANK"):
        if name in os.environ:
            return int(os.environ[name])
    return 0


def rank_zero_print(*values: Any, **kwargs: Any) -> None:
    """Use ordinary ``print`` once under both initialized and launcher DDP."""
    if process_rank() == 0:
        print(*values, **kwargs)


def resolve_precision(stage2: dict[str, Any], accelerator: str) -> str:
    """Resolve Lightning ``precision`` from config and accelerator.

    GPU defaults to ``bf16-mixed``; CPU falls back to ``32-true``. An explicit
    ``stage2.precision`` value always wins.
    """
    explicit = stage2.get("precision")
    if explicit:
        return str(explicit)
    if accelerator == "cpu":
        return "32-true"
    return "bf16-mixed"


def apply_step_based_validation(
    trainer_kwargs: dict[str, Any],
    batches_per_epoch: int,
    *,
    force: bool = False,
) -> dict[str, Any]:
    """Make an integer ``val_check_interval`` count train batches across epochs.

    Lightning reads an integer ``val_check_interval`` as a batch index *inside* one
    epoch and raises if it exceeds the epoch length. Virtual epochs (``sample_lens``
    divided by the batch size) are often much shorter than the configured interval,
    so switch Lightning to a cross-epoch train-batch counter instead of failing or
    validating at every short epoch boundary. This does *not* turn the unit into
    optimizer steps: with gradient accumulation ``k``, ``N`` train batches are
    approximately ``N / k`` optimizer steps. ``force=True`` also avoids validating
    against the pre-DDP DataLoader length before Lightning shards it. Mutates and
    returns ``trainer_kwargs``.
    """
    interval = trainer_kwargs.get("val_check_interval")
    if not isinstance(interval, int) or isinstance(interval, bool):
        return trainer_kwargs
    if force or (batches_per_epoch > 0 and interval > batches_per_epoch):
        trainer_kwargs["check_val_every_n_epoch"] = None
    return trainer_kwargs


def build_trainer_kwargs(
    config: dict[str, Any],
    devices: int,
    *,
    section: str = "stage2",
    limit_steps: int | None = None,
    accelerator: str = "gpu",
    max_epochs: int | None = None,
) -> dict[str, Any]:
    """Build keyword arguments for ``pytorch_lightning.Trainer`` from a config section."""
    stage = config.get(section, {})
    validation = stage.get("validation", {}) or {}

    if section == "stage1":
        configured_steps = int(stage.get("max_train_steps", 0))
        max_steps = limit_steps or configured_steps
        kwargs: dict[str, Any] = {
            "devices": devices,
            "strategy": "auto",
            "max_epochs": max_epochs if max_epochs is not None else int(stage.get("max_epochs", 1)),
            "max_steps": max_steps if max_steps else -1,
            "gradient_clip_val": float(stage.get("gradient_clip_val", 1.0)),
            "log_every_n_steps": int(stage.get("log_interval", 10)),
            "precision": resolve_precision(stage, accelerator),
        }
        if validation:
            kwargs["check_val_every_n_epoch"] = int(validation.get("check_val_every_n_epoch", 1))
        return kwargs

    configured_strategy = str(stage.get("strategy", "auto") or "auto")
    find_unused_parameters = bool(stage.get("find_unused_parameters", False))
    if devices > 1 and find_unused_parameters:
        trainer_strategy: str | Any = "ddp_find_unused_parameters_true"
    elif devices > 1 and configured_strategy != "auto":
        trainer_strategy = configured_strategy
    else:
        trainer_strategy = "auto"

    max_steps = limit_steps if limit_steps else int(stage.get("max_steps", 50000))
    val_check_interval = int(
        validation.get("val_check_interval", stage.get("val_check_interval", 1000))
    )

    return {
        "devices": devices,
        "strategy": trainer_strategy,
        "max_steps": max_steps,
        "accumulate_grad_batches": int(stage.get("accumulate_grad_batches", 1)),
        "gradient_clip_val": float(stage.get("gradient_clip_val", 1.0)),
        "val_check_interval": val_check_interval,
        "log_every_n_steps": int(stage.get("log_interval", 10)),
        "precision": resolve_precision(stage, accelerator),
    }
