"""DataLoader helper factories for training."""

from __future__ import annotations

from typing import Any


def build_loader_kwargs(num_workers: int, dataloader_cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    """Build optional DataLoader kwargs from ``stage2.dataloader`` config."""
    cfg = dataloader_cfg or {}
    kwargs: dict[str, Any] = {
        "pin_memory": bool(cfg.get("pin_memory", True)),
    }
    if num_workers > 0:
        if cfg.get("persistent_workers", True):
            kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = int(cfg.get("prefetch_factor", 4))
    return kwargs
