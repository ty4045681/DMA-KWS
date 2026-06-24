"""Optimizer and learning-rate scheduler factories."""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import get_cosine_schedule_with_warmup


def build_cosine_warmup_optimizer(
    module: nn.Module,
    lr: float,
    warmup_steps: int,
    total_steps: int,
) -> dict:
    """Build Adam + cosine warmup scheduler config for Lightning ``configure_optimizers``."""
    optimizer = torch.optim.Adam(module.parameters(), lr=lr)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    return {
        "optimizer": optimizer,
        "lr_scheduler": {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1,
        },
    }
