"""Accelerator resolution for Lightning trainers."""

from __future__ import annotations

import torch


def resolve_accelerator(device: str) -> tuple[str, int]:
    """Return ``(accelerator, devices)`` for Lightning given a device preference."""
    if device != "cpu" and torch.cuda.is_available():
        return "gpu", 1
    return "cpu", 1


def resolve_accelerator_and_devices(device: str, devices: int) -> tuple[str, int]:
    """Return ``(accelerator, devices)`` honoring a requested GPU count."""
    if device != "cpu" and torch.cuda.is_available():
        return "gpu", max(1, int(devices))
    return "cpu", 1
