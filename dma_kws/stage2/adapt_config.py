"""Shared, lightweight configuration contracts for keyword adaptation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


_METHOD_LABELS = {
    "lora": "LoRA",
    "qbyt_full": "Full QbyT",
    "encoder_qbyt_full": "Encoder + QbyT",
}


def resolve_adapt_method(adapt: Mapping[str, Any]) -> str:
    """Resolve the trainable parameter scope, preserving legacy LoRA defaults."""
    method = str(adapt.get("method", "lora")).strip().casefold()
    if method not in _METHOD_LABELS:
        raise ValueError(
            f"adapt.method must be one of {tuple(_METHOD_LABELS)}, got {adapt.get('method')!r}"
        )
    return method


def is_full_adapt_method(method: str) -> bool:
    """Whether adaptation updates base weights and exports a complete model."""
    return resolve_adapt_method({"method": method}) != "lora"


def adapt_method_label(method: str) -> str:
    """User-facing name shared by training and sweep summaries."""
    return _METHOD_LABELS[resolve_adapt_method({"method": method})]
