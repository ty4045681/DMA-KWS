"""Shared path helpers for DMA-KWS."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Mapping

from dma_kws.config import get_tokenizer_config

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def ensure_qbyt_on_path() -> Path:
    """Ensure vendored ``qbyt/`` is importable; return its root path."""
    qbyt_root = PROJECT_ROOT / "qbyt"
    qbyt_path = str(qbyt_root)
    if qbyt_path not in sys.path:
        sys.path.insert(0, qbyt_path)
    return qbyt_root


def resolve_dict_path(config: Mapping[str, Any], *, project_root: Path | None = None) -> Path:
    """Resolve tokenizer dict path from a loaded config."""
    root = project_root or PROJECT_ROOT
    tokenizer_cfg = get_tokenizer_config(dict(config))
    dict_path = Path(tokenizer_cfg["dict_path"])
    if not dict_path.is_absolute():
        dict_path = root / dict_path
    return dict_path
