"""Shared path helpers for DMA-KWS."""

from __future__ import annotations

import importlib.util
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


def load_qbyt_class() -> type[Any]:
    """Load ``QbyT`` from vendored ``qbyt/model.py`` without module-name collisions.

    This avoids ``from model import QbyT`` ambiguity when external repos (e.g. icefall)
    also expose a top-level ``model`` module on ``sys.path``.
    """
    qbyt_root = ensure_qbyt_on_path()
    qbyt_path = str(qbyt_root)

    # Keep qbyt first for imports inside qbyt/model.py such as "from models...".
    if sys.path and sys.path[0] != qbyt_path:
        try:
            sys.path.remove(qbyt_path)
        except ValueError:
            pass
        sys.path.insert(0, qbyt_path)

    model_path = qbyt_root / "model.py"
    if not model_path.exists():
        raise SystemExit(f"QbyT model file not found: {model_path}")

    module_name = "_dma_kws_qbyt_model"
    module = sys.modules.get(module_name)
    if module is None:
        spec = importlib.util.spec_from_file_location(module_name, model_path)
        if spec is None or spec.loader is None:
            raise SystemExit(f"Failed to load spec for {model_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

    qbyt_cls = getattr(module, "QbyT", None)
    if qbyt_cls is None:
        raise SystemExit(f"QbyT class not found in {model_path}")
    return qbyt_cls


def ensure_icefall_on_path() -> Path:
    """Ensure icefall Zipformer modules are importable.
    
    Requires ICEFALL_ROOT environment variable pointing to icefall repo root.
    Adds ``${ICEFALL_ROOT}/egs/gigaspeech/KWS/zipformer`` to sys.path.
    
    Returns:
        Path to zipformer recipe directory
    
    Raises:
        SystemExit: If ICEFALL_ROOT not set or path doesn't exist
    """
    import os
    
    icefall_root = os.getenv("ICEFALL_ROOT")
    if not icefall_root:
        raise SystemExit(
            "ICEFALL_ROOT environment variable not set. "
            "Please set it to your icefall repository root, e.g.: "
            "export ICEFALL_ROOT=/path/to/icefall"
        )
    
    icefall_root = Path(icefall_root)
    if not icefall_root.exists():
        raise SystemExit(f"ICEFALL_ROOT path does not exist: {icefall_root}")
    
    zipformer_recipe = icefall_root / "egs" / "gigaspeech" / "KWS" / "zipformer"
    if not zipformer_recipe.exists():
        raise SystemExit(
            f"Zipformer recipe not found at {zipformer_recipe}. "
            f"Ensure ICEFALL_ROOT points to the icefall repo with gigaspeech KWS recipes."
        )
    
    zipformer_path = str(zipformer_recipe)
    if zipformer_path not in sys.path:
        sys.path.insert(0, zipformer_path)
    
    return zipformer_recipe


def resolve_dict_path(config: Mapping[str, Any], *, project_root: Path | None = None) -> Path:
    """Resolve tokenizer dict path from a loaded config."""
    root = project_root or PROJECT_ROOT
    tokenizer_cfg = get_tokenizer_config(dict(config))
    dict_path = Path(tokenizer_cfg["dict_path"])
    if not dict_path.is_absolute():
        dict_path = root / dict_path
    return dict_path
