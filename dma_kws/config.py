"""Configuration helpers for DMA-KWS scripts."""

from __future__ import annotations

import os
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any, Iterable

import yaml


@dataclass(frozen=True)
class FbankConfig:
    """Shared Kaldi fbank feature extraction settings."""

    num_mel_bins: int = 80
    frame_length: int = 25
    frame_shift: int = 10
    dither: float = 0.1
    window_type: str = "povey"


def _expand_value(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expanduser(os.path.expandvars(value))
    if isinstance(value, list):
        return [_expand_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand_value(item) for key, item in value.items()}
    return value


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file and expand env/user markers in string values."""
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping at top level: {config_path}")
    return _expand_value(data)


def require_sections(config: dict[str, Any], sections: Iterable[str]) -> None:
    """Raise a readable error when required top-level config sections are absent."""
    missing = [section for section in sections if section not in config]
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"Missing required config sections: {joined}")


def get_tokenizer_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return the tokenizer section from a loaded config."""
    require_sections(config, ["tokenizer"])
    tokenizer = config["tokenizer"]
    if not isinstance(tokenizer, dict):
        raise ValueError("Config section 'tokenizer' must be a mapping")
    return tokenizer


def _fbank_field_names() -> tuple[str, ...]:
    return tuple(field.name for field in fields(FbankConfig))


def _resolve_num_mel_bins(config: dict[str, Any], fbank_section: dict[str, Any]) -> int:
    if "num_mel_bins" in fbank_section:
        return int(fbank_section["num_mel_bins"])
    stage1 = config.get("stage1")
    if isinstance(stage1, dict) and "input_dim" in stage1:
        return int(stage1["input_dim"])
    return FbankConfig.num_mel_bins


def get_fbank_config(config: dict[str, Any]) -> FbankConfig:
    """Return shared fbank settings from optional top-level ``fbank`` config."""
    fbank_section = config.get("fbank", {})
    if not isinstance(fbank_section, dict):
        raise ValueError("Config section 'fbank' must be a mapping")

    kwargs: dict[str, Any] = {
        "num_mel_bins": _resolve_num_mel_bins(config, fbank_section),
    }
    for name in _fbank_field_names():
        if name == "num_mel_bins":
            continue
        if name in fbank_section:
            kwargs[name] = fbank_section[name]
    return FbankConfig(**kwargs)


def get_eval_fbank_config(config: dict[str, Any]) -> FbankConfig:
    """Return fbank settings for eval prep, with optional ``stage2.eval.fbank`` overrides."""
    base = get_fbank_config(config)
    stage2 = config.get("stage2")
    if not isinstance(stage2, dict):
        return base

    eval_section = stage2.get("eval")
    if not isinstance(eval_section, dict):
        return base

    eval_fbank = eval_section.get("fbank")
    if eval_fbank is None:
        return base
    if not isinstance(eval_fbank, dict):
        raise ValueError("Config section 'stage2.eval.fbank' must be a mapping")

    overrides = {name: eval_fbank[name] for name in _fbank_field_names() if name in eval_fbank}
    return replace(base, **overrides)


def fbank_kwargs(cfg: FbankConfig) -> dict[str, Any]:
    """Return keyword arguments suitable for ``compute_fbank_for_clip``."""
    return {
        "num_mel_bins": cfg.num_mel_bins,
        "frame_length": cfg.frame_length,
        "frame_shift": cfg.frame_shift,
        "dither": cfg.dither,
        "window_type": cfg.window_type,
    }
