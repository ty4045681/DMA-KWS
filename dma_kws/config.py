"""Configuration helpers for DMA-KWS scripts."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterable

import yaml


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
