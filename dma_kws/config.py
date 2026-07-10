"""Configuration helpers for DMA-KWS scripts."""

from __future__ import annotations

import os
from dataclasses import fields, replace
from pathlib import Path
from typing import Any, Iterable

import yaml
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf

from dma_kws.configs.schema import DMAKWSConfig, FbankConfig

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "configs"


def _expand_value(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expanduser(os.path.expandvars(value))
    if isinstance(value, list):
        return [_expand_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand_value(item) for key, item in value.items()}
    return value


def _schema_defaults() -> DictConfig:
    return OmegaConf.structured(DMAKWSConfig)


def compose_config(
    experiment: str | None = None,
    overrides: Iterable[str] = (),
) -> DictConfig:
    """Compose a validated config via Hydra (groups + optional experiment overlay)."""
    GlobalHydra.instance().clear()
    override_list = list(overrides)
    if experiment:
        override_list.insert(0, f"+experiment={experiment}")
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        return compose(config_name="config", overrides=override_list)


def _merge_with_schema(cfg: DictConfig) -> DictConfig:
    base = OmegaConf.create(OmegaConf.to_container(_schema_defaults(), resolve=False))
    OmegaConf.set_struct(base, False)
    return OmegaConf.merge(base, cfg)


def config_to_dict(cfg: DictConfig) -> dict[str, Any]:
    """Resolve a composed DictConfig to a plain dict with env/user expansion."""
    merged = _merge_with_schema(cfg)
    container = OmegaConf.to_container(merged, resolve=True)
    if not isinstance(container, dict):
        raise ValueError("Config must resolve to a mapping")
    return _expand_value(container)


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file merged onto schema defaults (backward compatible)."""
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping at top level: {config_path}")

    base = OmegaConf.create(OmegaConf.to_container(_schema_defaults(), resolve=False))
    OmegaConf.set_struct(base, False)
    merged = OmegaConf.merge(base, OmegaConf.create(data))
    container = OmegaConf.to_container(merged, resolve=False)
    if not isinstance(container, dict):
        raise ValueError(f"Config must resolve to a mapping: {config_path}")
    return _expand_value(container)


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
        "backend": cfg.backend,
        "target_sample_rate": cfg.target_sample_rate,
        "snip_edges": cfg.snip_edges,
        "low_freq": cfg.low_freq,
        "high_freq": cfg.high_freq,
    }
