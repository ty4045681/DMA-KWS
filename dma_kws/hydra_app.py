"""Shared Hydra application helpers."""

from __future__ import annotations

from pathlib import Path

from omegaconf import DictConfig

from dma_kws.config import config_to_dict

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


def resolved_config(cfg: DictConfig) -> dict:
    """Merge composed Hydra config with schema defaults and return a plain dict."""
    return config_to_dict(cfg)
