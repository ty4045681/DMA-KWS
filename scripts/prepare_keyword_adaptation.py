#!/usr/bin/env python3
"""Prepare keyword adaptation data: scan raw wavs, G2P, fbank, train/eval manifests."""

from __future__ import annotations

import json

import hydra
from omegaconf import DictConfig, OmegaConf

from dma_kws.config import require_sections
from dma_kws.hydra_app import CONFIG_DIR, resolved_config
from dma_kws.stage2.adapt_console import prepare_with_console


@hydra.main(version_base=None, config_path=str(CONFIG_DIR), config_name="config")
def main(cfg: DictConfig) -> None:
    config = resolved_config(cfg)
    require_sections(config, ["paths", "adapt"])
    adapt = OmegaConf.to_container(cfg.adapt, resolve=True)
    if not isinstance(adapt, dict):
        raise SystemExit("adapt config section must be a mapping")
    prep = OmegaConf.to_container(cfg.prep, resolve=True)
    if not isinstance(prep, dict):
        prep = {}

    stats = prepare_with_console(config, adapt, prep)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
