#!/usr/bin/env python3
"""Prepare keyword adaptation data: scan raw wavs, G2P, fbank, train/eval manifests."""

from __future__ import annotations

from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from dma_kws.config import fbank_kwargs, get_fbank_config, require_sections
from dma_kws.hydra_app import CONFIG_DIR, resolved_config
from dma_kws.stage2.adapt_paths import adapt_data_root
from dma_kws.stage2.prepare_adapt import prepare_keyword_adaptation


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

    keyword = str(adapt.get("keyword", "")).strip()
    if not keyword:
        raise SystemExit("adapt.keyword is required")

    data_root = Path(adapt["data_root"]) if adapt.get("data_root") else adapt_data_root(config, keyword)
    manifest_csv = prep.get("manifest_csv") or adapt.get("manifest_csv")
    stats = prepare_keyword_adaptation(
        keyword=keyword,
        data_root=data_root,
        fbank_params=fbank_kwargs(get_fbank_config(config)),
        eval_fraction=float(adapt.get("eval_fraction", 0.2)),
        eval_seed=int(adapt.get("eval_seed", config.get("training", {}).get("seed", 2025))),
        manifest_csv=Path(manifest_csv) if manifest_csv else None,
        skip_existing=not bool(prep.get("no_skip_existing", False)),
    )
    print(stats)


if __name__ == "__main__":
    main()
