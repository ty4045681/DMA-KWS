#!/usr/bin/env python3
"""Precompute LibriPhrase eval fbank .npy files for Stage II validation."""

from __future__ import annotations

from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from dma_kws.config import fbank_kwargs, get_eval_fbank_config, require_sections
from dma_kws.hydra_app import CONFIG_DIR, resolved_config
from dma_kws.stage2.dataset import resolve_stage2_eval_paths
from dma_kws.stage2.fbank import FbankExtractor
from dma_kws.stage2.prepare_eval_fbank import prepare_eval_fbank, prepare_eval_fbank_from_csv


@hydra.main(version_base=None, config_path=str(CONFIG_DIR), config_name="config")
def main(cfg: DictConfig) -> None:
    config = resolved_config(cfg)
    require_sections(config, ["paths", "stage2"])
    prep = OmegaConf.to_container(cfg.prep, resolve=True)
    if not isinstance(prep, dict):
        prep = {}

    eval_paths = resolve_stage2_eval_paths(config)
    test_dir = Path(prep["test_dir"]) if prep.get("test_dir") else eval_paths["test_dir"]
    fbank_dir = eval_paths["fbank_dir"]
    if prep.get("test_dir") and fbank_dir == eval_paths["test_dir"]:
        fbank_dir = test_dir
    if not test_dir.exists():
        raise SystemExit(f"Eval test_dir not found: {test_dir}")

    skip_existing = not bool(prep.get("no_skip_existing", False))
    fbank_params = fbank_kwargs(get_eval_fbank_config(config))
    fbank_extractor = FbankExtractor(**fbank_params)
    limit = int(prep.get("limit", 0))
    log_interval = int(prep.get("log_interval", 1000))

    if bool(prep.get("from_csv", False)):
        written, skipped, failed = prepare_eval_fbank_from_csv(
            test_dir,
            fbank_dir=fbank_dir,
            csv_files=eval_paths["csv_files"],
            skip_existing=skip_existing,
            limit=limit,
            log_interval=log_interval,
            extractor=fbank_extractor,
            **fbank_params,
        )
    else:
        written, skipped, failed = prepare_eval_fbank(
            test_dir,
            fbank_dir=fbank_dir,
            skip_existing=skip_existing,
            limit=limit,
            log_interval=log_interval,
            extractor=fbank_extractor,
            **fbank_params,
        )

    print(
        f"Eval fbank prep complete under {fbank_dir}: "
        f"written={written}, skipped={skipped}, failed={failed}"
    )
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
