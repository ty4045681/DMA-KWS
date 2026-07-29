#!/usr/bin/env python3
"""Precompute LibriPhrase eval fbank .npy files for Stage II validation."""

from __future__ import annotations

from functools import partial
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from dma_kws.config import fbank_kwargs, get_eval_fbank_config, require_sections
from dma_kws.hydra_app import CONFIG_DIR, resolved_config
from dma_kws.stage2.dataset import resolve_stage2_eval_paths
from dma_kws.stage2.fbank import FbankExtractor
from dma_kws.stage2.prepare_eval_fbank import (
    compute_fbank_for_padded_clip,
    prepare_eval_fbank,
    prepare_eval_fbank_from_csv,
)


@hydra.main(version_base=None, config_path=str(CONFIG_DIR), config_name="config")
def main(cfg: DictConfig) -> None:
    config = resolved_config(cfg)
    require_sections(config, ["paths", "stage2"])
    prep = OmegaConf.to_container(cfg.prep, resolve=True)
    if not isinstance(prep, dict):
        prep = {}

    eval_paths = resolve_stage2_eval_paths(config)
    test_dir = Path(prep["test_dir"]) if prep.get("test_dir") else eval_paths["test_dir"]
    explicit_fbank_dir = str(prep.get("fbank_dir", "")).strip()
    fbank_dir = Path(explicit_fbank_dir) if explicit_fbank_dir else eval_paths["fbank_dir"]
    if prep.get("test_dir") and fbank_dir == eval_paths["test_dir"]:
        fbank_dir = test_dir
    if not test_dir.exists():
        raise SystemExit(f"Eval test_dir not found: {test_dir}")

    left_padding_ms = int(prep.get("left_padding_ms", 0))
    right_padding_ms = int(prep.get("right_padding_ms", 0))
    if left_padding_ms < 0 or right_padding_ms < 0:
        raise SystemExit("prep.left_padding_ms and prep.right_padding_ms must be >= 0")
    if (left_padding_ms or right_padding_ms) and not explicit_fbank_dir:
        raise SystemExit(
            "Padded eval features require +prep.fbank_dir=/a/separate/output/directory "
            "so the baseline fbank tree cannot be overwritten."
        )

    skip_existing = not bool(prep.get("no_skip_existing", False))
    fbank_params = fbank_kwargs(get_eval_fbank_config(config))
    fbank_extractor = FbankExtractor(**fbank_params)
    limit = int(prep.get("limit", 0))
    log_interval = int(prep.get("log_interval", 1000))
    compute_fn = partial(
        compute_fbank_for_padded_clip,
        left_padding_ms=left_padding_ms,
        right_padding_ms=right_padding_ms,
    )
    if left_padding_ms or right_padding_ms:
        print(
            f"Preparing padded eval fbank: left={left_padding_ms}ms "
            f"right={right_padding_ms}ms output={fbank_dir}"
        )

    if bool(prep.get("from_csv", False)):
        written, skipped, failed = prepare_eval_fbank_from_csv(
            test_dir,
            fbank_dir=fbank_dir,
            csv_files=eval_paths["csv_files"],
            skip_existing=skip_existing,
            limit=limit,
            log_interval=log_interval,
            extractor=fbank_extractor,
            compute_fn=compute_fn,
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
            compute_fn=compute_fn,
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
