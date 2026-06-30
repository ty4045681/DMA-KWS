#!/usr/bin/env python3
"""Recompute phoneme-level hard-negative distances for Stage II parquet prep."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import hydra
from omegaconf import DictConfig, OmegaConf

from dma_kws.config import require_sections
from dma_kws.hydra_app import CONFIG_DIR, resolved_config
from dma_kws.stage2.distances import build_distances_column
from dma_kws.stage2.prep_console import Stage2PrepReporter
from dma_kws.stage2.prepare_paper import OUTPUT_PARQUET_NAME

INPUT_PARQUET_NAME = "aggregated_segments_with_g2p.parquet"
_LIBRIPHRASE_ROOT_KEYS = ("libriphrase460_root", "libriphrase100_root", "libriphrase_root")


def _default_output(input_path: Path) -> Path:
    if input_path.name.endswith("_g2p.parquet"):
        return input_path.with_name(input_path.name.replace("_g2p.parquet", "_g2p_distance.parquet"))
    return input_path.with_name(OUTPUT_PARQUET_NAME)


def resolve_input_parquet(prep: dict[str, Any], paths: dict[str, Any]) -> Path:
    """Resolve the aggregated G2P parquet path from Hydra prep config."""
    explicit = prep.get("input_parquet", "")
    if explicit:
        return Path(explicit)

    for key in _LIBRIPHRASE_ROOT_KEYS:
        root = paths.get(key, "")
        if not root:
            continue
        candidate = Path(root) / INPUT_PARQUET_NAME
        if candidate.is_file():
            return candidate

    raise SystemExit(
        "No input parquet specified and no default found. Set "
        "prep.recompute_distances.input_parquet=/path/to/aggregated_segments_with_g2p.parquet "
        "or configure paths.libriphrase460_root / paths.libriphrase100_root / "
        "paths.libriphrase_root with that file present."
    )


def resolve_output_parquet(input_path: Path, output_parquet: str) -> Path:
    """Resolve the output parquet path from Hydra prep config."""
    if output_parquet:
        return Path(output_parquet)
    return _default_output(input_path)


@hydra.main(version_base=None, config_path=str(CONFIG_DIR), config_name="config")
def main(cfg: DictConfig) -> None:
    try:
        import pandas as pd
    except ImportError as exc:
        raise SystemExit("Missing dependency pandas. Install with: pip install pandas pyarrow") from exc

    config = resolved_config(cfg)
    require_sections(config, ["paths"])

    prep = OmegaConf.to_container(cfg.prep, resolve=True)
    if not isinstance(prep, dict):
        prep = {}

    distances_cfg = prep.get("recompute_distances", {})
    if not isinstance(distances_cfg, dict):
        distances_cfg = {}

    reporter = Stage2PrepReporter(use_rich=bool(prep.get("use_rich", True)))
    paths = config["paths"]

    input_path = resolve_input_parquet(distances_cfg, paths).resolve()
    if not input_path.is_file():
        raise SystemExit(f"Input parquet not found: {input_path}")

    output_path = resolve_output_parquet(
        input_path,
        str(distances_cfg.get("output_parquet", "")),
    ).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    top_k = int(distances_cfg.get("top_k", 100))
    block_size = int(distances_cfg.get("block_size", 1000))
    workers = int(distances_cfg.get("workers", -1))
    strip_stress = bool(distances_cfg.get("strip_stress", True))

    reporter.section("Plan")
    reporter.print_plan(
        [
            ("input_parquet", str(input_path)),
            ("output_parquet", str(output_path)),
            ("top_k", str(top_k)),
            ("block_size", str(block_size)),
            ("workers", str(workers)),
            ("strip_stress", str(strip_stress)),
        ],
        title="Stage II Distance Recompute",
    )

    reporter.section("Load")
    df = pd.read_parquet(input_path)
    required = {"ngram", "ngram_g2p"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"Input parquet missing required columns: {sorted(missing)}")
    reporter.info(f"Loaded {len(df)} anchors from {input_path}")

    reporter.section("Compute")
    last_done = 0

    def progress_callback(done: int, _total: int) -> None:
        nonlocal last_done
        delta = done - last_done
        last_done = done
        if bar is not None and delta:
            bar.update(delta)

    with reporter.track("Phoneme neighbor distances", total=len(df)) as bar:
        out_df = build_distances_column(
            df,
            top_k=top_k,
            strip_stress=strip_stress,
            block_size=block_size,
            workers=workers,
            progress_callback=progress_callback,
        )

    reporter.section("Write")
    out_df.to_parquet(output_path, index=False)
    reporter.info(f"Wrote {len(out_df)} rows to {output_path}")
    reporter.done("Stage II distance recompute complete.")


if __name__ == "__main__":
    main()
