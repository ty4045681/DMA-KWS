#!/usr/bin/env python3
"""Merge sharded ``eval_musan_fa.py`` outputs into one FA/hour result."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from dma_kws.inference.detection_plots import (
    DEFAULT_PLOT_DPI,
    write_false_accept_rate_plot,
)
from dma_kws.inference.musan_fa import (
    discover_shard_dirs,
    load_jsonl_records,
    merge_musan_summaries,
)


def merge_eval_dir(root: Path, output_dir: Path, *, plot_curves: bool = True) -> dict:
    """Pool ``shard_*/`` summaries under ``root`` into ``output_dir``."""

    shard_dirs = discover_shard_dirs(root)
    summaries = []
    results = []
    for shard_dir in shard_dirs:
        summary_path = shard_dir / "summary.json"
        results_path = shard_dir / "results.jsonl"
        if not results_path.is_file():
            raise SystemExit(f"Missing results.jsonl next to {summary_path}")
        with summary_path.open("r", encoding="utf-8") as handle:
            summary = json.load(handle)
        if not isinstance(summary, dict):
            raise SystemExit(f"{summary_path} must contain a JSON object")
        summaries.append(summary)
        results.append(load_jsonl_records(results_path))

    threshold = 0.5
    for summary in summaries:
        metrics = summary.get("metrics") or {}
        if "threshold" in metrics:
            threshold = float(metrics["threshold"])
            break
        plots = summary.get("plots") or {}
        if "deployment_threshold" in plots:
            threshold = float(plots["deployment_threshold"])
            break

    output_dir.mkdir(parents=True, exist_ok=True)
    merged, all_results = merge_musan_summaries(
        summaries,
        results,
        output_dir=output_dir,
        threshold=threshold,
    )

    results_path = output_dir / "results.jsonl"
    with results_path.open("w", encoding="utf-8") as handle:
        for record in all_results:
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")

    if plot_curves:
        merged["plots"] = write_false_accept_rate_plot(
            all_results,
            output_dir=output_dir,
            threshold=threshold,
            total_hours=float(merged.get("total_hours", 0.0)),
            dpi=DEFAULT_PLOT_DPI,
        )

    summary_path = output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(merged, handle, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps(merged, ensure_ascii=False, indent=2, allow_nan=False))
    return merged


def main() -> None:
    if len(sys.argv) not in {3, 4}:
        raise SystemExit(
            "Usage: python3 scripts/merge_musan_fa.py ROOT_DIR OUT_DIR [--no-plot]"
        )
    root = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])
    plot_curves = True
    if len(sys.argv) == 4:
        if sys.argv[3] != "--no-plot":
            raise SystemExit(f"Unknown option: {sys.argv[3]}")
        plot_curves = False
    merge_eval_dir(root, output_dir, plot_curves=plot_curves)


if __name__ == "__main__":
    main()
