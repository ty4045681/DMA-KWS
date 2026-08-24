#!/usr/bin/env python3
"""Merge sharded ``eval_musan_fa.py`` outputs into one FA/hour result."""

from __future__ import annotations

import argparse
import json
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


def merge_eval_dir(
    root: Path,
    output_dir: Path,
    *,
    plot_curves: bool = True,
    num_shards: int | None = None,
) -> dict:
    """Pool ``shard_*/`` summaries under ``root`` into ``output_dir``."""

    shard_dirs = discover_shard_dirs(root, num_shards=num_shards)
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
    parser = argparse.ArgumentParser(
        description="Merge sharded MUSAN FA outputs into one FA/hour result."
    )
    parser.add_argument("root_dir", type=Path)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument(
        "--num-shards",
        type=int,
        default=None,
        help="Only merge shard_0 … shard_{N-1}; ignore leftover higher-index dirs.",
    )
    args = parser.parse_args()
    if args.num_shards is not None and args.num_shards < 1:
        raise SystemExit("--num-shards must be >= 1")
    merge_eval_dir(
        args.root_dir,
        args.out_dir,
        plot_curves=not args.no_plot,
        num_shards=args.num_shards,
    )


if __name__ == "__main__":
    main()
