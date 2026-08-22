#!/usr/bin/env python3
"""Plot a MUSAN FA/hour curve from an existing eval directory.

The script never reruns inference. It reads ``results.jsonl`` written by
``eval_musan_fa.py`` or ``merge_musan_fa.py`` and writes
``fa_per_hour_curve.png`` with the same helper those scripts use.

Examples:

    python3 scripts/plot_musan_fa_curve.py /path/to/musan_test

    python3 scripts/plot_musan_fa_curve.py /path/to/musan_test/results.jsonl \
      --output-dir /path/to/plots --threshold 0.5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from dma_kws.inference.detection_plots import (
    DEFAULT_PLOT_DPI,
    write_false_accept_rate_plot,
)
from dma_kws.inference.musan_fa import load_jsonl_records


def resolve_results_path(path: Path) -> Path:
    """Accept a MUSAN eval directory or a ``results.jsonl`` file."""

    path = path.expanduser()
    if path.is_dir():
        path = path / "results.jsonl"
    if not path.is_file():
        raise SystemExit(f"results.jsonl not found: {path}")
    return path.resolve()


def load_summary(path: Path | None) -> dict[str, Any]:
    if path is None or not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Failed to read JSON from {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise SystemExit(f"Expected a JSON object in {path}")
    return value


def resolve_threshold(summary: dict[str, Any], requested: float | None) -> float:
    if requested is not None:
        return float(requested)
    metrics = summary.get("metrics")
    if isinstance(metrics, dict) and metrics.get("threshold") is not None:
        return float(metrics["threshold"])
    plots = summary.get("plots")
    if isinstance(plots, dict) and plots.get("deployment_threshold") is not None:
        return float(plots["deployment_threshold"])
    return 0.5


def resolve_total_hours(
    summary: dict[str, Any],
    requested: float | None,
) -> float:
    if requested is not None:
        hours = float(requested)
    elif summary.get("total_hours") is not None:
        hours = float(summary["total_hours"])
    else:
        raise SystemExit(
            "MUSAN FA/hour plots need total audio duration. Keep summary.json "
            "beside results.jsonl, pass --summary, or pass --total-hours."
        )
    if hours <= 0 or hours != hours:
        raise SystemExit("--total-hours / summary total_hours must be > 0")
    return hours


def plot_musan_fa_curve(
    source: Path,
    *,
    output_dir: Path | None = None,
    summary_path: Path | None = None,
    threshold: float | None = None,
    total_hours: float | None = None,
    dpi: int = DEFAULT_PLOT_DPI,
) -> dict[str, Any]:
    """Load one MUSAN eval output and write ``fa_per_hour_curve.png``."""

    results_path = resolve_results_path(source)
    if summary_path is None:
        summary_path = results_path.parent / "summary.json"
    elif not summary_path.is_file():
        raise SystemExit(f"summary.json not found: {summary_path}")
    summary = load_summary(summary_path)
    records = load_jsonl_records(results_path)
    if not records:
        raise SystemExit(f"No result rows found in {results_path}")
    hours = resolve_total_hours(summary, total_hours)
    deploy_threshold = resolve_threshold(summary, threshold)
    destination = (output_dir or results_path.parent).expanduser().resolve()
    plot_summary = write_false_accept_rate_plot(
        records,
        output_dir=destination,
        threshold=deploy_threshold,
        total_hours=hours,
        dpi=int(dpi),
    )
    if plot_summary.get("status") != "generated":
        reason = plot_summary.get("reason", "unknown plot failure")
        raise SystemExit(f"Did not write fa_per_hour_curve.png: {reason}")
    return plot_summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Plot fa_per_hour_curve.png from an eval_musan_fa.py directory "
            "or results.jsonl. Does not rerun inference."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "results",
        type=Path,
        help="MUSAN eval directory, or a results.jsonl path",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        help="summary.json path; defaults to the results.jsonl directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory for fa_per_hour_curve.png; defaults to the results directory",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        help="Deployment threshold to mark; defaults to summary.json then 0.5",
    )
    parser.add_argument(
        "--total-hours",
        type=float,
        help="Override total audio hours when summary.json is unavailable",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=DEFAULT_PLOT_DPI,
        help="PNG resolution",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.dpi <= 0:
        raise SystemExit("--dpi must be positive")
    if args.threshold is not None and (
        args.threshold != args.threshold
        or args.threshold < 0
        or args.threshold > 1
    ):
        raise SystemExit("--threshold must be a finite value in [0, 1]")
    plot_summary = plot_musan_fa_curve(
        args.results,
        output_dir=args.output_dir,
        summary_path=args.summary,
        threshold=args.threshold,
        total_hours=args.total_hours,
        dpi=args.dpi,
    )
    print(json.dumps(plot_summary, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main(sys.argv[1:])
