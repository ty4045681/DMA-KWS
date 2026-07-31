#!/usr/bin/env python3
"""Offline threshold scan for Stage-II clip and MUSAN evaluation results.

The script consumes the ``results.jsonl`` written by either
``eval_stage2_clips.py`` or ``eval_musan_fa.py``. It never reruns inference:
every operating point is recomputed from the saved ``qbyt_score`` values.

Examples:

    python3 scripts/scan_stage2_thresholds.py outputs/eval_stage2_clips \
      --max-fpr 0.001 --workers 8

    python3 scripts/scan_stage2_thresholds.py outputs/eval_musan_fa/results.jsonl \
      --max-fa-per-hour 0.5 --workers 8
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from dma_kws.inference.metrics import binary_auc, binary_eer


_CLIP_COLUMNS = (
    "threshold",
    "tp",
    "tn",
    "fp",
    "fn",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "fpr",
    "fnr",
    "youden_j",
)
_MUSAN_COLUMNS = (
    "threshold",
    "fp",
    "tn",
    "fpr",
    "fa_per_hour",
    "fa_per_1000_hours",
)


@dataclass(frozen=True)
class ScanInput:
    """Validated scores and metadata needed by the threshold scanner."""

    results_path: Path
    summary_path: Path | None
    mode: str
    labels: np.ndarray
    scores: np.ndarray
    total_hours: float | None
    subset_scores: dict[str, np.ndarray]
    subset_hours: dict[str, float]
    subset_names: dict[str, str]


def _safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    result = np.zeros_like(numerator, dtype=np.float64)
    np.divide(numerator, denominator, out=result, where=denominator != 0)
    return result


def _python_number(value: Any) -> int | float:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Failed to read JSON from {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise SystemExit(f"Expected a JSON object in {path}")
    return value


def _read_results(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise SystemExit(
                        f"Invalid JSON at {path}:{line_number}: {exc}"
                    ) from exc
                if not isinstance(record, dict):
                    raise SystemExit(
                        f"Expected a JSON object at {path}:{line_number}"
                    )
                records.append(record)
    except OSError as exc:
        raise SystemExit(f"Failed to read {path}: {exc}") from exc
    if not records:
        raise SystemExit(f"No result rows found in {path}")
    return records


def _resolve_results_path(path: Path) -> Path:
    path = path.expanduser()
    if path.is_dir():
        path = path / "results.jsonl"
    if not path.is_file():
        raise SystemExit(f"results.jsonl not found: {path}")
    return path.resolve()


def _resolve_summary_path(results_path: Path, requested: Path | None) -> Path | None:
    if requested is not None:
        path = requested.expanduser()
        if not path.is_file():
            raise SystemExit(f"summary.json not found: {path}")
        return path.resolve()
    candidate = results_path.parent / "summary.json"
    return candidate.resolve() if candidate.is_file() else None


def _has_musan_window_metadata(record: dict[str, Any]) -> bool:
    metadata = record.get("manifest_meta")
    return (
        isinstance(metadata, dict)
        and "subset" in metadata
        and "start_sec" in metadata
        and "end_sec" in metadata
    )


def _infer_mode(
    requested: str,
    records: Sequence[dict[str, Any]],
    summary: dict[str, Any],
) -> str:
    if requested != "auto":
        return requested
    if "musan_root" in summary or (
        "total_hours" in summary and all(_has_musan_window_metadata(row) for row in records)
    ):
        return "musan"
    return "clips"


def _subset_slug(name: str, used: set[str]) -> str:
    base = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_") or "other"
    candidate = base
    suffix = 2
    while candidate in used:
        candidate = f"{base}_{suffix}"
        suffix += 1
    used.add(candidate)
    return candidate


def _extract_subsets(
    records: Sequence[dict[str, Any]],
    summary: dict[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, float], dict[str, str]]:
    raw_scores: dict[str, list[float]] = {}
    for record in records:
        metadata = record.get("manifest_meta")
        if not isinstance(metadata, dict):
            continue
        raw_name = str(metadata.get("subset", "")).strip()
        if not raw_name:
            continue
        raw_scores.setdefault(raw_name, []).append(float(record["qbyt_score"]))

    summary_subsets = summary.get("subsets", {})
    if not isinstance(summary_subsets, dict):
        summary_subsets = {}

    used: set[str] = set()
    subset_scores: dict[str, np.ndarray] = {}
    subset_hours: dict[str, float] = {}
    subset_names: dict[str, str] = {}
    for raw_name in sorted(raw_scores):
        slug = _subset_slug(raw_name, used)
        subset_scores[slug] = np.sort(
            np.asarray(raw_scores[raw_name], dtype=np.float64)
        )
        subset_names[slug] = raw_name
        subset_summary = summary_subsets.get(raw_name, {})
        if isinstance(subset_summary, dict):
            hours = subset_summary.get("total_hours")
            if hours is not None and float(hours) > 0:
                subset_hours[slug] = float(hours)
    return subset_scores, subset_hours, subset_names


def load_scan_input(
    results: Path,
    *,
    mode: str = "auto",
    summary_path: Path | None = None,
    total_hours: float | None = None,
    include_subsets: bool = True,
) -> ScanInput:
    """Load and validate one ``results.jsonl`` evaluation output."""

    results_path = _resolve_results_path(results)
    resolved_summary = _resolve_summary_path(results_path, summary_path)
    summary = _read_json(resolved_summary) if resolved_summary else {}
    records = _read_results(results_path)
    resolved_mode = _infer_mode(mode, records, summary)

    scores: list[float] = []
    labels: list[int] = []
    for index, record in enumerate(records, start=1):
        if "qbyt_score" not in record:
            raise SystemExit(
                f"{results_path}:{index} has no qbyt_score; this is not a supported "
                "eval_stage2_clips/eval_musan_fa result row"
            )
        if "label" not in record:
            raise SystemExit(
                f"{results_path}:{index} has no label; threshold metrics require "
                "label=1 for positives and label=0 for negatives"
            )
        score = float(record["qbyt_score"])
        label = int(record["label"])
        if not math.isfinite(score):
            raise SystemExit(f"{results_path}:{index} has non-finite qbyt_score={score}")
        if label not in (0, 1):
            raise SystemExit(f"{results_path}:{index} has invalid label={label}; expected 0 or 1")
        scores.append(score)
        labels.append(label)

    score_array = np.asarray(scores, dtype=np.float64)
    label_array = np.asarray(labels, dtype=np.int64)
    positives = int(np.sum(label_array == 1))
    negatives = int(np.sum(label_array == 0))

    if resolved_mode == "clips" and (positives == 0 or negatives == 0):
        raise SystemExit(
            "Clip threshold scanning needs both positive and negative labeled rows; "
            f"found positives={positives}, negatives={negatives}. Use --mode musan "
            "for an intentional negative-only false-accept scan."
        )
    if resolved_mode == "musan" and positives:
        raise SystemExit(
            f"MUSAN results must be negative-only, but found {positives} positive rows"
        )

    resolved_hours = total_hours
    if resolved_hours is None and summary.get("total_hours") is not None:
        resolved_hours = float(summary["total_hours"])
    if resolved_mode == "musan" and (resolved_hours is None or resolved_hours <= 0):
        raise SystemExit(
            "MUSAN scanning needs total audio duration for FA/hour. Keep summary.json "
            "beside results.jsonl, pass --summary, or pass --total-hours."
        )

    subset_scores: dict[str, np.ndarray] = {}
    subset_hours: dict[str, float] = {}
    subset_names: dict[str, str] = {}
    if resolved_mode == "musan" and include_subsets:
        subset_scores, subset_hours, subset_names = _extract_subsets(records, summary)

    return ScanInput(
        results_path=results_path,
        summary_path=resolved_summary,
        mode=resolved_mode,
        labels=label_array,
        scores=score_array,
        total_hours=resolved_hours,
        subset_scores=subset_scores,
        subset_hours=subset_hours,
        subset_names=subset_names,
    )


def build_thresholds(scores: np.ndarray, *, step: float | None = None) -> np.ndarray:
    """Return descending thresholds covering every requested operating point."""

    if scores.size == 0:
        raise ValueError("scores must not be empty")
    reject_all = np.nextafter(float(np.max(scores)), math.inf)
    if step is None:
        observed = np.unique(scores)[::-1]
        return np.concatenate((np.asarray([reject_all]), observed))

    if not 0 < step <= 1:
        raise ValueError("threshold step must be in (0, 1]")
    if float(np.min(scores)) < 0 or float(np.max(scores)) > 1:
        raise ValueError("--threshold-step requires qbyt_score values in [0, 1]")

    grid = np.arange(0.0, 1.0 + step * 0.5, step, dtype=np.float64)
    grid = np.clip(grid, 0.0, 1.0)
    grid = np.unique(np.concatenate((grid, np.asarray([0.0, 1.0]))))[::-1]
    if reject_all <= grid[0]:
        return grid
    return np.concatenate((np.asarray([reject_all]), grid))


def _scan_chunk(
    thresholds: np.ndarray,
    *,
    positive_scores: np.ndarray,
    negative_scores: np.ndarray,
    mode: str,
    total_hours: float | None,
    subset_scores: dict[str, np.ndarray],
    subset_hours: dict[str, float],
) -> dict[str, np.ndarray]:
    positives = int(positive_scores.size)
    negatives = int(negative_scores.size)

    tp = positives - np.searchsorted(positive_scores, thresholds, side="left")
    fp = negatives - np.searchsorted(negative_scores, thresholds, side="left")
    fn = positives - tp
    tn = negatives - fp

    arrays: dict[str, np.ndarray] = {
        "threshold": thresholds,
        "tp": tp.astype(np.int64),
        "tn": tn.astype(np.int64),
        "fp": fp.astype(np.int64),
        "fn": fn.astype(np.int64),
    }
    if mode == "clips":
        precision = _safe_divide(tp, tp + fp)
        recall = _safe_divide(tp, tp + fn)
        fpr = _safe_divide(fp, fp + tn)
        fnr = _safe_divide(fn, fn + tp)
        arrays.update(
            {
                "accuracy": _safe_divide(tp + tn, np.full_like(tp, positives + negatives)),
                "precision": precision,
                "recall": recall,
                "f1": _safe_divide(2.0 * precision * recall, precision + recall),
                "fpr": fpr,
                "fnr": fnr,
                "youden_j": recall - fpr,
            }
        )
        return arrays

    assert total_hours is not None
    arrays["fpr"] = _safe_divide(fp, fp + tn)
    arrays["fa_per_hour"] = fp.astype(np.float64) / total_hours
    arrays["fa_per_1000_hours"] = fp.astype(np.float64) * 1000.0 / total_hours

    for slug, sorted_scores in subset_scores.items():
        subset_fp = sorted_scores.size - np.searchsorted(
            sorted_scores, thresholds, side="left"
        )
        subset_tn = sorted_scores.size - subset_fp
        arrays[f"subset_{slug}_fp"] = subset_fp.astype(np.int64)
        arrays[f"subset_{slug}_tn"] = subset_tn.astype(np.int64)
        arrays[f"subset_{slug}_fpr"] = _safe_divide(
            subset_fp, subset_fp + subset_tn
        )
        hours = subset_hours.get(slug)
        if hours is not None:
            arrays[f"subset_{slug}_fa_per_hour"] = (
                subset_fp.astype(np.float64) / hours
            )
    return arrays


def scan_thresholds(
    scan_input: ScanInput,
    thresholds: np.ndarray,
    *,
    workers: int = 0,
) -> tuple[dict[str, np.ndarray], int]:
    """Compute threshold metrics, optionally splitting thresholds across threads."""

    if workers < 0:
        raise ValueError("workers must be >= 0")
    requested_workers = workers or min(32, os.cpu_count() or 1)
    actual_workers = max(1, min(int(requested_workers), int(thresholds.size)))

    positive_scores = np.sort(scan_input.scores[scan_input.labels == 1])
    negative_scores = np.sort(scan_input.scores[scan_input.labels == 0])
    chunks = [
        chunk
        for chunk in np.array_split(thresholds, actual_workers)
        if chunk.size
    ]

    def scan(chunk: np.ndarray) -> dict[str, np.ndarray]:
        return _scan_chunk(
            chunk,
            positive_scores=positive_scores,
            negative_scores=negative_scores,
            mode=scan_input.mode,
            total_hours=scan_input.total_hours,
            subset_scores=scan_input.subset_scores,
            subset_hours=scan_input.subset_hours,
        )

    if actual_workers == 1:
        chunk_results = [scan(chunks[0])]
    else:
        with ThreadPoolExecutor(
            max_workers=actual_workers,
            thread_name_prefix="threshold-scan",
        ) as executor:
            chunk_results = list(executor.map(scan, chunks))

    keys = tuple(chunk_results[0])
    merged = {
        key: np.concatenate([chunk[key] for chunk in chunk_results])
        for key in keys
    }
    return merged, actual_workers


def _row_at(arrays: dict[str, np.ndarray], index: int) -> dict[str, int | float]:
    return {key: _python_number(values[index]) for key, values in arrays.items()}


def _best_index(
    arrays: dict[str, np.ndarray],
    *,
    primary: str,
    secondary: str | None = None,
) -> int:
    primary_values = arrays[primary]
    candidates = np.flatnonzero(primary_values == np.max(primary_values))
    if secondary is not None and candidates.size > 1:
        secondary_values = arrays[secondary][candidates]
        candidates = candidates[secondary_values == np.max(secondary_values)]
    if candidates.size > 1:
        thresholds = arrays["threshold"][candidates]
        candidates = np.asarray([candidates[int(np.argmax(thresholds))]])
    return int(candidates[0])


def select_operating_point(
    arrays: dict[str, np.ndarray],
    *,
    mode: str,
    max_fpr: float | None = None,
    min_recall: float | None = None,
    max_fa_per_hour: float | None = None,
) -> dict[str, Any] | None:
    """Select a constrained operating point, or return ``None`` without targets."""

    if max_fpr is None and min_recall is None and max_fa_per_hour is None:
        return None
    if mode == "musan" and min_recall is not None:
        raise ValueError("--min-recall is unavailable for negative-only MUSAN results")
    if mode == "clips" and max_fa_per_hour is not None:
        raise ValueError("--max-fa-per-hour is only available for MUSAN results")

    feasible = np.ones(arrays["threshold"].shape, dtype=bool)
    if max_fpr is not None:
        feasible &= arrays["fpr"] <= max_fpr
    if min_recall is not None:
        feasible &= arrays["recall"] >= min_recall
    if max_fa_per_hour is not None:
        feasible &= arrays["fa_per_hour"] <= max_fa_per_hour
    candidates = np.flatnonzero(feasible)

    constraints = {
        key: value
        for key, value in {
            "max_fpr": max_fpr,
            "min_recall": min_recall,
            "max_fa_per_hour": max_fa_per_hour,
        }.items()
        if value is not None
    }
    if candidates.size == 0:
        return {"constraints": constraints, "found": False}

    if mode == "musan":
        # Lower thresholds generally preserve more positive recall. Pick the
        # lowest threshold that still satisfies the false-accept constraint.
        index = int(candidates[int(np.argmin(arrays["threshold"][candidates]))])
    elif max_fpr is not None:
        best_recall = np.max(arrays["recall"][candidates])
        candidates = candidates[arrays["recall"][candidates] == best_recall]
        best_fpr = np.min(arrays["fpr"][candidates])
        candidates = candidates[arrays["fpr"][candidates] == best_fpr]
        index = int(candidates[int(np.argmax(arrays["threshold"][candidates]))])
    else:
        best_fpr = np.min(arrays["fpr"][candidates])
        candidates = candidates[arrays["fpr"][candidates] == best_fpr]
        best_recall = np.max(arrays["recall"][candidates])
        candidates = candidates[arrays["recall"][candidates] == best_recall]
        index = int(candidates[int(np.argmax(arrays["threshold"][candidates]))])

    return {
        "constraints": constraints,
        "found": True,
        "operating_point": _row_at(arrays, index),
    }


def _csv_columns(mode: str, arrays: dict[str, np.ndarray]) -> list[str]:
    base = list(_CLIP_COLUMNS if mode == "clips" else _MUSAN_COLUMNS)
    return base + sorted(key for key in arrays if key not in base and key not in {"tp", "fn"})


def write_curve_csv(
    arrays: dict[str, np.ndarray],
    *,
    mode: str,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    columns = _csv_columns(mode, arrays)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for index in range(arrays["threshold"].size):
            row = _row_at(arrays, index)
            writer.writerow({column: row[column] for column in columns})


def _validate_rate(name: str, value: float | None) -> None:
    if value is not None and not 0 <= value <= 1:
        raise SystemExit(f"{name} must be in [0, 1]")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Scan qbyt_score thresholds offline from eval_stage2_clips.py or "
            "eval_musan_fa.py results.jsonl."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "results",
        type=Path,
        help="results.jsonl path, or an evaluation directory containing it",
    )
    parser.add_argument(
        "--mode",
        choices=("auto", "clips", "musan"),
        default="auto",
        help="Input type; auto uses summary.json and row metadata",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        help="summary.json path; defaults to the results.jsonl directory",
    )
    parser.add_argument(
        "--total-hours",
        type=float,
        help="Override MUSAN total audio hours when summary.json is unavailable",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Threshold-scanning threads; 0 selects min(32, CPU count)",
    )
    parser.add_argument(
        "--threshold-step",
        type=float,
        help="Use a uniform [0,1] grid; omit to scan every observed score breakpoint",
    )
    parser.add_argument(
        "--max-fpr",
        type=float,
        help="Select the best operating point subject to FPR <= this value",
    )
    parser.add_argument(
        "--min-recall",
        type=float,
        help="For clip results, require recall >= this value",
    )
    parser.add_argument(
        "--max-fa-per-hour",
        type=float,
        help="For MUSAN results, select the lowest threshold meeting this FA/hour",
    )
    parser.add_argument(
        "--no-subsets",
        action="store_true",
        help="Do not add per-subset MUSAN columns",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        help="Curve CSV; defaults to <results-dir>/threshold_scan.csv",
    )
    parser.add_argument(
        "--out-summary",
        type=Path,
        help="Scan summary JSON; defaults to <results-dir>/threshold_scan_summary.json",
    )
    return parser


def run_scan(args: argparse.Namespace) -> dict[str, Any]:
    _validate_rate("--max-fpr", args.max_fpr)
    _validate_rate("--min-recall", args.min_recall)
    if args.max_fa_per_hour is not None and args.max_fa_per_hour < 0:
        raise SystemExit("--max-fa-per-hour must be >= 0")
    if args.total_hours is not None and args.total_hours <= 0:
        raise SystemExit("--total-hours must be > 0")
    if args.workers < 0:
        raise SystemExit("--workers must be >= 0")

    scan_input = load_scan_input(
        args.results,
        mode=args.mode,
        summary_path=args.summary,
        total_hours=args.total_hours,
        include_subsets=not args.no_subsets,
    )
    try:
        thresholds = build_thresholds(
            scan_input.scores,
            step=args.threshold_step,
        )
        arrays, actual_workers = scan_thresholds(
            scan_input,
            thresholds,
            workers=args.workers,
        )
        selection = select_operating_point(
            arrays,
            mode=scan_input.mode,
            max_fpr=args.max_fpr,
            min_recall=args.min_recall,
            max_fa_per_hour=args.max_fa_per_hour,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    output_csv = (
        args.out_csv.expanduser()
        if args.out_csv
        else scan_input.results_path.parent / "threshold_scan.csv"
    ).resolve()
    output_summary = (
        args.out_summary.expanduser()
        if args.out_summary
        else scan_input.results_path.parent / "threshold_scan_summary.json"
    ).resolve()
    write_curve_csv(arrays, mode=scan_input.mode, output_path=output_csv)

    positives = int(np.sum(scan_input.labels == 1))
    negatives = int(np.sum(scan_input.labels == 0))
    summary: dict[str, Any] = {
        "results": str(scan_input.results_path),
        "source_summary": str(scan_input.summary_path) if scan_input.summary_path else None,
        "mode": scan_input.mode,
        "num_samples": int(scan_input.scores.size),
        "positives": positives,
        "negatives": negatives,
        "score_min": float(np.min(scan_input.scores)),
        "score_max": float(np.max(scan_input.scores)),
        "num_thresholds": int(thresholds.size),
        "threshold_step": args.threshold_step,
        "workers": actual_workers,
        "curve_csv": str(output_csv),
    }
    if scan_input.mode == "clips":
        eer, eer_threshold = binary_eer(scan_input.labels, scan_input.scores)
        summary.update(
            {
                "auc": binary_auc(scan_input.labels, scan_input.scores),
                "eer": eer,
                "eer_threshold": eer_threshold,
                "best_youden": _row_at(
                    arrays, _best_index(arrays, primary="youden_j", secondary="f1")
                ),
                "best_f1": _row_at(
                    arrays, _best_index(arrays, primary="f1", secondary="youden_j")
                ),
            }
        )
    else:
        summary["total_hours"] = scan_input.total_hours
        summary["subsets"] = {
            slug: {
                "name": scan_input.subset_names[slug],
                "num_samples": int(scan_input.subset_scores[slug].size),
                "total_hours": scan_input.subset_hours.get(slug),
            }
            for slug in scan_input.subset_scores
        }
    if selection is not None:
        summary["selection"] = selection

    output_summary.parent.mkdir(parents=True, exist_ok=True)
    output_summary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(
        f"Scanned {summary['num_samples']} {scan_input.mode} rows at "
        f"{summary['num_thresholds']} thresholds with {actual_workers} thread(s)."
    )
    print(f"Curve:   {output_csv}")
    print(f"Summary: {output_summary}")
    if selection is not None:
        if selection["found"]:
            point = selection["operating_point"]
            details = [
                f"threshold={point['threshold']:.8g}",
                f"fpr={point['fpr']:.6g}",
            ]
            if "recall" in point:
                details.append(f"recall={point['recall']:.6g}")
            if "fa_per_hour" in point:
                details.append(f"fa/hour={point['fa_per_hour']:.6g}")
            print("Selected: " + ", ".join(details))
        else:
            print("Selected: no threshold satisfies the requested constraints")
    return summary


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    run_scan(args)


if __name__ == "__main__":
    main()
