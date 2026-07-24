#!/usr/bin/env python3
"""Summarize positive/negative sample counts and audio hours in a manifest.

Reads a CSV or JSONL manifest with ``audio_path``, ``keyword``, optional
``label`` columns, probes each audio file's duration in parallel, and reports
per-keyword and overall sample counts and hours.

Output schema loosely follows ``scripts/eval_musan_fa.py``:
  - ``summary.json`` contains overall fields plus a nested ``keywords`` dict.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from dma_kws.inference.manifest import load_manifest
from dma_kws.inference.musan_fa import audio_duration_sec


def resolve_num_workers(value: int) -> int:
    """Map ``num_workers`` (0 = auto) to a concrete worker count.

    Mirrors ``dma_kws.stage2.prep_console.resolve_num_workers`` without pulling
    in the torch-dependent ``dma_kws.stage2`` package.
    """
    if value < 0:
        raise ValueError(f"num_workers must be >= 0, got {value}")
    if value == 0:
        return max(1, min(8, os.cpu_count() or 1))
    return value


def probe_durations(
    audio_paths: list[str],
    *,
    num_workers: int,
) -> tuple[dict[str, float], dict[str, str]]:
    """Probe duration (seconds) for unique ``audio_paths`` in parallel.

    Returns ``(durations, errors)`` keyed by audio path.
    """
    try:
        import torchaudio  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "Missing torch/torchaudio. Install CUDA PyTorch on the remote training machine first."
        ) from exc

    unique_paths = sorted(set(audio_paths))
    durations: dict[str, float] = {}
    errors: dict[str, str] = {}

    worker_count = max(1, min(num_workers, len(unique_paths)))
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = {
            executor.submit(audio_duration_sec, path): path for path in unique_paths
        }
        for future in as_completed(futures):
            path = futures[future]
            try:
                durations[path] = float(future.result())
            except Exception as exc:  # noqa: BLE001 - report and keep going
                errors[path] = str(exc)
                print(f"WARNING: failed to probe {path}: {exc}", file=sys.stderr)
    return durations, errors


def _empty_bucket() -> dict[str, Any]:
    return {
        "num_samples": 0,
        "total_hours": 0.0,
        "num_positive": 0,
        "num_negative": 0,
        "positive_hours": 0.0,
        "negative_hours": 0.0,
        "num_unlabeled": 0,
        "num_errors": 0,
    }


def _accumulate(bucket: dict[str, Any], label: int | None, hours: float | None) -> None:
    bucket["num_samples"] += 1
    if hours is None:
        bucket["num_errors"] += 1
        hours = 0.0
    bucket["total_hours"] += hours
    if label == 1:
        bucket["num_positive"] += 1
        bucket["positive_hours"] += hours
    elif label == 0:
        bucket["num_negative"] += 1
        bucket["negative_hours"] += hours
    else:
        bucket["num_unlabeled"] += 1


def summarize_manifest(
    manifest_path: Path,
    *,
    num_workers: int,
) -> dict[str, Any]:
    rows = load_manifest(manifest_path)
    if not rows:
        raise SystemExit(f"Manifest is empty: {manifest_path}")

    durations, errors = probe_durations(
        [row["audio_path"] for row in rows],
        num_workers=num_workers,
    )

    overall = _empty_bucket()
    keywords: dict[str, dict[str, Any]] = {}
    for row in rows:
        label = row.get("label")
        hours_value = durations.get(row["audio_path"])
        hours = hours_value / 3600.0 if hours_value is not None else None
        _accumulate(overall, label, hours)
        bucket = keywords.setdefault(str(row["keyword"]), _empty_bucket())
        _accumulate(bucket, label, hours)

    summary: dict[str, Any] = {
        "manifest": str(manifest_path.resolve()),
        **overall,
        "num_unique_files": len(set(row["audio_path"] for row in rows)),
        "keywords": {keyword: keywords[keyword] for keyword in sorted(keywords)},
    }
    if errors:
        summary["probe_errors"] = errors
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="Path to CSV/JSONL manifest")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for summary.json (default: outputs/manifest_stats)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Parallel duration probes (0 = auto)",
    )
    args = parser.parse_args()

    num_workers = resolve_num_workers(args.num_workers)
    summary = summarize_manifest(args.manifest, num_workers=num_workers)

    output_dir = args.output_dir or Path("outputs/manifest_stats")
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Wrote summary -> {summary_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
