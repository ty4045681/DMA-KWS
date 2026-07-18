#!/usr/bin/env python3
"""Aggregate per-checkpoint/keyword summary.json files from eval_musan_fa.py."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path


def _keyword_slug(keyword: str) -> str:
    """Stable lowercase slug for directory names."""
    slug = keyword.lower().strip()
    slug = re.sub(r"[ _-]+", "_", slug)
    slug = slug.strip("_")
    return slug


def aggregate(root: Path) -> list[dict]:
    """Collect summary.json files under ``root`` into TSV rows."""
    rows: list[dict] = []
    for summary_path in sorted(root.rglob("summary.json")):
        with summary_path.open("r", encoding="utf-8") as handle:
            summary = json.load(handle)

        keyword = summary.get("keyword", "")
        stage2_ckpt = summary.get("stage2_ckpt", "")
        keyword_dir = _keyword_slug(keyword) if keyword else ""

        # Prefer the checkpoint stem stored in the summary; fall back to the
        # directory layout for older summaries.
        ckpt_name = Path(stage2_ckpt).stem if stage2_ckpt else ""
        if not ckpt_name and keyword_dir and summary_path.parent.name == keyword_dir:
            ckpt_name = summary_path.parent.parent.name
        if not ckpt_name:
            ckpt_name = summary_path.parent.name

        row: dict = {
            "keyword": keyword,
            "ckpt_name": ckpt_name,
            "output_dir": str(summary_path.parent),
            "total_files": summary.get("total_files", 0),
            "total_hours": summary.get("total_hours", 0.0),
            "num_samples": summary.get("num_samples", 0),
        }
        metrics = summary.get("metrics", {})
        row.update(
            {
                "fa_per_hour": metrics.get("fa_per_hour", 0.0),
                "fa_per_1000_hours": metrics.get("fa_per_1000_hours", 0.0),
                "fpr": metrics.get("fpr", 0.0),
                "fp": int(metrics.get("fp", 0)),
                "tn": int(metrics.get("tn", 0)),
            }
        )
        for subset, subset_data in summary.get("subsets", {}).items():
            sm = subset_data.get("metrics", {})
            row[f"subset_{subset}_fa_per_hour"] = sm.get("fa_per_hour", 0.0)
            row[f"subset_{subset}_total_hours"] = subset_data.get("total_hours", 0.0)
        rows.append(row)
    return rows


def write_tsv(rows: list[dict], out_path: Path) -> None:
    if not rows:
        print("No summary.json files found for aggregation.", file=sys.stderr)
        return

    # Use the union of keys so subset columns from any row are preserved.
    columns = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                columns.append(key)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        handle.write("\t".join(columns) + "\n")
        for row in rows:
            handle.write(
                "\t".join(str(row.get(col, "")) for col in columns) + "\n"
            )
    print(f"Wrote aggregated summary -> {out_path}")


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("Usage: python3 aggregate_musan_fa.py ROOT_DIR OUT_TSV")
    root = Path(sys.argv[1])
    out_path = Path(sys.argv[2])
    rows = aggregate(root)
    write_tsv(rows, out_path)


if __name__ == "__main__":
    main()
