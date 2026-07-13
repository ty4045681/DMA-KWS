#!/usr/bin/env python3
"""Collect batch_eval_stage2_clips.sh summary.json metrics into an Excel workbook.

Each per-out directory (one per checkpoint dir, containing <ckpt_name>/summary.json
subdirectories) becomes one sheet. Rows are checkpoints sorted by training step;
columns are: steps, recall, fpr (fp / (fp + tn)), then the remaining metrics.

Usage:
    python3 scripts/make_stage2_eval_excel.py \
        --out stage2_eval.xlsx \
        /path/to/per_out1 /path/to/per_out2 ...
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

_INVALID_SHEET_CHARS = re.compile(r"[\[\]:*?/\\]")
_MAX_SHEET_NAME_LEN = 31
_STEP_PATTERN = re.compile(r"(\d+)")


def _sheet_name(raw: str, used: set[str]) -> str:
    name = _INVALID_SHEET_CHARS.sub("_", raw).strip() or "sheet"
    name = name[:_MAX_SHEET_NAME_LEN]
    if name not in used:
        used.add(name)
        return name
    for idx in range(2, 1000):
        suffix = f"_{idx}"
        candidate = name[: _MAX_SHEET_NAME_LEN - len(suffix)] + suffix
        if candidate not in used:
            used.add(candidate)
            return candidate
    raise SystemExit(f"Could not derive a unique sheet name for {raw!r}")


def _extract_steps(name: str) -> int | None:
    matches = _STEP_PATTERN.findall(name)
    if not matches:
        return None
    return int(matches[-1])


def _false_positive_rate(metrics: dict) -> float:
    fp = float(metrics.get("fp", 0.0))
    tn = float(metrics.get("tn", 0.0))
    negatives = fp + tn
    if negatives == 0:
        return 0.0
    return fp / negatives


def _collect_rows(per_out: Path) -> pd.DataFrame:
    rows: list[dict] = []
    for summary_path in sorted(per_out.glob("*/summary.json")):
        with summary_path.open("r", encoding="utf-8") as handle:
            summary = json.load(handle)
        metrics = summary.get("metrics")
        if not isinstance(metrics, dict):
            print(f"WARNING: no metrics in {summary_path}, skipping.", file=sys.stderr)
            continue
        ckpt_name = summary_path.parent.name
        steps = _extract_steps(ckpt_name)
        row: dict = {"steps": steps if steps is not None else ckpt_name}
        row["recall"] = metrics.get("recall")
        row["fpr"] = _false_positive_rate(metrics)
        for key, value in metrics.items():
            if key not in ("recall",):
                row[key] = value
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    frame = pd.DataFrame(rows)
    frame = frame.sort_values(
        by="steps", key=lambda col: pd.to_numeric(col, errors="coerce")
    ).reset_index(drop=True)
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate stage2 eval summary.json files into an Excel workbook."
    )
    parser.add_argument(
        "per_out_dirs",
        nargs="+",
        type=Path,
        help="Per-out directories (each containing <ckpt_name>/summary.json subdirs); one sheet each.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("stage2_eval.xlsx"),
        help="Output .xlsx path (default: stage2_eval.xlsx)",
    )
    args = parser.parse_args()

    used_names: set[str] = set()
    sheets: list[tuple[str, pd.DataFrame]] = []
    for per_out in args.per_out_dirs:
        if not per_out.is_dir():
            print(f"WARNING: not a directory, skipping: {per_out}", file=sys.stderr)
            continue
        frame = _collect_rows(per_out)
        if frame.empty:
            print(f"WARNING: no summary.json found under {per_out}, skipping.", file=sys.stderr)
            continue
        sheets.append((_sheet_name(per_out.name, used_names), frame))

    if not sheets:
        raise SystemExit("No metrics collected; nothing to write.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(args.out, engine="openpyxl") as writer:
        for name, frame in sheets:
            frame.to_excel(writer, sheet_name=name, index=False)

    total_rows = sum(len(frame) for _, frame in sheets)
    print(f"Wrote {len(sheets)} sheet(s), {total_rows} row(s) -> {args.out}")


if __name__ == "__main__":
    main()
