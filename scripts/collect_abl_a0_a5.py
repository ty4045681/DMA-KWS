#!/usr/bin/env python3
"""Collect A0-A5 ablation validation metrics and compare against A0."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

ROOT = Path("/home/ubuntu/dma-kws")
ABL = Path(
    os.environ.get("ABL_ROOT", str(ROOT / "data/dma-kws/exp/stage2_qbyt/ablations"))
)
GROUPS = os.environ.get("ABL_GROUPS", "").split(",") or [
    "A0-control",
    "A1-sink-additive",
    "A2-sink-mixture",
    "A3-temp-fixed-0p1",
    "A4-temp-learnable",
    "A5-sink-identity",
]
GROUPS = [g for g in GROUPS if g]
CONTROL = os.environ.get("ABL_CONTROL", GROUPS[0])
KEYS = [
    "val/auc",
    "val/eer",
    "val/tpr_at_fpr_1e_2",
    "val/tpr_at_fpr_1e_3",
    "val/pauc_fpr_1e_2",
    "val/brier",
    "val/ece",
]


def load(group: str) -> dict:
    logs = sorted((ABL / group / "logs").glob("*/version_*/metrics.csv"))
    if not logs:
        return {"status": "missing"}
    frame = pd.read_csv(logs[-1])
    val_columns = [c for c in frame.columns if c.startswith("val/")]
    if "val/auc" not in frame.columns:
        return {"status": "no-validation", "columns": val_columns[:20]}
    rows = frame[frame["val/auc"].notna()]
    if rows.empty:
        return {"status": "validation-running", "columns": val_columns[:24]}
    last = rows.iloc[-1]
    out = {"status": "ok", "step": int(last.get("step", -1))}
    for key in KEYS:
        if key in frame.columns:
            out[key] = float(last[key])
    return out


def main() -> None:
    results = {group: load(group) for group in GROUPS}
    base = results.get(CONTROL, {})
    table = []
    for group in GROUPS:
        entry = dict(results[group])
        entry["group"] = group
        if entry.get("status") == "ok" and base.get("status") == "ok":
            for key in KEYS:
                if key in entry and key in base:
                    entry["delta_" + key] = entry[key] - base[key]
        table.append(entry)
    print(json.dumps({g: results[g] for g in GROUPS}, indent=1))
    out = Path(os.environ.get("ABL_OUT", str(ROOT / "outputs/v41_ablations/collect.json")))
    out.write_text(json.dumps(table, indent=1))
    print("wrote", out)


if __name__ == "__main__":
    main()
