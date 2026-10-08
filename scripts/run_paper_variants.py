#!/usr/bin/env python3
"""Isolate the two paper knobs after P1 collapsed.

P1 (unfrozen + hard_negative_ratio 100 + lr 1e-4, from scratch) aborted at step
5000 with AUC 0.672 vs C1's 0.857. Two factors were changed at once, so this
launcher runs them separately, both from scratch, with step-dependent floors:

  P2  encoder frozen,   hard_negative_ratio 100, lr 5e-4   (the paper mixture)
  P3  encoder unfrozen, hard_negative_ratio 1,   lr 2.5e-4 (the encoder fine-tune)
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path("/home/ubuntu/dma-kws")
PY = str(ROOT / ".venv/bin/python")
OUT = ROOT / "outputs/v41_paper_variants"
RUN_ROOT = "data/dma-kws/exp/stage2_qbyt/final"
EXP = "icefall_zipformer_stage2_eps_softmin_v41_musan_cached_zhen3m_50k"

GROUPS = {
    "P2-hardneg100-frozen": [
        "stage2.freeze_encoder=true",
        "stage2.hard_negative_ratio=100",
        "stage2.learning_rate=5e-4",
    ],
    "P3-unfreeze-1to1": [
        "stage2.freeze_encoder=false",
        "stage2.hard_negative_ratio=1",
        "stage2.learning_rate=2.5e-4",
    ],
}
FLOORS = [(2500, 0.70), (5000, 0.78), (10000, 0.84), (20000, 0.87), (30000, 0.89), (40000, 0.905)]


def command(name: str, extra: list[str]) -> list[str]:
    run_dir = f"{RUN_ROOT}/{name}"
    return [
        PY,
        "scripts/train_stage2_qbyt.py",
        f"+experiment={EXP}",
        "+stage2.qbyt_readout.sink_readout=additive",
        "+stage2.qbyt_readout.sink_zero_init=true",
        "stage2.warmup_steps=500",
        "stage2.max_steps=50000",
        "stage2.total_scheduler_steps=50000",
        "stage2.validation.val_check_interval=2500",
        "stage2.val_check_interval=2500",
        "stage2.num_workers=2",
        f"stage2.run_name={name}",
        f"stage2.checkpoint_dir={run_dir}/checkpoints",
        f"stage2.log_dir={run_dir}/logs",
        *extra,
    ]


def latest_val(name: str):
    paths = sorted((ROOT / RUN_ROOT / name / "logs").glob("*/version_*/metrics.csv"))
    if not paths:
        return None
    try:
        frame = pd.read_csv(paths[-1])
    except Exception:
        return None
    if "val/auc" not in frame.columns:
        return None
    rows = frame[frame["val/auc"].notna()]
    if rows.empty:
        return None
    last = rows.iloc[-1]
    return int(last["step"]), float(last["val/auc"])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="0")
    procs = {}
    logs = {}
    for name, extra in GROUPS.items():
        log = (OUT / f"{name}.log").open("w")
        procs[name] = subprocess.Popen(
            command(name, extra), cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT, env=env
        )
        logs[name] = log
        print("launched", name, flush=True)
    verdicts = {}
    while any(p.poll() is None for p in procs.values()):
        time.sleep(300)
        for name, proc in procs.items():
            if proc.poll() is not None:
                continue
            point = latest_val(name)
            if point is None:
                continue
            step, auc = point
            print("progress", name, step, "%.5f" % auc, flush=True)
            for floor_step, floor in FLOORS:
                if step >= floor_step and auc < floor:
                    proc.send_signal(signal.SIGTERM)
                    verdicts[name] = f"aborted at step {step}: AUC {auc:.5f} < floor {floor}"
                    print("ABORT", name, verdicts[name], flush=True)
                    break
    results = {}
    for name, proc in procs.items():
        logs[name].close()
        point = latest_val(name)
        results[name] = {
            "verdict": verdicts.get(name, "ok" if proc.returncode == 0 else f"exit={proc.returncode}"),
            "step": point[0] if point else None,
            "auc": point[1] if point else None,
        }
        print("finished", name, results[name], flush=True)
    (OUT / "results.json").write_text(json.dumps(results, indent=1))
    print("PAPERVARIANTSDONE", json.dumps(results), flush=True)


if __name__ == "__main__":
    main()
