#!/usr/bin/env python3
"""Fast ablations for the zh-en v4.2 model: unfreeze the encoder and/or use the
paper hard-negative sampling ratio (100:1).

All arms warm-start from the current v4.2 checkpoint (SS-zh-en, 3k sink-head
second stage, hard-split AUC 0.94106) with the sink loss still on, 3k steps:

  U1  encoder unfrozen, lr 2e-5
  U2  encoder unfrozen, lr 5e-5
  H1  encoder frozen,   hard_negative_ratio 100, lr 2e-4
  H2  encoder unfrozen, hard_negative_ratio 100, lr 5e-5

Validation is the full hard split (2.5k-step cadence in the base recipe, here
every 1.5k steps). A NaN or a val AUC below 0.92 kills the arm.
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
OUT = ROOT / "outputs/v41_improve"
RUN_ROOT = "data/dma-kws/exp/stage2_qbyt/improve"
EXP = "icefall_zipformer_stage2_eps_softmin_v41_musan_cached_zhen3m_50k"
BASE = "data/dma-kws/exp/stage2_qbyt/sinkhead_stage2/SS-zh-en/checkpoints/SS-zh-en/version_0/stage2_step003000.pt"

GROUPS = {
    "U1-unfreeze-lr2e5": [
        "stage2.freeze_encoder=false",
        "stage2.learning_rate=2e-5",
    ],
    "U2-unfreeze-lr5e5": [
        "stage2.freeze_encoder=false",
        "stage2.learning_rate=5e-5",
    ],
    "H1-hardneg100": [
        "stage2.freeze_encoder=true",
        "stage2.hard_negative_ratio=100",
        "stage2.learning_rate=2e-4",
    ],
    "H2-hardneg100-unfreeze": [
        "stage2.freeze_encoder=false",
        "stage2.hard_negative_ratio=100",
        "stage2.learning_rate=5e-5",
    ],
}
FLOOR = 0.92


def command(name: str, extra: list[str]) -> list[str]:
    run_dir = f"{RUN_ROOT}/{name}"
    return [
        PY,
        "scripts/train_stage2_qbyt.py",
        f"+experiment={EXP}",
        "+stage2.qbyt_readout.sink_readout=additive",
        "+stage2.qbyt_readout.sink_zero_init=true",
        "stage2.sink_loss.enabled=true",
        "stage2.sink_loss.weight=0.25",
        f"stage2.init_checkpoint={BASE}",
        "+stage2.init_allow_readout_mismatch=true",
        "stage2.warmup_steps=100",
        "stage2.max_steps=3000",
        "stage2.total_scheduler_steps=3000",
        "stage2.validation.val_check_interval=1500",
        "stage2.val_check_interval=1500",
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
    return int(last["step"]), float(last["val/auc"]), float(last["val/eer"])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="0")
    procs = {}
    logs = {}
    for name, extra in GROUPS.items():
        log = (OUT / f"{name}.log").open("w")
        procs[name] = subprocess.Popen(
            command(name, extra),
            cwd=str(ROOT),
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
        )
        logs[name] = log
        print("launched", name, flush=True)
    verdicts = {}
    while any(p.poll() is None for p in procs.values()):
        time.sleep(120)
        for name, proc in procs.items():
            if proc.poll() is not None:
                continue
            point = latest_val(name)
            if point is None:
                continue
            step, auc, eer = point
            print("progress", name, step, "%.5f" % auc, "%.5f" % eer, flush=True)
            if auc != auc or auc < FLOOR:
                proc.send_signal(signal.SIGTERM)
                verdicts[name] = f"aborted at step {step}: AUC {auc:.5f}"
                print("ABORT", name, verdicts[name], flush=True)
    results = {}
    for name, proc in procs.items():
        logs[name].close()
        point = latest_val(name)
        results[name] = {
            "verdict": verdicts.get(name, "ok" if proc.returncode == 0 else f"exit={proc.returncode}"),
            "step": point[0] if point else None,
            "auc": point[1] if point else None,
            "eer": point[2] if point else None,
        }
        print("finished", name, results[name], flush=True)
    (OUT / "results.json").write_text(json.dumps(results, indent=1))
    print("IMPROVEDONE", json.dumps(results), flush=True)


if __name__ == "__main__":
    main()
