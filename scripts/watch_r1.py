#!/usr/bin/env python3
"""Guard for R1: kill the run if it regresses against its own starting base.

The pipeline monitor compares R1 with C1's from-scratch curve, so it cannot see
a slow regression below the base model. This watcher reads R1's val rows every
5 minutes and terminates the training when either
  * two consecutive val AUCs are below 0.925, or
  * any single val AUC is below 0.915.
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

import pandas as pd

ROOT = Path("/home/ubuntu/dma-kws")
METRICS = "data/dma-kws/exp/stage2_qbyt/final/R1-unfreeze-hardneg100/logs/R1-unfreeze-hardneg100/version_0/metrics.csv"


def val_aucs():
    frame = pd.read_csv(ROOT / METRICS)
    rows = frame[frame["val/auc"].notna()]
    return [float(v) for v in rows["val/auc"]]


def kill_training(reason: str) -> None:
    out = subprocess.run(
        ["pgrep", "-f", "run_name=R1-unfreeze-hardneg100"], capture_output=True, text=True
    )
    for pid in out.stdout.split():
        subprocess.run(["kill", "-TERM", pid])
    print("KILLED R1:", reason, flush=True)


def main() -> None:
    while True:
        try:
            aucs = val_aucs()
        except Exception as exc:
            print("read error", exc, flush=True)
            aucs = []
        if aucs:
            print("val aucs:", ["%.5f" % a for a in aucs[-5:]], flush=True)
            if aucs[-1] < 0.915:
                kill_training(f"AUC {aucs[-1]:.5f} < 0.915")
                return
            if len(aucs) >= 2 and aucs[-1] < 0.925 and aucs[-2] < 0.925:
                kill_training(f"two consecutive AUCs below 0.925 ({aucs[-2]:.5f}, {aucs[-1]:.5f})")
                return
        check = subprocess.run(
            ["pgrep", "-f", "run_name=R1-unfreeze-hardneg100"], capture_output=True, text=True
        )
        if not check.stdout.strip():
            print("R1 finished, watcher exits", flush=True)
            return
        time.sleep(300)


if __name__ == "__main__":
    main()
