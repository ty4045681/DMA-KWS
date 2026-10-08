#!/usr/bin/env python3
"""Sink-loss round S: 3-group warm-start A/B from the C1 50k checkpoint.

S0  control            - continuation only (drift at LR 2e-4)
S1  sink_loss bce 0.25
S2  sink_loss bce 0.50

All runs start from the same C1 export, so the comparison is paired. Runs in
parallel with whatever else is on the GPU (num_workers=2 to limit CPU).

Usage (env sourced):
  .venv/bin/python scripts/run_sinkloss_ab.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path("/home/ubuntu/dma-kws")
EXP = "icefall_zipformer_stage2_eps_softmin_v41_musan_cached_zhen3m_50k"
CKPT = (
    "data/dma-kws/exp/stage2_qbyt/final/C1-sink-50k/"
    "checkpoints/C1-sink-50k/version_0/stage2_step050000.pt"
)
OUT = ROOT / "outputs/v41_sinkloss_ab"
RUN_ROOT = "data/dma-kws/exp/stage2_qbyt/sinkloss_ab"

GROUPS: dict[str, list[str]] = {
    "S0-control": [],
    "S1-bce-0p25": [
        "stage2.sink_loss.enabled=true",
        "stage2.sink_loss.weight=0.25",
        "stage2.sink_loss.form=bce",
    ],
    "S2-bce-0p5": [
        "stage2.sink_loss.enabled=true",
        "stage2.sink_loss.weight=0.5",
        "stage2.sink_loss.form=bce",
    ],
}


def command(name: str, extra: list[str]) -> list[str]:
    run_dir = f"{RUN_ROOT}/{name}"
    return [
        sys.executable,
        "scripts/train_stage2_qbyt.py",
        f"+experiment={EXP}",
        "+stage2.qbyt_readout.sink_readout=additive",
        "+stage2.qbyt_readout.sink_zero_init=true",
        f"stage2.init_checkpoint={CKPT}",
        "+stage2.init_allow_readout_mismatch=true",
        "stage2.learning_rate=2e-4",
        "stage2.warmup_steps=100",
        "stage2.max_steps=2000",
        "stage2.total_scheduler_steps=2000",
        "stage2.validation.val_check_interval=2000",
        "stage2.val_check_interval=2000",
        "stage2.num_workers=2",
        "stage2.checkpoint.every_n_train_steps=1000",
        "stage2.checkpoint.save_top_k=1",
        f"stage2.run_name={name}",
        f"stage2.checkpoint_dir={run_dir}/checkpoints",
        f"stage2.log_dir={run_dir}/logs",
        *extra,
    ]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="0")
    procs = {}
    for name, extra in GROUPS.items():
        log = (OUT / f"{name}.log").open("w")
        procs[name] = (
            subprocess.Popen(
                command(name, extra),
                cwd=str(ROOT),
                stdout=log,
                stderr=subprocess.STDOUT,
                env=env,
            ),
            log,
        )
        print("launched", name, flush=True)
    codes = {}
    for name, (proc, log) in procs.items():
        code = proc.wait()
        log.close()
        codes[name] = int(code)
        print("finished", name, "exit", code, flush=True)
    (OUT / "exit_codes.json").write_text(json.dumps(codes, indent=1))
    print("ALLDONE", json.dumps(codes), flush=True)


if __name__ == "__main__":
    main()
