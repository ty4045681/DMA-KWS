#!/usr/bin/env python3
"""Launch the A0-A5 warm-start ablations for the QbyT v4.1 sink head.

Every group continues from the same v4.1 47.5k checkpoint for 2,000 steps
(LR 5e-5, warmup 100, cosine to 0), validates once on the full LibriPhrase
hard split at step 2000, and exports the best checkpoint.

Usage (env must be sourced):
  source ~/.dma-kws-env.sh
  .venv/bin/python scripts/run_abl_a0_a5.py
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
    "data/dma-kws/exp/stage2_qbyt/checkpoints/v41-musan-zhen3m-50k/"
    "v41-musan-zhen3m-50k/version_9/stage2_step047500.pt"
)
OUT = ROOT / "outputs/v41_ablations"
RUN_ROOT = "data/dma-kws/exp/stage2_qbyt/ablations"

GROUPS: dict[str, list[str]] = {
    "A0-control": [],
    "A1-sink-additive": ["+stage2.qbyt_readout.sink_readout=additive"],
    "A2-sink-mixture": ["+stage2.qbyt_readout.sink_readout=mixture"],
    "A3-temp-fixed-0p1": ["stage2.qbyt_readout.temperature=0.1"],
    "A4-temp-learnable": [
        "stage2.qbyt_readout.temperature=0.1",
        "+stage2.qbyt_readout.temperature_learnable=true",
    ],
    "A5-sink-identity": ["+stage2.qbyt_readout.sink_identity=true"],
}


def command(name: str, extra: list[str]) -> list[str]:
    run_dir = f"{RUN_ROOT}/{name}"
    return [
        sys.executable,
        "scripts/train_stage2_qbyt.py",
        f"+experiment={EXP}",
        f"stage2.init_checkpoint={CKPT}",
        "+stage2.init_allow_readout_mismatch=true",
        "stage2.learning_rate=5e-5",
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
        proc = subprocess.Popen(
            command(name, extra),
            cwd=str(ROOT),
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
        )
        procs[name] = (proc, log)
        print("launched", name, flush=True)
    codes: dict[str, int] = {}
    for name, (proc, log) in procs.items():
        code = proc.wait()
        log.close()
        codes[name] = int(code)
        print("finished", name, "exit", code, flush=True)
    (OUT / "exit_codes.json").write_text(json.dumps(codes, indent=1))
    print("ALLDONE", json.dumps(codes), flush=True)


if __name__ == "__main__":
    main()
