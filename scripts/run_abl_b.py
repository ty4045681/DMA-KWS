#!/usr/bin/env python3
"""Round B: corrected sink-readout ablations (zero-init branch, 10x LR).

Round A showed the zero-initialised new heads barely moved at LR 5e-5 for 2,000
steps (sink_alpha +0.003, T unchanged, identity norm 0.02). Round B fixes the
parameterisation: the additive branch starts as a zero-weight linear layer with
alpha=1 (step-0 output identical, gradients reach the sink head immediately) and
uses LR 2e-4 with a matched control to absorb any drift of the pre-trained head.

Usage (env must be sourced):
  source ~/.dma-kws-env.sh
  .venv/bin/python scripts/run_abl_b.py
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
OUT = ROOT / "outputs/v41_ablations_b"
RUN_ROOT = "data/dma-kws/exp/stage2_qbyt/ablations_b"

ADDITIVE_ZERO = [
    "+stage2.qbyt_readout.sink_readout=additive",
    "+stage2.qbyt_readout.sink_zero_init=true",
]
MIXTURE_ZERO = [
    "+stage2.qbyt_readout.sink_readout=mixture",
    "+stage2.qbyt_readout.sink_zero_init=true",
]

GROUPS: dict[str, list[str]] = {
    "B0-control-lr2e4": [],
    "B1-additive-zero": ADDITIVE_ZERO,
    "B2-mixture-zero": MIXTURE_ZERO,
    "B3-additive-zero-temp01": ADDITIVE_ZERO + ["stage2.qbyt_readout.temperature=0.1"],
}


def command(name: str, extra: list[str]) -> list[str]:
    run_dir = f"{RUN_ROOT}/{name}"
    return [
        sys.executable,
        "scripts/train_stage2_qbyt.py",
        f"+experiment={EXP}",
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
