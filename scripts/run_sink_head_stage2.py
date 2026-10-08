#!/usr/bin/env python3
"""Second-stage sink-head fine-tunes: freeze the whole head, train only sink_fc
+ sink_alpha with the dedicated sink loss.

Bases:
  SS-zh-en            C1 50k export (best text head, zh-en-3M)
  SS-gsfinetune       v4.1 gs-finetune step 45000 export
  SS-gsbase-stream    v4.1 gs-base stream step 45000 export
  SS-gsbase-fullctx   v4.1 gs-base fullctx step 45000 export

Each run: 3000 steps, LR 1e-3, sink_loss bce 0.25, score temperature left at
training value (final scoring temperature applied at evaluation time).
Runs concurrently with the paperstage1 50k training.
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
OUT = ROOT / "outputs/v41_sinkhead_stage2"
RUN_ROOT = "data/dma-kws/exp/stage2_qbyt/sinkhead_stage2"
STATUS = OUT / "status.json"

GROUPS = {
    "SS-zh-en": {
        "exp": "icefall_zipformer_stage2_eps_softmin_v41_musan_cached_zhen3m_50k",
        "base": "data/dma-kws/exp/stage2_qbyt/final/C1-sink-50k/checkpoints/C1-sink-50k/version_0/stage2_step050000.pt",
        "floor": 0.925,
    },
    "SS-gsfinetune": {
        "exp": "icefall_zipformer_stage2_eps_softmin_v41_musan_cached_gs_50k",
        "base": "data/dma-kws/exp/stage2_qbyt/checkpoints/v41-musan-gs-50k/v41-musan-gs-50k/version_0/stage2_step045000.pt",
        "floor": 0.85,
    },
    "SS-gsbase-stream": {
        "exp": "icefall_zipformer_stage2_eps_softmin_v41_musan_cached_gsbase_stream_50k",
        "base": "data/dma-kws/exp/stage2_qbyt/checkpoints/v41-musan-gsbase-stream-50k/v41-musan-gsbase-stream-50k/version_0/stage2_step045000.pt",
        "floor": 0.90,
    },
    "SS-gsbase-fullctx": {
        "exp": "icefall_zipformer_stage2_eps_softmin_v41_musan_cached_gsbase_fullctx_50k",
        "base": "data/dma-kws/exp/stage2_qbyt/checkpoints/v41-musan-gsbase-fullctx-50k/v41-musan-gsbase-fullctx-50k/version_1/stage2_step045000.pt",
        "floor": 0.90,
    },
}


def command(name: str, cfg: dict) -> list[str]:
    run_dir = f"{RUN_ROOT}/{name}"
    return [
        PY,
        "scripts/train_stage2_qbyt.py",
        f"+experiment={cfg['exp']}",
        "+stage2.qbyt_readout.sink_readout=additive",
        "+stage2.qbyt_readout.sink_zero_init=true",
        "+stage2.freeze_all_but_sink=true",
        "stage2.sink_loss.enabled=true",
        "stage2.sink_loss.weight=0.25",
        f"stage2.init_checkpoint={cfg['base']}",
        "+stage2.init_allow_readout_mismatch=true",
        "stage2.learning_rate=1e-3",
        "stage2.warmup_steps=200",
        "stage2.max_steps=3000",
        "stage2.total_scheduler_steps=3000",
        "stage2.validation.val_check_interval=1500",
        "stage2.val_check_interval=1500",
        "stage2.num_workers=2",
        f"stage2.run_name={name}",
        f"stage2.checkpoint_dir={run_dir}/checkpoints",
        f"stage2.log_dir={run_dir}/logs",
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


def write_status(entries: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    current = {}
    if STATUS.exists():
        try:
            current = json.loads(STATUS.read_text())
        except ValueError:
            current = {}
    current.update(entries)
    current["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
    STATUS.write_text(json.dumps(current, indent=1, default=str))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="0")
    procs = {}
    logs = {}
    for name, cfg in GROUPS.items():
        log = (OUT / f"{name}.log").open("w")
        procs[name] = subprocess.Popen(
            command(name, cfg),
            cwd=str(ROOT),
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
        )
        logs[name] = log
        print("launched", name, flush=True)
    write_status({name: {"state": "running"} for name in GROUPS})

    verdicts = {}
    while any(p.poll() is None for p in procs.values()):
        time.sleep(120)
        for name, proc in procs.items():
            if proc.poll() is not None:
                continue
            point = latest_val(name)
            if point is None:
                continue
            step, auc = point
            write_status({name: {"state": "running", "step": step, "auc": auc}})
            if auc != auc:
                proc.send_signal(signal.SIGTERM)
                verdicts[name] = "aborted: NaN AUC"
                print("ABORT", name, verdicts[name], flush=True)
            elif auc < GROUPS[name]["floor"]:
                proc.send_signal(signal.SIGTERM)
                verdicts[name] = f"aborted: AUC {auc:.4f} below floor {GROUPS[name]['floor']}"
                print("ABORT", name, verdicts[name], flush=True)

    for name, proc in procs.items():
        logs[name].close()
        if name not in verdicts:
            verdicts[name] = "ok" if proc.returncode == 0 else f"exit={proc.returncode}"
        print("finished", name, verdicts[name], flush=True)
    write_status({name: {"state": "done", "verdict": verdicts[name]} for name in GROUPS})
    write_status({"all_done": True, "verdicts": verdicts})
    print("ALLDONE", json.dumps(verdicts), flush=True)


if __name__ == "__main__":
    main()
