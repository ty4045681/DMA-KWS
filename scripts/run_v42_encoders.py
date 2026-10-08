#!/usr/bin/env python3
"""Launch and babysit the four v4.2 encoder runs (paperstage1 + 3 GigaSpeech).

All four use the C1 recipe: the matching v4.1 overlay + additive-zero sink
readout (no sink loss), 50k steps from scratch, zh-en/GigaSpeech frozen
encoders. They run concurrently on the one V100; aggregate throughput is the
machine's ceiling either way, so one wave finishes everything fastest.

Early abort: per-encoder hard floors (well below each encoder's v4.1 curve)
plus NaN checks; a violating run is terminated and recorded, the others keep
going. Status: outputs/v41_encoders/status.json.
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
OUT = ROOT / "outputs/v41_encoders"
RUN_ROOT = "data/dma-kws/exp/stage2_qbyt/v42_encoders"
STATUS = OUT / "status.json"

GROUPS = {
    "V42-paperstage1": {
        "exp": "icefall_zipformer_stage2_eps_softmin_v41_musan_cached_paperstage1_50k",
        "floors": [(10000, 0.75), (20000, 0.85), (30000, 0.88), (40000, 0.90)],
    },
    "V42-gsfinetune": {
        "exp": "icefall_zipformer_stage2_eps_softmin_v41_musan_cached_gs_50k",
        "floors": [(10000, 0.70), (20000, 0.76), (30000, 0.80), (40000, 0.83)],
    },
    "V42-gsbase-stream": {
        "exp": "icefall_zipformer_stage2_eps_softmin_v41_musan_cached_gsbase_stream_50k",
        "floors": [(10000, 0.75), (20000, 0.84), (30000, 0.87), (40000, 0.89)],
    },
    "V42-gsbase-fullctx": {
        "exp": "icefall_zipformer_stage2_eps_softmin_v41_musan_cached_gsbase_fullctx_50k",
        "floors": [(10000, 0.75), (20000, 0.84), (30000, 0.87), (40000, 0.89)],
    },
}


def command(name: str, exp: str) -> list[str]:
    run_dir = f"{RUN_ROOT}/{name}"
    return [
        PY,
        "scripts/train_stage2_qbyt.py",
        f"+experiment={exp}",
        "+stage2.qbyt_readout.sink_readout=additive",
        "+stage2.qbyt_readout.sink_zero_init=true",
        "stage2.learning_rate=5e-4",
        "stage2.warmup_steps=500",
        "stage2.max_steps=50000",
        "stage2.total_scheduler_steps=50000",
        "stage2.validation.val_check_interval=2500",
        "stage2.val_check_interval=2500",
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
    selected = sys.argv[1:] or list(GROUPS)
    procs = {}
    logs = {}
    for name in selected:
        cfg = GROUPS[name]
        log = (OUT / f"{name}.log").open("w")
        procs[name] = subprocess.Popen(
            command(name, cfg["exp"]),
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
        time.sleep(300)
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
                continue
            for floor_step, floor in GROUPS[name]["floors"]:
                if step >= floor_step and auc < floor:
                    proc.send_signal(signal.SIGTERM)
                    verdicts[name] = f"aborted: AUC {auc:.4f} below floor {floor} at step {step}"
                    print("ABORT", name, verdicts[name], flush=True)
                    break

    for name, proc in procs.items():
        logs[name].close()
        if name not in verdicts:
            verdicts[name] = "ok" if proc.returncode == 0 else f"exit={proc.returncode}"
        print("finished", name, verdicts[name], flush=True)
    write_status({name: {"state": "done", "verdict": verdicts[name]} for name in selected})
    write_status({"all_done": True, "verdicts": verdicts})
    print("ALLDONE", json.dumps(verdicts), flush=True)


if __name__ == "__main__":
    main()
