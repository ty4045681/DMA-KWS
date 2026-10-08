#!/usr/bin/env python3
"""Queue and babysit the C1/C2 full 50k runs with early-abort monitoring.

Chain:
  1. wait for the round-B launcher to exit
  2. read round-B metrics, pick the C1 readout configuration
  3. C1 = 50k from scratch (v4.1 recipe + winning sink readout)
  4. offline probe + temperature analysis of C1
  5. C2 = 50k from scratch, same as C1 but text_position=sinusoidal
  6. offline probe + temperature analysis of C2

Monitoring: every 3 minutes the validation rows in metrics.csv are compared
against hard thresholds (calibrated to be well below the v2/v4.1 reference
curves) and against the reference curve itself. A run that breaks the hard
threshold, produces NaN, or tracks far below the reference is terminated
immediately, so a broken 50k run costs minutes instead of hours.

Status is written to outputs/v41_final/status.json after every event.
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
EXP = "icefall_zipformer_stage2_eps_softmin_v41_musan_cached_zhen3m_50k"
FINAL = ROOT / "outputs/v41_final"
RUN_ROOT = "data/dma-kws/exp/stage2_qbyt/final"
STATUS = FINAL / "status.json"

# v4.1 reference validation curve (same encoder + recipe, no sink readout).
REF_V41 = [
    (2500, 0.80), (5000, 0.85), (7500, 0.8740), (10000, 0.8854),
    (15000, 0.9010), (20000, 0.9092), (25000, 0.9203), (30000, 0.9222),
    (35000, 0.9275), (40000, 0.9330), (45000, 0.9342), (50000, 0.9348),
]
# Hard floors: clearly broken runs only.
HARD = [(2500, 0.74), (5000, 0.79), (10000, 0.84), (20000, 0.87), (30000, 0.89), (40000, 0.905)]
WARN_DROP = 0.010
ABORT_DROP = 0.020


def log(message: str) -> None:
    print(time.strftime("%H:%M:%S"), message, flush=True)


def write_status(**fields) -> None:
    FINAL.mkdir(parents=True, exist_ok=True)
    current = {}
    if STATUS.exists():
        try:
            current = json.loads(STATUS.read_text())
        except ValueError:
            current = {}
    current.update(fields)
    current["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
    STATUS.write_text(json.dumps(current, indent=1, default=str))


def wait_for_pid(pid: int) -> None:
    while True:
        try:
            os.kill(pid, 0)
        except OSError:
            return
        time.sleep(30)


def collect_round_b() -> dict:
    groups = "B0-control-lr2e4,B1-additive-zero,B2-mixture-zero,B3-additive-zero-temp01"
    env = dict(os.environ)
    env.update(
        {
            "ABL_ROOT": str(ROOT / "data/dma-kws/exp/stage2_qbyt/ablations_b"),
            "ABL_GROUPS": groups,
            "ABL_CONTROL": "B0-control-lr2e4",
            "ABL_OUT": str(FINAL / "collect_b.json"),
        }
    )
    subprocess.run(
        [PY, "scripts/collect_abl_a0_a5.py"],
        cwd=str(ROOT),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
        check=False,
    )
    path = FINAL / "collect_b.json"
    if not path.exists():
        return {}
    rows = json.loads(path.read_text())
    out = {}
    for row in rows:
        out[row["group"]] = row
    return out


def pick_c1_flags(rows: dict) -> tuple[list[str], str]:
    """Pick the C1 readout flags from the round-B comparison."""

    base = rows.get("B0-control-lr2e4", {})
    base_auc = base.get("val/auc")
    candidates = []
    for name in ("B1-additive-zero", "B2-mixture-zero", "B3-additive-zero-temp01"):
        row = rows.get(name)
        if not row or row.get("status") != "ok" or base_auc is None:
            continue
        dauc = row["val/auc"] - base_auc
        dtpr = row.get("val/tpr_at_fpr_1e_2", 0.0) - base.get("val/tpr_at_fpr_1e_2", 0.0)
        candidates.append((name, dauc, dtpr))
    usable = [c for c in candidates if c[1] > -0.002]
    if not usable:
        return (
            ["+stage2.qbyt_readout.sink_readout=additive", "+stage2.qbyt_readout.sink_zero_init=true"],
            "no round-B group usable; default additive-zero",
        )
    usable.sort(key=lambda c: max(c[1], c[2] / 10.0), reverse=True)
    winner = usable[0]
    if winner[0] == "B3-additive-zero-temp01":
        # S8b: training at low T did not reproduce the inference-time gain; only
        # keep it if it clearly beats plain additive-zero.
        b1 = [c for c in usable if c[0] == "B1-additive-zero"]
        if b1 and winner[1] - b1[0][1] < 0.0015:
            winner = b1[0]
    flags = {
        "B1-additive-zero": [
            "+stage2.qbyt_readout.sink_readout=additive",
            "+stage2.qbyt_readout.sink_zero_init=true",
        ],
        "B2-mixture-zero": [
            "+stage2.qbyt_readout.sink_readout=mixture",
            "+stage2.qbyt_readout.sink_zero_init=true",
        ],
        "B3-additive-zero-temp01": [
            "+stage2.qbyt_readout.sink_readout=additive",
            "+stage2.qbyt_readout.sink_zero_init=true",
            "stage2.qbyt_readout.temperature=0.1",
        ],
    }[winner[0]]
    reason = "%s: dAUC %+.5f dTPR@1%% %+.5f" % winner
    return flags, reason


def training_command(run_name: str, extra: list[str]) -> list[str]:
    run_dir = RUN_ROOT + "/" + run_name
    return [
        PY,
        "scripts/train_stage2_qbyt.py",
        "+experiment=" + EXP,
        "stage2.learning_rate=5e-4",
        "stage2.warmup_steps=500",
        "stage2.max_steps=50000",
        "stage2.total_scheduler_steps=50000",
        "stage2.validation.val_check_interval=2500",
        "stage2.val_check_interval=2500",
        "stage2.num_workers=8",
        "stage2.run_name=" + run_name,
        "stage2.checkpoint_dir=" + run_dir + "/checkpoints",
        "stage2.log_dir=" + run_dir + "/logs",
        *extra,
    ]


def latest_metrics(run_name: str):
    paths = sorted((ROOT / RUN_ROOT / run_name / "logs").glob("*/version_*/metrics.csv"))
    if not paths:
        return None
    try:
        return pd.read_csv(paths[-1])
    except Exception:
        return None


def reference_auc(step: int, ref) -> float:
    lo = ref[0]
    for hi in ref:
        if step <= hi[0]:
            span = hi[0] - lo[0]
            if span <= 0:
                return hi[1]
            frac = (step - lo[0]) / span
            return lo[1] + frac * (hi[1] - lo[1])
        lo = hi
    return ref[-1][1]


def monitor(proc: subprocess.Popen, run_name: str, ref, ref_name: str) -> str:
    """Watch a training run; terminate it on hard failures. Returns a verdict."""

    warned = False
    while proc.poll() is None:
        time.sleep(180)
        frame = latest_metrics(run_name)
        if frame is None:
            continue
        if "val/auc" not in frame.columns:
            continue
        rows = frame[frame["val/auc"].notna()]
        if rows.empty:
            continue
        last = rows.iloc[-1]
        step = int(last["step"])
        auc = float(last["val/auc"])
        reason = None
        for floor_step, floor in HARD:
            if step >= floor_step and auc < floor:
                reason = "AUC %.4f below hard floor %.4f at step %d" % (auc, floor, step)
                break
        if reason is None:
            ref_auc = reference_auc(step, ref)
            drop = ref_auc - auc
            if drop > ABORT_DROP and step >= 10000:
                reason = "AUC %.4f is %.4f below %s reference at step %d" % (auc, drop, ref_name, step)
            elif drop > WARN_DROP:
                log("WARN %s step %d AUC %.4f (%.4f below %s)" % (run_name, step, auc, drop, ref_name))
                warned = True
        write_status(**{
            run_name: {
                "step": step,
                "auc": auc,
                "eer": float(last.get("val/eer", float("nan"))),
                "tpr_at_fpr_1e_2": float(last.get("val/tpr_at_fpr_1e_2", float("nan"))),
                "tpr_at_fpr_1e_3": float(last.get("val/tpr_at_fpr_1e_3", float("nan"))),
                "pauc": float(last.get("val/pauc_fpr_1e_2", float("nan"))),
                "warned": warned,
            }
        })
        if reason:
            log("ABORT %s: %s" % (run_name, reason))
            proc.send_signal(signal.SIGTERM)
            try:
                proc.wait(timeout=120)
            except subprocess.TimeoutExpired:
                proc.kill()
            write_status(**{run_name + "_abort": reason})
            return "aborted: " + reason
    return "ok" if proc.returncode == 0 else "exit=%s" % proc.returncode


def exported_pt(run_name: str) -> Path | None:
    # Lightning writes checkpoints/<run_name>/version_*/; the exported .pt can
    # sit one or two levels below the run's checkpoint directory.
    found = sorted((ROOT / RUN_ROOT / run_name / "checkpoints").rglob("stage2_step*.pt"))
    return found[-1] if found else None


def probe(run_name: str, extra: list[str]) -> None:
    ckpt = exported_pt(run_name)
    if ckpt is None:
        log("probe skipped: no exported .pt for " + run_name)
        return
    out_dir = FINAL / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    npz = run_name + "_probe.npz"
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "0"
    command = [
        PY, "scripts/probe_qbyt_v41_offline.py",
        "+experiment=" + EXP,
        "prep.checkpoint=" + str(ckpt),
        "prep.output_dir=" + str(out_dir),
        "+prep.output_name=" + npz,
        "prep.batch_size=128",
        "prep.num_workers=8",
        *extra,
    ]
    log("probing " + run_name)
    subprocess.run(command, cwd=str(ROOT), env=env, check=False)
    analysis = out_dir / (run_name + "_analysis.json")
    subprocess.run(
        [PY, "scripts/analyze_v41_offline.py", str(out_dir / npz), str(analysis)],
        cwd=str(ROOT), env=env, check=False,
    )
    log("probe done " + run_name)


def run_training(run_name: str, extra: list[str], ref, ref_name: str) -> str:
    log("starting " + run_name)
    write_status(**{run_name + "_started": time.strftime("%Y-%m-%d %H:%M:%S")})
    log_path = FINAL / (run_name + ".log")
    with log_path.open("w") as handle:
        proc = subprocess.Popen(
            training_command(run_name, extra),
            cwd=str(ROOT),
            stdout=handle,
            stderr=subprocess.STDOUT,
            env=dict(os.environ, CUDA_VISIBLE_DEVICES="0"),
        )
        verdict = monitor(proc, run_name, ref, ref_name)
    log(run_name + " -> " + verdict)
    write_status(**{run_name + "_verdict": verdict})
    return verdict


def main() -> None:
    FINAL.mkdir(parents=True, exist_ok=True)
    wait_pid = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    if wait_pid:
        log("waiting for round-B launcher pid %d" % wait_pid)
        wait_for_pid(wait_pid)

    rows = collect_round_b()
    write_status(round_b=rows)
    flags, reason = pick_c1_flags(rows)
    log("C1 config: " + reason)
    write_status(c1_decision=reason, c1_flags=flags)

    c1_verdict = run_training("C1-sink-50k", flags, REF_V41, "v4.1")
    if not c1_verdict.startswith("ok"):
        write_status(pipeline="stopped after C1 failure; C2 not launched")
        log("C1 failed; C2 held back")
        return
    probe("C1-sink-50k", flags)

    c1_curve = []
    frame = latest_metrics("C1-sink-50k")
    if frame is not None and "val/auc" in frame.columns:
        rows_ok = frame[frame["val/auc"].notna()]
        c1_curve = [(int(r["step"]), float(r["val/auc"])) for _, r in rows_ok.iterrows()]
    if not c1_curve:
        c1_curve = REF_V41

    c2_flags = flags + ["stage2.qbyt_readout.text_position=sinusoidal"]
    c2_verdict = run_training("C2-sinusoidal-50k", c2_flags, c1_curve, "C1")
    if c2_verdict.startswith("ok"):
        probe("C2-sinusoidal-50k", c2_flags)
    write_status(pipeline="done", c1_verdict=c1_verdict, c2_verdict=c2_verdict)
    log("pipeline done")


if __name__ == "__main__":
    main()
