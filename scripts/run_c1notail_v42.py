#!/usr/bin/env python3
"""Wait for the user's C1-notail-50k to finish, then run the full v4.2 chain:
probe -> frozen-trunk sink-head second stage -> fitted-head write-back ->
verification probe. The probe flags adapt to the export's qbyt_readout spec.
"""

from __future__ import annotations

import glob
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch

ROOT = Path("/home/ubuntu/dma-kws")
EXP = "icefall_zipformer_stage2_eps_softmin_v41_musan_cached_zhen3m_50k"
RUN_DIR = ROOT / "data/dma-kws/exp/stage2_qbyt/final/C1-notail-50k"
OUT = ROOT / "outputs/v41_final/C1-notail-50k"
PY = sys.executable


def sh(args, log):
    with open(str(log), "w") as handle:
        return subprocess.run(
            args,
            cwd=str(ROOT),
            env={**os.environ, "CUDA_VISIBLE_DEVICES": "0"},
            stdout=handle,
            stderr=subprocess.STDOUT,
        ).returncode


def wait_training_exit():
    while True:
        check = subprocess.run(
            ["pgrep", "-f", "run_name=C1-notail-50k"],
            capture_output=True,
            text=True,
        )
        if not check.stdout.strip():
            return
        time.sleep(300)


def main() -> None:
    print("waiting for C1-notail-50k training to exit", flush=True)
    wait_training_exit()
    export = None
    for _ in range(40):
        found = sorted(
            glob.glob(str(RUN_DIR / "checkpoints/**/stage2_step*.pt"), recursive=True)
        )
        if found:
            export = found[-1]
            break
        time.sleep(30)
    if not export:
        print("NOEXPORT", flush=True)
        return
    print("export:", export, flush=True)
    checkpoint = torch.load(export, map_location="cpu", weights_only=False)
    spec = (checkpoint.get("config") or {}).get("stage2", {}).get("qbyt_readout") or {}
    print("spec:", json.dumps(spec, default=str)[:300], flush=True)
    flags = []
    if spec.get("sink_readout") == "additive":
        flags.append("+stage2.qbyt_readout.sink_readout=additive")
    if spec.get("sink_zero_init"):
        flags.append("+stage2.qbyt_readout.sink_zero_init=true")

    OUT.mkdir(parents=True, exist_ok=True)
    rc = sh(
        [
            PY, "scripts/probe_qbyt_v41_offline.py",
            f"+experiment={EXP}", *flags,
            f"prep.checkpoint={export}",
            f"prep.output_dir={OUT}",
            "+prep.output_name=probe.npz",
            "prep.batch_size=128", "prep.num_workers=4",
        ],
        OUT / "probe.log",
    )
    print("probe rc:", rc, flush=True)
    sh(
        [PY, "scripts/analyze_v41_offline.py", str(OUT / "probe.npz"), str(OUT / "analysis.json")],
        OUT / "analysis.log",
    )
    with open(OUT / "analysis.json") as handle:
        base = json.load(handle)["baseline"]
    print(
        "C1-notail deployed: AUC %.5f EER %.5f TPR@1%% %.5f TPR@0.1%% %.6f pAUC %.5f"
        % (base["auc"], base["eer"], base["tpr_at_fpr_1e_2"], base["tpr_at_fpr_1e_3"], base["pauc_fpr_1e_2"]),
        flush=True,
    )

    ss_dir = "data/dma-kws/exp/stage2_qbyt/sinkhead_stage2/SS-c1notail"
    rc = sh(
        [
            PY, "scripts/train_stage2_qbyt.py",
            f"+experiment={EXP}",
            "+stage2.qbyt_readout.sink_readout=additive",
            "+stage2.qbyt_readout.sink_zero_init=true",
            "+stage2.freeze_all_but_sink=true",
            "stage2.sink_loss.enabled=true",
            "stage2.sink_loss.weight=0.25",
            f"stage2.init_checkpoint={export}",
            "+stage2.init_allow_readout_mismatch=true",
            "stage2.learning_rate=1e-3",
            "stage2.warmup_steps=200",
            "stage2.max_steps=3000",
            "stage2.total_scheduler_steps=3000",
            "stage2.validation.val_check_interval=1500",
            "stage2.val_check_interval=1500",
            "stage2.num_workers=2",
            "stage2.run_name=SS-c1notail",
            f"stage2.checkpoint_dir={ss_dir}/checkpoints",
            f"stage2.log_dir={ss_dir}/logs",
        ],
        OUT / "ss.log",
    )
    print("second stage rc:", rc, flush=True)
    ss_exports = sorted(
        glob.glob(f"{ss_dir}/checkpoints/**/stage2_step*.pt", recursive=True)
    )
    if not ss_exports:
        print("NOSSEXPORT", flush=True)
        return
    ss_export = ss_exports[-1]
    print("ss export:", ss_export, flush=True)

    probe_dir = ROOT / "outputs/v41_sinkhead_stage2/probe_SS-c1notail"
    probe_dir.mkdir(parents=True, exist_ok=True)
    sh(
        [
            PY, "scripts/probe_qbyt_v41_offline.py",
            f"+experiment={EXP}",
            "+stage2.qbyt_readout.sink_readout=additive",
            "+stage2.qbyt_readout.sink_zero_init=true",
            f"prep.checkpoint={ss_export}",
            f"prep.output_dir={probe_dir}",
            "+prep.output_name=probe.npz",
            "prep.batch_size=128", "prep.num_workers=4",
        ],
        OUT / "ss_probe.log",
    )
    sh(
        [PY, "scripts/analyze_v41_offline.py", str(probe_dir / "probe.npz"), str(probe_dir / "analysis.json")],
        OUT / "ss_analysis.log",
    )
    sh(
        [PY, "scripts/write_sink_head.py", ss_export, str(probe_dir / "analysis.json"), ss_export + ".sinkfit.pt"],
        OUT / "writeback.log",
    )
    verify_dir = ROOT / "outputs/v41_sinkhead_stage2/probe_sinkfit_SS-c1notail"
    verify_dir.mkdir(parents=True, exist_ok=True)
    sh(
        [
            PY, "scripts/probe_qbyt_v41_offline.py",
            f"+experiment={EXP}",
            "+stage2.qbyt_readout.sink_readout=additive",
            "+stage2.qbyt_readout.sink_zero_init=true",
            f"prep.checkpoint={ss_export}.sinkfit.pt",
            f"prep.output_dir={verify_dir}",
            "+prep.output_name=probe.npz",
            "prep.batch_size=128", "prep.num_workers=4",
        ],
        OUT / "verify.log",
    )
    sh(
        [PY, "scripts/analyze_v41_offline.py", str(verify_dir / "probe.npz"), str(verify_dir / "analysis.json")],
        OUT / "verify_analysis.log",
    )
    with open(verify_dir / "analysis.json") as handle:
        base = json.load(handle)["baseline"]
    print(
        "C1-notail v4.2 (sinkfit): AUC %.5f EER %.5f TPR@1%% %.5f TPR@0.1%% %.6f pAUC %.5f"
        % (base["auc"], base["eer"], base["tpr_at_fpr_1e_2"], base["tpr_at_fpr_1e_3"], base["pauc_fpr_1e_2"]),
        flush=True,
    )
    print("C1NOTAILDONE", flush=True)


if __name__ == "__main__":
    main()
