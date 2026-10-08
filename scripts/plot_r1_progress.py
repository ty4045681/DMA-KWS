#!/usr/bin/env python3
"""Visualise R1 (paper-style continued fine-tune) against the base models."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path("/home/ubuntu/dma-kws")
FIG = ROOT / "outputs/v41_final/figures"
R1 = "data/dma-kws/exp/stage2_qbyt/final/R1-unfreeze-hardneg100/logs/R1-unfreeze-hardneg100/version_0/metrics.csv"
BASE_C1 = 0.93785      # C1 deployed (T=1)
BASE_SS = 0.94106      # SS-zh-en (v4.2 with 3k sink head), R1's starting point
TOTAL = 13300

frame = pd.read_csv(ROOT / R1)
train = frame[frame["step"].notna()]
val = frame[frame["val/auc"].notna()].reset_index(drop=True)
last_step = int(train.iloc[-1]["step"])
steps = val["step"].astype(int)
rate = last_step / max(
    1.0,
    (
        pd.Timestamp.now()
        - pd.Timestamp("2026-10-05 20:15:00")
    ).total_seconds(),
)

fig, axes = plt.subplots(2, 3, figsize=(17, 8.5))
ax = axes[0, 0]
ax.plot(steps, val["val/auc"], "o-", color="tab:red", label="R1 (unfreeze + 100:1)")
ax.axhline(BASE_SS, color="tab:blue", linestyle="--", label="SS-zh-en v4.2 base 0.94106")
ax.axhline(BASE_C1, color="tab:gray", linestyle=":", label="C1 0.93785")
ax.axhline(0.93765, color="tab:orange", linestyle=":", label="v2 0.93765")
ax.set_xlabel("optimizer step")
ax.set_ylabel("AUC")
ax.set_title("R1 AUC vs baselines")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

ax = axes[0, 1]
delta = val["val/auc"] - BASE_SS
ax.plot(steps, delta, "o-", color="black", label="R1 AUC - base 0.94106")
ax.axhline(0.0, color="tab:blue", linestyle="--", label="parity with base")
ax.fill_between(steps, delta, 0, where=(delta < 0), color="tab:red", alpha=0.15)
ax.set_xlabel("optimizer step")
ax.set_ylabel("delta AUC vs base")
ax.set_title("Regression vs the model R1 started from")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

ax = axes[0, 2]
ax.plot(steps, val["val/eer"], "o-", color="tab:purple")
ax.axhline(0.12571, color="tab:blue", linestyle="--", label="SS-zh-en EER 0.12571")
ax.axhline(0.12730, color="tab:gray", linestyle=":", label="C1 EER 0.12730")
ax.set_xlabel("optimizer step")
ax.set_ylabel("EER")
ax.set_title("R1 EER (lower better)")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

ax = axes[1, 0]
ax.plot(steps, val["val/tpr_at_fpr_1e_2"], "o-", color="tab:green", label="TPR@1%FPR")
ax.plot(steps, val["val/tpr_at_fpr_1e_3"], "s--", color="tab:brown", label="TPR@0.1%FPR")
ax.axhline(0.23556, color="tab:blue", linestyle="--", label="SS TPR@1% 0.23556")
ax.set_xlabel("optimizer step")
ax.set_ylabel("TPR")
ax.set_title("R1 low-FPR tail")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

ax = axes[1, 1]
grid = np.arange(0, TOTAL, 50)
lr = []
for step in grid:
    if step < 2500:
        lr.append(5e-4 * (step + 1) / 2500)
    else:
        progress = (step - 2500) / (100000 - 2500)
        lr.append(5e-4 * 0.5 * (1 + np.cos(np.pi * progress)))
ax.plot(grid, lr, color="tab:orange")
ax.axvline(last_step, color="k", linestyle=":", label=f"current step {last_step}")
ax.set_xlabel("optimizer step")
ax.set_ylabel("learning rate")
ax.set_title("R1 LR schedule (author's warmup 2500 + cosine to 100k, first 13.3k)")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

ax = axes[1, 2]
labels = ["v4.1", "v2", "C1", "SS base", "R1 latest"]
values = [0.93501, 0.93765, BASE_C1, BASE_SS, float(val["val/auc"].iloc[-1])]
colors = ["tab:gray", "tab:orange", "tab:blue", "tab:cyan", "tab:red"]
bars = ax.bar(labels, values, color=colors)
for bar, value in zip(bars, values):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        value + 0.0004,
        "%.4f" % value,
        ha="center",
        fontsize=8,
    )
ax.set_ylim(0.93, 0.945)
ax.set_ylabel("AUC")
ax.set_title("R1 latest val vs reference models")
ax.grid(alpha=0.3)

fig.suptitle(
    "R1 paper-style continued fine-tune (encoder unfrozen + hard_negative_ratio 100, batch 384x2, lr 5e-4)",
    fontsize=12,
)
fig.tight_layout()
FIG.mkdir(parents=True, exist_ok=True)
fig.savefig(FIG / "r1_progress.png", dpi=130)
print("wrote", FIG / "r1_progress.png")
print("last step", last_step, "of", TOTAL, "| val points", len(val))
print("val AUC:", [round(float(v), 5) for v in val["val/auc"]])
print("rate %.2f steps/s" % rate)
print("remaining %.2f h" % ((TOTAL - last_step) / rate / 3600))
