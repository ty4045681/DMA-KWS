#!/usr/bin/env python3
"""Visualize the two training tasks currently running in parallel."""
from __future__ import annotations

import datetime as dt
import glob

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

V42 = "data/dma-kws/exp/stage2_qbyt/v42_encoders/V42-paperstage1/logs/*/version_1"
NT = "data/dma-kws/exp/stage2_qbyt/final/C1-notail-50k/logs/*/version_0"
C1 = "data/dma-kws/exp/stage2_qbyt/final/C1-sink-50k/logs/*/version_0"
REF_V41 = [
    (2500, 0.80), (5000, 0.85), (7500, 0.8740), (10000, 0.8854),
    (15000, 0.9010), (20000, 0.9092), (25000, 0.9203), (30000, 0.9222),
    (35000, 0.9275), (40000, 0.9330), (45000, 0.9342), (50000, 0.9348),
]

vm = pd.read_csv(sorted(glob.glob(V42 + "/metrics.csv"))[-1])
ve = pd.read_csv(sorted(glob.glob(V42 + "/eval_history.csv"))[-1])
nm = pd.read_csv(sorted(glob.glob(NT + "/metrics.csv"))[-1])
ne = pd.read_csv(sorted(glob.glob(NT + "/eval_history.csv"))[-1])
c1 = pd.read_csv(sorted(glob.glob(C1 + "/eval_history.csv"))[-1])

v_step = int(vm["step"].iloc[-1])
n_step = int(nm["step"].iloc[-1])
NOW = dt.datetime(2026, 10, 5, 11, 20)

v_rate = v_step / ((NOW - dt.datetime(2026, 10, 5, 9, 25)).total_seconds() / 3600)
n_rate = n_step / ((NOW - dt.datetime(2026, 10, 5, 10, 21)).total_seconds() / 3600)
v_eta = NOW + dt.timedelta(hours=(50000 - v_step) / v_rate)
n_eta = NOW + dt.timedelta(hours=(50000 - n_step) / n_rate)

fig = plt.figure(figsize=(14, 9.6))
gs = fig.add_gridspec(2, 3, height_ratios=[0.62, 1.6], hspace=0.5, wspace=0.28)
ax_bar = fig.add_subplot(gs[0, :])

items = [
    ("V42-paperstage1", v_step, v_rate, v_eta, "#9467bd",
     f"AUC@37.5k {float(ve['val/auc'].iloc[-1]):.4f}  EER {float(ve['val/eer'].iloc[-1]):.4f}  (v4.2: paper Stage-I + C1 recipe, 09:25)",
     "launched by run_v42_encoders.py (monitored)"),
    ("C1-notail-50k", n_step, n_rate, n_eta, "#ff7f0e",
     f"AUC@17.5k {float(ne['val/auc'].iloc[-1]):.4f}  EER {float(ne['val/eer'].iloc[-1]):.4f}  (C1 recipe - no neg-tail loss, ls-gs-1460 data, 10:21)",
     "launched from interactive shell (unmonitored)"),
]
for i, (name, step, rate, eta, color, note, src) in enumerate(items):
    frac = step / 50000.0
    ax_bar.barh(i, 1.0, color="0.9", height=0.6, zorder=0)
    ax_bar.barh(i, frac, color=color, alpha=0.85, height=0.6)
    ax_bar.text(-0.008, i, name, va="center", ha="right", fontsize=10.5, color="0.2")
    ax_bar.text(frac + 0.008, i, f"{100 * frac:.1f}%  ({step:,}/50,000)", va="center", fontsize=10.5,
                fontweight="bold", color=color)
    ax_bar.text(1.008, i, note + "   •   " + src, va="center", fontsize=9.5, color="0.3")
ax_bar.set_yticks([])
ax_bar.set_xlim(-0.24, 1.62)
ax_bar.invert_yaxis()
ax_bar.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
ax_bar.set_xticklabels(["0", "12.5k", "25k", "37.5k", "50k"], fontsize=9)
ax_bar.spines[["top", "right", "left"]].set_visible(False)
ax_bar.set_title("Two trainings running in parallel on GPU 0 — 2026-10-05 11:20", fontsize=13, fontweight="bold", loc="left")

def ema(x, a=0.985):
    out = np.empty_like(x, dtype=float)
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = a * out[i - 1] + (1 - a) * x[i]
    return out

# ---- panel 1: V42 AUC/EER ----
ax1 = fig.add_subplot(gs[1, 0])
ax1.plot(c1["step"], c1["val/auc"], "-", lw=2, color="#1f77b4", label="C1-sink-50k (v4.1 best)")
ax1.plot([r[0] for r in REF_V41], [r[1] for r in REF_V41], "--", lw=1.2, color="0.5", label="v4.1 reference")
ax1.plot(ve["step"], ve["val/auc"], "-o", ms=4, lw=2, color="#9467bd", label="V42-paperstage1")
ax1.set_xlabel("step"); ax1.set_ylabel("val AUC")
ax1.grid(alpha=0.3); ax1.legend(fontsize=8, loc="lower right")
ax1.set_title("V42-paperstage1", fontsize=11, fontweight="bold")

# ---- panel 2: C1-notail AUC/EER ----
ax2 = fig.add_subplot(gs[1, 1])
ax2.plot(c1["step"], c1["val/auc"], "-", lw=2, color="#1f77b4", label="C1-sink-50k (with tail loss, zhen3m data)")
ax2.plot([r[0] for r in REF_V41], [r[1] for r in REF_V41], "--", lw=1.2, color="0.5", label="v4.1 reference")
ax2.plot(ne["step"], ne["val/auc"], "-o", ms=4, lw=2, color="#ff7f0e", label="C1-notail-50k")
ax2.set_xlabel("step"); ax2.set_ylabel("val AUC")
ax2.grid(alpha=0.3); ax2.legend(fontsize=8, loc="lower right")
ax2.set_title("C1-notail-50k (ablation: negative-tail loss off)", fontsize=11, fontweight="bold")

# ---- panel 3: loss overlay ----
ax3 = fig.add_subplot(gs[1, 2])
for name, m, color in [("V42-paperstage1", vm, "#9467bd"), ("C1-notail-50k", nm, "#ff7f0e")]:
    sub = m[["step", "train/microbatch/loss_total"]].dropna()
    ax3.plot(sub["step"], ema(sub["train/microbatch/loss_total"].values), lw=1.2, color=color, label=name)
ax3.set_yscale("log")
ax3.set_xlabel("step"); ax3.set_ylabel("train loss (EMA, log)")
ax3.grid(alpha=0.3, which="both"); ax3.legend(fontsize=8)
ax3.set_title("Training loss (microbatch, EMA-smoothed)", fontsize=11, fontweight="bold")

fig.text(0.01, 0.012,
         f"V42-paperstage1: ~{v_rate:,.0f} steps/h → ETA {v_eta.strftime('%H:%M')}    •    "
         f"C1-notail-50k: ~{n_rate:,.0f} steps/h → ETA {n_eta.strftime('%H:%M')}    •    "
         f"GPU 0 shared: ~4.6 GB / 32 GB, util ~54%",
         fontsize=9.5, color="0.3")
fig.savefig("outputs/v41_final/figures/training_progress_2026-10-04.png", dpi=110, bbox_inches="tight", facecolor="white")
print("saved;", v_step, n_step, f"{v_rate:.0f}/h ETA {v_eta.strftime('%H:%M')}", f"{n_rate:.0f}/h ETA {n_eta.strftime('%H:%M')}")
