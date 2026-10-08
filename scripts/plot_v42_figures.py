#!/usr/bin/env python3
"""Generate the remaining figures for the QbyT v4.2 design document.

Outputs (outputs/v41_final/figures/):
  temperature_sweep_multi.png - scoring-temperature sweep across 4 models
  sink_head_trajectory.png    - sink head learning trajectory (alpha / norm)
  v42_scoreboard.png          - final scoreboard: v4.1 / v2 / C1 / v4.2 paths
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path("/home/ubuntu/dma-kws")
FIG = ROOT / "outputs/v41_final/figures"
FIG.mkdir(parents=True, exist_ok=True)

V2 = {"auc": 0.937647, "eer": 0.128644, "tpr1": 0.202073, "tpr01": 0.023467, "pauc": 0.552360}

ANALYSES = {
    "C1 (sink readout)": ROOT / "outputs/v41_final/C1-sink-50k/C1-sink-50k_analysis.json",
    "S1 (warm 2k, bce .25)": ROOT / "outputs/v41_sinkloss_ab/probe_S1/analysis.json",
    "S2 (warm 2k, bce .5)": ROOT / "outputs/v41_sinkloss_ab/probe_S2/analysis.json",
    "C3 (50k, bce .25)": ROOT / "outputs/v41_final/C3-sinkloss-50k/C3-sinkloss-50k_analysis.json",
}
COLORS = ["#1f77b4", "#2ca02c", "#9467bd", "#d62728"]

fig, ax = plt.subplots(figsize=(9.5, 5.2))
for (name, path), color in zip(ANALYSES.items(), COLORS):
    if not path.exists():
        continue
    data = json.loads(path.read_text())
    sweep = data["temperature_sweep"]
    temps = [float(t) for t in sweep]
    ax.plot(temps, [sweep[str(t)]["auc"] for t in temps], "o-", color=color, label=name, markersize=4)
ax.axhline(V2["auc"], color="tab:red", linestyle=":", label="v2 final")
ax.set_xscale("log")
ax.set_xlabel("scoring temperature T (inference only)")
ax.set_ylabel("AUC")
ax.set_title("Scoring-temperature sweep across models (S8b generalises)")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(FIG / "temperature_sweep_multi.png", dpi=130)

CHECKPOINTS = {
    "v4.1": sorted((ROOT / "data/dma-kws/exp/stage2_qbyt/checkpoints/v41-musan-zhen3m-50k").rglob("stage2_step047500.pt")),
    "C1": sorted((ROOT / "data/dma-kws/exp/stage2_qbyt/final/C1-sink-50k/checkpoints").rglob("stage2_step*.pt")),
    "S1": sorted((ROOT / "data/dma-kws/exp/stage2_qbyt/sinkloss_ab/S1-bce-0p25/checkpoints").rglob("stage2_step*.pt")),
    "S2": sorted((ROOT / "data/dma-kws/exp/stage2_qbyt/sinkloss_ab/S2-bce-0p5/checkpoints").rglob("stage2_step*.pt")),
    "C3": sorted((ROOT / "data/dma-kws/exp/stage2_qbyt/final/C3-sinkloss-50k/checkpoints").rglob("stage2_step*.pt")),
}
names, norms, alphas = [], [], []
for name, paths in CHECKPOINTS.items():
    if not paths:
        continue
    state = torch.load(paths[-1], map_location="cpu", weights_only=False)
    sd = state["model_state_dict"] if "model_state_dict" in state else state["state_dict"]
    names.append(name)
    norms.append(float(sd["qbyt.sink_fc.weight"].norm()) if "qbyt.sink_fc.weight" in sd else np.nan)
    alphas.append(float(sd["qbyt.sink_alpha"]) if "qbyt.sink_alpha" in sd else np.nan)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
x = np.arange(len(names))
ax = axes[0]
bars = ax.bar(x, norms, color=["tab:gray", "tab:blue", "tab:green", "tab:purple", "tab:red"])
ax.set_xticks(x)
ax.set_xticklabels(names)
ax.set_ylabel("sink_fc weight norm")
ax.set_title("Sink head capacity learned (0 = zero init)")
ax.grid(alpha=0.3)
for bar, value in zip(bars, norms):
    if np.isfinite(value):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.01, f"{value:.2f}", ha="center", fontsize=8)
ax = axes[1]
bars = ax.bar(x, alphas, color=["tab:gray", "tab:blue", "tab:green", "tab:purple", "tab:red"])
ax.set_xticks(x)
ax.set_xticklabels(names)
ax.set_ylabel("alpha (sink branch weight)")
ax.set_title("Alpha gate trajectory (1.0 = full sink contribution)")
ax.grid(alpha=0.3)
for bar, value in zip(bars, alphas):
    if np.isfinite(value):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f"{value:.2f}", ha="center", fontsize=8)
fig.suptitle("Sink head learning across checkpoints (sink_loss drives the head, see text for the trade-off)")
fig.tight_layout()
fig.savefig(FIG / "sink_head_trajectory.png", dpi=130)

SCOREBOARD = [
    ("v4.1", 0.93501, 0.13106, 0.18323, 0.01993),
    ("v2", 0.937647, 0.128644, 0.202073, 0.023467),
    ("C1 (T=1)", 0.93785, 0.12730, 0.20385, 0.028919),
    ("C1 + T=0.25", 0.93985, 0.12610, 0.22468, 0.019454),
    ("v4.2 achieved = SS + sink fit", 0.94258, 0.12360, 0.24595, 0.03992),
]
labels = [row[0] for row in SCOREBOARD]
aucs = [row[1] for row in SCOREBOARD]
eers = [row[2] for row in SCOREBOARD]
tpr1 = [row[3] for row in SCOREBOARD]
tpr01 = [row[4] for row in SCOREBOARD]
x = np.arange(len(SCOREBOARD))
fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
ax = axes[0]
bars = ax.bar(x, aucs, color=["tab:gray", "tab:red", "tab:blue", "tab:blue", "tab:purple"])
ax.axhline(V2["auc"], color="tab:red", linestyle=":", linewidth=1)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=7)
ax.set_ylim(0.93, 0.945)
ax.set_ylabel("AUC")
ax.set_title("AUC")
ax.grid(alpha=0.3)
for bar, value in zip(bars, aucs):
    ax.text(bar.get_x() + bar.get_width() / 2, value + 0.0004, f"{value:.4f}", ha="center", fontsize=7)
ax = axes[1]
bars = ax.bar(x, eers, color=["tab:gray", "tab:red", "tab:blue", "tab:blue", "tab:purple"])
ax.axhline(V2["eer"], color="tab:red", linestyle=":", linewidth=1)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=7)
ax.set_ylim(0.122, 0.134)
ax.set_ylabel("EER")
ax.set_title("EER (lower is better)")
ax.grid(alpha=0.3)
for bar, value in zip(bars, eers):
    ax.text(bar.get_x() + bar.get_width() / 2, value - 0.0006, f"{value:.4f}", ha="center", fontsize=7)
ax = axes[2]
width = 0.38
bars1 = ax.bar(x - width / 2, tpr1, width, color=["tab:gray", "tab:red", "tab:blue", "tab:blue", "tab:purple"], label="TPR@1%FPR")
bars2 = ax.bar(x + width / 2, [v * 10 for v in tpr01], width, color=["0.75", "0.85", "0.55", "0.55", "0.8"], label="TPR@0.1%FPR x10")
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=7)
ax.set_ylabel("TPR")
ax.set_title("Low-FPR tail (TPR@0.1%FPR shown x10)")
ax.legend(fontsize=7)
ax.grid(alpha=0.3)
for bar, value in zip(bars1, tpr1):
    ax.text(bar.get_x() + bar.get_width() / 2, value + 0.004, f"{value:.3f}", ha="center", fontsize=6.5)
for bar, value in zip(bars2, [v * 10 for v in tpr01]):
    ax.text(bar.get_x() + bar.get_width() / 2, value + 0.004, f"{value:.2f}", ha="center", fontsize=6.5)
fig.suptitle("QbyT v4.2 scoreboard: frozen-trunk sink head takes v4.1 past v2", fontsize=12)
fig.tight_layout()
fig.savefig(FIG / "v42_scoreboard.png", dpi=130)

print("wrote", sorted(p.name for p in FIG.glob("*.png")))
