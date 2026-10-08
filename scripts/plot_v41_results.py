#!/usr/bin/env python3
"""Visualize the C1/C2 training runs and the A/B ablations."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path("/home/ubuntu/dma-kws")
FIG = ROOT / "outputs/v41_final/figures"
FIG.mkdir(parents=True, exist_ok=True)
C1 = ROOT / "data/dma-kws/exp/stage2_qbyt/final/C1-sink-50k/logs/C1-sink-50k/version_0/metrics.csv"
C2 = ROOT / "data/dma-kws/exp/stage2_qbyt/final/C2-sinusoidal-50k/logs/C2-sinusoidal-50k/version_0/metrics.csv"

V41 = [
    (7500, 0.8740, 0.2038, 0.0072, 0.1430),
    (10000, 0.8854, 0.1858, 0.0059, 0.1345),
    (15000, 0.9010, 0.1728, 0.0099, 0.1289),
    (20000, 0.9092, 0.1627, 0.0126, 0.1196),
    (25000, 0.9203, 0.1505, 0.0135, 0.1154),
    (30000, 0.9222, 0.1440, 0.0136, 0.1186),
    (35000, 0.9275, 0.1394, 0.0177, 0.1168),
    (40000, 0.9330, 0.1341, 0.0180, 0.1119),
    (45000, 0.9342, 0.1318, 0.0191, 0.1117),
    (50000, 0.9348, 0.1313, 0.0205, 0.1098),
]
V2 = {"auc": 0.937647, "eer": 0.128644, "tpr1": 0.202073, "pauc": 0.552360}


def val_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    return frame[frame["val/auc"].notna()].reset_index(drop=True)


def panel(ax, x, y, color, label, ylabel, marker="o"):
    ax.plot(x, y, color=color, marker=marker, markersize=4, linewidth=2, label=label)
    ax.set_xlabel("optimizer step")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)


c1 = val_frame(C1)
c2 = val_frame(C2)

fig, axes = plt.subplots(2, 2, figsize=(13, 9))
ax = axes[0, 0]
panel(ax, c1["step"], c1["val/auc"], "tab:blue", "C1 (sink additive)", "AUC")
ax.plot([r[0] for r in V41], [r[1] for r in V41], "k--", marker="s", markersize=4, label="v4.1 (no sink)")
ax.axhline(V2["auc"], color="tab:red", linestyle=":", label="v2 final")
ax.set_title("LibriPhrase hard split - AUC")
ax.legend(fontsize=8)

ax = axes[0, 1]
panel(ax, c1["step"], c1["val/eer"], "tab:blue", "C1", "EER")
ax.plot([r[0] for r in V41], [r[2] for r in V41], "k--", marker="s", markersize=4, label="v4.1")
ax.axhline(V2["eer"], color="tab:red", linestyle=":", label="v2 final")
ax.set_title("EER (lower is better)")
ax.legend(fontsize=8)

ax = axes[1, 0]
panel(ax, c1["step"], c1["val/tpr_at_fpr_1e_2"], "tab:blue", "C1", "TPR@1%FPR")
ax.plot([r[0] for r in V41], [r[3] for r in V41], "k--", marker="s", markersize=4, label="v4.1")
ax.axhline(V2["tpr1"], color="tab:red", linestyle=":", label="v2 final")
ax.set_title("TPR@1%FPR (higher is better)")
ax.legend(fontsize=8)

ax = axes[1, 1]
panel(ax, c1["step"], c1["val/pauc_fpr_1e_2"], "tab:blue", "C1", "pAUC(<=1%)")
ax.axhline(V2["pauc"], color="tab:red", linestyle=":", label="v2 final")
ax.set_title("pAUC in the low-FPR region")
ax.legend(fontsize=8)
fig.suptitle("C1: v4.1 recipe + additive sink readout, 50k steps from scratch", fontsize=13)
fig.tight_layout()
fig.savefig(FIG / "c1_curves.png", dpi=130)

fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
ax = axes[0]
ax.plot(c1["step"], c1["val/auc"], "o-", color="tab:blue", label="C1 (learned text PE)")
ax.plot(c2["step"], c2["val/auc"], "o-", color="tab:orange", label="C2 (sinusoidal text PE)")
ax.axhline(V2["auc"], color="tab:red", linestyle=":", label="v2 final")
ax.set_xlabel("optimizer step")
ax.set_ylabel("AUC")
ax.set_title("C1 vs C2 (same recipe, text positional encoding)")
ax.grid(alpha=0.3)
ax.legend(fontsize=8)
ax = axes[1]
ax.plot(c1["step"], c1["val/tpr_at_fpr_1e_2"], "o-", color="tab:blue", label="C1")
ax.plot(c2["step"], c2["val/tpr_at_fpr_1e_2"], "o-", color="tab:orange", label="C2")
ax.axhline(V2["tpr1"], color="tab:red", linestyle=":", label="v2 final")
ax.set_xlabel("optimizer step")
ax.set_ylabel("TPR@1%FPR")
ax.set_title("Low-FPR tail: C1 vs C2")
ax.grid(alpha=0.3)
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(FIG / "c1_vs_c2.png", dpi=130)

a_rows = json.loads((ROOT / "outputs/v41_ablations/collect_early.json").read_text())
b_rows = json.loads((ROOT / "outputs/v41_ablations_b/collect_final.json").read_text())

fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
ax = axes[0]
names = [r["group"].split("-", 1)[0] for r in a_rows]
dauc = [r.get("delta_val/auc", 0) for r in a_rows]
dtpr = [r.get("delta_val/tpr_at_fpr_1e_2", 0) for r in a_rows]
x = range(len(names))
ax.bar([i - 0.2 for i in x], dauc, width=0.4, color="tab:blue", label="dAUC")
ax.bar([i + 0.2 for i in x], dtpr, width=0.4, color="tab:green", label="dTPR@1%FPR")
ax.axhline(0, color="k", linewidth=0.8)
ax.set_xticks(list(x))
ax.set_xticklabels(names)
ax.set_title("Round A (LR 5e-5, 2k warm start) vs A0")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
ax = axes[1]
names = [r["group"].split("-", 1)[0] for r in b_rows]
dauc = [r.get("delta_val/auc", 0) for r in b_rows]
dtpr = [r.get("delta_val/tpr_at_fpr_1e_2", 0) for r in b_rows]
x = range(len(names))
ax.bar([i - 0.2 for i in x], dauc, width=0.4, color="tab:blue", label="dAUC")
ax.bar([i + 0.2 for i in x], dtpr, width=0.4, color="tab:green", label="dTPR@1%FPR")
ax.axhline(0, color="k", linewidth=0.8)
ax.set_xticks(list(x))
ax.set_xticklabels(names)
ax.set_title("Round B (LR 2e-4, zero-init) vs B0")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
fig.suptitle("Ablations: change relative to the matched control", fontsize=13)
fig.tight_layout()
fig.savefig(FIG / "ablations_ab.png", dpi=130)

analysis = json.loads((ROOT / "outputs/v41_final/C1-sink-50k/C1-sink-50k_analysis.json").read_text())
sweep = analysis["temperature_sweep"]
temps = [float(t) for t in sweep]
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
ax = axes[0, 0]
ax.plot(temps, [sweep[str(t)]["auc"] for t in temps], "o-", color="tab:blue")
ax.axhline(V2["auc"], color="tab:red", linestyle=":", label="v2 final")
ax.axhline(analysis["baseline"]["auc"], color="k", linestyle="--", label="C1 deployed (T=1, with sink)")
ax.set_xscale("log")
ax.set_xlabel("scoring temperature T")
ax.set_ylabel("AUC")
ax.set_title("AUC vs scoring temperature (text pooling, zero training)")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
ax = axes[0, 1]
ax.plot(temps, [sweep[str(t)]["eer"] for t in temps], "o-", color="tab:orange")
ax.axhline(V2["eer"], color="tab:red", linestyle=":", label="v2 final")
ax.set_xscale("log")
ax.set_xlabel("scoring temperature T")
ax.set_ylabel("EER")
ax.set_title("EER vs scoring temperature")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
ax = axes[1, 0]
ax.plot(temps, [sweep[str(t)]["tpr_at_fpr_1e_2"] for t in temps], "o-", color="tab:green")
ax.axhline(V2["tpr1"], color="tab:red", linestyle=":", label="v2 final")
ax.axhline(analysis["sink_probe"]["combo_metrics_eval_half"]["tpr_at_fpr_1e_2"], color="tab:purple", linestyle="-.", label="C1 + fitted sink head")
ax.set_xscale("log")
ax.set_xlabel("scoring temperature T")
ax.set_ylabel("TPR@1%FPR")
ax.set_title("Tail gain at 1% FPR")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
ax = axes[1, 1]
ax.plot(temps, [sweep[str(t)]["tpr_at_fpr_1e_3"] for t in temps], "o-", color="tab:purple")
ax.axhline(0.023467, color="tab:brown", linestyle=":", label="v2 TPR@0.1%")
ax.axhline(analysis["sink_probe"]["combo_metrics_eval_half"]["tpr_at_fpr_1e_3"], color="tab:purple", linestyle="-.", label="C1 + fitted sink head")
ax.set_xscale("log")
ax.set_xlabel("scoring temperature T")
ax.set_ylabel("TPR@0.1%FPR")
ax.set_title("Tail gain at 0.1% FPR")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
fig.suptitle("C1: zero-training scoring tweaks (temperature sweep, fitted sink readout)", fontsize=13)
fig.tight_layout()
fig.savefig(FIG / "c1_temperature.png", dpi=130)

S_PATHS = {
    name: ROOT / f"data/dma-kws/exp/stage2_qbyt/sinkloss_ab/{name}/logs/{name}/version_0/metrics.csv"
    for name in ("S0-control", "S1-bce-0p25", "S2-bce-0p5")
}
fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
ax = axes[0]
for name, path in S_PATHS.items():
    if not path.exists():
        continue
    frame = pd.read_csv(path)
    train = frame[frame["step"].notna()]
    ax.plot(train["step"], train["train/microbatch/loss_total"], linewidth=1.6, label=name)
ax.set_xlabel("step")
ax.set_ylabel("train loss (combined objective)")
ax.set_title("S round: warm-start continuation (note: objectives differ, compare trends only)")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
ax = axes[1]
for name in ("S1-bce-0p25", "S2-bce-0p5"):
    path = S_PATHS[name]
    if not path.exists():
        continue
    frame = pd.read_csv(path)
    train = frame[frame["step"].notna()]
    ax.plot(train["step"], train["train/microbatch/loss_sink_raw"], linewidth=1.6, label=name + " (raw)")
ax.set_xlabel("step")
ax.set_ylabel("sink head BCE (raw, pre-alpha)")
ax.set_title("S round: dedicated sink loss is training")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(FIG / "sinkloss_ab_progress.png", dpi=130)

s_collect = ROOT / "outputs/v41_sinkloss_ab/collect.json"
if s_collect.exists():
    s_rows = json.loads(s_collect.read_text())
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    ax = axes[0]
    names = [r["group"].split("-", 1)[0] for r in s_rows]
    dauc = [r.get("delta_val/auc", 0.0) for r in s_rows]
    dtpr = [r.get("delta_val/tpr_at_fpr_1e_2", 0.0) for r in s_rows]
    x = range(len(names))
    ax.bar([i - 0.2 for i in x], dauc, width=0.4, color="tab:blue", label="dAUC")
    ax.bar([i + 0.2 for i in x], dtpr, width=0.4, color="tab:green", label="dTPR@1%FPR")
    ax.axhline(0, color="k", linewidth=0.8)
    ax.set_xticks(list(x))
    ax.set_xticklabels(names)
    ax.set_title("S round (vs S0 control): sink loss works")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax = axes[1]
    labels = []
    aucs = []
    for r in s_rows:
        if r.get("status") != "ok":
            continue
        labels.append(r["group"].split("-", 1)[0])
        aucs.append(r["val/auc"])
    bars = ax.bar(labels, aucs, color=["tab:gray", "tab:blue", "tab:purple"])
    ax.axhline(0.937647, color="tab:red", linestyle=":", label="v2 final")
    ax.set_ylabel("AUC")
    ax.set_ylim(0.93, 0.94)
    ax.set_title("S round absolute AUC (2k warm start)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "sinkloss_ab_vals.png", dpi=130)

C3 = ROOT / "data/dma-kws/exp/stage2_qbyt/final/C3-sinkloss-50k/logs/C3-sinkloss-50k/version_0/metrics.csv"
c3 = val_frame(C3)
fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
ax = axes[0]
ax.plot(c1["step"], c1["val/auc"], "o-", color="tab:blue", label="C1 (sink readout, no sink loss)")
ax.plot(c2["step"], c2["val/auc"], "o-", color="tab:orange", label="C2 (sinusoidal PE)")
ax.plot(c3["step"], c3["val/auc"], "o-", color="tab:green", label="C3 (sink loss 0.25 from scratch)")
ax.plot([r[0] for r in V41], [r[1] for r in V41], "k--", marker="s", markersize=3, label="v4.1 ref")
ax.axhline(V2["auc"], color="tab:red", linestyle=":", label="v2 final")
ax.set_xlabel("optimizer step")
ax.set_ylabel("AUC")
ax.set_title("Full 50k runs: C1 vs C2 vs C3")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
ax = axes[1]
ax.plot(c1["step"], c1["val/tpr_at_fpr_1e_2"], "o-", color="tab:blue", label="C1")
ax.plot(c2["step"], c2["val/tpr_at_fpr_1e_2"], "o-", color="tab:orange", label="C2")
ax.plot(c3["step"], c3["val/tpr_at_fpr_1e_2"], "o-", color="tab:green", label="C3")
ax.axhline(V2["tpr1"], color="tab:red", linestyle=":", label="v2 final")
ax.set_xlabel("optimizer step")
ax.set_ylabel("TPR@1%FPR")
ax.set_title("Low-FPR tail of the three 50k runs")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(FIG / "c1_c2_c3.png", dpi=130)

ANALYSES = {
    "C1": ROOT / "outputs/v41_final/C1-sink-50k/C1-sink-50k_analysis.json",
    "S1 (bce .25)": ROOT / "outputs/v41_sinkloss_ab/probe_S1/analysis.json",
    "S2 (bce .5)": ROOT / "outputs/v41_sinkloss_ab/probe_S2/analysis.json",
    "C3 (50k)": ROOT / "outputs/v41_final/C3-sinkloss-50k/C3-sinkloss-50k_analysis.json",
}
rows = []
for name, path in ANALYSES.items():
    if not path.exists():
        continue
    data = json.loads(path.read_text())
    probe = data["sink_probe"]
    rows.append(
        (
            name,
            data["temperature_sweep"]["1.0"]["auc"],
            probe["sink_only_auc_eval_half"],
            probe["pooled_plus_sink_auc_eval_half"],
        )
    )
if rows:
    import numpy as np

    names = [r[0] for r in rows]
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(11, 5))
    width = 0.26
    ax.bar(x - width, [r[1] for r in rows], width, color="tab:blue", label="text head only (full split, T=1)")
    ax.bar(x, [r[2] for r in rows], width, color="tab:green", label="sink head only (fitted, eval half)")
    ax.bar(x + width, [r[3] for r in rows], width, color="tab:purple", label="text + fitted sink (eval half)")
    ax.axhline(V2["auc"], color="tab:red", linestyle=":", label="v2 final AUC")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylim(0.925, 0.945)
    ax.set_ylabel("AUC")
    ax.set_title("Head decomposition: sink-loss training improves the sink head but damages the text head")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "head_decomposition.png", dpi=130)

print("wrote", sorted(p.name for p in FIG.glob("*.png")))
print("C1 final:", c1.iloc[-1][["step", "val/auc", "val/eer", "val/tpr_at_fpr_1e_2", "val/tpr_at_fpr_1e_3"]].to_dict())
print("C2 latest:", c2.iloc[-1][["step", "val/auc", "val/tpr_at_fpr_1e_2"]].to_dict())
