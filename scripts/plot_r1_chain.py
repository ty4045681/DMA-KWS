#!/usr/bin/env python3
"""Scoreboard for the zh-en ladder: v4.1 -> v2 -> C1 -> v4.2(C1) -> R1 chain."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FIG = Path("/home/ubuntu/dma-kws/outputs/v41_final/figures")
ROWS = [
    ("v4.1", 0.93501, 0.13106, 0.18323, 0.01993, 0.54320),
    ("v2", 0.93765, 0.12864, 0.20207, 0.02347, 0.55236),
    ("C1", 0.93785, 0.12730, 0.20385, 0.02892, 0.55118),
    ("v4.2 (C1)", 0.94258, 0.12360, 0.24595, 0.03992, 0.56721),
    ("R1", 0.94630, 0.11638, 0.23585, 0.03114, 0.56284),
    ("SS-R1", 0.94779, 0.11498, 0.25135, 0.03272, 0.57032),
    ("v4.2 (R1), write-back", 0.95028, 0.11201, 0.26827, 0.04770, 0.57158),
]
labels = [r[0] for r in ROWS]
x = np.arange(len(ROWS))
colors = ["0.75", "tab:orange", "0.55", "tab:blue", "tab:green", "tab:cyan", "tab:red"]

fig, axes = plt.subplots(1, 4, figsize=(17, 4.6))
for ax, index, title, fmt in (
    (axes[0], 1, "AUC", "%.4f"),
    (axes[1], 2, "EER (lower better)", "%.4f"),
    (axes[2], 3, "TPR@1%FPR", "%.4f"),
    (axes[3], 4, "TPR@0.1%FPR", "%.4f"),
):
    values = [r[index] for r in ROWS]
    bars = ax.bar(x, values, color=colors)
    for bar, value in zip(bars, values):
        offset = max(values) * 0.01
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + offset,
            fmt % value,
            ha="center",
            fontsize=7,
            rotation=90,
        )
    low = min(values) - (max(values) - min(values)) * 0.25
    ax.set_ylim(low, max(values) + (max(values) - min(values)) * 0.35)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=7)
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.3, axis="y")

fig.suptitle(
    "zh-en ladder: R1 (unfrozen encoder + 100:1 continued fine-tune) + v4.2 sink pipeline",
    fontsize=12,
)
fig.tight_layout()
fig.savefig(FIG / "r1_chain_scoreboard.png", dpi=130)
print("wrote", FIG / "r1_chain_scoreboard.png")
