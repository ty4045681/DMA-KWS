#!/usr/bin/env python3
"""ROC (AUC) and DET (EER) panels for the TTS whole-clip evaluation.

8 tasks (4 models x 2 corpora), each with an ROC and a DET panel, arranged as
4 rows (models) x 4 columns (eva ROC, eva DET, google ROC, google DET).
Labels: 1 = exact keyword variant, 0 = confusable near-miss.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

ROOT = Path("/home/ubuntu/dma-kws")
CLIP_FA = ROOT / "data/dma-kws/exp/stage2_qbyt/fa_clip"
OUT = ROOT / "outputs/v41_keyword_eval"
MODELS = ["v42-zh", "v42-paper", "v2-zh", "v2-paper", "v42r1"]
CORPORA = ["eva", "google"]


def load(model: str, corpus: str):
    path = CLIP_FA / f"clip-{model}-{corpus}" / "results.jsonl"
    labels, scores = [], []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("skipped"):
            continue
        labels.append(int(row["label"]))
        scores.append(float(row["qbyt_score"]))
    return np.asarray(labels), np.asarray(scores)


def eer_point(fpr, tpr):
    fnr = 1.0 - tpr
    index = int(np.nanargmin(np.abs(fnr - fpr)))
    return float((fnr[index] + fpr[index]) / 2.0), float(fpr[index])


fig, axes = plt.subplots(5, 4, figsize=(17, 18.5))
rows = []
for i, model in enumerate(MODELS):
    for j, corpus in enumerate(CORPORA):
        labels, scores = load(model, corpus)
        fpr, tpr, _ = roc_curve(labels, scores)
        auc = float(roc_auc_score(labels, scores))
        eer, eer_fpr = eer_point(fpr, tpr)
        rows.append((model, corpus, auc, eer, len(labels), int(labels.sum())))

        ax = axes[i, 2 * j]
        ax.plot(fpr, tpr, color="tab:blue", linewidth=1.6)
        ax.plot([0, 1], [0, 1], "k:", linewidth=0.8)
        ax.set_title(f"{model} | {corpus}: ROC AUC={auc:.4f}", fontsize=9)
        ax.set_xlabel("FPR", fontsize=8)
        ax.set_ylabel("TPR", fontsize=8)
        ax.grid(alpha=0.3)

        ax = axes[i, 2 * j + 1]
        fnr = 1.0 - tpr
        ax.plot(fpr, fnr, color="tab:red", linewidth=1.6)
        ax.plot([0, 1], [0, 1], "k:", linewidth=0.8)
        ax.plot([eer_fpr], [eer], "ko", markersize=4)
        ax.set_title(f"{model} | {corpus}: EER={eer * 100:.2f}%", fontsize=9)
        ax.set_xlabel("FPR", fontsize=8)
        ax.set_ylabel("FNR", fontsize=8)
        ax.grid(alpha=0.3)

fig.suptitle(
    "TTS whole-clip evaluation: ROC (AUC) and DET (EER), exact variants vs confusable near-misses",
    fontsize=13,
)
fig.tight_layout()
fig.savefig(OUT / "tts_clip_roc_eer.png", dpi=120)
print("wrote", OUT / "tts_clip_roc_eer.png")
print("| model | corpus | clips | exact | AUC | EER |")
print("|---|---|---|---|---|---|")
for model, corpus, auc, eer, total, positives in rows:
    print("| %s | %s | %d | %d | %.4f | %.2f%% |" % (model, corpus, total, positives, auc, eer * 100))
