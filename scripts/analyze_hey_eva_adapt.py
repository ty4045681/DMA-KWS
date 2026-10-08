#!/usr/bin/env python3
"""Score-slice analysis for the "hey eva" adaptation.

Joins an ``eval_stage2_clips.py`` ``results.jsonl`` back to the corpus metadata
that produced it, so a single held-out number can be broken down by speaker,
recording provenance (original vs augmentation), augmentation type and spoken
text.  The point is to keep the headline metric explainable: a real-data AUC is
dominated by the augmentation mix, and a per-slice view says which part of the
domain actually moved.

Usage:
  python scripts/analyze_hey_eva_adapt.py \
    --eval-dir outputs/.../eval_clips_adapted \
    --adapt-manifest data/dma-kws/processed/adapt/hey_eva_v42/manifests/real_eval.csv \
    --source-manifest data/dma-kws/raw/hey_eva_real_v2/manifests/real_reviewed_abs.csv \
    --out outputs/hey_eva_v42_eval/real_adapted.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve


def load_scores(eval_dir: Path) -> pd.DataFrame:
    rows = []
    with (eval_dir / "results.jsonl").open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("skipped"):
                continue
            rows.append(
                {
                    "audio_path": row["audio_path"],
                    "score": float(row["qbyt_score"]),
                    "label": int(row["label"]),
                    "detected": bool(row.get("detected", False)),
                }
            )
    if not rows:
        raise SystemExit(f"no scored rows in {eval_dir / 'results.jsonl'}")
    return pd.DataFrame(rows)


def metrics(labels: np.ndarray, scores: np.ndarray) -> dict:
    if len(np.unique(labels)) < 2:
        return {"num_pos": int((labels == 1).sum()), "num_neg": int((labels == 0).sum())}
    fpr, tpr, thresholds = roc_curve(labels, scores)
    auc = float(roc_auc_score(labels, scores))
    fnr = 1.0 - tpr
    eer_index = int(np.nanargmin(np.abs(fnr - fpr)))
    out = {
        "num_pos": int((labels == 1).sum()),
        "num_neg": int((labels == 0).sum()),
        "auc": auc,
        "eer": float((fpr[eer_index] + fnr[eer_index]) / 2.0),
        "eer_threshold": float(thresholds[eer_index]),
        "tpr_at_fpr_1e_2": float(tpr[np.searchsorted(fpr, 1e-2)]),
        "tpr_at_fpr_1e_3": float(tpr[np.searchsorted(fpr, 1e-3)]),
        "deploy_tpr": float((scores[labels == 1] > 0.5).mean()),
        "deploy_fpr": float((scores[labels == 0] > 0.5).mean()),
        "score_pos_median": float(np.median(scores[labels == 1])),
        "score_neg_p95": float(np.percentile(scores[labels == 0], 95)),
    }
    return out


def slice_metrics(frame: pd.DataFrame, field: str, *, min_rows: int = 10) -> dict:
    out = {}
    for value, group in frame.groupby(field, dropna=False):
        if len(group) < min_rows or group["label"].nunique() < 2:
            continue
        out[str(value)] = metrics(group["label"].to_numpy(), group["score"].to_numpy())
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-dir", type=Path, required=True)
    parser.add_argument("--adapt-manifest", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--source", choices=["real", "tts"], required=True)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    scores = load_scores(args.eval_dir)
    adapt = pd.read_csv(args.adapt_manifest)
    source = pd.read_csv(args.source_manifest)

    # The prepared manifest keeps a pointer back into the corpus manifest; use it
    # rather than re-parsing paths so renamed files stay traceable.  Only the
    # provenance columns come from the corpus manifest: both frames carry
    # audio_path/text/label, and merging them unprefixed would suffix both.
    source = source.reset_index().rename(columns={"index": "source_row"})
    provenance = [
        column
        for column in ("is_original", "aug_type", "category", "wake_word", "tts_provider")
        if column in source.columns
    ]
    meta = source[["source_row", *provenance]]
    joined = adapt.rename(columns={"source_row": "corpus_row"}).merge(
        meta, left_on="corpus_row", right_on="source_row", how="left"
    )
    if "audio_path" not in joined.columns:
        raise SystemExit("adapt manifest lost its audio_path column during the merge")
    joined["audio_path"] = joined["audio_path"].astype(str)
    scores["audio_path"] = scores["audio_path"].astype(str)

    merge_columns = ["audio_path", "text", "speaker_id", *provenance]
    merged = scores.merge(joined[merge_columns], on="audio_path", how="left")
    unmatched = int(merged["speaker_id"].isna().sum())
    if unmatched:
        raise SystemExit(f"{unmatched}/{len(merged)} scored clips did not join the manifest")

    def present(*fields: str) -> list[str]:
        return [field for field in fields if field in merged.columns]

    report = {
        "eval_dir": str(args.eval_dir),
        "manifest": str(args.adapt_manifest),
        "overall": metrics(merged["label"].to_numpy(), merged["score"].to_numpy()),
        "by_original": {},
        "by_aug_type": slice_metrics(merged, "aug_type") if "aug_type" in merged else {},
        "by_speaker": slice_metrics(merged, "speaker_id"),
        "by_text": slice_metrics(merged, "text"),
    }
    for is_original, group in merged.groupby("is_original") if "is_original" in merged else []:
        report["by_original"]["original" if is_original else "augmented"] = metrics(
            group["label"].to_numpy(), group["score"].to_numpy()
        )
    positives = merged[merged["label"] == 1]
    negatives = merged[merged["label"] == 0]
    group_field = "aug_type" if "aug_type" in merged else "tts_provider"
    report["positives_by_group"] = {
        str(k): {
            "clips": int(len(g)),
            "score_median": float(g["score"].median()),
            "wake_rate_at_0.5": float((g["score"] > 0.5).mean()),
        }
        for k, g in positives.groupby(group_field)
    }
    report["negatives_by_text"] = {
        str(k): {
            "clips": int(len(g)),
            "score_median": float(g["score"].median()),
            "trigger_rate_at_0.5": float((g["score"] > 0.5).mean()),
        }
        for k, g in negatives.groupby("text")
    }

    text = json.dumps(report, indent=1, ensure_ascii=False)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
        print(f"wrote {args.out}")
    overall = report["overall"]
    print(
        f"overall: AUC {overall['auc']:.4f} EER {overall['eer']:.4f} "
        f"TPR@1%FPR {overall['tpr_at_fpr_1e_2']:.4f} "
        f"recall@0.5 {overall['deploy_tpr']:.4f} FTR@0.5 {overall['deploy_fpr']:.4f}"
    )


if __name__ == "__main__":
    main()
