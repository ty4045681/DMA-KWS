#!/usr/bin/env python3
"""Does lengthening a held-out "Hey eva" recording change the wake rate?

The reviewed clips are scored at their native length (the evaluator's
``audio_padding_ms`` is 0/0), and their median length is about 2.4 s. This probe
centre-pads silence onto every clip that is shorter than a target length, then
re-scores the whole held-out set, so the only thing that changes between cells
is how much silence surrounds the phrase. Clips already at or above the target
are left byte-identical.

Results are reported per native-length cohort as well as overall, because the
overall number mixes "clips that were padded" with "clips that were not".

Usage:
  .venv/bin/python scripts/probe_hey_eva_length_wake.py \
    --models joint=<ckpt> base=<ckpt> --lengths 2.0 2.5 3.0 \
    --out outputs/hey_eva_len_probe
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from sklearn.metrics import roc_auc_score, roc_curve

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PY = str(PROJECT_ROOT / ".venv/bin/python")
EXPERIMENT = "adapt_hey_eva_v42"
SOURCE_MANIFEST = PROJECT_ROOT / "outputs/hey_eva_v42_eval/real_eval_clips.csv"

COHORTS = {"<2.0 s": (0.0, 2.0), "2.0-2.5 s": (2.0, 2.5), ">=2.5 s": (2.5, 99.0)}


def native_durations(frame: pd.DataFrame) -> dict[str, float]:
    out = {}
    for path in frame["audio_path"]:
        info = sf.info(str(path))
        out[str(path)] = float(info.frames) / float(info.samplerate)
    return out


def build_padded(
    frame: pd.DataFrame, target: float, out_root: Path
) -> tuple[pd.DataFrame, int]:
    """Centre-pad silence to ``target`` seconds; return a manifest on the copies."""
    padded = frame.copy()
    changed = 0
    for index, row in frame.iterrows():
        source = Path(str(row["audio_path"]))
        audio, rate = sf.read(str(source), dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        length = len(audio) / rate
        if length >= target:
            continue
        changed += 1
        want = int(round(target * rate))
        total = want - len(audio)
        left = total // 2
        right = total - left
        padded_audio = np.concatenate(
            [np.zeros(left, dtype=np.float32), audio, np.zeros(right, dtype=np.float32)]
        )
        dest = out_root / f"{length:.3f}_{source.parent.name}" / source.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(dest), padded_audio, rate, subtype="PCM_16")
        # Absolute on purpose: eval_stage2_clips.py resolves a relative
        # audio_path against the manifest's own directory.
        padded.at[index, "audio_path"] = str(dest.resolve())
    return padded, changed


def metrics(labels: np.ndarray, scores: np.ndarray) -> dict:
    if len(np.unique(labels)) < 2:
        return {}
    fpr, tpr, thresholds = roc_curve(labels, scores)
    fnr = 1.0 - tpr
    eer_index = int(np.nanargmin(np.abs(fnr - fpr)))
    return {
        "auc": float(roc_auc_score(labels, scores)),
        "eer": float((fpr[eer_index] + fnr[eer_index]) / 2.0),
        "tpr_at_fpr_1e_2": float(tpr[np.searchsorted(fpr, 1e-2)]),
        "wake_rate_at_0.5": float((scores[labels == 1] > 0.5).mean()),
        "near_miss_trigger_at_0.5": float((scores[labels == 0] > 0.5).mean()),
    }


def score(model_path: str, manifest: Path, out_dir: Path) -> pd.DataFrame:
    if not (out_dir / "results.jsonl").is_file():
        args = [
            PY,
            str(PROJECT_ROOT / "scripts/eval_stage2_clips.py"),
            f"+experiment={EXPERIMENT}",
            f"prep.manifest={manifest}",
            f"prep.stage2_ckpt={model_path}",
            f"prep.output_dir={out_dir}",
            "run.device=cuda",
        ]
        result = subprocess.run(
            args,
            cwd=str(PROJECT_ROOT),
            env={**os.environ, "CUDA_VISIBLE_DEVICES": "0"},
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise SystemExit(f"eval failed: {manifest}\n{result.stdout[-1500:]}\n{result.stderr[-1500:]}")
    rows = []
    with (out_dir / "results.jsonl").open() as handle:
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
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", required=True, help="NAME=CHECKPOINT")
    parser.add_argument("--lengths", nargs="+", type=float, default=[2.0, 2.5, 3.0])
    parser.add_argument("--out", type=Path, default=Path("outputs/hey_eva_len_probe"))
    args = parser.parse_args()

    models = {}
    for spec in args.models:
        name, _, path = spec.partition("=")
        if not Path(path).is_file():
            raise SystemExit(f"checkpoint not found: {path!r}")
        models[name] = path

    frame = pd.read_csv(SOURCE_MANIFEST)
    native = native_durations(frame)
    frame["native_dur"] = frame["audio_path"].map(native)
    frame["cohort"] = pd.cut(
        frame["native_dur"],
        bins=[0.0, 2.0, 2.5, 99.0],
        labels=list(COHORTS),
        right=False,
    )

    report: dict = {
        "source_manifest": str(SOURCE_MANIFEST),
        "native_length_stats": {
            "positives": frame[frame.label == 1]["native_dur"].describe().to_dict(),
            "negatives": frame[frame.label == 0]["native_dur"].describe().to_dict(),
            "positives_below": {
                str(bound): float((frame[frame.label == 1]["native_dur"] < bound).mean())
                for bound in (2.0, 2.5, 3.0)
            },
        },
        "cells": {},
    }
    print("native positive lengths:", json.dumps(report["native_length_stats"]["positives"], default=float))

    conditions: list[tuple[str, pd.DataFrame, int]] = [
        ("native", frame, 0),
        *[
            (f"pad{target:g}", *build_padded(frame, target, args.out / f"audio_{target:g}"))
            for target in args.lengths
        ],
    ]

    for label, table, changed in conditions:
        manifest = args.out / f"manifest_{label}.csv"
        table.to_csv(manifest, index=False)
        print(f"condition {label}: {changed} clips padded -> {manifest}", flush=True)
        for model_name, model_path in models.items():
            out_dir = args.out / "cells" / f"{model_name}_{label}"
            scores = score(model_path, manifest, out_dir)
            merged = scores.merge(
                table[["keyword" if False else "audio_path", "native_dur", "cohort", "speaker_id"]],
                on="audio_path",
                how="left",
            )
            if merged["native_dur"].isna().any():
                raise SystemExit(f"{label}/{model_name}: score rows did not join the manifest")
            overall = metrics(merged["label"].to_numpy(), merged["score"].to_numpy())
            by_cohort = {}
            for cohort, group in merged.groupby("cohort", observed=True):
                if group["label"].nunique() < 2:
                    continue
                by_cohort[str(cohort)] = {
                    "clips": int(len(group)),
                    **metrics(group["label"].to_numpy(), group["score"].to_numpy()),
                }
            report["cells"][f"{model_name}|{label}"] = {
                "padded_clips": changed,
                "overall": overall,
                "by_cohort": by_cohort,
            }
            print(
                f"  {model_name:6s} {label:8s} AUC={overall['auc']:.4f} "
                f"wake@0.5={overall['wake_rate_at_0.5']:.4f} "
                f"near-miss@0.5={overall['near_miss_trigger_at_0.5']:.4f}",
                flush=True,
            )

    (args.out / "length_probe.json").write_text(json.dumps(report, indent=1, default=float))
    print(f"wrote {args.out / 'length_probe.json'}", flush=True)


if __name__ == "__main__":
    main()
