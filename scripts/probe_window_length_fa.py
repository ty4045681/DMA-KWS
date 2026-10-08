#!/usr/bin/env python3
"""Does a shorter scoring window false-wake more often?

Sweeps the verifier's sliding-window length over a fixed subset of MUSAN
(music/noise) and LibriSpeech (speech) for two checkpoints, keeping
``window_sec == hop_sec`` so every window is an independent draw and the
"false alarms per hour" reading stays comparable across the grid.

Two numbers per cell matter and they answer different questions:

  per-window rate  does a shorter window make the model *decide* yes more often
  FA per hour      what an always-on deployment actually experiences, which
                   includes the fact that shorter windows mean more draws

Usage:
  .venv/bin/python scripts/probe_window_length_fa.py \
    --models base=<ckpt> joint=<ckpt> --hours 1.5 \
    --windows 0.5 1.0 2.0 3.0 --out outputs/hey_eva_window_probe
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PY = str(PROJECT_ROOT / ".venv/bin/python")
EXPERIMENT = "adapt_hey_eva_v42"
KEYWORD = "hey eva"
PHONEMES = "HH EY1 IY1 V AH0"

CORPORA = {
    "musan": (
        PROJECT_ROOT / "data/dma-kws/raw/musan",
        PROJECT_ROOT / "data/dma-kws/processed/musan_split/eval_musan.list",
    ),
    "librispeech": (
        PROJECT_ROOT / "data/dma-kws/raw/LibriSpeech",
        PROJECT_ROOT / "data/dma-kws/processed/ls-other-500-stride6.list",
    ),
}


def audio_duration(path: Path) -> float:
    import soundfile as sf

    info = sf.info(str(path))
    return float(info.frames) / float(info.samplerate)


def build_subset(source_list: Path, out_list: Path, target_hours: float) -> tuple[list[str], float]:
    """Take files in list order until the target duration is reached.

    Deterministic and shared by every cell of the grid, so window length is the
    only variable that changes between runs.
    """
    files = [line.strip() for line in source_list.read_text().splitlines() if line.strip()]
    chosen: list[str] = []
    total = 0.0
    for entry in files:
        path = Path(entry)
        if not path.is_file():
            continue
        try:
            seconds = audio_duration(path)
        except Exception:
            continue
        chosen.append(entry)
        total += seconds
        if total >= target_hours * 3600:
            break
    if not chosen:
        raise SystemExit(f"no readable audio found under {source_list}")
    out_list.write_text("\n".join(chosen) + "\n")
    return chosen, total / 3600.0


def run_cell(
    *,
    model_path: str,
    corpus_root: Path,
    list_path: Path,
    window: float,
    out_dir: Path,
) -> dict:
    if (out_dir / "summary.json").is_file():
        return json.loads((out_dir / "summary.json").read_text())
    args = [
        PY,
        str(PROJECT_ROOT / "scripts/eval_musan_fa.py"),
        f"+experiment={EXPERIMENT}",
        f"prep.keyword={KEYWORD!r}",
        f"prep.keyword_phonemes={PHONEMES!r}",
        f"prep.musan_root={corpus_root}",
        f"prep.musan_audio_list_path={list_path}",
        f"prep.stage2_ckpt={model_path}",
        f"prep.window_sec={window}",
        f"prep.hop_sec={window}",
        "prep.batch_size=64",
        "prep.num_workers=4",
        "prep.amp=fp16",
        f"prep.output_dir={out_dir}",
    ]
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "0"}
    result = subprocess.run(
        args, cwd=str(PROJECT_ROOT), env=env, capture_output=True, text=True
    )
    if result.returncode != 0 or not (out_dir / "summary.json").is_file():
        raise SystemExit(
            f"eval failed for window={window} ckpt={model_path}\n{result.stdout[-2000:]}\n{result.stderr[-2000:]}"
        )
    return json.loads((out_dir / "summary.json").read_text())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="NAME=CHECKPOINT pairs",
    )
    parser.add_argument("--hours", type=float, default=1.5)
    parser.add_argument("--windows", nargs="+", type=float, default=[0.5, 1.0, 2.0, 3.0])
    parser.add_argument("--out", type=Path, default=Path("outputs/hey_eva_window_probe"))
    args = parser.parse_args()

    models = {}
    for spec in args.models:
        name, _, path = spec.partition("=")
        if not path or not Path(path).is_file():
            raise SystemExit(f"checkpoint not found for {name}: {path!r}")
        models[name] = path

    args.out.mkdir(parents=True, exist_ok=True)
    subsets = {}
    for corpus, (root, source_list) in CORPORA.items():
        list_path = args.out / f"{corpus}_subset.list"
        files, hours = build_subset(source_list, list_path, args.hours)
        subsets[corpus] = {"list": str(list_path), "files": len(files), "hours": hours}
        print(f"subset {corpus}: {len(files)} files, {hours:.2f} h", flush=True)

    table: dict[str, dict] = {}
    for model_name, model_path in models.items():
        for corpus, (root, source_list) in CORPORA.items():
            list_path = Path(subsets[corpus]["list"])
            for window in args.windows:
                out_dir = args.out / "cells" / f"{model_name}_{corpus}_w{window:g}"
                summary = run_cell(
                    model_path=model_path,
                    corpus_root=root,
                    list_path=list_path,
                    window=window,
                    out_dir=out_dir,
                )
                metrics = summary.get("metrics", {}) or {}
                windows = float(metrics.get("num_samples") or 0.0)
                false_alarms = float(metrics.get("fp") or 0.0)
                cell = {
                    "window_sec": window,
                    "hours": summary.get("total_hours"),
                    "windows": windows,
                    "false_alarms": false_alarms,
                    "per_window_rate": (false_alarms / windows) if windows else None,
                    "fa_per_hour": metrics.get("fa_per_hour"),
                    "max_score": None,
                }
                # Score tail, independent of any threshold.
                scores = []
                with (out_dir / "results.jsonl").open() as handle:
                    for line in handle:
                        if line.strip():
                            row = json.loads(line)
                            if not row.get("skipped"):
                                scores.append(float(row["qbyt_score"]))
                if scores:
                    scores.sort()
                    cell["max_score"] = scores[-1]
                    cell["score_p99"] = scores[int(0.99 * (len(scores) - 1))]
                table[f"{model_name}|{corpus}|{window:g}"] = cell
                print(
                    f"{model_name:6s} {corpus:12s} w={window:<4g} "
                    f"windows={int(windows):6d} FA={int(false_alarms):5d} "
                    f"per-window={cell['per_window_rate']:.5f} FA/h={cell['fa_per_hour']:.3f}",
                    flush=True,
                )

    payload = {"models": models, "subsets": subsets, "cells": table}
    (args.out / "window_probe.json").write_text(json.dumps(payload, indent=1))
    print(f"wrote {args.out / 'window_probe.json'}", flush=True)


if __name__ == "__main__":
    main()
