#!/usr/bin/env python3
"""Whole-clip TTS wake / false-trigger evaluation (no sliding windows).

Builds a JSONL manifest per TTS corpus (label 1 = exact keyword variant, 0 =
confusable near-miss) and scores every clip as one full-clip Stage-II input via
scripts/eval_stage2_clips.py. Then aggregates per-variant trigger rates and
plots a wake-vs-confusable summary.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/home/ubuntu/dma-kws")
PY = sys.executable
OUT = ROOT / "outputs/v41_keyword_eval"
CLIP_FA = ROOT / "data/dma-kws/exp/stage2_qbyt/fa_clip"
OUT.mkdir(parents=True, exist_ok=True)

MODELS = {
    "v42-zh": {
        "exp": "icefall_zipformer_stage2_eps_softmin_v41_musan_cached_zhen3m_50k",
        "sink": True,
        "ckpt": "data/dma-kws/exp/stage2_qbyt/sinkhead_stage2/SS-zh-en/checkpoints/SS-zh-en/version_0/stage2_step003000.pt",
    },
    "v42-paper": {
        "exp": "icefall_zipformer_stage2_eps_softmin_v41_musan_cached_paperstage1_50k",
        "sink": True,
        "ckpt": "data/dma-kws/exp/stage2_qbyt/sinkhead_stage2/SS-paperstage1/checkpoints/SS-paperstage1/version_0/stage2_step003000.pt",
    },
    "v2-zh": {
        "exp": "icefall_zipformer_stage2_v2_musan_cached_zhen3m_stream_50k",
        "sink": False,
        "ckpt": "data/dma-kws/exp/stage2_qbyt/checkpoints/v2-musan-zhen3m-stream-50k/v2-musan-zhen3m-stream-50k/version_1/stage2_step050000.pt",
    },
    "v2-paper": {
        "exp": "icefall_zipformer_stage2_v2_musan_cached_paperstage1_50k",
        "sink": False,
        "ckpt": "data/dma-kws/exp/stage2_qbyt/checkpoints/v2-musan-paperstage1-50k/v2-musan-paperstage1-50k/version_0/stage2_step050000.pt",
    },
}
CORPORA = {
    "eva": ("outputs/tts/hey_eva", "hey eva", "HH EY1 IY1 V AH0"),
    "google": ("outputs/tts/hey_google", "hey google", "HH EY1 G UW1 G AH0 L"),
}


def normalize(text: str) -> str:
    return " ".join(re.sub(r"[^a-z ]", "", text.lower()).split())


def build_manifest(corpus: str) -> Path:
    corpus_dir, keyword, phonemes = CORPORA[corpus]
    rows = []
    index_path = ROOT / corpus_dir / "index.jsonl"
    for line in index_path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        text = row["text"]
        rows.append(
            {
                "audio_path": str((ROOT / corpus_dir / row["audio_path"]).resolve()),
                "keyword": keyword,
                "keyword_phonemes": phonemes,
                "label": int(normalize(text) == keyword),
                "text": text,
            }
        )
    manifest = OUT / f"manifest_{corpus}.jsonl"
    manifest.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    positives = sum(r["label"] for r in rows)
    print(f"manifest {corpus}: {len(rows)} rows, {positives} exact / {len(rows) - positives} confusable", flush=True)
    return manifest


def run_clip_eval(tag: str, model: dict, corpus: str, manifest: Path) -> None:
    outdir = CLIP_FA / tag
    if (outdir / "summary.json").exists():
        print("SKIP", tag, flush=True)
        return
    args = [PY, "scripts/eval_stage2_clips.py", f"+experiment={model['exp']}"]
    if model["sink"]:
        args += [
            "+stage2.qbyt_readout.sink_readout=additive",
            "+stage2.qbyt_readout.sink_zero_init=true",
        ]
    args += [
        f"prep.manifest={manifest}",
        f"prep.stage2_ckpt={model['ckpt']}",
        f"prep.output_dir={outdir}",
        "prep.batch_size=64",
        "prep.num_workers=2",
    ]
    print("START", tag, flush=True)
    with open(OUT / "clip_eval.log", "a") as log:
        log.write(" ".join(args) + "\n")
        rc = subprocess.run(
            args,
            cwd=str(ROOT),
            env={**os.environ, "CUDA_VISIBLE_DEVICES": "0"},
            stdout=log,
            stderr=subprocess.STDOUT,
        ).returncode
    print("END", tag, "rc=", rc, flush=True)


def aggregate(corpus: str) -> dict:
    results = {}
    for name in MODELS:
        tag = f"clip-{name}-{corpus}"
        path = CLIP_FA / tag / "results.jsonl"
        if not path.exists():
            results[name] = None
            continue
        per_variant = {}
        exact_hits = exact_total = conf_hits = conf_total = 0
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            text = row.get("text") or ""
            hit = bool(row.get("detected"))
            label = int(row.get("label", 0))
            key = normalize(text)
            per_variant.setdefault(key, [0, 0])
            per_variant[key][0] += int(hit)
            per_variant[key][1] += 1
            if label == 1:
                exact_total += 1
                exact_hits += int(hit)
            else:
                conf_total += 1
                conf_hits += int(hit)
        results[name] = {
            "exact_wake_rate": exact_hits / exact_total if exact_total else None,
            "confusable_trigger_rate": conf_hits / conf_total if conf_total else None,
            "per_variant": {k: v[0] / v[1] for k, v in sorted(per_variant.items())},
        }
    return results


def main() -> None:
    manifests = {c: build_manifest(c) for c in CORPORA}
    jobs = []
    for name, model in MODELS.items():
        for corpus in CORPORA:
            tag = f"clip-{name}-{corpus}"
            if not (CLIP_FA / tag / "summary.json").exists():
                jobs.append((tag, model, corpus))
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [
            pool.submit(run_clip_eval, tag, model, corpus, manifests[corpus])
            for tag, model, corpus in jobs
        ]
        for future in futures:
            future.result()

    all_results = {c: aggregate(c) for c in CORPORA}
    (OUT / "tts_clip_results.json").write_text(json.dumps(all_results, indent=1))
    print(json.dumps(all_results, indent=1), flush=True)

    names = list(MODELS)
    x = range(len(names))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for ax, corpus, color in ((axes[0], "eva", "tab:blue"), (axes[1], "google", "tab:green")):
        res = all_results[corpus]
        wake = [res[n]["exact_wake_rate"] if res[n] else 0 for n in names]
        conf = [res[n]["confusable_trigger_rate"] if res[n] else 0 for n in names]
        width = 0.35
        ax.bar([i - width / 2 for i in x], wake, width, color=color, label="exact wake rate")
        ax.bar([i + width / 2 for i in x], conf, width, color="tab:red", label="confusable trigger rate")
        ax.set_xticks(list(x))
        ax.set_xticklabels(names)
        ax.set_ylim(0, 1)
        ax.set_ylabel("rate")
        ax.set_title(f"{corpus} TTS clips (whole-clip scoring)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle("Whole-clip TTS wake rate vs confusable false-trigger rate (threshold 0.5)", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "tts_clip_wake.png", dpi=130)
    print("CLIPEVALDONE", flush=True)


if __name__ == "__main__":
    main()
