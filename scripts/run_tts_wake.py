#!/usr/bin/env python3
"""TTS wake-rate evaluation: 4 models x 2 TTS corpora (hey eva / hey google).

The TTS corpora contain exact-phrase variants AND confusable near-misses
(Hey Eve / Ivan / Eric / Ava... and Hey Goggle / Gaggle / Goober...), so per
variant-group wake and false-trigger rates are reported alongside the overall
clip trigger rate. Scoring runs through eval_musan_fa.py (same verifier,
threshold 0.5, 3 s window); per-window decisions come from results.jsonl.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/home/ubuntu/dma-kws")
PY = sys.executable
FA = ROOT / "data/dma-kws/exp/stage2_qbyt/fa"
OUT = ROOT / "outputs/v41_keyword_eval"
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
# TTS clips are ~1 s long; the FA window is 3 s, so the sliding-window path
# skips anything shorter than the window. outputs/tts_padded holds silence-padded
# (centred, 3.0 s) copies of the same clips plus a copy of index.jsonl.
TTS = {
    "eva": ("outputs/tts_padded/hey_eva", "hey eva", "HH EY1 IY1 V AH0"),
    "google": ("outputs/tts_padded/hey_google", "hey google", "HH EY1 G UW1 G AH0 L"),
}


def sh(args, log_path):
    with open(str(log_path), "a") as handle:
        handle.write(" ".join(str(a) for a in args) + "\n")
        return subprocess.run(
            args,
            cwd=str(ROOT),
            env={**os.environ, "CUDA_VISIBLE_DEVICES": "0"},
            stdout=handle,
            stderr=subprocess.STDOUT,
        ).returncode


def run_tts_fa(tag, model, tts_dir, keyword, phonemes):
    outdir = FA / tag
    if (outdir / "summary.json").exists():
        print("SKIP", tag, flush=True)
        return
    wavs = sorted(str(p.resolve()) for p in Path(tts_dir).rglob("*.wav"))
    list_path = OUT / f"list_{tag}.txt"
    list_path.write_text("\n".join(wavs) + "\n")
    args = [
        PY, "scripts/eval_musan_fa.py",
        f"+experiment={model['exp']}",
    ]
    if model["sink"]:
        args += [
            "+stage2.qbyt_readout.sink_readout=additive",
            "+stage2.qbyt_readout.sink_zero_init=true",
        ]
    args += [
        f"prep.keyword={keyword}",
        f"prep.keyword_phonemes={phonemes}",
        f"prep.stage2_ckpt={model['ckpt']}",
        f"prep.musan_root={tts_dir}",
        f"prep.musan_audio_list_path={list_path}",
        "prep.window_sec=3.0",
        "prep.hop_sec=3.0",
        "prep.batch_size=128",
        "prep.num_workers=2",
        "prep.amp=fp16",
        f"prep.output_dir={outdir}",
    ]
    print("START", tag, flush=True)
    rc = sh(args, OUT / "tts_fa.log")
    print("END", tag, "rc=", rc, flush=True)


def load_index(tts_dir):
    rows = {}
    for line in (Path(tts_dir) / "index.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        rows[Path(row["audio_path"]).name] = row["text"]
    return rows


def normalize(text):
    lowered = re.sub(r"[^a-z ]", "", text.lower()).strip()
    return " ".join(lowered.split())


def analyze(tag, tts_name):
    results_path = FA / tag / "results.jsonl"
    if not results_path.exists():
        return None
    index = load_index(TTS[tts_name][0])
    per_clip = {}
    for line in results_path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        base = Path(row["audio_path"]).name
        if base not in index:
            continue
        detected = bool(row.get("detected"))
        per_clip.setdefault(base, False)
        per_clip[base] = per_clip[base] or detected
    total = len(per_clip)
    if total == 0:
        return None
    fired = sum(per_clip.values())
    exact_keys = {"hey eva": {"hey eva"}, "hey google": {"hey google"}}[tts_name]
    exact = 0
    confusable = 0
    exact_total = 0
    confusable_total = 0
    for base, hit in per_clip.items():
        norm = normalize(index[base])
        if norm in exact_keys:
            exact_total += 1
            exact += int(hit)
        else:
            confusable_total += 1
            confusable += int(hit)
    return {
        "clips": total,
        "overall_wake_rate": fired / total,
        "exact_wake_rate": (exact / exact_total) if exact_total else None,
        "exact_total": exact_total,
        "confusable_trigger_rate": (confusable / confusable_total) if confusable_total else None,
        "confusable_total": confusable_total,
    }


def main() -> None:
    results = {}
    jobs = []
    for name, model in MODELS.items():
        for tts_name, (tts_dir, keyword, phonemes) in TTS.items():
            tag = f"tts-{name}-{tts_name}"
            outdir = FA / tag
            if not (outdir / "summary.json").exists():
                jobs.append((tag, name, tts_name, model, tts_dir, keyword, phonemes))
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [
            pool.submit(run_tts_fa, tag, model, tts_dir, keyword, phonemes)
            for tag, _n, _t, model, tts_dir, keyword, phonemes in jobs
        ]
        for future in futures:
            future.result()
    for name in MODELS:
        for tts_name in TTS:
            tag = f"tts-{name}-{tts_name}"
            results[f"{name}-{tts_name}"] = analyze(tag, tts_name)
            print(tag, json.dumps(results[f"{name}-{tts_name}"]), flush=True)
    (OUT / "tts_results.json").write_text(json.dumps({"matching": results}, indent=1))

    missing = [key for key, value in results.items() if value is None]
    if missing:
        print("missing results:", missing, flush=True)
    names = list(MODELS)
    x = range(len(names))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    ax = axes[0]
    width = 0.35
    eva_exact = [results[f"{n}-eva"]["exact_wake_rate"] for n in names]
    eva_conf = [results[f"{n}-eva"]["confusable_trigger_rate"] for n in names]
    ax.bar([i - width / 2 for i in x], eva_exact, width, color="tab:blue", label="Hey Eva exact wake rate")
    ax.bar([i + width / 2 for i in x], eva_conf, width, color="tab:orange", label="confusable trigger rate")
    ax.set_xticks(list(x))
    ax.set_xticklabels(names)
    ax.set_ylim(0, 1)
    ax.set_ylabel("rate")
    ax.set_title("Hey Eva TTS: wake vs confusable triggers")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax = axes[1]
    google_exact = [results[f"{n}-google"]["exact_wake_rate"] for n in names]
    google_conf = [results[f"{n}-google"]["confusable_trigger_rate"] for n in names]
    ax.bar([i - width / 2 for i in x], google_exact, width, color="tab:green", label="Hey Google exact wake rate")
    ax.bar([i + width / 2 for i in x], google_conf, width, color="tab:red", label="confusable trigger rate")
    ax.set_xticks(list(x))
    ax.set_xticklabels(names)
    ax.set_ylim(0, 1)
    ax.set_ylabel("rate")
    ax.set_title("Hey Google TTS: wake vs confusable triggers")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.suptitle("TTS wake-rate evaluation (threshold 0.5, 3 s window)", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "tts_wake.png", dpi=130)
    print("TTSDONE", flush=True)


if __name__ == "__main__":
    main()
