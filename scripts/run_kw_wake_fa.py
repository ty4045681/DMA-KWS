#!/usr/bin/env python3
"""Wake-rate / false-alarm evaluation for 4 models x 2 keywords.

Models:
  v42-zh     SS-zh-en (zh-en-3M streaming, v4.2, 3k sink-head second stage)
  v42-paper  SS-paperstage1 (paper-stage1 conformer fullctx, v4.2, 3k second stage)
  v2-zh      v2-musan-zhen3m-stream-50k (50k)
  v2-paper   v2-musan-paperstage1-50k (50k)
Keywords:
  hey eva    HH EY1 IY1 V AH0
  hey google HH EY1 G UW1 G AH0 L

FA: MUSAN held-out, 3 s and 1 s grids, threshold 0.5, fp16, batch 64.
Wake rate: recall at threshold 0.5 over the hard-split positives (hey eva audio;
no hey-google positive corpus exists, so hey-google wake rate is reported N/A).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/home/ubuntu/dma-kws")
PY = sys.executable
FA = ROOT / "data/dma-kws/exp/stage2_qbyt/fa"
MUSAN_LIST = "data/dma-kws/processed/musan_split/eval_musan.list"
OUT = ROOT / "outputs/v41_keyword_eval"
OUT.mkdir(parents=True, exist_ok=True)

MODELS = {
    "v42-zh": {
        "exp": "icefall_zipformer_stage2_eps_softmin_v41_musan_cached_zhen3m_50k",
        "sink": True,
        "ckpt": "data/dma-kws/exp/stage2_qbyt/sinkhead_stage2/SS-zh-en/checkpoints/SS-zh-en/version_0/stage2_step003000.pt",
        "probe": "outputs/v41_sinkhead_stage2/probe_SS-zh-en/probe.npz",
    },
    "v42-paper": {
        "exp": "icefall_zipformer_stage2_eps_softmin_v41_musan_cached_paperstage1_50k",
        "sink": True,
        "ckpt": "data/dma-kws/exp/stage2_qbyt/sinkhead_stage2/SS-paperstage1/checkpoints/SS-paperstage1/version_0/stage2_step003000.pt",
        "probe": "outputs/v41_sinkhead_stage2/probe_SS-paperstage1/probe.npz",
    },
    "v2-zh": {
        "exp": "icefall_zipformer_stage2_v2_musan_cached_zhen3m_stream_50k",
        "sink": False,
        "ckpt": "data/dma-kws/exp/stage2_qbyt/checkpoints/v2-musan-zhen3m-stream-50k/v2-musan-zhen3m-stream-50k/version_1/stage2_step050000.pt",
        "probe": "outputs/v41_keyword_eval/probe_v2-zh/probe.npz",
    },
    "v2-paper": {
        "exp": "icefall_zipformer_stage2_v2_musan_cached_paperstage1_50k",
        "sink": False,
        "ckpt": "data/dma-kws/exp/stage2_qbyt/checkpoints/v2-musan-paperstage1-50k/v2-musan-paperstage1-50k/version_0/stage2_step050000.pt",
        "probe": "outputs/v41_keyword_eval/probe_v2-paper/probe.npz",
    },
}
KEYWORDS = {
    "hey eva": "HH EY1 IY1 V AH0",
    "hey google": "HH EY1 G UW1 G AH0 L",
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


def run_fa(tag, model, keyword, phonemes, window, hop):
    outdir = FA / tag
    if (outdir / "summary.json").exists():
        print("SKIP", tag, flush=True)
        return
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
        "prep.musan_root=data/dma-kws/raw/musan",
        f"prep.musan_audio_list_path={MUSAN_LIST}",
        f"prep.stage2_ckpt={model['ckpt']}",
        f"prep.window_sec={window}",
        f"prep.hop_sec={hop}",
        "prep.batch_size=64",
        "prep.num_workers=4",
        "prep.amp=fp16",
        f"prep.output_dir={outdir}",
    ]
    print("START", tag, flush=True)
    rc = sh(args, OUT / "fa.log")
    print("END", tag, "rc=", rc, flush=True)


def run_probe(name, model):
    outdir = OUT / f"probe_{name}"
    outdir.mkdir(parents=True, exist_ok=True)
    if (outdir / "probe.npz").exists():
        print("SKIP probe", name, flush=True)
        return
    args = [PY, "scripts/probe_qbyt_v41_offline.py", f"+experiment={model['exp']}"]
    if model["sink"]:
        args += [
            "+stage2.qbyt_readout.sink_readout=additive",
            "+stage2.qbyt_readout.sink_zero_init=true",
        ]
    args += [
        f"prep.checkpoint={model['ckpt']}",
        f"prep.output_dir={outdir}",
        "+prep.output_name=probe.npz",
        "prep.batch_size=128",
        "prep.num_workers=4",
    ]
    print("PROBE", name, flush=True)
    rc = sh(args, OUT / "probe.log")
    print("PROBE rc=", rc, flush=True)


def wake_rate(probe_path):
    import numpy as np

    data = np.load(str(probe_path))
    labels = np.asarray(data["labels"]).reshape(-1)
    logits = np.asarray(data["logits"]).reshape(-1)
    positive = logits[labels == 1]
    if positive.size == 0:
        return None
    return float((positive > 0.5).mean())


def main() -> None:
    run_probe("v2-zh", MODELS["v2-zh"])
    run_probe("v2-paper", MODELS["v2-paper"])
    for grid_name, (window, hop) in (("musan", (3.0, 3.0)), ("musan-1s0", (1.0, 1.0))):
        for name, model in MODELS.items():
            for keyword, phonemes in KEYWORDS.items():
                kw_tag = "eva" if keyword == "hey eva" else "google"
                tag = f"kw-{name}-{kw_tag}-{grid_name}"
                run_fa(tag, model, keyword, phonemes, window, hop)

    results = {}
    for name, model in MODELS.items():
        results[name] = {"wake_eva": wake_rate(model["probe"])}
        for keyword, kw_tag in (("hey eva", "eva"), ("hey google", "google")):
            for grid in ("musan", "musan-1s0"):
                summary = FA / f"kw-{name}-{kw_tag}-{grid}" / "summary.json"
                fah = None
                if summary.exists():
                    metrics = json.loads(summary.read_text()).get("metrics", {}) or {}
                    fah = metrics.get("fa_per_hour")
                results[name][f"fa_{kw_tag}_{grid}"] = fah
    (OUT / "results.json").write_text(json.dumps(results, indent=1))
    print("RESULTS", json.dumps(results, indent=1), flush=True)
    print("KEYWORDEVALDONE", flush=True)


if __name__ == "__main__":
    main()
