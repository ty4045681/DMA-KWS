#!/usr/bin/env python3
"""R1: paper-style continued fine-tune for zh-en (encoder unfrozen + 100:1).

Faithful to the author's train/two_stage/train_2_2ft.py:
  * start from a converged stage-2 checkpoint (our C1, 50k steps)
  * encoder trainable, single Adam over encoder + QbyT, lr 5e-4
  * warmup 2500, cosine to 100000 steps (the author's schedule) -- we run the
    first 13300 steps, so the LR trajectory matches the author's prefix
  * hard_negative_ratio 100, effective batch 768 (384 x accumulate 2)
  * gradient_clip_val 1.0, val every 2500 steps

Monitoring tracks C1's curve: a >0.01 drop warns, >0.02 aborts (plus the shared
hard floors from scripts/pipeline_c1_c2.py).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path("/home/ubuntu/dma-kws")
sys.path.insert(0, str(ROOT))


def _load_pipeline():
    spec = importlib.util.spec_from_file_location(
        "pipeline_c1_c2", str(ROOT / "scripts/pipeline_c1_c2.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PIPE = _load_pipeline()

FLAGS = [
    "+stage2.qbyt_readout.sink_readout=additive",
    "+stage2.qbyt_readout.sink_zero_init=true",
    "stage2.freeze_encoder=false",
    "stage2.hard_negative_ratio=100",
    "stage2.learning_rate=5e-4",
    "stage2.batch_size_per_gpu=384",
    "stage2.accumulate_grad_batches=2",
    "stage2.init_checkpoint=data/dma-kws/exp/stage2_qbyt/final/C1-sink-50k/checkpoints/C1-sink-50k/version_0/stage2_step050000.pt",
    "+stage2.init_allow_readout_mismatch=true",
    "stage2.warmup_steps=2500",
    "stage2.total_scheduler_steps=100000",
    "stage2.max_steps=13300",
    "stage2.num_workers=4",
]


def c1_curve():
    import pandas as pd

    paths = sorted(
        (ROOT / "data/dma-kws/exp/stage2_qbyt/final/C1-sink-50k/logs").glob(
            "*/version_*/metrics.csv"
        )
    )
    frame = pd.read_csv(paths[-1])
    rows = frame[frame["val/auc"].notna()]
    return [(int(r["step"]), float(r["val/auc"])) for _, r in rows.iterrows()]


def main() -> None:
    ref = c1_curve()
    print("C1 reference points:", len(ref), flush=True)
    PIPE.write_status(r1_started="yes")
    verdict = PIPE.run_training("R1-unfreeze-hardneg100", FLAGS, ref, "C1")
    if verdict.startswith("ok"):
        PIPE.probe("R1-unfreeze-hardneg100", FLAGS)
    PIPE.write_status(r1_verdict=verdict)


if __name__ == "__main__":
    main()
