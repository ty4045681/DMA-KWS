#!/usr/bin/env python3
"""Paper-faithful 50k run for zh-en: encoder unfrozen + hard_negative_ratio 100.

The 3k warm-start arms (U1/U2/H1/H2) all landed below the converged base, so the
paper settings only make sense as a from-scratch recipe. This run starts from the
zh-en-3M encoder init (no stage-2 checkpoint) with:

  stage2.freeze_encoder=false      # encoder fine-tuned end to end
  stage2.hard_negative_ratio=100   # paper sampling ratio (we used 1)
  stage2.learning_rate=1e-4        # 5x lower than the frozen-encoder C1 recipe

Monitoring and the post-run probe reuse scripts/pipeline_c1_c2.py (same floors,
REF_V41 tracking).
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
    "stage2.learning_rate=1e-4",
    "stage2.warmup_steps=500",
]


def main() -> None:
    PIPE.write_status(p1_paper_50k_queued="yes")
    verdict = PIPE.run_training("P1-paper-50k", FLAGS, PIPE.REF_V41, "v4.1")
    if verdict.startswith("ok"):
        PIPE.probe("P1-paper-50k", FLAGS)
    PIPE.write_status(pipeline="p1 done", p1_verdict=verdict)


if __name__ == "__main__":
    main()
