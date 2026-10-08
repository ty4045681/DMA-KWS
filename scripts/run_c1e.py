#!/usr/bin/env python3
"""C1-E: 50k from scratch, C1 recipe minus the negative-tail CVaR loss.

E1 showed the CVaR term suppresses the low-FPR tail (TPR@1%FPR +0.017 vs the
matched control in 2k warm-start steps). This run is the from-scratch version;
if it lands at or above C1, it becomes the v4.2 main-training recipe.
Reuses the C1 monitoring/export/probe logic (same floors, REF_V41).
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

C1E_FLAGS = [
    "+stage2.qbyt_readout.sink_readout=additive",
    "+stage2.qbyt_readout.sink_zero_init=true",
    "stage2.negative_tail_loss.enabled=false",
]


def main() -> None:
    PIPE.write_status(c1e_queued="yes")
    verdict = PIPE.run_training("C1E-cvaroff-50k", C1E_FLAGS, PIPE.REF_V41, "v4.1")
    if verdict.startswith("ok"):
        PIPE.probe("C1E-cvaroff-50k", C1E_FLAGS)
    PIPE.write_status(pipeline="c1e done", c1e_verdict=verdict)


if __name__ == "__main__":
    main()
