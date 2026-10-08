#!/usr/bin/env python3
"""Launch, monitor and probe the C3 50k run (v4.2 candidate).

C3 = v4.1 recipe + additive-zero sink readout + stage2.sink_loss (bce, 0.25),
trained from scratch for 50k steps. Reuses the monitoring/export/probe logic
from scripts/pipeline_c1_c2.py, including the same early-abort floors.
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

C3_FLAGS = [
    "+stage2.qbyt_readout.sink_readout=additive",
    "+stage2.qbyt_readout.sink_zero_init=true",
    "stage2.sink_loss.enabled=true",
    "stage2.sink_loss.weight=0.25",
    "stage2.sink_loss.form=bce",
]


def main() -> None:
    PIPE.write_status(c3_queued="yes")
    verdict = PIPE.run_training("C3-sinkloss-50k", C3_FLAGS, PIPE.REF_V41, "v4.1")
    if verdict.startswith("ok"):
        PIPE.probe("C3-sinkloss-50k", C3_FLAGS)
    PIPE.write_status(pipeline="c3 done", c3_verdict=verdict)


if __name__ == "__main__":
    main()
