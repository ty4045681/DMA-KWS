#!/usr/bin/env python3
"""Torchrun worker for one fixed-parameter LoRA adaptation trial."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from dma_kws.stage2.adapt import Stage2AdaptArgs
from dma_kws.stage2.sweep_adapt import (
    run_adaptation_training_trial,
    serialize_training_result,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True, type=Path)
    args = parser.parse_args()

    request = json.loads(args.request.read_text(encoding="utf-8"))
    result = run_adaptation_training_trial(
        request["config"],
        params=request["params"],
        base_args=Stage2AdaptArgs(**request["base_args"]),
        single_phase=bool(request.get("single_phase", False)),
    )

    if int(os.environ.get("RANK", "0")) == 0:
        result_path = Path(request["result_path"])
        temporary_path = result_path.with_suffix(result_path.suffix + ".tmp")
        temporary_path.write_text(
            json.dumps(serialize_training_result(result), indent=2),
            encoding="utf-8",
        )
        temporary_path.replace(result_path)


if __name__ == "__main__":
    main()
