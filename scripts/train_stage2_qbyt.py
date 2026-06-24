#!/usr/bin/env python3
"""Train Stage II QbyT verifier with LibriPhrase parquet + dual loss."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dma_kws.config import load_config
from dma_kws.stage2.train import Stage2TrainArgs, run_stage2_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument(
        "--init-checkpoint",
        default="",
        help="Override stage2.init_checkpoint for partial encoder/QbyT weight init",
    )
    parser.add_argument(
        "--resume-checkpoint",
        default="",
        help="Override stage2.resume_checkpoint for full Lightning resume",
    )
    parser.add_argument(
        "--resume-from",
        default="",
        help='True Lightning resume of full training state from a checkpoint path or "last"',
    )
    parser.add_argument("--device", default="cuda", help="cuda or cpu")
    parser.add_argument("--devices", type=int, default=1, help="Number of visible GPUs to use")
    parser.add_argument("--limit-steps", type=int, default=0, help="Optional training step cap for smoke runs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    stage2 = config.get("stage2", {})
    resume_checkpoint = (
        args.resume_checkpoint
        or stage2.get("resume_checkpoint", "")
    )
    run_stage2_training(
        config,
        Stage2TrainArgs(
            init_checkpoint=args.init_checkpoint,
            resume_checkpoint=resume_checkpoint,
            resume_from=args.resume_from,
            device=args.device,
            devices=args.devices,
            limit_steps=args.limit_steps,
        ),
    )


if __name__ == "__main__":
    main()
