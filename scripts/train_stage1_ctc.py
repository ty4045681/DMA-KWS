#!/usr/bin/env python3
"""Train Stage I phoneme CTC with Wenet-aligned CharTokenizer targets."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dma_kws.config import load_config
from dma_kws.stage1.wenet_ctc import Stage1TrainArgs, run_stage1_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--train-manifest", default="", help="Override train manifest path")
    parser.add_argument("--dev-manifest", default="", help="Override dev manifest path")
    parser.add_argument("--device", default="cuda", help="cuda or cpu")
    parser.add_argument("--devices", type=int, default=1, help="Number of devices for the Lightning Trainer")
    parser.add_argument("--limit-steps", type=int, default=0, help="Optional training step cap for smoke runs")
    parser.add_argument(
        "--resume-from",
        default="",
        help='Resume full training state from a checkpoint path, or "last" to use <checkpoint_dir>/last.ckpt',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    run_stage1_training(
        config,
        Stage1TrainArgs(
            train_manifest=args.train_manifest,
            dev_manifest=args.dev_manifest,
            device=args.device,
            devices=args.devices,
            limit_steps=args.limit_steps,
            resume_from=args.resume_from,
        ),
    )


if __name__ == "__main__":
    main()
