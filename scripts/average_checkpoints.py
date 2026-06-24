#!/usr/bin/env python3
"""Average the last K checkpoints from a training run."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dma_kws.training.checkpoint_avg import average_lightning_checkpoints


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path, help="Directory containing checkpoints")
    parser.add_argument(
        "--pattern",
        default="*.ckpt",
        help='Glob pattern for checkpoint files (default: "*.ckpt")',
    )
    parser.add_argument(
        "--last-k",
        type=int,
        default=10,
        help="Number of most recent checkpoints to average (default: 10)",
    )
    parser.add_argument("--output", required=True, type=Path, help="Output checkpoint path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.last_k <= 0:
        raise SystemExit("--last-k must be positive")

    input_dir = args.input_dir
    if not input_dir.is_dir():
        raise SystemExit(f"Input directory not found: {input_dir}")

    candidates = sorted(input_dir.glob(args.pattern))
    if not candidates:
        raise SystemExit(f"No checkpoints matched pattern {args.pattern!r} in {input_dir}")

    selected = candidates[-args.last_k :]
    output_path = average_lightning_checkpoints(selected, args.output)
    print(f"Averaged {len(selected)} checkpoints -> {output_path}")


if __name__ == "__main__":
    main()
