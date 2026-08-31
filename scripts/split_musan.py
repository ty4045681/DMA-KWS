#!/usr/bin/env python3
"""Split a complete MUSAN tree into disjoint train and evaluation allowlists."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dma_kws.data_prep.musan_split import (
    DEFAULT_MUSAN_SPLIT_SEED,
    DEFAULT_MUSAN_TRAIN_RATIO,
    build_musan_split,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--musan-root",
        type=Path,
        required=True,
        help="Complete MUSAN root containing music/, noise/ and speech/",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="New directory for train_background.list, eval_musan.list and split.json",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=DEFAULT_MUSAN_TRAIN_RATIO,
        help=(
            "Target train duration inside every category/source stratum "
            f"(default: {DEFAULT_MUSAN_TRAIN_RATIO:g})"
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_MUSAN_SPLIT_SEED,
        help=f"Deterministic split seed (default: {DEFAULT_MUSAN_SPLIT_SEED})",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> dict:
    args = build_parser().parse_args(argv)
    summary = build_musan_split(
        args.musan_root,
        args.output_dir,
        train_ratio=args.train_ratio,
        seed=args.seed,
    )
    brief_keys = (
        "list",
        "recordings",
        "groups",
        "duration_hours",
        "catalog_sha256",
    )
    concise = {
        "output_dir": str(args.output_dir.expanduser().resolve()),
        "train": {
            key: summary["splits"]["train"][key] for key in brief_keys
        },
        "eval": {key: summary["splits"]["eval"][key] for key in brief_keys},
        "train_ratio_target": summary["train_ratio_target"],
        "train_ratio_actual": summary["train_ratio_actual"],
    }
    print(json.dumps(concise, ensure_ascii=False, indent=2, sort_keys=True))
    return summary


if __name__ == "__main__":
    main()
