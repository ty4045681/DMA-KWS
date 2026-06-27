#!/usr/bin/env python3
"""Precompute LibriPhrase eval fbank .npy files for Stage II validation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dma_kws.config import fbank_kwargs, get_eval_fbank_config, load_config, require_sections
from dma_kws.stage2.dataset import resolve_stage2_eval_paths
from dma_kws.stage2.prepare_eval_fbank import prepare_eval_fbank, prepare_eval_fbank_from_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument(
        "--test-dir",
        default="",
        help="Override stage2.eval.test_dir (default: resolved from config)",
    )
    parser.add_argument(
        "--from-csv",
        action="store_true",
        help="Only convert wav files referenced by eval CSV anchor/comparison columns",
    )
    parser.add_argument("--limit", type=int, default=0, help="Optional max wav files to convert")
    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Recompute fbank even when the output .npy already exists",
    )
    parser.add_argument("--log-interval", type=int, default=1000, help="Progress print interval")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    require_sections(config, ["paths", "stage2"])

    eval_paths = resolve_stage2_eval_paths(config)
    test_dir = Path(args.test_dir) if args.test_dir else eval_paths["test_dir"]
    if not test_dir.exists():
        raise SystemExit(f"Eval test_dir not found: {test_dir}")

    skip_existing = not args.no_skip_existing
    fbank_params = fbank_kwargs(get_eval_fbank_config(config))

    if args.from_csv:
        written, skipped, failed = prepare_eval_fbank_from_csv(
            test_dir,
            csv_files=eval_paths["csv_files"],
            skip_existing=skip_existing,
            limit=args.limit,
            log_interval=args.log_interval,
            **fbank_params,
        )
    else:
        written, skipped, failed = prepare_eval_fbank(
            test_dir,
            skip_existing=skip_existing,
            limit=args.limit,
            log_interval=args.log_interval,
            **fbank_params,
        )

    print(
        f"Eval fbank prep complete under {test_dir}: "
        f"written={written}, skipped={skipped}, failed={failed}"
    )
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
