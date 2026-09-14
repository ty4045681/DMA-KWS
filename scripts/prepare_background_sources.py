#!/usr/bin/env python3
"""Prepare MUSAN, DNS, and FSD50K background recording catalogs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dma_kws.data_prep.background_sources import (
    prepare_background_sources,
    prepare_config_from_mapping,
)


DESCRIPTION = """Prepare MUSAN/DNS/FSD50K background source catalogs.

Reads a YAML --config (yaml.safe_load). Does not download data and does not
scan directories other than each source root / metadata path.

MUSAN expected layout:
  <root>/music/ ...
  <root>/noise/ ...
  <root>/speech/ ...
  optional split_dir with train_background.list, eval_musan.list, split.json.
  Old eval members are mapped to test; val is carved from remaining train groups.

DNS expected layout:
  Walks only the configured noise root for audio files (.wav/.flac/.mp3/.m4a).
  Does not scan sibling clean or speech trees. Directory presence is not
  eligibility; provide eligible_ids_file. Optional metadata CSV may supply
  relative_path, group_id, origin_id.

FSD50K expected layout:
  <root>/FSD50K.dev_audio/<fname>.wav
  <root>/FSD50K.eval_audio/<fname>.wav
  Ground truth under <root>/FSD50K.ground_truth or <metadata>:
    dev.csv  columns: fname, labels, split
    eval.csv columns: fname, labels
  Clip metadata under <metadata> or <root>/FSD50K.metadata:
    dev_clips.csv  columns: fname, username, license
    eval_clips.csv columns: fname, username, license
  Official eval is mapped to test and is never reshuffled into train.
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=DESCRIPTION,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="YAML prepare config (see configs/background_sources/example.yaml)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> dict:
    args = build_parser().parse_args(argv)
    config_path = args.config.expanduser()
    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"prepare config must be a mapping: {config_path}")
    config = prepare_config_from_mapping(payload, base_dir=config_path.parent)
    result = prepare_background_sources(config)
    summary = {
        "output_dir": str(result.output_dir),
        "provenance_complete": result.audit.provenance_complete,
        "capability_limit": result.audit.capability_limit,
        "sources": {
            source_id: dict(catalog.stats)
            for source_id, catalog in result.catalogs.items()
        },
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return summary


if __name__ == "__main__":
    main()
