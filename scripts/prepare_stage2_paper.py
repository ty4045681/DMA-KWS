#!/usr/bin/env python3
"""Prepare paper-format Stage II QbyT training data from LibriPhrase parquet."""

from __future__ import annotations

import argparse
import sys
from functools import partial
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from dma_kws.config import fbank_kwargs, get_fbank_config, load_config, require_sections
from dma_kws.stage2.pairs import clip_to_audio_rel, iter_decoded_audio_rows
from dma_kws.stage2.prepare_paper import (
    OUTPUT_PARQUET_NAME,
    compute_fbank_for_clip,
    convert_aggregated_to_paper_parquet,
    parse_clips,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--input-parquet", default="", help="Aggregated LibriPhrase parquet")
    parser.add_argument(
        "--decoded-parquet-root",
        default="",
        help="Directory or file with LP-100-decoded-*.parquet shards",
    )
    parser.add_argument("--limit-anchors", type=int, default=0, help="Optional anchor cap for smoke runs")
    parser.add_argument("--seed", type=int, default=2025, help="Reserved for future deterministic sampling")
    return parser.parse_args()


def find_default_parquet(root: Path) -> Path:
    matches = sorted(root.rglob("*.parquet"))
    if not matches:
        raise SystemExit(f"No parquet files found under {root}; pass --input-parquet explicitly")
    return matches[0]


def find_decoded_parquets(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    matches = sorted(root.rglob("LP-100-decoded-*.parquet"))
    if not matches:
        matches = sorted(root.rglob("*.parquet"))
    if not matches:
        raise SystemExit(f"No decoded parquet shards found under {root}")
    return matches


def _read_parquet(path: Path):
    import pandas as pd

    return pd.read_parquet(path)


def collect_needed_audio_keys(df, *, limit_anchors: int = 0) -> set[str]:
    needed: set[str] = set()
    count = 0
    for _, row in df.iterrows():
        clips = parse_clips(row.get("clips"))
        if not clips:
            continue
        for clip in clips:
            needed.add(clip_to_audio_rel(clip["audio_path"]))
        count += 1
        if limit_anchors and count >= limit_anchors:
            break
    return needed


def load_decoded_audio(
    decoded_parquet_paths: list[Path],
    needed_keys: set[str],
    *,
    read_parquet=_read_parquet,
) -> dict[str, tuple[np.ndarray, int]]:
    audio_by_rel: dict[str, tuple[np.ndarray, int]] = {}
    for audio_rel, audio, sample_rate in iter_decoded_audio_rows(
        decoded_parquet_paths,
        needed_keys,
        read_parquet=read_parquet,
    ):
        array = np.asarray(audio, dtype=np.float32)
        if array.ndim != 1:
            raise SystemExit(
                f"Expected mono 1-D audio for {audio_rel}, got shape {array.shape}"
            )
        audio_by_rel[audio_rel] = (array, int(sample_rate))
    return audio_by_rel


def main() -> None:
    args = parse_args()
    try:
        import pandas as pd
    except ImportError as exc:
        raise SystemExit("Missing dependency pandas/pyarrow. Install with: pip install pandas pyarrow") from exc

    _ = args.seed  # reserved
    config = load_config(args.config)
    require_sections(config, ["paths"])
    paths = config["paths"]

    input_path = Path(args.input_parquet) if args.input_parquet else find_default_parquet(Path(paths["libriphrase100_root"]))
    processed_root = Path(paths["processed_root"])
    feature_root = Path(paths.get("feature_root", processed_root.parent / "features"))

    output_dir = processed_root / "stage2_qbyt"
    clips_dir = output_dir / "clips"
    distances_dir = output_dir / "distances"
    fbank_dir = feature_root / "fbank"
    output_parquet = output_dir / OUTPUT_PARQUET_NAME

    df = pd.read_parquet(input_path)
    needed_keys = collect_needed_audio_keys(df, limit_anchors=args.limit_anchors)

    libriphrase_root = Path(paths["libriphrase100_root"])
    decoded_root = Path(args.decoded_parquet_root) if args.decoded_parquet_root else libriphrase_root
    decoded_parquet_paths = find_decoded_parquets(decoded_root)
    audio_by_rel = load_decoded_audio(decoded_parquet_paths, needed_keys)

    fbank_cfg = get_fbank_config(config)
    compute_fbank_fn = partial(compute_fbank_for_clip, **fbank_kwargs(fbank_cfg))

    paper_df, stats = convert_aggregated_to_paper_parquet(
        df,
        clips_dir=clips_dir,
        distances_dir=distances_dir,
        fbank_dir=fbank_dir,
        audio_by_rel=audio_by_rel,
        limit_anchors=args.limit_anchors,
        compute_fbank=compute_fbank_fn,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    paper_df.to_parquet(output_parquet, index=False)

    missing = needed_keys - set(audio_by_rel)
    print(f"Read {stats['anchors']} anchors from {input_path}")
    print(f"Wrote paper parquet to {output_parquet}")
    print(f"Wrote clips npy under {clips_dir}")
    print(f"Wrote distances npy under {distances_dir}")
    print(f"Wrote {stats['fbank_written']} fbank npy files under {fbank_dir}")
    if missing:
        print(
            f"WARNING: {len(missing)} referenced clips were not found in decoded parquet "
            f"({stats['missing_audio']} fbank files skipped)"
        )


if __name__ == "__main__":
    main()
