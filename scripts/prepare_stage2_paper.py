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
from dma_kws.stage2.pairs import (
    clip_to_audio_rel,
    decoded_glob_for_dataset,
    infer_dataset_id,
    iter_decoded_audio_rows,
    resolve_data_root,
)
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
        help="Directory or file with decoded-parquet shards",
    )
    parser.add_argument(
        "--dataset-root",
        default="",
        help="Alias for --decoded-parquet-root (dataset-specific decoded shard root)",
    )
    parser.add_argument(
        "--output-subdir",
        default="",
        help="Subdirectory under paths.processed_root for paper parquet output "
        "(default: stage2_qbyt, or stage2.prep.output_subdir from config)",
    )
    parser.add_argument("--limit-anchors", type=int, default=0, help="Optional anchor cap for smoke runs")
    parser.add_argument("--seed", type=int, default=2025, help="Reserved for future deterministic sampling")
    return parser.parse_args()


def find_default_parquet(root: Path) -> Path:
    matches = sorted(root.rglob("*.parquet"))
    if not matches:
        raise SystemExit(f"No parquet files found under {root}; pass --input-parquet explicitly")
    return matches[0]


def find_decoded_parquets(root: Path, *, decoded_glob: str | None = None) -> list[Path]:
    if root.is_file():
        return [root]
    glob_pattern = decoded_glob or "LP-100-decoded-*.parquet"
    matches = sorted(root.rglob(glob_pattern))
    if not matches and decoded_glob:
        print(
            f"WARNING: No shards matching {glob_pattern!r} under {root}; "
            "falling back to *.parquet (may include unrelated files)"
        )
        matches = sorted(root.rglob("*.parquet"))
    elif not matches:
        matches = sorted(root.rglob("*.parquet"))
    if not matches:
        raise SystemExit(f"No decoded parquet shards found under {root}")
    return matches


def _read_parquet(path: Path):
    import pandas as pd

    return pd.read_parquet(path)


def iter_clip_audio_paths(df, *, limit_anchors: int = 0):
    count = 0
    for _, row in df.iterrows():
        clips = parse_clips(row.get("clips"))
        if not clips:
            continue
        for clip in clips:
            yield clip["audio_path"]
        count += 1
        if limit_anchors and count >= limit_anchors:
            break


def collect_needed_audio_keys(
    df,
    *,
    limit_anchors: int = 0,
    dataset_id: str | None = None,
) -> set[str]:
    needed: set[str] = set()
    count = 0
    for _, row in df.iterrows():
        clips = parse_clips(row.get("clips"))
        if not clips:
            continue
        for clip in clips:
            needed.add(clip_to_audio_rel(clip["audio_path"], dataset_id=dataset_id))
        count += 1
        if limit_anchors and count >= limit_anchors:
            break
    return needed


def resolve_decoded_root(
    args: argparse.Namespace,
    config: dict,
    paths: dict,
    dataset_id: str | None,
) -> Path:
    if args.decoded_parquet_root:
        return Path(args.decoded_parquet_root)
    if args.dataset_root:
        return Path(args.dataset_root)
    stage2_prep = (config.get("stage2") or {}).get("prep") or {}
    if stage2_prep.get("data_root"):
        return Path(stage2_prep["data_root"])
    if dataset_id:
        root = resolve_data_root(paths, dataset_id)
        if root is not None:
            return root
    return Path(paths["libriphrase100_root"])


def resolve_output_subdir(args: argparse.Namespace, config: dict) -> str:
    if args.output_subdir:
        return args.output_subdir
    stage2_prep = (config.get("stage2") or {}).get("prep") or {}
    return stage2_prep.get("output_subdir", "stage2_qbyt")


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

    output_subdir = resolve_output_subdir(args, config)
    output_dir = processed_root / output_subdir
    clips_dir = output_dir / "clips"
    distances_dir = output_dir / "distances"
    fbank_dir = feature_root / "fbank"
    output_parquet = output_dir / OUTPUT_PARQUET_NAME

    df = pd.read_parquet(input_path)
    dataset_id = infer_dataset_id(iter_clip_audio_paths(df, limit_anchors=args.limit_anchors))
    if dataset_id:
        print(f"Detected dataset: {dataset_id}")
    else:
        print("Detected dataset: unknown (using LP-100 defaults)")

    needed_keys = collect_needed_audio_keys(
        df,
        limit_anchors=args.limit_anchors,
        dataset_id=dataset_id,
    )

    decoded_root = resolve_decoded_root(args, config, paths, dataset_id)
    decoded_glob = decoded_glob_for_dataset(dataset_id) if dataset_id else "LP-100-decoded-*.parquet"
    print(f"Using decoded glob: {decoded_glob}")
    decoded_parquet_paths = find_decoded_parquets(decoded_root, decoded_glob=decoded_glob)
    print(f"Found {len(decoded_parquet_paths)} shards under {decoded_root}")

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
