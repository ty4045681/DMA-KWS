#!/usr/bin/env python3
"""Prepare Stage II QbyT positive/negative pairs from LibriPhrase-style parquet."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from dma_kws.config import load_config, require_sections
from dma_kws.g2p import clean_phoneme_tokens, make_g2p, text_to_phonemes
from dma_kws.stage2.pairs import (
    AnchorExample,
    PairRecord,
    clip_to_audio_rel,
    iter_decoded_audio_rows,
    make_pair_records,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--input-parquet", default="", help="Input parquet; defaults to first parquet under libriphrase100_root")
    parser.add_argument("--limit-anchors", type=int, default=0, help="Optional anchor cap for smoke runs")
    parser.add_argument("--negatives-per-anchor", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument(
        "--decoded-parquet-root",
        default="",
        help="Directory or file with LP-100-decoded-*.parquet shards; defaults to libriphrase100_root",
    )
    parser.add_argument(
        "--eval-parquet",
        default="",
        help="Optional separate parquet for the dev split; when given, dev pairs are built from it "
        "instead of carving a hold-out from the training anchors",
    )
    parser.add_argument(
        "--holdout-anchor-fraction",
        type=float,
        default=None,
        help="Fraction of anchors held out for dev when --eval-parquet is not given; "
        "overrides config stage2.dev.holdout_anchor_fraction (default 0.1)",
    )
    return parser.parse_args()


def parse_clips(raw_clips: Any) -> list[str]:
    if isinstance(raw_clips, str):
        try:
            raw_clips = json.loads(raw_clips)
        except json.JSONDecodeError:
            return [raw_clips]
    clips: list[str] = []
    if isinstance(raw_clips, list):
        for item in raw_clips:
            if isinstance(item, dict) and "audio_path" in item:
                clips.append(str(item["audio_path"]))
            elif isinstance(item, str):
                clips.append(item)
    return clips


def find_default_parquet(root: Path) -> Path:
    matches = sorted(root.rglob("*.parquet"))
    if not matches:
        raise SystemExit(f"No parquet files found under {root}; pass --input-parquet explicitly")
    return matches[0]


def _read_parquet(path: Path):
    import pandas as pd

    return pd.read_parquet(path)


def find_decoded_parquets(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    matches = sorted(root.rglob("LP-100-decoded-*.parquet"))
    if not matches:
        matches = sorted(root.rglob("*.parquet"))
    if not matches:
        raise SystemExit(f"No decoded parquet shards found under {root}")
    return matches


def materialize_pairs(
    pairs: list[PairRecord],
    *,
    decoded_parquet_paths: list[Path],
    audio_dir: Path,
    read_parquet=_read_parquet,
) -> tuple[list[PairRecord], int]:
    """Extract referenced clips to .npy and rewrite each pair's wav_path.

    Returns (rewritten_pairs, unmatched_count). Pairs whose audio could not
    be found in any decoded shard are dropped from the returned list.
    """
    needed = {clip_to_audio_rel(pair.wav_path) for pair in pairs}
    audio_dir.mkdir(parents=True, exist_ok=True)
    rel_for_key: dict[str, str] = {}
    for audio_rel, audio, _sr in iter_decoded_audio_rows(
        decoded_parquet_paths, needed, read_parquet=read_parquet
    ):
        out_path = audio_dir / f"{audio_rel}.npy"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        array = np.asarray(audio, dtype=np.float32)
        if array.ndim != 1:
            raise SystemExit(
                f"Expected mono 1-D audio for {audio_rel}, got shape {array.shape}"
            )
        np.save(out_path, array)
        # wav_path is relative to audio_dir.parent (the stage2_qbyt dir)
        rel_for_key[audio_rel] = f"{audio_dir.name}/{audio_rel}.npy"

    rewritten: list[PairRecord] = []
    unmatched = 0
    for pair in pairs:
        key = clip_to_audio_rel(pair.wav_path)
        rel = rel_for_key.get(key)
        if rel is None:
            unmatched += 1
            continue
        rewritten.append(
            PairRecord(
                anchor_text=pair.anchor_text,
                anchor_phonemes=pair.anchor_phonemes,
                wav_path=rel,
                label=pair.label,
                sample_rate=pair.sample_rate,
            )
        )
    return rewritten, unmatched


def read_anchors(df, *, limit_anchors: int = 0) -> list[AnchorExample]:
    """Build AnchorExample rows from a LibriPhrase parquet frame."""
    required = {"ngram", "clips"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise SystemExit(f"Input parquet missing required columns {missing}. Available columns: {list(df.columns)}")

    g2p = None if "ngram_g2p" in df.columns else make_g2p()
    anchors: list[AnchorExample] = []
    for _, row in df.iterrows():
        text = str(row["ngram"])
        if "ngram_g2p" in df.columns and row.get("ngram_g2p"):
            phonemes = clean_phoneme_tokens(str(row["ngram_g2p"]).split())
        else:
            phonemes = text_to_phonemes(g2p, text)
        clips = parse_clips(row["clips"])
        if clips and phonemes:
            anchors.append(AnchorExample(text=text, phonemes=phonemes, clips=clips))
        if limit_anchors and len(anchors) >= limit_anchors:
            break
    return anchors


def split_anchors(
    anchors: list[AnchorExample],
    *,
    holdout_fraction: float,
    seed: int,
) -> tuple[list[AnchorExample], list[AnchorExample]]:
    """Split anchors into (train, dev) by anchor so no phrase leaks across splits.

    Returns an empty dev list when there are too few anchors to hold any out.
    """
    if holdout_fraction <= 0.0:
        return list(anchors), []
    shuffled = list(anchors)
    random.Random(seed).shuffle(shuffled)
    holdout_count = int(len(shuffled) * holdout_fraction)
    if holdout_count <= 0:
        return shuffled, []
    dev_anchors = shuffled[:holdout_count]
    train_anchors = shuffled[holdout_count:]
    return train_anchors, dev_anchors


def build_split(
    anchors: list[AnchorExample],
    output_path: Path,
    *,
    negatives_per_anchor: int,
    seed: int,
    decoded_parquet_paths: list[Path],
    audio_dir: Path,
) -> tuple[int, int]:
    """Generate pairs for ``anchors``, materialize audio, and write ``output_path``.

    Returns (written_pairs, unmatched_pairs).
    """
    pairs = make_pair_records(
        anchors,
        negatives_per_anchor=negatives_per_anchor,
        seed=seed,
    )
    rewritten, unmatched = materialize_pairs(
        pairs,
        decoded_parquet_paths=decoded_parquet_paths,
        audio_dir=audio_dir,
    )
    with output_path.open("w", encoding="utf-8") as writer:
        for pair in rewritten:
            writer.write(json.dumps(pair.to_json_dict(), ensure_ascii=False) + "\n")
    return len(rewritten), unmatched


def main() -> None:
    args = parse_args()
    try:
        import pandas as pd
    except ImportError as exc:
        raise SystemExit("Missing dependency pandas/pyarrow. Install with: pip install pandas pyarrow") from exc

    config = load_config(args.config)
    require_sections(config, ["paths"])
    paths = config["paths"]
    dev_config = config.get("stage2", {}).get("dev", {})
    holdout_fraction = (
        args.holdout_anchor_fraction
        if args.holdout_anchor_fraction is not None
        else float(dev_config.get("holdout_anchor_fraction", 0.1))
    )
    dev_seed = int(dev_config.get("seed", 2025))

    input_path = Path(args.input_parquet) if args.input_parquet else find_default_parquet(Path(paths["libriphrase100_root"]))
    output_dir = Path(paths["processed_root"]) / "stage2_qbyt"
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train.jsonl"
    dev_path = output_dir / "dev.jsonl"

    df = pd.read_parquet(input_path)
    anchors = read_anchors(df, limit_anchors=args.limit_anchors)

    libriphrase_root = Path(paths["libriphrase100_root"])
    decoded_root = Path(args.decoded_parquet_root) if args.decoded_parquet_root else libriphrase_root
    decoded_parquet_paths = find_decoded_parquets(decoded_root)
    audio_dir = output_dir / "audio"

    if args.eval_parquet:
        # Dev pairs come from a separate parquet; train uses all main-input anchors.
        train_anchors = anchors
        eval_df = pd.read_parquet(Path(args.eval_parquet))
        dev_anchors = read_anchors(eval_df)
        print(f"Read {len(anchors)} anchors from {input_path}")
        print(f"Read {len(dev_anchors)} dev anchors from {args.eval_parquet}")
    else:
        # Carve a hold-out split BY ANCHOR before pairing so no phrase leaks across splits.
        train_anchors, dev_anchors = split_anchors(
            anchors, holdout_fraction=holdout_fraction, seed=dev_seed
        )
        print(f"Read {len(anchors)} anchors from {input_path}")
        if dev_anchors:
            print(
                f"Held out {len(dev_anchors)} dev anchors "
                f"(fraction {holdout_fraction}, seed {dev_seed}); {len(train_anchors)} train anchors"
            )
        else:
            print(
                f"NOTE: too few anchors ({len(anchors)}) to hold out fraction {holdout_fraction}; "
                f"no dev.jsonl will be produced"
            )

    train_written, train_unmatched = build_split(
        train_anchors,
        train_path,
        negatives_per_anchor=args.negatives_per_anchor,
        seed=args.seed,
        decoded_parquet_paths=decoded_parquet_paths,
        audio_dir=audio_dir,
    )
    print(f"Materialized {train_written} train pairs to {train_path} (audio under {audio_dir})")
    if train_unmatched:
        print(f"WARNING: {train_unmatched} train pairs had clips not found in decoded parquet and were skipped")

    if dev_anchors:
        dev_written, dev_unmatched = build_split(
            dev_anchors,
            dev_path,
            negatives_per_anchor=args.negatives_per_anchor,
            seed=args.seed,
            decoded_parquet_paths=decoded_parquet_paths,
            audio_dir=audio_dir,
        )
        print(f"Materialized {dev_written} dev pairs to {dev_path} (audio under {audio_dir})")
        if dev_unmatched:
            print(f"WARNING: {dev_unmatched} dev pairs had clips not found in decoded parquet and were skipped")
    elif dev_path.exists():
        # Avoid a stale dev split from a prior run shadowing the "no dev" decision.
        dev_path.unlink()


if __name__ == "__main__":
    main()
