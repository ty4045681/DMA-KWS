#!/usr/bin/env python3
"""Prepare Stage II QbyT positive/negative pairs from LibriPhrase-style parquet."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dma_kws.config import load_config, require_sections
from dma_kws.phonemes import normalize_english_text
from dma_kws.stage1.librispeech import strip_stress_marker
from dma_kws.stage2.pairs import AnchorExample, make_pair_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--input-parquet", default="", help="Input parquet; defaults to first parquet under libriphrase100_root")
    parser.add_argument("--limit-anchors", type=int, default=0, help="Optional anchor cap for smoke runs")
    parser.add_argument("--negatives-per-anchor", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2025)
    return parser.parse_args()


def make_g2p():
    try:
        from g2p_en import G2p
    except ImportError as exc:
        raise SystemExit("Missing dependency g2p_en. Install it with: pip install g2p_en") from exc
    return G2p()


def text_to_phonemes(g2p, text: str) -> list[str]:
    normalized = normalize_english_text(text)
    return clean_phoneme_tokens(g2p(normalized))


def clean_phoneme_tokens(tokens) -> list[str]:
    phonemes: list[str] = []
    for phone in tokens:
        if phone == " ":
            continue
        cleaned = strip_stress_marker(str(phone)).strip()
        if cleaned:
            phonemes.append(cleaned)
    return phonemes


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


def main() -> None:
    args = parse_args()
    try:
        import pandas as pd
    except ImportError as exc:
        raise SystemExit("Missing dependency pandas/pyarrow. Install with: pip install pandas pyarrow") from exc

    config = load_config(args.config)
    require_sections(config, ["paths"])
    paths = config["paths"]
    input_path = Path(args.input_parquet) if args.input_parquet else find_default_parquet(Path(paths["libriphrase100_root"]))
    output_dir = Path(paths["processed_root"]) / "stage2_qbyt"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "train.jsonl"

    df = pd.read_parquet(input_path)
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
        if args.limit_anchors and len(anchors) >= args.limit_anchors:
            break

    pairs = make_pair_records(anchors, negatives_per_anchor=args.negatives_per_anchor, seed=args.seed)
    with output_path.open("w", encoding="utf-8") as writer:
        for pair in pairs:
            writer.write(json.dumps(pair.to_json_dict(), ensure_ascii=False) + "\n")

    print(f"Read {len(anchors)} anchors from {input_path}")
    print(f"Wrote {len(pairs)} pairs to {output_path}")


if __name__ == "__main__":
    main()
