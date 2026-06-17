#!/usr/bin/env python3
"""Prepare LibriSpeech phoneme manifests for Stage I CTC training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from typing import Iterable

from dma_kws.config import load_config, require_sections
from dma_kws.phonemes import PhonemeVocabulary, normalize_english_text
from dma_kws.stage1.librispeech import iter_librispeech_utterances, strip_stress_marker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--limit", type=int, default=0, help="Optional max utterances per split for smoke runs")
    return parser.parse_args()


def make_g2p():
    try:
        from g2p_en import G2p
    except ImportError as exc:
        raise SystemExit("Missing dependency g2p_en. Install it with: pip install g2p_en") from exc
    return G2p()


def text_to_phonemes(g2p, text: str, *, strip_stress: bool = True) -> list[str]:
    normalized = normalize_english_text(text)
    phones = [phone for phone in g2p(normalized) if phone != " "]
    if strip_stress:
        phones = [strip_stress_marker(phone) for phone in phones]
    return [phone for phone in phones if phone]


def prepare_split(
    *,
    g2p,
    librispeech_root: Path,
    splits: Iterable[str],
    output_path: Path,
    limit: int,
) -> list[str]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    all_phones: list[str] = []
    count = 0
    with output_path.open("w", encoding="utf-8") as writer:
        for utt in iter_librispeech_utterances(librispeech_root, splits):
            phones = text_to_phonemes(g2p, utt.text)
            all_phones.extend(phones)
            record = {
                "utt_id": utt.utt_id,
                "split": utt.split,
                "wav_path": str(utt.audio_path),
                "text": utt.text,
                "normalized_text": normalize_english_text(utt.text),
                "phonemes": phones,
            }
            writer.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
            if limit and count >= limit:
                break
    print(f"Wrote {count} utterances to {output_path}")
    return all_phones


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    require_sections(config, ["paths", "stage1"])

    stage1 = config["stage1"]
    paths = config["paths"]
    librispeech_root = Path(paths["librispeech_root"])
    output_dir = Path(paths["processed_root"]) / "stage1_phoneme_ctc"

    train_splits = stage1.get("train_splits", ["train-clean-100"])
    dev_splits = stage1.get("dev_splits", ["dev-clean"])

    g2p = make_g2p()
    train_phones = prepare_split(
        g2p=g2p,
        librispeech_root=librispeech_root,
        splits=train_splits,
        output_path=output_dir / "train.jsonl",
        limit=args.limit,
    )
    prepare_split(
        g2p=g2p,
        librispeech_root=librispeech_root,
        splits=dev_splits,
        output_path=output_dir / "dev.jsonl",
        limit=args.limit,
    )

    vocab = PhonemeVocabulary.build(train_phones)
    vocab_path = output_dir / "phoneme_vocab.txt"
    vocab.write(vocab_path)
    print(f"Wrote vocab of size {len(vocab.token_to_id)} to {vocab_path}")


if __name__ == "__main__":
    main()
