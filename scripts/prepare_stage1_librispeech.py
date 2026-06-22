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
from dma_kws.stage1.librispeech import (
    ParquetAudioUtterance,
    iter_librispeech_parquet_utterances,
    iter_librispeech_utterances,
    strip_stress_marker,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--limit", type=int, default=0, help="Optional max utterances per split for smoke runs")
    parser.add_argument(
        "--input-format",
        choices=("librispeech-dir", "hf-parquet"),
        default="librispeech-dir",
        help="Input format for the training split. Default keeps the official LibriSpeech directory layout.",
    )
    parser.add_argument("--parquet-root", default="", help="Directory or file containing HuggingFace parquet shard(s)")
    parser.add_argument(
        "--parquet-split",
        default="",
        help="Split label to write into the training manifest for --input-format hf-parquet",
    )
    parser.add_argument(
        "--parquet-audio-dir",
        default="",
        help="Optional directory for audio extracted from parquet bytes",
    )
    parser.add_argument(
        "--dev-parquet-root",
        default="",
        help="Optional parquet file/directory for the dev split when --input-format hf-parquet is used",
    )
    parser.add_argument(
        "--dev-parquet-split",
        default="",
        help="Split label to write into dev.jsonl for --dev-parquet-root",
    )
    parser.add_argument(
        "--dev-parquet-audio-dir",
        default="",
        help="Optional directory for dev audio extracted from parquet bytes",
    )
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


def _parquet_utterance_audio_path(utt: ParquetAudioUtterance, audio_output_dir: Path) -> Path:
    if utt.audio_bytes is not None:
        speaker_id = utt.speaker_id or "_unknown_speaker"
        chapter_id = utt.chapter_id or "_unknown_chapter"
        extension = utt.audio_extension or ".flac"
        output_path = audio_output_dir / speaker_id / chapter_id / f"{utt.utt_id}{extension}"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(utt.audio_bytes)
        return output_path
    if utt.audio_path is not None:
        return utt.audio_path
    raise ValueError(
        f"Parquet utterance {utt.utt_id} has neither audio bytes nor a local audio path; "
        "download parquet shards with the audio column or convert them to local audio files first."
    )


def prepare_parquet_split(
    *,
    g2p,
    parquet_root: Path,
    split: str,
    output_path: Path,
    audio_output_dir: Path,
    limit: int,
) -> list[str]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    all_phones: list[str] = []
    count = 0
    with output_path.open("w", encoding="utf-8") as writer:
        for utt in iter_librispeech_parquet_utterances(parquet_root, split=split):
            wav_path = _parquet_utterance_audio_path(utt, audio_output_dir)
            phones = text_to_phonemes(g2p, utt.text)
            all_phones.extend(phones)
            record = {
                "utt_id": utt.utt_id,
                "split": utt.split,
                "wav_path": str(wav_path),
                "text": utt.text,
                "normalized_text": normalize_english_text(utt.text),
                "phonemes": phones,
            }
            writer.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
            if limit and count >= limit:
                break
    print(f"Wrote {count} parquet utterances to {output_path}")
    return all_phones


def has_librispeech_transcripts(librispeech_root: Path, splits: Iterable[str]) -> bool:
    for split in splits:
        if any((librispeech_root / split).rglob("*.trans.txt")):
            return True
    return False


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
    if args.input_format == "hf-parquet":
        if not args.parquet_root:
            raise SystemExit("--parquet-root is required when --input-format hf-parquet")
        parquet_split = args.parquet_split or (train_splits[0] if train_splits else "train-clean-360")
        parquet_audio_dir = Path(args.parquet_audio_dir) if args.parquet_audio_dir else output_dir / "audio" / parquet_split
        train_phones = prepare_parquet_split(
            g2p=g2p,
            parquet_root=Path(args.parquet_root),
            split=parquet_split,
            output_path=output_dir / "train.jsonl",
            audio_output_dir=parquet_audio_dir,
            limit=args.limit,
        )
    else:
        train_phones = prepare_split(
            g2p=g2p,
            librispeech_root=librispeech_root,
            splits=train_splits,
            output_path=output_dir / "train.jsonl",
            limit=args.limit,
        )
    if args.input_format == "hf-parquet":
        if args.dev_parquet_root:
            dev_parquet_split = args.dev_parquet_split or (dev_splits[0] if dev_splits else "dev-clean")
            dev_parquet_audio_dir = (
                Path(args.dev_parquet_audio_dir)
                if args.dev_parquet_audio_dir
                else output_dir / "audio" / dev_parquet_split
            )
            prepare_parquet_split(
                g2p=g2p,
                parquet_root=Path(args.dev_parquet_root),
                split=dev_parquet_split,
                output_path=output_dir / "dev.jsonl",
                audio_output_dir=dev_parquet_audio_dir,
                limit=args.limit,
            )
        elif has_librispeech_transcripts(librispeech_root, dev_splits):
            prepare_split(
                g2p=g2p,
                librispeech_root=librispeech_root,
                splits=dev_splits,
                output_path=output_dir / "dev.jsonl",
                limit=args.limit,
            )
        else:
            raise SystemExit(
                "No LibriSpeech dev transcripts found under paths.librispeech_root. "
                "Pass --dev-parquet-root when using --input-format hf-parquet in a parquet-only environment."
            )
    else:
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
