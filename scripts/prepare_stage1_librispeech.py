#!/usr/bin/env python3
"""Prepare LibriSpeech phoneme manifests for Stage I CTC training."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import hydra
from omegaconf import DictConfig, OmegaConf

from dma_kws.config import require_sections
from dma_kws.g2p import make_g2p, text_to_phonemes
from dma_kws.hydra_app import CONFIG_DIR, resolved_config
from dma_kws.phonemes import normalize_english_text
from dma_kws.stage1.librispeech import (
    ParquetAudioUtterance,
    iter_librispeech_parquet_utterances,
    iter_librispeech_utterances,
)
from dma_kws.stage1.wenet_ctc import phonemes_to_g2p_string


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
                "phonemes_g2p": phonemes_to_g2p_string(phones),
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
                "phonemes_g2p": phonemes_to_g2p_string(phones),
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


def run_prepare_stage1_librispeech(config: dict, prep: dict) -> None:
    """Prepare Stage I manifests from config and prep option overrides."""
    require_sections(config, ["paths", "stage1"])

    stage1 = config["stage1"]
    paths = config["paths"]
    librispeech_root = Path(paths["librispeech_root"])
    output_dir = Path(paths["processed_root"]) / "stage1_phoneme_ctc"
    train_splits = stage1.get("train_splits", ["train-clean-100"])
    dev_splits = stage1.get("dev_splits", ["dev-clean"])
    limit = int(prep.get("limit", 0))
    input_format = str(prep.get("input_format", "librispeech-dir"))

    g2p = make_g2p()
    if input_format == "hf-parquet":
        parquet_root = str(prep.get("parquet_root", ""))
        if not parquet_root:
            raise SystemExit("--parquet-root is required when --input-format hf-parquet")
        parquet_split = str(prep.get("parquet_split", "")) or (train_splits[0] if train_splits else "train-clean-360")
        parquet_audio_dir_value = str(prep.get("parquet_audio_dir", ""))
        parquet_audio_dir = Path(parquet_audio_dir_value) if parquet_audio_dir_value else output_dir / "audio" / parquet_split
        prepare_parquet_split(
            g2p=g2p,
            parquet_root=Path(parquet_root),
            split=parquet_split,
            output_path=output_dir / "train.jsonl",
            audio_output_dir=parquet_audio_dir,
            limit=limit,
        )
    else:
        prepare_split(
            g2p=g2p,
            librispeech_root=librispeech_root,
            splits=train_splits,
            output_path=output_dir / "train.jsonl",
            limit=limit,
        )
    if input_format == "hf-parquet":
        dev_parquet_root = str(prep.get("dev_parquet_root", ""))
        if dev_parquet_root:
            dev_parquet_split = str(prep.get("dev_parquet_split", "")) or (dev_splits[0] if dev_splits else "dev-clean")
            dev_parquet_audio_dir_value = str(prep.get("dev_parquet_audio_dir", ""))
            dev_parquet_audio_dir = (
                Path(dev_parquet_audio_dir_value)
                if dev_parquet_audio_dir_value
                else output_dir / "audio" / dev_parquet_split
            )
            prepare_parquet_split(
                g2p=g2p,
                parquet_root=Path(dev_parquet_root),
                split=dev_parquet_split,
                output_path=output_dir / "dev.jsonl",
                audio_output_dir=dev_parquet_audio_dir,
                limit=limit,
            )
        elif has_librispeech_transcripts(librispeech_root, dev_splits):
            prepare_split(
                g2p=g2p,
                librispeech_root=librispeech_root,
                splits=dev_splits,
                output_path=output_dir / "dev.jsonl",
                limit=limit,
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
            limit=limit,
        )

    print(
        "Stage I targets use Wenet CharTokenizer dict from config tokenizer.dict_path "
        f"(e.g. data/dict/lang_char.txt). Manifests include phonemes_g2p for training."
    )


@hydra.main(version_base=None, config_path=str(CONFIG_DIR), config_name="config")
def main(cfg: DictConfig) -> None:
    prep = OmegaConf.to_container(cfg.prep, resolve=True)
    if not isinstance(prep, dict):
        prep = {}
    run_prepare_stage1_librispeech(resolved_config(cfg), prep)


if __name__ == "__main__":
    main()
