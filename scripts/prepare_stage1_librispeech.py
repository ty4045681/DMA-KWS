#!/usr/bin/env python3
"""Prepare LibriSpeech phoneme manifests for Stage I CTC training."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import json
from pathlib import Path
import tempfile
from typing import Iterable

import hydra
from omegaconf import DictConfig, OmegaConf

from dma_kws.config import require_sections
from dma_kws.g2p import make_g2p, text_to_phonemes
from dma_kws.hydra_app import CONFIG_DIR, resolved_config
from dma_kws.phonemes import normalize_english_text
from dma_kws.stage1.librispeech import (
    ParquetAudioUtterance,
    iter_librispeech_parquet_shard_utterances,
    iter_librispeech_parquet_utterances,
    iter_librispeech_utterances,
    list_librispeech_parquet_shards,
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


@dataclass(frozen=True)
class _ParquetShardTask:
    shard_path: Path
    split: str
    audio_output_dir: Path
    tmp_jsonl_path: Path


@dataclass(frozen=True)
class _ParquetShardResult:
    shard_path: Path
    tmp_jsonl_path: Path
    count: int


def _resolve_num_workers(value: int) -> int:
    if value < 0:
        raise ValueError(f"num_workers must be >= 0, got {value}")
    if value == 0:
        import os

        return max(1, min(8, os.cpu_count() or 1))
    return value


def _prepare_parquet_shard(task: _ParquetShardTask) -> _ParquetShardResult:
    g2p = make_g2p()
    count = 0
    task.tmp_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with task.tmp_jsonl_path.open("w", encoding="utf-8") as writer:
        for utt in iter_librispeech_parquet_shard_utterances(task.shard_path, split=task.split):
            wav_path = _parquet_utterance_audio_path(utt, task.audio_output_dir)
            phones = text_to_phonemes(g2p, utt.text)
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
    return _ParquetShardResult(shard_path=task.shard_path, tmp_jsonl_path=task.tmp_jsonl_path, count=count)


def prepare_parquet_split(
    *,
    g2p,
    parquet_root: Path,
    split: str,
    output_path: Path,
    audio_output_dir: Path,
    limit: int,
    num_workers: int = 1,
) -> list[str]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shard_paths = list_librispeech_parquet_shards(parquet_root)
    if num_workers <= 1 or len(shard_paths) <= 1:
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

    all_phones: list[str] = []
    count = 0
    with tempfile.TemporaryDirectory(prefix="stage1_librispeech_shards_") as tmp_dir:
        tmp_root = Path(tmp_dir)
        tasks = [
            _ParquetShardTask(
                shard_path=shard_path,
                split=split,
                audio_output_dir=audio_output_dir,
                tmp_jsonl_path=tmp_root / f"{shard_path.stem}.jsonl",
            )
            for shard_path in shard_paths
        ]

        results: dict[Path, _ParquetShardResult] = {}
        worker_count = min(num_workers, len(tasks))
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            future_to_task = {executor.submit(_prepare_parquet_shard, task): task for task in tasks}
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    result = future.result()
                except Exception as exc:
                    raise RuntimeError(f"Failed to process parquet shard: {task.shard_path}") from exc
                results[result.shard_path] = result

        with output_path.open("w", encoding="utf-8") as writer:
            for shard_path in shard_paths:
                result = results[shard_path]
                with result.tmp_jsonl_path.open("r", encoding="utf-8") as reader:
                    for line in reader:
                        if limit and count >= limit:
                            break
                        writer.write(line)
                        record = json.loads(line)
                        phonemes = record.get("phonemes")
                        if isinstance(phonemes, list):
                            all_phones.extend(str(phone) for phone in phonemes)
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
    num_workers = _resolve_num_workers(int(prep.get("num_workers", 0)))
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
            num_workers=num_workers,
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
                num_workers=num_workers,
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
