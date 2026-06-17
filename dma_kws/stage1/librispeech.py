"""LibriSpeech helpers for Stage I phoneme CTC preparation."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from typing import Any, Iterable, Iterator

_STRESS_RE = re.compile(r"\d+$")


@dataclass(frozen=True)
class TranscriptUtterance:
    """One LibriSpeech utterance and its transcript."""

    utt_id: str
    text: str
    audio_path: Path
    split: str


@dataclass(frozen=True)
class ParquetAudioUtterance:
    """One HuggingFace LibriSpeech parquet utterance and its audio payload."""

    utt_id: str
    text: str
    split: str
    speaker_id: str
    chapter_id: str
    audio_bytes: bytes | None
    audio_path: Path | None
    audio_extension: str = ".flac"


def strip_stress_marker(phoneme: str) -> str:
    """Remove ARPAbet stress digits, e.g. AH0 -> AH."""
    return _STRESS_RE.sub("", phoneme)


def _find_audio_path(transcript_path: Path, utt_id: str) -> Path:
    for suffix in (".flac", ".wav"):
        candidate = transcript_path.parent / f"{utt_id}{suffix}"
        if candidate.exists():
            return candidate
    return transcript_path.parent / f"{utt_id}.flac"


def iter_librispeech_utterances(librispeech_root: str | Path, splits: Iterable[str]) -> Iterator[TranscriptUtterance]:
    """Yield utterances from LibriSpeech `.trans.txt` files for selected splits."""
    root = Path(librispeech_root)
    for split in splits:
        split_root = root / split
        for transcript_path in sorted(split_root.rglob("*.trans.txt")):
            with transcript_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    stripped = line.strip()
                    if not stripped:
                        continue
                    utt_id, text = stripped.split(maxsplit=1)
                    yield TranscriptUtterance(
                        utt_id=utt_id,
                        text=text,
                        audio_path=_find_audio_path(transcript_path, utt_id),
                        split=split,
                    )


def _iter_parquet_records(parquet_path: Path) -> Iterator[dict[str, Any]]:
    """Yield dict records from one parquet shard.

    Kept as a small helper so tests can monkeypatch parquet reading without
    requiring pandas/pyarrow in lightweight local environments.
    """
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError("Missing dependency pandas/pyarrow. Install requirements.txt before reading parquet data.") from exc

    try:
        dataframe = pd.read_parquet(parquet_path)
    except Exception as exc:  # pragma: no cover - exact pandas/pyarrow errors vary
        raise RuntimeError(f"Failed to read parquet shard: {parquet_path}") from exc
    yield from dataframe.to_dict(orient="records")


def _is_missing(value: Any) -> bool:
    return value is None or (isinstance(value, float) and value != value)


def _string_or_empty(value: Any) -> str:
    if _is_missing(value):
        return ""
    if isinstance(value, Integral):
        return str(int(value))
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _coerce_audio_bytes(value: Any) -> bytes | None:
    if _is_missing(value):
        return None
    if isinstance(value, bytes):
        return value
    if isinstance(value, bytearray):
        return bytes(value)
    if isinstance(value, memoryview):
        return value.tobytes()
    return None


def _infer_speaker_chapter(record: Mapping[str, Any], utt_id: str) -> tuple[str, str]:
    speaker_id = _string_or_empty(record.get("speaker_id"))
    chapter_id = _string_or_empty(record.get("chapter_id"))
    parts = utt_id.split("-")
    if not speaker_id and len(parts) >= 1:
        speaker_id = parts[0]
    if not chapter_id and len(parts) >= 2:
        chapter_id = parts[1]
    return speaker_id, chapter_id


def _resolve_existing_audio_path(source_path: str, parquet_path: Path) -> Path | None:
    if not source_path:
        return None
    raw_path = Path(source_path)
    if raw_path.exists():
        return raw_path
    for base in (parquet_path.parent, parquet_path.parent.parent, parquet_path.parent.parent.parent):
        candidate = base / raw_path
        if candidate.exists():
            return candidate
    return None


def _parquet_record_to_utterance(
    record: Mapping[str, Any],
    *,
    split: str,
    parquet_path: Path,
) -> ParquetAudioUtterance:
    text = _string_or_empty(record.get("text")).strip()
    audio = record.get("audio")

    audio_bytes: bytes | None = None
    audio_path_text = ""
    if isinstance(audio, Mapping):
        audio_bytes = _coerce_audio_bytes(audio.get("bytes"))
        audio_path_text = _string_or_empty(audio.get("path"))
    else:
        audio_bytes = _coerce_audio_bytes(audio)

    file_path_text = _string_or_empty(record.get("file"))
    source_path_text = audio_path_text or file_path_text
    utt_id = _string_or_empty(record.get("id")).strip()
    if not utt_id and source_path_text:
        utt_id = Path(source_path_text).stem
    if not utt_id:
        raise ValueError(f"Missing utterance id in parquet record from {parquet_path}")
    if not text:
        raise ValueError(f"Missing transcript text for utterance {utt_id} in {parquet_path}")

    speaker_id, chapter_id = _infer_speaker_chapter(record, utt_id)
    audio_extension = Path(source_path_text).suffix or ".flac"
    return ParquetAudioUtterance(
        utt_id=utt_id,
        text=text,
        split=split,
        speaker_id=speaker_id,
        chapter_id=chapter_id,
        audio_bytes=audio_bytes,
        audio_path=_resolve_existing_audio_path(source_path_text, parquet_path),
        audio_extension=audio_extension,
    )


def iter_librispeech_parquet_utterances(
    parquet_root: str | Path,
    *,
    split: str,
) -> Iterator[ParquetAudioUtterance]:
    """Yield utterances from HuggingFace `openslr/librispeech_asr` parquet shards."""
    root = Path(parquet_root)
    parquet_paths = [root] if root.is_file() else sorted(root.glob("*.parquet"))
    if not parquet_paths:
        raise FileNotFoundError(f"No parquet shards found in {root}")

    for parquet_path in parquet_paths:
        for record in _iter_parquet_records(parquet_path):
            yield _parquet_record_to_utterance(record, split=split, parquet_path=parquet_path)
