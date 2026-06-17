"""LibriSpeech helpers for Stage I phoneme CTC preparation."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

_STRESS_RE = re.compile(r"\d+$")


@dataclass(frozen=True)
class TranscriptUtterance:
    """One LibriSpeech utterance and its transcript."""

    utt_id: str
    text: str
    audio_path: Path
    split: str


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
