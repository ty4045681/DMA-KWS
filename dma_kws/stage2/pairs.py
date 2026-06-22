"""Pair generation utilities for Stage II QbyT training."""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from typing import Sequence


def clip_to_audio_rel(clip_path: str) -> str:
    """Map a LibriPhrase `clips` path to a decoded-parquet `audio_rel` key.

    The aggregated parquet stores clip paths like `LP-100/<ngram>/<id>.wav`
    while the decoded shards key audio by `audio_rel` = `<ngram>/<id>.wav`.
    """
    prefix = "LP-100/"
    if clip_path.startswith(prefix):
        return clip_path[len(prefix):]
    return clip_path


@dataclass(frozen=True)
class AnchorExample:
    """Phrase anchor and available audio clips."""

    text: str
    phonemes: list[str]
    clips: list[str]


@dataclass(frozen=True)
class PairRecord:
    """One QbyT training/eval pair."""

    anchor_text: str
    anchor_phonemes: list[str]
    wav_path: str
    label: int
    sample_rate: int

    def to_json_dict(self) -> dict:
        return asdict(self)


def make_pair_records(
    anchors: Sequence[AnchorExample],
    *,
    negatives_per_anchor: int = 1,
    seed: int = 2025,
    sample_rate: int = 16000,
) -> list[PairRecord]:
    """Create positive pairs plus random negative pairs from phrase anchors."""
    rng = random.Random(seed)
    usable = [anchor for anchor in anchors if anchor.clips]
    pairs: list[PairRecord] = []

    for index, anchor in enumerate(usable):
        positive_clip = anchor.clips[0]
        pairs.append(
            PairRecord(
                anchor_text=anchor.text,
                anchor_phonemes=anchor.phonemes,
                wav_path=positive_clip,
                label=1,
                sample_rate=sample_rate,
            )
        )

        negative_pool = [candidate for candidate_index, candidate in enumerate(usable) if candidate_index != index]
        for _ in range(negatives_per_anchor):
            if not negative_pool:
                break
            negative_anchor = rng.choice(negative_pool)
            negative_clip = rng.choice(negative_anchor.clips)
            pairs.append(
                PairRecord(
                    anchor_text=anchor.text,
                    anchor_phonemes=anchor.phonemes,
                    wav_path=negative_clip,
                    label=0,
                    sample_rate=sample_rate,
                )
            )

    return pairs
