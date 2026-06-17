"""Pair generation utilities for Stage II QbyT training."""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from typing import Sequence


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

    def to_json_dict(self) -> dict:
        return asdict(self)


def make_pair_records(
    anchors: Sequence[AnchorExample],
    *,
    negatives_per_anchor: int = 1,
    seed: int = 2025,
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
                )
            )

    return pairs
