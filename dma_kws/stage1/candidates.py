"""Candidate generation from decoded phoneme frames."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class PhonemeFrame:
    """A decoded phoneme with an approximate time span and log score."""

    phoneme: str
    start_sec: float
    end_sec: float
    log_score: float = 0.0


@dataclass(frozen=True)
class KeywordCandidate:
    """A Stage I candidate region for Stage II verification."""

    start_sec: float
    end_sec: float
    stage1_score: float
    phonemes: list[str]


def find_keyword_candidates(
    decoded_frames: Sequence[PhonemeFrame],
    keyword_phonemes: Sequence[str],
    *,
    margin_sec: float = 0.0,
    max_insertions: int = 0,
) -> list[KeywordCandidate]:
    """Find approximate keyword subsequence matches in decoded phoneme frames."""
    if not keyword_phonemes:
        return []

    candidates: list[KeywordCandidate] = []
    frame_count = len(decoded_frames)
    keyword_count = len(keyword_phonemes)

    for start_index in range(frame_count):
        if decoded_frames[start_index].phoneme != keyword_phonemes[0]:
            continue

        keyword_index = 1
        insertions = 0
        end_index = start_index
        cursor = start_index + 1

        while cursor < frame_count and keyword_index < keyword_count:
            if decoded_frames[cursor].phoneme == keyword_phonemes[keyword_index]:
                keyword_index += 1
                end_index = cursor
            else:
                insertions += 1
                if insertions > max_insertions:
                    break
                end_index = cursor
            cursor += 1

        if keyword_index == keyword_count:
            window = list(decoded_frames[start_index : end_index + 1])
            candidates.append(
                KeywordCandidate(
                    start_sec=round(max(0.0, window[0].start_sec - margin_sec), 6),
                    end_sec=round(window[-1].end_sec + margin_sec, 6),
                    stage1_score=sum(frame.log_score for frame in window),
                    phonemes=[frame.phoneme for frame in window],
                )
            )

    return candidates
