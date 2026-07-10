"""Audio span helpers shared by inference locators and verifiers."""

from __future__ import annotations


def num_fbank_frames(
    num_samples: int,
    *,
    sample_rate: int,
    frame_length_ms: float = 25.0,
    frame_shift_ms: float = 10.0,
    snip_edges: bool = True,
) -> int:
    if num_samples <= 0:
        return 0
    frame_length_samples = round(sample_rate * frame_length_ms / 1000.0)
    frame_shift_samples = round(sample_rate * frame_shift_ms / 1000.0)
    if not snip_edges:
        return (num_samples + frame_shift_samples // 2) // frame_shift_samples
    if num_samples < frame_length_samples:
        return 0
    return 1 + (num_samples - frame_length_samples) // frame_shift_samples


def min_samples_for_fbank_frames(
    min_frames: int,
    *,
    sample_rate: int,
    frame_length_ms: float = 25.0,
    frame_shift_ms: float = 10.0,
    snip_edges: bool = True,
) -> int:
    if min_frames <= 0:
        return 0
    frame_length_samples = round(sample_rate * frame_length_ms / 1000.0)
    frame_shift_samples = round(sample_rate * frame_shift_ms / 1000.0)
    if not snip_edges:
        return max(1, min_frames * frame_shift_samples - frame_shift_samples // 2)
    return frame_length_samples + (min_frames - 1) * frame_shift_samples


def has_min_fbank_frames(
    num_samples: int,
    *,
    min_frames: int,
    sample_rate: int,
    frame_length_ms: float = 25.0,
    frame_shift_ms: float = 10.0,
    snip_edges: bool = True,
) -> bool:
    return num_samples >= min_samples_for_fbank_frames(
        min_frames,
        sample_rate=sample_rate,
        frame_length_ms=frame_length_ms,
        frame_shift_ms=frame_shift_ms,
        snip_edges=snip_edges,
    )


def apply_margin_to_span(
    start_sec: float,
    end_sec: float,
    *,
    margin_sec: float,
    audio_duration_sec: float | None = None,
) -> tuple[float, float]:
    start = max(0.0, start_sec - margin_sec)
    end = end_sec + margin_sec
    if audio_duration_sec is not None:
        end = min(audio_duration_sec, end)
    return round(start, 6), round(end, 6)
