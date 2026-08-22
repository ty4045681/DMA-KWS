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


def window_fbank_frame_span(
    start_sample: int,
    end_sample: int,
    *,
    sample_rate: int,
    frame_length_ms: float = 25.0,
    frame_shift_ms: float = 10.0,
    snip_edges: bool = True,
) -> tuple[int, int]:
    """Return ``[start_frame, end_frame)`` in a full-file fbank for one window.

    The width matches :func:`num_fbank_frames` on the isolated window. When
    ``snip_edges`` is true, ``dither`` is zero, and ``start_sample`` is a
    multiple of the frame shift, slicing the full-file features over this span
    equals extracting fbank on ``waveform[:, start_sample:end_sample]``.
    ``snip_edges=false`` uses the same width so callers can score file slices,
    but those frames are not bit-identical to an independent window extract.
    """
    if start_sample < 0 or end_sample < start_sample:
        raise ValueError("window samples must satisfy 0 <= start <= end")
    num_frames = num_fbank_frames(
        end_sample - start_sample,
        sample_rate=sample_rate,
        frame_length_ms=frame_length_ms,
        frame_shift_ms=frame_shift_ms,
        snip_edges=snip_edges,
    )
    if num_frames <= 0:
        return 0, 0
    frame_shift_samples = round(sample_rate * frame_shift_ms / 1000.0)
    if frame_shift_samples <= 0:
        raise ValueError("frame shift must be positive")
    if snip_edges:
        start_frame = start_sample // frame_shift_samples
    else:
        start_frame = (
            start_sample + frame_shift_samples // 2
        ) // frame_shift_samples
    return start_frame, start_frame + num_frames


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
