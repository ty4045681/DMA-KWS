"""Background source headers, crop specs, and waveform materialization.

This module is the shared contract between online Stage II sampling and the
offline crop-cache builder. It must not import Lightning or QbyT.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
import torch


@dataclass(frozen=True)
class BackgroundSourceInfo:
    source_id: str
    path: Path
    sample_rate: int
    num_frames: int
    channels: int


@dataclass(frozen=True)
class BackgroundCropSpec:
    source_id: str
    duration_seconds: float
    read_start_frame: int
    read_num_frames: int
    target_num_samples: int
    final_offset: int


def probe_source_info(
    path: str | Path,
    *,
    source_id: str | None = None,
) -> BackgroundSourceInfo:
    """Read container headers without decoding waveform samples."""
    source_path = Path(path)
    with sf.SoundFile(str(source_path)) as handle:
        sample_rate = int(handle.samplerate)
        num_frames = int(len(handle))
        channels = int(handle.channels)
    if sample_rate <= 0:
        raise ValueError(f"Audio sample rate must be positive: {source_path}")
    if num_frames <= 0:
        raise ValueError(f"Audio contains no samples: {source_path}")
    if channels <= 0:
        raise ValueError(f"Audio contains no channels: {source_path}")
    return BackgroundSourceInfo(
        source_id=source_id if source_id is not None else str(source_path),
        path=source_path,
        sample_rate=sample_rate,
        num_frames=num_frames,
        channels=channels,
    )


def draw_crop_spec(
    source: BackgroundSourceInfo,
    duration_seconds: float,
    *,
    rng: random.Random,
) -> BackgroundCropSpec:
    """Draw a partial-read crop for an already chosen source.

    Consumes RNG for the decode start (long files) and the final slice/repeat
    phase, matching online ``TrainingBackgroundSampler.extract``. Does not
    choose among recordings.
    """
    duration_seconds = float(duration_seconds)
    sample_rate = int(source.sample_rate)
    num_frames = int(source.num_frames)
    if sample_rate <= 0:
        raise ValueError("source sample_rate must be positive")
    if num_frames <= 0:
        raise ValueError("source num_frames must be positive")

    # Microsecond units preserve fractional seconds without opening the file
    # merely to discover its sample rate. The online path used target_samples
    # at 1e6 Hz so ceil(duration * source_rate) keeps sub-millisecond duration.
    duration_units = max(1, round(duration_seconds * 1_000_000))
    frames = max(1, math.ceil(duration_units * sample_rate / 1_000_000))
    if num_frames > frames:
        read_start_frame = rng.randint(0, num_frames - frames)
        read_num_frames = frames
    else:
        read_start_frame = 0
        read_num_frames = num_frames

    target_num_samples = max(1, round(duration_seconds * sample_rate))
    if read_num_frames >= target_num_samples:
        max_offset = read_num_frames - target_num_samples
        final_offset = rng.randint(0, max_offset) if max_offset else 0
    else:
        final_offset = rng.randrange(read_num_frames) if read_num_frames > 1 else 0

    return BackgroundCropSpec(
        source_id=source.source_id,
        duration_seconds=duration_seconds,
        read_start_frame=int(read_start_frame),
        read_num_frames=int(read_num_frames),
        target_num_samples=int(target_num_samples),
        final_offset=int(final_offset),
    )


def materialize_crop(
    source: BackgroundSourceInfo,
    spec: BackgroundCropSpec,
) -> tuple[torch.Tensor, int]:
    """Decode ``spec`` and return a mono ``[1, T]`` float32 crop.

    Does not draw RNG. Reads exactly ``spec.read_num_frames`` from
    ``spec.read_start_frame``, converts to mono, then applies the final
    slice or periodic repeat using ``spec.final_offset``.
    """
    _validate_crop_spec(source, spec)
    with sf.SoundFile(str(source.path)) as handle:
        handle.seek(spec.read_start_frame)
        array = handle.read(
            frames=spec.read_num_frames,
            dtype="float32",
            always_2d=True,
        )
    if int(array.shape[0]) != spec.read_num_frames:
        raise ValueError(
            f"Decoded {array.shape[0]} frames from {source.path}, "
            f"expected {spec.read_num_frames}"
        )
    if int(array.shape[1]) != int(source.channels):
        raise ValueError(
            f"Decoded {array.shape[1]} channels from {source.path}, "
            f"expected {source.channels}"
        )
    waveform = torch.from_numpy(np.asarray(array, dtype=np.float32).T.copy())
    waveform = _as_mono(waveform, source=source.path)
    waveform = _apply_final_crop(waveform, spec)
    return waveform.to(dtype=torch.float32).contiguous(), int(source.sample_rate)


def _validate_crop_spec(source: BackgroundSourceInfo, spec: BackgroundCropSpec) -> None:
    if spec.source_id != source.source_id:
        raise ValueError(
            f"Crop spec source_id {spec.source_id!r} does not match "
            f"{source.source_id!r}"
        )
    if spec.read_start_frame < 0 or spec.read_num_frames < 1:
        raise ValueError("Crop read window must be a positive-length span")
    read_end = spec.read_start_frame + spec.read_num_frames
    if read_end > source.num_frames:
        raise ValueError(
            f"Crop read window [{spec.read_start_frame}, {read_end}) "
            f"exceeds source frames {source.num_frames}"
        )
    if spec.target_num_samples < 1:
        raise ValueError("Crop target_num_samples must be positive")
    if spec.final_offset < 0:
        raise ValueError("Crop final_offset must be non-negative")
    if spec.read_num_frames >= spec.target_num_samples:
        if spec.final_offset + spec.target_num_samples > spec.read_num_frames:
            raise ValueError("Crop final_offset exceeds the decoded window")
    elif spec.final_offset >= spec.read_num_frames:
        raise ValueError("Crop repeat phase exceeds the decoded window")


def _as_mono(waveform: torch.Tensor, *, source: Path) -> torch.Tensor:
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    if waveform.dim() != 2 or waveform.size(0) == 0:
        raise ValueError(
            f"Expected audio shaped (channels, samples) for {source}, "
            f"got {tuple(waveform.shape)}"
        )
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if waveform.size(1) == 0:
        raise ValueError(f"Audio contains no samples: {source}")
    return waveform.to(dtype=torch.float32).contiguous()


def _apply_final_crop(waveform: torch.Tensor, spec: BackgroundCropSpec) -> torch.Tensor:
    noise_samples = int(waveform.size(1))
    target_samples = int(spec.target_num_samples)
    offset = int(spec.final_offset)
    if noise_samples >= target_samples:
        return waveform[:, offset : offset + target_samples]

    # Repeat one extra period so a random phase still yields a complete
    # target-length slice instead of always starting at noise sample zero.
    repeats = math.ceil((target_samples + offset) / noise_samples)
    tiled = waveform.repeat(1, repeats)
    return tiled[:, offset : offset + target_samples]
