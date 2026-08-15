"""Deterministic MUSAN interference mixing for clip evaluation."""

from __future__ import annotations

import hashlib
import math
import random
import sys
from array import array
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from dma_kws.audio import load_audio
from dma_kws.inference.manifest import iter_audio_files


_RMS_EPSILON = 1.0e-8
_MIX_KEYS = {
    "seed",
    "noise",
    "music",
    "speech",
    "stationary_noise",
    "burst_noise",
    "volume_variation",
}
_STATIONARY_NOISE_KINDS = {"white_gaussian"}
_BURST_SNR_SCOPES = {"active_event", "whole_clip"}


@dataclass(frozen=True)
class _ComponentRecipe:
    kind: str
    source_path: str
    source: str
    level_db: float
    offset_fraction: float
    recipe_seed: int

    def metadata(self) -> dict[str, Any]:
        if self.kind == "speech":
            level_name = "relative_db"
        elif self.kind in {"noise", "music"}:
            level_name = "snr_db"
        else:
            raise ValueError(f"Unsupported MUSAN component kind: {self.kind}")
        return {
            "source": self.source,
            level_name: self.level_db,
            "offset_fraction": self.offset_fraction,
            "recipe_seed": self.recipe_seed,
        }


@dataclass(frozen=True)
class _MixRecipe:
    noise: _ComponentRecipe | None = None
    music: _ComponentRecipe | None = None
    speech: _ComponentRecipe | None = None
    stationary_noise_seed: int | None = None
    burst_noise: "_BurstNoiseRecipe | None" = None
    volume_variation_seed: int | None = None


@dataclass(frozen=True)
class _BurstEventRecipe:
    event_index: int
    component: _ComponentRecipe
    duration_ms: float
    placement_fraction: float

    def metadata(self) -> dict[str, Any]:
        return {
            "event_index": self.event_index,
            "source": self.component.source,
            "duration_ms": self.duration_ms,
            "start_fraction": self.placement_fraction,
            "source_offset_fraction": self.component.offset_fraction,
            "recipe_seed": self.component.recipe_seed,
        }


@dataclass(frozen=True)
class _BurstNoiseRecipe:
    recipe_seed: int
    events: tuple[_BurstEventRecipe, ...]


@dataclass(frozen=True)
class _StationaryNoiseConfig:
    enabled: bool
    kind: str
    snr_db: float


@dataclass(frozen=True)
class _BurstNoiseConfig:
    enabled: bool
    snr_db: float
    snr_scope: str
    event_count_min: int
    event_count_max: int
    duration_ms_min: float
    duration_ms_max: float
    fade_ms: float
    allow_overlap: bool
    min_gap_ms: float


@dataclass(frozen=True)
class _VolumeVariationConfig:
    enabled: bool
    low_gain_db: float
    high_gain_db: float
    segment_ms_min: float
    segment_ms_max: float
    transition_ms: float


def _require_mapping(value: object, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field} must be a mapping")
    return value


def _reject_unknown_keys(
    mapping: Mapping[str, Any],
    allowed: set[str],
    *,
    field: str,
) -> None:
    unknown = sorted(str(key) for key in mapping if key not in allowed)
    if unknown:
        raise ValueError(f"{field} has unsupported keys: {', '.join(unknown)}")


def _require_bool(value: object, *, field: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{field} must be true or false")
    return value


def _require_finite_float(value: object, *, field: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{field} must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{field} must be a finite number") from exc
    if not math.isfinite(number):
        raise ValueError(f"{field} must be a finite number")
    return number


def _require_int(
    value: object,
    *,
    field: str,
    minimum: int,
    maximum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field} must be an integer")
    number = int(value)
    if number < minimum or (maximum is not None and number > maximum):
        suffix = f" and <= {maximum}" if maximum is not None else ""
        raise ValueError(f"{field} must be >= {minimum}{suffix}")
    return number


def _require_nonnegative_float(value: object, *, field: str) -> float:
    number = _require_finite_float(value, field=field)
    if number < 0.0:
        raise ValueError(f"{field} must be >= 0")
    return number


def _require_positive_float(value: object, *, field: str) -> float:
    number = _require_finite_float(value, field=field)
    if number <= 0.0:
        raise ValueError(f"{field} must be > 0")
    return number


def _gain_ratio(level_db: float, *, field: str) -> float:
    try:
        ratio = 10.0 ** (level_db / 20.0)
    except OverflowError as exc:
        raise ValueError(f"{field} is outside the supported numeric range") from exc
    if ratio <= 0.0 or not math.isfinite(ratio):
        raise ValueError(f"{field} is outside the supported numeric range")
    return ratio


def _target_amplitude_ratio(kind: str, level_db: float, *, field: str) -> float:
    if kind == "speech":
        exponent = level_db / 20.0
    elif kind in {"noise", "music"}:
        exponent = -level_db / 20.0
    else:
        raise ValueError(f"Unsupported MUSAN component kind: {kind}")
    try:
        ratio = 10.0**exponent
    except OverflowError as exc:
        raise ValueError(f"{field} is outside the supported numeric range") from exc
    if ratio <= 0.0 or not math.isfinite(ratio):
        raise ValueError(f"{field} is outside the supported numeric range")
    return ratio


def _derive_seed(base_seed: int, index: int, audio_path: str, kind: str) -> int:
    payload = f"{base_seed}\0{index}\0{audio_path}\0{kind}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _derive_event_seed(component_seed: int, event_index: int) -> int:
    payload = f"{component_seed}\0burst_noise_event\0{event_index}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _discover_subset(root: Path, subset: str) -> tuple[Path, ...]:
    subset_root = root / subset
    if not subset_root.is_dir():
        raise NotADirectoryError(f"MUSAN {subset} directory not found: {subset_root}")
    files = tuple(iter_audio_files(subset_root))
    if not files:
        raise ValueError(f"No supported audio files found under {subset_root}")
    return files


def _component_recipe(
    *,
    root: Path,
    files: Sequence[Path],
    kind: str,
    level_db: float,
    base_seed: int,
    index: int,
    audio_path: str,
) -> _ComponentRecipe:
    recipe_seed = _derive_seed(base_seed, index, audio_path, kind)
    rng = random.Random(recipe_seed)
    selected_path = files[rng.randrange(len(files))]
    source_path = selected_path.resolve()
    return _ComponentRecipe(
        kind=kind,
        source_path=str(source_path),
        source=selected_path.relative_to(root).as_posix(),
        level_db=level_db,
        offset_fraction=rng.random(),
        recipe_seed=recipe_seed,
    )


def _burst_noise_recipe(
    *,
    root: Path,
    files: Sequence[Path],
    config: _BurstNoiseConfig,
    base_seed: int,
    index: int,
    audio_path: str,
) -> _BurstNoiseRecipe:
    recipe_seed = _derive_seed(base_seed, index, audio_path, "burst_noise")
    event_count_rng = random.Random(recipe_seed)
    event_count = event_count_rng.randint(
        config.event_count_min,
        config.event_count_max,
    )
    events = []
    for event_index in range(event_count):
        event_seed = _derive_event_seed(recipe_seed, event_index)
        rng = random.Random(event_seed)
        selected_path = files[rng.randrange(len(files))]
        component = _ComponentRecipe(
            kind="noise",
            source_path=str(selected_path.resolve()),
            source=selected_path.relative_to(root).as_posix(),
            level_db=config.snr_db,
            offset_fraction=rng.random(),
            recipe_seed=event_seed,
        )
        events.append(
            _BurstEventRecipe(
                event_index=event_index,
                component=component,
                duration_ms=rng.uniform(
                    config.duration_ms_min,
                    config.duration_ms_max,
                ),
                placement_fraction=rng.random(),
            )
        )
    return _BurstNoiseRecipe(recipe_seed=recipe_seed, events=tuple(events))


def _import_torch():
    try:
        import torch
    except ImportError as exc:
        raise SystemExit(
            "Missing torch/torchaudio. Install CUDA PyTorch on the remote training machine first."
        ) from exc
    return torch


def _validate_finite_waveform(waveform, *, field: str) -> None:
    torch = _import_torch()

    if waveform.numel() == 0:
        raise ValueError(f"{field} is empty")
    if not bool(torch.isfinite(waveform).all().item()):
        raise ValueError(f"{field} contains non-finite samples")


def _waveform_rms(waveform, *, field: str) -> float:
    _validate_finite_waveform(waveform, field=field)
    torch = _import_torch()
    rms = float(waveform.to(dtype=torch.float64).square().mean().sqrt().item())
    if rms <= _RMS_EPSILON:
        raise ValueError(
            f"{field} RMS must be greater than {_RMS_EPSILON:g} for level-controlled mixing"
        )
    return rms


def _sample_rate(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("sample_rate must be a positive integer")
    if value <= 0:
        raise ValueError("sample_rate must be a positive integer")
    return int(value)


def _milliseconds_to_samples(
    milliseconds: float,
    sample_rate: int,
    *,
    allow_zero: bool,
) -> int:
    samples = int(round(milliseconds * sample_rate / 1000.0))
    return max(0 if allow_zero else 1, samples)


def _volume_envelope(
    *,
    target_samples: int,
    sample_rate: int,
    config: _VolumeVariationConfig,
    recipe_seed: int,
) -> np.ndarray:
    if target_samples <= 0:
        raise ValueError("clean waveform is empty")
    minimum = _milliseconds_to_samples(
        config.segment_ms_min,
        sample_rate,
        allow_zero=False,
    )
    maximum = _milliseconds_to_samples(
        config.segment_ms_max,
        sample_rate,
        allow_zero=False,
    )
    transition = _milliseconds_to_samples(
        config.transition_ms,
        sample_rate,
        allow_zero=True,
    )
    rng = random.Random(recipe_seed)
    envelope = np.empty(target_samples, dtype=np.float64)
    cursor = 0
    current_gain_db = (
        config.low_gain_db if rng.randrange(2) == 0 else config.high_gain_db
    )
    previous_gain_db = None
    while cursor < target_samples:
        segment_samples = rng.randint(minimum, maximum)
        end = min(target_samples, cursor + segment_samples)
        gain = _gain_ratio(current_gain_db, field="volume_variation gain")
        envelope[cursor:end] = gain
        if previous_gain_db is not None and transition > 0:
            width = min(transition, end - cursor)
            phase = np.arange(1, width + 1, dtype=np.float64) / (width + 1.0)
            interpolation = 0.5 - (0.5 * np.cos(np.pi * phase))
            transition_db = (
                previous_gain_db
                + interpolation * (current_gain_db - previous_gain_db)
            )
            envelope[cursor : cursor + width] = (
                np.power(10.0, transition_db / 20.0)
            )
        previous_gain_db = current_gain_db
        current_gain_db = (
            config.high_gain_db
            if current_gain_db == config.low_gain_db
            else config.low_gain_db
        )
        cursor = end
    return envelope


def _fade_envelope(event_samples: int, fade_samples: int) -> np.ndarray:
    envelope = np.ones(event_samples, dtype=np.float64)
    fade_samples = min(fade_samples, (event_samples - 1) // 2)
    if fade_samples <= 0:
        return envelope
    phase = np.arange(fade_samples, dtype=np.float64) / fade_samples
    ramp = 0.5 - (0.5 * np.cos(np.pi * phase))
    envelope[:fade_samples] = ramp
    envelope[-fade_samples:] = ramp[::-1]
    return envelope


def _burst_start(
    *,
    target_samples: int,
    event_samples: int,
    placement_fraction: float,
    allow_overlap: bool,
    min_gap_samples: int,
    placed: Sequence[tuple[int, int]],
) -> int:
    maximum_start = target_samples - event_samples
    if maximum_start < 0:
        raise ValueError("burst event duration exceeds clean waveform length")
    if allow_overlap or not placed:
        return min(
            maximum_start,
            int(placement_fraction * (maximum_start + 1)),
        )

    valid_ranges = [(0, maximum_start)]
    for occupied_start, occupied_end in placed:
        forbidden_start = occupied_start - event_samples - min_gap_samples + 1
        forbidden_end = occupied_end + min_gap_samples - 1
        remaining = []
        for range_start, range_end in valid_ranges:
            if forbidden_end < range_start or forbidden_start > range_end:
                remaining.append((range_start, range_end))
                continue
            if range_start < forbidden_start:
                remaining.append((range_start, forbidden_start - 1))
            if forbidden_end < range_end:
                remaining.append((forbidden_end + 1, range_end))
        valid_ranges = remaining
        if not valid_ranges:
            break
    if not valid_ranges:
        raise ValueError(
            "burst_noise events cannot fit without overlap at the configured "
            "durations and min_gap_ms"
        )

    available = sum(end - start + 1 for start, end in valid_ranges)
    rank = min(available - 1, int(placement_fraction * available))
    for start, end in valid_ranges:
        size = end - start + 1
        if rank < size:
            return start + rank
        rank -= size
    raise RuntimeError("failed to select deterministic burst start")  # pragma: no cover


def _match_length(waveform, target_samples: int, offset_fraction: float, *, field: str):
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    if waveform.dim() != 2 or waveform.size(0) != 1:
        raise ValueError(
            f"{field} must be mono with shape (1, samples), got {tuple(waveform.shape)}"
        )
    source_samples = int(waveform.size(1))
    if source_samples <= 0:
        raise ValueError(f"{field} is empty")
    if target_samples <= 0:
        raise ValueError("clean waveform is empty")

    if source_samples >= target_samples:
        max_start = source_samples - target_samples
        start = min(max_start, int(offset_fraction * (max_start + 1)))
        return waveform[:, start : start + target_samples]

    start = min(source_samples - 1, int(offset_fraction * source_samples))
    repeats = math.ceil((start + target_samples) / source_samples)
    return waveform.repeat(1, repeats)[:, start : start + target_samples]


def _load_source_audio(
    component: _ComponentRecipe,
    *,
    target_samples: int,
    sample_rate: int,
):
    """Read only the needed span when the source length can be inspected."""

    source_info = _source_audio_info(component.source_path)
    if source_info is None:
        return load_audio(component.source_path, sample_rate=sample_rate)
    total_frames, source_sample_rate = source_info

    # Short sources are read in full so _match_length can repeat them from the
    # deterministic phase. ceil() provides enough input frames for rate
    # conversion without shrinking the valid crop-start range.
    source_frames = math.ceil(target_samples * source_sample_rate / sample_rate)
    source_is_short = total_frames < source_frames
    frames_to_read = total_frames if source_is_short else source_frames
    if source_is_short:
        frame_offset = 0
    else:
        max_start = total_frames - frames_to_read
        frame_offset = min(
            max_start,
            int(component.offset_fraction * (max_start + 1)),
        )
    partial = _read_pcm_wav_span(
        component.source_path,
        frame_offset=frame_offset,
        num_frames=frames_to_read,
    )
    if partial is not None:
        waveform, source_rate = partial
    else:
        try:
            import torchaudio

            waveform, source_rate = torchaudio.load(
                component.source_path,
                frame_offset=frame_offset,
                num_frames=frames_to_read,
            )
        except (ImportError, OSError, RuntimeError, ValueError):
            return load_audio(component.source_path, sample_rate=sample_rate)
    if waveform.size(1) != frames_to_read:
        return load_audio(component.source_path, sample_rate=sample_rate)

    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if int(source_rate) != int(sample_rate):
        try:
            import torchaudio
        except ImportError:
            return load_audio(component.source_path, sample_rate=sample_rate)
        waveform = torchaudio.functional.resample(
            waveform,
            int(source_rate),
            int(sample_rate),
        )
    if not source_is_short and waveform.size(1) < target_samples:
        return load_audio(component.source_path, sample_rate=sample_rate)
    if source_is_short:
        return waveform.contiguous(), int(sample_rate)
    return waveform[:, :target_samples].contiguous(), int(sample_rate)


def _read_pcm_wav_span(
    path: str,
    *,
    frame_offset: int,
    num_frames: int,
):
    """Decode an uncompressed PCM WAV span without optional codec packages."""

    if Path(path).suffix.lower() != ".wav":
        return None

    try:
        import wave

        with wave.open(path, "rb") as handle:
            if handle.getcomptype() != "NONE":
                return None
            channels = handle.getnchannels()
            sample_width = handle.getsampwidth()
            sample_rate = handle.getframerate()
            if channels <= 0 or sample_rate <= 0 or sample_width not in {1, 2, 3, 4}:
                return None
            handle.setpos(frame_offset)
            payload = handle.readframes(num_frames)
    except (EOFError, OSError, wave.Error):
        return None

    bytes_per_frame = channels * sample_width
    if len(payload) != num_frames * bytes_per_frame:
        return None

    torch = _import_torch()
    if sample_width == 1:
        values = array("B")
        values.frombytes(payload)
        samples = (torch.tensor(values, dtype=torch.float32) - 128.0) / 128.0
    elif sample_width == 2:
        values = array("h")
        values.frombytes(payload)
        if sys.byteorder != "little":
            values.byteswap()
        samples = torch.tensor(values, dtype=torch.float32) / 32768.0
    elif sample_width == 4:
        values = array("i")
        if values.itemsize != 4:
            return None
        values.frombytes(payload)
        if sys.byteorder != "little":
            values.byteswap()
        samples = torch.tensor(values, dtype=torch.float32) / 2147483648.0
    else:
        packed = torch.frombuffer(bytearray(payload), dtype=torch.uint8)
        packed = packed.reshape(-1, 3).to(dtype=torch.int32)
        samples = packed[:, 0] | (packed[:, 1] << 8) | (packed[:, 2] << 16)
        samples = torch.where(
            (samples & 0x800000) != 0,
            samples - 0x1000000,
            samples,
        ).to(dtype=torch.float32)
        samples = samples / 8388608.0

    waveform = samples.reshape(-1, channels).transpose(0, 1).contiguous()
    return waveform, int(sample_rate)


def _source_audio_info(path: str) -> tuple[int, int] | None:
    if Path(path).suffix.lower() == ".wav":
        try:
            import wave

            with wave.open(path, "rb") as handle:
                return handle.getnframes(), handle.getframerate()
        except (EOFError, OSError, wave.Error):
            pass

    try:
        import soundfile as sf

        info = sf.info(path)
    except (ImportError, OSError, RuntimeError, ValueError):
        return None
    if info.frames <= 0 or info.samplerate <= 0:
        return None
    return int(info.frames), int(info.samplerate)


def _parse_stationary_noise(mix: Mapping[str, Any]) -> _StationaryNoiseConfig:
    section = _require_mapping(
        mix.get("stationary_noise", {}),
        field="prep.musan_mix.stationary_noise",
    )
    _reject_unknown_keys(
        section,
        {"enabled", "kind", "snr_db"},
        field="prep.musan_mix.stationary_noise",
    )
    enabled = _require_bool(
        section.get("enabled", False),
        field="prep.musan_mix.stationary_noise.enabled",
    )
    kind = section.get("kind", "white_gaussian")
    if not isinstance(kind, str) or kind not in _STATIONARY_NOISE_KINDS:
        raise ValueError(
            "prep.musan_mix.stationary_noise.kind must be white_gaussian"
        )
    snr_db = _require_finite_float(
        section.get("snr_db", 20.0),
        field="prep.musan_mix.stationary_noise.snr_db",
    )
    _target_amplitude_ratio(
        "noise",
        snr_db,
        field="prep.musan_mix.stationary_noise.snr_db",
    )
    return _StationaryNoiseConfig(enabled=enabled, kind=kind, snr_db=snr_db)


def _parse_burst_noise(mix: Mapping[str, Any]) -> _BurstNoiseConfig:
    section = _require_mapping(
        mix.get("burst_noise", {}),
        field="prep.musan_mix.burst_noise",
    )
    allowed = {
        "enabled",
        "snr_db",
        "snr_scope",
        "event_count_min",
        "event_count_max",
        "duration_ms_min",
        "duration_ms_max",
        "fade_ms",
        "allow_overlap",
        "min_gap_ms",
    }
    _reject_unknown_keys(section, allowed, field="prep.musan_mix.burst_noise")
    enabled = _require_bool(
        section.get("enabled", False),
        field="prep.musan_mix.burst_noise.enabled",
    )
    snr_db = _require_finite_float(
        section.get("snr_db", 10.0),
        field="prep.musan_mix.burst_noise.snr_db",
    )
    _target_amplitude_ratio(
        "noise",
        snr_db,
        field="prep.musan_mix.burst_noise.snr_db",
    )
    snr_scope = section.get("snr_scope", "active_event")
    if not isinstance(snr_scope, str) or snr_scope not in _BURST_SNR_SCOPES:
        raise ValueError(
            "prep.musan_mix.burst_noise.snr_scope must be active_event or "
            "whole_clip"
        )
    event_count_min = _require_int(
        section.get("event_count_min", 1),
        field="prep.musan_mix.burst_noise.event_count_min",
        minimum=0,
    )
    event_count_max = _require_int(
        section.get("event_count_max", 1),
        field="prep.musan_mix.burst_noise.event_count_max",
        minimum=0,
    )
    if event_count_max < event_count_min:
        raise ValueError(
            "prep.musan_mix.burst_noise.event_count_max must be >= "
            "event_count_min"
        )
    duration_ms_min = _require_positive_float(
        section.get("duration_ms_min", 100.0),
        field="prep.musan_mix.burst_noise.duration_ms_min",
    )
    duration_ms_max = _require_positive_float(
        section.get("duration_ms_max", 400.0),
        field="prep.musan_mix.burst_noise.duration_ms_max",
    )
    if duration_ms_max < duration_ms_min:
        raise ValueError(
            "prep.musan_mix.burst_noise.duration_ms_max must be >= "
            "duration_ms_min"
        )
    fade_ms = _require_nonnegative_float(
        section.get("fade_ms", 10.0),
        field="prep.musan_mix.burst_noise.fade_ms",
    )
    allow_overlap = _require_bool(
        section.get("allow_overlap", False),
        field="prep.musan_mix.burst_noise.allow_overlap",
    )
    min_gap_ms = _require_nonnegative_float(
        section.get("min_gap_ms", 50.0),
        field="prep.musan_mix.burst_noise.min_gap_ms",
    )
    return _BurstNoiseConfig(
        enabled=enabled,
        snr_db=snr_db,
        snr_scope=snr_scope,
        event_count_min=event_count_min,
        event_count_max=event_count_max,
        duration_ms_min=duration_ms_min,
        duration_ms_max=duration_ms_max,
        fade_ms=fade_ms,
        allow_overlap=allow_overlap,
        min_gap_ms=min_gap_ms,
    )


def _parse_volume_variation(mix: Mapping[str, Any]) -> _VolumeVariationConfig:
    section = _require_mapping(
        mix.get("volume_variation", {}),
        field="prep.musan_mix.volume_variation",
    )
    allowed = {
        "enabled",
        "low_gain_db",
        "high_gain_db",
        "segment_ms_min",
        "segment_ms_max",
        "transition_ms",
    }
    _reject_unknown_keys(section, allowed, field="prep.musan_mix.volume_variation")
    enabled = _require_bool(
        section.get("enabled", False),
        field="prep.musan_mix.volume_variation.enabled",
    )
    low_gain_db = _require_finite_float(
        section.get("low_gain_db", -12.0),
        field="prep.musan_mix.volume_variation.low_gain_db",
    )
    high_gain_db = _require_finite_float(
        section.get("high_gain_db", 6.0),
        field="prep.musan_mix.volume_variation.high_gain_db",
    )
    if high_gain_db <= low_gain_db:
        raise ValueError(
            "prep.musan_mix.volume_variation.high_gain_db must be > low_gain_db"
        )
    _gain_ratio(low_gain_db, field="prep.musan_mix.volume_variation.low_gain_db")
    _gain_ratio(high_gain_db, field="prep.musan_mix.volume_variation.high_gain_db")
    segment_ms_min = _require_positive_float(
        section.get("segment_ms_min", 250.0),
        field="prep.musan_mix.volume_variation.segment_ms_min",
    )
    segment_ms_max = _require_positive_float(
        section.get("segment_ms_max", 750.0),
        field="prep.musan_mix.volume_variation.segment_ms_max",
    )
    if segment_ms_max < segment_ms_min:
        raise ValueError(
            "prep.musan_mix.volume_variation.segment_ms_max must be >= "
            "segment_ms_min"
        )
    transition_ms = _require_nonnegative_float(
        section.get("transition_ms", 50.0),
        field="prep.musan_mix.volume_variation.transition_ms",
    )
    if transition_ms > segment_ms_min:
        raise ValueError(
            "prep.musan_mix.volume_variation.transition_ms must be <= "
            "segment_ms_min"
        )
    return _VolumeVariationConfig(
        enabled=enabled,
        low_gain_db=low_gain_db,
        high_gain_db=high_gain_db,
        segment_ms_min=segment_ms_min,
        segment_ms_max=segment_ms_max,
        transition_ms=transition_ms,
    )


class MusanWaveformMixer:
    """Apply a precomputed, per-row MUSAN mixing recipe.

    Recipes are generated before DataLoader workers start. Their stable seeds
    depend on the configured seed, row index, clean path and component kind, so
    worker scheduling and batch size cannot change source selection or offsets.
    """

    def __init__(
        self,
        *,
        seed: int,
        musan_root: Path | None,
        noise_enabled: bool,
        noise_snr_db: float,
        noise_files: Sequence[Path],
        music_enabled: bool,
        music_snr_db: float,
        music_files: Sequence[Path],
        speech_enabled: bool,
        speech_relative_db: float,
        speech_files: Sequence[Path],
        stationary_noise: _StationaryNoiseConfig,
        burst_noise: _BurstNoiseConfig,
        volume_variation: _VolumeVariationConfig,
        recipes: Sequence[_MixRecipe],
    ) -> None:
        self._seed = seed
        self._musan_root = musan_root
        self._noise_enabled = noise_enabled
        self._noise_snr_db = noise_snr_db
        self._noise_files = tuple(noise_files)
        self._music_enabled = music_enabled
        self._music_snr_db = music_snr_db
        self._music_files = tuple(music_files)
        self._speech_enabled = speech_enabled
        self._speech_relative_db = speech_relative_db
        self._speech_files = tuple(speech_files)
        self._stationary_noise = stationary_noise
        self._burst_noise = burst_noise
        self._volume_variation = volume_variation
        self._recipes = tuple(recipes)

    @classmethod
    def from_prep(
        cls,
        prep: Mapping[str, Any],
        *,
        audio_paths: Sequence[str | Path],
    ) -> "MusanWaveformMixer":
        """Build resolved source pools and deterministic recipes from prep config."""

        prep = _require_mapping(prep, field="prep")
        mix_raw = prep.get("musan_mix", {})
        if mix_raw is None:
            mix_raw = {}
        mix = _require_mapping(mix_raw, field="prep.musan_mix")
        _reject_unknown_keys(mix, _MIX_KEYS, field="prep.musan_mix")

        seed_raw = mix.get("seed", 2025)
        if isinstance(seed_raw, bool) or not isinstance(seed_raw, int):
            raise TypeError("prep.musan_mix.seed must be a non-negative integer")
        if seed_raw < 0:
            raise ValueError("prep.musan_mix.seed must be a non-negative integer")
        seed = int(seed_raw)

        noise = _require_mapping(
            mix.get("noise", {}),
            field="prep.musan_mix.noise",
        )
        music = _require_mapping(
            mix.get("music", {}),
            field="prep.musan_mix.music",
        )
        speech = _require_mapping(
            mix.get("speech", {}),
            field="prep.musan_mix.speech",
        )
        _reject_unknown_keys(
            noise,
            {"enabled", "snr_db"},
            field="prep.musan_mix.noise",
        )
        _reject_unknown_keys(
            music,
            {"enabled", "snr_db"},
            field="prep.musan_mix.music",
        )
        _reject_unknown_keys(
            speech,
            {"enabled", "relative_db"},
            field="prep.musan_mix.speech",
        )
        noise_enabled = _require_bool(
            noise.get("enabled", False),
            field="prep.musan_mix.noise.enabled",
        )
        music_enabled = _require_bool(
            music.get("enabled", False),
            field="prep.musan_mix.music.enabled",
        )
        speech_enabled = _require_bool(
            speech.get("enabled", False),
            field="prep.musan_mix.speech.enabled",
        )
        noise_snr_db = _require_finite_float(
            noise.get("snr_db", 20.0),
            field="prep.musan_mix.noise.snr_db",
        )
        music_snr_db = _require_finite_float(
            music.get("snr_db", 20.0),
            field="prep.musan_mix.music.snr_db",
        )
        speech_relative_db = _require_finite_float(
            speech.get("relative_db", 0.0),
            field="prep.musan_mix.speech.relative_db",
        )
        _target_amplitude_ratio(
            "noise",
            noise_snr_db,
            field="prep.musan_mix.noise.snr_db",
        )
        _target_amplitude_ratio(
            "music",
            music_snr_db,
            field="prep.musan_mix.music.snr_db",
        )
        _target_amplitude_ratio(
            "speech",
            speech_relative_db,
            field="prep.musan_mix.speech.relative_db",
        )
        stationary_noise = _parse_stationary_noise(mix)
        burst_noise = _parse_burst_noise(mix)
        volume_variation = _parse_volume_variation(mix)
        clean_paths = tuple(str(path) for path in audio_paths)

        any_enabled = any(
            (
                noise_enabled,
                music_enabled,
                speech_enabled,
                stationary_noise.enabled,
                burst_noise.enabled,
                volume_variation.enabled,
            )
        )
        if not any_enabled:
            return cls(
                seed=seed,
                musan_root=None,
                noise_enabled=False,
                noise_snr_db=noise_snr_db,
                noise_files=(),
                music_enabled=False,
                music_snr_db=music_snr_db,
                music_files=(),
                speech_enabled=False,
                speech_relative_db=speech_relative_db,
                speech_files=(),
                stationary_noise=stationary_noise,
                burst_noise=burst_noise,
                volume_variation=volume_variation,
                recipes=(_MixRecipe() for _ in clean_paths),
            )

        musan_required = any(
            (noise_enabled, music_enabled, speech_enabled, burst_noise.enabled)
        )
        root = None
        if musan_required:
            root_raw = prep.get("musan_root", "")
            if not isinstance(root_raw, (str, Path)) or not str(root_raw).strip():
                raise ValueError(
                    "prep.musan_root is required when MUSAN mixing is enabled"
                )
            root = Path(root_raw).expanduser().resolve()
            if not root.is_dir():
                raise NotADirectoryError(f"MUSAN root not found: {root}")

        noise_files = (
            _discover_subset(root, "noise")
            if root is not None and (noise_enabled or burst_noise.enabled)
            else ()
        )
        music_files = (
            _discover_subset(root, "music")
            if root is not None and music_enabled
            else ()
        )
        speech_files = (
            _discover_subset(root, "speech")
            if root is not None and speech_enabled
            else ()
        )
        recipes = []
        for index, audio_path in enumerate(clean_paths):
            recipes.append(
                _MixRecipe(
                    noise=(
                        _component_recipe(
                            root=root,
                            files=noise_files,
                            kind="noise",
                            level_db=noise_snr_db,
                            base_seed=seed,
                            index=index,
                            audio_path=audio_path,
                        )
                        if noise_enabled
                        else None
                    ),
                    music=(
                        _component_recipe(
                            root=root,
                            files=music_files,
                            kind="music",
                            level_db=music_snr_db,
                            base_seed=seed,
                            index=index,
                            audio_path=audio_path,
                        )
                        if music_enabled
                        else None
                    ),
                    speech=(
                        _component_recipe(
                            root=root,
                            files=speech_files,
                            kind="speech",
                            level_db=speech_relative_db,
                            base_seed=seed,
                            index=index,
                            audio_path=audio_path,
                        )
                        if speech_enabled
                        else None
                    ),
                    stationary_noise_seed=(
                        _derive_seed(
                            seed,
                            index,
                            audio_path,
                            "stationary_noise",
                        )
                        if stationary_noise.enabled
                        else None
                    ),
                    burst_noise=(
                        _burst_noise_recipe(
                            root=root,
                            files=noise_files,
                            config=burst_noise,
                            base_seed=seed,
                            index=index,
                            audio_path=audio_path,
                        )
                        if burst_noise.enabled
                        else None
                    ),
                    volume_variation_seed=(
                        _derive_seed(
                            seed,
                            index,
                            audio_path,
                            "volume_variation",
                        )
                        if volume_variation.enabled
                        else None
                    ),
                )
            )

        return cls(
            seed=seed,
            musan_root=root,
            noise_enabled=noise_enabled,
            noise_snr_db=noise_snr_db,
            noise_files=noise_files,
            music_enabled=music_enabled,
            music_snr_db=music_snr_db,
            music_files=music_files,
            speech_enabled=speech_enabled,
            speech_relative_db=speech_relative_db,
            speech_files=speech_files,
            stationary_noise=stationary_noise,
            burst_noise=burst_noise,
            volume_variation=volume_variation,
            recipes=recipes,
        )

    @property
    def enabled(self) -> bool:
        return any(
            (
                self._noise_enabled,
                self._music_enabled,
                self._speech_enabled,
                self._stationary_noise.enabled,
                self._burst_noise.enabled,
                self._volume_variation.enabled,
            )
        )

    @property
    def additive_enabled(self) -> bool:
        return any(
            (
                self._noise_enabled,
                self._music_enabled,
                self._speech_enabled,
                self._stationary_noise.enabled,
                self._burst_noise.enabled,
            )
        )

    @property
    def pre_mix_enabled(self) -> bool:
        return self._volume_variation.enabled

    def summary(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "seed": self._seed,
            "musan_root": (
                str(self._musan_root) if self._musan_root is not None else None
            ),
            "noise": {
                "enabled": self._noise_enabled,
                "snr_db": self._noise_snr_db,
                "num_files": len(self._noise_files) if self._noise_enabled else 0,
            },
            "music": {
                "enabled": self._music_enabled,
                "snr_db": self._music_snr_db,
                "num_files": len(self._music_files),
            },
            "speech": {
                "enabled": self._speech_enabled,
                "relative_db": self._speech_relative_db,
                "num_files": len(self._speech_files),
            },
            "stationary_noise": {
                "enabled": self._stationary_noise.enabled,
                "kind": self._stationary_noise.kind,
                "snr_db": self._stationary_noise.snr_db,
            },
            "burst_noise": {
                "enabled": self._burst_noise.enabled,
                "snr_db": self._burst_noise.snr_db,
                "snr_scope": self._burst_noise.snr_scope,
                "event_count_min": self._burst_noise.event_count_min,
                "event_count_max": self._burst_noise.event_count_max,
                "duration_ms_min": self._burst_noise.duration_ms_min,
                "duration_ms_max": self._burst_noise.duration_ms_max,
                "fade_ms": self._burst_noise.fade_ms,
                "allow_overlap": self._burst_noise.allow_overlap,
                "min_gap_ms": self._burst_noise.min_gap_ms,
                "num_files": len(self._noise_files) if self._burst_noise.enabled else 0,
            },
            "volume_variation": {
                "enabled": self._volume_variation.enabled,
                "low_gain_db": self._volume_variation.low_gain_db,
                "high_gain_db": self._volume_variation.high_gain_db,
                "segment_ms_min": self._volume_variation.segment_ms_min,
                "segment_ms_max": self._volume_variation.segment_ms_max,
                "transition_ms": self._volume_variation.transition_ms,
            },
            "length_policy": "random_crop_or_repeat",
            "mix_policy": "scale_each_against_clean_rms_then_sum",
            "processing_order": (
                "volume_variation_then_scale_all_additive_components_against_"
                "the_same_varied_clean_then_sum_without_clipping"
            ),
        }

    def recipe_metadata(self, index: int) -> dict[str, Any]:
        recipe = self._recipe_at(index)
        result: dict[str, Any] = {"seed": self._seed, "row_index": index}
        if recipe.noise is not None:
            result["noise"] = recipe.noise.metadata()
        if recipe.music is not None:
            result["music"] = recipe.music.metadata()
        if recipe.speech is not None:
            result["speech"] = recipe.speech.metadata()
        if recipe.stationary_noise_seed is not None:
            result["stationary_noise"] = {
                "kind": self._stationary_noise.kind,
                "snr_db": self._stationary_noise.snr_db,
                "recipe_seed": recipe.stationary_noise_seed,
            }
        if recipe.burst_noise is not None:
            result["burst_noise"] = {
                "snr_db": self._burst_noise.snr_db,
                "snr_scope": self._burst_noise.snr_scope,
                "recipe_seed": recipe.burst_noise.recipe_seed,
                "event_count": len(recipe.burst_noise.events),
                "events": [event.metadata() for event in recipe.burst_noise.events],
            }
        if recipe.volume_variation_seed is not None:
            result["volume_variation"] = {
                "low_gain_db": self._volume_variation.low_gain_db,
                "high_gain_db": self._volume_variation.high_gain_db,
                "segment_ms_min": self._volume_variation.segment_ms_min,
                "segment_ms_max": self._volume_variation.segment_ms_max,
                "transition_ms": self._volume_variation.transition_ms,
                "recipe_seed": recipe.volume_variation_seed,
            }
        return result

    def apply_pre_mix(self, index: int, waveform, sample_rate: int):
        if not self.pre_mix_enabled:
            return waveform
        recipe = self._recipe_at(index)
        if recipe.volume_variation_seed is None:  # pragma: no cover
            raise RuntimeError("missing volume_variation recipe seed")
        if waveform.dim() != 2 or waveform.size(0) != 1:
            raise ValueError(
                "clean waveform must be mono with shape (1, samples), got "
                f"{tuple(waveform.shape)} for row {index}"
            )
        _validate_finite_waveform(waveform, field=f"clean waveform for row {index}")
        resolved_sample_rate = _sample_rate(sample_rate)
        target_samples = int(waveform.size(1))
        envelope = _volume_envelope(
            target_samples=target_samples,
            sample_rate=resolved_sample_rate,
            config=self._volume_variation,
            recipe_seed=recipe.volume_variation_seed,
        )
        torch = _import_torch()
        envelope_tensor = torch.from_numpy(envelope).to(
            device=waveform.device,
            dtype=waveform.dtype,
        )
        varied = waveform * envelope_tensor.unsqueeze(0)
        _validate_finite_waveform(
            varied,
            field=f"volume-varied clean waveform for row {index}",
        )
        return varied

    def apply_additive_delta(self, index: int, waveform, sample_rate: int):
        torch = _import_torch()
        if not self.additive_enabled:
            return torch.zeros_like(waveform)
        return self._mix_additives(index, waveform, sample_rate) - waveform

    def _mix_additives(self, index: int, waveform, sample_rate: int):
        torch = _import_torch()
        recipe = self._recipe_at(index)
        if waveform.dim() != 2 or waveform.size(0) != 1:
            raise ValueError(
                "clean waveform must be mono with shape (1, samples), got "
                f"{tuple(waveform.shape)} for row {index}"
            )
        resolved_sample_rate = _sample_rate(sample_rate)
        has_resolved_component = any(
            (
                recipe.noise is not None,
                recipe.music is not None,
                recipe.speech is not None,
                recipe.stationary_noise_seed is not None,
                recipe.burst_noise is not None and bool(recipe.burst_noise.events),
            )
        )
        if not has_resolved_component:
            _validate_finite_waveform(
                waveform,
                field=f"post-volume clean waveform for row {index}",
            )
            return waveform
        clean_rms = _waveform_rms(
            waveform,
            field=f"post-volume clean waveform for row {index}",
        )
        target_samples = int(waveform.size(1))
        components = []

        for component in (recipe.noise, recipe.music, recipe.speech):
            if component is None:
                continue
            source_waveform, source_rate = _load_source_audio(
                component,
                target_samples=target_samples,
                sample_rate=resolved_sample_rate,
            )
            if int(source_rate) != resolved_sample_rate:
                raise ValueError(
                    f"MUSAN {component.kind} loader returned sample rate "
                    f"{source_rate}, expected {resolved_sample_rate}: "
                    f"{component.source_path}"
                )
            source_waveform = source_waveform.to(
                device=waveform.device,
                dtype=waveform.dtype,
            )
            _validate_finite_waveform(
                source_waveform,
                field=f"MUSAN {component.kind} source {component.source_path}",
            )
            segment = _match_length(
                source_waveform,
                target_samples,
                component.offset_fraction,
                field=f"MUSAN {component.kind} source {component.source_path}",
            )
            source_rms = _waveform_rms(
                segment,
                field=f"MUSAN {component.kind} segment {component.source_path}",
            )
            target_rms = clean_rms * _target_amplitude_ratio(
                component.kind,
                component.level_db,
                field=f"MUSAN {component.kind} level",
            )
            components.append(segment * (target_rms / source_rms))

        if recipe.stationary_noise_seed is not None:
            noise_rng = np.random.default_rng(recipe.stationary_noise_seed)
            stationary = noise_rng.standard_normal(target_samples)
            if target_samples > 1:
                stationary = stationary - float(stationary.mean())
            stationary_tensor = torch.from_numpy(stationary).to(
                device=waveform.device,
                dtype=waveform.dtype,
            ).unsqueeze(0)
            source_rms = _waveform_rms(
                stationary_tensor,
                field=f"stationary noise for row {index}",
            )
            target_rms = clean_rms * _target_amplitude_ratio(
                "noise",
                self._stationary_noise.snr_db,
                field="stationary_noise.snr_db",
            )
            components.append(stationary_tensor * (target_rms / source_rms))

        if recipe.burst_noise is not None:
            components.append(
                self._burst_noise_delta(
                    index=index,
                    waveform=waveform,
                    clean_rms=clean_rms,
                    sample_rate=resolved_sample_rate,
                    recipe=recipe.burst_noise,
                )
            )

        mixed = waveform
        for component in components:
            mixed = mixed + component
        _validate_finite_waveform(mixed, field=f"mixed waveform for row {index}")
        return mixed

    def __call__(self, index: int, waveform, sample_rate: int):
        if not self.enabled:
            return waveform
        varied = self.apply_pre_mix(index, waveform, sample_rate)
        mixed = (
            self._mix_additives(index, varied, sample_rate)
            if self.additive_enabled
            else varied
        )
        _validate_finite_waveform(mixed, field=f"mixed waveform for row {index}")
        return mixed

    def _burst_noise_delta(
        self,
        *,
        index: int,
        waveform,
        clean_rms: float,
        sample_rate: int,
        recipe: _BurstNoiseRecipe,
    ):
        torch = _import_torch()
        target_samples = int(waveform.size(1))
        gap_samples = _milliseconds_to_samples(
            self._burst_noise.min_gap_ms,
            sample_rate,
            allow_zero=True,
        )
        configured_fade_samples = _milliseconds_to_samples(
            self._burst_noise.fade_ms,
            sample_rate,
            allow_zero=True,
        )
        placed: list[tuple[int, int]] = []
        raw_events = []
        for event in recipe.events:
            event_samples = min(
                target_samples,
                _milliseconds_to_samples(
                    event.duration_ms,
                    sample_rate,
                    allow_zero=False,
                ),
            )
            start = _burst_start(
                target_samples=target_samples,
                event_samples=event_samples,
                placement_fraction=event.placement_fraction,
                allow_overlap=self._burst_noise.allow_overlap,
                min_gap_samples=gap_samples,
                placed=placed,
            )
            end = start + event_samples
            placed.append((start, end))

            source_waveform, source_rate = _load_source_audio(
                event.component,
                target_samples=event_samples,
                sample_rate=sample_rate,
            )
            if int(source_rate) != sample_rate:
                raise ValueError(
                    "MUSAN burst noise loader returned sample rate "
                    f"{source_rate}, expected {sample_rate}: "
                    f"{event.component.source_path}"
                )
            source_waveform = source_waveform.to(
                device=waveform.device,
                dtype=waveform.dtype,
            )
            _validate_finite_waveform(
                source_waveform,
                field=f"MUSAN burst source {event.component.source_path}",
            )
            segment = _match_length(
                source_waveform,
                event_samples,
                event.component.offset_fraction,
                field=f"MUSAN burst source {event.component.source_path}",
            )
            fade = _fade_envelope(event_samples, configured_fade_samples)
            fade_tensor = torch.from_numpy(fade).to(
                device=waveform.device,
                dtype=waveform.dtype,
            ).unsqueeze(0)
            shaped = segment * fade_tensor
            shaped_rms = _waveform_rms(
                shaped,
                field=f"MUSAN burst event {event.event_index} after fade",
            )
            raw_events.append((start, end, shaped / shaped_rms))

        delta = torch.zeros_like(waveform)
        if not raw_events:
            return delta
        target_ratio = _target_amplitude_ratio(
            "noise",
            self._burst_noise.snr_db,
            field="burst_noise.snr_db",
        )
        if self._burst_noise.snr_scope == "active_event":
            for start, end, normalized_event in raw_events:
                delta[:, start:end] += normalized_event * (
                    clean_rms * target_ratio
                )
            return delta

        for start, end, normalized_event in raw_events:
            delta[:, start:end] += normalized_event
        burst_rms = _waveform_rms(
            delta,
            field=f"whole-clip burst noise for row {index}",
        )
        return delta * ((clean_rms * target_ratio) / burst_rms)

    def _recipe_at(self, index: int) -> _MixRecipe:
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError("MUSAN mix row index must be an integer")
        if index < 0 or index >= len(self._recipes):
            raise IndexError(f"MUSAN mix row index out of range: {index}")
        return self._recipes[index]


__all__ = ["MusanWaveformMixer"]
