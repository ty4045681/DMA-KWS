"""Deterministic in-memory waveform augmentations for Stage II evaluation.

The implementations in this module are compatible with the nine methods used
by the external ``audio_aug`` recipes, but are native DMA-KWS code. Recipes are
derived deterministically for each row, and the transform object contains only
pickle-friendly values.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Any, Mapping, Sequence

import numpy as np


AUDIO_AUG_COMPAT_REVISION = "58410f27c5beecdb9fa438e98833daccb69db895"

PRE_MIX_METHODS = ("speed_change", "volume_gain")
ADDITIVE_METHODS = ("noise_mix",)
POST_MIX_METHODS = (
    "amp_distortion",
    "subband_eq",
    "band_limit",
    "narrowband",
    "spectral_mask",
    "signal_mimic",
)
SUPPORTED_METHODS = PRE_MIX_METHODS + ADDITIVE_METHODS + POST_MIX_METHODS

_STOCHASTIC_METHODS = {
    "noise_mix",
    "subband_eq",
    "spectral_mask",
    "amp_distortion",
    "signal_mimic",
}
_SIGNAL_MIMIC_OVERLAPS = {
    "subband_eq",
    "band_limit",
    "narrowband",
    "spectral_mask",
}
_TOP_LEVEL_KEYS = {
    "seed",
    "speed_length_policy",
    "pcm_policy",
    "allow_signal_mimic_overlap",
    "transforms",
}
_SPEED_LENGTH_POLICIES = {"variable", "center_crop_or_zero_pad"}
_PCM_POLICIES = {"clip_round_each_stage", "float_unclipped"}
_PCM16_NEGATIVE_LIMIT = -32768.0
_PCM16_POSITIVE_LIMIT = 32767.0
_STAGE2_PCM_SCALE = 32768.0
_UPSTREAM_PCM_SCALE = 32767.0

_DEFAULT_PARAMS: dict[str, dict[str, Any]] = {
    "volume_gain": {"gain_db": 3.0},
    "speed_change": {"speed_factor": 1.05},
    "noise_mix": {"snr_db": 30.0, "snr_mode": "exact_rms"},
    "subband_eq": {
        "low_min_gain_db": -3.0,
        "high_min_gain_db": -5.0,
    },
    "band_limit": {
        "mode": "iir",
        "cutoff_hz": 6500.0,
        "filter_order": 4,
        "target_sample_rate": 8000,
    },
    "narrowband": {"target_sample_rate": 12000},
    "spectral_mask": {
        "frequency_masks": 1,
        "time_masks": 0,
        "min_gain": 0.6,
        "max_gain": 0.9,
    },
    "amp_distortion": {
        "distortion_type": "gain_db",
        "rate": 0.25,
        "gain_db": 3.0,
        "max_db": -0.03,
        "mask_number": 4,
        "a": 1.0,
        "m": 1,
        "n": 1,
    },
    "signal_mimic": {
        "subband_probability": 0.6,
        "mute_probability": 0.0,
        "band_limit_probability": 0.0,
        "spectral_mask_probability": 0.0,
        "narrowband_probability": 0.0,
    },
}


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


def _finite_float(value: object, *, field: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{field} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{field} must be a finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{field} must be a finite number")
    return result


def _bounded_float(
    value: object,
    *,
    field: str,
    minimum: float,
    maximum: float,
    minimum_inclusive: bool = True,
) -> float:
    result = _finite_float(value, field=field)
    below = result < minimum if minimum_inclusive else result <= minimum
    if below or result > maximum:
        left = "[" if minimum_inclusive else "("
        raise ValueError(
            f"{field} must be in {left}{minimum:g}, {maximum:g}], got {result:g}"
        )
    return result


def _bounded_int(
    value: object,
    *,
    field: str,
    minimum: int,
    maximum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field} must be an integer")
    result = int(value)
    if result < minimum or (maximum is not None and result > maximum):
        suffix = f" and <= {maximum}" if maximum is not None else ""
        raise ValueError(f"{field} must be >= {minimum}{suffix}, got {result}")
    return result


def _db_to_amplitude(db: float) -> float:
    try:
        result = 10.0 ** (float(db) / 20.0)
    except OverflowError as exc:
        raise ValueError(f"dB value is outside the supported numeric range: {db}") from exc
    if result <= 0.0 or not math.isfinite(result):
        raise ValueError(f"dB value is outside the supported numeric range: {db}")
    return result


def _derive_seed(base_seed: int, index: int, audio_path: str, method: str) -> int:
    payload = f"{base_seed}\0{index}\0{audio_path}\0audio_aug:{method}".encode(
        "utf-8"
    )
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _phase_for(method: str) -> str:
    if method in PRE_MIX_METHODS:
        return "pre_mix"
    if method in ADDITIVE_METHODS:
        return "additive"
    if method in POST_MIX_METHODS:
        return "post_mix"
    raise ValueError(f"Unsupported audio_aug method: {method}")


def _validate_method_params(
    method: str,
    section: Mapping[str, Any],
) -> tuple[bool, dict[str, Any]]:
    prefix = f"prep.audio_aug.transforms.{method}"
    defaults = _DEFAULT_PARAMS[method]
    _reject_unknown_keys(section, {"enabled", *defaults}, field=prefix)
    enabled = _require_bool(section.get("enabled", False), field=f"{prefix}.enabled")

    if method == "volume_gain":
        gain_db = _bounded_float(
            section.get("gain_db", defaults["gain_db"]),
            field=f"{prefix}.gain_db",
            minimum=-30.0,
            maximum=30.0,
        )
        if gain_db == 0.0:
            raise ValueError(f"{prefix}.gain_db must not be 0")
        return enabled, {"gain_db": gain_db}

    if method == "speed_change":
        factor = _bounded_float(
            section.get("speed_factor", defaults["speed_factor"]),
            field=f"{prefix}.speed_factor",
            minimum=0.5,
            maximum=2.0,
        )
        if factor == 1.0:
            raise ValueError(f"{prefix}.speed_factor must not be 1")
        return enabled, {"speed_factor": factor}

    if method == "noise_mix":
        snr_mode = str(section.get("snr_mode", defaults["snr_mode"]))
        if snr_mode not in {"exact_rms", "upstream_std"}:
            raise ValueError(
                f"{prefix}.snr_mode must be exact_rms or upstream_std"
            )
        return enabled, {
            "snr_db": _bounded_float(
                section.get("snr_db", defaults["snr_db"]),
                field=f"{prefix}.snr_db",
                minimum=-5.0,
                maximum=40.0,
            ),
            "snr_mode": snr_mode,
        }

    if method == "subband_eq":
        low = _bounded_float(
            section.get("low_min_gain_db", defaults["low_min_gain_db"]),
            field=f"{prefix}.low_min_gain_db",
            minimum=-60.0,
            maximum=0.0,
        )
        high = _bounded_float(
            section.get("high_min_gain_db", defaults["high_min_gain_db"]),
            field=f"{prefix}.high_min_gain_db",
            minimum=-60.0,
            maximum=0.0,
        )
        if high > low:
            raise ValueError(
                f"{prefix}.high_min_gain_db must be <= low_min_gain_db"
            )
        return enabled, {
            "low_min_gain_db": low,
            "high_min_gain_db": high,
        }

    if method == "band_limit":
        mode = str(section.get("mode", defaults["mode"]))
        if mode not in {"freq", "iir", "resample"}:
            raise ValueError(f"{prefix}.mode must be freq, iir, or resample")
        return enabled, {
            "mode": mode,
            "cutoff_hz": _bounded_float(
                section.get("cutoff_hz", defaults["cutoff_hz"]),
                field=f"{prefix}.cutoff_hz",
                minimum=20.0,
                maximum=192000.0,
            ),
            "filter_order": _bounded_int(
                section.get("filter_order", defaults["filter_order"]),
                field=f"{prefix}.filter_order",
                minimum=1,
                maximum=12,
            ),
            "target_sample_rate": _bounded_int(
                section.get(
                    "target_sample_rate", defaults["target_sample_rate"]
                ),
                field=f"{prefix}.target_sample_rate",
                minimum=1000,
            ),
        }

    if method == "narrowband":
        return enabled, {
            "target_sample_rate": _bounded_int(
                section.get(
                    "target_sample_rate", defaults["target_sample_rate"]
                ),
                field=f"{prefix}.target_sample_rate",
                minimum=1000,
            )
        }

    if method == "spectral_mask":
        frequency_masks = _bounded_int(
            section.get("frequency_masks", defaults["frequency_masks"]),
            field=f"{prefix}.frequency_masks",
            minimum=1,
            maximum=8,
        )
        time_masks = _bounded_int(
            section.get("time_masks", defaults["time_masks"]),
            field=f"{prefix}.time_masks",
            minimum=0,
            maximum=8,
        )
        min_gain = _bounded_float(
            section.get("min_gain", defaults["min_gain"]),
            field=f"{prefix}.min_gain",
            minimum=0.0,
            maximum=1.0,
        )
        max_gain = _bounded_float(
            section.get("max_gain", defaults["max_gain"]),
            field=f"{prefix}.max_gain",
            minimum=min_gain,
            maximum=1.0,
        )
        return enabled, {
            "frequency_masks": frequency_masks,
            "time_masks": time_masks,
            "min_gain": min_gain,
            "max_gain": max_gain,
        }

    if method == "amp_distortion":
        distortion_type = str(
            section.get("distortion_type", defaults["distortion_type"])
        )
        allowed_types = {
            "gain_db",
            "max_distortion",
            "fence_distortion",
            "jag_distortion",
            "poly_distortion",
            "quad_distortion",
        }
        if distortion_type not in allowed_types:
            raise ValueError(
                f"{prefix}.distortion_type must be one of: "
                + ", ".join(sorted(allowed_types))
            )
        return enabled, {
            "distortion_type": distortion_type,
            "rate": _bounded_float(
                section.get("rate", defaults["rate"]),
                field=f"{prefix}.rate",
                minimum=0.0,
                maximum=1.0,
                minimum_inclusive=False,
            ),
            "gain_db": _bounded_float(
                section.get("gain_db", defaults["gain_db"]),
                field=f"{prefix}.gain_db",
                minimum=-30.0,
                maximum=30.0,
            ),
            "max_db": _bounded_float(
                section.get("max_db", defaults["max_db"]),
                field=f"{prefix}.max_db",
                minimum=-120.0,
                maximum=0.0,
            ),
            "mask_number": _bounded_int(
                section.get("mask_number", defaults["mask_number"]),
                field=f"{prefix}.mask_number",
                minimum=0,
                maximum=12,
            ),
            "a": _bounded_float(
                section.get("a", defaults["a"]),
                field=f"{prefix}.a",
                minimum=-10.0,
                maximum=10.0,
            ),
            "m": _bounded_int(
                section.get("m", defaults["m"]),
                field=f"{prefix}.m",
                minimum=1,
                maximum=8,
            ),
            "n": _bounded_int(
                section.get("n", defaults["n"]),
                field=f"{prefix}.n",
                minimum=1,
                maximum=8,
            ),
        }

    if method == "signal_mimic":
        result = {}
        for name, default in defaults.items():
            result[name] = _bounded_float(
                section.get(name, default),
                field=f"{prefix}.{name}",
                minimum=0.0,
                maximum=1.0,
            )
        return enabled, result

    raise ValueError(f"Unsupported audio_aug method: {method}")


_SCIPY_SIGNAL = None
_SCIPY_VERSION: str | None = None


def _require_scipy():
    global _SCIPY_SIGNAL, _SCIPY_VERSION
    if _SCIPY_SIGNAL is not None:
        return _SCIPY_SIGNAL
    try:
        import scipy
        from scipy import signal
    except ImportError as exc:
        raise ImportError(
            "Enabled audio_aug transform requires scipy>=1.15; install the "
            "DMA-KWS audio augmentation dependencies"
        ) from exc
    _SCIPY_SIGNAL = signal
    _SCIPY_VERSION = str(scipy.__version__)
    return signal


def _method_requires_scipy(method: str, params: Mapping[str, Any]) -> bool:
    if method in {"subband_eq", "narrowband", "spectral_mask", "signal_mimic"}:
        return True
    return method == "band_limit" and params["mode"] in {"iir", "resample"}


@dataclass(frozen=True)
class _StageRecipe:
    name: str
    phase: str
    recipe_seed: int


@dataclass(frozen=True)
class _RowRecipe:
    stages: tuple[_StageRecipe, ...]


@dataclass(frozen=True)
class _SignalMimicStep:
    name: str
    values_items: tuple[tuple[str, int], ...] = ()

    def values(self) -> dict[str, int]:
        return dict(self.values_items)


def _signal_mimic_plan(
    params: Mapping[str, Any],
    *,
    recipe_seed: int,
    sample_rate: int,
    frame_count: int,
) -> tuple[_SignalMimicStep, ...]:
    """Resolve the pinned upstream's single-RNG schedule without processing audio."""

    rng = np.random.default_rng(recipe_seed)
    steps: list[_SignalMimicStep] = []

    def child_seed() -> int:
        return int(rng.integers(0, (2**31) - 1))

    if float(rng.random()) < float(params["subband_probability"]):
        steps.append(
            _SignalMimicStep(
                "subband_eq",
                (("seed", child_seed()),),
            )
        )

    if frame_count > 0 and float(rng.random()) < float(params["mute_probability"]):
        start = int(rng.integers(0, max(1, frame_count // 2)))
        length = int(
            rng.integers(
                max(1, frame_count // 20),
                max(2, frame_count // 4),
            )
        )
        steps.append(
            _SignalMimicStep(
                "mute",
                (("start", start), ("length", length)),
            )
        )

    if float(rng.random()) < float(params["band_limit_probability"]):
        cutoff_upper = int(min((sample_rate / 2.0) - 1.0, 4500.0))
        if cutoff_upper >= 20:
            cutoff_lower = min(3000, max(20, int(cutoff_upper * 0.5)))
            cutoff = int(rng.integers(cutoff_lower, cutoff_upper + 1))
            steps.append(
                _SignalMimicStep(
                    "band_limit",
                    (("cutoff_hz", cutoff),),
                )
            )

    if float(rng.random()) < float(params["spectral_mask_probability"]):
        steps.append(
            _SignalMimicStep(
                "spectral_mask",
                (("seed", child_seed()),),
            )
        )

    if (
        float(rng.random()) < float(params["narrowband_probability"])
        and sample_rate > 8000
    ):
        steps.append(_SignalMimicStep("narrowband"))

    if not steps:
        steps.append(
            _SignalMimicStep(
                "subband_eq",
                (("seed", child_seed()),),
            )
        )
    return tuple(steps)


def _import_torch():
    try:
        import torch
    except ImportError as exc:
        raise SystemExit(
            "Missing torch. Install CUDA PyTorch on the evaluation machine first."
        ) from exc
    return torch


def _validate_waveform(waveform, sample_rate: int, *, field: str) -> None:
    torch = _import_torch()
    if not isinstance(waveform, torch.Tensor):
        raise TypeError(f"{field} must be a torch.Tensor")
    if waveform.dim() != 2 or waveform.size(0) != 1:
        raise ValueError(
            f"{field} must be mono with shape (1, samples), got {tuple(waveform.shape)}"
        )
    if waveform.numel() == 0:
        raise ValueError(f"{field} is empty")
    if not waveform.is_floating_point():
        raise TypeError(f"{field} must use a floating-point dtype")
    if not bool(torch.isfinite(waveform).all().item()):
        raise ValueError(f"{field} contains non-finite samples")
    if isinstance(sample_rate, bool) or not isinstance(sample_rate, int):
        raise TypeError("sample_rate must be a positive integer")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be a positive integer")


def _to_numpy(waveform) -> np.ndarray:
    return (
        waveform.detach()
        .to(device="cpu")
        .to(dtype=_import_torch().float64)
        .transpose(0, 1)
        .contiguous()
        .numpy()
        .copy()
    )


def _from_numpy(samples: np.ndarray, *, like):
    torch = _import_torch()
    contiguous = np.ascontiguousarray(samples.T, dtype=np.float64)
    return torch.from_numpy(contiguous).to(device=like.device, dtype=like.dtype)


def _apply_pcm_policy(samples: np.ndarray, policy: str) -> np.ndarray:
    """Quantize upstream-normalized samples and return Stage II decode scaling."""

    if policy == "float_unclipped":
        return np.asarray(samples, dtype=np.float64)
    pcm = np.clip(
        np.rint(np.asarray(samples) * _UPSTREAM_PCM_SCALE),
        _PCM16_NEGATIVE_LIMIT,
        _PCM16_POSITIVE_LIMIT,
    )
    return np.asarray(pcm / _STAGE2_PCM_SCALE, dtype=np.float64)


def _to_upstream_pcm_scale(samples: np.ndarray, policy: str) -> np.ndarray:
    """Map Stage II's q/32768 decode convention to upstream's q/32767."""

    if policy == "float_unclipped":
        return samples
    return np.asarray(
        samples * (_STAGE2_PCM_SCALE / _UPSTREAM_PCM_SCALE),
        dtype=np.float64,
    )


def _match_center_zero(samples: np.ndarray, frame_count: int) -> np.ndarray:
    current = int(samples.shape[0])
    if current == frame_count:
        return samples
    if current > frame_count:
        start = (current - frame_count) // 2
        return samples[start : start + frame_count]
    missing = frame_count - current
    left = missing // 2
    return np.pad(samples, ((left, missing - left), (0, 0)), mode="constant")


def _match_head_edge(samples: np.ndarray, frame_count: int) -> np.ndarray:
    current = int(samples.shape[0])
    if current >= frame_count:
        return samples[:frame_count]
    if current == 0:
        return np.zeros((frame_count, samples.shape[1]), dtype=np.float64)
    return np.pad(samples, ((0, frame_count - current), (0, 0)), mode="edge")


def _stft_channel(channel: np.ndarray, sample_rate: int):
    signal = _require_scipy()
    frame_count = int(channel.shape[0])
    nperseg = min(480, frame_count)
    hop = min(160, max(1, nperseg // 2))
    noverlap = max(0, nperseg - hop)
    nfft = max(512, nperseg)
    frequencies, times, spectrum = signal.stft(
        channel,
        fs=sample_rate,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft,
        boundary="zeros",
        padded=True,
    )
    return frequencies, times, spectrum, (nfft, nperseg, noverlap)


def _istft_channel(
    spectrum: np.ndarray,
    sample_rate: int,
    settings: tuple[int, int, int],
    frame_count: int,
) -> np.ndarray:
    signal = _require_scipy()
    nfft, nperseg, noverlap = settings
    _, samples = signal.istft(
        spectrum,
        fs=sample_rate,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft,
        input_onesided=True,
        boundary=True,
    )
    samples = np.asarray(samples, dtype=np.float64).reshape(-1, 1)
    return _match_head_edge(samples, frame_count)[:, 0]


def _roundtrip_resample(
    samples: np.ndarray,
    source_rate: int,
    target_rate: int,
) -> np.ndarray:
    signal = _require_scipy()
    common = math.gcd(source_rate, target_rate)
    down = signal.resample_poly(
        samples,
        target_rate // common,
        source_rate // common,
        axis=0,
    )
    restored = signal.resample_poly(
        down,
        source_rate // common,
        target_rate // common,
        axis=0,
    )
    return _match_head_edge(np.asarray(restored, dtype=np.float64), samples.shape[0])


def _volume_gain(samples: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
    return samples * _db_to_amplitude(float(params["gain_db"]))


def _speed_change(samples: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
    factor = float(params["speed_factor"])
    frame_count = int(samples.shape[0])
    output_count = max(1, int(round(frame_count / factor)))
    source_positions = np.arange(frame_count, dtype=np.float64)
    output_positions = np.minimum(
        np.arange(output_count, dtype=np.float64) * factor,
        frame_count - 1,
    )
    channels = [
        np.interp(output_positions, source_positions, samples[:, channel])
        for channel in range(samples.shape[1])
    ]
    return np.column_stack(channels)


def _noise_mix(samples: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
    rng = np.random.default_rng(int(params["seed"]))
    signal_rms = float(np.sqrt(np.mean(np.square(samples))))
    # Match the external recipe's silent-input fallback in normalized PCM units.
    noise_rms = (
        signal_rms / _db_to_amplitude(float(params["snr_db"]))
        if signal_rms > 0.0
        else 100.0 / 32767.0
    )
    if params["snr_mode"] == "upstream_std":
        return samples + rng.normal(0.0, noise_rms, size=samples.shape)

    noise = rng.normal(0.0, 1.0, size=samples.shape)
    realized_rms = float(np.sqrt(np.mean(np.square(noise))))
    if realized_rms <= 0.0 or not math.isfinite(realized_rms):  # pragma: no cover
        raise RuntimeError("noise_mix failed to generate finite non-silent noise")
    return samples + noise * (noise_rms / realized_rms)


def _subband_boundaries(
    rng: np.random.Generator,
    bin_count: int,
) -> tuple[int, ...]:
    if bin_count < 1:
        raise ValueError("subband bin_count must be positive")
    candidates = [0, 1]
    for position in range(2, 11):
        if position < 4:
            step = int(rng.integers(1, 3))
        elif position < 7:
            step = int(rng.integers(8, 10))
        elif position < 10:
            step = int(rng.integers(32, 64))
        else:
            step = bin_count
        candidates.append(min(bin_count, candidates[-1] + step))
    candidates[-1] = bin_count

    boundaries = []
    for candidate in candidates:
        if not boundaries or candidate > boundaries[-1]:
            boundaries.append(candidate)
    if boundaries[-1] != bin_count:
        boundaries.append(bin_count)
    return tuple(boundaries)


def _subband_gain_floor_db(
    band_index: int,
    boundaries: Sequence[int],
    params: Mapping[str, Any],
) -> float:
    if band_index >= len(boundaries) - 4:
        return float(params["high_min_gain_db"])
    return float(params["low_min_gain_db"])


def _subband_eq(samples: np.ndarray, sample_rate: int, params: Mapping[str, Any]) -> np.ndarray:
    rng = np.random.default_rng(int(params["seed"]))
    output = []
    for channel_index in range(samples.shape[1]):
        frequencies, _, spectrum, settings = _stft_channel(
            samples[:, channel_index], sample_rate
        )
        bin_count = len(frequencies)
        boundaries = _subband_boundaries(rng, bin_count)
        gains = np.ones(bin_count, dtype=np.float64)
        for band_index, (start, end) in enumerate(
            zip(boundaries, boundaries[1:])
        ):
            minimum = _subband_gain_floor_db(
                band_index,
                boundaries,
                params,
            )
            gains[start:end] = _db_to_amplitude(float(rng.uniform(minimum, 0.0)))
        output.append(
            _istft_channel(
                spectrum * gains[:, None],
                sample_rate,
                settings,
                samples.shape[0],
            )
        )
    return np.column_stack(output)


def _band_limit(samples: np.ndarray, sample_rate: int, params: Mapping[str, Any]) -> np.ndarray:
    mode = str(params["mode"])
    cutoff = float(params["cutoff_hz"])
    nyquist = sample_rate / 2.0
    if not 20.0 <= cutoff < nyquist:
        raise ValueError(
            f"band_limit.cutoff_hz must be in [20, {nyquist:g}) for sample_rate={sample_rate}"
        )
    if mode == "freq":
        spectrum = np.fft.rfft(samples, axis=0)
        frequencies = np.fft.rfftfreq(samples.shape[0], d=1.0 / sample_rate)
        spectrum[frequencies > cutoff, :] = 0.0
        return np.fft.irfft(spectrum, n=samples.shape[0], axis=0)
    if mode == "iir":
        signal = _require_scipy()
        sos = signal.butter(
            int(params["filter_order"]),
            cutoff,
            btype="lowpass",
            fs=sample_rate,
            output="sos",
        )
        output = []
        for channel_index in range(samples.shape[1]):
            channel = samples[:, channel_index]
            if channel.shape[0] > int(params["filter_order"]) * 6:
                filtered = signal.sosfiltfilt(sos, channel)
            else:
                filtered = signal.sosfilt(sos, channel)
            output.append(np.asarray(filtered, dtype=np.float64))
        return np.column_stack(output)
    if mode == "resample":
        target_rate = int(params["target_sample_rate"])
        if not 1000 <= target_rate < sample_rate:
            raise ValueError(
                "band_limit.target_sample_rate must be >= 1000 and lower than "
                f"sample_rate={sample_rate}"
            )
        return _roundtrip_resample(samples, sample_rate, target_rate)
    raise ValueError(f"Unsupported band_limit mode: {mode}")


def _narrowband(samples: np.ndarray, sample_rate: int, params: Mapping[str, Any]) -> np.ndarray:
    target_rate = int(params["target_sample_rate"])
    if not 1000 <= target_rate < sample_rate:
        raise ValueError(
            "narrowband.target_sample_rate must be >= 1000 and lower than "
            f"sample_rate={sample_rate}"
        )
    return _roundtrip_resample(samples, sample_rate, target_rate)


def _spectral_frequency_mask_width(
    rng: np.random.Generator,
    bin_count: int,
) -> int:
    return int(
        rng.integers(
            max(1, bin_count // 24),
            max(2, bin_count // 5),
        )
    )


def _spectral_mask(samples: np.ndarray, sample_rate: int, params: Mapping[str, Any]) -> np.ndarray:
    rng = np.random.default_rng(int(params["seed"]))
    output = []
    for channel_index in range(samples.shape[1]):
        frequencies, times, spectrum, settings = _stft_channel(
            samples[:, channel_index], sample_rate
        )
        mask = np.ones(spectrum.shape, dtype=np.float64)
        for _ in range(int(params["frequency_masks"])):
            width = _spectral_frequency_mask_width(rng, len(frequencies))
            width = min(width, len(frequencies))
            start = int(rng.integers(0, max(1, len(frequencies) - width + 1)))
            mask[start : start + width, :] *= float(
                rng.uniform(float(params["min_gain"]), float(params["max_gain"]))
            )
        for _ in range(int(params["time_masks"])):
            maximum_width = max(2, len(times) // 4 + 1)
            width = min(len(times), int(rng.integers(1, maximum_width)))
            start = int(rng.integers(0, max(1, len(times) - width + 1)))
            mask[:, start : start + width] *= float(
                rng.uniform(float(params["min_gain"]), float(params["max_gain"]))
            )
        output.append(
            _istft_channel(
                spectrum * mask,
                sample_rate,
                settings,
                samples.shape[0],
            )
        )
    return np.column_stack(output)


def _amplitude_masks(
    rng: np.random.Generator,
    count: int,
) -> tuple[tuple[float, float], ...]:
    if count == 0:
        db_ranges = (
            (-110.0, -95.0),
            (-90.0, -80.0),
            (-65.0, -60.0),
            (-50.0, -30.0),
            (-15.0, 0.0),
        )
    else:
        increments = rng.uniform(0.5, 1.0, size=(2 * count) - 1)
        positions = np.concatenate(([0.0], np.cumsum(increments)))
        scale = float(positions[-1])
        db_ranges = tuple(
            (
                ((float(positions[2 * index]) - scale) / scale) * 100.0,
                ((float(positions[(2 * index) + 1]) - scale) / scale) * 100.0,
            )
            for index in range(count)
        )
    return tuple(
        (_db_to_amplitude(left), _db_to_amplitude(right))
        for left, right in db_ranges
    )


def _inside_masks(values: np.ndarray, masks: Sequence[tuple[float, float]]) -> np.ndarray:
    result = np.zeros(values.shape, dtype=bool)
    for lower, upper in masks:
        result |= (values >= lower) & (values <= upper)
    return result


def _polynomial_distortion(
    normalized: np.ndarray,
    *,
    a: float,
    m: int,
    n: int,
) -> np.ndarray:
    absolute = np.abs(normalized)
    db_position = np.clip(
        (20.0 * np.log10(np.maximum(absolute, 1.0e-12)) / 100.0) + 1.0,
        0.0,
        1.0,
    )
    shaped = np.clip(
        db_position
        + a * np.power(db_position, m) * np.power(1.0 - db_position, n),
        0.0,
        1.0,
    )
    amplitude = np.minimum(
        0.9997,
        np.power(10.0, ((shaped - 1.0) * 100.0) / 20.0),
    )
    return np.where(absolute < 1.0e-6, 0.0, np.sign(normalized) * amplitude)


def _amp_distortion(samples: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
    rng = np.random.default_rng(int(params["seed"]))
    normalized = np.clip(samples, -1.0, 1.0)
    mutate = rng.random(normalized.shape) < float(params["rate"])
    if normalized.size and not bool(mutate.any()):
        non_silent = np.flatnonzero(np.abs(normalized.reshape(-1)) > 1.0e-12)
        candidates = non_silent if non_silent.size else np.arange(normalized.size)
        mutate.reshape(-1)[int(rng.choice(candidates))] = True

    distortion_type = str(params["distortion_type"])
    if distortion_type == "gain_db":
        transformed = np.clip(
            normalized * _db_to_amplitude(float(params["gain_db"])),
            -0.997,
            0.997,
        )
    elif distortion_type == "max_distortion":
        maximum = min(0.997, _db_to_amplitude(float(params["max_db"])))
        transformed = np.sign(normalized) * maximum
    elif distortion_type in {"fence_distortion", "jag_distortion"}:
        count = int(params["mask_number"])
        positive = _inside_masks(np.abs(normalized), _amplitude_masks(rng, count))
        negative = _inside_masks(np.abs(normalized), _amplitude_masks(rng, count))
        included = np.where(normalized >= 0.0, positive, negative)
        if distortion_type == "fence_distortion":
            maximum = min(0.997, _db_to_amplitude(float(params["max_db"])))
            transformed = np.where(
                included, np.sign(normalized) * maximum, 0.0
            )
        else:
            transformed = np.where(included, normalized, 0.0)
    else:
        if distortion_type == "quad_distortion":
            a, m, n = 1.0, 1, 1
        else:
            a, m, n = float(params["a"]), int(params["m"]), int(params["n"])
        transformed = _polynomial_distortion(normalized, a=a, m=m, n=n)
    return np.where(mutate, transformed, normalized)


def _signal_mimic(
    samples: np.ndarray,
    sample_rate: int,
    params: Mapping[str, Any],
) -> np.ndarray:
    seed = int(params["seed"])
    output = samples
    for step in _signal_mimic_plan(
        params,
        recipe_seed=seed,
        sample_rate=sample_rate,
        frame_count=samples.shape[0],
    ):
        values = step.values()
        if step.name == "subband_eq":
            output = _subband_eq(
                output,
                sample_rate,
                {
                    "low_min_gain_db": -10.0,
                    "high_min_gain_db": -20.0,
                    "seed": values["seed"],
                },
            )
        elif step.name == "mute":
            output = output.copy()
            start = values["start"]
            end = min(output.shape[0], start + values["length"])
            output[start:end, :] = 0.0
        elif step.name == "band_limit":
            output = _band_limit(
                output,
                sample_rate,
                {
                    "mode": "freq",
                    "cutoff_hz": float(values["cutoff_hz"]),
                    "filter_order": 6,
                    "target_sample_rate": max(
                        1000,
                        int(values["cutoff_hz"] * 2),
                    ),
                },
            )
        elif step.name == "spectral_mask":
            output = _spectral_mask(
                output,
                sample_rate,
                {
                    "frequency_masks": 2,
                    "time_masks": 1,
                    "min_gain": 0.05,
                    "max_gain": 0.6,
                    "seed": values["seed"],
                },
            )
        elif step.name == "narrowband":
            output = _narrowband(
                output,
                sample_rate,
                {"target_sample_rate": 8000},
            )
        else:  # pragma: no cover - guarded by the fixed plan names
            raise RuntimeError(f"Unknown signal_mimic child: {step.name}")
    return output


def _apply_numpy_method(
    method: str,
    samples: np.ndarray,
    sample_rate: int,
    params: Mapping[str, Any],
) -> np.ndarray:
    if method == "volume_gain":
        return _volume_gain(samples, params)
    if method == "speed_change":
        return _speed_change(samples, params)
    if method == "noise_mix":
        return _noise_mix(samples, params)
    if method == "subband_eq":
        return _subband_eq(samples, sample_rate, params)
    if method == "band_limit":
        return _band_limit(samples, sample_rate, params)
    if method == "narrowband":
        return _narrowband(samples, sample_rate, params)
    if method == "spectral_mask":
        return _spectral_mask(samples, sample_rate, params)
    if method == "amp_distortion":
        return _amp_distortion(samples, params)
    if method == "signal_mimic":
        return _signal_mimic(samples, sample_rate, params)
    raise ValueError(f"Unsupported audio_aug method: {method}")


class AudioAugWaveformTransform:
    """Apply deterministic per-row audio augmentations in three mix phases."""

    def __init__(
        self,
        *,
        seed: int,
        speed_length_policy: str,
        pcm_policy: str,
        allow_signal_mimic_overlap: bool,
        scipy_version: str | None,
        method_configs: Mapping[str, Mapping[str, Any]],
        enabled_methods: Sequence[str],
        audio_paths: Sequence[str],
    ) -> None:
        self._seed = int(seed)
        self._speed_length_policy = str(speed_length_policy)
        self._pcm_policy = str(pcm_policy)
        self._allow_signal_mimic_overlap = bool(allow_signal_mimic_overlap)
        self._scipy_version = scipy_version
        self._method_configs = {
            method: dict(method_configs[method]) for method in SUPPORTED_METHODS
        }
        self._enabled_methods = tuple(enabled_methods)
        self._audio_paths = tuple(audio_paths)

    @classmethod
    def from_prep(
        cls,
        prep: Mapping[str, Any],
        *,
        audio_paths: Sequence[object],
    ) -> "AudioAugWaveformTransform":
        prep = _require_mapping(prep, field="prep")
        raw_audio_aug = prep.get("audio_aug", {})
        if raw_audio_aug is None:
            raw_audio_aug = {}
        audio_aug = _require_mapping(raw_audio_aug, field="prep.audio_aug")
        _reject_unknown_keys(audio_aug, _TOP_LEVEL_KEYS, field="prep.audio_aug")

        raw_seed = audio_aug.get("seed", 2025)
        if isinstance(raw_seed, bool) or not isinstance(raw_seed, int):
            raise TypeError("prep.audio_aug.seed must be a non-negative integer")
        if raw_seed < 0:
            raise ValueError("prep.audio_aug.seed must be a non-negative integer")
        seed = int(raw_seed)

        speed_length_policy = str(
            audio_aug.get("speed_length_policy", "variable")
        )
        if speed_length_policy not in _SPEED_LENGTH_POLICIES:
            raise ValueError(
                "prep.audio_aug.speed_length_policy must be variable or "
                "center_crop_or_zero_pad"
            )
        pcm_policy = str(audio_aug.get("pcm_policy", "clip_round_each_stage"))
        if pcm_policy not in _PCM_POLICIES:
            raise ValueError(
                "prep.audio_aug.pcm_policy must be clip_round_each_stage or "
                "float_unclipped"
            )
        allow_overlap = _require_bool(
            audio_aug.get("allow_signal_mimic_overlap", False),
            field="prep.audio_aug.allow_signal_mimic_overlap",
        )

        raw_transforms = _require_mapping(
            audio_aug.get("transforms", {}),
            field="prep.audio_aug.transforms",
        )
        _reject_unknown_keys(
            raw_transforms,
            set(SUPPORTED_METHODS),
            field="prep.audio_aug.transforms",
        )
        method_configs: dict[str, dict[str, Any]] = {}
        enabled_methods = []
        for method in SUPPORTED_METHODS:
            section = _require_mapping(
                raw_transforms.get(method, {}),
                field=f"prep.audio_aug.transforms.{method}",
            )
            method_enabled, params = _validate_method_params(method, section)
            method_configs[method] = params
            if method_enabled:
                enabled_methods.append(method)

        if "signal_mimic" in enabled_methods and not allow_overlap:
            overlaps = sorted(_SIGNAL_MIMIC_OVERLAPS.intersection(enabled_methods))
            if overlaps:
                raise ValueError(
                    "prep.audio_aug.transforms.signal_mimic overlaps explicit "
                    f"transforms {', '.join(overlaps)}; set "
                    "prep.audio_aug.allow_signal_mimic_overlap=true to allow "
                    "repeated degradation"
                )

        scipy_version = None
        for method in enabled_methods:
            if _method_requires_scipy(method, method_configs[method]):
                _require_scipy()
                scipy_version = _SCIPY_VERSION

        clean_paths = tuple(str(path) for path in audio_paths)
        return cls(
            seed=seed,
            speed_length_policy=speed_length_policy,
            pcm_policy=pcm_policy,
            allow_signal_mimic_overlap=allow_overlap,
            scipy_version=scipy_version,
            method_configs=method_configs,
            enabled_methods=enabled_methods,
            audio_paths=clean_paths,
        )

    @property
    def enabled(self) -> bool:
        return bool(self._enabled_methods)

    @property
    def additive_enabled(self) -> bool:
        return "noise_mix" in self._enabled_methods

    @property
    def changes_duration(self) -> bool:
        return (
            "speed_change" in self._enabled_methods
            and self._speed_length_policy == "variable"
        )

    def summary(self) -> dict[str, Any]:
        enabled_set = set(self._enabled_methods)
        return {
            "enabled": self.enabled,
            "seed": self._seed,
            "compat_revision": AUDIO_AUG_COMPAT_REVISION,
            "numpy_version": str(np.__version__),
            "scipy_version": self._scipy_version,
            "speed_length_policy": self._speed_length_policy,
            "changes_duration": self.changes_duration,
            "pcm_policy": self._pcm_policy,
            "allow_signal_mimic_overlap": self._allow_signal_mimic_overlap,
            "pre_mix_order": [
                method for method in PRE_MIX_METHODS if method in enabled_set
            ],
            "additive_order": [
                method for method in ADDITIVE_METHODS if method in enabled_set
            ],
            "post_mix_order": [
                method for method in POST_MIX_METHODS if method in enabled_set
            ],
            "transforms": {
                method: {
                    "enabled": method in enabled_set,
                    "phase": _phase_for(method),
                    **dict(self._method_configs[method]),
                }
                for method in SUPPORTED_METHODS
            },
        }

    def recipe_metadata(self, index: int) -> dict[str, Any]:
        recipe = self._recipe_at(index)
        stages = []
        for stage in recipe.stages:
            params = self._stage_params(stage)
            metadata: dict[str, Any] = {
                "name": stage.name,
                "phase": stage.phase,
                "params": params,
                "recipe_seed": stage.recipe_seed,
            }
            if stage.name == "noise_mix":
                metadata["requested_snr_db"] = float(params["snr_db"])
            stages.append(metadata)
        return {
            "seed": self._seed,
            "row_index": index,
            "compat_revision": AUDIO_AUG_COMPAT_REVISION,
            "speed_length_policy": self._speed_length_policy,
            "pcm_policy": self._pcm_policy,
            "stages": stages,
        }

    def apply_pre_mix(self, index: int, waveform, sample_rate: int):
        if not any(method in PRE_MIX_METHODS for method in self._enabled_methods):
            return waveform
        return self._apply_phase(index, waveform, sample_rate, "pre_mix")

    def apply_additive_delta(self, index: int, waveform, sample_rate: int):
        torch = _import_torch()
        if not self.additive_enabled:
            return torch.zeros_like(waveform)
        transformed = self._apply_phase(index, waveform, sample_rate, "additive")
        if tuple(transformed.shape) != tuple(waveform.shape):
            raise RuntimeError(
                "audio_aug additive transform must preserve waveform shape: "
                f"before={tuple(waveform.shape)}, after={tuple(transformed.shape)}"
            )
        return transformed - waveform

    def apply_post_mix(self, index: int, waveform, sample_rate: int):
        if not any(method in POST_MIX_METHODS for method in self._enabled_methods):
            return waveform
        return self._apply_phase(index, waveform, sample_rate, "post_mix")

    def __call__(self, index: int, waveform, sample_rate: int):
        if not self.enabled:
            return waveform
        clean = self.apply_pre_mix(index, waveform, sample_rate)
        additive = self.apply_additive_delta(index, clean, sample_rate)
        mixed = clean + additive
        return self.apply_post_mix(index, mixed, sample_rate)

    def _apply_phase(self, index: int, waveform, sample_rate: int, phase: str):
        _validate_waveform(
            waveform,
            sample_rate,
            field=f"audio_aug {phase} waveform for row {index}",
        )
        result = waveform
        for stage in self._recipe_at(index).stages:
            if stage.phase != phase:
                continue
            original_frames = int(result.size(1))
            samples = _to_numpy(result)
            samples = _to_upstream_pcm_scale(samples, self._pcm_policy)
            transformed = _apply_numpy_method(
                stage.name,
                samples,
                int(sample_rate),
                self._stage_params(stage),
            )
            transformed = np.asarray(transformed, dtype=np.float64)
            if transformed.ndim != 2 or transformed.shape[1] != 1:
                raise RuntimeError(
                    f"audio_aug {stage.name} returned invalid shape {transformed.shape}"
                )
            if stage.name == "speed_change" and self._speed_length_policy == "center_crop_or_zero_pad":
                transformed = _match_center_zero(transformed, original_frames)
            if stage.name != "speed_change" and transformed.shape[0] != original_frames:
                raise RuntimeError(
                    f"audio_aug {stage.name} must preserve sample count: "
                    f"before={original_frames}, after={transformed.shape[0]}"
                )
            transformed = _apply_pcm_policy(transformed, self._pcm_policy)
            if transformed.size == 0 or not bool(np.isfinite(transformed).all()):
                raise ValueError(
                    f"audio_aug {stage.name} produced empty or non-finite samples "
                    f"for row {index}"
                )
            result = _from_numpy(transformed, like=result)
        return result

    def _recipe_at(self, index: int) -> _RowRecipe:
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError("audio_aug row index must be an integer")
        if index < 0 or index >= len(self._audio_paths):
            raise IndexError(f"audio_aug row index out of range: {index}")
        audio_path = self._audio_paths[index]
        return _RowRecipe(
            stages=tuple(
                _StageRecipe(
                    name=method,
                    phase=_phase_for(method),
                    recipe_seed=_derive_seed(
                        self._seed,
                        index,
                        audio_path,
                        method,
                    ),
                )
                for method in SUPPORTED_METHODS
                if method in self._enabled_methods
            )
        )

    def _stage_params(self, stage: _StageRecipe) -> dict[str, Any]:
        params = dict(self._method_configs[stage.name])
        if stage.name in _STOCHASTIC_METHODS:
            params["seed"] = stage.recipe_seed
        return params


__all__ = [
    "ADDITIVE_METHODS",
    "AUDIO_AUG_COMPAT_REVISION",
    "AudioAugWaveformTransform",
    "POST_MIX_METHODS",
    "PRE_MIX_METHODS",
    "SUPPORTED_METHODS",
]
