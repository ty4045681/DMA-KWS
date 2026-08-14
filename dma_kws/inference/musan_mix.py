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

from dma_kws.audio import load_audio
from dma_kws.inference.manifest import iter_audio_files


_RMS_EPSILON = 1.0e-8


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


def _require_mapping(value: object, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field} must be a mapping")
    return value


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
        clean_paths = tuple(str(path) for path in audio_paths)

        if not noise_enabled and not music_enabled and not speech_enabled:
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
                recipes=(_MixRecipe() for _ in clean_paths),
            )

        root_raw = prep.get("musan_root", "")
        if not isinstance(root_raw, (str, Path)) or not str(root_raw).strip():
            raise ValueError(
                "prep.musan_root is required when MUSAN mixing is enabled"
            )
        root = Path(root_raw).expanduser().resolve()
        if not root.is_dir():
            raise NotADirectoryError(f"MUSAN root not found: {root}")

        noise_files = _discover_subset(root, "noise") if noise_enabled else ()
        music_files = _discover_subset(root, "music") if music_enabled else ()
        speech_files = _discover_subset(root, "speech") if speech_enabled else ()
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
            recipes=recipes,
        )

    @property
    def enabled(self) -> bool:
        return self._noise_enabled or self._music_enabled or self._speech_enabled

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
                "num_files": len(self._noise_files),
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
            "length_policy": "random_crop_or_repeat",
            "mix_policy": "scale_each_against_clean_rms_then_sum",
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
        return result

    def __call__(self, index: int, waveform, sample_rate: int):
        if not self.enabled:
            return waveform
        recipe = self._recipe_at(index)

        if waveform.dim() != 2 or waveform.size(0) != 1:
            raise ValueError(
                "clean waveform must be mono with shape (1, samples), got "
                f"{tuple(waveform.shape)} for row {index}"
            )
        clean_rms = _waveform_rms(waveform, field=f"clean waveform for row {index}")
        target_samples = int(waveform.size(1))
        components = []

        for component in (recipe.noise, recipe.music, recipe.speech):
            if component is None:
                continue
            source_waveform, source_rate = _load_source_audio(
                component,
                target_samples=target_samples,
                sample_rate=int(sample_rate),
            )
            if int(source_rate) != int(sample_rate):
                raise ValueError(
                    f"MUSAN {component.kind} loader returned sample rate "
                    f"{source_rate}, expected {sample_rate}: {component.source_path}"
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

        mixed = waveform
        for component in components:
            mixed = mixed + component
        _validate_finite_waveform(mixed, field=f"mixed waveform for row {index}")
        return mixed

    def _recipe_at(self, index: int) -> _MixRecipe:
        if index < 0 or index >= len(self._recipes):
            raise IndexError(f"MUSAN mix row index out of range: {index}")
        return self._recipes[index]


__all__ = ["MusanWaveformMixer"]
