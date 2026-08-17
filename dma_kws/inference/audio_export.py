"""Deterministic export of selected evaluation waveforms as float WAV files."""

from __future__ import annotations

import hashlib
import json
import os
import re
import struct
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


_CONFIG_KEY = "audio_export"
_MODES = {"disabled", "random", "all"}
_ROW_FILENAME = re.compile(r"^row_[0-9]{8}\.wav$")
_INDEX_FILENAME = "index.jsonl"


@dataclass(frozen=True)
class WaveformExportConfig:
    """Validated selection settings for model-input waveform export."""

    mode: str = "disabled"
    count: int = 5
    seed: int = 2025

    @property
    def enabled(self) -> bool:
        return self.mode != "disabled"


def _require_plain_int(value: object, *, field: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field} must be an integer")
    if value < minimum:
        raise ValueError(f"{field} must be >= {minimum}")
    return int(value)


def parse_waveform_export_config(
    prep: Mapping[str, Any],
    *,
    key: str = _CONFIG_KEY,
) -> WaveformExportConfig:
    """Parse and strictly validate ``prep.audio_export``."""

    if not isinstance(prep, Mapping):
        raise TypeError("prep must be a mapping")
    raw = prep.get(key, {})
    if not isinstance(raw, Mapping):
        raise TypeError(f"prep.{key} must be a mapping")
    unknown = sorted(str(name) for name in raw if name not in {"mode", "count", "seed"})
    if unknown:
        raise ValueError(
            f"prep.{key} has unsupported keys: {', '.join(unknown)}"
        )

    mode = raw.get("mode", "disabled")
    if not isinstance(mode, str) or mode not in _MODES:
        supported = ", ".join(sorted(_MODES))
        raise ValueError(f"prep.{key}.mode must be one of: {supported}")
    count = _require_plain_int(
        raw.get("count", 5),
        field=f"prep.{key}.count",
        minimum=1 if mode == "random" else 0,
    )
    seed = _require_plain_int(
        raw.get("seed", 2025),
        field=f"prep.{key}.seed",
        minimum=0,
    )
    return WaveformExportConfig(mode=mode, count=count, seed=seed)


def _selection_digest(*, seed: int, row_index: int, audio_path: str) -> bytes:
    payload = f"{seed}\0{row_index}\0{audio_path}".encode("utf-8")
    return hashlib.sha256(payload).digest()


def select_waveform_indices(
    audio_paths: Sequence[str],
    config: WaveformExportConfig,
) -> tuple[int, ...]:
    """Return stable manifest indices selected independently of worker order."""

    if not isinstance(config, WaveformExportConfig):
        raise TypeError("config must be a WaveformExportConfig")
    if not config.enabled:
        return ()
    if config.mode == "all":
        return tuple(range(len(audio_paths)))

    ranked = sorted(
        range(len(audio_paths)),
        key=lambda index: (
            _selection_digest(
                seed=config.seed,
                row_index=index,
                audio_path=str(audio_paths[index]),
            ),
            index,
        ),
    )
    return tuple(sorted(ranked[: min(config.count, len(ranked))]))


def _atomic_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            for record in records:
                handle.write(
                    json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n"
                )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


@dataclass(frozen=True)
class _FloatWavInfo:
    sample_rate: int
    num_samples: int
    duration_sec: float
    peak_abs: float


def _write_float32_mono_wav(
    path: Path,
    samples: Any,
    *,
    sample_rate: int,
) -> None:
    """Atomically write a mono IEEE-float WAV without an external codec."""

    import numpy as np

    values = np.asarray(samples, dtype="<f4")
    if values.ndim != 1 or values.size <= 0:
        raise ValueError("samples must be a non-empty one-dimensional array")
    if not bool(np.isfinite(values).all()):
        raise ValueError("samples contain non-finite values")
    sample_bytes = values.tobytes(order="C")
    if len(sample_bytes) > 0xFFFFFFFF:
        raise ValueError("waveform is too large for a RIFF/WAVE data chunk")

    # WAVE_FORMAT_IEEE_FLOAT (format tag 3), mono, 32 bits per sample. Float
    # WAV conventionally includes a fact chunk containing the frame count.
    block_align = 4
    byte_rate = sample_rate * block_align
    fmt_payload = (
        struct.pack(
            "<HHIIHH",
            3,
            1,
            sample_rate,
            byte_rate,
            block_align,
            32,
        )
        + struct.pack("<H", 0)
    )
    fact_payload = struct.pack("<I", int(values.size))
    riff_size = (
        4
        + 8
        + len(fmt_payload)
        + 8
        + len(fact_payload)
        + 8
        + len(sample_bytes)
    )
    if riff_size > 0xFFFFFFFF:
        raise ValueError("waveform is too large for a RIFF/WAVE file")

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}.",
        suffix=".wav",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(struct.pack("<4sI4s", b"RIFF", riff_size, b"WAVE"))
            handle.write(struct.pack("<4sI", b"fmt ", len(fmt_payload)))
            handle.write(fmt_payload)
            handle.write(struct.pack("<4sI", b"fact", len(fact_payload)))
            handle.write(fact_payload)
            handle.write(struct.pack("<4sI", b"data", len(sample_bytes)))
            handle.write(sample_bytes)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _inspect_float32_mono_wav(path: Path) -> _FloatWavInfo:
    """Parse and validate the exact IEEE-float WAV layout used by the exporter."""

    import numpy as np

    payload = path.read_bytes()
    if len(payload) < 12:
        raise ValueError("file is shorter than a RIFF/WAVE header")
    riff_id, riff_size, wave_id = struct.unpack_from("<4sI4s", payload, 0)
    if riff_id != b"RIFF" or wave_id != b"WAVE":
        raise ValueError("file is not RIFF/WAVE")
    if riff_size + 8 != len(payload):
        raise ValueError("RIFF size does not match file size")

    fmt_payload = None
    fact_samples = None
    data_payload = None
    offset = 12
    while offset < len(payload):
        if offset + 8 > len(payload):
            raise ValueError("truncated RIFF chunk header")
        chunk_id, chunk_size = struct.unpack_from("<4sI", payload, offset)
        offset += 8
        chunk_end = offset + chunk_size
        if chunk_end > len(payload):
            raise ValueError(f"truncated {chunk_id!r} RIFF chunk")
        chunk_payload = payload[offset:chunk_end]
        if chunk_id == b"fmt ":
            if fmt_payload is not None:
                raise ValueError("duplicate fmt chunk")
            fmt_payload = chunk_payload
        elif chunk_id == b"fact":
            if fact_samples is not None or len(chunk_payload) < 4:
                raise ValueError("invalid fact chunk")
            fact_samples = struct.unpack_from("<I", chunk_payload, 0)[0]
        elif chunk_id == b"data":
            if data_payload is not None:
                raise ValueError("duplicate data chunk")
            data_payload = chunk_payload
        offset = chunk_end + (chunk_size & 1)
    if offset != len(payload):
        raise ValueError("invalid RIFF chunk padding")
    if fmt_payload is None or len(fmt_payload) < 16:
        raise ValueError("missing or invalid fmt chunk")
    if data_payload is None:
        raise ValueError("missing data chunk")

    (
        format_tag,
        channels,
        sample_rate,
        byte_rate,
        block_align,
        bits_per_sample,
    ) = struct.unpack_from("<HHIIHH", fmt_payload, 0)
    if (
        format_tag != 3
        or channels != 1
        or sample_rate <= 0
        or byte_rate != sample_rate * 4
        or block_align != 4
        or bits_per_sample != 32
    ):
        raise ValueError("WAV is not mono 32-bit IEEE float audio")
    if len(data_payload) == 0 or len(data_payload) % block_align:
        raise ValueError("invalid IEEE-float WAV data size")
    num_samples = len(data_payload) // block_align
    if fact_samples is None or fact_samples != num_samples:
        raise ValueError("fact chunk does not match WAV frame count")

    samples = np.frombuffer(data_payload, dtype="<f4")
    if not bool(np.isfinite(samples).all()):
        raise ValueError("WAV contains non-finite samples")
    peak_abs = float(np.abs(samples).max())
    return _FloatWavInfo(
        sample_rate=int(sample_rate),
        num_samples=int(num_samples),
        duration_sec=float(num_samples / sample_rate),
        peak_abs=peak_abs,
    )


class SelectedWaveformExporter:
    """Pickle-friendly observer that writes a deterministic subset of waveforms.

    Instances contain only regular Python values so they can be copied into
    ``DataLoader`` workers. Each selected row owns one unique filename; final
    validation and index generation happen in the parent process by inspecting
    those files rather than relying on mutable worker-local counters.
    """

    def __init__(
        self,
        *,
        output_dir: str | Path,
        audio_paths: Sequence[str],
        mode: str = "disabled",
        count: int = 5,
        seed: int = 2025,
    ) -> None:
        config = parse_waveform_export_config(
            {_CONFIG_KEY: {"mode": mode, "count": count, "seed": seed}}
        )
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.audio_paths = tuple(str(path) for path in audio_paths)
        self.config = config
        self.selected_indices = select_waveform_indices(self.audio_paths, config)
        self._selected_set = frozenset(self.selected_indices)

    @classmethod
    def from_prep(
        cls,
        prep: Mapping[str, Any],
        *,
        output_dir: str | Path,
        audio_paths: Sequence[str],
        key: str = _CONFIG_KEY,
    ) -> "SelectedWaveformExporter":
        config = parse_waveform_export_config(prep, key=key)
        return cls(
            output_dir=output_dir,
            audio_paths=audio_paths,
            mode=config.mode,
            count=config.count,
            seed=config.seed,
        )

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    @property
    def index_path(self) -> Path:
        return self.output_dir / _INDEX_FILENAME

    def path_for_index(self, index: int) -> Path | None:
        """Return the selected row's expected path, or ``None`` if unselected."""

        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError("index must be an integer")
        if index < 0 or index >= len(self.audio_paths):
            raise IndexError(f"row index out of range: {index}")
        if index not in self._selected_set:
            return None
        return self.output_dir / f"row_{index:08d}.wav"

    def result_path(self, index: int) -> str | None:
        """Return an exported absolute path for a result row when it exists."""

        path = self.path_for_index(index)
        if path is None or not path.is_file():
            return None
        return str(path.resolve())

    def prepare(self) -> None:
        """Create the output directory and remove only files owned by this exporter."""

        if not self.output_dir.exists():
            if self.enabled:
                self.output_dir.mkdir(parents=True, exist_ok=True)
            return
        if not self.output_dir.is_dir():
            raise NotADirectoryError(
                f"audio export path is not a directory: {self.output_dir}"
            )
        for path in self.output_dir.iterdir():
            if path.is_file() and (
                _ROW_FILENAME.fullmatch(path.name) or path.name == _INDEX_FILENAME
            ):
                path.unlink()

    def __call__(self, index: int, waveform: Any, sample_rate: int) -> None:
        path = self.path_for_index(index)
        if path is None:
            return
        if isinstance(sample_rate, bool) or not isinstance(sample_rate, int):
            raise TypeError("sample_rate must be an integer")
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")

        try:
            shape = tuple(waveform.shape)
        except (AttributeError, TypeError) as exc:
            raise TypeError("waveform must be a tensor with a shape") from exc
        if len(shape) != 2 or shape[0] != 1 or shape[1] <= 0:
            raise ValueError(
                "waveform must be non-empty mono audio with shape (1, samples), "
                f"got {shape}"
            )
        try:
            values = waveform.detach().cpu().to(dtype=self._torch_float32()).squeeze(0)
            samples = values.contiguous().numpy()
        except (AttributeError, TypeError, RuntimeError) as exc:
            raise TypeError("waveform must be a torch-compatible tensor") from exc
        if not bool(self._numpy_isfinite(samples).all()):
            raise ValueError("waveform contains non-finite samples")

        _write_float32_mono_wav(path, samples, sample_rate=sample_rate)

    @staticmethod
    def _torch_float32():
        import torch

        return torch.float32

    @staticmethod
    def _numpy_isfinite(values):
        import numpy as np

        return np.isfinite(values)

    def finalize(self) -> dict[str, Any]:
        """Validate exports, atomically write ``index.jsonl``, and summarize them."""

        base = {
            "mode": self.config.mode,
            "requested_count": (
                self.config.count
                if self.config.mode == "random"
                else ("all" if self.config.mode == "all" else 0)
            ),
            "seed": self.config.seed,
            "format": "WAV",
            "subtype": "FLOAT",
            "stage": "post_augmentation_pre_padding",
            "num_selected": len(self.selected_indices),
            "row_indices": list(self.selected_indices),
        }
        if not self.enabled:
            return {
                **base,
                "status": "disabled",
                "directory": None,
                "manifest": None,
                "num_exported": 0,
            }

        selected_paths = []
        missing = []
        for index in self.selected_indices:
            path = self.path_for_index(index)
            assert path is not None
            selected_paths.append((index, path))
            if not path.is_file():
                missing.append(path)
        if missing:
            preview = ", ".join(str(path) for path in missing[:3])
            suffix = " ..." if len(missing) > 3 else ""
            raise FileNotFoundError(
                f"missing {len(missing)} selected waveform exports: {preview}{suffix}"
            )

        records = []
        for index, path in selected_paths:
            try:
                info = _inspect_float32_mono_wav(path)
            except (OSError, ValueError) as exc:
                raise ValueError(
                    f"invalid exported WAV for row {index}: {path}"
                ) from exc
            records.append(
                {
                    "row_index": index,
                    "source_audio_path": self.audio_paths[index],
                    "exported_audio_path": str(path.resolve()),
                    "sample_rate": info.sample_rate,
                    "num_samples": info.num_samples,
                    "duration_sec": info.duration_sec,
                    "peak_abs": info.peak_abs,
                }
            )
        _atomic_jsonl(self.index_path, records)
        return {
            **base,
            "status": "generated",
            "directory": str(self.output_dir),
            "manifest": str(self.index_path.resolve()),
            "num_exported": len(records),
        }


__all__ = [
    "SelectedWaveformExporter",
    "WaveformExportConfig",
    "parse_waveform_export_config",
    "select_waveform_indices",
]
