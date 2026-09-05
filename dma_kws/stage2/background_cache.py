"""Mmap reader for the Stage II background fbank crop cache (format v1).

Training-start checks are intentionally lighter than ``--verify-only``: this
module validates manifest identity, recordings/index digests, shard
headers/sizes, and crop index ranges. It does not SHA256 multi-GB shard
payloads or scan them for finite values.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
import hashlib
import json
import os
from pathlib import Path
import random
from typing import Any

import numpy as np
from numpy.lib import format as npy_format
import torch

from dma_kws.data_prep.stage2_background import (
    CROP_DTYPE,
    FORMAT_VERSION,
    _cache_id_for,
    _normalize_fbank,
)


def _require_max_open_shards(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(
            "stage2.background_negative.max_open_shards must be a "
            f"positive int, got {value!r}"
        )
    return value


def _sha256_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            records.append(payload)
    return records


def _read_list_entries(list_path: Path) -> list[str]:
    if not list_path.is_file():
        raise FileNotFoundError(f"Background audio list not found: {list_path}")
    entries: list[str] = []
    with list_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            entries.append(line)
    return entries


def _read_npy_header(path: Path) -> tuple[tuple[int, ...], np.dtype, bool]:
    with path.open("rb") as handle:
        version = npy_format.read_magic(handle)
        if version == (1, 0):
            shape, fortran, dtype = npy_format.read_array_header_1_0(handle)
        elif version == (2, 0):
            shape, fortran, dtype = npy_format.read_array_header_2_0(handle)
        else:
            raise ValueError(f"Unsupported npy header version {version} in {path}")
    return tuple(int(dim) for dim in shape), np.dtype(dtype), bool(fortran)


def _close_mmap(array: np.ndarray) -> None:
    mmap_obj = getattr(array, "_mmap", None)
    if mmap_obj is None:
        return
    try:
        mmap_obj.close()
    except (BufferError, ValueError, OSError):
        pass


class BackgroundFeatureCache:
    """Lazy mmap LRU over a format-v1 background crop cache."""

    def __init__(
        self,
        manifest_path,
        *,
        expected_fbank_kwargs,
        duration_seconds_min,
        duration_seconds_max,
        audio_list_path="",
        max_open_shards=8,
    ) -> None:
        self.max_open_shards = _require_max_open_shards(max_open_shards)
        self._open_shards: OrderedDict[int, np.ndarray] = OrderedDict()
        self._pid = os.getpid()
        self.manifest_path = Path(manifest_path).expanduser()
        try:
            self._init_from_manifest(
                expected_fbank_kwargs=expected_fbank_kwargs,
                duration_seconds_min=duration_seconds_min,
                duration_seconds_max=duration_seconds_max,
                audio_list_path=audio_list_path,
            )
        except Exception:
            self.close()
            raise

    def _init_from_manifest(
        self,
        *,
        expected_fbank_kwargs,
        duration_seconds_min,
        duration_seconds_max,
        audio_list_path,
    ) -> None:
        manifest_path = self.manifest_path
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"Background cache manifest not found: {manifest_path}"
            )
        manifest_path = manifest_path.resolve()
        self.manifest_path = manifest_path
        cache_dir = manifest_path.parent
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid background cache manifest: {manifest_path}") from exc
        if not isinstance(manifest, dict):
            raise ValueError(f"manifest.json must be a mapping: {manifest_path}")

        if int(manifest.get("format_version", -1)) != FORMAT_VERSION:
            raise ValueError(
                f"Unsupported background cache format_version="
                f"{manifest.get('format_version')!r} in {manifest_path}"
            )
        if manifest.get("split_role") != "train":
            raise ValueError(
                f"Background cache split_role must be 'train', got "
                f"{manifest.get('split_role')!r} in {manifest_path}"
            )
        expected_id = _cache_id_for(manifest)
        stamped = manifest.get("cache_id")
        if not stamped or stamped != expected_id:
            raise ValueError(
                f"manifest cache_id mismatch for {manifest_path}: "
                f"stamped {stamped!r}, recomputed {expected_id!r}"
            )
        if manifest.get("dtype") != "float32":
            raise ValueError(
                f"Background cache dtype must be float32, got "
                f"{manifest.get('dtype')!r} in {manifest_path}"
            )

        try:
            seed = int(manifest["seed"])
            k = int(manifest["K"])
            num_sources = int(manifest["num_sources"])
            num_crops = int(manifest["num_crops"])
            feature_dim = int(manifest["feature_dim"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"Background cache manifest is missing seed/K/counts: {manifest_path}"
            ) from exc
        if k < 1 or num_sources < 1 or num_crops < 1 or feature_dim < 1:
            raise ValueError(
                f"Background cache seed/K/counts are invalid in {manifest_path}: "
                f"seed={seed!r} K={k!r} num_sources={num_sources!r} "
                f"num_crops={num_crops!r} feature_dim={feature_dim!r}"
            )

        stored_min = float(manifest["duration_seconds_min"])
        stored_max = float(manifest["duration_seconds_max"])
        duration_min = float(duration_seconds_min)
        duration_max = float(duration_seconds_max)
        if stored_min != duration_min or stored_max != duration_max:
            raise ValueError(
                f"Background cache duration range mismatch for {manifest_path}: "
                f"cache ({stored_min}, {stored_max}) != training "
                f"({duration_min}, {duration_max})"
            )

        stored_fbank = manifest.get("fbank")
        if not isinstance(stored_fbank, Mapping):
            raise ValueError(f"Background cache missing fbank dict: {manifest_path}")
        stored_fbank = _normalize_fbank(stored_fbank)
        if float(stored_fbank.get("dither", 0.0)) != 0.0:
            raise ValueError(
                f"fbank_cache requires dither=0, cache has {stored_fbank.get('dither')!r} "
                f"in {manifest_path}"
            )
        expected_fbank = _normalize_fbank(dict(expected_fbank_kwargs or {}))
        if float(expected_fbank.get("dither", 0.0)) != 0.0:
            raise ValueError(
                "stage2.background_negative.mode=fbank_cache "
                f"requires fbank dither=0, got {expected_fbank.get('dither')!r}"
            )
        if expected_fbank != stored_fbank:
            raise ValueError(
                f"fbank config mismatch for {manifest_path}: "
                f"cache {stored_fbank!r} != training {expected_fbank!r}"
            )

        recordings_spec = manifest.get("recordings")
        crops_spec = manifest.get("crops")
        shard_specs = manifest.get("shards")
        if not isinstance(recordings_spec, Mapping) or not isinstance(
            crops_spec, Mapping
        ):
            raise ValueError(
                f"manifest.json must include recordings and crops file records: "
                f"{manifest_path}"
            )
        if not isinstance(shard_specs, list) or not shard_specs:
            raise ValueError(f"manifest.json shards list is empty: {manifest_path}")

        recordings_path = cache_dir / str(recordings_spec["path"])
        crops_path = cache_dir / str(crops_spec["path"])
        self._check_digest(
            recordings_path,
            str(recordings_spec["sha256"]),
            int(recordings_spec["size"]),
        )
        self._check_digest(
            crops_path,
            str(crops_spec["sha256"]),
            int(crops_spec["size"]),
        )

        recordings = _read_jsonl(recordings_path)
        if len(recordings) != num_sources:
            raise ValueError(
                f"recordings.jsonl length {len(recordings)} != num_sources "
                f"{num_sources}: {recordings_path}"
            )
        crops = np.load(crops_path, allow_pickle=False)
        if getattr(crops.dtype, "names", None) != CROP_DTYPE.names:
            raise ValueError(
                f"crops.npy dtype fields are {crops.dtype.names!r} in {crops_path}"
            )
        if len(crops) != num_crops:
            raise ValueError(
                f"crops.npy length {len(crops)} != manifest num_crops {num_crops}: "
                f"{crops_path}"
            )

        shard_paths: list[Path] = []
        shard_frames: list[int] = []
        for shard_index, shard_spec in enumerate(shard_specs):
            if not isinstance(shard_spec, Mapping):
                raise ValueError(f"manifest shard records must be mappings: {manifest_path}")
            shard_path = cache_dir / str(shard_spec["path"])
            example_crop = _example_crop_id(crops, shard_index)
            if not shard_path.is_file():
                raise FileNotFoundError(
                    f"Missing feature shard {shard_path} for crop_id {example_crop}"
                )
            actual_size = shard_path.stat().st_size
            expected_size = int(shard_spec["size"])
            if actual_size != expected_size:
                raise ValueError(
                    f"size mismatch for {shard_path} (crop_id {example_crop}): "
                    f"expected {expected_size}, got {actual_size}"
                )
            shape, dtype, fortran = _read_npy_header(shard_path)
            if dtype != np.dtype(np.float32) or len(shape) != 2 or fortran:
                raise ValueError(
                    f"Feature shard {shard_path} for crop_id {example_crop} has "
                    f"dtype={dtype} shape={shape} fortran={fortran}, expected "
                    f"C-contiguous float32 [frames, {feature_dim}]"
                )
            if int(shape[1]) != feature_dim:
                raise ValueError(
                    f"Feature shard {shard_path} for crop_id {example_crop} "
                    f"feature_dim {shape[1]} != {feature_dim}"
                )
            header_frames = int(shape[0])
            spec_frames = int(shard_spec["num_frames"])
            if header_frames != spec_frames:
                raise ValueError(
                    f"Feature shard {shard_path} for crop_id {example_crop} "
                    f"num_frames {header_frames} != {spec_frames}"
                )
            shard_paths.append(shard_path)
            shard_frames.append(header_frames)

        for crop_id, row in enumerate(crops):
            shard_index = int(row["shard_index"])
            frame_offset = int(row["frame_offset"])
            num_frames = int(row["num_frames"])
            if shard_index < 0 or shard_index >= len(shard_paths):
                raise ValueError(
                    f"crop_id {crop_id} shard_index {shard_index} is out of range "
                    f"for {manifest_path}"
                )
            shard_path = shard_paths[shard_index]
            if (
                num_frames < 1
                or frame_offset < 0
                or frame_offset + num_frames > shard_frames[shard_index]
            ):
                raise ValueError(
                    f"crop_id {crop_id} frame_offset {frame_offset} num_frames "
                    f"{num_frames} out of range for {shard_path} with "
                    f"{shard_frames[shard_index]} frames"
                )

        crop_starts = np.empty(num_sources, dtype=np.int64)
        crop_counts = np.empty(num_sources, dtype=np.int32)
        list_entries: list[str] = []
        source_ids: list[str] = []
        for index, record in enumerate(recordings):
            start = int(record["crop_start"])
            count = int(record["crop_count"])
            if count != k:
                raise ValueError(
                    f"recordings.jsonl crop_count {count} != K {k} for "
                    f"{record.get('source_id')!r} in {recordings_path}"
                )
            if start < 0 or start + count > num_crops:
                raise ValueError(
                    f"recordings.jsonl crop window out of range for "
                    f"{record.get('source_id')!r} in {recordings_path}"
                )
            crop_starts[index] = start
            crop_counts[index] = count
            list_entries.append(str(record.get("list_entry", "")))
            source_ids.append(str(record.get("source_id", "")))

        audio_list = str(audio_list_path or "").strip()
        if audio_list:
            self._check_audio_list(
                Path(audio_list),
                list_entries=list_entries,
                source_ids=source_ids,
                musan_root=str((manifest.get("split") or {}).get("musan_root", "")),
            )

        crop_starts.setflags(write=False)
        crop_counts.setflags(write=False)
        shard_index_arr = np.ascontiguousarray(crops["shard_index"], dtype=np.int32)
        frame_offset_arr = np.ascontiguousarray(crops["frame_offset"], dtype=np.int64)
        num_frames_arr = np.ascontiguousarray(crops["num_frames"], dtype=np.int32)
        shard_index_arr.setflags(write=False)
        frame_offset_arr.setflags(write=False)
        num_frames_arr.setflags(write=False)

        self.cache_id = str(stamped)
        self.format_version = FORMAT_VERSION
        self.num_crops = num_crops
        self.num_sources = num_sources
        self.feature_dim = feature_dim
        self.K = k
        self.seed = seed
        self.fbank = dict(stored_fbank)
        self._shard_paths = tuple(shard_paths)
        self._shard_frames = tuple(shard_frames)
        self._crop_starts = crop_starts
        self._crop_counts = crop_counts
        self._shard_index = shard_index_arr
        self._frame_offset = frame_offset_arr
        self._num_frames = num_frames_arr

    @staticmethod
    def _check_digest(path: Path, expected_sha256: str, expected_size: int) -> None:
        if not path.is_file():
            raise FileNotFoundError(f"Background cache file not found: {path}")
        digest, size = _sha256_file(path)
        if digest != expected_sha256 or size != int(expected_size):
            raise ValueError(
                f"Checksum mismatch for {path}: expected sha256={expected_sha256} "
                f"size={expected_size}, got sha256={digest} size={size}"
            )

    @staticmethod
    def _check_audio_list(
        list_path: Path,
        *,
        list_entries: list[str],
        source_ids: list[str],
        musan_root: str,
    ) -> None:
        listed = _read_list_entries(list_path)
        if sorted(listed) == sorted(list_entries):
            return
        root = Path(musan_root).expanduser() if musan_root else None
        derived: list[str] = []
        base = list_path.parent
        for entry in listed:
            path = Path(entry).expanduser()
            if not path.is_absolute():
                path = base / path
            if root is not None and str(root):
                try:
                    derived.append(
                        path.resolve(strict=False)
                        .relative_to(root.resolve(strict=False))
                        .as_posix()
                    )
                    continue
                except ValueError:
                    pass
            derived.append(entry)
        if sorted(derived) == sorted(source_ids):
            return
        raise ValueError(
            f"Background audio list {list_path} does not match the cache "
            f"source-directory / canonical entries"
        )

    def run_record_fields(self) -> dict[str, Any]:
        return {
            "background_cache_id": self.cache_id,
            "background_cache_manifest": str(self.manifest_path),
            "background_cache_format_version": int(self.format_version),
            "background_crop_count": int(self.num_crops),
            "background_fbank": dict(self.fbank),
        }

    def extract(self, *, rng: random.Random) -> torch.Tensor:
        if not isinstance(rng, random.Random):
            raise TypeError("BackgroundFeatureCache.extract requires the caller rng")
        source_index = rng.randrange(self.num_sources)
        ordinal = rng.randrange(int(self._crop_counts[source_index]))
        crop_id = int(self._crop_starts[source_index]) + int(ordinal)
        return self.read_crop(crop_id)

    def read_crop(self, crop_id: int) -> torch.Tensor:
        crop_id = int(crop_id)
        if crop_id < 0 or crop_id >= self.num_crops:
            raise ValueError(
                f"crop_id {crop_id} out of range for {self.manifest_path} "
                f"(num_crops={self.num_crops})"
            )
        shard_index = int(self._shard_index[crop_id])
        frame_offset = int(self._frame_offset[crop_id])
        num_frames = int(self._num_frames[crop_id])
        path = self._shard_paths[shard_index]
        try:
            array = self._shard_mmap(shard_index)
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"Missing feature shard {path} for crop_id {crop_id}"
            ) from exc
        if array.dtype != np.float32 or array.ndim != 2:
            raise ValueError(
                f"Feature shard {path} for crop_id {crop_id} has dtype={array.dtype} "
                f"shape={array.shape}, expected float32 [frames, {self.feature_dim}]"
            )
        if int(array.shape[1]) != self.feature_dim:
            raise ValueError(
                f"Feature shard {path} for crop_id {crop_id} feature_dim "
                f"{array.shape[1]} != {self.feature_dim}"
            )
        end = frame_offset + num_frames
        if frame_offset < 0 or num_frames < 1 or end > int(array.shape[0]):
            raise ValueError(
                f"crop_id {crop_id} frame_offset {frame_offset} num_frames "
                f"{num_frames} out of range for {path} with {array.shape[0]} frames"
            )
        copied = np.empty((num_frames, self.feature_dim), dtype=np.float32)
        copied[...] = array[frame_offset:end]
        return torch.from_numpy(copied)

    def close(self) -> None:
        maps = list(self._open_shards.values())
        self._open_shards.clear()
        if os.getpid() == self._pid:
            for array in maps:
                _close_mmap(array)

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_open_shards"] = OrderedDict()
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._open_shards = OrderedDict()
        self._pid = os.getpid()

    def _ensure_pid(self) -> None:
        pid = os.getpid()
        if pid == self._pid:
            return
        self._open_shards = OrderedDict()
        self._pid = pid

    def _shard_mmap(self, shard_index: int) -> np.ndarray:
        self._ensure_pid()
        cached = self._open_shards.get(shard_index)
        if cached is not None:
            self._open_shards.move_to_end(shard_index)
            return cached
        path = self._shard_paths[shard_index]
        if not path.is_file():
            raise FileNotFoundError(path)
        while len(self._open_shards) >= self.max_open_shards:
            _, evicted = self._open_shards.popitem(last=False)
            _close_mmap(evicted)
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        self._open_shards[shard_index] = array
        return array


def _example_crop_id(crops: np.ndarray, shard_index: int) -> int:
    for row in crops:
        if int(row["shard_index"]) == shard_index:
            return int(row["crop_id"])
    return -1
