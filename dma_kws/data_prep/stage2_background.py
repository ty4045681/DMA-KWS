"""Pre-generate the Stage II background fbank crop cache (format v1).

Build writes a finite library of K waveform-crop fbank features per MUSAN
train-background recording. Crops are drawn with
``SHA256(seed, source_id, crop_ordinal)`` RNGs and materialized through the T1
crop contract, then extracted with the composed training fbank.

Checksum policy:

* **Build** hashes the split/list files, source audio, ``recordings.jsonl``,
  ``crops.npy``, and every ``features-*.npy`` shard, then stamps ``cache_id``.
* **``--verify-only``** re-hashes the full index and every shard and checks
  shapes, offsets, and finite values. This is the bit-rot detector.
* **Training start (T3)** should only check manifest identity, recordings/index
  digests, shard headers/sizes, and index ranges — not scan shard payloads.

Cache-internal paths in the manifest are relative to the cache directory so the
tree can be moved as a unit.
"""

from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import random
import shutil
import tempfile
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

import numpy as np
import torch

from dma_kws.config import compose_config, config_to_dict, fbank_kwargs, get_fbank_config
from dma_kws.stage2.background_sampling import (
    BackgroundCropSpec,
    draw_crop_spec,
    materialize_crop,
    probe_source_info,
)
from dma_kws.stage2.fbank import FbankExtractor


FORMAT_VERSION = 1
DEFAULT_EXPERIMENT = "icefall_zipformer_stage2_eps_softmin_v41"
DEFAULT_CROPS_PER_RECORDING = 128
DEFAULT_SEED = 2025
DEFAULT_WORKERS = 4
DEFAULT_SHARD_SIZE_MIB = 256
RECORDINGS_NAME = "recordings.jsonl"
CROPS_NAME = "crops.npy"
MANIFEST_NAME = "manifest.json"
SPLIT_JSON_NAME = "split.json"
TRAIN_LIST_NAME = "train_background.list"
EVAL_LIST_NAME = "eval_musan.list"
CACHE_ID_EXCLUDED_FIELDS = ("cache_id", "build")
SOURCE_ID_MAX_CHARS = 1024
BYTES_PER_FLOAT32 = 4
ALLOWED_TRAIN_CATEGORIES = ("music", "noise")

CROP_DTYPE = np.dtype(
    [
        ("crop_id", np.int64),
        ("source_index", np.int32),
        ("ordinal", np.int32),
        ("source_id", np.dtype(f"U{SOURCE_ID_MAX_CHARS}")),
        ("duration_seconds", np.float64),
        ("read_start_frame", np.int64),
        ("read_num_frames", np.int64),
        ("target_num_samples", np.int64),
        ("final_offset", np.int64),
        ("shard_index", np.int32),
        ("frame_offset", np.int64),
        ("num_frames", np.int32),
    ]
)

_WORKER_FBANK_KWARGS: dict[str, Any] | None = None
_WORKER_EXTRACTOR: FbankExtractor | None = None


@dataclass(frozen=True)
class SourceJob:
    source_index: int
    source_id: str
    path: str
    list_entry: str
    seed: int
    crops_per_recording: int
    duration_seconds_min: float
    duration_seconds_max: float
    feature_dim: int


@dataclass(frozen=True)
class SourceCrops:
    source_index: int
    source_id: str
    list_entry: str
    relative_path: str
    sample_rate: int
    num_frames: int
    channels: int
    content_sha256: str
    specs: tuple[BackgroundCropSpec, ...]
    features: tuple[np.ndarray, ...]


def crop_rng(seed: int, source_id: str, crop_ordinal: int) -> random.Random:
    """Return the per-crop RNG. Payload is ``f"{seed}\\0{source_id}\\0{ordinal}"``."""
    payload = f"{int(seed)}\0{source_id}\0{int(crop_ordinal)}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def _require_int(value: object, *, field: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{field} must be an int >= {minimum}, got {value!r}")
    return value


def _parse_int_at_least(minimum: int):
    def parser(value: str) -> int:
        if str(value).strip().casefold() in {"true", "false"}:
            raise argparse.ArgumentTypeError(
                f"must be an int >= {minimum}, got {value!r}"
            )
        try:
            parsed = int(value)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"must be an int >= {minimum}, got {value!r}"
            ) from exc
        if parsed < minimum:
            raise argparse.ArgumentTypeError(
                f"must be an int >= {minimum}, got {value!r}"
            )
        return parsed

    return parser


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


def _write_text(path: Path, text: str) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())


def _save_npy(path: Path, array: np.ndarray) -> None:
    np.save(path, np.ascontiguousarray(array), allow_pickle=False)
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _json_ready(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, sort_keys=True))


def _canonical_manifest_bytes(manifest: Mapping[str, Any]) -> bytes:
    payload = {
        key: value
        for key, value in manifest.items()
        if key not in CACHE_ID_EXCLUDED_FIELDS
    }
    ready = _json_ready(payload)
    return json.dumps(
        ready, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _cache_id_for(manifest: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_manifest_bytes(manifest)).hexdigest()


def _catalog_sha256(relative_paths: Iterable[str]) -> str:
    payload = "".join(f"{path}\n" for path in sorted(relative_paths)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _normalize_fbank(fbank: Mapping[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key, value in fbank.items():
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, bool) or value is None or isinstance(value, str):
            normalized[key] = value
        elif isinstance(value, int):
            normalized[key] = int(value)
        elif isinstance(value, float):
            normalized[key] = float(value)
        else:
            normalized[key] = value
    return _json_ready(normalized)


def _dependency_versions() -> dict[str, str]:
    versions = {
        "numpy": np.__version__,
        "torch": torch.__version__,
    }
    for name in ("soundfile", "lhotse", "torchaudio"):
        try:
            module = __import__(name)
        except ImportError:
            continue
        versions[name] = str(getattr(module, "__version__", "unknown"))
    return versions


def _load_list_entries(list_path: Path) -> list[tuple[str, Path]]:
    if not list_path.is_file():
        raise FileNotFoundError(f"Stage II background list file not found: {list_path}")

    entries: list[tuple[str, Path]] = []
    base = list_path.parent
    with list_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            path = Path(line).expanduser()
            if not path.is_absolute():
                path = base / path
            entries.append((line, path))
    return entries


def _source_id_for(path: Path, musan_root: Path) -> str:
    resolved = path.resolve()
    root = musan_root.resolve()
    try:
        relative = resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            f"MUSAN audio path is outside musan_root: {resolved} (root: {root})"
        ) from exc
    source_id = relative.as_posix()
    if not source_id or source_id.startswith("../"):
        raise ValueError(f"Invalid MUSAN-relative source_id for {resolved}: {source_id!r}")
    if len(source_id) > SOURCE_ID_MAX_CHARS:
        raise ValueError(
            f"source_id exceeds {SOURCE_ID_MAX_CHARS} characters: {source_id!r}"
        )
    return source_id


def _category_of(source_id: str) -> str:
    return source_id.split("/", 1)[0]


def _resolve_list_sources(
    list_path: Path,
    *,
    musan_root: Path,
    expected_count: int,
    expected_catalog_sha256: str,
    label: str,
) -> list[tuple[str, str, Path]]:
    entries = _load_list_entries(list_path)
    if not entries:
        if expected_count == 0:
            computed = _catalog_sha256([])
            if computed != expected_catalog_sha256:
                raise ValueError(
                    f"{label} catalog_sha256 does not match split.json: {list_path}"
                )
            return []
        raise ValueError(f"{label} list is empty: {list_path}")

    missing = next((path for _, path in entries if not path.is_file()), None)
    if missing is not None:
        raise FileNotFoundError(f"{label} audio not found: {missing}")

    resolved_rows: list[tuple[str, str, Path]] = []
    seen_ids: dict[str, Path] = {}
    seen_paths: dict[Path, str] = {}
    for list_entry, path in entries:
        resolved = path.resolve()
        source_id = _source_id_for(path, musan_root)
        previous = seen_ids.get(source_id)
        if previous is not None:
            raise ValueError(
                f"Duplicate source_id {source_id!r} in {list_path}: {previous} and {resolved}"
            )
        previous_id = seen_paths.get(resolved)
        if previous_id is not None:
            raise ValueError(
                f"Duplicate canonical source path in {list_path}: {resolved}"
            )
        seen_ids[source_id] = resolved
        seen_paths[resolved] = source_id
        resolved_rows.append((source_id, list_entry, resolved))

    relative_ids = [source_id for source_id, _, _ in resolved_rows]
    if len(relative_ids) != expected_count:
        raise ValueError(
            f"{label} list count {len(relative_ids)} does not match split.json "
            f"recordings={expected_count}: {list_path}"
        )
    computed = _catalog_sha256(relative_ids)
    if computed != expected_catalog_sha256:
        raise ValueError(
            f"{label} list catalog_sha256 does not match split.json: {list_path}"
        )
    return resolved_rows


def _load_split(split_dir: Path) -> dict[str, Any]:
    split_path = split_dir / SPLIT_JSON_NAME
    if not split_path.is_file():
        raise FileNotFoundError(f"split.json not found: {split_path}")
    document = json.loads(split_path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError(f"split.json must be a mapping: {split_path}")
    return document


def _split_side(document: Mapping[str, Any], side: str) -> dict[str, Any]:
    splits = document.get("splits")
    if not isinstance(splits, Mapping) or side not in splits:
        raise ValueError(f"split.json missing splits.{side}")
    payload = splits[side]
    if not isinstance(payload, Mapping):
        raise ValueError(f"split.json splits.{side} must be a mapping")
    return dict(payload)


def _validate_train_categories(document: Mapping[str, Any]) -> list[str]:
    policy = document.get("policy")
    if not isinstance(policy, Mapping):
        raise ValueError("split.json missing policy.train_categories")
    raw = policy.get("train_categories")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)) or not raw:
        raise ValueError("split.json policy.train_categories must be a non-empty list")
    categories = [str(item) for item in raw]
    if "speech" in categories:
        raise ValueError(
            "v4.1 background cache requires policy.train_categories to be only "
            f"music and noise; speech must stay on eval, got {categories}"
        )
    if tuple(categories) != ALLOWED_TRAIN_CATEGORIES and set(categories) != set(
        ALLOWED_TRAIN_CATEGORIES
    ):
        raise ValueError(
            "v4.1 background cache requires policy.train_categories to be only "
            f"music and noise, got {categories}"
        )
    return categories


def _validate_no_speech(rows: Sequence[tuple[str, str, Path]], *, label: str) -> None:
    for source_id, _list_entry, path in rows:
        if _category_of(source_id) == "speech":
            raise ValueError(
                f"{label} contains speech recording {source_id}: {path}"
            )


def _configure_torch_threads() -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def _init_worker(fbank_params: dict[str, Any]) -> None:
    global _WORKER_FBANK_KWARGS, _WORKER_EXTRACTOR
    _configure_torch_threads()
    _WORKER_FBANK_KWARGS = dict(fbank_params)
    _WORKER_EXTRACTOR = None


def _worker_extractor() -> FbankExtractor:
    global _WORKER_EXTRACTOR
    if _WORKER_EXTRACTOR is None:
        if _WORKER_FBANK_KWARGS is None:
            raise RuntimeError("Worker fbank kwargs were not initialized")
        _WORKER_EXTRACTOR = FbankExtractor(**_WORKER_FBANK_KWARGS)
    return _WORKER_EXTRACTOR


def _validate_feature(
    array: np.ndarray,
    *,
    feature_dim: int,
    source_path: Path,
) -> None:
    if array.ndim != 2 or int(array.shape[0]) < 1 or int(array.shape[1]) != feature_dim:
        raise ValueError(
            f"Background fbank must be non-empty 2-D [frames, {feature_dim}] "
            f"for {source_path}, got {array.shape}"
        )
    if array.dtype != np.float32:
        raise ValueError(
            f"Background fbank dtype must be float32 for {source_path}, got {array.dtype}"
        )
    if not bool(np.isfinite(array).all()):
        raise ValueError(f"Non-finite background fbank for {source_path}")


def _extract_source_crops(
    job: SourceJob,
    *,
    extractor: FbankExtractor | None = None,
) -> SourceCrops:
    path = Path(job.path)
    if extractor is None:
        extractor = _worker_extractor()
    try:
        source = probe_source_info(path, source_id=job.source_id)
    except Exception as exc:
        raise RuntimeError(
            f"Could not read background source {job.source_id}: {path}"
        ) from exc
    content_sha256, _size = _sha256_file(path)
    specs: list[BackgroundCropSpec] = []
    features: list[np.ndarray] = []
    for ordinal in range(job.crops_per_recording):
        rng = crop_rng(job.seed, job.source_id, ordinal)
        duration = rng.uniform(job.duration_seconds_min, job.duration_seconds_max)
        spec = draw_crop_spec(source, duration, rng=rng)
        try:
            waveform, sample_rate = materialize_crop(source, spec)
            feat = extractor.extract(waveform, sample_rate)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to materialize/extract crop {ordinal} from {path}"
            ) from exc
        array = np.ascontiguousarray(feat.detach().cpu().numpy(), dtype=np.float32)
        _validate_feature(array, feature_dim=job.feature_dim, source_path=path)
        specs.append(spec)
        features.append(array)
    return SourceCrops(
        source_index=job.source_index,
        source_id=job.source_id,
        list_entry=job.list_entry,
        relative_path=job.source_id,
        sample_rate=int(source.sample_rate),
        num_frames=int(source.num_frames),
        channels=int(source.channels),
        content_sha256=content_sha256,
        specs=tuple(specs),
        features=tuple(features),
    )


def _worker_extract(job: SourceJob) -> SourceCrops:
    return _extract_source_crops(job)


def _iter_ordered_pool_results(
    n_jobs: int,
    *,
    workers: int,
    submit: Callable[[int], Any],
    collect: Callable[[Any, int], Any],
    wait_completed: Callable[[Mapping[Any, int]], Iterable[Any]],
) -> Iterator[Any]:
    """Yield job results in index order with a bounded submit window.

    At most ``workers`` jobs may be submitted ahead of the next index to
    yield. Completed-but-unconsumed results therefore cannot grow with N:
    ``len(in_flight) + len(ready) <= workers``.
    """
    next_submit = 0
    next_yield = 0
    in_flight: dict[Any, int] = {}
    ready: dict[int, Any] = {}
    while next_yield < n_jobs:
        while next_submit < n_jobs and next_submit < next_yield + workers:
            handle = submit(next_submit)
            in_flight[handle] = next_submit
            next_submit += 1
        if not in_flight:
            raise RuntimeError("Background crop workers produced no in-flight jobs")
        for handle in wait_completed(in_flight):
            index = in_flight.pop(handle)
            ready[index] = collect(handle, index)
        if len(in_flight) + len(ready) > workers:
            raise RuntimeError(
                "in-flight background-cache results exceeded the worker window"
            )
        while next_yield in ready:
            yield ready.pop(next_yield)
            next_yield += 1


def _iter_source_results(
    jobs: Sequence[SourceJob],
    *,
    workers: int,
    fbank_params: Mapping[str, Any],
) -> Iterator[SourceCrops]:
    if workers == 1:
        _configure_torch_threads()
        extractor = FbankExtractor(**dict(fbank_params))
        for job in jobs:
            yield _extract_source_crops(job, extractor=extractor)
        return

    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=context,
        initializer=_init_worker,
        initargs=(dict(fbank_params),),
    ) as executor:

        def submit(index: int) -> Any:
            return executor.submit(_worker_extract, jobs[index])

        def collect(handle: Any, index: int) -> SourceCrops:
            try:
                return handle.result()
            except Exception as exc:
                job = jobs[index]
                raise RuntimeError(
                    f"Failed to extract background crops from {job.path}"
                ) from exc

        def wait_completed(in_flight: Mapping[Any, int]) -> Iterable[Any]:
            done, _pending = wait(tuple(in_flight), return_when=FIRST_COMPLETED)
            return done

        yield from _iter_ordered_pool_results(
            len(jobs),
            workers=workers,
            submit=submit,
            collect=collect,
            wait_completed=wait_completed,
        )


def _max_frames_per_shard(shard_size_mib: int, feature_dim: int) -> int:
    return max(
        1,
        (int(shard_size_mib) * 1024 * 1024) // (int(feature_dim) * BYTES_PER_FLOAT32),
    )


def _shard_name(index: int) -> str:
    return f"features-{index:05d}.npy"


class _ShardWriter:
    def __init__(
        self,
        staging: Path,
        *,
        feature_dim: int,
        max_frames: int,
    ) -> None:
        self.staging = staging
        self.feature_dim = int(feature_dim)
        self.max_frames = int(max_frames)
        self._buffer: list[np.ndarray] = []
        self._buffer_frames = 0
        self._crops_in_shard = 0
        self.shard_index = 0
        self.records: list[dict[str, Any]] = []

    def add(self, features: np.ndarray) -> tuple[int, int]:
        frames = int(features.shape[0])
        if self._buffer and self._buffer_frames + frames > self.max_frames:
            self.flush()
        frame_offset = self._buffer_frames
        self._buffer.append(features)
        self._buffer_frames += frames
        self._crops_in_shard += 1
        return self.shard_index, frame_offset

    def flush(self) -> None:
        if not self._buffer:
            return
        array = np.ascontiguousarray(
            np.concatenate(self._buffer, axis=0), dtype=np.float32
        )
        path = self.staging / _shard_name(self.shard_index)
        _save_npy(path, array)
        digest, size = _sha256_file(path)
        self.records.append(
            {
                "path": path.name,
                "sha256": digest,
                "size": size,
                "num_frames": int(array.shape[0]),
                "num_crops": self._crops_in_shard,
                "feature_dim": self.feature_dim,
            }
        )
        self.shard_index += 1
        self._buffer = []
        self._buffer_frames = 0
        self._crops_in_shard = 0

    def close(self) -> list[dict[str, Any]]:
        self.flush()
        return self.records


def _print_capacity_plan(
    *,
    num_sources: int,
    crops_per_recording: int,
    fbank: Mapping[str, Any],
    duration_min: float,
    duration_max: float,
    config: Mapping[str, Any],
) -> None:
    feature_dim = int(fbank["num_mel_bins"])
    frame_shift = float(fbank["frame_shift"])
    bytes_per_second = feature_dim * BYTES_PER_FLOAT32 * (1000.0 / frame_shift)
    mean_duration = 0.5 * (float(duration_min) + float(duration_max))
    bytes_per_crop = bytes_per_second * mean_duration
    total_bytes = num_sources * crops_per_recording * bytes_per_crop
    print(
        "Estimated feature capacity: "
        f"{num_sources} sources × {crops_per_recording} crops × "
        f"~{bytes_per_crop / 1024.0:.0f} kB/crop ≈ {total_bytes / (1024.0 * 1024.0):.1f} MiB "
        f"(~{bytes_per_second / 1024.0:.0f} kB/s at {feature_dim}-dim / "
        f"{frame_shift:g} ms shift FP32; mean {mean_duration:g} s), plus index."
    )
    stage2 = config.get("stage2") if isinstance(config.get("stage2"), Mapping) else {}
    background = (
        stage2.get("background_negative")
        if isinstance(stage2.get("background_negative"), Mapping)
        else {}
    )
    steps = int(stage2.get("max_steps") or 0)
    accum = int(stage2.get("accumulate_grad_batches") or 1)
    batch = int(stage2.get("batch_size_per_gpu") or 0)
    probability = float(background.get("probability") or 0.0)
    world_size = 1
    sampling_count = steps * accum * batch * world_size * 0.5 * probability
    print(
        "Sampling-count estimate: "
        f"{steps} steps × {accum} accum × {batch} batch/gpu × {world_size} world × "
        f"0.5 × {probability:g} probability ≈ {sampling_count:.0f} background draws "
        "(world_size assumed 1)."
    )
    suggested = max(128, math.ceil(2 * 160000 / max(num_sources, 1)))
    print(
        f"Suggested K: max(128, ceil(2*160000/N)) = {suggested} for the default "
        f"10k/128/accum=1 case (N={num_sources}). Not applied; using "
        f"--crops-per-recording={crops_per_recording}."
    )


def _duration_bounds(config: Mapping[str, Any]) -> tuple[float, float]:
    stage2 = config.get("stage2")
    if not isinstance(stage2, Mapping):
        raise ValueError("Composed config is missing stage2")
    background = stage2.get("background_negative")
    if not isinstance(background, Mapping):
        raise ValueError("Composed config is missing stage2.background_negative")
    duration_min = float(background.get("duration_seconds_min", 1.0))
    duration_max = float(background.get("duration_seconds_max", 3.0))
    if not math.isfinite(duration_min) or not math.isfinite(duration_max):
        raise ValueError("stage2.background_negative duration bounds must be finite")
    if duration_min <= 0.0:
        raise ValueError("stage2.background_negative.duration_seconds_min must be positive")
    if duration_min > duration_max:
        raise ValueError(
            "stage2.background_negative.duration_seconds_min must be <= duration_seconds_max"
        )
    return duration_min, duration_max


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


def _file_record(path: Path, *, relative_name: str) -> dict[str, Any]:
    digest, size = _sha256_file(path)
    return {"path": relative_name, "sha256": digest, "size": size}


def _check_digest(path: Path, expected_sha256: str, expected_size: int) -> None:
    digest, size = _sha256_file(path)
    if digest != expected_sha256 or size != int(expected_size):
        raise ValueError(
            f"Checksum mismatch for {path.name}: expected sha256={expected_sha256} "
            f"size={expected_size}, got sha256={digest} size={size}"
        )


def verify_stage2_background_cache(output_dir: str | Path) -> dict[str, Any]:
    """Re-hash index and shards and check shapes/offsets/finite values."""
    cache_dir = Path(output_dir).expanduser().resolve()
    manifest_path = cache_dir / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Background cache manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError(f"manifest.json must be a mapping: {manifest_path}")
    if int(manifest.get("format_version", -1)) != FORMAT_VERSION:
        raise ValueError(
            f"Unsupported background cache format_version={manifest.get('format_version')!r}"
        )
    if manifest.get("split_role") != "train":
        raise ValueError(
            f"Background cache split_role must be 'train', got {manifest.get('split_role')!r}"
        )
    if manifest.get("dtype") != "float32":
        raise ValueError(f"Background cache dtype must be float32, got {manifest.get('dtype')!r}")

    expected_id = _cache_id_for(manifest)
    stamped = manifest.get("cache_id")
    if stamped != expected_id:
        raise ValueError(
            f"manifest cache_id mismatch: stamped {stamped!r}, recomputed {expected_id!r}"
        )

    recordings_spec = manifest.get("recordings")
    crops_spec = manifest.get("crops")
    shard_specs = manifest.get("shards")
    if not isinstance(recordings_spec, Mapping) or not isinstance(crops_spec, Mapping):
        raise ValueError("manifest.json must include recordings and crops file records")
    if not isinstance(shard_specs, list) or not shard_specs:
        raise ValueError("manifest.json shards list is empty")

    recordings_path = cache_dir / str(recordings_spec["path"])
    crops_path = cache_dir / str(crops_spec["path"])
    _check_digest(
        recordings_path,
        str(recordings_spec["sha256"]),
        int(recordings_spec["size"]),
    )
    _check_digest(crops_path, str(crops_spec["sha256"]), int(crops_spec["size"]))

    feature_dim = int(manifest["feature_dim"])
    crops = np.load(crops_path, allow_pickle=False)
    if crops.dtype != CROP_DTYPE:
        # Accept equivalent dtypes with the same field names/order.
        if getattr(crops.dtype, "names", None) != CROP_DTYPE.names:
            raise ValueError(f"crops.npy dtype fields are {crops.dtype.names!r}")
    if int(manifest["num_crops"]) != len(crops):
        raise ValueError(
            f"crops.npy length {len(crops)} != manifest num_crops {manifest['num_crops']}"
        )

    seen_shard_names: list[str] = []
    for shard_spec in shard_specs:
        if not isinstance(shard_spec, Mapping):
            raise ValueError("manifest shard records must be mappings")
        shard_path = cache_dir / str(shard_spec["path"])
        seen_shard_names.append(shard_path.name)
        _check_digest(shard_path, str(shard_spec["sha256"]), int(shard_spec["size"]))
        array = np.load(shard_path, mmap_mode="r", allow_pickle=False)
        if array.dtype != np.float32 or array.ndim != 2:
            raise ValueError(
                f"{shard_path.name} must be float32 [frames, dim], got "
                f"dtype={array.dtype} shape={array.shape}"
            )
        if int(array.shape[1]) != feature_dim:
            raise ValueError(
                f"{shard_path.name} feature_dim {array.shape[1]} != {feature_dim}"
            )
        if int(array.shape[0]) != int(shard_spec["num_frames"]):
            raise ValueError(
                f"{shard_path.name} num_frames {array.shape[0]} != {shard_spec['num_frames']}"
            )
        if not array.flags["C_CONTIGUOUS"]:
            raise ValueError(f"{shard_path.name} is not C-contiguous")
        if not bool(np.isfinite(array).all()):
            raise ValueError(f"Non-finite values in {shard_path.name}")

    on_disk = sorted(path.name for path in cache_dir.glob("features-*.npy"))
    if on_disk != sorted(seen_shard_names):
        raise ValueError(
            f"Shard files on disk {on_disk} do not match manifest {seen_shard_names}"
        )

    by_shard: dict[int, list[np.void]] = {}
    for row in crops:
        by_shard.setdefault(int(row["shard_index"]), []).append(row)
    for shard_index, shard_spec in enumerate(shard_specs):
        rows = sorted(
            by_shard.get(shard_index, []),
            key=lambda row: int(row["frame_offset"]),
        )
        cursor = 0
        for row in rows:
            offset = int(row["frame_offset"])
            frames = int(row["num_frames"])
            if frames < 1:
                raise ValueError(f"Crop {int(row['crop_id'])} has no feature frames")
            if offset != cursor:
                raise ValueError(
                    f"Shard {shard_index} crop {int(row['crop_id'])} frame_offset "
                    f"{offset} != {cursor}"
                )
            cursor += frames
        if cursor != int(shard_spec["num_frames"]):
            raise ValueError(
                f"Shard {shard_index} indexed frames {cursor} != {shard_spec['num_frames']}"
            )

    recordings = _read_jsonl(recordings_path)
    if len(recordings) != int(manifest["num_sources"]):
        raise ValueError(
            f"recordings.jsonl length {len(recordings)} != num_sources {manifest['num_sources']}"
        )
    expected_ids: list[str] = []
    for record in recordings:
        start = int(record["crop_start"])
        count = int(record["crop_count"])
        source_index = int(record["source_index"])
        source_id = str(record["source_id"])
        expected_ids.append(source_id)
        slice_rows = crops[start : start + count]
        if len(slice_rows) != count:
            raise ValueError(f"recordings.jsonl crop window out of range for {source_id}")
        if any(int(row["source_index"]) != source_index for row in slice_rows):
            raise ValueError(f"Crop source_index mismatch for {source_id}")
        if [int(row["ordinal"]) for row in slice_rows] != list(range(count)):
            raise ValueError(f"Crop ordinals are not 0..K-1 for {source_id}")
    if expected_ids != sorted(expected_ids):
        raise ValueError("recordings.jsonl is not sorted by source_id")

    return {
        "ok": True,
        "output_dir": str(cache_dir),
        "cache_id": str(stamped),
        "format_version": FORMAT_VERSION,
        "num_sources": int(manifest["num_sources"]),
        "num_crops": int(manifest["num_crops"]),
        "num_shards": len(shard_specs),
        "K": int(manifest["K"]),
    }


def _build_jobs(
    train_rows: Sequence[tuple[str, str, Path]],
    *,
    seed: int,
    crops_per_recording: int,
    duration_min: float,
    duration_max: float,
    feature_dim: int,
) -> list[SourceJob]:
    ordered = sorted(train_rows, key=lambda item: item[0])
    return [
        SourceJob(
            source_index=index,
            source_id=source_id,
            path=str(path),
            list_entry=list_entry,
            seed=seed,
            crops_per_recording=crops_per_recording,
            duration_seconds_min=duration_min,
            duration_seconds_max=duration_max,
            feature_dim=feature_dim,
        )
        for index, (source_id, list_entry, path) in enumerate(ordered)
    ]


def _write_cache(
    staging: Path,
    *,
    jobs: Sequence[SourceJob],
    workers: int,
    fbank_params: Mapping[str, Any],
    feature_dim: int,
    shard_size_mib: int,
    split_document: Mapping[str, Any],
    split_dir: Path,
    musan_root: Path,
    train_categories: Sequence[str],
    experiment: str,
    overrides: Sequence[str],
    seed: int,
    crops_per_recording: int,
    duration_min: float,
    duration_max: float,
    train_list_path: Path,
    eval_list_path: Path,
) -> dict[str, Any]:
    writer = _ShardWriter(
        staging,
        feature_dim=feature_dim,
        max_frames=_max_frames_per_shard(shard_size_mib, feature_dim),
    )
    crop_rows = np.empty(len(jobs) * crops_per_recording, dtype=CROP_DTYPE)
    recordings_path = staging / RECORDINGS_NAME
    crop_id = 0
    with recordings_path.open("w", encoding="utf-8", newline="\n") as handle:
        for result in _iter_source_results(
            jobs, workers=workers, fbank_params=fbank_params
        ):
            crop_start = crop_id
            record = {
                "source_index": result.source_index,
                "source_id": result.source_id,
                "relative_path": result.relative_path,
                "list_entry": result.list_entry,
                "sample_rate": result.sample_rate,
                "num_frames": result.num_frames,
                "channels": result.channels,
                "content_sha256": result.content_sha256,
                "crop_start": crop_start,
                "crop_count": len(result.specs),
            }
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            for ordinal, spec, features in zip(
                range(len(result.specs)), result.specs, result.features
            ):
                shard_index, frame_offset = writer.add(features)
                row = crop_rows[crop_id]
                row["crop_id"] = crop_id
                row["source_index"] = result.source_index
                row["ordinal"] = ordinal
                row["source_id"] = spec.source_id
                row["duration_seconds"] = spec.duration_seconds
                row["read_start_frame"] = spec.read_start_frame
                row["read_num_frames"] = spec.read_num_frames
                row["target_num_samples"] = spec.target_num_samples
                row["final_offset"] = spec.final_offset
                row["shard_index"] = shard_index
                row["frame_offset"] = frame_offset
                row["num_frames"] = int(features.shape[0])
                crop_id += 1
        handle.flush()
        os.fsync(handle.fileno())
    shard_records = writer.close()
    if crop_id != len(crop_rows):
        raise RuntimeError(
            f"Wrote {crop_id} crops, expected {len(crop_rows)}"
        )
    if not shard_records:
        raise RuntimeError("Background cache produced no feature shards")

    crops_path = staging / CROPS_NAME
    _save_npy(crops_path, crop_rows)

    split_json_path = split_dir / SPLIT_JSON_NAME
    recordings_record = _file_record(recordings_path, relative_name=RECORDINGS_NAME)
    crops_record = _file_record(crops_path, relative_name=CROPS_NAME)
    train_digest, train_size = _sha256_file(train_list_path)
    eval_digest, eval_size = _sha256_file(eval_list_path)
    split_digest, split_size = _sha256_file(split_json_path)
    train_side = _split_side(split_document, "train")
    eval_side = _split_side(split_document, "eval")
    normalized_fbank = _normalize_fbank(fbank_params)

    manifest: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "split_role": "train",
        "seed": seed,
        "K": crops_per_recording,
        "num_sources": len(jobs),
        "num_crops": len(crop_rows),
        "duration_seconds_min": duration_min,
        "duration_seconds_max": duration_max,
        "dtype": "float32",
        "feature_dim": feature_dim,
        "fbank": normalized_fbank,
        "experiment": experiment,
        "overrides": list(overrides),
        "shard_size_mib": shard_size_mib,
        "feature_lib": {
            "extractor": "dma_kws.stage2.fbank.FbankExtractor",
            "backend": normalized_fbank.get("backend"),
        },
        "dependency_versions": _dependency_versions(),
        "split": {
            "musan_root": str(musan_root),
            "schema_version": split_document.get("schema_version"),
            "split_json": SPLIT_JSON_NAME,
            "split_json_sha256": split_digest,
            "split_json_size": split_size,
            "train_list": train_list_path.name,
            "train_list_sha256": train_digest,
            "train_list_size": train_size,
            "eval_list": eval_list_path.name,
            "eval_list_sha256": eval_digest,
            "eval_list_size": eval_size,
            "train_catalog_sha256": train_side.get("catalog_sha256"),
            "eval_catalog_sha256": eval_side.get("catalog_sha256"),
            "train_recordings": int(train_side.get("recordings", len(jobs))),
            "eval_recordings": int(eval_side.get("recordings", 0)),
            "train_categories": list(train_categories),
        },
        "recordings": recordings_record,
        "crops": crops_record,
        "shards": shard_records,
        "build": {
            "built_at": datetime.now(timezone.utc).isoformat(),
            "workers": workers,
        },
    }
    manifest["cache_id"] = _cache_id_for(manifest)
    _write_text(
        staging / MANIFEST_NAME,
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    return manifest


def prepare_stage2_background(
    *,
    split_dir: str | Path,
    output_dir: str | Path,
    experiment: str = DEFAULT_EXPERIMENT,
    overrides: Sequence[str] = (),
    crops_per_recording: int = DEFAULT_CROPS_PER_RECORDING,
    seed: int = DEFAULT_SEED,
    workers: int = DEFAULT_WORKERS,
    shard_size_mib: int = DEFAULT_SHARD_SIZE_MIB,
    verify_only: bool = False,
) -> dict[str, Any]:
    """Build or verify a format-v1 Stage II background fbank crop cache."""
    crops_per_recording = _require_int(
        crops_per_recording, field="crops_per_recording", minimum=1
    )
    seed = _require_int(seed, field="seed", minimum=0)
    workers = _require_int(workers, field="workers", minimum=1)
    shard_size_mib = _require_int(shard_size_mib, field="shard_size_mib", minimum=1)

    destination = Path(output_dir).expanduser().resolve()
    if verify_only:
        return verify_stage2_background_cache(destination)

    split_root = Path(split_dir).expanduser().resolve()
    override_list = [str(item) for item in overrides]
    composed = config_to_dict(compose_config(experiment, override_list))
    fbank_params = _normalize_fbank(fbank_kwargs(get_fbank_config(composed)))
    if float(fbank_params["dither"]) != 0.0:
        raise ValueError(
            "fbank_cache v1 accepts dither=0 only; composed fbank dither is "
            f"{fbank_params['dither']!r}. Pass an experiment with dither=0 "
            "(do not silently zero it)."
        )
    duration_min, duration_max = _duration_bounds(composed)
    feature_dim = int(fbank_params["num_mel_bins"])

    document = _load_split(split_root)
    musan_root = Path(str(document.get("musan_root", ""))).expanduser()
    if not str(document.get("musan_root", "")).strip():
        raise ValueError("split.json missing musan_root")
    train_categories = _validate_train_categories(document)
    train_side = _split_side(document, "train")
    eval_side = _split_side(document, "eval")
    train_list_path = split_root / str(train_side.get("list") or TRAIN_LIST_NAME)
    eval_list_path = split_root / str(eval_side.get("list") or EVAL_LIST_NAME)

    train_rows = _resolve_list_sources(
        train_list_path,
        musan_root=musan_root,
        expected_count=int(train_side.get("recordings", -1)),
        expected_catalog_sha256=str(train_side.get("catalog_sha256") or ""),
        label="train",
    )
    eval_rows = _resolve_list_sources(
        eval_list_path,
        musan_root=musan_root,
        expected_count=int(eval_side.get("recordings", -1)),
        expected_catalog_sha256=str(eval_side.get("catalog_sha256") or ""),
        label="eval",
    )
    train_paths = {path.resolve() for _sid, _entry, path in train_rows}
    eval_paths = {path.resolve() for _sid, _entry, path in eval_rows}
    overlap = train_paths & eval_paths
    if overlap:
        sample = ", ".join(sorted(str(path) for path in overlap)[:5])
        raise ValueError(f"Train and eval recordings overlap: {sample}")
    _validate_no_speech(train_rows, label="Train background list")

    if destination.exists():
        raise FileExistsError(
            f"Refusing to overwrite an existing Stage II background cache: {destination}"
        )

    jobs = _build_jobs(
        train_rows,
        seed=seed,
        crops_per_recording=crops_per_recording,
        duration_min=duration_min,
        duration_max=duration_max,
        feature_dim=feature_dim,
    )
    _print_capacity_plan(
        num_sources=len(jobs),
        crops_per_recording=crops_per_recording,
        fbank=fbank_params,
        duration_min=duration_min,
        duration_max=duration_max,
        config=composed,
    )
    for job in jobs:
        try:
            probe_source_info(job.path, source_id=job.source_id)
        except Exception as exc:
            raise RuntimeError(
                f"Could not read background source {job.source_id}: {job.path}"
            ) from exc

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(dir=destination.parent, prefix=f".{destination.name}.")
    )
    try:
        _write_cache(
            staging,
            jobs=jobs,
            workers=workers,
            fbank_params=fbank_params,
            feature_dim=feature_dim,
            shard_size_mib=shard_size_mib,
            split_document=document,
            split_dir=split_root,
            musan_root=musan_root.resolve(),
            train_categories=train_categories,
            experiment=str(experiment),
            overrides=override_list,
            seed=seed,
            crops_per_recording=crops_per_recording,
            duration_min=duration_min,
            duration_max=duration_max,
            train_list_path=train_list_path,
            eval_list_path=eval_list_path,
        )
        verify_stage2_background_cache(staging)
        os.replace(staging, destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)

    manifest = json.loads((destination / MANIFEST_NAME).read_text(encoding="utf-8"))
    return {
        "output_dir": str(destination),
        "cache_id": manifest["cache_id"],
        "format_version": FORMAT_VERSION,
        "num_sources": manifest["num_sources"],
        "num_crops": manifest["num_crops"],
        "num_shards": len(manifest["shards"]),
        "K": manifest["K"],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Pre-generate the Stage II background fbank crop cache."
    )
    parser.add_argument(
        "--split-dir",
        type=Path,
        required=True,
        help="Existing MUSAN split directory (split.json + train/eval lists)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="New cache directory; refused if it already exists",
    )
    parser.add_argument(
        "--experiment",
        default=DEFAULT_EXPERIMENT,
        help=f"Hydra experiment name (default: {DEFAULT_EXPERIMENT})",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Repeatable Hydra override forwarded to compose_config",
    )
    parser.add_argument(
        "--crops-per-recording",
        type=_parse_int_at_least(1),
        default=DEFAULT_CROPS_PER_RECORDING,
        help=f"K crops per recording (default: {DEFAULT_CROPS_PER_RECORDING})",
    )
    parser.add_argument(
        "--seed",
        type=_parse_int_at_least(0),
        default=DEFAULT_SEED,
        help=f"Crop RNG seed (default: {DEFAULT_SEED})",
    )
    parser.add_argument(
        "--workers",
        type=_parse_int_at_least(1),
        default=DEFAULT_WORKERS,
        help=f"Process workers (default: {DEFAULT_WORKERS})",
    )
    parser.add_argument(
        "--shard-size-mib",
        type=_parse_int_at_least(1),
        default=DEFAULT_SHARD_SIZE_MIB,
        help=f"Feature shard size in MiB (default: {DEFAULT_SHARD_SIZE_MIB})",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Validate an existing output-dir and exit; do not generate",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = build_parser().parse_args(argv)
    summary = prepare_stage2_background(
        split_dir=args.split_dir,
        output_dir=args.output_dir,
        experiment=args.experiment,
        overrides=tuple(args.override or ()),
        crops_per_recording=args.crops_per_recording,
        seed=args.seed,
        workers=args.workers,
        shard_size_mib=args.shard_size_mib,
        verify_only=bool(args.verify_only),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return summary


if __name__ == "__main__":
    main()
