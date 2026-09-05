"""Reproducible Stage II input-path benchmark (background / loader / train).

The three modes are independent measurements. This module never collapses them
into a single speedup number, and train mode refuses a missing checkpoint
rather than reporting toy-encoder throughput.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
import json
import os
from pathlib import Path
import random
import resource
import subprocess
import sys
import time
from typing import Any

import numpy as np
import torch

from dma_kws.config import (
    compose_config,
    config_to_dict,
    fbank_kwargs,
    get_fbank_config,
    get_tokenizer_config,
)
from dma_kws.data_prep.stage2_background import DEFAULT_EXPERIMENT
from dma_kws.pathing import PROJECT_ROOT, resolve_dict_path
from dma_kws.stage2.train_data import (
    build_stage2_train_dataloader,
    build_stage2_train_dataset,
)
from dma_kws.tokenizer import load_char_tokenizer
from dma_kws.training.ddp import resolve_precision


class InputBenchmarkError(Exception):
    """Fatal benchmark configuration error."""


class TrainModeUnavailable(InputBenchmarkError):
    """Train mode cannot produce an official throughput result."""


def padding_ratio(lengths: torch.Tensor | Sequence[int]) -> float:
    """Return ``1 - sum(lengths) / (B * max_length)`` for one batch."""
    if not torch.is_tensor(lengths):
        tensor = torch.as_tensor(list(lengths), dtype=torch.float64)
    else:
        tensor = lengths.detach().to(dtype=torch.float64).reshape(-1)
    if tensor.numel() == 0:
        return 0.0
    max_length = float(tensor.max().item())
    if max_length <= 0.0:
        return 0.0
    return float(1.0 - float(tensor.sum().item()) / (float(tensor.numel()) * max_length))


def align_up(value: int, multiple: int) -> int:
    """Raise ``value`` to the next multiple of ``multiple`` (0 stays 0)."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"count must be a non-negative int, got {value!r}")
    if isinstance(multiple, bool) or not isinstance(multiple, int) or multiple < 1:
        raise ValueError(f"alignment multiple must be a positive int, got {multiple!r}")
    if value == 0:
        return 0
    remainder = value % multiple
    if remainder == 0:
        return value
    return value + (multiple - remainder)


def _require_int(value: Any, *, field: str, minimum: int) -> int:
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


def _git_commit() -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(PROJECT_ROOT),
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return "unavailable"
    if completed.returncode != 0:
        return "unavailable"
    return completed.stdout.strip() or "unavailable"


def process_rss_bytes() -> int:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return int(usage)
    return int(usage) * 1024


def gpu_util() -> Any:
    """Return nvidia-smi GPU util, or ``\"unavailable\"`` — never a fake 0."""
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return "unavailable"
    if completed.returncode != 0:
        return "unavailable"
    values: list[float] = []
    for line in completed.stdout.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        token = stripped.split(",")[0].strip()
        try:
            values.append(float(token))
        except ValueError:
            return "unavailable"
    if not values:
        return "unavailable"
    if len(values) == 1:
        return values[0]
    return values


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(_json_safe(payload), ensure_ascii=False, indent=2, sort_keys=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text + "\n", encoding="utf-8")
    temporary.replace(path)


def _percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(item) for item in values)
    if len(ordered) == 1:
        return ordered[0]
    index = (len(ordered) - 1) * q
    low = int(index)
    high = min(low + 1, len(ordered) - 1)
    frac = index - low
    return ordered[low] * (1.0 - frac) + ordered[high] * frac


def _frame_length_stats(lengths: Sequence[int]) -> dict[str, float | int]:
    if not lengths:
        return {"min": 0, "max": 0, "mean": 0.0, "p50": 0.0, "p95": 0.0, "count": 0}
    numeric = [float(item) for item in lengths]
    return {
        "min": int(min(lengths)),
        "max": int(max(lengths)),
        "mean": float(sum(numeric) / len(numeric)),
        "p50": _percentile(numeric, 0.5),
        "p95": _percentile(numeric, 0.95),
        "count": int(len(lengths)),
    }


def _synchronize(device: str) -> None:
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()


def _seed_all(config: dict[str, Any]) -> int:
    seed = int((config.get("training") or {}).get("seed", 2025))
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    return seed


def _load_tokenizer(config: dict[str, Any]):
    tokenizer_cfg = get_tokenizer_config(config)
    dict_path = resolve_dict_path(config)
    return load_char_tokenizer(
        dict_path,
        split_with_space=tokenizer_cfg.get("split_with_space", " "),
    )


def _cache_id_from_config(config: dict[str, Any], cache: Any | None = None) -> str | None:
    if cache is not None:
        return str(getattr(cache, "cache_id", "") or "") or None
    background = (config.get("stage2") or {}).get("background_negative") or {}
    manifest = str(background.get("cache_manifest") or "").strip()
    if not manifest:
        return None
    path = Path(manifest)
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    cache_id = payload.get("cache_id")
    return str(cache_id) if cache_id else None


def _loader_settings(
    config: dict[str, Any], *, device: str = "cpu"
) -> tuple[int, int, int, int, str]:
    stage1 = config.get("stage1") or {}
    stage2 = config.get("stage2") or {}
    dataloader_cfg = stage2.get("dataloader") or {}
    batch_size = int(stage2.get("batch_size_per_gpu", 64))
    num_workers = int(stage2.get("num_workers", stage1.get("num_workers", 2)))
    prefetch = (
        int(dataloader_cfg.get("prefetch_factor", 4)) if num_workers > 0 else 0
    )
    accumulate = max(1, int(stage2.get("accumulate_grad_batches", 1)))
    accelerator = "gpu" if device == "cuda" else "cpu"
    return (
        batch_size,
        num_workers,
        prefetch,
        accumulate,
        str(resolve_precision(stage2, accelerator)),
    )


def _frame_shift_seconds(config: dict[str, Any]) -> float:
    return float(get_fbank_config(config).frame_shift) / 1000.0


def _effective_audio_rate(
    lengths: Sequence[int], config: dict[str, Any], wall_seconds: float
) -> float:
    audio_seconds = float(sum(int(item) for item in lengths)) * _frame_shift_seconds(
        config
    )
    return audio_seconds / max(float(wall_seconds), 1e-9)


def _composition_from_batch(batch: dict[str, Any]) -> tuple[int, int, int, list[int]]:
    labels = batch["label"].detach().cpu()
    lengths_tensor = batch["feat_lengths"].detach().cpu()
    query_lengths = batch.get("query_lengths")
    positives = int((labels == 1).sum().item())
    if query_lengths is not None:
        query = query_lengths.detach().cpu()
        background = int(((labels == 0) & (query == 0)).sum().item())
        negatives = int(((labels == 0) & (query > 0)).sum().item())
    else:
        background = 0
        negatives = int((labels == 0).sum().item())
    lengths = [int(item) for item in lengths_tensor.tolist()]
    return positives, negatives, background, lengths


def _iter_forever(loader) -> Iterator[Any]:
    while True:
        yielded = 0
        for batch in loader:
            yielded += 1
            yield batch
        if yielded == 0:
            raise RuntimeError(
                "Stage II train DataLoader produced no batches "
                "(drop_last=True requires len(dataset) >= batch_size)"
            )


def _next_timed(iterator: Iterator[Any]) -> tuple[Any, float]:
    started = time.perf_counter()
    batch = next(iterator)
    return batch, time.perf_counter() - started


@contextmanager
def _profile_cm(enabled: bool, output: Path, device: str):
    info: dict[str, Any] = {}
    if not enabled:
        yield info
        return
    from torch.profiler import ProfilerActivity, profile

    activities = [ProfilerActivity.CPU]
    if device == "cuda" and torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)
    with profile(activities=activities) as prof:
        yield info
    trace_path = output.with_name(f"{output.stem}.profile.json")
    prof.export_chrome_trace(str(trace_path))
    info["trace_path"] = str(trace_path)
    info["enabled"] = True


def _base_payload(
    *,
    mode: str,
    config: dict[str, Any],
    device: str,
    warmup_batches: int,
    measure_batches: int,
    batch_size: int,
    workers: int,
    prefetch: int,
    precision: str,
    accumulate: int,
    cache_id: str | None,
) -> dict[str, Any]:
    cuda_version = getattr(torch.version, "cuda", None)
    return {
        "mode": mode,
        "status": "ok",
        "git_commit": _git_commit(),
        "config": _json_safe(config),
        "cache_id": cache_id,
        "pytorch_version": str(torch.__version__),
        "cuda_version": cuda_version,
        "cuda_available": bool(torch.cuda.is_available()),
        "device": device,
        "cpu_cores": int(os.cpu_count() or 0),
        "workers": int(workers),
        "prefetch": int(prefetch),
        "batch_size": int(batch_size),
        "precision": precision,
        "accumulate_grad_batches": int(accumulate),
        "warmup_batches_requested": int(warmup_batches),
        "measure_batches_requested": int(measure_batches),
        "gpu_util": gpu_util(),
        "process_rss_bytes": process_rss_bytes(),
    }


def _finish_payload(
    payload: dict[str, Any],
    *,
    warmup_actual: int,
    measure_actual: int,
    optimizer_requested: int,
    optimizer_actual: int,
    first_next_seconds: float,
    warmup_seconds: float,
    measured_wall_seconds: float,
    waits: Sequence[float],
    lengths: Sequence[int],
    padding_ratios: Sequence[float],
    positive_count: int,
    negative_count: int,
    background_count: int,
    samples: int,
    config: dict[str, Any],
    cache_hits: Any = 0,
    cache_misses: Any = 0,
    open_shard_high_water: Any = 0,
) -> dict[str, Any]:
    wait_ms = [float(item) * 1000.0 for item in waits]
    wall = max(float(measured_wall_seconds), 1e-9)
    payload.update(
        {
            "warmup_batches_actual": int(warmup_actual),
            "measure_batches_actual": int(measure_actual),
            "optimizer_steps_requested": int(optimizer_requested),
            "optimizer_steps_actual": int(optimizer_actual),
            "first_next_seconds": float(first_next_seconds),
            "warmup_seconds": float(warmup_seconds),
            "measured_wall_seconds": float(measured_wall_seconds),
            "wait_p50_ms": _percentile(wait_ms, 0.5),
            "wait_p95_ms": _percentile(wait_ms, 0.95),
            "samples_per_second": float(samples) / wall,
            "effective_audio_seconds_per_second": _effective_audio_rate(
                lengths, config, measured_wall_seconds
            ),
            "frame_length": _frame_length_stats(lengths),
            "padding_ratio": (
                float(sum(padding_ratios) / len(padding_ratios))
                if padding_ratios
                else 0.0
            ),
            "positive_count": int(positive_count),
            "negative_count": int(negative_count),
            "background_count": int(background_count),
            "cache_hits": cache_hits,
            "cache_misses": cache_misses,
            "open_shard_high_water": open_shard_high_water,
            "process_rss_bytes": process_rss_bytes(),
            "gpu_util": gpu_util(),
        }
    )
    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                payload = json.loads(stripped)
                if isinstance(payload, dict):
                    rows.append(payload)
    return rows


def _resolve_source_path(record: dict[str, Any], split_dir: Path) -> Path:
    split_path = split_dir / "split.json"
    musan_root = Path()
    if split_path.is_file():
        document = json.loads(split_path.read_text(encoding="utf-8"))
        if isinstance(document, dict) and document.get("musan_root"):
            musan_root = Path(str(document["musan_root"])).expanduser()
    source_id = str(record.get("source_id") or "")
    relative = str(record.get("relative_path") or source_id)
    list_entry = str(record.get("list_entry") or "")
    candidates: list[Path] = []
    if str(musan_root):
        candidates.append(musan_root / relative)
        if source_id:
            candidates.append(musan_root / source_id)
    if list_entry:
        entry = Path(list_entry).expanduser()
        if not entry.is_absolute():
            entry = split_dir / entry
        candidates.append(entry)
    if relative:
        candidates.append(split_dir / relative)
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(
        f"Cannot resolve original audio for {source_id!r} under {split_dir}"
    )


def _spec_from_row(row: Any):
    from dma_kws.stage2.background_sampling import BackgroundCropSpec

    return BackgroundCropSpec(
        source_id=str(row["source_id"]),
        duration_seconds=float(row["duration_seconds"]),
        read_start_frame=int(row["read_start_frame"]),
        read_num_frames=int(row["read_num_frames"]),
        target_num_samples=int(row["target_num_samples"]),
        final_offset=int(row["final_offset"]),
    )


def _online_crop_timed(source, spec, extractor) -> tuple[torch.Tensor, dict[str, float]]:
    import soundfile as sf

    from dma_kws.stage2.background_sampling import _apply_final_crop, _as_mono

    started = time.perf_counter()
    with sf.SoundFile(str(source.path)) as handle:
        handle.seek(spec.read_start_frame)
        array = handle.read(
            frames=spec.read_num_frames,
            dtype="float32",
            always_2d=True,
        )
    read_seconds = time.perf_counter() - started
    started = time.perf_counter()
    waveform = torch.from_numpy(np.asarray(array, dtype=np.float32).T.copy())
    waveform = _as_mono(waveform, source=source.path)
    waveform = _apply_final_crop(waveform, spec)
    waveform = waveform.to(dtype=torch.float32).contiguous()
    crop_seconds = time.perf_counter() - started
    sample_rate = int(source.sample_rate)
    started = time.perf_counter()
    prepared, output_rate = extractor.prepare_waveform(waveform, sample_rate)
    resample_seconds = time.perf_counter() - started
    started = time.perf_counter()
    if extractor.backend == "lhotse_fbank":
        feats = extractor._extract_lhotse(prepared, output_rate)
    else:
        feats = extractor._extract_torchaudio(prepared, output_rate)
    fbank_seconds = time.perf_counter() - started
    return feats, {
        "read_seconds": read_seconds,
        "crop_seconds": crop_seconds,
        "resample_seconds": resample_seconds,
        "fbank_seconds": fbank_seconds,
    }


def _open_background_cache(config: dict[str, Any], manifest: Path):
    from dma_kws.stage2.background_cache import BackgroundFeatureCache

    stage2 = config["stage2"]
    background = stage2.get("background_negative") or {}
    return BackgroundFeatureCache(
        manifest,
        expected_fbank_kwargs=fbank_kwargs(get_fbank_config(config)),
        duration_seconds_min=float(background.get("duration_seconds_min", 1.0)),
        duration_seconds_max=float(background.get("duration_seconds_max", 3.0)),
        max_open_shards=background.get("max_open_shards", 8),
    )


def _run_background(
    *,
    config: dict[str, Any],
    output: Path,
    warmup_batches: int,
    measure_batches: int,
    device: str,
    manifest: Path,
    split_dir: Path,
    profile: bool,
) -> dict[str, Any]:
    from dma_kws.stage2.background_sampling import probe_source_info
    from dma_kws.stage2.fbank import FbankExtractor

    batch_size, _workers, _prefetch, accumulate, precision = _loader_settings(
        config, device=device
    )
    cache = _open_background_cache(config, manifest)
    crops_path = cache.manifest_path.parent / "crops.npy"
    recordings_path = cache.manifest_path.parent / "recordings.jsonl"
    crops = np.load(crops_path, allow_pickle=False)
    recordings = _read_jsonl(recordings_path)
    by_source = {str(row["source_id"]): row for row in recordings}
    extractor = FbankExtractor(**fbank_kwargs(get_fbank_config(config)))
    sources: dict[str, Any] = {}

    def source_for(source_id: str):
        cached = sources.get(source_id)
        if cached is not None:
            return cached
        record = by_source.get(source_id)
        if record is None:
            raise KeyError(f"crop source_id {source_id!r} is missing from recordings.jsonl")
        info = probe_source_info(
            _resolve_source_path(record, split_dir),
            source_id=source_id,
        )
        sources[source_id] = info
        return info

    total_batches = warmup_batches + measure_batches
    if total_batches < 1:
        raise InputBenchmarkError("warmup-batches + measure-batches must be at least 1")
    crop_ids = [index % int(cache.num_crops) for index in range(total_batches * batch_size)]
    unique_ids = list(dict.fromkeys(crop_ids))
    for crop_id in unique_ids:
        row = crops[crop_id]
        spec = _spec_from_row(row)
        source = source_for(str(row["source_id"]))
        online, _parts = _online_crop_timed(source, spec, extractor)
        stored = cache.read_crop(int(crop_id))
        if tuple(stored.shape) != tuple(online.shape):
            raise RuntimeError(
                f"crop_id {crop_id} frame/feature shape mismatch: "
                f"cache {tuple(stored.shape)} vs online {tuple(online.shape)}"
            )
        torch.testing.assert_close(stored, online.cpu(), rtol=1e-5, atol=1e-5)
    # Drop verification mmaps so timed reads observe cold open-shard misses.
    cache.close()

    warmup_ids = crop_ids[: warmup_batches * batch_size]
    measure_ids = crop_ids[warmup_batches * batch_size :]
    first_next = 0.0
    warmup_seconds = 0.0
    if warmup_batches:
        started = time.perf_counter()
        for index, crop_id in enumerate(warmup_ids):
            row = crops[crop_id]
            spec = _spec_from_row(row)
            source = source_for(str(row["source_id"]))
            batch_started = time.perf_counter()
            _online_crop_timed(source, spec, extractor)
            cache.read_crop(int(crop_id))
            if index == 0:
                first_next = time.perf_counter() - batch_started
        warmup_seconds = time.perf_counter() - started

    component = {
        "read_seconds": 0.0,
        "crop_seconds": 0.0,
        "resample_seconds": 0.0,
        "fbank_seconds": 0.0,
        "cache_read_seconds": 0.0,
        "online_total_seconds": 0.0,
    }
    waits: list[float] = []
    lengths: list[int] = []
    padding_ratios: list[float] = []
    cache_hits = 0
    cache_misses = 0
    high_water = 0

    def _shard_index(crop_id: int) -> int:
        return int(crops[crop_id]["shard_index"])

    with _profile_cm(profile, output, device) as profile_info:
        measure_started = time.perf_counter()
        for batch_index in range(measure_batches):
            batch_ids = measure_ids[
                batch_index * batch_size : (batch_index + 1) * batch_size
            ]
            batch_started = time.perf_counter()
            batch_lengths: list[int] = []
            for crop_id in batch_ids:
                open_before = _shard_index(crop_id) in cache._open_shards
                cached_started = time.perf_counter()
                stored = cache.read_crop(int(crop_id))
                component["cache_read_seconds"] += time.perf_counter() - cached_started
                if open_before:
                    cache_hits += 1
                else:
                    cache_misses += 1
                high_water = max(high_water, len(cache._open_shards))
                batch_lengths.append(int(stored.shape[0]))
            waits.append(time.perf_counter() - batch_started)
            lengths.extend(batch_lengths)
            padding_ratios.append(padding_ratio(batch_lengths))
            if warmup_batches == 0 and batch_index == 0:
                first_next = waits[0]
        cache_wall = time.perf_counter() - measure_started
        online_started = time.perf_counter()
        for crop_id in measure_ids:
            row = crops[crop_id]
            spec = _spec_from_row(row)
            source = source_for(str(row["source_id"]))
            _feats, parts = _online_crop_timed(source, spec, extractor)
            for key, value in parts.items():
                component[key] += float(value)
        component["online_total_seconds"] = time.perf_counter() - online_started
        measured_wall = cache_wall

    unique_measure = len(set(measure_ids))
    payload = _base_payload(
        mode="background",
        config=config,
        device=device,
        warmup_batches=warmup_batches,
        measure_batches=measure_batches,
        batch_size=batch_size,
        workers=0,
        prefetch=0,
        precision=precision,
        accumulate=accumulate,
        cache_id=str(cache.cache_id),
    )
    payload = _finish_payload(
        payload,
        warmup_actual=warmup_batches,
        measure_actual=measure_batches,
        optimizer_requested=0,
        optimizer_actual=0,
        first_next_seconds=first_next,
        warmup_seconds=warmup_seconds,
        measured_wall_seconds=measured_wall,
        waits=waits,
        lengths=lengths,
        padding_ratios=padding_ratios,
        positive_count=0,
        negative_count=0,
        background_count=len(measure_ids),
        samples=len(measure_ids),
        config=config,
        cache_hits=cache_hits,
        cache_misses=cache_misses,
        open_shard_high_water=high_water,
    )
    payload["component_times"] = component
    payload["crop_repeat_rate"] = (
        1.0 - (unique_measure / len(measure_ids)) if measure_ids else 0.0
    )
    payload["unique_crop_count"] = unique_measure
    payload["selected_crop_count"] = len(measure_ids)
    payload["selected_crop_ids_sample"] = [int(item) for item in measure_ids[:8]]
    payload["selected_source_ids_sample"] = [
        str(crops[crop_id]["source_id"]) for crop_id in measure_ids[:8]
    ]
    if profile_info:
        payload["profile"] = dict(profile_info)
    cache.close()
    return payload


def _run_loader(
    *,
    config: dict[str, Any],
    output: Path,
    warmup_batches: int,
    measure_batches: int,
    device: str,
    profile: bool,
    dataset: Any | None,
    tokenizer: Any | None,
) -> dict[str, Any]:
    batch_size, workers, prefetch, accumulate, precision = _loader_settings(
        config, device=device
    )
    if dataset is None:
        tokenizer = tokenizer or _load_tokenizer(config)
        dataset = build_stage2_train_dataset(config, tokenizer)
    loader = build_stage2_train_dataloader(config, dataset)
    iterator = _iter_forever(loader)
    first_next = 0.0
    warmup_seconds = 0.0
    if warmup_batches:
        started = time.perf_counter()
        for index in range(warmup_batches):
            _batch, wait = _next_timed(iterator)
            if index == 0:
                first_next = wait
        warmup_seconds = time.perf_counter() - started

    waits: list[float] = []
    lengths: list[int] = []
    padding_ratios: list[float] = []
    positive = negative = background = 0
    with _profile_cm(profile, output, device) as profile_info:
        _synchronize(device)
        measure_started = time.perf_counter()
        for index in range(measure_batches):
            batch, wait = _next_timed(iterator)
            waits.append(wait)
            pos, neg, bg, batch_lengths = _composition_from_batch(batch)
            positive += pos
            negative += neg
            background += bg
            lengths.extend(batch_lengths)
            padding_ratios.append(padding_ratio(batch_lengths))
            if warmup_batches == 0 and index == 0:
                first_next = wait
        _synchronize(device)
        measured_wall = time.perf_counter() - measure_started

    background_cfg = (config.get("stage2") or {}).get("background_negative") or {}
    cache_mode = str(background_cfg.get("mode") or "online") == "fbank_cache" and bool(
        background_cfg.get("enabled", False)
    )
    cache_hits: Any = "unavailable" if cache_mode else 0
    cache_misses: Any = "unavailable" if cache_mode else 0
    high_water: Any = "unavailable" if cache_mode else 0
    payload = _base_payload(
        mode="loader",
        config=config,
        device=device,
        warmup_batches=warmup_batches,
        measure_batches=measure_batches,
        batch_size=batch_size,
        workers=workers,
        prefetch=prefetch,
        precision=precision,
        accumulate=accumulate,
        cache_id=_cache_id_from_config(config),
    )
    payload = _finish_payload(
        payload,
        warmup_actual=warmup_batches,
        measure_actual=measure_batches,
        optimizer_requested=0,
        optimizer_actual=0,
        first_next_seconds=first_next,
        warmup_seconds=warmup_seconds,
        measured_wall_seconds=measured_wall,
        waits=waits,
        lengths=lengths,
        padding_ratios=padding_ratios,
        positive_count=positive,
        negative_count=negative,
        background_count=background,
        samples=measure_batches * batch_size,
        config=config,
        cache_hits=cache_hits,
        cache_misses=cache_misses,
        open_shard_high_water=high_water,
    )
    if profile_info:
        payload["profile"] = dict(profile_info)
    return payload


class _TimedCyclingLoader:
    """Iterable wrapper that records real ``next(iterator)`` wait times."""

    def __init__(self, loader: Any) -> None:
        self.loader = loader
        self.waits: list[float] = []
        self.batch_size = getattr(loader, "batch_size", None)

    @property
    def dataset(self) -> Any:
        return self.loader.dataset

    def __len__(self) -> int:
        return len(self.loader)

    def __iter__(self) -> Iterator[Any]:
        while True:
            iterator = iter(self.loader)
            yielded = 0
            while True:
                started = time.perf_counter()
                try:
                    batch = next(iterator)
                except StopIteration:
                    break
                self.waits.append(time.perf_counter() - started)
                yielded += 1
                yield batch
            if yielded == 0:
                raise RuntimeError(
                    "Stage II train DataLoader produced no batches "
                    "(drop_last=True requires len(dataset) >= batch_size)"
                )


def _run_train(
    *,
    config: dict[str, Any],
    output: Path,
    warmup_batches: int,
    measure_batches: int,
    device: str,
    profile: bool,
    dataset: Any | None,
    tokenizer: Any | None,
) -> dict[str, Any]:
    if device == "cuda" and not torch.cuda.is_available():
        raise TrainModeUnavailable(
            "CUDA was requested (--device cuda) but torch.cuda.is_available() is False"
        )
    stage2 = config.get("stage2") or {}
    init_checkpoint = str(stage2.get("init_checkpoint") or "").strip()
    if not init_checkpoint:
        raise TrainModeUnavailable(
            "train mode requires a real encoder init_checkpoint; "
            "refusing to report toy-encoder throughput. Set stage2.init_checkpoint "
            "or pass --override stage2.init_checkpoint=PATH."
        )
    if not Path(init_checkpoint).expanduser().is_file():
        raise TrainModeUnavailable(
            "train mode requires a real encoder init_checkpoint; "
            f"file not found: {init_checkpoint}. Refusing to report toy-encoder throughput."
        )

    try:
        import pytorch_lightning as pl
    except ImportError as exc:
        raise TrainModeUnavailable(
            "train mode requires pytorch-lightning"
        ) from exc

    from dma_kws.stage2.module import Stage2LightningModule
    from dma_kws.training.ddp import build_trainer_kwargs
    from dma_kws.training.device import resolve_accelerator_and_devices

    batch_size, workers, prefetch, accumulate, _precision = _loader_settings(
        config, device=device
    )
    warmup_actual = align_up(warmup_batches, accumulate)
    measure_actual = align_up(measure_batches, accumulate)
    if measure_actual < accumulate:
        measure_actual = accumulate
    optimizer_steps = (warmup_actual + measure_actual) // accumulate
    tokenizer = tokenizer or _load_tokenizer(config)
    if dataset is None:
        dataset = build_stage2_train_dataset(config, tokenizer)
    loader = build_stage2_train_dataloader(config, dataset)
    timed = _TimedCyclingLoader(loader)

    class _TrainWindowCallback(pl.Callback):
        def __init__(self) -> None:
            super().__init__()
            self.micro = 0
            self.warmup_seconds = 0.0
            self.measured_wall_seconds = 0.0
            self._warmup_t0: float | None = None
            self._measure_t0: float | None = None
            self.positive_count = 0
            self.negative_count = 0
            self.background_count = 0
            self.lengths: list[int] = []
            self.padding_ratios: list[float] = []

        def on_train_batch_start(self, trainer, pl_module, batch, batch_idx) -> None:
            if self.micro == 0 and warmup_actual > 0:
                self._warmup_t0 = time.perf_counter()
            if self.micro == warmup_actual:
                if self._warmup_t0 is not None:
                    self.warmup_seconds = time.perf_counter() - self._warmup_t0
                _synchronize(device)
                self._measure_t0 = time.perf_counter()

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
            in_measure = warmup_actual <= self.micro < warmup_actual + measure_actual
            if in_measure:
                pos, neg, bg, batch_lengths = _composition_from_batch(batch)
                self.positive_count += pos
                self.negative_count += neg
                self.background_count += bg
                self.lengths.extend(batch_lengths)
                self.padding_ratios.append(padding_ratio(batch_lengths))
            self.micro += 1
            if self.micro == warmup_actual + measure_actual:
                _synchronize(device)
                if self._measure_t0 is not None:
                    self.measured_wall_seconds = time.perf_counter() - self._measure_t0

    window = _TrainWindowCallback()
    vocab_size = len(tokenizer._symbol_table)
    model = Stage2LightningModule(
        config,
        vocab_size=vocab_size,
        freeze_encoder=bool(stage2.get("freeze_encoder", False)),
        init_checkpoint=init_checkpoint,
    )
    accelerator, devices = resolve_accelerator_and_devices(device, 1)
    trainer_kwargs = build_trainer_kwargs(
        config,
        devices,
        limit_steps=optimizer_steps,
        accelerator=accelerator,
    )
    trainer_kwargs["logger"] = False
    trainer_kwargs["enable_checkpointing"] = False
    trainer_kwargs["enable_model_summary"] = False
    trainer_kwargs["num_sanity_val_steps"] = 0
    trainer_kwargs["limit_val_batches"] = 0
    trainer_kwargs.pop("val_check_interval", None)
    trainer_kwargs["check_val_every_n_epoch"] = None
    precision = str(trainer_kwargs.get("precision") or resolve_precision(stage2, accelerator))

    with _profile_cm(profile, output, device) as profile_info:
        trainer = pl.Trainer(
            accelerator=accelerator,
            callbacks=[window],
            **trainer_kwargs,
        )
        trainer.fit(model, train_dataloaders=timed)

    measure_waits = timed.waits[warmup_actual : warmup_actual + measure_actual]
    first_next = float(timed.waits[0]) if timed.waits else 0.0
    background_cfg = stage2.get("background_negative") or {}
    cache_mode = str(background_cfg.get("mode") or "online") == "fbank_cache" and bool(
        background_cfg.get("enabled", False)
    )
    payload = _base_payload(
        mode="train",
        config=config,
        device=device,
        warmup_batches=warmup_batches,
        measure_batches=measure_batches,
        batch_size=batch_size,
        workers=workers,
        prefetch=prefetch,
        precision=precision,
        accumulate=accumulate,
        cache_id=_cache_id_from_config(config),
    )
    payload = _finish_payload(
        payload,
        warmup_actual=warmup_actual,
        measure_actual=measure_actual,
        optimizer_requested=optimizer_steps,
        optimizer_actual=int(getattr(trainer, "global_step", 0)),
        first_next_seconds=first_next,
        warmup_seconds=window.warmup_seconds,
        measured_wall_seconds=window.measured_wall_seconds,
        waits=measure_waits,
        lengths=window.lengths,
        padding_ratios=window.padding_ratios,
        positive_count=window.positive_count,
        negative_count=window.negative_count,
        background_count=window.background_count,
        samples=measure_actual * batch_size,
        config=config,
        cache_hits="unavailable" if cache_mode else 0,
        cache_misses="unavailable" if cache_mode else 0,
        open_shard_high_water="unavailable" if cache_mode else 0,
    )
    payload["microbatches_requested"] = {
        "warmup": warmup_batches,
        "measure": measure_batches,
    }
    payload["microbatches_actual"] = {
        "warmup": warmup_actual,
        "measure": measure_actual,
    }
    if profile_info:
        payload["profile"] = dict(profile_info)
    return payload


def run_input_benchmark(
    *,
    mode: str,
    output: str | Path,
    config: dict[str, Any] | None = None,
    warmup_batches: int = 20,
    measure_batches: int = 200,
    device: str = "cpu",
    manifest: str | Path | None = None,
    split_dir: str | Path | None = None,
    profile: bool = False,
    experiment: str | None = None,
    overrides: Sequence[str] = (),
    dataset: Any | None = None,
    tokenizer: Any | None = None,
) -> dict[str, Any]:
    """Run one benchmark mode and write the timing JSON to ``output``."""
    mode = str(mode).strip()
    if mode not in {"background", "loader", "train"}:
        raise InputBenchmarkError(
            f"--mode must be background, loader, or train, got {mode!r}"
        )
    device = str(device).strip()
    if device not in {"cpu", "cuda"}:
        raise InputBenchmarkError(f"--device must be cpu or cuda, got {device!r}")
    warmup_batches = _require_int(warmup_batches, field="warmup-batches", minimum=0)
    measure_batches = _require_int(measure_batches, field="measure-batches", minimum=1)
    if device == "cuda" and not torch.cuda.is_available():
        message = "CUDA was requested (--device cuda) but is not available"
        if mode == "train":
            raise TrainModeUnavailable(message)
        raise InputBenchmarkError(message)
    if config is None:
        config = config_to_dict(
            compose_config(experiment or DEFAULT_EXPERIMENT, list(overrides))
        )
    _seed_all(config)
    output_path = Path(output)
    if mode == "background":
        if manifest is None or split_dir is None:
            raise InputBenchmarkError(
                "background mode requires --manifest and --split-dir"
            )
        payload = _run_background(
            config=config,
            output=output_path,
            warmup_batches=warmup_batches,
            measure_batches=measure_batches,
            device=device,
            manifest=Path(manifest),
            split_dir=Path(split_dir),
            profile=profile,
        )
    elif mode == "loader":
        payload = _run_loader(
            config=config,
            output=output_path,
            warmup_batches=warmup_batches,
            measure_batches=measure_batches,
            device=device,
            profile=profile,
            dataset=dataset,
            tokenizer=tokenizer,
        )
    else:
        payload = _run_train(
            config=config,
            output=output_path,
            warmup_batches=warmup_batches,
            measure_batches=measure_batches,
            device=device,
            profile=profile,
            dataset=dataset,
            tokenizer=tokenizer,
        )
    _write_json(output_path, payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark Stage II background crops, DataLoader wait, or short train."
    )
    parser.add_argument(
        "--mode",
        choices=("background", "loader", "train"),
        default="loader",
        help="background | loader | train (default: loader)",
    )
    parser.add_argument(
        "--warmup-batches",
        type=_parse_int_at_least(0),
        default=20,
        help="Warmup microbatches (default: 20)",
    )
    parser.add_argument(
        "--measure-batches",
        type=_parse_int_at_least(1),
        default=200,
        help="Measured microbatches (default: 200)",
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cpu",
        help="cpu | cuda (default: cpu; train/cuda requires real CUDA)",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Background cache manifest.json (required for --mode background)",
    )
    parser.add_argument(
        "--split-dir",
        type=Path,
        default=None,
        help="MUSAN split directory with original source audio (background mode)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Timing JSON output path (independent of training checkpoints)",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Write a Chrome trace next to --output; not mixed into timing fields",
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
    return parser


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = build_parser().parse_args(argv)
    try:
        payload = run_input_benchmark(
            mode=args.mode,
            output=args.output,
            warmup_batches=args.warmup_batches,
            measure_batches=args.measure_batches,
            device=args.device,
            manifest=args.manifest,
            split_dir=args.split_dir,
            profile=bool(args.profile),
            experiment=args.experiment,
            overrides=tuple(args.override or ()),
        )
    except TrainModeUnavailable as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2) from exc
    except InputBenchmarkError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2) from exc
    summary = {
        "mode": payload["mode"],
        "status": payload["status"],
        "output": str(args.output),
        "samples_per_second": payload["samples_per_second"],
        "measured_wall_seconds": payload["measured_wall_seconds"],
        "padding_ratio": payload["padding_ratio"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return payload


if __name__ == "__main__":
    main()
