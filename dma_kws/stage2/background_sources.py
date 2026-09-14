"""Training-side background sampling: legacy adapter, online crops, multi-source."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
import random
from typing import Any, Protocol

import torch

from dma_kws.configs.schema import (
    active_background_sources,
    background_source_batch_index,
    normalize_source_weights,
    validate_background_negative_config,
)
from dma_kws.data_prep.background_manifest import (
    CATALOG_JSON_NAME,
    assert_catalog_source_id,
    read_catalog,
    read_recordings_jsonl,
    require_eligible_train_records,
)
from dma_kws.stage2.background_sampling import (
    draw_crop_spec,
    materialize_crop,
    probe_source_info,
)
from dma_kws.stage2.fbank import FbankExtractor


@dataclass(frozen=True)
class BackgroundSample:
    feat: torch.Tensor
    background_source_id: int
    recording_id: str
    crop_id: str | None


class BackgroundSampler(Protocol):
    def sample(self, *, rng: random.Random) -> BackgroundSample: ...

    def run_record_fields(self) -> dict[str, Any]: ...

    def close(self) -> None: ...


def select_weighted_source_index(
    weights: Sequence[float], rng: random.Random
) -> int:
    """Choose among sorted active sources. A single source does not draw RNG."""
    if len(weights) == 1:
        return 0
    draw = rng.random()
    cumulative = 0.0
    last = len(weights) - 1
    for index, weight in enumerate(weights):
        cumulative += float(weight)
        if index == last or draw < cumulative:
            return index
    return last


class LegacyBackgroundSamplerAdapter:
    """Expose ``sample()`` around an existing ``extract(rng) -> Tensor`` sampler."""

    def __init__(self, inner: Any, *, background_source_id: int = 0) -> None:
        self.inner = inner
        self._background_source_id = int(background_source_id)

    def extract(self, *, rng: random.Random) -> torch.Tensor:
        return self.inner.extract(rng=rng)

    def sample(self, *, rng: random.Random) -> BackgroundSample:
        feat = self.extract(rng=rng)
        return BackgroundSample(
            feat=feat,
            background_source_id=self._background_source_id,
            recording_id="",
            crop_id=None,
        )

    def run_record_fields(self) -> dict[str, Any]:
        record = getattr(self.inner, "run_record_fields", None)
        if callable(record):
            return dict(record())
        return {}

    def close(self) -> None:
        close = getattr(self.inner, "close", None)
        if callable(close):
            close()


class OnlineBackgroundSource:
    """Uniform-over-recordings online crop for one catalog source."""

    def __init__(
        self,
        *,
        source_id: str,
        manifest_path: str | Path,
        records: Sequence[Any],
        duration_seconds_min: float,
        duration_seconds_max: float,
        fbank_kwargs: Mapping[str, Any] | None = None,
        background_source_id: int,
    ) -> None:
        if not records:
            raise ValueError(f"no eligible train recordings for source {source_id}")
        self.source_id = source_id
        self.manifest_path = Path(manifest_path)
        self._recordings = tuple(records)
        self.duration_seconds_min = float(duration_seconds_min)
        self.duration_seconds_max = float(duration_seconds_max)
        self._fbank_kwargs = dict(fbank_kwargs or {})
        self._background_source_id = int(background_source_id)
        self._fbank_extractor: FbankExtractor | None = None
        self._source_info_cache: dict[str, Any] = {}

    def _fbank(self) -> FbankExtractor:
        if self._fbank_extractor is None:
            self._fbank_extractor = FbankExtractor(**self._fbank_kwargs)
        return self._fbank_extractor

    def _source_info(self, path: Path, *, source_id: str):
        key = str(path)
        cached = self._source_info_cache.get(key)
        if cached is not None:
            return cached
        info = probe_source_info(path, source_id=source_id)
        self._source_info_cache[key] = info
        return info

    def sample(self, *, rng: random.Random) -> BackgroundSample:
        duration_seconds = rng.uniform(
            self.duration_seconds_min,
            self.duration_seconds_max,
        )
        record = rng.choice(self._recordings)
        audio_path = Path(record.audio_path)
        if not audio_path.is_absolute():
            audio_path = self.manifest_path.parent / audio_path
        source = self._source_info(audio_path, source_id=record.recording_id)
        spec = draw_crop_spec(source, duration_seconds, rng=rng)
        waveform, sample_rate = materialize_crop(source, spec)
        feat = self._fbank().extract(waveform, sample_rate)
        return BackgroundSample(
            feat=feat,
            background_source_id=self._background_source_id,
            recording_id=record.recording_id,
            crop_id=None,
        )

    def extract(self, *, rng: random.Random) -> torch.Tensor:
        return self.sample(rng=rng).feat

    def run_record_fields(self) -> dict[str, Any]:
        return {
            "background_source_id": self.source_id,
            "background_manifest": str(self.manifest_path),
            "background_recording_count": len(self._recordings),
        }

    def close(self) -> None:
        self._fbank_extractor = None
        self._source_info_cache.clear()


class CachedBackgroundSource:
    """Cache v2 strategy; Task 4 fills the reader."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise ValueError(
            "stage2.background_negative.mode=fbank_cache with non-empty sources "
            "is not yet wired (CachedBackgroundSource)"
        )

    def sample(self, *, rng: random.Random) -> BackgroundSample:
        raise ValueError(
            "stage2.background_negative.mode=fbank_cache with non-empty sources "
            "is not yet wired (CachedBackgroundSource)"
        )

    def run_record_fields(self) -> dict[str, Any]:
        raise ValueError(
            "stage2.background_negative.mode=fbank_cache with non-empty sources "
            "is not yet wired (CachedBackgroundSource)"
        )

    def close(self) -> None:
        return None


class MultiSourceBackgroundSampler:
    """Weighted source draw, then one inner ``sample()``. No probability gate."""

    def __init__(self, sources: Sequence[tuple[str, float, Any]]) -> None:
        by_id: dict[str, tuple[float, Any]] = {}
        for source_id, weight, sampler in sources:
            if source_id in by_id:
                raise ValueError(f"duplicate source id: {source_id!r}")
            weight_value = float(weight)
            if weight_value <= 0.0:
                continue
            by_id[source_id] = (weight_value, sampler)
        if not by_id:
            raise ValueError(
                "stage2.background_negative.sources must include at least one "
                "source with weight > 0"
            )
        ordered_ids = tuple(sorted(by_id))
        self._source_ids = ordered_ids
        self._index = background_source_batch_index(ordered_ids)
        self._weights = tuple(
            normalize_source_weights([by_id[source_id][0] for source_id in ordered_ids])
        )
        self._samplers = tuple(by_id[source_id][1] for source_id in ordered_ids)

    def sample(self, *, rng: random.Random) -> BackgroundSample:
        index = select_weighted_source_index(self._weights, rng)
        drawn = self._samplers[index].sample(rng=rng)
        return BackgroundSample(
            feat=drawn.feat,
            background_source_id=self._index[self._source_ids[index]],
            recording_id=drawn.recording_id,
            crop_id=drawn.crop_id,
        )

    def extract(self, *, rng: random.Random) -> torch.Tensor:
        return self.sample(rng=rng).feat

    def run_record_fields(self) -> dict[str, Any]:
        fields: dict[str, Any] = {
            "background_mode": "online",
            "background_source_ids": list(self._source_ids),
            "background_source_weights": list(self._weights),
            "background_source_index": dict(self._index),
        }
        for sampler in self._samplers:
            record = getattr(sampler, "run_record_fields", None)
            if callable(record):
                inner = record()
                source_id = inner.get("background_source_id")
                if source_id:
                    fields[f"background_source:{source_id}"] = inner
        return fields

    def close(self) -> None:
        for sampler in self._samplers:
            close = getattr(sampler, "close", None)
            if callable(close):
                close()


def _load_online_records(source: Any) -> tuple[Any, ...]:
    source_id = source.id
    manifest = Path(str(source.manifest or "")).expanduser()
    if not manifest.is_file():
        raise FileNotFoundError(
            f"source id={source_id!r} field=manifest; expected an existing "
            f"recordings.jsonl, got {str(manifest)!r}"
        )
    records = read_recordings_jsonl(manifest)
    mismatched = next(
        (record for record in records if record.dataset_id != source_id),
        None,
    )
    if mismatched is not None:
        raise ValueError(
            f"source id={source_id!r} recording_id={mismatched.recording_id!r} "
            f"field=dataset_id; expected {source_id!r}, got "
            f"{mismatched.dataset_id!r}"
        )
    catalog_path = manifest.parent / CATALOG_JSON_NAME
    if catalog_path.is_file():
        assert_catalog_source_id(read_catalog(catalog_path), source_id)
    eligible = require_eligible_train_records(records, source_id=source_id)
    ordered = tuple(sorted(eligible, key=lambda item: item.recording_id))
    for record in ordered:
        audio_path = Path(record.audio_path)
        if not audio_path.is_absolute():
            audio_path = manifest.parent / audio_path
        if not audio_path.is_file():
            raise FileNotFoundError(
                f"source id={source_id!r} recording_id={record.recording_id!r} "
                f"field=audio_path; expected an accessible audio file, got "
                f"{str(audio_path)!r}"
            )
    return ordered


def _build_legacy_sampler(
    payload: Mapping[str, Any],
    *,
    fbank_kwargs: Mapping[str, Any] | None,
) -> BackgroundSampler:
    mode = str(payload.get("mode") or "online").strip() or "online"
    duration_min = float(payload.get("duration_seconds_min", 1.0))
    duration_max = float(payload.get("duration_seconds_max", 3.0))
    if mode == "fbank_cache":
        cache_manifest = str(payload.get("cache_manifest", "") or "").strip()
        if not cache_manifest:
            raise ValueError(
                "stage2.background_negative.cache_manifest is required "
                "when mode=fbank_cache"
            )
        fbank = dict(fbank_kwargs or {})
        if "dither" in fbank and float(fbank["dither"]) != 0.0:
            raise ValueError(
                "stage2.background_negative.mode=fbank_cache "
                f"requires fbank dither=0, got {fbank['dither']!r}"
            )
        from dma_kws.stage2.background_cache import BackgroundFeatureCache

        inner = BackgroundFeatureCache(
            cache_manifest,
            expected_fbank_kwargs=fbank,
            duration_seconds_min=duration_min,
            duration_seconds_max=duration_max,
            audio_list_path=str(payload.get("audio_list_path", "") or "").strip(),
            max_open_shards=payload.get("max_open_shards", 8),
        )
        return LegacyBackgroundSamplerAdapter(inner, background_source_id=0)

    audio_list_path = str(payload.get("audio_list_path", "") or "").strip()
    if not audio_list_path:
        raise ValueError(
            "stage2.background_negative.audio_list_path is required when enabled"
        )
    from dma_kws.stage2.features import TrainingBackgroundSampler

    inner = TrainingBackgroundSampler(
        audio_list_path=audio_list_path,
        duration_seconds_min=duration_min,
        duration_seconds_max=duration_max,
        fbank_kwargs=fbank_kwargs,
    )
    return LegacyBackgroundSamplerAdapter(inner, background_source_id=0)


def build_background_sampler(
    config: Mapping[str, Any],
    *,
    fbank_kwargs: Mapping[str, Any] | None = None,
) -> BackgroundSampler | None:
    payload = dict(config or {})
    validate_background_negative_config(payload)
    if not bool(payload.get("enabled", False)):
        return None
    sources = payload.get("sources") or []
    mode = str(payload.get("mode") or "online").strip() or "online"
    if sources:
        if mode == "fbank_cache":
            raise ValueError(
                "stage2.background_negative.mode=fbank_cache with non-empty "
                "sources is not yet wired (CachedBackgroundSource)"
            )
        active = active_background_sources(sources)
        index = background_source_batch_index([source.id for source in active])
        duration_min = float(payload.get("duration_seconds_min", 1.0))
        duration_max = float(payload.get("duration_seconds_max", 3.0))
        built: list[tuple[str, float, BackgroundSampler]] = []
        for source in active:
            records = _load_online_records(source)
            inner = OnlineBackgroundSource(
                source_id=source.id,
                manifest_path=source.manifest,
                records=records,
                duration_seconds_min=duration_min,
                duration_seconds_max=duration_max,
                fbank_kwargs=fbank_kwargs,
                background_source_id=index[source.id],
            )
            built.append((source.id, float(source.weight), inner))
        return MultiSourceBackgroundSampler(built)
    return _build_legacy_sampler(payload, fbank_kwargs=fbank_kwargs)
