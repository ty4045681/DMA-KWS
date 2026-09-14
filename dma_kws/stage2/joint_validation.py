"""Fixed held-out background crops for validation (clip FPR, not FA/h)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
from pathlib import Path
import random
from typing import Any

import torch
from torch.utils.data import Dataset

from dma_kws.configs.schema import (
    active_background_sources,
    background_source_batch_index,
)
from dma_kws.data_prep.background_manifest import read_recordings_jsonl
from dma_kws.stage2.background_identity import musan_metric_alias_allowed
from dma_kws.stage2.background_sampling import draw_crop_spec, materialize_crop, probe_source_info
from dma_kws.stage2.fbank import FbankExtractor
from dma_kws.stage2.features import TrainingBackgroundSampler


def validation_crop_rng(
    seed: int, source_id: str, recording_id: str, ordinal: int
) -> random.Random:
    payload = f"{int(seed)}\0{source_id}\0{recording_id}\0{int(ordinal)}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def _eligible_val_records(records: Sequence[Any], *, source_id: str) -> tuple[Any, ...]:
    eligible = tuple(
        sorted(
            (
                record
                for record in records
                if record.split == "val" and record.background_eligible is True
            ),
            key=lambda item: item.recording_id,
        )
    )
    if not eligible:
        raise ValueError(
            f"source id={source_id!r} field=split; expected eligible val "
            "recordings, got 0"
        )
    return eligible


class BackgroundValidationDataset(Dataset):
    def __init__(
        self, *, audio_list_path, anchor_seq, num_samples=512, seed=2026,
        duration_seconds_min=1.0, duration_seconds_max=3.0, fbank_kwargs=None,
    ):
        if isinstance(num_samples, bool) or int(num_samples) != num_samples or num_samples < 1:
            raise ValueError("adapt.joint.background_val_samples must be a positive integer")
        self.num_samples = int(num_samples)
        self.seed = int(seed)
        self.anchor_seq = torch.as_tensor(anchor_seq, dtype=torch.long).clone()
        if not self.anchor_seq.numel():
            raise ValueError("Background validation requires a nonempty keyword")
        self.background_sampler = TrainingBackgroundSampler(
            audio_list_path=audio_list_path,
            duration_seconds_min=duration_seconds_min,
            duration_seconds_max=duration_seconds_max,
            fbank_kwargs=fbank_kwargs,
        )

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        seed = self.seed + int(index)
        # Audio crops and feature dither must both stay fixed across validations
        # and worker counts, without perturbing the caller's RNG stream.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            feat = self.background_sampler.extract(rng=random.Random(seed))
        return {
            "sample_id": torch.tensor(index, dtype=torch.long),
            "anchor_seq": self.anchor_seq.clone(),
            "feat": feat,
            "label": torch.tensor(0, dtype=torch.long),
        }


class MultiSourceBackgroundValidationDataset(Dataset):
    """Deterministic per-source val crops; isolated from the training RNG."""

    def __init__(
        self,
        *,
        source_id: str,
        manifest_path: str | Path,
        records: Sequence[Any],
        anchor_seq,
        samples_per_source: int,
        seed: int,
        duration_seconds_min: float,
        duration_seconds_max: float,
        fbank_kwargs: Mapping[str, Any] | None,
        background_source_id: int,
    ) -> None:
        if (
            isinstance(samples_per_source, bool)
            or int(samples_per_source) != samples_per_source
            or samples_per_source < 1
        ):
            raise ValueError(
                "stage2.background_negative.validation.samples_per_source must "
                f"be a positive integer, got {samples_per_source!r}"
            )
        self.source_id = str(source_id)
        self.manifest_path = Path(manifest_path)
        self.records = _eligible_val_records(records, source_id=self.source_id)
        self.num_samples = int(samples_per_source)
        self.seed = int(seed)
        self.duration_seconds_min = float(duration_seconds_min)
        self.duration_seconds_max = float(duration_seconds_max)
        self.anchor_seq = torch.as_tensor(anchor_seq, dtype=torch.long).clone()
        if not self.anchor_seq.numel():
            raise ValueError("Background validation requires a nonempty keyword")
        self._fbank_kwargs = dict(fbank_kwargs or {})
        self._fbank_extractor: FbankExtractor | None = None
        self.background_source_id = int(background_source_id)
        for record in self.records:
            audio_path = Path(record.audio_path)
            if not audio_path.is_absolute():
                audio_path = self.manifest_path.parent / audio_path
            if not audio_path.is_file():
                raise FileNotFoundError(
                    f"source id={self.source_id!r} recording_id="
                    f"{record.recording_id!r} field=audio_path; expected an "
                    f"accessible val waveform, got {str(audio_path)!r}"
                )

    def _fbank(self) -> FbankExtractor:
        if self._fbank_extractor is None:
            self._fbank_extractor = FbankExtractor(**self._fbank_kwargs)
        return self._fbank_extractor

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> dict[str, Any]:
        ordinal = int(index)
        record = self.records[ordinal % len(self.records)]
        rng = validation_crop_rng(
            self.seed, self.source_id, record.recording_id, ordinal
        )
        duration = rng.uniform(self.duration_seconds_min, self.duration_seconds_max)
        audio_path = Path(record.audio_path)
        if not audio_path.is_absolute():
            audio_path = self.manifest_path.parent / audio_path
        source = probe_source_info(audio_path, source_id=record.recording_id)
        spec = draw_crop_spec(source, duration, rng=rng)
        digest = hashlib.sha256(
            f"{self.seed}\0{self.source_id}\0{record.recording_id}\0{ordinal}".encode(
                "utf-8"
            )
        ).digest()
        torch_seed = int.from_bytes(digest[:8], "big") % (2**32)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(torch_seed)
            waveform, sample_rate = materialize_crop(source, spec)
            feat = self._fbank().extract(waveform, sample_rate)
        return {
            "sample_id": torch.tensor(ordinal, dtype=torch.long),
            "anchor_seq": self.anchor_seq.clone(),
            "feat": feat,
            "label": torch.tensor(0, dtype=torch.long),
            "background_source_id": torch.tensor(
                self.background_source_id, dtype=torch.long
            ),
            "recording_id": record.recording_id,
        }


def build_background_validation_datasets(
    background_cfg: Mapping[str, Any],
    *,
    anchor_seq,
    fbank_kwargs: Mapping[str, Any] | None = None,
) -> list[tuple[str, Dataset]]:
    """Return ``(role, dataset)`` pairs for enabled per-source val crops."""
    payload = dict(background_cfg or {})
    validation = dict(payload.get("validation") or {})
    if not bool(validation.get("enabled", False)):
        return []
    active = active_background_sources(payload.get("sources") or [])
    index = background_source_batch_index([source.id for source in active])
    samples = int(validation.get("samples_per_source", 256))
    seed = int(validation.get("seed", 2025))
    duration_min = float(payload.get("duration_seconds_min", 1.0))
    duration_max = float(payload.get("duration_seconds_max", 3.0))
    built: list[tuple[str, Dataset]] = []
    for source in active:
        records = read_recordings_jsonl(source.manifest)
        dataset = MultiSourceBackgroundValidationDataset(
            source_id=source.id,
            manifest_path=source.manifest,
            records=records,
            anchor_seq=anchor_seq,
            samples_per_source=samples,
            seed=seed,
            duration_seconds_min=duration_min,
            duration_seconds_max=duration_max,
            fbank_kwargs=fbank_kwargs,
            background_source_id=index[source.id],
        )
        built.append((f"background:{source.id}", dataset))
    return built


def background_clip_stats(
    scores: torch.Tensor,
    *,
    threshold: float,
    sample_ids: torch.Tensor | None = None,
) -> dict[str, float]:
    """Clip FPR is the threshold hit-rate on background scores, not FA/h."""
    scores = scores.detach().reshape(-1).to(dtype=torch.float32)
    if sample_ids is not None and sample_ids.numel():
        sample_ids = sample_ids.detach().reshape(-1)
        order = torch.argsort(sample_ids, stable=True)
        ordered_ids = sample_ids[order]
        first = torch.ones_like(ordered_ids, dtype=torch.bool)
        first[1:] = ordered_ids[1:] != ordered_ids[:-1]
        scores = scores[order[first]]
    count = int(scores.numel())
    nan = float("nan")
    if count == 0:
        return {
            "count": 0,
            "clip_fpr": nan,
            "score_mean": nan,
            "score_p95": nan,
            "score_p99": nan,
            "score_max": nan,
        }
    quantiles = torch.quantile(
        scores, torch.tensor([0.95, 0.99], device=scores.device, dtype=scores.dtype)
    )
    return {
        "count": count,
        "clip_fpr": float((scores >= float(threshold)).to(dtype=scores.dtype).mean()),
        "score_mean": float(scores.mean()),
        "score_p95": float(quantiles[0]),
        "score_p99": float(quantiles[1]),
        "score_max": float(scores.max()),
    }


def format_background_val_metrics(
    per_source: Mapping[str, Mapping[str, float]],
    *,
    overall: Mapping[str, float] | None = None,
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    total_count = 0
    weighted_fpr = 0.0
    weighted_mean = 0.0
    score_max = float("-inf")
    for source_id, stats in per_source.items():
        prefix = f"val/background/{source_id}"
        for key, value in stats.items():
            metrics[f"{prefix}/{key}"] = value
        count = int(stats.get("count", 0) or 0)
        total_count += count
        if count:
            weighted_fpr += float(stats.get("clip_fpr", 0.0)) * count
            weighted_mean += float(stats.get("score_mean", 0.0)) * count
            score_max = max(score_max, float(stats.get("score_max", float("-inf"))))
    nan = float("nan")
    if overall is None:
        overall = {
            "count": total_count,
            "clip_fpr": (weighted_fpr / total_count) if total_count else nan,
            "score_mean": (weighted_mean / total_count) if total_count else nan,
            "score_p95": nan,
            "score_p99": nan,
            "score_max": score_max if total_count else nan,
        }
    for key, value in overall.items():
        metrics[f"val/background/overall/{key}"] = value
    if musan_metric_alias_allowed(list(per_source)) and "musan" in per_source:
        metrics["val/musan_deploy_fpr"] = per_source["musan"]["clip_fpr"]
    return metrics
