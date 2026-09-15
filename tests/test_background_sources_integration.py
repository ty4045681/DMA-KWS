from __future__ import annotations

import json
from pathlib import Path
import random
import shutil

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from dma_kws.data_prep.background_manifest import (
    CATALOG_JSON_NAME,
    background_record_from_mapping,
    build_catalog,
    write_catalog,
    write_recordings_jsonl,
)
from dma_kws.stage2.collate import train_collate_fn
from dma_kws.stage2.dataset import LibriPhraseTrainDataset
from tests.test_prepare_stage2_background import (
    DEFAULT_EXPERIMENT,
    K,
    SEED,
    _extraction_overrides,
    _load_recordings,
    _make_generic_catalog,
    _make_synthetic_split,
    _prepare_kwargs,
    _prepare_v2_kwargs,
)
from tests.test_stage2_background_cache import (
    _composed_cache_args,
    _mock_speech_metadata,
    _patch_online_feature_path,
)
from tests.test_stage2_background_config import _FakeTokenizer, _mock_dataframe
from tests.test_stage2_background_sources import _online_config, _source_config


def _boom(*_args, **_kwargs):
    raise AssertionError("cache path must not decode audio or construct FbankExtractor")


def _cache_config(layout, manifest_path, *, dataset_id: str, **overrides):
    args = _composed_cache_args()
    payload = _online_config(
        [
            _source_config(
                dataset_id,
                layout["manifest"],
                cache_manifest=str(manifest_path),
            )
        ],
        mode="fbank_cache",
        duration_seconds_min=args["duration_seconds_min"],
        duration_seconds_max=args["duration_seconds_max"],
        max_open_shards=4,
    )
    payload.update(overrides)
    return payload, args


def _imported_manifest_for_v1(
    tmp_path: Path,
    cache_dir: Path,
    *,
    dataset_id: str = "musan",
    move_last_to_val: bool = False,
    drop_last: bool = False,
    provenance_complete: bool = False,
) -> Path:
    recordings = _load_recordings(cache_dir)
    records = []
    for index, row in enumerate(recordings):
        source_id = str(row["source_id"])
        split = "train"
        eligible = True
        if move_last_to_val and index == len(recordings) - 1:
            split = "val"
        record = background_record_from_mapping(
            {
                "schema_version": 1,
                "dataset_id": dataset_id,
                "recording_id": f"{dataset_id}:{source_id}",
                "relative_path": source_id,
                "audio_path": f"/relocated/{source_id}",
                "group_id": f"{dataset_id}:group:{index}",
                "origin_ids": [],
                "split": split,
                "categories": ["noise"],
                "background_eligible": eligible,
                "eligibility_basis": "imported_v1",
                "duration_seconds": 2.0,
                "sample_rate": int(row.get("sample_rate") or 16000),
                "channels": int(row.get("channels") or 1),
                "audio_sha256": str(row.get("content_sha256") or "a" * 64),
                "license_id": "",
                "provenance_complete": provenance_complete,
            }
        )
        records.append(record)
    if drop_last:
        records = records[:-1]
    catalog_dir = tmp_path / "imported" / dataset_id
    catalog_dir.mkdir(parents=True)
    manifest_path = catalog_dir / "recordings.jsonl"
    write_recordings_jsonl(manifest_path, records)
    catalog = build_catalog(
        dataset_id=dataset_id,
        root="/relocated",
        records=records,
        recordings_path=manifest_path,
        filter_policy_version="imported_v1",
        filter_policy_hash="c" * 64,
        split_seed=SEED,
        split_rules={"policy": "preserve"},
    )
    write_catalog(catalog_dir / CATALOG_JSON_NAME, catalog)
    return manifest_path


def test_v2_prepare_verify_reader_dataset_matches_online_crop(tmp_path, monkeypatch):
    from dma_kws.data_prep.stage2_background import (
        prepare_stage2_background,
        verify_stage2_background_cache,
    )
    from dma_kws.stage2.background_cache import BackgroundFeatureCache
    from dma_kws.stage2.background_sources import (
        CachedBackgroundSource,
        MultiSourceBackgroundSampler,
        build_background_sampler,
    )

    layout = _make_generic_catalog(tmp_path)
    output_dir = tmp_path / "cache"
    prepare_stage2_background(**_prepare_v2_kwargs(layout["manifest"], output_dir))
    verify_stage2_background_cache(output_dir)
    args = _composed_cache_args()
    cache = BackgroundFeatureCache(
        output_dir / "manifest.json",
        expected_fbank_kwargs=args["expected_fbank_kwargs"],
        duration_seconds_min=args["duration_seconds_min"],
        duration_seconds_max=args["duration_seconds_max"],
    )
    try:
        stored = cache.read_crop(0)
        assert stored.dtype == torch.float32
        assert stored.ndim == 2
    finally:
        cache.close()

    _mock_speech_metadata(monkeypatch)
    config, args = _cache_config(
        layout, output_dir / "manifest.json", dataset_id=layout["dataset_id"]
    )
    sampler = build_background_sampler(config, fbank_kwargs=args["expected_fbank_kwargs"])
    assert isinstance(sampler, MultiSourceBackgroundSampler)
    inner = sampler._samplers[0]
    assert isinstance(inner, CachedBackgroundSource)
    sample = sampler.sample(rng=random.Random(0))
    assert sample.recording_id in set(layout["train_ids"])
    assert sample.crop_id
    assert sample.feat.dtype == torch.float32
    sampler.close()

    dataset = LibriPhraseTrainDataset(
        wav_dir="/data/segments",
        tokenizer=_FakeTokenizer(),
        df=_mock_dataframe(),
        sample_lens=8,
        seed=0,
        background_negative={**config, "probability": 1.0},
        fbank_kwargs=args["expected_fbank_kwargs"],
    )
    item = dataset[0]
    assert item["label"].item() == 0
    assert item["query_seq"].numel() == 0
    assert int(item["background_source_id"]) == 0
    assert item["recording_id"] in set(layout["train_ids"])
    dataset._background_sampler.close()


def test_cache_only_training_batch_after_wavs_removed(tmp_path, monkeypatch):
    from dma_kws.data_prep.stage2_background import prepare_stage2_background

    layout = _make_generic_catalog(tmp_path)
    output_dir = tmp_path / "cache"
    prepare_stage2_background(**_prepare_v2_kwargs(layout["manifest"], output_dir))
    shutil.rmtree(layout["root"])
    _mock_speech_metadata(monkeypatch)
    _patch_online_feature_path(monkeypatch)
    monkeypatch.setattr("dma_kws.stage2.background_sampling.probe_source_info", _boom)
    monkeypatch.setattr("dma_kws.stage2.background_sampling.materialize_crop", _boom)
    monkeypatch.setattr("dma_kws.stage2.fbank.FbankExtractor", _boom)
    monkeypatch.setattr("soundfile.SoundFile", _boom)

    config, args = _cache_config(
        layout,
        output_dir / "manifest.json",
        dataset_id=layout["dataset_id"],
        probability=1.0,
        validation={"enabled": False, "samples_per_source": 256, "seed": 2025},
    )
    dataset = LibriPhraseTrainDataset(
        wav_dir="/data/segments",
        tokenizer=_FakeTokenizer(),
        df=_mock_dataframe(),
        sample_lens=4,
        seed=0,
        background_negative=config,
        fbank_kwargs=args["expected_fbank_kwargs"],
    )
    loader = DataLoader(dataset, batch_size=2, collate_fn=train_collate_fn)
    batch = next(iter(loader))
    assert batch["feat"].ndim == 3
    assert batch["label"].tolist() == [0, 0] or 0 in batch["label"].tolist()
    assert batch["background_source_id"].dtype == torch.long
    dataset._background_sampler.close()


def test_v1_adapter_maps_matching_imported_manifest_and_refuses_member_mismatch(
    tmp_path, monkeypatch
):
    from dma_kws.data_prep.stage2_background import prepare_stage2_background
    from dma_kws.stage2.background_sources import (
        CachedBackgroundSource,
        build_background_sampler,
    )

    layout = _make_synthetic_split(tmp_path)
    output_dir = tmp_path / "cache-v1"
    prepare_stage2_background(**_prepare_kwargs(layout["split_dir"], output_dir))
    shutil.rmtree(layout["musan_root"])
    matching = _imported_manifest_for_v1(tmp_path / "match", output_dir)
    args = _composed_cache_args()
    sampler = build_background_sampler(
        _online_config(
            [
                _source_config(
                    "musan",
                    matching,
                    cache_manifest=str(output_dir / "manifest.json"),
                )
            ],
            mode="fbank_cache",
            duration_seconds_min=args["duration_seconds_min"],
            duration_seconds_max=args["duration_seconds_max"],
        ),
        fbank_kwargs=args["expected_fbank_kwargs"],
    )
    inner = sampler._samplers[0]
    assert isinstance(inner, CachedBackgroundSource)
    assert inner._cache.format_version == 1
    sample = sampler.sample(rng=random.Random(1))
    assert sample.recording_id.startswith("musan:")
    assert sample.feat.dtype == torch.float32
    sampler.close()

    mismatched = _imported_manifest_for_v1(
        tmp_path / "mismatch", output_dir, move_last_to_val=True
    )
    with pytest.raises(ValueError, match=r"dataset_id|member|train"):
        build_background_sampler(
            _online_config(
                [
                    _source_config(
                        "musan",
                        mismatched,
                        cache_manifest=str(output_dir / "manifest.json"),
                    )
                ],
                mode="fbank_cache",
                duration_seconds_min=args["duration_seconds_min"],
                duration_seconds_max=args["duration_seconds_max"],
            ),
            fbank_kwargs=args["expected_fbank_kwargs"],
        )

    dropped = _imported_manifest_for_v1(
        tmp_path / "dropped", output_dir, drop_last=True
    )
    with pytest.raises(ValueError, match=r"dataset_id|member|train"):
        build_background_sampler(
            _online_config(
                [
                    _source_config(
                        "musan",
                        dropped,
                        cache_manifest=str(output_dir / "manifest.json"),
                    )
                ],
                mode="fbank_cache",
                duration_seconds_min=args["duration_seconds_min"],
                duration_seconds_max=args["duration_seconds_max"],
            ),
            fbank_kwargs=args["expected_fbank_kwargs"],
        )


def test_v1_adapter_does_not_mark_missing_provenance_complete(tmp_path):
    from dma_kws.data_prep.background_manifest import read_recordings_jsonl
    from dma_kws.data_prep.stage2_background import prepare_stage2_background
    from dma_kws.stage2.background_sources import build_background_sampler

    layout = _make_synthetic_split(tmp_path)
    output_dir = tmp_path / "cache-v1"
    prepare_stage2_background(**_prepare_kwargs(layout["split_dir"], output_dir))
    imported = _imported_manifest_for_v1(
        tmp_path / "prov", output_dir, provenance_complete=False
    )
    before = read_recordings_jsonl(imported)
    assert all(record.provenance_complete is False for record in before)
    assert all(record.origin_ids == () for record in before)
    args = _composed_cache_args()
    sampler = build_background_sampler(
        _online_config(
            [
                _source_config(
                    "musan",
                    imported,
                    cache_manifest=str(output_dir / "manifest.json"),
                )
            ],
            mode="fbank_cache",
            duration_seconds_min=args["duration_seconds_min"],
            duration_seconds_max=args["duration_seconds_max"],
        ),
        fbank_kwargs=args["expected_fbank_kwargs"],
    )
    sampler.close()
    after = read_recordings_jsonl(imported)
    assert all(record.provenance_complete is False for record in after)
    assert (output_dir / "manifest.json").is_file()
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["format_version"] == 1


def test_factory_errors_name_dataset_and_cache_identity(tmp_path):
    from dma_kws.data_prep.stage2_background import prepare_stage2_background
    from dma_kws.stage2.background_sources import build_background_sampler

    layout = _make_generic_catalog(tmp_path, dataset_id="dns")
    output_dir = tmp_path / "cache"
    prepare_stage2_background(**_prepare_v2_kwargs(layout["manifest"], output_dir))
    other = _make_generic_catalog(tmp_path / "other", dataset_id="fsd50k")
    args = _composed_cache_args()
    with pytest.raises(ValueError, match=r"dataset_id=.+dns.+fsd50k|dataset_id"):
        build_background_sampler(
            _online_config(
                [
                    _source_config(
                        "fsd50k",
                        other["manifest"],
                        cache_manifest=str(output_dir / "manifest.json"),
                    )
                ],
                mode="fbank_cache",
                duration_seconds_min=args["duration_seconds_min"],
                duration_seconds_max=args["duration_seconds_max"],
            ),
            fbank_kwargs=args["expected_fbank_kwargs"],
        )
