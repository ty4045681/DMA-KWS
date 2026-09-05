from __future__ import annotations

import hashlib
import json
import multiprocessing
from pathlib import Path
import pickle
import random
import shutil

import numpy as np
import pytest
import torch

from dma_kws.config import compose_config, config_to_dict, fbank_kwargs, get_fbank_config
from dma_kws.stage2.background_sampling import (
    BackgroundCropSpec,
    draw_crop_spec,
    materialize_crop,
    probe_source_info,
)
from dma_kws.stage2.dataset import LibriPhraseTrainDataset
from dma_kws.stage2.fbank import FbankExtractor
from dma_kws.stage2.train import background_negative_run_paths
from dma_kws.training.metrics_history import collect_hparams
from tests.test_prepare_stage2_background import (
    DEFAULT_EXPERIMENT,
    SEED,
    _extraction_overrides,
    _make_synthetic_split,
    _oracle_crop_rng,
    _prepare_kwargs,
)
from tests.test_stage2_background_config import _FakeTokenizer, _mock_dataframe


def _composed_cache_args() -> dict:
    composed = config_to_dict(
        compose_config(DEFAULT_EXPERIMENT, _extraction_overrides())
    )
    background = composed["stage2"]["background_negative"]
    return {
        "expected_fbank_kwargs": fbank_kwargs(get_fbank_config(composed)),
        "duration_seconds_min": float(background["duration_seconds_min"]),
        "duration_seconds_max": float(background["duration_seconds_max"]),
        "composed": composed,
    }


def _build_cache(tmp_path: Path, **overrides) -> dict:
    from dma_kws.data_prep.stage2_background import prepare_stage2_background

    layout = _make_synthetic_split(tmp_path)
    output_dir = tmp_path / "cache"
    kwargs = _prepare_kwargs(layout["split_dir"], output_dir)
    kwargs.update(overrides)
    prepare_stage2_background(**kwargs)
    args = _composed_cache_args()
    return {
        "layout": layout,
        "output_dir": output_dir,
        "manifest_path": output_dir / "manifest.json",
        **args,
    }


def _open_cache(built: dict, **overrides):
    from dma_kws.stage2.background_cache import BackgroundFeatureCache

    kwargs = {
        "expected_fbank_kwargs": built["expected_fbank_kwargs"],
        "duration_seconds_min": built["duration_seconds_min"],
        "duration_seconds_max": built["duration_seconds_max"],
    }
    kwargs.update(overrides)
    return BackgroundFeatureCache(built["manifest_path"], **kwargs)


def _load_manifest(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _restamp_manifest(cache_dir: Path, manifest: dict) -> dict:
    from dma_kws.data_prep.stage2_background import _cache_id_for

    manifest["cache_id"] = _cache_id_for(manifest)
    (cache_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def _file_digest(path: Path) -> tuple[str, int]:
    payload = path.read_bytes()
    return hashlib.sha256(payload).hexdigest(), len(payload)


def _boom(*_args, **_kwargs):
    raise AssertionError("cache path must not decode audio or construct FbankExtractor")


def _patch_online_feature_path(monkeypatch) -> None:
    monkeypatch.setattr("dma_kws.stage2.features._load_audio", _boom)
    monkeypatch.setattr("dma_kws.stage2.fbank.FbankExtractor", _boom)
    monkeypatch.setattr("dma_kws.stage2.features.FbankExtractor", _boom)


def _mock_speech_metadata(monkeypatch) -> None:
    clips = {
        "clips-2-a.npy": np.array(
            [{"audio_path": "LP-460/hello/a.wav"}, {"audio_path": "LP-460/hello/b.wav"}],
            dtype=object,
        ),
        "clips-2-b.npy": np.array(
            [{"audio_path": "LP-460/world/c.wav"}, {"audio_path": "LP-460/world/d.wav"}],
            dtype=object,
        ),
    }
    distances = {
        "dist-0-a.npy": np.array([], dtype=object),
        "dist-2-b.npy": np.array([{"ngram": "hello"}, {"ngram": "hello"}], dtype=object),
    }
    fbank = np.ones((5, 80), dtype=np.float32)
    original_load = np.load

    def fake_load(path, *args, **kwargs):
        name = Path(path).name
        if name in clips:
            return clips[name]
        if name in distances:
            return distances[name]
        if name.endswith(".npy") and "fbank" in str(path):
            return fbank
        return original_load(path, *args, **kwargs)

    monkeypatch.setattr("dma_kws.stage2.dataset.np.load", fake_load)


def _spawn_extract_from_pickle(payload: bytes, seed: int, queue) -> None:
    cache = pickle.loads(payload)
    try:
        tensor = cache.extract(rng=random.Random(seed))
        queue.put(
            {
                "shape": tuple(tensor.shape),
                "dtype": str(tensor.dtype),
                "device": tensor.device.type,
                "mean": float(tensor.mean().item()),
                "writable": bool(tensor.is_contiguous() and tensor.dtype == torch.float32),
            }
        )
    finally:
        cache.close()


def test_extract_works_after_original_audio_is_deleted(tmp_path, monkeypatch):
    built = _build_cache(tmp_path)
    shutil.rmtree(built["layout"]["musan_root"])
    _patch_online_feature_path(monkeypatch)
    cache = _open_cache(
        built,
        audio_list_path=str(built["layout"]["split_dir"] / "train_background.list"),
    )
    try:
        feats = cache.extract(rng=random.Random(0))
        assert feats.dtype == torch.float32
        assert feats.device.type == "cpu"
        assert feats.ndim == 2
        assert feats.shape[0] >= 1
        assert feats.shape[1] == int(built["expected_fbank_kwargs"]["num_mel_bins"])
        assert bool(torch.isfinite(feats).all())
        feats.add_(1.0)
    finally:
        cache.close()


def test_dataset_getitem_background_survives_deleted_audio_and_online_monkeypatch(
    tmp_path, monkeypatch
):
    built = _build_cache(tmp_path)
    shutil.rmtree(built["layout"]["musan_root"])
    _mock_speech_metadata(monkeypatch)
    _patch_online_feature_path(monkeypatch)

    dataset = LibriPhraseTrainDataset(
        wav_dir="/data/segments",
        tokenizer=_FakeTokenizer(),
        df=_mock_dataframe(),
        sample_lens=8,
        seed=0,
        background_negative={
            "enabled": True,
            "mode": "fbank_cache",
            "probability": 1.0,
            "cache_manifest": str(built["manifest_path"]),
            "audio_list_path": "",
            "duration_seconds_min": built["duration_seconds_min"],
            "duration_seconds_max": built["duration_seconds_max"],
            "max_open_shards": 8,
        },
        fbank_kwargs=built["expected_fbank_kwargs"],
    )
    sample = dataset[0]
    assert sample["label"].item() == 0
    assert sample["query_seq"].numel() == 0
    assert set(sample.keys()) == {
        "anchor_seq",
        "query_seq",
        "feat",
        "label",
        "seq_label",
    }
    assert sample["feat"].dtype == torch.float32
    assert sample["feat"].device.type == "cpu"
    assert sample["feat"].ndim == 2
    assert sample["seq_label"].numel() == sample["anchor_seq"].numel()


def test_disabled_online_and_cached_dataset_construction(tmp_path, monkeypatch):
    from dma_kws.stage2.background_cache import BackgroundFeatureCache
    from dma_kws.stage2.features import TrainingBackgroundSampler

    built = _build_cache(tmp_path)

    class _MustNotConstructCache:
        def __init__(self, *_args, **_kwargs):
            raise AssertionError("disabled path must not open the cache")

    monkeypatch.setattr(
        "dma_kws.stage2.background_cache.BackgroundFeatureCache",
        _MustNotConstructCache,
    )
    disabled = LibriPhraseTrainDataset(
        wav_dir="/data/segments",
        tokenizer=_FakeTokenizer(),
        df=_mock_dataframe(),
        background_negative={
            "enabled": False,
            "mode": "fbank_cache",
            "cache_manifest": str(built["manifest_path"]),
            "max_open_shards": 4,
        },
        fbank_kwargs=built["expected_fbank_kwargs"],
    )
    assert disabled._background_sampler is None
    monkeypatch.undo()

    constructed = {}

    class _FakeOnline:
        def __init__(self, **kwargs):
            constructed.update(kwargs)

        def extract(self, *, rng):
            raise AssertionError("online extract is not under test")

    monkeypatch.setattr(
        "dma_kws.stage2.features.TrainingBackgroundSampler",
        _FakeOnline,
    )
    online = LibriPhraseTrainDataset(
        wav_dir="/data/segments",
        tokenizer=_FakeTokenizer(),
        df=_mock_dataframe(),
        background_negative={
            "enabled": True,
            "mode": "online",
            "audio_list_path": "/background/musan.list",
            "duration_seconds_min": 1.0,
            "duration_seconds_max": 3.0,
        },
        fbank_kwargs={"dither": 0.0},
    )
    assert constructed["audio_list_path"] == "/background/musan.list"
    assert online._background_sampler is not None
    assert not isinstance(online._background_sampler, BackgroundFeatureCache)
    monkeypatch.undo()

    cached = LibriPhraseTrainDataset(
        wav_dir="/data/segments",
        tokenizer=_FakeTokenizer(),
        df=_mock_dataframe(),
        background_negative={
            "enabled": True,
            "mode": "fbank_cache",
            "cache_manifest": str(built["manifest_path"]),
            "audio_list_path": "",
            "duration_seconds_min": built["duration_seconds_min"],
            "duration_seconds_max": built["duration_seconds_max"],
        },
        fbank_kwargs=built["expected_fbank_kwargs"],
    )
    assert isinstance(cached._background_sampler, BackgroundFeatureCache)
    assert not isinstance(cached._background_sampler, TrainingBackgroundSampler)
    cached._background_sampler.close()


def test_unknown_keys_and_nonzero_dither_still_rejected(tmp_path):
    built = _build_cache(tmp_path)
    with pytest.raises(ValueError, match="Unknown stage2.background_negative fields"):
        LibriPhraseTrainDataset(
            wav_dir="/data/segments",
            tokenizer=_FakeTokenizer(),
            df=_mock_dataframe(),
            background_negative={"enabled": False, "typo": True},
        )
    with pytest.raises(ValueError, match="dither"):
        LibriPhraseTrainDataset(
            wav_dir="/data/segments",
            tokenizer=_FakeTokenizer(),
            df=_mock_dataframe(),
            background_negative={
                "enabled": True,
                "mode": "fbank_cache",
                "cache_manifest": str(built["manifest_path"]),
            },
            fbank_kwargs={"dither": 0.1},
        )


def test_max_open_shards_true_rejected_on_dataset_and_reader(tmp_path):
    from dma_kws.stage2.background_cache import BackgroundFeatureCache

    built = _build_cache(tmp_path)
    with pytest.raises(
        ValueError,
        match=r"stage2\.background_negative\.max_open_shards must be a positive int",
    ):
        LibriPhraseTrainDataset(
            wav_dir="/data/segments",
            tokenizer=_FakeTokenizer(),
            df=_mock_dataframe(),
            background_negative={
                "enabled": False,
                "max_open_shards": True,
            },
        )
    with pytest.raises(ValueError, match="max_open_shards"):
        BackgroundFeatureCache(
            built["manifest_path"],
            expected_fbank_kwargs=built["expected_fbank_kwargs"],
            duration_seconds_min=built["duration_seconds_min"],
            duration_seconds_max=built["duration_seconds_max"],
            max_open_shards=True,
        )


def test_missing_shard_wrong_offset_wrong_dtype_name_crop_and_path(tmp_path):
    built = _build_cache(tmp_path)
    manifest = _load_manifest(built["manifest_path"])
    shard_name = manifest["shards"][0]["path"]
    shard_path = built["output_dir"] / shard_name
    missing_dir = tmp_path / "missing-shard"
    shutil.copytree(built["output_dir"], missing_dir)
    (missing_dir / shard_name).unlink()
    with pytest.raises((FileNotFoundError, ValueError), match=shard_name):
        cache = _open_cache({**built, "manifest_path": missing_dir / "manifest.json"})
        cache.close()

    crops_path = built["output_dir"] / "crops.npy"
    crops = np.load(crops_path, allow_pickle=False)
    crop_id = int(crops[0]["crop_id"])
    crops[0]["frame_offset"] = 10**9
    np.save(crops_path, crops, allow_pickle=False)
    digest, size = _file_digest(crops_path)
    manifest["crops"]["sha256"] = digest
    manifest["crops"]["size"] = size
    _restamp_manifest(built["output_dir"], manifest)
    with pytest.raises(ValueError, match=str(crop_id)):
        cache = _open_cache(built)
        cache.close()

    clean = _build_cache(tmp_path / "clean")
    clean_manifest = _load_manifest(clean["manifest_path"])
    shard_name = clean_manifest["shards"][0]["path"]
    clean_shard = clean["output_dir"] / shard_name
    array = np.load(clean_shard, allow_pickle=False)
    np.save(clean_shard, array.astype(np.float64), allow_pickle=False)
    digest, size = _file_digest(clean_shard)
    clean_manifest["shards"][0]["sha256"] = digest
    clean_manifest["shards"][0]["size"] = size
    _restamp_manifest(clean["output_dir"], clean_manifest)
    with pytest.raises(ValueError, match=shard_name):
        cache = _open_cache(clean)
        cache.close()


def test_reader_is_picklable_and_spawn_worker_can_extract(tmp_path):
    built = _build_cache(tmp_path)
    cache = _open_cache(built, max_open_shards=2)
    try:
        first = cache.read_crop(0)
        state = cache.__getstate__()
        assert dict(state["_open_shards"]) == {}
        payload = pickle.dumps(cache)
        restored = pickle.loads(payload)
        try:
            assert len(restored._open_shards) == 0
            again = restored.extract(rng=random.Random(3))
            assert again.dtype == torch.float32
            assert again.device.type == "cpu"
            assert again.ndim == 2
        finally:
            restored.close()

        ctx = multiprocessing.get_context("spawn")
        queue = ctx.Queue()
        proc = ctx.Process(
            target=_spawn_extract_from_pickle,
            args=(payload, 11, queue),
        )
        proc.start()
        result = queue.get(timeout=60)
        proc.join(timeout=60)
        assert proc.exitcode == 0
        assert result["dtype"] == "torch.float32"
        assert result["device"] == "cpu"
        assert result["shape"][1] == int(built["expected_fbank_kwargs"]["num_mel_bins"])
        assert first.shape[1] == result["shape"][1]
    finally:
        cache.close()


def test_lru_high_water_and_returned_tensor_survives_eviction(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "dma_kws.data_prep.stage2_background._max_frames_per_shard",
        lambda *_args, **_kwargs: 1,
    )
    built = _build_cache(tmp_path)
    manifest = _load_manifest(built["manifest_path"])
    assert len(manifest["shards"]) >= 3
    cache = _open_cache(built, max_open_shards=2)
    try:
        high_water = 0
        tensors = []
        clones = []
        for crop_id in range(int(manifest["num_crops"])):
            tensor = cache.read_crop(crop_id)
            tensors.append(tensor)
            clones.append(tensor.clone())
            high_water = max(high_water, len(cache._open_shards))
        assert high_water <= 2
        assert len(cache._open_shards) <= 2
        assert torch.equal(tensors[0], clones[0])
        tensors[0].add_(1.5)
        assert not torch.equal(tensors[0], clones[0])
        torch.testing.assert_close(clones[0], cache.read_crop(0))
        cache.read_crop(int(manifest["num_crops"]) - 1)
        assert torch.equal(tensors[1], clones[1])
    finally:
        cache.close()


def test_read_crop_matches_extract_selected_id_and_stored_frames(tmp_path):
    built = _build_cache(tmp_path)
    cache = _open_cache(built)
    try:
        crops = np.load(built["output_dir"] / "crops.npy", allow_pickle=False)
        recordings = [
            json.loads(line)
            for line in (built["output_dir"] / "recordings.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ]
        starts = [int(row["crop_start"]) for row in recordings]
        counts = [int(row["crop_count"]) for row in recordings]
        rng = random.Random(2025)
        oracle = random.Random(2025)
        for _ in range(16):
            source = oracle.randrange(len(recordings))
            ordinal = oracle.randrange(counts[source])
            crop_id = starts[source] + ordinal
            extracted = cache.extract(rng=rng)
            stored = cache.read_crop(crop_id)
            torch.testing.assert_close(extracted, stored)
            row = crops[crop_id]
            shard = np.load(
                built["output_dir"]
                / f"features-{int(row['shard_index']):05d}.npy",
                allow_pickle=False,
            )
            frames = shard[
                int(row["frame_offset"]) : int(row["frame_offset"])
                + int(row["num_frames"])
            ]
            torch.testing.assert_close(
                stored,
                torch.from_numpy(np.ascontiguousarray(frames)),
            )
    finally:
        cache.close()


def test_duration_does_not_change_source_then_crop_selection(tmp_path):
    built = _build_cache(tmp_path)
    cache = _open_cache(built)
    try:
        before = [cache.extract(rng=random.Random(i)).clone() for i in range(8)]
    finally:
        cache.close()

    crops_path = built["output_dir"] / "crops.npy"
    crops = np.load(crops_path, allow_pickle=False)
    crops["duration_seconds"] = crops["duration_seconds"][::-1] * 3.0
    np.save(crops_path, crops, allow_pickle=False)
    manifest = _load_manifest(built["manifest_path"])
    digest, size = _file_digest(crops_path)
    manifest["crops"]["sha256"] = digest
    manifest["crops"]["size"] = size
    _restamp_manifest(built["output_dir"], manifest)

    cache = _open_cache(built)
    try:
        after = [cache.extract(rng=random.Random(i)) for i in range(8)]
        for left, right in zip(before, after):
            torch.testing.assert_close(left, right)
    finally:
        cache.close()


def test_read_crop_matches_online_fbank_of_stored_spec(tmp_path):
    built = _build_cache(tmp_path)
    cache = _open_cache(built)
    try:
        crops = np.load(built["output_dir"] / "crops.npy", allow_pickle=False)
        layout = built["layout"]
        sources = {
            layout["long_id"]: probe_source_info(
                layout["long"], source_id=layout["long_id"]
            ),
            layout["short_id"]: probe_source_info(
                layout["short"], source_id=layout["short_id"]
            ),
        }
        extractor = FbankExtractor(**built["expected_fbank_kwargs"])
        for row in crops:
            source_id = str(row["source_id"])
            source = sources[source_id]
            oracle_rng = _oracle_crop_rng(SEED, source_id, int(row["ordinal"]))
            duration = oracle_rng.uniform(
                built["duration_seconds_min"], built["duration_seconds_max"]
            )
            spec = draw_crop_spec(source, duration, rng=oracle_rng)
            assert spec == BackgroundCropSpec(
                source_id=source_id,
                duration_seconds=float(row["duration_seconds"]),
                read_start_frame=int(row["read_start_frame"]),
                read_num_frames=int(row["read_num_frames"]),
                target_num_samples=int(row["target_num_samples"]),
                final_offset=int(row["final_offset"]),
            )
            waveform, sample_rate = materialize_crop(source, spec)
            expected = extractor.extract(waveform, sample_rate)
            stored = cache.read_crop(int(row["crop_id"]))
            assert stored.shape == tuple(expected.shape)
            torch.testing.assert_close(stored, expected.cpu(), rtol=1e-5, atol=1e-5)
    finally:
        cache.close()


def test_fbank_and_duration_mismatch_have_no_online_fallback(tmp_path):
    built = _build_cache(tmp_path)
    with pytest.raises(ValueError, match="fbank"):
        _open_cache(
            built,
            expected_fbank_kwargs={
                **built["expected_fbank_kwargs"],
                "num_mel_bins": 40,
            },
        )
    with pytest.raises(ValueError, match="duration"):
        _open_cache(built, duration_seconds_min=0.5)


def test_audio_list_identity_does_not_require_wavs(tmp_path):
    built = _build_cache(tmp_path)
    list_path = built["layout"]["split_dir"] / "train_background.list"
    shutil.rmtree(built["layout"]["musan_root"])
    cache = _open_cache(built, audio_list_path=str(list_path))
    cache.close()

    other = tmp_path / "other.list"
    other.write_text("not-a-source.wav\n", encoding="utf-8")
    with pytest.raises((ValueError, FileNotFoundError), match="other.list|audio list"):
        _open_cache(built, audio_list_path=str(other))

    missing = tmp_path / "missing.list"
    with pytest.raises(FileNotFoundError, match="missing.list"):
        _open_cache(built, audio_list_path=str(missing))


def test_run_record_fields_and_cached_overlay_keep_readout_v4(tmp_path):
    built = _build_cache(tmp_path)
    cache = _open_cache(built)
    try:
        manifest = _load_manifest(built["manifest_path"])
        fields = cache.run_record_fields()
        assert fields["background_cache_id"] == manifest["cache_id"]
        assert fields["background_cache_manifest"] == str(cache.manifest_path)
        assert fields["background_cache_format_version"] == 1
        assert fields["background_crop_count"] == int(manifest["num_crops"])
        assert fields["background_fbank"] == manifest["fbank"]
        paths = background_negative_run_paths(
            {
                "enabled": True,
                "mode": "fbank_cache",
                "audio_list_path": "",
            },
            cache,
        )
        assert paths["background_audio_list"] == ""
        assert paths["background_cache_id"] == manifest["cache_id"]
        assert "source_mode" not in paths
    finally:
        cache.close()

    cached = config_to_dict(
        compose_config("icefall_zipformer_stage2_eps_softmin_v41_cached")
    )
    assert cached["stage2"]["qbyt_readout_version"] == 4
    hparams = collect_hparams(cached)
    assert hparams["qbyt_readout_version"] == 4
    assert hparams["qbyt_readout_mode"] == "eps_softmin"
    assert "source_mode" not in hparams


def test_cache_init_does_not_open_feature_mmaps(tmp_path, monkeypatch):
    built = _build_cache(tmp_path)
    loads: list[tuple[str, object]] = []
    original = np.load

    def tracking_load(path, *args, **kwargs):
        loads.append((str(path), kwargs.get("mmap_mode")))
        return original(path, *args, **kwargs)

    monkeypatch.setattr("dma_kws.stage2.background_cache.np.load", tracking_load)
    cache = _open_cache(built)
    try:
        mmap_loads = [item for item in loads if item[1] == "r"]
        assert mmap_loads == []
        cache.read_crop(0)
        mmap_loads = [item for item in loads if item[1] == "r"]
        assert mmap_loads
    finally:
        cache.close()
