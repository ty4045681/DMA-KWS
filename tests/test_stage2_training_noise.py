import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from dma_kws.stage2.dataset import LibriPhraseTrainDataset
from dma_kws.stage2.features import TrainingNoiseAugmenter
from dma_kws.stage2.prepare_paper import (
    build_anchor_metadata,
    convert_aggregated_to_paper_parquet,
    resolve_fbank_rel_path,
    stream_fbank_from_decoded,
)
from dma_kws.stage2.train import _build_val_dataloader
from scripts.prepare_stage2_paper import resolve_training_waveform_dir


class _FakeTokenizer:
    def tokenize(self, text: str):
        tokens = text.split()
        return tokens, list(range(1, len(tokens) + 1))


def _training_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ngram": ["hello"],
            "ngram_g2p": ["HH AH0 L OW1"],
            "clips_file": ["clips-1-a.npy"],
            "distances_file": ["dist-0-a.npy"],
        }
    )


def _prep_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ngram": ["hello"],
            "ngram_g2p": ["HH AH0 L OW1"],
            "clips": [[{"audio_path": "LP-100/hello/a.wav"}]],
        }
    )


def _patch_positive_clip_loader(monkeypatch, *, cached_fbank: np.ndarray | None):
    clips = np.array(
        [{"audio_path": "LP-460/hello/a.wav"}],
        dtype=object,
    )
    loaded: list[str] = []

    def fake_load(path, allow_pickle=False):
        del allow_pickle
        loaded.append(str(path))
        if Path(path).name == "clips-1-a.npy":
            return clips
        if cached_fbank is not None and str(path).endswith(".npy"):
            return cached_fbank
        raise AssertionError(f"Unexpected np.load during training sample: {path}")

    monkeypatch.setattr("dma_kws.stage2.dataset.np.load", fake_load)
    return loaded


def test_noise_augmentation_is_off_by_default_and_uses_cached_fbank(monkeypatch):
    cached = np.full((5, 80), 2.0, dtype=np.float32)
    loaded = _patch_positive_clip_loader(monkeypatch, cached_fbank=cached)

    class _MustNotBeConstructed:
        def __init__(self, **_kwargs):
            raise AssertionError("disabled noise augmentation constructed its helper")

    monkeypatch.setattr(
        "dma_kws.stage2.features.TrainingNoiseAugmenter",
        _MustNotBeConstructed,
    )
    dataset = LibriPhraseTrainDataset(
        wav_dir="/features",
        tokenizer=_FakeTokenizer(),
        df=_training_df(),
        sample_lens=1,
        seed=1,
    )

    sample = dataset[0]

    assert torch.equal(sample["feat"], torch.from_numpy(cached))
    assert any("LP-460-fbank/hello/a.npy" in path for path in loaded)


def test_enabled_training_noise_uses_online_features_and_configured_fbank(monkeypatch):
    _patch_positive_clip_loader(monkeypatch, cached_fbank=None)
    calls: dict[str, object] = {}

    class _FakeNoiseAugmenter:
        def __init__(self, **kwargs):
            calls["init"] = kwargs

        def extract(self, wav_path: str, *, rng: random.Random) -> torch.Tensor:
            calls["wav_path"] = wav_path
            calls["rng"] = rng
            return torch.full((7, 80), 3.0)

    monkeypatch.setattr(
        "dma_kws.stage2.features.TrainingNoiseAugmenter",
        _FakeNoiseAugmenter,
    )
    fbank_kwargs = {
        "num_mel_bins": 80,
        "backend": "lhotse_fbank",
        "target_sample_rate": 16_000,
        "dither": 0.0,
        "snip_edges": False,
    }
    dataset = LibriPhraseTrainDataset(
        wav_dir="/features",
        tokenizer=_FakeTokenizer(),
        df=_training_df(),
        sample_lens=1,
        seed=1,
        noise_augmentation={
            "enabled": True,
            "probability": 1.0,
            "waveform_dir": "/waveforms",
            "noise_list_path": "/noise/noise.list",
            "snr_db_min": 10.0,
            "snr_db_max": 20.0,
        },
        fbank_kwargs=fbank_kwargs,
    )

    sample = dataset[0]

    assert sample["feat"].shape == (7, 80)
    assert torch.isfinite(sample["feat"]).all()
    assert calls["wav_path"] == "LP-460/hello/a.wav"
    assert calls["rng"] is dataset._rng
    assert calls["init"] == {
        "waveform_dir": "/waveforms",
        "noise_list_path": "/noise/noise.list",
        "snr_db_min": 10.0,
        "snr_db_max": 20.0,
        "fbank_kwargs": fbank_kwargs,
    }


def test_zero_noise_probability_keeps_cached_training_feature(monkeypatch):
    cached = np.full((4, 80), 4.0, dtype=np.float32)
    _patch_positive_clip_loader(monkeypatch, cached_fbank=cached)

    class _FakeNoiseAugmenter:
        def __init__(self, **_kwargs):
            pass

        def extract(self, _wav_path: str, *, rng: random.Random) -> torch.Tensor:
            del rng
            raise AssertionError("probability=0 must not invoke online augmentation")

    monkeypatch.setattr(
        "dma_kws.stage2.features.TrainingNoiseAugmenter",
        _FakeNoiseAugmenter,
    )
    dataset = LibriPhraseTrainDataset(
        wav_dir="/features",
        tokenizer=_FakeTokenizer(),
        df=_training_df(),
        sample_lens=1,
        seed=1,
        noise_augmentation={
            "enabled": True,
            "probability": 0.0,
            "waveform_dir": "/waveforms",
            "noise_list_path": "/noise/noise.list",
        },
    )

    assert torch.equal(dataset[0]["feat"], torch.from_numpy(cached))


def test_training_noise_is_seeded_deterministic_finite_and_80_dimensional(
    monkeypatch,
    tmp_path,
):
    waveform_dir = tmp_path / "waveforms"
    source = waveform_dir / "LP-460" / "hello" / "a.wav"
    source.parent.mkdir(parents=True)
    source.touch()
    noise = tmp_path / "noise.wav"
    noise.touch()
    noise_list = tmp_path / "noise.list"
    noise_list.write_text(f"{noise}\n", encoding="utf-8")

    clean_waveform = torch.linspace(-0.8, 0.8, 32).unsqueeze(0)
    noise_waveform = torch.tensor(([1.0, -0.5, 0.25, -0.75] * 16)).unsqueeze(0)

    def fake_load_audio(path: str | Path, **_kwargs):
        if Path(path) == noise:
            return noise_waveform.clone(), 16_000
        if Path(path) == source:
            return clean_waveform.clone(), 16_000
        raise AssertionError(path)

    class _FakeFbankExtractor:
        def __init__(self, **kwargs):
            assert kwargs["num_mel_bins"] == 80
            assert kwargs["dither"] == 0.0

        def prepare_waveform(self, waveform: torch.Tensor, sample_rate: int):
            return waveform, sample_rate

        def extract(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
            assert sample_rate == 16_000
            # Preserve some waveform dependence while emulating a [T, mel] fbank.
            return waveform[0, :6].unsqueeze(1).repeat(1, 80)

    monkeypatch.setattr("dma_kws.stage2.features._load_audio", fake_load_audio)
    monkeypatch.setattr("dma_kws.stage2.features.FbankExtractor", _FakeFbankExtractor)
    kwargs = {
        "waveform_dir": waveform_dir,
        "noise_list_path": noise_list,
        "snr_db_min": 10.0,
        "snr_db_max": 20.0,
        "fbank_kwargs": {"num_mel_bins": 80, "dither": 0.0},
    }

    first = TrainingNoiseAugmenter(**kwargs).extract(
        "LP-460/hello/a.wav",
        rng=random.Random(2025),
    )
    second = TrainingNoiseAugmenter(**kwargs).extract(
        "LP-460/hello/a.wav",
        rng=random.Random(2025),
    )

    assert first.shape == (6, 80)
    assert torch.isfinite(first).all()
    assert torch.equal(first, second)


def test_training_noise_resamples_clean_before_drawing_snr(monkeypatch, tmp_path):
    waveform_dir = tmp_path / "waveforms"
    source = waveform_dir / "LP-460" / "hello" / "a.wav"
    source.parent.mkdir(parents=True)
    source.touch()
    noise = tmp_path / "noise.wav"
    noise.touch()
    noise_list = tmp_path / "noise.list"
    noise_list.write_text(f"{noise}\n", encoding="utf-8")
    events: list[str] = []

    def fake_load_audio(path: str | Path, **_kwargs):
        if Path(path) == source:
            events.append("load_clean_8k")
            return torch.linspace(-0.5, 0.5, 8).unsqueeze(0), 8_000
        if Path(path) == noise:
            events.append("load_noise_16k")
            return torch.tensor([[1.0, -1.0] * 8]), 16_000
        raise AssertionError(path)

    class _FakeFbankExtractor:
        def __init__(self, **_kwargs):
            pass

        def prepare_waveform(self, waveform: torch.Tensor, sample_rate: int):
            assert sample_rate == 8_000
            events.append("prepare_clean_16k")
            return waveform.repeat_interleave(2, dim=1), 16_000

        def extract(self, waveform: torch.Tensor, sample_rate: int):
            assert sample_rate == 16_000
            events.append("extract_fbank_16k")
            return waveform[:, :4].T.repeat(1, 80)

    monkeypatch.setattr("dma_kws.stage2.features._load_audio", fake_load_audio)
    monkeypatch.setattr(
        "dma_kws.stage2.features.FbankExtractor", _FakeFbankExtractor
    )
    augmenter = TrainingNoiseAugmenter(
        waveform_dir=waveform_dir,
        noise_list_path=noise_list,
        snr_db_min=10.0,
        snr_db_max=10.0,
        fbank_kwargs={"target_sample_rate": 16_000},
    )

    feat = augmenter.extract("LP-460/hello/a.wav", rng=random.Random(4))

    assert feat.shape == (4, 80)
    assert events == [
        "load_clean_8k",
        "prepare_clean_16k",
        "load_noise_16k",
        "extract_fbank_16k",
    ]


def test_training_noise_mixer_hits_configured_snr(monkeypatch, tmp_path):
    waveform_dir = tmp_path / "waveforms"
    waveform_dir.mkdir()
    noise = tmp_path / "noise.wav"
    noise.touch()
    noise_list = tmp_path / "noise.list"
    noise_list.write_text(f"{noise}\n", encoding="utf-8")
    noise_waveform = torch.tensor([[1.0, -1.0] * 8])
    load_kwargs: dict[str, object] = {}

    def fake_load_audio(path: str | Path, **kwargs):
        assert Path(path) == noise
        load_kwargs.update(kwargs)
        return noise_waveform.clone(), 16_000

    monkeypatch.setattr("dma_kws.stage2.features._load_audio", fake_load_audio)
    augmenter = TrainingNoiseAugmenter(
        waveform_dir=waveform_dir,
        noise_list_path=noise_list,
        snr_db_min=10.0,
        snr_db_max=10.0,
    )
    clean = torch.full((1, 16), 0.5)

    rng = random.Random(9)
    mixed, sampled_snr = augmenter.mix(clean, 16_000, rng=rng)

    delta_power = (mixed - clean).square().mean()
    measured_snr = 10.0 * math.log10(float(clean.square().mean() / delta_power))
    assert sampled_snr == pytest.approx(10.0)
    assert measured_snr == pytest.approx(10.0, abs=1e-5)
    assert mixed.shape == clean.shape
    assert torch.isfinite(mixed).all()
    assert load_kwargs == {
        "rng": rng,
        "target_samples": 16,
        "target_sample_rate": 16_000,
    }


@pytest.mark.parametrize("probability", [-0.01, 1.01])
def test_training_noise_rejects_invalid_probability(monkeypatch, probability):
    class _MustNotBeConstructed:
        def __init__(self, **_kwargs):
            raise AssertionError("probability must be checked before source I/O")

    monkeypatch.setattr(
        "dma_kws.stage2.features.TrainingNoiseAugmenter",
        _MustNotBeConstructed,
    )

    with pytest.raises(ValueError, match="probability must be between 0 and 1"):
        LibriPhraseTrainDataset(
            wav_dir="/features",
            tokenizer=_FakeTokenizer(),
            df=_training_df(),
            noise_augmentation={
                "enabled": True,
                "probability": probability,
                "waveform_dir": "/waveforms",
                "noise_list_path": "/noise/noise.list",
            },
        )


def test_validation_loader_ignores_training_noise_augmentation(monkeypatch, tmp_path):
    test_dir = tmp_path / "eval"
    aggregate = test_dir / "evaluation_set" / "test_all_phrase.csv"
    aggregate.parent.mkdir(parents=True)
    pd.DataFrame(
        {
            "anchor_text": ["hello", "hello"],
            "anchor": ["a.wav", "b.wav"],
            "anchor_dur": [1.0, 1.0],
            "comparison_text": ["hello", "world"],
            "comparison": ["pos.wav", "hard.wav"],
            "comparison_dur": [1.0, 1.0],
            "target": [1, 0],
            "type": ["diffspk_positive", "diffspk_hardneg"],
        }
    ).to_csv(aggregate, index=False)
    cached = np.full((4, 80), 5.0, dtype=np.float32)
    monkeypatch.setattr("dma_kws.stage2.dataset.np.load", lambda *_args, **_kwargs: cached)
    monkeypatch.setattr(
        "dma_kws.stage2.dataset.make_g2p",
        lambda: lambda text: text.upper().split(),
    )
    config = {
        "paths": {},
        "tokenizer": {"dict_path": "unused", "split_with_space": " "},
        "stage2": {
            "noise_augmentation": {
                "enabled": True,
                "probability": 1.0,
                # Deliberately invalid: validation must not inspect training-only
                # waveform/noise sources or attempt online feature extraction.
                "waveform_dir": str(tmp_path / "missing-waveforms"),
                "noise_list_path": str(tmp_path / "missing-noise.list"),
            },
            "eval": {
                "test_dir": str(test_dir),
                "split": "hard",
                "aggregate_csv": "evaluation_set/test_all_phrase.csv",
                "batch_size": 2,
                "num_workers": 0,
            },
        },
    }

    batch = next(iter(_build_val_dataloader(config, _FakeTokenizer())))

    assert batch["feat"].shape == (2, 4, 80)
    assert torch.all(batch["feat"] == 5.0)


def test_prep_emits_waveform_only_target_when_fbank_is_already_cached(tmp_path):
    fbank_dir = tmp_path / "fbank"
    cached_fbank = fbank_dir / resolve_fbank_rel_path("LP-100/hello/a.wav")
    cached_fbank.parent.mkdir(parents=True)
    np.save(cached_fbank, np.ones((4, 80), dtype=np.float32))
    waveform_dir = tmp_path / "waveforms"

    _paper_df, targets, stats = build_anchor_metadata(
        _prep_df(),
        clips_dir=tmp_path / "clips",
        distances_dir=tmp_path / "distances",
        fbank_dir=fbank_dir,
        waveform_dir=waveform_dir,
    )

    assert set(targets) == {"hello/a.wav"}
    target = targets["hello/a.wav"]
    assert target.fbank_path == cached_fbank
    assert target.write_fbank is False
    assert target.waveform_path == waveform_dir / "LP-100/hello/a.wav"
    assert stats["fbank_skipped"] == 1
    assert stats["waveform_skipped"] == 0


def test_streaming_prep_fills_only_missing_waveform_cache(
    monkeypatch,
    tmp_path,
):
    fbank_dir = tmp_path / "fbank"
    cached_fbank = fbank_dir / resolve_fbank_rel_path("LP-100/hello/a.wav")
    cached_fbank.parent.mkdir(parents=True)
    np.save(cached_fbank, np.ones((4, 80), dtype=np.float32))
    waveform_dir = tmp_path / "waveforms"
    _paper_df, targets, _stats = build_anchor_metadata(
        _prep_df(),
        clips_dir=tmp_path / "clips",
        distances_dir=tmp_path / "distances",
        fbank_dir=fbank_dir,
        waveform_dir=waveform_dir,
    )
    decoded = pd.DataFrame(
        {
            "audio_rel": ["hello/a.wav"],
            "audio": [np.linspace(-0.5, 0.5, 160, dtype=np.float32)],
            "sampling_rate": [16_000],
        }
    )
    cached_waveforms: list[tuple[Path, int, int]] = []

    def fake_write(path, waveform, sample_rate):
        cached_waveforms.append((Path(path), len(waveform), int(sample_rate)))
        return str(path)

    def must_not_recompute_fbank(*_args, **_kwargs):
        raise AssertionError("waveform-only streaming must not recompute fbank")

    monkeypatch.setattr(
        "dma_kws.stage2.prepare_paper._write_waveform_cache",
        fake_write,
    )
    progress: list[tuple[str, int]] = []
    written, missing = stream_fbank_from_decoded(
        [tmp_path / "decoded.parquet"],
        targets,
        read_parquet=lambda _path: decoded,
        compute_fbank=must_not_recompute_fbank,
        on_progress=lambda stage, value: progress.append((stage, value)),
    )

    assert written == 0
    assert missing == 0
    assert cached_waveforms == [
        (waveform_dir / "LP-100/hello/a.wav", 160, 16_000)
    ]
    assert ("fbank_total", 0) in progress
    assert ("waveform_total", 1) in progress
    assert ("waveform", 1) in progress


def test_prep_rejects_two_missing_sources_with_same_decoded_audio_key(tmp_path):
    frame = pd.DataFrame(
        {
            "ngram": ["hello"],
            "ngram_g2p": ["HH AH0 L OW1"],
            "clips": [
                [
                    {"audio_path": "LP-460/hello/a.wav"},
                    {"audio_path": "GP-1000/hello/a.wav"},
                ]
            ],
        }
    )

    with pytest.raises(ValueError, match="decoded audio key collision"):
        build_anchor_metadata(
            frame,
            clips_dir=tmp_path / "clips",
            distances_dir=tmp_path / "distances",
            fbank_dir=tmp_path / "fbank",
            waveform_dir=tmp_path / "waveforms",
        )


def test_convert_writes_loadable_training_waveform_and_counts_it(tmp_path):
    sf = pytest.importorskip("soundfile")
    fbank_dir = tmp_path / "fbank"
    cached_fbank = fbank_dir / resolve_fbank_rel_path("LP-100/hello/a.wav")
    cached_fbank.parent.mkdir(parents=True)
    np.save(cached_fbank, np.ones((4, 80), dtype=np.float32))
    waveform_dir = tmp_path / "waveforms"
    waveform = np.linspace(-0.75, 0.75, 320, dtype=np.float32)

    def must_not_recompute_fbank(*_args, **_kwargs):
        raise AssertionError("cached fbank must not be recomputed for waveform caching")

    _paper_df, stats = convert_aggregated_to_paper_parquet(
        _prep_df(),
        clips_dir=tmp_path / "clips",
        distances_dir=tmp_path / "distances",
        fbank_dir=fbank_dir,
        waveform_dir=waveform_dir,
        audio_by_rel={"hello/a.wav": (waveform, 8_000)},
        compute_fbank=must_not_recompute_fbank,
    )

    cached_wav = waveform_dir / "LP-100/hello/a.wav"
    restored, sample_rate = sf.read(cached_wav, dtype="float32")
    assert sample_rate == 8_000
    assert restored.shape == waveform.shape
    assert np.allclose(restored, waveform, atol=1.0 / 32768.0)
    assert stats["fbank_written"] == 0
    assert stats["fbank_skipped"] == 1
    assert stats["waveform_written"] == 1


def test_resolve_training_waveform_dir_prefers_prep_then_enabled_stage2_fallback():
    stage2 = {
        "noise_augmentation": {
            "enabled": True,
            "waveform_dir": "/from-stage2",
        }
    }

    with pytest.raises(SystemExit, match="must point to the same cache"):
        resolve_training_waveform_dir({"waveform_dir": "/from-prep"}, stage2)
    assert resolve_training_waveform_dir(
        {"waveform_dir": "/from-stage2"}, stage2
    ) == Path("/from-stage2")
    assert resolve_training_waveform_dir({}, stage2) == Path("/from-stage2")
    assert (
        resolve_training_waveform_dir(
            {},
            {
                "noise_augmentation": {
                    "enabled": False,
                    "waveform_dir": "/disabled",
                }
            },
        )
        is None
    )

    with pytest.raises(SystemExit, match="waveform_dir is required"):
        resolve_training_waveform_dir(
            {},
            {"noise_augmentation": {"enabled": True, "waveform_dir": ""}},
        )
