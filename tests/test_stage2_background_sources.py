from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import soundfile as sf
import torch

from dma_kws.data_prep.background_manifest import (
    background_record_from_mapping,
    write_recordings_jsonl,
)
from dma_kws.stage2.background_sources import (
    BackgroundSample,
    CachedBackgroundSource,
    LegacyBackgroundSamplerAdapter,
    MultiSourceBackgroundSampler,
    OnlineBackgroundSource,
    build_background_sampler,
)
from dma_kws.stage2.collate import train_collate_fn
from dma_kws.stage2.dataset import LibriPhraseTrainDataset
from dma_kws.stage2.features import TrainingBackgroundSampler


def _write_wav(path: Path, array: np.ndarray, sample_rate: int = 16_000) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(
        str(path),
        np.asarray(array, dtype=np.float32),
        sample_rate,
        subtype="FLOAT",
    )
    return path


def _record(
    *,
    dataset_id: str,
    recording_id: str,
    audio_path: str | Path,
    split: str = "train",
    eligible: bool = True,
) -> Any:
    return background_record_from_mapping(
        {
            "schema_version": 1,
            "dataset_id": dataset_id,
            "recording_id": recording_id,
            "relative_path": Path(audio_path).name,
            "audio_path": str(audio_path),
            "group_id": f"{dataset_id}:group",
            "origin_ids": [recording_id],
            "split": split,
            "categories": ["noise"],
            "background_eligible": eligible,
            "eligibility_basis": "curated_allowlist_v1",
            "duration_seconds": 1.0,
            "sample_rate": 16000,
            "channels": 1,
            "audio_sha256": "a" * 64,
            "license_id": "cc0",
            "provenance_complete": True,
        }
    )


def _write_manifest(path: Path, records) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_recordings_jsonl(path, records)
    return path


def _source_config(
    source_id: str,
    manifest: str | Path,
    *,
    weight: float = 1.0,
    cache_manifest: str = "",
) -> dict[str, Any]:
    return {
        "id": source_id,
        "weight": weight,
        "manifest": str(manifest),
        "cache_manifest": cache_manifest,
    }


def _online_config(sources: list[dict[str, Any]], **overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "enabled": True,
        "probability": 0.25,
        "mode": "online",
        "audio_list_path": "",
        "cache_manifest": "",
        "duration_seconds_min": 1.0,
        "duration_seconds_max": 1.0,
        "max_open_shards": 8,
        "sources": sources,
        "validation": {"enabled": False, "samples_per_source": 256, "seed": 2025},
    }
    payload.update(overrides)
    return payload


class _FakeSourceSampler:
    def __init__(self, name: str, feat: torch.Tensor | None = None) -> None:
        self.name = name
        self.feat = torch.ones(2, 3) if feat is None else feat
        self.draws: list[str] = []
        self.closed = False

    def sample(self, *, rng: random.Random) -> BackgroundSample:
        self.draws.append(self.name)
        return BackgroundSample(
            feat=self.feat,
            background_source_id=-99,
            recording_id=f"{self.name}:rec",
            crop_id=None,
        )

    def run_record_fields(self) -> dict[str, Any]:
        return {"name": self.name}

    def close(self) -> None:
        self.closed = True


class _CountingRandom(random.Random):
    def __init__(self, seed: int) -> None:
        super().__init__(seed)
        self.random_calls = 0

    def random(self) -> float:
        self.random_calls += 1
        return super().random()


class _ScriptedRandom(random.Random):
    def __init__(self, values: list[float]) -> None:
        super().__init__(0)
        self._values = list(values)

    def random(self) -> float:
        return self._values.pop(0)


class _IdentityFbank:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def extract(self, waveform, sample_rate):
        return waveform.transpose(0, 1)


def _patch_online_fbank(monkeypatch) -> None:
    monkeypatch.setattr(
        "dma_kws.stage2.background_sources.FbankExtractor",
        _IdentityFbank,
    )


def _mock_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ngram": ["hello", "world"],
            "ngram_g2p": ["HH AH0 L OW1", "W ER1 L D"],
            "clips_file": ["clips-2-a.npy", "clips-2-b.npy"],
            "distances_file": ["dist-0-a.npy", "dist-2-b.npy"],
        }
    )


class _FakeTokenizer:
    def tokenize(self, text: str):
        tokens = text.split()
        ids = [hash(token) % 100 + 1 for token in tokens]
        return tokens, ids


@pytest.fixture
def mock_npy_loader(monkeypatch):
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

    def fake_load(path, allow_pickle=False):
        name = Path(path).name
        if name in clips:
            return clips[name]
        if name in distances:
            return distances[name]
        if name.endswith(".npy") and "fbank" in str(path):
            return fbank
        raise FileNotFoundError(path)

    monkeypatch.setattr("dma_kws.stage2.dataset.np.load", fake_load)
    return fbank


def _make_three_source_sampler() -> MultiSourceBackgroundSampler:
    return MultiSourceBackgroundSampler(
        [
            ("musan", 0.2, _FakeSourceSampler("musan")),
            ("dns", 0.4, _FakeSourceSampler("dns")),
            ("fsd50k", 0.4, _FakeSourceSampler("fsd50k")),
        ]
    )


def _bin_tolerance(probability: float, n: int) -> float:
    return max(0.005, 6.0 * math.sqrt(probability * (1.0 - probability) / n))


def test_build_background_sampler_disabled_returns_none_without_io(monkeypatch):
    def _forbid_open(*_args, **_kwargs):
        raise AssertionError("disabled factory must not open files")

    monkeypatch.setattr("builtins.open", _forbid_open)
    sampler = build_background_sampler(
        {
            "enabled": False,
            "mode": "online",
            "audio_list_path": "",
            "sources": [
                {
                    "id": "musan",
                    "weight": 1.0,
                    "manifest": "/missing/recordings.jsonl",
                }
            ],
        },
        fbank_kwargs={},
    )
    assert sampler is None


def test_build_background_sampler_legacy_online_wraps_training_sampler(
    tmp_path, monkeypatch
):
    audio = _write_wav(tmp_path / "noise.wav", np.linspace(-0.2, 0.2, 1600))
    audio_list = tmp_path / "background.list"
    audio_list.write_text(f"{audio.name}\n", encoding="utf-8")
    constructed: dict[str, Any] = {}

    class _FakeTraining(TrainingBackgroundSampler):
        def __init__(self, **kwargs):
            constructed.update(kwargs)
            self.audio_paths = [audio]

        def extract(self, *, rng):
            rng.uniform(0.0, 1.0)
            return torch.full((3, 4), 2.0)

    monkeypatch.setattr(
        "dma_kws.stage2.features.TrainingBackgroundSampler",
        _FakeTraining,
    )
    sampler = build_background_sampler(
        {
            "enabled": True,
            "mode": "online",
            "audio_list_path": str(audio_list),
            "duration_seconds_min": 1.0,
            "duration_seconds_max": 3.0,
            "sources": [],
        },
        fbank_kwargs={"dither": 0.0},
    )
    assert isinstance(sampler, LegacyBackgroundSamplerAdapter)
    assert constructed == {
        "audio_list_path": str(audio_list),
        "duration_seconds_min": 1.0,
        "duration_seconds_max": 3.0,
        "fbank_kwargs": {"dither": 0.0},
    }
    rng = random.Random(3)
    sample = sampler.sample(rng=rng)
    assert torch.equal(sample.feat, torch.full((3, 4), 2.0))
    assert sample.background_source_id == 0
    leftover = rng.random()
    oracle = random.Random(3)
    sampler.extract(rng=oracle)
    assert leftover == oracle.random()


def test_legacy_adapter_sample_does_not_draw_rng_twice():
    class _Inner:
        def extract(self, *, rng):
            return torch.tensor([rng.random()])

    adapter = LegacyBackgroundSamplerAdapter(_Inner(), background_source_id=0)
    rng = random.Random(9)
    sample = adapter.sample(rng=rng)
    after_sample = rng.random()
    oracle = random.Random(9)
    feat = adapter.extract(rng=oracle)
    after_extract = oracle.random()
    assert torch.equal(sample.feat, feat)
    assert after_sample == after_extract


def test_fbank_cache_with_nonempty_sources_raises_not_yet_wired():
    with pytest.raises(
        ValueError,
        match=r"fbank_cache.*non-empty sources.*not yet wired.*CachedBackgroundSource",
    ):
        build_background_sampler(
            _online_config(
                [
                    _source_config(
                        "musan",
                        "/data/musan/recordings.jsonl",
                        cache_manifest="/data/musan/cache/manifest.json",
                    )
                ],
                mode="fbank_cache",
            ),
            fbank_kwargs={"dither": 0.0},
        )
    with pytest.raises(ValueError, match="not yet wired"):
        CachedBackgroundSource()


def test_single_positive_weight_source_does_not_consume_source_selection_rng():
    sampler = MultiSourceBackgroundSampler(
        [("musan", 1.0, _FakeSourceSampler("musan"))]
    )
    rng = _CountingRandom(0)
    sample = sampler.sample(rng=rng)
    assert rng.random_calls == 0
    assert sample.background_source_id == 0
    assert sample.recording_id == "musan:rec"


def test_source_selection_last_bin_includes_one_and_boundaries():
    sampler = _make_three_source_sampler()
    # Sorted active ids: dns, fsd50k, musan with 0.4 / 0.4 / 0.2.
    expected = [
        (0.0, 0, "dns"),
        (0.399999, 0, "dns"),
        (0.4, 1, "fsd50k"),
        (0.799999, 1, "fsd50k"),
        (0.8, 2, "musan"),
        (0.999, 2, "musan"),
        (1.0, 2, "musan"),
    ]
    for draw, source_id, name in expected:
        sample = sampler.sample(rng=_ScriptedRandom([draw]))
        assert sample.background_source_id == source_id
        assert sample.recording_id == f"{name}:rec"


def test_t04_weighted_source_selection_is_reproducible_and_order_invariant():
    n = 100_000
    seed = 2025
    sampler_a = _make_three_source_sampler()
    sampler_b = MultiSourceBackgroundSampler(
        [
            ("fsd50k", 0.4, _FakeSourceSampler("fsd50k")),
            ("dns", 0.4, _FakeSourceSampler("dns")),
            ("musan", 0.2, _FakeSourceSampler("musan")),
        ]
    )
    rng_a = random.Random(seed)
    rng_b = random.Random(seed)
    counts = [0, 0, 0]
    sequence_a = []
    sequence_b = []
    for _ in range(n):
        sample_a = sampler_a.sample(rng=rng_a)
        sample_b = sampler_b.sample(rng=rng_b)
        sequence_a.append(sample_a.background_source_id)
        sequence_b.append(sample_b.background_source_id)
        counts[sample_a.background_source_id] += 1
    assert sequence_a == sequence_b
    expected = [0.4, 0.4, 0.2]
    for source_id, probability in enumerate(expected):
        observed = counts[source_id] / n
        assert abs(observed - probability) <= _bin_tolerance(probability, n)


def test_t04_recordings_are_uniform_even_with_unequal_crop_room(
    tmp_path, monkeypatch
):
    _patch_online_fbank(monkeypatch)
    long_wav = _write_wav(
        tmp_path / "long.wav",
        np.linspace(-0.5, 0.5, 64_000, dtype=np.float32),
    )
    short_wav = _write_wav(
        tmp_path / "short.wav",
        np.linspace(0.1, -0.1, 8_000, dtype=np.float32),
    )
    records = [
        _record(
            dataset_id="musan",
            recording_id="musan:short",
            audio_path=short_wav,
        ),
        _record(
            dataset_id="musan",
            recording_id="musan:long",
            audio_path=long_wav,
        ),
    ]
    manifest = _write_manifest(tmp_path / "recordings.jsonl", records)
    source = OnlineBackgroundSource(
        source_id="musan",
        manifest_path=manifest,
        records=tuple(sorted(records, key=lambda item: item.recording_id)),
        duration_seconds_min=1.0,
        duration_seconds_max=1.0,
        fbank_kwargs={"dither": 0.0},
        background_source_id=0,
    )
    n = 20_000
    rng = random.Random(11)
    counts = {"musan:long": 0, "musan:short": 0}
    for _ in range(n):
        sample = source.sample(rng=rng)
        counts[sample.recording_id] += 1
        assert sample.background_source_id == 0
        assert sample.crop_id is None
    for recording_id, count in counts.items():
        observed = count / n
        assert abs(observed - 0.5) <= _bin_tolerance(0.5, n), recording_id


def test_t05_shared_sampler_has_no_probability_gate():
    inner = _FakeSourceSampler("musan")
    sampler = MultiSourceBackgroundSampler([("musan", 1.0, inner)])
    assert not hasattr(sampler, "probability")
    rng = random.Random(0)
    for _ in range(64):
        sample = sampler.sample(rng=rng)
        assert sample.feat is not None
        assert sample.background_source_id == 0
    assert len(inner.draws) == 64


def test_t05_dataset_background_rate_is_not_multiplied_by_source_count(
    tmp_path, monkeypatch, mock_npy_loader
):
    _patch_online_fbank(monkeypatch)
    sources = []
    for source_id in ("dns", "fsd50k", "musan"):
        wav = _write_wav(
            tmp_path / source_id / "clip.wav",
            np.linspace(-0.3, 0.3, 16_000, dtype=np.float32),
        )
        manifest = _write_manifest(
            tmp_path / source_id / "recordings.jsonl",
            [
                _record(
                    dataset_id=source_id,
                    recording_id=f"{source_id}:clip",
                    audio_path=wav,
                )
            ],
        )
        sources.append(_source_config(source_id, manifest, weight=1.0))
    dataset = LibriPhraseTrainDataset(
        wav_dir="/data/segments",
        tokenizer=_FakeTokenizer(),
        df=_mock_dataframe(),
        sample_lens=1,
        seed=0,
        background_negative=_online_config(sources, probability=0.25),
        fbank_kwargs={"dither": 0.0},
    )
    n = 8_000
    n_background = 0
    n_non_background = 0
    for _ in range(n):
        sample = dataset[0]
        source_id = int(sample["background_source_id"])
        if sample["query_seq"].numel() == 0 and int(sample["label"]) == 0:
            n_background += 1
            assert source_id in {0, 1, 2}
        else:
            n_non_background += 1
            assert source_id == -1
    rate = n_background / n
    expected = 0.5 * 0.25
    assert abs(rate - expected) <= _bin_tolerance(expected, n)
    assert n_non_background == n - n_background


def test_t05_joint_kind_background_does_not_apply_second_gate(
    tmp_path, monkeypatch, mock_npy_loader
):
    _patch_online_fbank(monkeypatch)
    wav = _write_wav(
        tmp_path / "musan" / "clip.wav",
        np.linspace(-0.2, 0.2, 16_000, dtype=np.float32),
    )
    manifest = _write_manifest(
        tmp_path / "musan" / "recordings.jsonl",
        [
            _record(
                dataset_id="musan",
                recording_id="musan:clip",
                audio_path=wav,
            )
        ],
    )
    dataset = LibriPhraseTrainDataset(
        wav_dir="/data/segments",
        tokenizer=_FakeTokenizer(),
        df=_mock_dataframe(),
        sample_lens=1,
        seed=0,
        background_negative=_online_config(
            [_source_config("musan", manifest)],
            probability=0.0,
        ),
        fbank_kwargs={"dither": 0.0},
    )
    sample = dataset.sample_pair(0, "background", rng=random.Random(4))
    assert int(sample["label"]) == 0
    assert sample["query_seq"].numel() == 0
    assert int(sample["background_source_id"]) == 0
    assert sample["seq_label"].tolist() == [0] * sample["anchor_seq"].numel()


def test_t06_background_labels_and_non_background_id(
    tmp_path, monkeypatch, mock_npy_loader
):
    _patch_online_fbank(monkeypatch)
    wav = _write_wav(
        tmp_path / "dns" / "clip.wav",
        np.linspace(-0.4, 0.4, 16_000, dtype=np.float32),
    )
    manifest = _write_manifest(
        tmp_path / "dns" / "recordings.jsonl",
        [
            _record(
                dataset_id="dns",
                recording_id="dns:clip",
                audio_path=wav,
            )
        ],
    )
    dataset = LibriPhraseTrainDataset(
        wav_dir="/data/segments",
        tokenizer=_FakeTokenizer(),
        df=_mock_dataframe(),
        sample_lens=1,
        seed=0,
        background_negative=_online_config(
            [_source_config("dns", manifest)],
            probability=1.0,
        ),
        fbank_kwargs={"dither": 0.0},
    )
    background = dataset.sample_pair(0, "background", rng=random.Random(1))
    assert int(background["label"]) == 0
    assert background["query_seq"].numel() == 0
    assert background["seq_label"].tolist() == [0] * background["anchor_seq"].numel()
    assert int(background["background_source_id"]) == 0
    assert background["recording_id"] == "dns:clip"
    assert "source" not in background
    assert "domain_source" not in background

    positive = dataset.sample_pair(0, "positive", rng=random.Random(1))
    assert int(positive["label"]) == 1
    assert int(positive["background_source_id"]) == -1
    assert "source" not in positive
    assert "domain_source" not in positive

    mixed = {
        "anchor_seq": background["anchor_seq"],
        "query_seq": background["query_seq"],
        "feat": torch.zeros(2, 8),
        "label": background["label"],
        "seq_label": background["seq_label"],
        "background_source_id": background["background_source_id"],
        "recording_id": background["recording_id"],
        "source": 7,
        "domain_source": 3,
    }
    speech = {
        "anchor_seq": positive["anchor_seq"],
        "query_seq": positive["query_seq"],
        "feat": torch.ones(3, 8),
        "label": positive["label"],
        "seq_label": positive["seq_label"],
        "background_source_id": positive["background_source_id"],
        "source": 7,
        "domain_source": 3,
    }
    collated = train_collate_fn([mixed, speech])
    assert collated["source"].tolist() == [7, 7]
    assert collated["domain_source"].tolist() == [3, 3]
    assert collated["background_source_id"].tolist() == [0, -1]


def test_collate_defaults_missing_background_source_id_to_minus_one_but_keeps_zero():
    def _item(*, source_id: int | None, label: int) -> dict:
        item = {
            "anchor_seq": torch.tensor([1, 2], dtype=torch.long),
            "feat": torch.ones(2, 8),
            "label": torch.tensor(label, dtype=torch.long),
            "seq_label": torch.tensor([0, 0], dtype=torch.long),
            "query_seq": torch.tensor([], dtype=torch.long),
        }
        if source_id is not None:
            item["background_source_id"] = source_id
        return item

    collated = train_collate_fn(
        [
            _item(source_id=0, label=0),
            _item(source_id=2, label=0),
            _item(source_id=None, label=1),
        ]
    )
    assert collated["background_source_id"].dtype == torch.long
    assert collated["background_source_id"].tolist() == [0, 2, -1]


def test_collate_keeps_recording_id_as_cpu_string_list():
    batch = [
        {
            "anchor_seq": torch.tensor([1], dtype=torch.long),
            "feat": torch.ones(2, 4),
            "label": torch.tensor(0, dtype=torch.long),
            "seq_label": torch.tensor([0], dtype=torch.long),
            "background_source_id": 1,
            "recording_id": "dns:clip",
            "crop_id": None,
        },
        {
            "anchor_seq": torch.tensor([1], dtype=torch.long),
            "feat": torch.ones(3, 4),
            "label": torch.tensor(1, dtype=torch.long),
            "seq_label": torch.tensor([1], dtype=torch.long),
        },
    ]
    collated = train_collate_fn(batch)
    assert collated["recording_id"] == ["dns:clip", ""]
    assert collated["crop_id"] == [None, None]
    assert not torch.is_tensor(collated["recording_id"])
    assert collated["background_source_id"].tolist() == [1, -1]


def test_zero_weight_source_is_not_opened(tmp_path, monkeypatch):
    _patch_online_fbank(monkeypatch)
    wav = _write_wav(
        tmp_path / "musan" / "clip.wav",
        np.linspace(-0.1, 0.1, 16_000, dtype=np.float32),
    )
    musan_manifest = _write_manifest(
        tmp_path / "musan" / "recordings.jsonl",
        [
            _record(
                dataset_id="musan",
                recording_id="musan:clip",
                audio_path=wav,
            )
        ],
    )
    sampler = build_background_sampler(
        _online_config(
            [
                _source_config(
                    "dns",
                    tmp_path / "missing" / "recordings.jsonl",
                    weight=0.0,
                ),
                _source_config("musan", musan_manifest, weight=1.0),
            ]
        ),
        fbank_kwargs={"dither": 0.0},
    )
    assert isinstance(sampler, MultiSourceBackgroundSampler)
    sample = sampler.sample(rng=random.Random(0))
    assert sample.background_source_id == 0
    assert sample.recording_id == "musan:clip"


def test_factory_rejects_no_eligible_train_recordings(tmp_path):
    wav = _write_wav(
        tmp_path / "musan" / "clip.wav",
        np.linspace(-0.1, 0.1, 16_000, dtype=np.float32),
    )
    manifest = _write_manifest(
        tmp_path / "musan" / "recordings.jsonl",
        [
            _record(
                dataset_id="musan",
                recording_id="musan:val",
                audio_path=wav,
                split="val",
            )
        ],
    )
    with pytest.raises(ValueError, match=r"no eligible train recordings for source musan"):
        build_background_sampler(
            _online_config([_source_config("musan", manifest)]),
            fbank_kwargs={"dither": 0.0},
        )


def test_dataset_enabled_nonempty_sources_uses_factory(
    tmp_path, monkeypatch, mock_npy_loader
):
    _patch_online_fbank(monkeypatch)

    class _MustNotConstruct:
        def __init__(self, **_kwargs):
            raise AssertionError("legacy TrainingBackgroundSampler must not be used")

    monkeypatch.setattr(
        "dma_kws.stage2.features.TrainingBackgroundSampler",
        _MustNotConstruct,
    )
    wav = _write_wav(
        tmp_path / "musan" / "clip.wav",
        np.linspace(-0.2, 0.2, 16_000, dtype=np.float32),
    )
    manifest = _write_manifest(
        tmp_path / "musan" / "recordings.jsonl",
        [
            _record(
                dataset_id="musan",
                recording_id="musan:clip",
                audio_path=wav,
            )
        ],
    )
    dataset = LibriPhraseTrainDataset(
        wav_dir="/data/segments",
        tokenizer=_FakeTokenizer(),
        df=_mock_dataframe(),
        background_negative=_online_config([_source_config("musan", manifest)]),
        fbank_kwargs={"dither": 0.0},
    )
    assert isinstance(dataset._background_sampler, MultiSourceBackgroundSampler)


def test_multi_source_close_closes_inners():
    inners = [
        _FakeSourceSampler("dns"),
        _FakeSourceSampler("musan"),
    ]
    sampler = MultiSourceBackgroundSampler(
        [("dns", 0.5, inners[0]), ("musan", 0.5, inners[1])]
    )
    sampler.close()
    assert all(inner.closed for inner in inners)
