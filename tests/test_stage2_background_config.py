import pandas as pd
import pytest

from dma_kws.config import compose_config, config_to_dict, load_config
from dma_kws.configs.schema import (
    Stage2BackgroundNegativeConfig,
    Stage2MetadataCacheConfig,
)
from dma_kws.stage2.dataset import LibriPhraseTrainDataset
from dma_kws.stage2.readout import resolve_qbyt_score_spec
from dma_kws.stage2.readout_pooling import QbyTReadoutConfig


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


def _make_dataset(*, background_negative, fbank_kwargs=None):
    return LibriPhraseTrainDataset(
        wav_dir="/data/segments",
        tokenizer=_FakeTokenizer(),
        df=_mock_dataframe(),
        background_negative=background_negative,
        fbank_kwargs=fbank_kwargs,
    )


class _MustNotConstructSampler:
    def __init__(self, **_kwargs):
        raise AssertionError("TrainingBackgroundSampler must not be constructed")


def test_default_compose_uses_online_background_and_metadata_cache_caps():
    stage2 = config_to_dict(compose_config())["stage2"]
    background = stage2["background_negative"]
    metadata = stage2["metadata_cache"]

    assert background["mode"] == "online"
    assert background["cache_manifest"] == ""
    assert background["max_open_shards"] == 8
    assert metadata["max_entries"] == 128
    assert metadata["max_bytes"] == 33554432


def test_old_yaml_without_new_fields_still_fills_schema_defaults(tmp_path):
    config_file = tmp_path / "legacy.yaml"
    config_file.write_text(
        "stage2:\n"
        "  background_negative:\n"
        "    enabled: false\n"
        "    probability: 0.25\n"
        "    audio_list_path: \"\"\n",
        encoding="utf-8",
    )

    config = load_config(config_file)
    background = config["stage2"]["background_negative"]
    metadata = config["stage2"]["metadata_cache"]

    assert background["mode"] == "online"
    assert background["cache_manifest"] == ""
    assert background["max_open_shards"] == 8
    assert metadata["max_entries"] == 128
    assert metadata["max_bytes"] == 33554432


def test_v41_experiment_stays_online_with_train_background_list():
    stage2 = config_to_dict(
        compose_config("icefall_zipformer_stage2_eps_softmin_v41")
    )["stage2"]
    background = stage2["background_negative"]

    assert background["mode"] == "online"
    assert background["audio_list_path"].endswith("train_background.list")
    assert background["audio_list_path"]


def test_cached_overlay_flips_data_source_and_keeps_v41_readout():
    v41 = config_to_dict(compose_config("icefall_zipformer_stage2_eps_softmin_v41"))
    cached = config_to_dict(
        compose_config("icefall_zipformer_stage2_eps_softmin_v41_cached")
    )
    v41_stage2 = v41["stage2"]
    cached_stage2 = cached["stage2"]
    background = cached_stage2["background_negative"]
    score = resolve_qbyt_score_spec(cached_stage2)

    assert background["mode"] == "fbank_cache"
    assert background["audio_list_path"] == ""
    assert background["cache_manifest"].endswith(
        "stage2_background_cache/manifest.json"
    )
    assert score.version == 4
    assert score.value == QbyTReadoutConfig(
        mode="eps_softmin",
        temperature=1.0,
        sink_token=True,
        text_position="learned",
        audio_position="relative_bias",
    )
    assert cached_stage2["run_name"] == (
        "icefall-zipformer-frozen-eps-softmin-v41-cached"
    )
    assert cached_stage2["run_name"] != v41_stage2["run_name"]
    assert cached_stage2["checkpoint_dir"] != v41_stage2["checkpoint_dir"]
    assert cached_stage2["log_dir"] != v41_stage2["log_dir"]
    assert cached_stage2["checkpoint_dir"].endswith(
        "icefall-zipformer-frozen-eps-softmin-v41-cached"
    )
    assert cached_stage2["log_dir"].endswith(
        "icefall-zipformer-frozen-eps-softmin-v41-cached"
    )
    assert cached_stage2["qbyt_readout_version"] == 4
    assert cached_stage2["negative_tail_loss"] == v41_stage2["negative_tail_loss"]


def test_dataset_accepts_new_background_keys_when_disabled(monkeypatch):
    monkeypatch.setattr(
        "dma_kws.stage2.features.TrainingBackgroundSampler",
        _MustNotConstructSampler,
    )
    dataset = _make_dataset(
        background_negative={
            "enabled": False,
            "mode": "fbank_cache",
            "cache_manifest": "/unused/manifest.json",
            "max_open_shards": 4,
            "audio_list_path": "",
        }
    )

    assert dataset._background_sampler is None
    assert dataset._background_probability == 0.0


def test_dataset_omitted_mode_defaults_to_online_and_constructs_sampler(monkeypatch):
    constructed = {}

    class _FakeSampler:
        def __init__(self, **kwargs):
            constructed.update(kwargs)

    monkeypatch.setattr(
        "dma_kws.stage2.features.TrainingBackgroundSampler",
        _FakeSampler,
    )
    dataset = _make_dataset(
        background_negative={
            "enabled": True,
            "probability": 0.25,
            "audio_list_path": "/background/musan.list",
            "duration_seconds_min": 1.0,
            "duration_seconds_max": 3.0,
        },
        fbank_kwargs={"dither": 0.0},
    )

    assert dataset._background_sampler is not None
    assert constructed == {
        "audio_list_path": "/background/musan.list",
        "duration_seconds_min": 1.0,
        "duration_seconds_max": 3.0,
        "fbank_kwargs": {"dither": 0.0},
    }


def test_dataset_online_mode_still_requires_audio_list_path(monkeypatch):
    monkeypatch.setattr(
        "dma_kws.stage2.features.TrainingBackgroundSampler",
        _MustNotConstructSampler,
    )
    with pytest.raises(
        ValueError,
        match="stage2.background_negative.audio_list_path is required when enabled",
    ):
        _make_dataset(
            background_negative={
                "enabled": True,
                "mode": "online",
                "audio_list_path": "",
            }
        )


def test_dataset_fbank_cache_requires_cache_manifest(monkeypatch):
    monkeypatch.setattr(
        "dma_kws.stage2.features.TrainingBackgroundSampler",
        _MustNotConstructSampler,
    )
    with pytest.raises(
        ValueError,
        match="stage2.background_negative.cache_manifest is required",
    ):
        _make_dataset(
            background_negative={
                "enabled": True,
                "mode": "fbank_cache",
                "audio_list_path": "",
                "cache_manifest": "",
            }
        )


def test_dataset_fbank_cache_raises_not_wired_without_constructing_sampler(
    monkeypatch,
):
    monkeypatch.setattr(
        "dma_kws.stage2.features.TrainingBackgroundSampler",
        _MustNotConstructSampler,
    )
    with pytest.raises(ValueError, match="not wired"):
        _make_dataset(
            background_negative={
                "enabled": True,
                "mode": "fbank_cache",
                "audio_list_path": "",
                "cache_manifest": "/tmp/stage2_background_cache/manifest.json",
                "max_open_shards": 8,
            }
        )


def test_dataset_fbank_cache_does_not_load_audio_list_even_when_set(monkeypatch):
    monkeypatch.setattr(
        "dma_kws.stage2.features.TrainingBackgroundSampler",
        _MustNotConstructSampler,
    )
    with pytest.raises(ValueError, match="fbank_cache"):
        _make_dataset(
            background_negative={
                "enabled": True,
                "mode": "fbank_cache",
                "audio_list_path": "/missing/train_background.list",
                "cache_manifest": "manifest.json",
            }
        )


def test_dataset_fbank_cache_rejects_nonzero_explicit_dither(monkeypatch):
    monkeypatch.setattr(
        "dma_kws.stage2.features.TrainingBackgroundSampler",
        _MustNotConstructSampler,
    )
    with pytest.raises(ValueError, match="dither"):
        _make_dataset(
            background_negative={
                "enabled": True,
                "mode": "fbank_cache",
                "cache_manifest": "manifest.json",
            },
            fbank_kwargs={"dither": 0.1},
        )


def test_dataset_rejects_unknown_mode(monkeypatch):
    monkeypatch.setattr(
        "dma_kws.stage2.features.TrainingBackgroundSampler",
        _MustNotConstructSampler,
    )
    with pytest.raises(ValueError, match=r"stage2\.background_negative\.mode"):
        _make_dataset(
            background_negative={
                "enabled": False,
                "mode": "disk",
            }
        )


@pytest.mark.parametrize("max_open_shards", [True, 0, -1])
def test_dataset_rejects_invalid_max_open_shards(max_open_shards):
    with pytest.raises(
        ValueError,
        match=r"stage2\.background_negative\.max_open_shards must be a positive int",
    ):
        _make_dataset(
            background_negative={
                "enabled": False,
                "max_open_shards": max_open_shards,
            }
        )


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"max_entries": True}, r"stage2\.metadata_cache\.max_entries"),
        ({"max_entries": -1}, r"stage2\.metadata_cache\.max_entries"),
        ({"max_bytes": True}, r"stage2\.metadata_cache\.max_bytes"),
        ({"max_bytes": 0}, r"stage2\.metadata_cache\.max_bytes"),
    ],
)
def test_metadata_cache_schema_rejects_invalid_capacities(kwargs, match):
    with pytest.raises(ValueError, match=match):
        Stage2MetadataCacheConfig(**kwargs)


def test_background_negative_schema_rejects_bool_and_zero_max_open_shards():
    with pytest.raises(
        ValueError,
        match=r"stage2\.background_negative\.max_open_shards",
    ):
        Stage2BackgroundNegativeConfig(max_open_shards=True)
    with pytest.raises(
        ValueError,
        match=r"stage2\.background_negative\.max_open_shards",
    ):
        Stage2BackgroundNegativeConfig(max_open_shards=0)


def test_background_negative_schema_rejects_unknown_mode():
    with pytest.raises(ValueError, match=r"stage2\.background_negative\.mode"):
        Stage2BackgroundNegativeConfig(mode="disk")


def test_dataset_still_rejects_unknown_background_field():
    with pytest.raises(ValueError, match="Unknown stage2.background_negative fields"):
        _make_dataset(background_negative={"enabled": False, "typo": True})
