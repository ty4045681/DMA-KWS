from __future__ import annotations

import math
from typing import Any

import pytest

from dma_kws.configs.schema import (
    Stage2BackgroundNegativeConfig,
    Stage2BackgroundSourceConfig,
    Stage2BackgroundValidationConfig,
    active_background_sources,
    background_source_batch_index,
    normalize_source_weights,
    validate_background_negative_config,
)


def _source(
    source_id: str = "musan",
    *,
    weight: Any = 1.0,
    manifest: str = "/data/musan/recordings.jsonl",
    cache_manifest: str = "",
) -> dict[str, Any]:
    return {
        "id": source_id,
        "weight": weight,
        "manifest": manifest,
        "cache_manifest": cache_manifest,
    }


def _payload(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "enabled": False,
        "probability": 0.25,
        "audio_list_path": "",
        "duration_seconds_min": 1.0,
        "duration_seconds_max": 3.0,
        "mode": "online",
        "cache_manifest": "",
        "max_open_shards": 8,
        "sources": [_source()],
        "validation": {
            "enabled": False,
            "samples_per_source": 256,
            "seed": 2025,
        },
    }
    payload.update(overrides)
    return payload


def _assert_schema_and_helper_raise(payload: dict[str, Any], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        validate_background_negative_config(payload)
    with pytest.raises(ValueError, match=match):
        Stage2BackgroundNegativeConfig(**payload)


def test_schema_defaults_empty_sources_and_validation_off():
    cfg = Stage2BackgroundNegativeConfig()
    assert cfg.sources == []
    assert cfg.validation.enabled is False
    assert cfg.validation.samples_per_source == 256
    assert cfg.validation.seed == 2025


@pytest.mark.parametrize("weight", [True, False, float("nan"), float("inf"), -0.1])
def test_bool_nan_inf_negative_weight_rejected(weight):
    _assert_schema_and_helper_raise(
        _payload(sources=[_source(weight=weight)]),
        r"stage2\.background_negative\.sources\[\]\.weight",
    )


def test_duplicate_source_id_rejected():
    _assert_schema_and_helper_raise(
        _payload(sources=[_source("musan"), _source("musan", manifest="/other.jsonl")]),
        "duplicate",
    )


def test_nonempty_sources_reject_top_level_audio_list_path():
    _assert_schema_and_helper_raise(
        _payload(audio_list_path="/legacy/train_background.list"),
        r"audio_list_path.*sources is non-empty",
    )


def test_nonempty_sources_reject_top_level_cache_manifest():
    _assert_schema_and_helper_raise(
        _payload(
            mode="fbank_cache",
            cache_manifest="/legacy/cache/manifest.json",
            sources=[
                _source(
                    cache_manifest="/data/musan/cache/manifest.json",
                )
            ],
        ),
        r"cache_manifest.*sources is non-empty",
    )


def test_enabled_false_does_not_open_or_stat_paths(monkeypatch):
    def _forbid_open(*_args, **_kwargs):
        raise AssertionError("validators must not open files when enabled=false")

    def _forbid_exists(self):
        raise AssertionError("validators must not stat paths when enabled=false")

    monkeypatch.setattr("builtins.open", _forbid_open)
    monkeypatch.setattr("pathlib.Path.exists", _forbid_exists)
    monkeypatch.setattr("pathlib.Path.is_file", _forbid_exists)
    monkeypatch.setattr("pathlib.Path.stat", _forbid_exists)

    payload = _payload(
        enabled=False,
        mode="fbank_cache",
        sources=[
            _source(
                manifest="/missing/recordings.jsonl",
                cache_manifest="/missing/cache/manifest.json",
            )
        ],
    )
    validate_background_negative_config(payload)
    Stage2BackgroundNegativeConfig(**payload)


def test_zero_weight_only_sources_rejected():
    _assert_schema_and_helper_raise(
        _payload(sources=[_source(weight=0.0, manifest="")]),
        r"at least one source with weight > 0",
    )


def test_mixed_zero_weight_kept_in_resolved_config_and_dropped_from_active():
    cfg = Stage2BackgroundNegativeConfig(
        **_payload(
            sources=[
                _source("dns", weight=0.0, manifest=""),
                _source("musan", weight=2.0),
                _source("fsd50k", weight=2.0, manifest="/data/fsd50k/recordings.jsonl"),
            ]
        )
    )
    assert [source.id for source in cfg.sources] == ["dns", "musan", "fsd50k"]
    assert cfg.sources[0].weight == 0.0
    active = active_background_sources(cfg.sources)
    assert [source.id for source in active] == ["fsd50k", "musan"]
    assert normalize_source_weights([source.weight for source in active]) == [0.5, 0.5]
    assert background_source_batch_index([source.id for source in active]) == {
        "fsd50k": 0,
        "musan": 1,
    }


def test_source_batch_index_is_sorted_active_ids():
    assert background_source_batch_index(["musan", "dns", "fsd50k"]) == {
        "dns": 0,
        "fsd50k": 1,
        "musan": 2,
    }


@pytest.mark.parametrize("source_id", ["MUSAN", "dns-v5", "dns.v5", "", " musan"])
def test_source_id_must_match_ascii_pattern(source_id):
    _assert_schema_and_helper_raise(
        _payload(sources=[_source(source_id)]),
        r"\[a-z0-9_\]\+",
    )


def test_dns_v5_source_id_is_accepted():
    cfg = Stage2BackgroundNegativeConfig(
        **_payload(sources=[_source("dns_v5", manifest="/data/dns/recordings.jsonl")])
    )
    assert cfg.sources[0].id == "dns_v5"


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"probability": 1.5}, r"probability must be between 0 and 1"),
        ({"probability": -0.01}, r"probability must be between 0 and 1"),
        ({"probability": float("nan")}, r"probability must be between 0 and 1"),
        ({"probability": float("inf")}, r"probability must be between 0 and 1"),
        ({"probability": True}, r"probability"),
        ({"duration_seconds_min": float("nan")}, r"duration bounds must be finite"),
        ({"duration_seconds_max": float("inf")}, r"duration bounds must be finite"),
        ({"duration_seconds_min": 0.0}, r"duration_seconds_min must be positive"),
        (
            {"duration_seconds_min": 3.0, "duration_seconds_max": 1.0},
            r"duration_seconds_min must be <= duration_seconds_max",
        ),
        ({"max_open_shards": True}, r"max_open_shards must be a positive int"),
        ({"max_open_shards": 0}, r"max_open_shards must be a positive int"),
        (
            {
                "validation": {
                    "enabled": False,
                    "samples_per_source": True,
                    "seed": 2025,
                }
            },
            r"samples_per_source must be a positive int",
        ),
        (
            {
                "validation": {
                    "enabled": False,
                    "samples_per_source": 0,
                    "seed": 2025,
                }
            },
            r"samples_per_source must be a positive int",
        ),
    ],
)
def test_schema_and_dict_helper_raise_consistently_for_invalid_scalars(overrides, match):
    _assert_schema_and_helper_raise(_payload(**overrides), match)


def test_online_active_source_requires_manifest_and_empty_cache():
    _assert_schema_and_helper_raise(
        _payload(mode="online", sources=[_source("musan", manifest="")]),
        r"source id='musan'.+field=manifest.+required when mode=online",
    )
    _assert_schema_and_helper_raise(
        _payload(
            mode="online",
            sources=[
                _source(
                    "musan",
                    cache_manifest="/data/musan/cache/manifest.json",
                )
            ],
        ),
        r"source id='musan'.+field=cache_manifest.+must be empty when mode=online",
    )


def test_fbank_cache_active_source_requires_manifest_and_cache():
    _assert_schema_and_helper_raise(
        _payload(
            mode="fbank_cache",
            sources=[
                _source(
                    "fsd50k",
                    manifest="",
                    cache_manifest="/cache/manifest.json",
                )
            ],
        ),
        r"source id='fsd50k'.+field=manifest.+required when mode=fbank_cache",
    )
    _assert_schema_and_helper_raise(
        _payload(
            mode="fbank_cache",
            sources=[_source("dns_v5", cache_manifest="")],
        ),
        r"source id='dns_v5'.+field=cache_manifest.+required when mode=fbank_cache",
    )


def test_validation_enabled_requires_active_source_manifest():
    _assert_schema_and_helper_raise(
        _payload(
            mode="online",
            sources=[
                _source("dns", weight=0.0, manifest=""),
                _source("musan", manifest=""),
            ],
            validation={"enabled": True, "samples_per_source": 256, "seed": 2025},
        ),
        r"source id='musan'.+field=manifest.+required when validation.enabled=true",
    )


def test_zero_weight_source_skips_mode_path_requirements():
    cfg = Stage2BackgroundNegativeConfig(
        **_payload(
            mode="fbank_cache",
            sources=[
                _source("dns", weight=0.0, manifest="", cache_manifest=""),
                _source(
                    "musan",
                    weight=1.0,
                    cache_manifest="/data/musan/cache/manifest.json",
                ),
            ],
        )
    )
    assert cfg.sources[0].manifest == ""
    assert [source.id for source in active_background_sources(cfg.sources)] == ["musan"]


def test_normalize_source_weights_is_proportional():
    weights = normalize_source_weights([0.4, 0.4, 0.2])
    assert weights == pytest.approx([0.4, 0.4, 0.2])
    assert normalize_source_weights([2.0, 2.0, 1.0]) == pytest.approx([0.4, 0.4, 0.2])
    assert math.isclose(sum(weights), 1.0)


def test_empty_sources_legacy_enabled_still_requires_top_level_paths():
    _assert_schema_and_helper_raise(
        _payload(enabled=True, sources=[], audio_list_path=""),
        r"audio_list_path is required when enabled",
    )
    _assert_schema_and_helper_raise(
        _payload(
            enabled=True,
            mode="fbank_cache",
            sources=[],
            cache_manifest="",
        ),
        r"cache_manifest is required",
    )


def test_source_and_validation_dataclasses_use_independent_defaults():
    first = Stage2BackgroundSourceConfig(id="musan", weight=1.0)
    second = Stage2BackgroundSourceConfig(id="dns", weight=0.5)
    first.manifest = "/mutated.jsonl"
    assert second.manifest == ""
    validation_a = Stage2BackgroundValidationConfig()
    validation_b = Stage2BackgroundValidationConfig()
    validation_a.enabled = True
    assert validation_b.enabled is False
