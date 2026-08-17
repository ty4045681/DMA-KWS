import pytest

from dma_kws.config import compose_config, config_to_dict


_MUSAN_COMPONENTS = {
    "noise",
    "music",
    "speech",
    "stationary_noise",
    "burst_noise",
    "volume_variation",
}

_AUDIO_AUG_TRANSFORMS = {
    "volume_gain",
    "speed_change",
    "noise_mix",
    "subband_eq",
    "band_limit",
    "narrowband",
    "spectral_mask",
    "amp_distortion",
    "signal_mimic",
}

_CONDITIONS = [
    ("clean", set(), {}),
    (
        "stationary_snr10",
        {"stationary_noise"},
        {"stationary_noise": {"kind": "white_gaussian", "snr_db": 10.0}},
    ),
    (
        "stationary_snr20",
        {"stationary_noise"},
        {"stationary_noise": {"kind": "white_gaussian", "snr_db": 20.0}},
    ),
    (
        "volume_variation",
        {"volume_variation"},
        {
            "volume_variation": {
                "low_gain_db": -12.0,
                "high_gain_db": 6.0,
                "segment_ms_min": 250.0,
                "segment_ms_max": 750.0,
                "transition_ms": 50.0,
            }
        },
    ),
    (
        "burst_snr10",
        {"burst_noise"},
        {
            "burst_noise": {
                "snr_db": 10.0,
                "snr_scope": "active_event",
                "event_count_min": 1,
                "event_count_max": 1,
                "duration_ms_min": 100.0,
                "duration_ms_max": 400.0,
                "fade_ms": 10.0,
                "allow_overlap": False,
                "min_gap_ms": 50.0,
            }
        },
    ),
    (
        "burst_snr20",
        {"burst_noise"},
        {
            "burst_noise": {
                "snr_db": 20.0,
                "snr_scope": "active_event",
                "event_count_min": 1,
                "event_count_max": 1,
                "duration_ms_min": 100.0,
                "duration_ms_max": 400.0,
                "fade_ms": 10.0,
                "allow_overlap": False,
                "min_gap_ms": 50.0,
            }
        },
    ),
    (
        "musan_noise_snr10",
        {"noise"},
        {"noise": {"snr_db": 10.0}},
    ),
    (
        "musan_noise_snr20",
        {"noise"},
        {"noise": {"snr_db": 20.0}},
    ),
    (
        "musan_noise_snr10_music_snr10",
        {"noise", "music"},
        {"noise": {"snr_db": 10.0}, "music": {"snr_db": 10.0}},
    ),
    (
        "musan_noise_snr20_music_snr20",
        {"noise", "music"},
        {"noise": {"snr_db": 20.0}, "music": {"snr_db": 20.0}},
    ),
    (
        "musan_noise_snr10_speech_equal",
        {"noise", "speech"},
        {"noise": {"snr_db": 10.0}, "speech": {"relative_db": 0.0}},
    ),
    (
        "musan_noise_snr20_speech_equal",
        {"noise", "speech"},
        {"noise": {"snr_db": 20.0}, "speech": {"relative_db": 0.0}},
    ),
    (
        "musan_speech_quieter",
        {"speech"},
        {"speech": {"relative_db": -6.0}},
    ),
    (
        "musan_speech_equal",
        {"speech"},
        {"speech": {"relative_db": 0.0}},
    ),
    (
        "musan_speech_louder",
        {"speech"},
        {"speech": {"relative_db": 6.0}},
    ),
]


@pytest.mark.parametrize(
    ("condition", "enabled_components", "expected_values"),
    _CONDITIONS,
    ids=[condition for condition, _, _ in _CONDITIONS],
)
def test_eval_condition_composes_isolated_deterministic_recipe(
    condition,
    enabled_components,
    expected_values,
):
    config = config_to_dict(
        compose_config(overrides=[f"+eval_condition={condition}"])
    )
    musan_mix = config["prep"]["musan_mix"]
    audio_aug = config["prep"]["audio_aug"]

    assert musan_mix["seed"] == 2025
    assert audio_aug["seed"] == 2025
    assert set(musan_mix) >= {"seed", *_MUSAN_COMPONENTS}
    assert {
        name
        for name in _MUSAN_COMPONENTS
        if musan_mix[name]["enabled"]
    } == enabled_components

    assert set(audio_aug["transforms"]) == _AUDIO_AUG_TRANSFORMS
    assert all(
        transform["enabled"] is False
        for transform in audio_aug["transforms"].values()
    )

    for component, values in expected_values.items():
        for key, expected in values.items():
            assert musan_mix[component][key] == expected


def test_eval_condition_composes_alongside_model_experiment():
    config = config_to_dict(
        compose_config(
            "icefall_zipformer_stage2",
            overrides=["+eval_condition=musan_noise_snr10_speech_equal"],
        )
    )

    assert config["stage1"]["encoder_type"] == "icefall_zipformer"
    assert config["stage2"]["freeze_encoder"] is True
    assert config["prep"]["musan_mix"]["noise"] == {
        "enabled": True,
        "snr_db": 10.0,
    }
    assert config["prep"]["musan_mix"]["speech"] == {
        "enabled": True,
        "relative_db": 0.0,
    }
