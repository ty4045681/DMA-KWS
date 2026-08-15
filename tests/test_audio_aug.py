from __future__ import annotations

import math
import pickle

import numpy as np
import pytest

from dma_kws.inference import audio_aug as audio_aug_module
from dma_kws.inference.audio_aug import (
    ADDITIVE_METHODS,
    POST_MIX_METHODS,
    PRE_MIX_METHODS,
    SUPPORTED_METHODS,
    AudioAugWaveformTransform,
)


torch = pytest.importorskip("torch")


def _prep(
    transforms: dict[str, dict[str, object]] | None = None,
    **options: object,
) -> dict[str, object]:
    return {
        "audio_aug": {
            **options,
            "transforms": transforms or {},
        }
    }


def _transform(
    transforms: dict[str, dict[str, object]] | None = None,
    *,
    audio_paths: tuple[str, ...] = ("clean/a.wav",),
    **options: object,
) -> AudioAugWaveformTransform:
    return AudioAugWaveformTransform.from_prep(
        _prep(transforms, **options),
        audio_paths=audio_paths,
    )


def _sine(
    *,
    frames: int = 4096,
    sample_rate: int = 16000,
    frequency: float = 440.0,
    amplitude: float = 0.2,
):
    time = torch.arange(frames, dtype=torch.float64) / sample_rate
    return (amplitude * torch.sin(2.0 * math.pi * frequency * time)).unsqueeze(0)


def _stage(metadata: dict[str, object], name: str) -> dict[str, object]:
    stages = metadata["stages"]
    assert isinstance(stages, list)
    return next(item for item in stages if item["name"] == name)


def test_noop_is_lazy_and_preserves_object_identity(monkeypatch):
    monkeypatch.setattr(
        audio_aug_module,
        "_require_scipy",
        lambda: (_ for _ in ()).throw(AssertionError("scipy must remain lazy")),
    )
    monkeypatch.setattr(audio_aug_module, "_SCIPY_VERSION", None)
    transform = AudioAugWaveformTransform.from_prep(
        {},
        audio_paths=["clean.wav"],
    )
    sentinel = object()

    assert transform.enabled is False
    assert transform.changes_duration is False
    assert transform(0, sentinel, 16000) is sentinel
    assert transform.apply_pre_mix(0, sentinel, 16000) is sentinel
    assert transform.apply_post_mix(0, sentinel, 16000) is sentinel
    assert transform.recipe_metadata(0)["stages"] == []
    summary = transform.summary()
    assert summary["enabled"] is False
    assert summary["seed"] == 2025
    assert summary["speed_length_policy"] == "variable"
    assert summary["pcm_policy"] == "clip_round_each_stage"
    assert summary["scipy_version"] is None
    assert summary["numpy_version"] == np.__version__
    assert set(summary["transforms"]) == set(SUPPORTED_METHODS)
    assert all(
        section["enabled"] is False
        for section in summary["transforms"].values()
    )


def test_disabled_additive_phase_returns_same_shape_zero_delta():
    transform = _transform(
        {"volume_gain": {"enabled": True, "gain_db": 3.0}},
        pcm_policy="float_unclipped",
    )
    waveform = _sine(frames=37).to(torch.float32)

    delta = transform.apply_additive_delta(0, waveform, 16000)

    assert delta.shape == waveform.shape
    assert delta.dtype == waveform.dtype
    assert delta.device == waveform.device
    assert torch.count_nonzero(delta).item() == 0


@pytest.mark.parametrize(
    ("audio_aug", "error_type", "match"),
    [
        ({"unknown": 1}, ValueError, "unsupported keys"),
        ({"seed": True}, TypeError, "non-negative integer"),
        ({"seed": -1}, ValueError, "non-negative integer"),
        ({"speed_length_policy": "head_crop"}, ValueError, "speed_length_policy"),
        ({"pcm_policy": "normalize"}, ValueError, "pcm_policy"),
        ({"allow_signal_mimic_overlap": 1}, TypeError, "true or false"),
        (
            {"transforms": {"unknown": {"enabled": True}}},
            ValueError,
            "unsupported keys",
        ),
        (
            {"transforms": {"volume_gain": {"enabled": "yes"}}},
            TypeError,
            "enabled must be true or false",
        ),
        (
            {"transforms": {"volume_gain": {"enabled": True, "gain_db": 0}}},
            ValueError,
            "must not be 0",
        ),
        (
            {
                "transforms": {
                    "speed_change": {"enabled": True, "speed_factor": 1.0}
                }
            },
            ValueError,
            "must not be 1",
        ),
        (
            {"transforms": {"noise_mix": {"enabled": True, "snr_db": "nan"}}},
            ValueError,
            "finite number",
        ),
        (
            {
                "transforms": {
                    "noise_mix": {"enabled": True, "snr_mode": "nominal"}
                }
            },
            ValueError,
            "snr_mode",
        ),
        (
            {
                "transforms": {
                    "subband_eq": {
                        "enabled": True,
                        "low_min_gain_db": -8,
                        "high_min_gain_db": -2,
                    }
                }
            },
            ValueError,
            "must be <=",
        ),
        (
            {"transforms": {"band_limit": {"enabled": True, "mode": "fft"}}},
            ValueError,
            "freq, iir, or resample",
        ),
        (
            {
                "transforms": {
                    "spectral_mask": {
                        "enabled": True,
                        "min_gain": 0.9,
                        "max_gain": 0.2,
                    }
                }
            },
            ValueError,
            "max_gain",
        ),
        (
            {
                "transforms": {
                    "amp_distortion": {
                        "enabled": True,
                        "distortion_type": "unknown",
                    }
                }
            },
            ValueError,
            "distortion_type",
        ),
        (
            {
                "transforms": {
                    "signal_mimic": {
                        "enabled": True,
                        "mute_probability": 1.1,
                    }
                }
            },
            ValueError,
            "mute_probability",
        ),
    ],
)
def test_config_validation_is_strict(audio_aug, error_type, match):
    with pytest.raises(error_type, match=match):
        AudioAugWaveformTransform.from_prep(
            {"audio_aug": audio_aug},
            audio_paths=["clean.wav"],
        )


def test_scipy_is_required_only_for_enabled_scipy_transform(monkeypatch):
    def unavailable():
        raise ImportError("scipy test sentinel")

    monkeypatch.setattr(audio_aug_module, "_require_scipy", unavailable)
    volume = _transform(
        {"volume_gain": {"enabled": True, "gain_db": 2.0}},
        pcm_policy="float_unclipped",
    )
    assert volume.enabled

    with pytest.raises(ImportError, match="scipy test sentinel"):
        _transform({"subband_eq": {"enabled": True}})


def test_signal_mimic_overlap_requires_explicit_opt_in():
    config = {
        "signal_mimic": {"enabled": True},
        "subband_eq": {"enabled": True},
    }
    with pytest.raises(ValueError, match="allow_signal_mimic_overlap=true"):
        _transform(config)

    transform = _transform(config, allow_signal_mimic_overlap=True)
    assert transform.enabled
    assert transform.summary()["allow_signal_mimic_overlap"] is True


def test_recipe_seeds_are_stable_method_local_and_pickle_safe():
    transforms = {
        "noise_mix": {"enabled": True, "snr_db": 17.0},
        "amp_distortion": {
            "enabled": True,
            "distortion_type": "gain_db",
            "rate": 0.4,
            "gain_db": 2.0,
        },
    }
    paths = ("clean/a.wav", "clean/b.wav")
    first = _transform(
        transforms,
        audio_paths=paths,
        seed=91,
        pcm_policy="float_unclipped",
    )
    rebuilt = _transform(
        transforms,
        audio_paths=paths,
        seed=91,
        pcm_policy="float_unclipped",
    )
    restored = pickle.loads(pickle.dumps(first))
    noise_only = _transform(
        {"noise_mix": transforms["noise_mix"]},
        audio_paths=paths,
        seed=91,
        pcm_policy="float_unclipped",
    )

    assert first.recipe_metadata(0) == rebuilt.recipe_metadata(0)
    assert first.recipe_metadata(1) == restored.recipe_metadata(1)
    seed_a = _stage(first.recipe_metadata(0), "noise_mix")["recipe_seed"]
    seed_b = _stage(first.recipe_metadata(1), "noise_mix")["recipe_seed"]
    assert seed_a != seed_b
    assert seed_a == _stage(noise_only.recipe_metadata(0), "noise_mix")[
        "recipe_seed"
    ]

    waveform = _sine()
    assert torch.equal(first(0, waveform, 16000), rebuilt(0, waveform, 16000))
    assert torch.equal(first(0, waveform, 16000), restored(0, waveform, 16000))


def test_recipe_seed_changes_with_global_seed_and_audio_path():
    config = {"noise_mix": {"enabled": True}}
    base = _transform(config, seed=1)
    other_seed = _transform(config, seed=2)
    other_path = _transform(config, seed=1, audio_paths=("other.wav",))

    def recipe_seed(transform):
        return _stage(transform.recipe_metadata(0), "noise_mix")["recipe_seed"]

    assert len({recipe_seed(base), recipe_seed(other_seed), recipe_seed(other_path)}) == 3


def test_summary_and_recipe_expose_fixed_phase_order_and_resolved_params():
    transforms = {
        "volume_gain": {"enabled": True, "gain_db": 4.0},
        "speed_change": {"enabled": True, "speed_factor": 1.25},
        "noise_mix": {"enabled": True, "snr_db": 13.0},
        "band_limit": {
            "enabled": True,
            "mode": "freq",
            "cutoff_hz": 3000.0,
        },
        "amp_distortion": {
            "enabled": True,
            "distortion_type": "gain_db",
        },
    }
    transform = _transform(
        transforms,
        seed=7,
        pcm_policy="float_unclipped",
    )
    summary = transform.summary()
    metadata = transform.recipe_metadata(0)

    assert summary["pre_mix_order"] == ["speed_change", "volume_gain"]
    assert summary["additive_order"] == ["noise_mix"]
    assert summary["post_mix_order"] == ["amp_distortion", "band_limit"]
    assert [stage["name"] for stage in metadata["stages"]] == [
        "speed_change",
        "volume_gain",
        "noise_mix",
        "amp_distortion",
        "band_limit",
    ]
    noise = _stage(metadata, "noise_mix")
    assert noise["phase"] == "additive"
    assert noise["params"]["snr_db"] == 13.0
    assert noise["params"]["seed"] == noise["recipe_seed"]
    assert noise["requested_snr_db"] == 13.0


def test_volume_gain_float_math_and_pcm_policy():
    waveform = torch.tensor([[-0.75, -0.25, 0.25, 0.75]], dtype=torch.float64)
    floating = _transform(
        {"volume_gain": {"enabled": True, "gain_db": 6.0}},
        pcm_policy="float_unclipped",
    )
    pcm = _transform(
        {"volume_gain": {"enabled": True, "gain_db": 6.0}},
        pcm_policy="clip_round_each_stage",
    )

    floating_output = floating(0, waveform, 16000)
    pcm_output = pcm(0, waveform, 16000)

    assert torch.allclose(
        floating_output,
        waveform * (10.0 ** (6.0 / 20.0)),
        atol=1e-12,
        rtol=1e-12,
    )
    assert floating_output.abs().max().item() > 1.0
    assert pcm_output.max().item() <= 32767.0 / 32768.0
    assert pcm_output.min().item() >= -1.0
    assert torch.allclose(pcm_output * 32768.0, torch.round(pcm_output * 32768.0))


def test_volume_gain_matches_upstream_for_real_stage2_decoded_pcm():
    pcm = np.array(
        [-32768, -30000, -10001, -1, 0, 1, 10001, 25000, 32767],
        dtype=np.float64,
    )
    waveform = torch.from_numpy((pcm / 32768.0).reshape(1, -1))
    gain_db = 3.0
    gain = 10.0 ** (gain_db / 20.0)
    expected_pcm = np.clip(np.rint(pcm * gain), -32768.0, 32767.0)
    expected = torch.from_numpy((expected_pcm / 32768.0).reshape(1, -1))
    transform = _transform(
        {"volume_gain": {"enabled": True, "gain_db": gain_db}},
        pcm_policy="clip_round_each_stage",
    )

    output = transform(0, waveform, 16000)

    assert torch.equal(output, expected)


def test_pcm_scale_conversion_is_repeated_between_volume_and_amp_stages():
    pcm = np.array(
        [-32768, -28000, -12001, -123, 0, 123, 12001, 28000, 32767],
        dtype=np.float64,
    )
    waveform = torch.from_numpy((pcm / 32768.0).reshape(1, -1))
    volume_gain_db = 3.0
    amp_gain_db = -3.0

    after_volume = np.clip(
        np.rint(pcm * (10.0 ** (volume_gain_db / 20.0))),
        -32768.0,
        32767.0,
    )
    upstream_normalized = np.clip(after_volume / 32767.0, -1.0, 1.0)
    after_amp_normalized = np.clip(
        upstream_normalized * (10.0 ** (amp_gain_db / 20.0)),
        -0.997,
        0.997,
    )
    expected_pcm = np.clip(
        np.rint(after_amp_normalized * 32767.0),
        -32768.0,
        32767.0,
    )
    expected = torch.from_numpy((expected_pcm / 32768.0).reshape(1, -1))
    transform = _transform(
        {
            "volume_gain": {"enabled": True, "gain_db": volume_gain_db},
            "amp_distortion": {
                "enabled": True,
                "distortion_type": "gain_db",
                "rate": 1.0,
                "gain_db": amp_gain_db,
            },
        },
        pcm_policy="clip_round_each_stage",
    )

    output = transform(0, waveform, 16000)

    assert torch.equal(output, expected)


def test_noise_upstream_std_matches_pcm_write_parity_for_decoded_audio():
    pcm = np.resize(
        np.array([-20000, -7001, -1, 0, 1, 7001, 20000], dtype=np.float64),
        257,
    )
    waveform = torch.from_numpy((pcm / 32768.0).reshape(1, -1))
    snr_db = 12.0
    transform = _transform(
        {
            "noise_mix": {
                "enabled": True,
                "snr_db": snr_db,
                "snr_mode": "upstream_std",
            }
        },
        seed=29,
        pcm_policy="clip_round_each_stage",
    )
    stage = _stage(transform.recipe_metadata(0), "noise_mix")
    noise_std = float(np.sqrt(np.mean(np.square(pcm)))) / (
        10.0 ** (snr_db / 20.0)
    )
    noise = np.random.default_rng(stage["recipe_seed"]).normal(
        0.0,
        noise_std,
        size=(pcm.size, 1),
    )[:, 0]
    expected_pcm = np.clip(np.rint(pcm + noise), -32768.0, 32767.0)
    expected = torch.from_numpy((expected_pcm / 32768.0).reshape(1, -1))

    output = transform(0, waveform, 16000)

    assert torch.equal(output, expected)


def test_speed_variable_and_compatibility_length_policies():
    waveform = torch.arange(8, dtype=torch.float64).unsqueeze(0)
    variable = _transform(
        {"speed_change": {"enabled": True, "speed_factor": 2.0}},
        speed_length_policy="variable",
        pcm_policy="float_unclipped",
    )
    compatibility = _transform(
        {"speed_change": {"enabled": True, "speed_factor": 2.0}},
        speed_length_policy="center_crop_or_zero_pad",
        pcm_policy="float_unclipped",
    )

    changed = variable(0, waveform, 16000)
    restored = compatibility(0, waveform, 16000)

    assert variable.changes_duration is True
    assert compatibility.changes_duration is False
    assert changed.shape == (1, 4)
    assert torch.equal(changed, torch.tensor([[0.0, 2.0, 4.0, 6.0]]))
    assert restored.shape == waveform.shape
    assert torch.equal(
        restored,
        torch.tensor([[0.0, 0.0, 0.0, 2.0, 4.0, 6.0, 0.0, 0.0]]),
    )


@pytest.mark.parametrize("snr_db", [0.0, 10.0, 20.0, 30.0])
def test_noise_delta_has_exact_requested_snr(snr_db):
    transform = _transform(
        {"noise_mix": {"enabled": True, "snr_db": snr_db}},
        seed=3,
        pcm_policy="float_unclipped",
    )
    waveform = _sine(frames=8192)
    stage = _stage(transform.recipe_metadata(0), "noise_mix")

    delta = transform.apply_additive_delta(0, waveform, 16000)
    signal_rms = torch.sqrt(torch.mean(waveform.square())).item()
    noise_rms = torch.sqrt(torch.mean(delta.square())).item()
    achieved = 20.0 * math.log10(signal_rms / noise_rms)

    assert delta.shape == waveform.shape
    assert stage["params"]["snr_mode"] == "exact_rms"
    assert transform.summary()["transforms"]["noise_mix"]["snr_mode"] == "exact_rms"
    assert achieved == pytest.approx(snr_db, abs=1e-10)
    assert torch.allclose(transform(0, waveform, 16000), waveform + delta)


def test_noise_upstream_std_mode_matches_pinned_nominal_std_semantics():
    snr_db = 20.0
    transform = _transform(
        {
            "noise_mix": {
                "enabled": True,
                "snr_db": snr_db,
                "snr_mode": "upstream_std",
            }
        },
        seed=3,
        pcm_policy="float_unclipped",
    )
    waveform = _sine(frames=128)
    stage = _stage(transform.recipe_metadata(0), "noise_mix")
    recipe_seed = stage["recipe_seed"]
    signal_rms = torch.sqrt(torch.mean(waveform.square())).item()
    nominal_noise_rms = signal_rms / (10.0 ** (snr_db / 20.0))
    expected_noise = np.random.default_rng(recipe_seed).normal(
        0.0,
        nominal_noise_rms,
        size=(waveform.shape[1], 1),
    )

    delta = transform.apply_additive_delta(0, waveform, 16000)
    achieved = 20.0 * math.log10(
        signal_rms / torch.sqrt(torch.mean(delta.square())).item()
    )

    assert stage["params"]["snr_mode"] == "upstream_std"
    assert transform.summary()["transforms"]["noise_mix"]["snr_mode"] == "upstream_std"
    assert np.allclose(delta.numpy().T, expected_noise, atol=1e-16, rtol=1e-13)
    assert abs(achieved - snr_db) > 0.01


def test_pre_additive_post_api_matches_call_and_preserves_component_delta():
    transform = _transform(
        {
            "speed_change": {"enabled": True, "speed_factor": 1.25},
            "volume_gain": {"enabled": True, "gain_db": 2.0},
            "noise_mix": {"enabled": True, "snr_db": 15.0},
            "amp_distortion": {
                "enabled": True,
                "distortion_type": "gain_db",
                "rate": 1.0,
                "gain_db": -2.0,
            },
        },
        seed=11,
        pcm_policy="float_unclipped",
    )
    waveform = _sine(frames=2000)

    pre = transform.apply_pre_mix(0, waveform, 16000)
    delta = transform.apply_additive_delta(0, pre, 16000)
    post = transform.apply_post_mix(0, pre + delta, 16000)

    assert pre.shape == delta.shape
    assert post.shape == pre.shape
    assert torch.equal(transform(0, waveform, 16000), post)


def test_frequency_band_limit_suppresses_bins_above_cutoff():
    sample_rate = 16000
    frames = 16000
    low = _sine(
        frames=frames,
        sample_rate=sample_rate,
        frequency=500.0,
        amplitude=0.3,
    )
    high = _sine(
        frames=frames,
        sample_rate=sample_rate,
        frequency=6000.0,
        amplitude=0.3,
    )
    waveform = low + high
    transform = _transform(
        {
            "band_limit": {
                "enabled": True,
                "mode": "freq",
                "cutoff_hz": 2000.0,
            }
        },
        pcm_policy="float_unclipped",
    )

    output = transform(0, waveform, sample_rate)
    spectrum = torch.fft.rfft(output[0])
    frequencies = torch.fft.rfftfreq(frames, d=1.0 / sample_rate)
    low_magnitude = spectrum[torch.argmin((frequencies - 500.0).abs())].abs()
    high_magnitude = spectrum[torch.argmin((frequencies - 6000.0).abs())].abs()

    assert output.shape == waveform.shape
    assert low_magnitude.item() > 1000.0
    assert high_magnitude.item() < low_magnitude.item() * 1e-10


@pytest.mark.parametrize(
    "distortion_type",
    [
        "gain_db",
        "max_distortion",
        "fence_distortion",
        "jag_distortion",
        "poly_distortion",
        "quad_distortion",
    ],
)
def test_all_amplitude_distortion_modes_are_deterministic(distortion_type):
    transform = _transform(
        {
            "amp_distortion": {
                "enabled": True,
                "distortion_type": distortion_type,
                "rate": 1.0,
                "gain_db": 3.0,
                "max_db": -3.0,
                "mask_number": 3,
                "a": 0.5,
                "m": 2,
                "n": 2,
            }
        },
        seed=27,
        pcm_policy="float_unclipped",
    )
    waveform = torch.linspace(-0.9, 0.9, 2001, dtype=torch.float64).unsqueeze(0)

    first = transform(0, waveform, 16000)
    second = transform(0, waveform, 16000)

    assert first.shape == waveform.shape
    assert torch.isfinite(first).all()
    assert torch.equal(first, second)
    assert not torch.equal(first, waveform)


@pytest.mark.parametrize(
    ("method", "params"),
    [
        ("volume_gain", {"gain_db": -2.0}),
        ("speed_change", {"speed_factor": 1.1}),
        ("noise_mix", {"snr_db": 18.0}),
        ("subband_eq", {"low_min_gain_db": -3.0, "high_min_gain_db": -6.0}),
        ("band_limit", {"mode": "iir", "cutoff_hz": 3000.0}),
        ("narrowband", {"target_sample_rate": 8000}),
        (
            "spectral_mask",
            {
                "frequency_masks": 2,
                "time_masks": 1,
                "min_gain": 0.2,
                "max_gain": 0.8,
            },
        ),
        (
            "amp_distortion",
            {"distortion_type": "gain_db", "rate": 0.5, "gain_db": 2.0},
        ),
        (
            "signal_mimic",
            {
                "subband_probability": 0.0,
                "mute_probability": 1.0,
                "band_limit_probability": 0.0,
                "spectral_mask_probability": 0.0,
                "narrowband_probability": 0.0,
            },
        ),
    ],
)
def test_each_supported_transform_smoke(method, params):
    transform = _transform(
        {method: {"enabled": True, **params}},
        seed=13,
        pcm_policy="float_unclipped",
    )
    waveform = _sine(frames=4096)

    first = transform(0, waveform, 16000)
    second = transform(0, waveform, 16000)

    expected_frames = (
        round(waveform.shape[1] / float(params["speed_factor"]))
        if method == "speed_change"
        else waveform.shape[1]
    )
    assert first.shape == (1, expected_frames)
    assert torch.isfinite(first).all()
    assert torch.equal(first, second)
    assert not torch.equal(first, waveform)


def test_signal_mimic_recipe_records_params_and_seed_without_fixed_rate_plan():
    transform = _transform(
        {
            "signal_mimic": {
                "enabled": True,
                "subband_probability": 0.0,
                "mute_probability": 1.0,
                "band_limit_probability": 0.0,
                "spectral_mask_probability": 0.0,
                "narrowband_probability": 0.0,
            }
        },
        seed=101,
    )
    stage = _stage(transform.recipe_metadata(0), "signal_mimic")

    assert "planned_stages_at_16000_hz" not in stage
    assert stage["params"]["seed"] == stage["recipe_seed"]


@pytest.mark.parametrize(
    ("waveform", "sample_rate", "error_type", "match"),
    [
        (torch.ones(8), 16000, ValueError, "shape"),
        (torch.ones(2, 8), 16000, ValueError, "mono"),
        (torch.ones(1, 0), 16000, ValueError, "empty"),
        (torch.ones(1, 8, dtype=torch.int16), 16000, TypeError, "floating-point"),
        (torch.tensor([[0.0, float("nan")]]), 16000, ValueError, "non-finite"),
        (torch.ones(1, 8), 0, ValueError, "positive integer"),
        (torch.ones(1, 8), True, TypeError, "positive integer"),
    ],
)
def test_enabled_transform_rejects_invalid_waveforms(
    waveform,
    sample_rate,
    error_type,
    match,
):
    transform = _transform(
        {"volume_gain": {"enabled": True, "gain_db": 2.0}},
        pcm_policy="float_unclipped",
    )
    with pytest.raises(error_type, match=match):
        transform(0, waveform, sample_rate)


def test_recipe_index_validation():
    transform = _transform()
    with pytest.raises(TypeError, match="row index"):
        transform.recipe_metadata(True)
    with pytest.raises(IndexError, match="out of range"):
        transform.recipe_metadata(-1)
    with pytest.raises(IndexError, match="out of range"):
        transform.recipe_metadata(1)


def test_constants_cover_each_supported_method_exactly_once():
    combined = PRE_MIX_METHODS + ADDITIVE_METHODS + POST_MIX_METHODS
    assert combined == SUPPORTED_METHODS
    assert len(combined) == 9
    assert len(set(combined)) == 9


def test_subband_boundary_rng_matches_pinned_golden():
    boundaries = audio_aug_module._subband_boundaries(
        np.random.default_rng(12345),
        257,
    )

    assert boundaries == (0, 1, 3, 4, 13, 21, 29, 86, 138, 191, 257)


def test_subband_high_gain_floor_matches_pinned_three_highest_bands():
    boundaries = (0, 1, 3, 4, 13, 21, 29, 86, 138, 191, 257)
    params = {"low_min_gain_db": -3.0, "high_min_gain_db": -7.0}

    floors = [
        audio_aug_module._subband_gain_floor_db(index, boundaries, params)
        for index in range(len(boundaries) - 1)
    ]

    assert floors == [-3.0] * 7 + [-7.0] * 3


def test_amplitude_mask_rng_matches_pinned_golden():
    masks = audio_aug_module._amplitude_masks(
        np.random.default_rng(12345),
        4,
    )

    assert np.allclose(
        masks,
        (
            (1.0e-5, 3.921863107697936e-5),
            (0.0001699127816326277, 0.0012570795482706107),
            (0.008127087955805438, 0.03824916521344766),
            (0.16870169432501253, 1.0),
        ),
        atol=0.0,
        rtol=1e-15,
    )


def test_signal_mimic_single_rng_schedule_matches_pinned_golden():
    all_enabled = {
        "subband_probability": 1.0,
        "mute_probability": 1.0,
        "band_limit_probability": 1.0,
        "spectral_mask_probability": 1.0,
        "narrowband_probability": 1.0,
    }
    steps = audio_aug_module._signal_mimic_plan(
        all_enabled,
        recipe_seed=12345,
        sample_rate=16000,
        frame_count=1000,
    )

    assert [(step.name, step.values()) for step in steps] == [
        ("subband_eq", {"seed": 1693606510}),
        ("mute", {"start": 158, "length": 178}),
        ("band_limit", {"cutoff_hz": 3772}),
        ("spectral_mask", {"seed": 1218704077}),
        ("narrowband", {}),
    ]


def test_signal_mimic_skipped_children_do_not_consume_child_seeds():
    params = {
        "subband_probability": 0.6,
        "mute_probability": 0.2,
        "band_limit_probability": 0.5,
        "spectral_mask_probability": 0.4,
        "narrowband_probability": 0.5,
    }
    steps = audio_aug_module._signal_mimic_plan(
        params,
        recipe_seed=4,
        sample_rate=16000,
        frame_count=1000,
    )

    assert [(step.name, step.values()) for step in steps] == [
        ("spectral_mask", {"seed": 973933017}),
        ("narrowband", {}),
    ]


def test_spectral_frequency_width_rng_matches_exclusive_upstream_bound():
    width = audio_aug_module._spectral_frequency_mask_width(
        np.random.default_rng(12345),
        257,
    )

    assert width == 38
    assert width < 257 // 5


def test_iir_branch_uses_strict_order_times_six_length_threshold(monkeypatch):
    calls = []

    class FakeSignal:
        @staticmethod
        def butter(*_args, **_kwargs):
            return object()

        @staticmethod
        def sosfilt(_sos, channel):
            calls.append(("sosfilt", len(channel)))
            return channel

        @staticmethod
        def sosfiltfilt(_sos, channel):
            calls.append(("sosfiltfilt", len(channel)))
            return channel

    monkeypatch.setattr(audio_aug_module, "_require_scipy", lambda: FakeSignal())
    params = {
        "mode": "iir",
        "cutoff_hz": 3000.0,
        "filter_order": 4,
        "target_sample_rate": 8000,
    }

    audio_aug_module._band_limit(np.ones((24, 1)), 16000, params)
    audio_aug_module._band_limit(np.ones((25, 1)), 16000, params)

    assert calls == [("sosfilt", 24), ("sosfiltfilt", 25)]


def test_scipy_version_is_an_instance_snapshot_not_global_history(monkeypatch):
    monkeypatch.setattr(audio_aug_module, "_SCIPY_SIGNAL", object())
    monkeypatch.setattr(audio_aug_module, "_SCIPY_VERSION", "9.9.test")

    scipy_enabled = _transform({"subband_eq": {"enabled": True}})
    disabled = _transform()
    numpy_only = _transform(
        {"volume_gain": {"enabled": True, "gain_db": 2.0}}
    )

    assert scipy_enabled.summary()["scipy_version"] == "9.9.test"
    assert disabled.summary()["scipy_version"] is None
    assert numpy_only.summary()["scipy_version"] is None


def test_large_manifest_pickle_size_does_not_scale_with_enabled_stage_count():
    paths = tuple(f"audio/{index:06d}.wav" for index in range(10000))
    one_stage = _transform(
        {"noise_mix": {"enabled": True}},
        audio_paths=paths,
    )
    every_stage = _transform(
        {method: {"enabled": True} for method in SUPPORTED_METHODS},
        audio_paths=paths,
        allow_signal_mimic_overlap=True,
    )

    one_stage_size = len(pickle.dumps(one_stage))
    every_stage_size = len(pickle.dumps(every_stage))

    assert every_stage_size < 300_000
    assert every_stage_size - one_stage_size < 10_000
