from pathlib import Path

import pytest

from dma_kws.config import (
    compose_config,
    config_to_dict,
    get_tokenizer_config,
    load_config,
    require_sections,
)
from dma_kws.configs.schema import AdaptSweepConfig
from dma_kws.stage2.readout import resolve_qbyt_score_spec

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_load_config_expands_user_and_env_paths(tmp_path, monkeypatch):
    monkeypatch.setenv("DMA_KWS_TEST_ROOT", str(tmp_path))
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        "paths:\n"
        "  data_root: ${DMA_KWS_TEST_ROOT}/data\n"
        "  cache_root: ~/dma-kws-cache\n"
        "stage1:\n"
        "  batch_size_per_gpu: 48\n",
        encoding="utf-8",
    )

    config = load_config(config_file)

    assert config["paths"]["data_root"] == str(tmp_path / "data")
    assert config["paths"]["cache_root"].startswith(str(Path.home()))
    assert config["stage1"]["batch_size_per_gpu"] == 48


def test_compose_config_expands_oc_env_paths(tmp_path, monkeypatch):
    monkeypatch.setenv("DMA_KWS_TEST_ROOT", str(tmp_path))
    cfg = compose_config(overrides=[f"paths.processed_root=${{oc.env:DMA_KWS_TEST_ROOT}}/processed"])
    config = config_to_dict(cfg)

    assert config["paths"]["processed_root"] == str(tmp_path / "processed")


def test_require_sections_reports_missing_sections():
    with pytest.raises(ValueError, match="Missing required config sections: stage2, demo"):
        require_sections({"paths": {}, "stage1": {}}, ["paths", "stage1", "stage2", "demo"])


def test_demo_config_loads_with_tokenizer_and_training_seed():
    config = config_to_dict(compose_config("demo_librispeech100"))

    tokenizer = get_tokenizer_config(config)
    assert tokenizer["dict_path"] == "data/dict/lang_char.txt"
    assert tokenizer["split_with_space"] == " "
    assert config["training"]["seed"] == 2025

    stage2 = config["stage2"]
    assert stage2["parquet_file"].endswith("aggregated_segments_with_g2p_distance.parquet")
    assert stage2["wav_dir"].endswith("features/fbank")
    assert stage2["negative_ratio"] == 1
    assert stage2["learning_rate"] == 0.0005
    assert stage2["warmup_steps"] == 2500
    assert stage2["sequence_loss"] == {
        "target_mode": "ordered_contiguous_prefix",
        "progress_weight": 0.3,
        "normalization": "sample",
    }
    assert stage2["qbyt_readout_version"] == 7
    assert stage2["qbyt_readout"] is None
    assert stage2["qbyt_alignment"] == {
        "min_phone_duration_frames": 1,
        "max_phone_duration_frames": 8,
        "max_inter_phone_gap_frames": 3,
        "max_keyword_span_frames": 30,
        "local_context_kernel": 5,
        "weakest_phone_temperature": 0.2,
        "weakest_phone_weight": 1.0,
        "topology": "keyword_filler_segmental_crf_v1",
        "temperature": None,
    }
    assert "allow_legacy_qbyt_readout" not in stage2
    assert stage2["checkpoint"]["monitor"] == "val_auc"
    assert stage2["checkpoint"]["mode"] == "max"


def test_alignment_experiment_inherits_the_single_stage2_alignment_config():
    config = config_to_dict(compose_config("icefall_zipformer_stage2_alignment"))

    assert config["training"]["recipe"] == "icefall-zipformer-frozen-segmental-crf-v6"
    assert config["stage2"]["qbyt_readout_version"] == 7
    assert config["stage2"]["qbyt_readout"] is None
    assert config["stage2"]["qbyt_alignment"] == {
        "min_phone_duration_frames": 1,
        "max_phone_duration_frames": 8,
        "max_inter_phone_gap_frames": 3,
        "max_keyword_span_frames": 30,
        "local_context_kernel": 5,
        "weakest_phone_temperature": 0.2,
        "weakest_phone_weight": 1.0,
        "topology": "keyword_filler_segmental_crf_v1",
        "temperature": None,
    }
    assert "allow_legacy_qbyt_readout" not in config["stage2"]


@pytest.mark.parametrize(
    ("experiment", "version", "family", "detail"),
    [
        ("icefall_zipformer_stage2_pooling", 4, "pooling", "gru_last"),
        ("icefall_zipformer_stage2_eps", 4, "pooling", "eps_mean"),
        ("icefall_zipformer_stage2_eps_softmin", 4, "pooling", "eps_softmin"),
        ("icefall_zipformer_stage2_bounded", 5, "bounded", "bounded_segmental_v1"),
        (
            "icefall_zipformer_stage2_v6",
            6,
            "keyword_filler",
            "query_relative",
        ),
        (
            "icefall_zipformer_stage2_alignment",
            7,
            "keyword_filler",
            "one_vs_rest",
        ),
    ],
)
def test_readout_experiments_select_the_declared_score_family(
    experiment: str, version: int, family: str, detail: str
) -> None:
    config = config_to_dict(compose_config(experiment))
    score = resolve_qbyt_score_spec(config["stage2"])
    assert score.version == version
    assert score.family == family
    if family == "pooling":
        from dma_kws.stage2.readout_pooling import QbyTReadoutConfig

        assert score.value.mode == detail
        assert score.value == QbyTReadoutConfig(
            mode=detail, temperature=score.value.temperature
        )
        assert config["stage2"]["negative_tail_loss"]["enabled"] is False
    elif family == "bounded":
        assert score.value.topology == detail
        assert config["stage2"]["negative_tail_loss"]["enabled"] is False
    else:
        assert score.emission == detail


def test_eps_softmin_v41_experiment_resolves_extension_knobs_and_tail_loss():
    from dma_kws.stage2.readout_pooling import QbyTReadoutConfig

    config = config_to_dict(compose_config("icefall_zipformer_stage2_eps_softmin_v41"))
    score = resolve_qbyt_score_spec(config["stage2"])
    assert score.version == 4
    assert score.family == "pooling"
    assert score.value == QbyTReadoutConfig(
        mode="eps_softmin",
        temperature=1.0,
        sink_token=True,
        text_position="learned",
        audio_position="relative_bias",
    )
    assert score.value != QbyTReadoutConfig(mode="eps_softmin", temperature=1.0)
    stage2 = config["stage2"]
    assert stage2["negative_tail_loss"]["enabled"] is True
    assert stage2["negative_tail_loss"]["weight"] == 0.5
    assert stage2["background_negative"]["enabled"] is True
    assert stage2["noise_augmentation"]["enabled"] is False
    assert stage2["background_negative"]["audio_list_path"].endswith(
        "musan_split/train_background.list"
    )


def test_checkpoint_monitor_and_mode_accept_structured_overrides():
    config = config_to_dict(
        compose_config(
            overrides=["stage2.checkpoint.monitor=", "stage2.checkpoint.mode=min"]
        )
    )

    assert config["stage2"]["checkpoint"]["monitor"] == ""
    assert config["stage2"]["checkpoint"]["mode"] == "min"


def test_musan_mix_config_accepts_regular_hydra_overrides():
    config = config_to_dict(
        compose_config(
            overrides=[
                "prep.musan_mix.seed=7",
                "prep.musan_mix.noise.enabled=true",
                "prep.musan_mix.noise.snr_db=10",
                "prep.musan_mix.music.enabled=true",
                "prep.musan_mix.music.snr_db=12",
                "prep.musan_mix.speech.enabled=true",
                "prep.musan_mix.speech.relative_db=6",
                "prep.musan_mix.stationary_noise.enabled=true",
                "prep.musan_mix.stationary_noise.kind=white_gaussian",
                "prep.musan_mix.stationary_noise.snr_db=18",
                "prep.musan_mix.burst_noise.enabled=true",
                "prep.musan_mix.burst_noise.snr_db=8",
                "prep.musan_mix.burst_noise.snr_scope=whole_clip",
                "prep.musan_mix.burst_noise.event_count_min=2",
                "prep.musan_mix.burst_noise.event_count_max=4",
                "prep.musan_mix.burst_noise.duration_ms_min=80",
                "prep.musan_mix.burst_noise.duration_ms_max=240",
                "prep.musan_mix.burst_noise.fade_ms=12",
                "prep.musan_mix.burst_noise.allow_overlap=true",
                "prep.musan_mix.burst_noise.min_gap_ms=25",
                "prep.musan_mix.volume_variation.enabled=true",
                "prep.musan_mix.volume_variation.low_gain_db=-9",
                "prep.musan_mix.volume_variation.high_gain_db=3",
                "prep.musan_mix.volume_variation.segment_ms_min=200",
                "prep.musan_mix.volume_variation.segment_ms_max=600",
                "prep.musan_mix.volume_variation.transition_ms=40",
            ]
        )
    )

    assert config["prep"]["musan_mix"] == {
        "seed": 7,
        "noise": {"enabled": True, "snr_db": 10.0},
        "music": {"enabled": True, "snr_db": 12.0},
        "speech": {"enabled": True, "relative_db": 6.0},
        "stationary_noise": {
            "enabled": True,
            "kind": "white_gaussian",
            "snr_db": 18.0,
        },
        "burst_noise": {
            "enabled": True,
            "snr_db": 8.0,
            "snr_scope": "whole_clip",
            "event_count_min": 2,
            "event_count_max": 4,
            "duration_ms_min": 80.0,
            "duration_ms_max": 240.0,
            "fade_ms": 12.0,
            "allow_overlap": True,
            "min_gap_ms": 25.0,
        },
        "volume_variation": {
            "enabled": True,
            "low_gain_db": -9.0,
            "high_gain_db": 3.0,
            "segment_ms_min": 200.0,
            "segment_ms_max": 600.0,
            "transition_ms": 40.0,
        },
    }


def test_audio_export_config_defaults_disabled_and_accepts_overrides():
    defaults = config_to_dict(compose_config())
    assert defaults["prep"]["audio_export"] == {
        "mode": "disabled",
        "count": 5,
        "seed": 2025,
    }

    configured = config_to_dict(
        compose_config(
            overrides=[
                "prep.audio_export.mode=all",
                "prep.audio_export.count=0",
                "prep.audio_export.seed=17",
            ]
        )
    )
    assert configured["prep"]["audio_export"] == {
        "mode": "all",
        "count": 0,
        "seed": 17,
    }


def test_musan_mix_music_defaults_to_disabled_at_20_db():
    config = config_to_dict(compose_config())

    assert config["prep"]["musan_mix"]["music"] == {
        "enabled": False,
        "snr_db": 20.0,
    }


def test_musan_mix_synthetic_and_volume_defaults_are_disabled():
    config = config_to_dict(compose_config())

    assert config["prep"]["musan_mix"]["stationary_noise"] == {
        "enabled": False,
        "kind": "white_gaussian",
        "snr_db": 20.0,
    }
    assert config["prep"]["musan_mix"]["burst_noise"] == {
        "enabled": False,
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
    assert config["prep"]["musan_mix"]["volume_variation"] == {
        "enabled": False,
        "low_gain_db": -12.0,
        "high_gain_db": 6.0,
        "segment_ms_min": 250.0,
        "segment_ms_max": 750.0,
        "transition_ms": 50.0,
    }


def test_audio_aug_config_accepts_regular_hydra_overrides():
    config = config_to_dict(
        compose_config(
            overrides=[
                "prep.audio_aug.seed=7",
                "prep.audio_aug.speed_length_policy=center_crop_or_zero_pad",
                "prep.audio_aug.pcm_policy=float_unclipped",
                "prep.audio_aug.allow_signal_mimic_overlap=true",
                "prep.audio_aug.transforms.volume_gain.enabled=true",
                "prep.audio_aug.transforms.volume_gain.gain_db=-3",
                "prep.audio_aug.transforms.speed_change.enabled=true",
                "prep.audio_aug.transforms.speed_change.speed_factor=0.95",
                "prep.audio_aug.transforms.noise_mix.enabled=true",
                "prep.audio_aug.transforms.noise_mix.snr_db=10",
                "prep.audio_aug.transforms.noise_mix.snr_mode=upstream_std",
                "prep.audio_aug.transforms.subband_eq.enabled=true",
                "prep.audio_aug.transforms.subband_eq.low_min_gain_db=-4",
                "prep.audio_aug.transforms.subband_eq.high_min_gain_db=-6",
                "prep.audio_aug.transforms.band_limit.enabled=true",
                "prep.audio_aug.transforms.band_limit.mode=resample",
                "prep.audio_aug.transforms.band_limit.cutoff_hz=3000",
                "prep.audio_aug.transforms.band_limit.filter_order=6",
                "prep.audio_aug.transforms.band_limit.target_sample_rate=8000",
                "prep.audio_aug.transforms.narrowband.enabled=true",
                "prep.audio_aug.transforms.narrowband.target_sample_rate=10000",
                "prep.audio_aug.transforms.spectral_mask.enabled=true",
                "prep.audio_aug.transforms.spectral_mask.frequency_masks=2",
                "prep.audio_aug.transforms.spectral_mask.time_masks=1",
                "prep.audio_aug.transforms.spectral_mask.min_gain=0.2",
                "prep.audio_aug.transforms.spectral_mask.max_gain=0.8",
                "prep.audio_aug.transforms.amp_distortion.enabled=true",
                "prep.audio_aug.transforms.amp_distortion.distortion_type=poly_distortion",
                "prep.audio_aug.transforms.amp_distortion.rate=0.5",
                "prep.audio_aug.transforms.amp_distortion.gain_db=6",
                "prep.audio_aug.transforms.amp_distortion.max_db=-0.1",
                "prep.audio_aug.transforms.amp_distortion.mask_number=5",
                "prep.audio_aug.transforms.amp_distortion.a=0.5",
                "prep.audio_aug.transforms.amp_distortion.m=2",
                "prep.audio_aug.transforms.amp_distortion.n=3",
                "prep.audio_aug.transforms.signal_mimic.enabled=true",
                "prep.audio_aug.transforms.signal_mimic.subband_probability=0.1",
                "prep.audio_aug.transforms.signal_mimic.mute_probability=0.2",
                "prep.audio_aug.transforms.signal_mimic.band_limit_probability=0.3",
                "prep.audio_aug.transforms.signal_mimic.spectral_mask_probability=0.4",
                "prep.audio_aug.transforms.signal_mimic.narrowband_probability=0.5",
            ]
        )
    )

    assert config["prep"]["audio_aug"] == {
        "seed": 7,
        "speed_length_policy": "center_crop_or_zero_pad",
        "pcm_policy": "float_unclipped",
        "allow_signal_mimic_overlap": True,
        "transforms": {
            "volume_gain": {"enabled": True, "gain_db": -3.0},
            "speed_change": {"enabled": True, "speed_factor": 0.95},
            "noise_mix": {
                "enabled": True,
                "snr_db": 10.0,
                "snr_mode": "upstream_std",
            },
            "subband_eq": {
                "enabled": True,
                "low_min_gain_db": -4.0,
                "high_min_gain_db": -6.0,
            },
            "band_limit": {
                "enabled": True,
                "mode": "resample",
                "cutoff_hz": 3000.0,
                "filter_order": 6,
                "target_sample_rate": 8000,
            },
            "narrowband": {
                "enabled": True,
                "target_sample_rate": 10000,
            },
            "spectral_mask": {
                "enabled": True,
                "frequency_masks": 2,
                "time_masks": 1,
                "min_gain": 0.2,
                "max_gain": 0.8,
            },
            "amp_distortion": {
                "enabled": True,
                "distortion_type": "poly_distortion",
                "rate": 0.5,
                "gain_db": 6.0,
                "max_db": -0.1,
                "mask_number": 5,
                "a": 0.5,
                "m": 2,
                "n": 3,
            },
            "signal_mimic": {
                "enabled": True,
                "subband_probability": 0.1,
                "mute_probability": 0.2,
                "band_limit_probability": 0.3,
                "spectral_mask_probability": 0.4,
                "narrowband_probability": 0.5,
            },
        },
    }


def test_audio_aug_defaults_match_fixed_upstream_recipe():
    config = config_to_dict(compose_config())
    audio_aug = config["prep"]["audio_aug"]

    assert audio_aug["seed"] == 2025
    assert audio_aug["speed_length_policy"] == "variable"
    assert audio_aug["pcm_policy"] == "clip_round_each_stage"
    assert audio_aug["allow_signal_mimic_overlap"] is False
    assert set(audio_aug["transforms"]) == {
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
    assert all(
        transform["enabled"] is False
        for transform in audio_aug["transforms"].values()
    )
    assert audio_aug["transforms"]["noise_mix"]["snr_db"] == 30.0
    assert audio_aug["transforms"]["noise_mix"]["snr_mode"] == "exact_rms"
    assert audio_aug["transforms"]["speed_change"]["speed_factor"] == 1.05
    assert audio_aug["transforms"]["signal_mimic"] == {
        "enabled": False,
        "subband_probability": 0.6,
        "mute_probability": 0.0,
        "band_limit_probability": 0.0,
        "spectral_mask_probability": 0.0,
        "narrowband_probability": 0.0,
    }


def test_adapt_sweep_plot_defaults_match_structured_config():
    config = config_to_dict(compose_config())
    structured = AdaptSweepConfig()

    assert config["prep"]["plot_min_recall"] is None
    assert config["prep"]["plot_max_fpr"] is None
    assert config["adapt"]["sweep"]["plot_curves"] is structured.plot_curves
    assert config["adapt"]["sweep"]["plot_dpi"] == structured.plot_dpi
    assert structured.plot_min_recall is None
    assert structured.plot_max_fpr is None
    assert (
        config["adapt"]["sweep"]["plot_min_recall"]
        is structured.plot_min_recall
    )
    assert (
        config["adapt"]["sweep"]["plot_max_fpr"]
        is structured.plot_max_fpr
    )


def test_adapter_v2_stage2_config_freezes_the_domain_adapted_trunk(monkeypatch):
    monkeypatch.setenv("ICEFALL_CHECKPOINT", "/checkpoints/icefall.pt")
    monkeypatch.setenv("PHONEME_ADAPTER_V2_CHECKPOINT", "/checkpoints/adapter-v2.pt")
    config = config_to_dict(compose_config("icefall_zipformer_stage2_adapter_v2"))

    adapter = config["stage2"]["phoneme_adapter"]
    assert config["stage2"]["init_checkpoint"] == "/checkpoints/icefall.pt"
    assert adapter["enabled"] is True
    assert adapter["init_checkpoint"] == "/checkpoints/adapter-v2.pt"
    assert adapter["freeze"] is True
    assert adapter["ctc_weight"] == 0.0
    assert adapter["trunk"]["output_dim"] == 192
    assert config["stage2"]["run_name"].endswith("adapter-v2")


def test_hey_eva_adapter_v2_config_uses_the_complete_stage2_base(monkeypatch):
    monkeypatch.setenv("STAGE2_ADAPTER_V2_CHECKPOINT", "/checkpoints/stage2-v2.pt")
    config = config_to_dict(compose_config("adapt_hey_eva_icefall_adapter_v2"))

    adapter = config["stage2"]["phoneme_adapter"]
    assert config["prep"]["stage2_ckpt"] == "/checkpoints/stage2-v2.pt"
    assert adapter["enabled"] is True
    assert adapter["init_checkpoint"] == ""
    assert adapter["freeze"] is True
    assert adapter["ctc_weight"] == 0.0
    assert adapter["trunk"]["output_dim"] == 192
    assert config["adapt"]["data_root"] == (
        "/home/q00931063/DMA-KWS/data/dma-kws/"
        "chinese_accent_english_datasets/views/hey_eva_adapt"
    )
    assert config["prep"]["manifest_csv"].endswith(
        "views/hey_eva_adapt/manifests/real_source.csv"
    )
    assert config["adapt"]["train_phases"] == ["real"]
    assert config["adapt"]["exp_root"].endswith("lora/hey-eva-adapter-v2")


@pytest.mark.parametrize(
    "experiment",
    ["paper_ls460", "paper_ls_gs1460"],
)
def test_paper_configs_load(experiment):
    config = config_to_dict(compose_config(experiment))
    assert "paths" in config
    assert "stage2" in config


@pytest.mark.parametrize(
    ("experiment", "recipe", "hard_negative_ratio"),
    [
        ("paper_ls460", "init-ls-460", 1),
        ("paper_ls_gs1460", "ft-ls-gs-1460", 100),
    ],
)
def test_paper_configs_have_recipe_and_hard_negative_ratio(
    experiment, recipe, hard_negative_ratio
):
    config = config_to_dict(compose_config(experiment))

    assert config["training"]["recipe"] == recipe
    assert config["stage2"]["hard_negative_ratio"] == hard_negative_ratio
