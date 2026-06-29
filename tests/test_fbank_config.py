from pathlib import Path

import pytest

from dma_kws.config import (
    FbankConfig,
    compose_config,
    config_to_dict,
    fbank_kwargs,
    get_eval_fbank_config,
    get_fbank_config,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_fbank_config_defaults_without_sections():
    cfg = get_fbank_config({})

    assert cfg == FbankConfig()


def test_fbank_config_uses_stage1_input_dim_when_fbank_missing():
    cfg = get_fbank_config({"stage1": {"input_dim": 64}})

    assert cfg.num_mel_bins == 64
    assert cfg.dither == 0.1
    assert cfg.frame_length == 25
    assert cfg.frame_shift == 10
    assert cfg.window_type == "povey"


def test_fbank_config_yaml_overrides_top_level_section():
    config = config_to_dict(compose_config("wenet_asr_stage2"))

    cfg = get_fbank_config(config)

    assert cfg == FbankConfig(
        num_mel_bins=80,
        frame_length=25,
        frame_shift=10,
        dither=0.1,
        window_type="povey",
    )


def test_fbank_config_falls_back_to_stage1_input_dim_in_paper_config():
    config = config_to_dict(compose_config("paper_ls460"))

    cfg = get_fbank_config(config)

    assert cfg.num_mel_bins == config["stage1"]["input_dim"]
    assert cfg == FbankConfig()


def test_fbank_config_partial_yaml_override():
    config = {
        "stage1": {"input_dim": 80},
        "fbank": {"dither": 0.0, "frame_shift": 8},
    }

    cfg = get_fbank_config(config)

    assert cfg.num_mel_bins == 80
    assert cfg.dither == 0.0
    assert cfg.frame_shift == 8
    assert cfg.frame_length == 25
    assert cfg.window_type == "povey"


def test_eval_fbank_config_overrides_dither_only():
    config = {
        "fbank": {
            "num_mel_bins": 80,
            "frame_length": 25,
            "frame_shift": 10,
            "dither": 0.1,
            "window_type": "povey",
        },
        "stage2": {
            "eval": {
                "fbank": {
                    "dither": 0.0,
                }
            }
        },
    }

    cfg = get_eval_fbank_config(config)

    assert cfg.dither == 0.0
    assert cfg.num_mel_bins == 80
    assert cfg.frame_length == 25
    assert cfg.frame_shift == 10
    assert cfg.window_type == "povey"


def test_fbank_kwargs_matches_compute_fbank_for_clip_signature():
    cfg = FbankConfig(dither=0.0, frame_shift=8)

    assert fbank_kwargs(cfg) == {
        "num_mel_bins": 80,
        "frame_length": 25,
        "frame_shift": 8,
        "dither": 0.0,
        "window_type": "povey",
    }


def test_get_fbank_config_rejects_non_mapping_section():
    with pytest.raises(ValueError, match="Config section 'fbank' must be a mapping"):
        get_fbank_config({"fbank": "invalid"})


def test_get_eval_fbank_config_rejects_non_mapping_eval_fbank():
    config = {"stage2": {"eval": {"fbank": "invalid"}}}

    with pytest.raises(ValueError, match="Config section 'stage2.eval.fbank' must be a mapping"):
        get_eval_fbank_config(config)
