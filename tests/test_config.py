from pathlib import Path

import pytest

from dma_kws.config import (
    compose_config,
    config_to_dict,
    get_tokenizer_config,
    load_config,
    require_sections,
)

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
        "progress_weight": 0.5,
        "completion_weight": 0.5,
        "normalization": "sample",
    }


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
