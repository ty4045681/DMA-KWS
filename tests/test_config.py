from pathlib import Path

import pytest

from dma_kws.config import get_tokenizer_config, load_config, require_sections

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


def test_require_sections_reports_missing_sections():
    with pytest.raises(ValueError, match="Missing required config sections: stage2, demo"):
        require_sections({"paths": {}, "stage1": {}}, ["paths", "stage1", "stage2", "demo"])


def test_demo_config_loads_with_tokenizer_and_training_seed():
    config = load_config(REPO_ROOT / "configs" / "demo_librispeech100.yaml")

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


@pytest.mark.parametrize(
    "config_name",
    ["paper_ls460.yaml", "paper_ls_gs1460.yaml"],
)
def test_paper_configs_load(config_name):
    config = load_config(REPO_ROOT / "configs" / config_name)
    assert "paths" in config
    assert "stage2" in config


@pytest.mark.parametrize(
    ("config_name", "recipe", "hard_negative_ratio"),
    [
        ("paper_ls460.yaml", "init-ls-460", 1),
        ("paper_ls_gs1460.yaml", "ft-ls-gs-1460", 100),
    ],
)
def test_paper_configs_have_recipe_and_hard_negative_ratio(
    config_name, recipe, hard_negative_ratio
):
    config = load_config(REPO_ROOT / "configs" / config_name)

    assert config["training"]["recipe"] == recipe
    assert config["stage2"]["hard_negative_ratio"] == hard_negative_ratio
