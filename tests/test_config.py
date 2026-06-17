from pathlib import Path

import pytest

from dma_kws.config import load_config, require_sections


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
