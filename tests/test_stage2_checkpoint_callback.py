"""Tests for Stage II checkpoint callback builder."""

from __future__ import annotations

import pytest

pytest.importorskip("pytorch_lightning")

from dma_kws.training.checkpoint_callback import build_stage2_checkpoint_callback


def _base_config(tmp_path) -> dict:
    return {
        "paths": {"exp_root": str(tmp_path / "exp")},
        "stage2": {
            "checkpoint_dir": str(tmp_path / "checkpoints"),
        },
    }


def test_init_recipe_uses_step_only_filename(tmp_path):
    callback = build_stage2_checkpoint_callback(_base_config(tmp_path), "init-ls-460")

    assert callback.filename == "step_{step:06d}"
    assert callback.save_top_k == -1
    assert callback._every_n_train_steps == 1000
    assert str(callback.dirpath).endswith("checkpoints")


def test_finetune_recipe_uses_auc_filename(tmp_path):
    callback = build_stage2_checkpoint_callback(_base_config(tmp_path), "ft-ls-gs-1460")

    assert callback.filename == "step_{step:06d}_auc_{val_auc:.6f}"
    assert callback.save_top_k == -1
    assert callback._every_n_train_steps == 1000


def test_checkpoint_section_overrides_defaults(tmp_path):
    config = _base_config(tmp_path)
    config["stage2"]["checkpoint"] = {
        "every_n_train_steps": 500,
        "init_filename": "custom_{step:04d}",
        "finetune_filename": "ft_{step:04d}_auc_{val_auc:.4f}",
    }

    init_cb = build_stage2_checkpoint_callback(config, "init-ls-460")
    assert init_cb._every_n_train_steps == 500
    assert init_cb.filename == "custom_{step:04d}"

    ft_cb = build_stage2_checkpoint_callback(config, "ft-ls-gs-1460")
    assert ft_cb.filename == "ft_{step:04d}_auc_{val_auc:.4f}"


def test_frozen_encoder_recipe_uses_init_filename(tmp_path):
    callback = build_stage2_checkpoint_callback(_base_config(tmp_path), "frozen-wenet-encoder")

    assert callback.filename == "step_{step:06d}"
