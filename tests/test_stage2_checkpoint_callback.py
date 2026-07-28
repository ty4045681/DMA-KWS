"""Tests for Stage II checkpoint callback builder."""

from __future__ import annotations

from pathlib import Path

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


def test_explicit_checkpoint_dir_overrides_stage2_section(tmp_path):
    trial_dir = tmp_path / "sweep" / "trial_1" / "tts" / "checkpoints"

    callback = build_stage2_checkpoint_callback(
        _base_config(tmp_path), "icefall-zipformer-frozen", checkpoint_dir=trial_dir
    )

    assert str(callback.dirpath).endswith(str(Path("trial_1") / "tts" / "checkpoints"))


def test_frozen_encoder_recipe_uses_init_filename(tmp_path):
    callback = build_stage2_checkpoint_callback(_base_config(tmp_path), "frozen-wenet-encoder")

    assert callback.filename == "step_{step:06d}"


def test_validation_metric_is_monitored_by_default(tmp_path):
    """Stage II must keep the best-AUC checkpoint, not only periodic snapshots.

    ``every_n_train_steps`` alone saves on a step grid; when val/auc peaks early
    and then degrades, nothing marks the peak, and picking it after the fact means
    re-reading the logs and guessing which step file matches.
    """
    callback = build_stage2_checkpoint_callback(_base_config(tmp_path), "init-ls-460")

    assert callback.monitor == "val_auc"
    assert callback.mode == "max"


def test_monitor_can_be_disabled_for_step_grid_only_runs(tmp_path):
    config = _base_config(tmp_path)
    config["stage2"]["checkpoint"] = {"monitor": ""}

    callback = build_stage2_checkpoint_callback(config, "init-ls-460")

    assert callback.monitor is None


def test_monitor_and_mode_are_configurable(tmp_path):
    config = _base_config(tmp_path)
    config["stage2"]["checkpoint"] = {"monitor": "val/eer", "mode": "min"}

    callback = build_stage2_checkpoint_callback(config, "init-ls-460")

    assert callback.monitor == "val/eer"
    assert callback.mode == "min"
