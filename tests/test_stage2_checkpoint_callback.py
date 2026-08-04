"""Tests for Stage II checkpoint callback builder."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("pytorch_lightning")

from dma_kws.training.checkpoint_callback import (
    FreshValidationModelCheckpoint,
    build_stage2_checkpoint_callback,
    resolve_stage2_val_check_interval,
)


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
    assert isinstance(callback, FreshValidationModelCheckpoint)
    assert callback.fresh_every_n_train_steps == 1000
    assert callback._every_n_train_steps == 0
    assert str(callback.dirpath).endswith("checkpoints")


def test_finetune_recipe_uses_auc_filename(tmp_path):
    callback = build_stage2_checkpoint_callback(_base_config(tmp_path), "ft-ls-gs-1460")

    assert callback.filename == "step_{step:06d}_auc_{val_auc:.6f}"
    assert callback.save_top_k == -1
    assert callback.fresh_every_n_train_steps == 1000


def test_checkpoint_section_overrides_defaults(tmp_path):
    config = _base_config(tmp_path)
    config["stage2"]["validation"] = {"val_check_interval": 500}
    config["stage2"]["checkpoint"] = {
        "every_n_train_steps": 500,
        "init_filename": "custom_{step:04d}",
        "finetune_filename": "ft_{step:04d}_auc_{val_auc:.4f}",
    }

    init_cb = build_stage2_checkpoint_callback(config, "init-ls-460")
    assert init_cb.fresh_every_n_train_steps == 500
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


def test_non_auc_monitor_does_not_leave_auc_in_default_ft_filename(tmp_path):
    config = _base_config(tmp_path)
    config["stage2"]["checkpoint"] = {"monitor": "val/eer", "mode": "min"}

    callback = build_stage2_checkpoint_callback(config, "ft-ls-gs-1460")

    assert callback.filename == "step_{step:06d}"


def test_runtime_monitor_override_is_explicit(tmp_path):
    callback = build_stage2_checkpoint_callback(
        _base_config(tmp_path),
        "adapt",
        monitor_override="val_target_auc",
    )

    assert callback.monitor == "val_target_auc"


def test_adapt_filename_override_uses_the_same_metric_as_monitor(tmp_path):
    callback = build_stage2_checkpoint_callback(
        _base_config(tmp_path),
        "ft-ls-gs-1460",
        monitor_override="val_target_auc",
        filename_override="step_{step:06d}_target_auc_{val_target_auc:.6f}",
    )

    assert callback.monitor == "val_target_auc"
    assert callback.filename == "step_{step:06d}_target_auc_{val_target_auc:.6f}"
    assert "val_auc" not in callback.filename.replace("val_target_auc", "")


def test_validation_section_controls_checkpoint_cadence_check():
    stage2 = {
        "val_check_interval": 500,
        "validation": {"val_check_interval": 2000},
    }

    assert resolve_stage2_val_check_interval(stage2) == 2000


def test_monitored_checkpoint_cadence_need_not_equal_batch_validation_cadence(tmp_path):
    config = _base_config(tmp_path)
    config["stage2"].update(
        {
            "validation": {"val_check_interval": 1000},
            "checkpoint": {"every_n_train_steps": 1500},
        }
    )

    callback = build_stage2_checkpoint_callback(config, "init-ls-460")

    assert callback.fresh_every_n_train_steps == 1500
    assert callback._every_n_train_steps == 0


def test_monitored_checkpoint_cadence_accepts_validation_multiple(tmp_path):
    config = _base_config(tmp_path)
    config["stage2"].update(
        {
            "validation": {"val_check_interval": 1000},
            "checkpoint": {"every_n_train_steps": 2000},
        }
    )

    callback = build_stage2_checkpoint_callback(config, "init-ls-460")

    assert callback.monitor == "val_auc"
    assert callback.fresh_every_n_train_steps == 2000


def test_step_grid_without_monitor_does_not_require_validation_alignment(tmp_path):
    config = _base_config(tmp_path)
    config["stage2"].update(
        {
            "validation": {"val_check_interval": 1000},
            "checkpoint": {"every_n_train_steps": 1500, "monitor": ""},
        }
    )

    callback = build_stage2_checkpoint_callback(config, "init-ls-460")

    assert callback.monitor is None
    assert callback._every_n_train_steps == 1500


def test_explicit_runtime_validation_interval_controls_cadence_check(tmp_path):
    config = _base_config(tmp_path)
    config["stage2"].update(
        {
            "validation": {"val_check_interval": 1000},
            "checkpoint": {"every_n_train_steps": 500},
        }
    )

    callback = build_stage2_checkpoint_callback(
        config,
        "adapt",
        val_check_interval=500,
    )

    assert callback.fresh_every_n_train_steps == 500


def test_monitored_checkpoint_only_saves_at_fresh_validation_spacing(tmp_path, monkeypatch):
    callback = build_stage2_checkpoint_callback(_base_config(tmp_path), "init-ls-460")
    trainer = type("Trainer", (), {"global_step": 0})()
    saved_steps: list[int] = []

    monkeypatch.setattr(callback, "_should_skip_saving_checkpoint", lambda trainer: False)
    monkeypatch.setattr(callback, "_monitor_candidates", lambda trainer: {"val_auc": 0.9})
    monkeypatch.setattr(
        callback,
        "_save_topk_checkpoint",
        lambda trainer, metrics: saved_steps.append(int(trainer.global_step)),
    )
    monkeypatch.setattr(callback, "_save_last_checkpoint", lambda trainer, metrics: None)

    for step in (400, 999, 1000, 1400, 2100):
        trainer.global_step = step
        callback.on_validation_end(trainer, None)

    assert saved_steps == [1000, 2100]


def test_fresh_validation_checkpoint_persists_spacing_state(tmp_path):
    callback = build_stage2_checkpoint_callback(_base_config(tmp_path), "init-ls-460")
    callback._last_fresh_validation_step = 1234
    callback._last_recovery_step = 1200
    state = callback.state_dict()

    restored = build_stage2_checkpoint_callback(_base_config(tmp_path), "init-ls-460")
    restored.load_state_dict(state)

    assert restored._last_fresh_validation_step == 1234
    assert restored._last_recovery_step == 1200


def test_fresh_validation_step_is_visible_to_same_step_checkpoint_saves(
    tmp_path, monkeypatch
):
    callback = build_stage2_checkpoint_callback(
        _base_config(tmp_path), "init-ls-460"
    )
    trainer = type("Trainer", (), {"global_step": 1000})()
    persisted_steps: list[int] = []

    monkeypatch.setattr(
        callback, "_should_skip_saving_checkpoint", lambda trainer: False
    )
    monkeypatch.setattr(
        callback, "_monitor_candidates", lambda trainer: {"val_auc": 0.9}
    )

    def capture_state(trainer, metrics):
        persisted_steps.append(
            int(callback.state_dict()["last_fresh_validation_step"])
        )

    monkeypatch.setattr(callback, "_save_topk_checkpoint", capture_state)
    monkeypatch.setattr(callback, "_save_last_checkpoint", capture_state)

    callback.on_validation_end(trainer, None)

    assert persisted_steps == [1000, 1000]


def test_recovery_save_does_not_suppress_fresh_validation(tmp_path, monkeypatch):
    callback = build_stage2_checkpoint_callback(_base_config(tmp_path), "init-ls-460")
    trainer = type(
        "Trainer",
        (),
        {"global_step": 1000, "fast_dev_run": False, "sanity_checking": False},
    )()
    events: list[str] = []

    monkeypatch.setattr(callback, "_monitor_candidates", lambda trainer: {"val_auc": 0.8})

    def save_last(trainer, metrics):
        events.append("last")
        callback._last_global_step_saved = int(trainer.global_step)

    monkeypatch.setattr(callback, "_save_last_checkpoint", save_last)
    monkeypatch.setattr(callback, "_should_skip_saving_checkpoint", lambda trainer: False)
    monkeypatch.setattr(
        callback,
        "_save_topk_checkpoint",
        lambda trainer, metrics: events.append("topk"),
    )

    callback.on_train_batch_end(trainer, None, None, None, 0)
    assert callback._last_global_step_saved != trainer.global_step
    callback.on_validation_end(trainer, None)

    assert events == ["last", "topk", "last"]


def test_real_trainer_uses_current_validation_metric_with_accumulation(tmp_path):
    import pytorch_lightning as pl
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    class TinyModule(pl.LightningModule):
        def __init__(self) -> None:
            super().__init__()
            self.layer = torch.nn.Linear(1, 1)
            self.validation_steps: list[int] = []

        def training_step(self, batch, batch_idx):
            (x,) = batch
            return self.layer(x).square().mean()

        def validation_step(self, batch, batch_idx):
            return None

        def on_validation_epoch_end(self) -> None:
            step = int(self.global_step)
            self.validation_steps.append(step)
            value = torch.tensor(1.0 / step)
            self.log("val/per", value)
            self.log("val_per", value, logger=False)

        def configure_optimizers(self):
            return torch.optim.SGD(self.parameters(), lr=0.01)

    callback = FreshValidationModelCheckpoint(
        fresh_every_n_train_steps=1,
        dirpath=tmp_path,
        monitor="val/per",
        mode="min",
        save_top_k=1,
        save_last=True,
        filename="adapter_{step:02d}_{val_per:.3f}",
    )
    model = TinyModule()
    train_loader = DataLoader(TensorDataset(torch.ones(4, 1)), batch_size=1)
    val_loader = DataLoader(TensorDataset(torch.ones(1, 1)), batch_size=1)
    trainer = pl.Trainer(
        accelerator="cpu",
        devices=1,
        logger=False,
        callbacks=[callback],
        max_steps=2,
        accumulate_grad_batches=2,
        val_check_interval=2,
        check_val_every_n_epoch=None,
        num_sanity_val_steps=0,
        enable_progress_bar=False,
        enable_model_summary=False,
    )

    trainer.fit(model, train_loader, val_loader)

    assert model.validation_steps == [1, 2]
    assert float(callback.best_model_score) == pytest.approx(0.5)
    assert "step=02" in Path(callback.best_model_path).name
    assert (tmp_path / "last.ckpt").is_file()
