"""Step A runner wiring that is easy to get silently wrong."""

from pathlib import Path

import pytest

from dma_kws.phoneme_adapter.runner import build_adapter_callbacks, resolve_val_check_interval
from dma_kws.training.checkpoint_callback import FreshValidationModelCheckpoint
from dma_kws.training.ddp import apply_step_based_validation

# Neither module imports torch or pytorch_lightning at module level, so the
# imports above are safe unguarded. The guard still has to be here: the
# ModelCheckpoint import inside build_adapter_callbacks needs Lightning.
pytest.importorskip("pytorch_lightning")


def test_resolve_val_check_interval_prefers_the_validation_section():
    """``build_trainer_kwargs`` reads ``validation.val_check_interval`` first, so
    the checkpoint cadence check has to look at the same value."""
    cfg = {"val_check_interval": 500, "validation": {"val_check_interval": 2000}}

    assert resolve_val_check_interval(cfg) == 2000
    assert resolve_val_check_interval({"val_check_interval": 500}) == 500


def test_checkpoint_cadence_need_not_be_a_multiple_of_validation(tmp_path):
    cfg = {
        "validation": {"val_check_interval": 2000},
        "checkpoint": {"every_n_train_steps": 1500, "save_top_k": 3},
    }

    _, checkpoint_callback = build_adapter_callbacks(tmp_path, cfg)

    assert isinstance(checkpoint_callback, FreshValidationModelCheckpoint)
    assert checkpoint_callback.fresh_every_n_train_steps == 1500
    assert checkpoint_callback._every_n_train_steps == 0


def test_aligned_checkpoint_cadence_is_accepted(tmp_path):
    cfg = {
        "validation": {"val_check_interval": 2000},
        "checkpoint": {"every_n_train_steps": 4000, "save_top_k": 3},
    }

    callbacks, checkpoint_callback = build_adapter_callbacks(Path(tmp_path), cfg)

    assert callbacks == [checkpoint_callback]
    assert checkpoint_callback.monitor == "val/per"
    assert isinstance(checkpoint_callback, FreshValidationModelCheckpoint)


def test_epoch_only_checkpointing_skips_the_cadence_check(tmp_path):
    callbacks, _ = build_adapter_callbacks(
        Path(tmp_path), {"validation": {"val_check_interval": 2000}, "checkpoint": {}}
    )

    assert len(callbacks) == 1


def test_adapter_checkpoint_rejects_a_misleading_monitor_override(tmp_path):
    with pytest.raises(SystemExit, match="requires.*val/per.*min"):
        build_adapter_callbacks(
            tmp_path,
            {
                "validation": {"val_check_interval": 100},
                "checkpoint": {"monitor": "val/loss", "mode": "min"},
            },
        )


def test_step_based_validation_is_applied_for_short_epochs():
    """A small manifest or a limit_steps smoke run has fewer batches than the
    validation interval, and Lightning raises rather than clamping."""
    trainer_kwargs = {"val_check_interval": 2000}
    apply_step_based_validation(trainer_kwargs, batches_per_epoch=50)

    assert trainer_kwargs["check_val_every_n_epoch"] is None

    long_epoch = {"val_check_interval": 2000}
    apply_step_based_validation(long_epoch, batches_per_epoch=5000)

    assert "check_val_every_n_epoch" not in long_epoch
