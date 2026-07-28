"""Stage II checkpoint callbacks aligned with main ``qbyt/train*.py``."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pytorch_lightning.callbacks import ModelCheckpoint

_DEFAULT_EVERY_N_TRAIN_STEPS = 1000
_DEFAULT_SAVE_TOP_K = -1
_INIT_FILENAME = "step_{step:06d}"
_FINETUNE_FILENAME = "step_{step:06d}_auc_{val_auc:.6f}"

#: Metric that decides which checkpoint is "best". ``val_auc`` is the logger-free
#: alias ``Stage2LightningModule.on_validation_epoch_end`` publishes alongside
#: ``val/auc`` precisely so a filename and a monitor can reference it without a
#: slash. Set ``stage2.checkpoint.monitor`` to "" to go back to a plain step grid.
_DEFAULT_MONITOR = "val_auc"
_DEFAULT_MONITOR_MODE = "max"


def build_stage2_checkpoint_callback(
    config: dict[str, Any],
    recipe: str,
    *,
    checkpoint_dir: str | Path | None = None,
) -> ModelCheckpoint:
    """Build a Lightning ``ModelCheckpoint`` for Stage II init or finetune training.

    ``checkpoint_dir`` overrides ``stage2.checkpoint_dir`` for runs that own their
    own output tree (LoRA adaptation phases, sweep trials), so they never write
    into the pretrained Stage II checkpoint directory.
    """
    stage2 = config["stage2"]
    ckpt_cfg = stage2.get("checkpoint", {}) or {}

    if checkpoint_dir is None:
        checkpoint_dir = stage2.get(
            "checkpoint_dir", Path(config["paths"]["exp_root"]) / "stage2_qbyt" / "checkpoints"
        )
    checkpoint_dir = Path(checkpoint_dir)
    every_n_train_steps = int(ckpt_cfg.get("every_n_train_steps", _DEFAULT_EVERY_N_TRAIN_STEPS))
    save_top_k = int(ckpt_cfg.get("save_top_k", _DEFAULT_SAVE_TOP_K))

    if recipe.startswith("ft-"):
        filename = str(ckpt_cfg.get("finetune_filename", _FINETUNE_FILENAME))
    else:
        filename = str(ckpt_cfg.get("init_filename", _INIT_FILENAME))

    # Empty string is the opt-out, so ``.get`` cannot collapse it into the default.
    raw_monitor = ckpt_cfg.get("monitor", _DEFAULT_MONITOR)
    monitor = str(raw_monitor).strip() or None
    mode = str(ckpt_cfg.get("mode", _DEFAULT_MONITOR_MODE))

    return ModelCheckpoint(
        dirpath=str(checkpoint_dir),
        filename=filename,
        monitor=monitor,
        mode=mode,
        save_top_k=save_top_k,
        save_on_train_epoch_end=False,
        every_n_train_steps=every_n_train_steps,
        save_last=True,
    )
