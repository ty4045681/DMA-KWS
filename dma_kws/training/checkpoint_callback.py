"""Stage II checkpoint callbacks aligned with main ``qbyt/train*.py``."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pytorch_lightning.callbacks import ModelCheckpoint

_DEFAULT_EVERY_N_TRAIN_STEPS = 1000
_DEFAULT_SAVE_TOP_K = -1
_INIT_FILENAME = "step_{step:06d}"
_FINETUNE_FILENAME = "step_{step:06d}_auc_{val_auc:.6f}"


def build_stage2_checkpoint_callback(config: dict[str, Any], recipe: str) -> ModelCheckpoint:
    """Build a Lightning ``ModelCheckpoint`` for Stage II init or finetune training."""
    stage2 = config["stage2"]
    ckpt_cfg = stage2.get("checkpoint", {}) or {}

    checkpoint_dir = Path(stage2.get("checkpoint_dir", Path(config["paths"]["exp_root"]) / "stage2_qbyt" / "checkpoints"))
    every_n_train_steps = int(ckpt_cfg.get("every_n_train_steps", _DEFAULT_EVERY_N_TRAIN_STEPS))
    save_top_k = int(ckpt_cfg.get("save_top_k", _DEFAULT_SAVE_TOP_K))

    if recipe.startswith("ft-"):
        filename = str(ckpt_cfg.get("finetune_filename", _FINETUNE_FILENAME))
    else:
        filename = str(ckpt_cfg.get("init_filename", _INIT_FILENAME))

    return ModelCheckpoint(
        dirpath=str(checkpoint_dir),
        filename=filename,
        save_top_k=save_top_k,
        save_on_train_epoch_end=False,
        every_n_train_steps=every_n_train_steps,
    )
