"""Stage II checkpoint callbacks."""

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
_MONITOR_UNSET = object()


class FreshValidationModelCheckpoint(ModelCheckpoint):
    """A monitored checkpoint that only saves from fresh validation metrics.

    Lightning's integer ``val_check_interval`` is expressed in train batches,
    while ``every_n_train_steps`` is expressed in optimizer steps.  With
    gradient accumulation, comparing the two configured integers is therefore
    insufficient.  More importantly, a train-batch checkpoint hook can see the
    metric left by the previous validation and save against a stale value.

    This callback keeps top-k selection in ``on_validation_end`` and checks the
    requested optimizer-step spacing there.  A monitored save is therefore
    made at the first fresh validation at or after the requested spacing, never
    against a metric from an earlier validation pass.  ``last.ckpt`` remains a
    separate crash-recovery snapshot: it is refreshed on the configured
    optimizer-step grid and once more at normal train end without participating
    in top-k selection.
    """

    def __init__(self, *, fresh_every_n_train_steps: int, **kwargs: Any) -> None:
        self.fresh_every_n_train_steps = max(0, int(fresh_every_n_train_steps))
        self._last_fresh_validation_step = 0
        self._last_recovery_step = 0
        super().__init__(
            every_n_train_steps=0,
            every_n_epochs=0,
            save_on_train_epoch_end=False,
            **kwargs,
        )

    def _save_recovery_checkpoint(self, trainer) -> None:
        """Refresh ``last.ckpt`` without suppressing same-step validation.

        ``ModelCheckpoint._save_last_checkpoint`` records the global step and
        its normal skip guard would then treat the fresh validation at that same
        step as already saved.  Preserve that guard state around the recovery
        write so top-k selection can still consume the new validation metric.
        """
        if not self.save_last:
            return
        previous_saved_step = self._last_global_step_saved
        self._last_recovery_step = int(trainer.global_step)
        self._save_last_checkpoint(trainer, self._monitor_candidates(trainer))
        self._last_global_step_saved = previous_saved_step

    def on_train_batch_end(
        self,
        trainer,
        pl_module,
        outputs,
        batch,
        batch_idx: int,
    ) -> None:
        step = int(trainer.global_step)
        if (
            self.fresh_every_n_train_steps <= 0
            or step <= 0
            or step == self._last_recovery_step
            or step % self.fresh_every_n_train_steps != 0
            or trainer.fast_dev_run
            or trainer.sanity_checking
        ):
            return
        self._save_recovery_checkpoint(trainer)

    def on_validation_end(self, trainer, pl_module) -> None:
        if self._should_skip_saving_checkpoint(trainer):
            return
        step = int(trainer.global_step)
        if (
            self.fresh_every_n_train_steps > 0
            and step - self._last_fresh_validation_step
            < self.fresh_every_n_train_steps
        ):
            return

        monitor_candidates = self._monitor_candidates(trainer)
        # ``trainer.save_checkpoint`` serializes callback state synchronously.
        # Record this validation before either save so both top-k and last.ckpt
        # restore the spacing boundary that produced them, not the previous one.
        self._last_fresh_validation_step = step
        self._save_topk_checkpoint(trainer, monitor_candidates)
        self._save_last_checkpoint(trainer, monitor_candidates)

    def on_train_end(self, trainer, pl_module) -> None:
        # The base implementation only writes when no prior ``last.ckpt``
        # exists.  Always refresh it when training stops between two periodic
        # recovery points so ``last.ckpt`` really means the final optimizer
        # state of this invocation.
        if self.save_last and int(trainer.global_step) != self._last_recovery_step:
            self._save_recovery_checkpoint(trainer)

    def state_dict(self) -> dict[str, Any]:
        state = super().state_dict()
        state["last_fresh_validation_step"] = self._last_fresh_validation_step
        state["last_recovery_step"] = self._last_recovery_step
        return state

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        super().load_state_dict(state_dict)
        self._last_fresh_validation_step = int(
            state_dict.get("last_fresh_validation_step", 0)
        )
        self._last_recovery_step = int(state_dict.get("last_recovery_step", 0))


def resolve_stage2_val_check_interval(stage2: dict[str, Any]) -> int:
    """Return the validation cadence used by ``build_trainer_kwargs``."""
    validation = stage2.get("validation", {}) or {}
    return int(
        validation.get("val_check_interval", stage2.get("val_check_interval", 1000))
    )


def build_stage2_checkpoint_callback(
    config: dict[str, Any],
    recipe: str,
    *,
    checkpoint_dir: str | Path | None = None,
    val_check_interval: int | None = None,
    monitor_override: str | None | object = _MONITOR_UNSET,
    filename_override: str | None = None,
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

    # Empty string is the opt-out, so ``.get`` cannot collapse it into the default.
    raw_monitor = (
        ckpt_cfg.get("monitor", _DEFAULT_MONITOR)
        if monitor_override is _MONITOR_UNSET
        else monitor_override
    )
    monitor = str(raw_monitor).strip() or None
    mode = str(ckpt_cfg.get("mode", _DEFAULT_MONITOR_MODE))

    if filename_override is not None:
        filename = str(filename_override)
    elif recipe.startswith("ft-"):
        candidate = str(ckpt_cfg.get("finetune_filename", _FINETUNE_FILENAME))
        # The default fine-tune template embeds val_auc. If selection is based
        # on another metric (or disabled), showing AUC in the filename implies
        # false provenance. A genuinely custom template remains authoritative.
        filename = (
            str(ckpt_cfg.get("init_filename", _INIT_FILENAME))
            if monitor != _DEFAULT_MONITOR and candidate == _FINETUNE_FILENAME
            else candidate
        )
    else:
        filename = str(ckpt_cfg.get("init_filename", _INIT_FILENAME))

    if monitor is not None:
        effective_val_interval = (
            resolve_stage2_val_check_interval(stage2)
            if val_check_interval is None
            else int(val_check_interval)
        )
        if effective_val_interval <= 0:
            raise SystemExit(
                "stage2.validation.val_check_interval must be positive when "
                f"stage2.checkpoint.monitor={monitor!r}."
            )

    common_kwargs = {
        "dirpath": str(checkpoint_dir),
        "filename": filename,
        "monitor": monitor,
        "mode": mode,
        "save_top_k": save_top_k,
        "save_last": True,
    }
    if monitor is not None:
        return FreshValidationModelCheckpoint(
            fresh_every_n_train_steps=every_n_train_steps,
            **common_kwargs,
        )
    return ModelCheckpoint(
        save_on_train_epoch_end=False,
        every_n_train_steps=every_n_train_steps,
        **common_kwargs,
    )
