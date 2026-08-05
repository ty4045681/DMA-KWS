"""Dense per-validation metrics history and cross-run comparison CSVs.

Complements the sparse Lightning ``metrics.csv`` (train/val metrics land on
different rows there) with two dense, comparison-friendly tables:

- ``eval_history.csv``: one wide row per validation pass (step, timing, latest
  train metrics, all val metrics) written next to the Lightning logs.
- ``runs.csv``: one row per completed run (hyperparameters + final/best
  metrics) appended under the experiment root for cross-run comparison.

The CSV helpers and hyperparameter collection are torch-free; the Lightning
callback imports pytorch_lightning lazily via ``build_metrics_history_callback``.
"""

from __future__ import annotations

import csv
import math
import os
import tempfile
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

#: Metric-name fragments whose improvement direction is "higher is better".
_MAX_METRIC_HINTS = (
    "auc",
    "acc",
    "f1",
    "precision",
    "recall",
    "tpr",
)
_MIN_METRIC_HINTS = (
    "loss",
    "eer",
    "error",
    "brier",
    "ece",
    "fpr",
    "fnr",
    "skip_rate",
)

#: Compatibility/checkpoint-only aliases excluded from dense history/final rows.
#: They remain in Lightning's callback metric namespace for old consumers, but
#: writing them next to the canonical hierarchy would make CSV columns
#: ambiguous and duplicate the same scalar under several names.
_ALIAS_METRICS = frozenset(
    {
        "train/loss",
        "train/utt_loss",
        "train/seq_loss",
        "train/seq_progress_loss",
        "train/seq_completion_loss",
        "train/ctc_loss",
        "train/lr",
        "train/grad_norm",
        "val_auc",
        "val_target_auc",
        "val_lph_auc",
        "val_per",
    }
)


def metric_direction(name: str) -> str:
    """Return ``"max"`` or ``"min"``, the improvement direction for a metric."""
    lowered = name.lower()
    return "max" if any(hint in lowered for hint in _MAX_METRIC_HINTS) else "min"


def tracks_best_metric(name: str) -> bool:
    """Whether a metric has a meaningful monotonic "best" direction.

    Thresholds, sample counts and score quantiles belong in validation history,
    but calling their numeric minimum a "best" value is misleading.  Only
    objective/performance metrics participate in the cross-run best summary.
    """
    lowered = name.lower()
    if "threshold" in lowered:
        return False
    leaf = lowered.rsplit("/", 1)[-1]
    if leaf == "per":
        return True
    return any(
        hint in lowered for hint in (*_MAX_METRIC_HINTS, *_MIN_METRIC_HINTS)
    )


def _format_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


@contextmanager
def _exclusive_csv_lock(path: Path):
    """Serialize cross-process updates to a shared history CSV."""
    lock_path = path.with_suffix(f"{path.suffix}.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as lock_handle:
        try:
            import fcntl
        except ImportError:  # pragma: no cover - Windows fallback
            yield
            return
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)


def append_wide_row(csv_path: str | Path, row: dict[str, Any]) -> Path:
    """Append ``row`` to a wide CSV, expanding the header when new columns appear.

    Existing rows keep their values; cells for newly added columns stay empty so
    every row remains aligned with the (stable, insertion-ordered) header.
    """
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    formatted = {key: _format_value(value) for key, value in row.items()}

    with _exclusive_csv_lock(path):
        _append_wide_row_unlocked(path, formatted)
    return path


def _append_wide_row_unlocked(path: Path, formatted: dict[str, str]) -> None:
    """Implementation of :func:`append_wide_row` under its sidecar lock."""

    if not path.exists():
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(formatted))
            writer.writeheader()
            writer.writerow(formatted)
        return

    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        new_columns = [key for key in formatted if key not in fieldnames]
        existing_rows = list(reader) if new_columns else []

    if new_columns:
        fieldnames.extend(new_columns)
        fd, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
        )
        try:
            with os.fdopen(fd, "w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames, restval="")
                writer.writeheader()
                writer.writerows(existing_rows)
                writer.writerow(formatted)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_name, path)
        finally:
            if os.path.exists(temporary_name):
                os.unlink(temporary_name)
    else:
        with path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, restval="", extrasaction="ignore")
            writer.writerow(formatted)


def collect_hparams(
    config: dict[str, Any],
    *,
    section: str = "stage2",
    effective_max_steps: int | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a flat hyperparameter dict for ``log_hyperparams`` and ``runs.csv``.

    ``section`` selects where run-specific settings come from (``stage2`` or
    ``adapt``), mirroring ``build_run_summary_rows`` so logged values match the
    actual training configuration.
    """
    stage2 = config.get("stage2") or {}
    stage = config.get(section) or {}
    trainer_stage = stage2 if section == "adapt" else stage

    def setting(key: str, default: Any) -> Any:
        value = stage.get(key)
        return stage2.get(key, default) if value is None else value

    if section == "adapt":
        from dma_kws.training.adapt_params import resolve_adapt_lr

        lr, _ = resolve_adapt_lr(stage)
        hparams: dict[str, Any] = {
            "learning_rate": lr,
            "optimizer": str(stage.get("optimizer", "adam")).lower(),
            "weight_decay": float(stage.get("weight_decay", 0.0)),
            "warmup_steps": int(stage.get("warmup_steps", 100)),
            "max_steps": int(stage.get("max_steps", 3000)),
            "rank": int(stage.get("rank", 16)),
            "alpha": float(stage.get("alpha", 32)),
            "mix_ratio": float(stage.get("mix_ratio", 0.5)),
            "keyword": str(stage.get("keyword", "")),
            "phase": str(stage.get("phase", "tts")),
        }
    elif section == "stage1":
        configured_steps = int(stage.get("max_train_steps", 0))
        hparams = {
            "learning_rate": float(stage.get("learning_rate", 1e-3)),
            "optimizer": "adam",
            "weight_decay": 0.0,
            "warmup_steps": int(stage.get("warmup_steps", 0)),
            "max_steps": configured_steps if configured_steps else -1,
            "max_epochs": int(stage.get("max_epochs", 1)),
        }
    else:
        from dma_kws.training.scheduler import build_optimizer_config

        opt_cfg = build_optimizer_config(stage)
        hparams = {
            "learning_rate": opt_cfg["lr"],
            "optimizer": opt_cfg["optimizer"],
            "weight_decay": opt_cfg["weight_decay"],
            "warmup_steps": opt_cfg["warmup_steps"],
            "max_steps": int(stage.get("max_steps", 50000)),
        }

    hparams.update(
        {
            "batch_size_per_gpu": int(setting("batch_size_per_gpu", 64)),
            "accumulate_grad_batches": int(
                trainer_stage.get("accumulate_grad_batches", 1)
            ),
            "seed": int((config.get("training") or {}).get("seed", 2025)),
            "recipe": str((config.get("training") or {}).get("recipe", "")),
        }
    )
    validation_cfg = stage.get("validation", {}) or {}
    val_interval = validation_cfg.get("val_check_interval", stage.get("val_check_interval"))
    if isinstance(val_interval, int) and not isinstance(val_interval, bool):
        accumulate = max(1, int(trainer_stage.get("accumulate_grad_batches", 1)))
        hparams["val_check_interval_train_batches"] = int(val_interval)
        hparams["val_check_interval_optimizer_steps_approx"] = math.ceil(
            int(val_interval) / accumulate
        )
    if effective_max_steps is not None:
        hparams["max_steps"] = int(effective_max_steps)
    if section in {"stage2", "adapt"}:
        from dma_kws.stage2.readout import resolve_qbyt_readout

        sequence_loss = stage2.get("sequence_loss", {}) or {}
        validation = stage2.get("validation", {}) or {}
        readout = resolve_qbyt_readout(stage2)
        hparams.update(
            {
                "qbyt_readout_mode": readout.mode,
                "qbyt_readout_temperature": readout.temperature,
                "qbyt_deployment_threshold": float(
                    ((config.get("demo") or {}).get("qbyt_threshold", 0.5))
                ),
                "score_ece_num_bins": int(validation.get("ece_num_bins", 15)),
                "seq_diagnostic_threshold": float(
                    validation.get("seq_diagnostic_threshold", 0.5)
                ),
                "seq_target_mode": str(
                    sequence_loss.get("target_mode", "ordered_contiguous_prefix")
                ),
                "seq_progress_weight": float(
                    sequence_loss.get("progress_weight", 0.5)
                ),
                "seq_completion_weight": float(
                    sequence_loss.get("completion_weight", 0.5)
                ),
                "seq_normalization": str(
                    sequence_loss.get("normalization", "sample")
                ),
            }
        )
    if extra:
        hparams.update(extra)
    return hparams


def build_run_record(
    *,
    run_name: str,
    hparams: dict[str, Any],
    final_metrics: dict[str, float],
    best_metrics: dict[str, float],
    global_step: int,
    duration_seconds: float,
    metric_step: int | None = None,
    best_steps: dict[str, int] | None = None,
    identity: dict[str, Any] | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one flat ``runs.csv`` row: identity, hyperparameters, final/best metrics."""
    record: dict[str, Any] = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "run": run_name,
    }
    if identity:
        record.update(identity)
    record.update(hparams)
    record["global_step"] = int(global_step)
    record["metric_step"] = "" if metric_step is None else int(metric_step)
    record["duration_s"] = round(float(duration_seconds), 1)
    if provenance:
        record.update(provenance)
    for name in sorted(final_metrics):
        record[f"final/{name}"] = final_metrics[name]
    for name in sorted(best_metrics):
        record[f"best/{name}"] = best_metrics[name]
        if best_steps and name in best_steps:
            record[f"best_step/{name}"] = int(best_steps[name])
    return record


def numeric_callback_metrics(callback_metrics: dict[str, Any]) -> dict[str, float]:
    """Extract float-convertible metrics, dropping console/checkpoint aliases."""
    metrics: dict[str, float] = {}
    for name, value in callback_metrics.items():
        if name in _ALIAS_METRICS:
            continue
        try:
            metrics[name] = float(value)
        except (TypeError, ValueError):
            continue
    return metrics


def update_best_metrics(best: dict[str, float], metrics: dict[str, float]) -> set[str]:
    """Fold ``metrics`` into ``best`` in place, honoring each metric's direction."""
    import math

    updated: set[str] = set()
    for name, value in metrics.items():
        if not tracks_best_metric(name) or not math.isfinite(value):
            continue
        current = best.get(name)
        if current is None or (value > current if metric_direction(name) == "max" else value < current):
            best[name] = value
            updated.add(name)
    return updated


def build_metrics_history_callback(
    *,
    run_name: str,
    run_id: str | None = None,
    default_dir: str | Path,
    filename: str = "eval_history.csv",
):
    """Return a Lightning callback appending one dense row per validation pass.

    The CSV lands inside the run's Lightning log directory (``version_N``) when
    available, else under ``default_dir``. The callback also tracks best val
    metrics and total fit duration for the ``runs.csv`` record.
    """
    import pytorch_lightning as pl

    class MetricsHistoryCallback(pl.Callback):
        def __init__(self) -> None:
            super().__init__()
            self.run_name = run_name
            self.run_id = run_id or run_name
            self.csv_path: Path | None = None
            self.best: dict[str, float] = {}
            self.best_steps: dict[str, int] = {}
            self.last_validation_step: int | None = None
            self._duration_before_resume = 0.0
            self._fit_start: float | None = None
            self._fit_elapsed = 0.0
            self._last_row_time: float | None = None
            self._last_row_step = 0

        @property
        def duration_seconds(self) -> float:
            current = self._fit_elapsed
            if self._fit_start is not None:
                current = time.monotonic() - self._fit_start
            return self._duration_before_resume + current

        def state_dict(self) -> dict[str, Any]:
            return {
                "best": dict(self.best),
                "best_steps": dict(self.best_steps),
                "last_validation_step": self.last_validation_step,
                "duration_seconds": self.duration_seconds,
            }

        def load_state_dict(self, state_dict: dict[str, Any]) -> None:
            self.best = {
                str(name): float(value)
                for name, value in (state_dict.get("best") or {}).items()
            }
            self.best_steps = {
                str(name): int(value)
                for name, value in (state_dict.get("best_steps") or {}).items()
            }
            validation_step = state_dict.get("last_validation_step")
            self.last_validation_step = (
                None if validation_step is None else int(validation_step)
            )
            self._duration_before_resume = float(
                state_dict.get("duration_seconds", 0.0)
            )

        def on_fit_start(self, trainer, pl_module) -> None:
            # ``default_dir`` is RunContext.run_dir, the canonical location
            # shared by every backend. The first logger may be W&B/Trackio and
            # expose a backend-specific directory, so it must not decide where
            # this local history file lands.
            self.csv_path = Path(default_dir) / filename
            self._fit_start = time.monotonic()
            self._fit_elapsed = 0.0
            self._last_row_time = self._fit_start
            self._last_row_step = int(trainer.global_step)

        def on_fit_end(self, trainer, pl_module) -> None:
            if self._fit_start is None:
                return
            self._fit_elapsed = time.monotonic() - self._fit_start
            self._fit_start = None

        def on_validation_end(self, trainer, pl_module) -> None:
            if trainer.sanity_checking or not trainer.is_global_zero:
                return
            metrics = numeric_callback_metrics(dict(trainer.callback_metrics))
            step = int(trainer.global_step)
            updated = update_best_metrics(
                self.best,
                {name: value for name, value in metrics.items() if name.startswith("val")},
            )
            for name in updated:
                self.best_steps[name] = step
            self.last_validation_step = step

            now = time.monotonic()
            interval = now - (self._last_row_time or now)
            steps_per_sec = (step - self._last_row_step) / interval if interval > 0 else 0.0
            row: dict[str, Any] = {
                "run": self.run_name,
                "run_id": self.run_id,
                "step": step,
                "epoch": int(trainer.current_epoch),
                "wall_time_s": round(self.duration_seconds, 3),
                "steps_per_sec": round(steps_per_sec, 4),
            }
            row.update({name: metrics[name] for name in sorted(metrics)})
            self._last_row_time = now
            self._last_row_step = step
            if self.csv_path is not None:
                append_wide_row(self.csv_path, row)

    return MetricsHistoryCallback()
