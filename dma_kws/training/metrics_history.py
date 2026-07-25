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
import time
from datetime import datetime
from pathlib import Path
from typing import Any

#: Metric-name fragments whose improvement direction is "higher is better".
_MAX_METRIC_HINTS = ("auc", "acc", "f1", "precision", "recall")

#: Console/checkpoint-only aliases excluded from history rows.
_ALIAS_METRICS = ("val_auc",)


def metric_direction(name: str) -> str:
    """Return ``"max"`` or ``"min"``, the improvement direction for a metric."""
    lowered = name.lower()
    return "max" if any(hint in lowered for hint in _MAX_METRIC_HINTS) else "min"


def _format_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def append_wide_row(csv_path: str | Path, row: dict[str, Any]) -> Path:
    """Append ``row`` to a wide CSV, expanding the header when new columns appear.

    Existing rows keep their values; cells for newly added columns stay empty so
    every row remains aligned with the (stable, insertion-ordered) header.
    """
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    formatted = {key: _format_value(value) for key, value in row.items()}

    if not path.exists():
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(formatted))
            writer.writeheader()
            writer.writerow(formatted)
        return path

    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        new_columns = [key for key in formatted if key not in fieldnames]
        existing_rows = list(reader) if new_columns else []

    if new_columns:
        fieldnames.extend(new_columns)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, restval="")
            writer.writeheader()
            writer.writerows(existing_rows)
            writer.writerow(formatted)
    else:
        with path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, restval="", extrasaction="ignore")
            writer.writerow(formatted)
    return path


def collect_hparams(
    config: dict[str, Any],
    *,
    section: str = "stage2",
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a flat hyperparameter dict for ``log_hyperparams`` and ``runs.csv``.

    ``section`` selects where run-specific settings come from (``stage2`` or
    ``adapt``), mirroring ``build_run_summary_rows`` so logged values match the
    actual training configuration.
    """
    stage2 = config.get("stage2") or {}
    stage = stage2 if section == "stage2" else (config.get(section) or {})

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
    else:
        from dma_kws.training.scheduler import build_optimizer_config

        opt_cfg = build_optimizer_config(stage2)
        hparams = {
            "learning_rate": opt_cfg["lr"],
            "optimizer": opt_cfg["optimizer"],
            "weight_decay": opt_cfg["weight_decay"],
            "warmup_steps": opt_cfg["warmup_steps"],
            "max_steps": int(stage2.get("max_steps", 50000)),
        }

    hparams.update(
        {
            "batch_size_per_gpu": int(setting("batch_size_per_gpu", 64)),
            "accumulate_grad_batches": int(stage2.get("accumulate_grad_batches", 1)),
            "seed": int((config.get("training") or {}).get("seed", 2025)),
            "recipe": str((config.get("training") or {}).get("recipe", "")),
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
    identity: dict[str, Any] | None = None,
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
    record["duration_s"] = round(float(duration_seconds), 1)
    for name in sorted(final_metrics):
        record[f"final/{name}"] = final_metrics[name]
    for name in sorted(best_metrics):
        record[f"best/{name}"] = best_metrics[name]
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


def update_best_metrics(best: dict[str, float], metrics: dict[str, float]) -> None:
    """Fold ``metrics`` into ``best`` in place, honoring each metric's direction."""
    for name, value in metrics.items():
        current = best.get(name)
        if current is None or (value > current if metric_direction(name) == "max" else value < current):
            best[name] = value


def build_metrics_history_callback(
    *,
    run_name: str,
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
            self.csv_path: Path | None = None
            self.best: dict[str, float] = {}
            self._fit_start: float | None = None
            self._last_row_time: float | None = None
            self._last_row_step = 0

        @property
        def duration_seconds(self) -> float:
            return 0.0 if self._fit_start is None else time.monotonic() - self._fit_start

        def on_fit_start(self, trainer, pl_module) -> None:
            log_dir = getattr(trainer.loggers[0], "log_dir", None) if trainer.loggers else None
            self.csv_path = Path(log_dir or default_dir) / filename
            self._fit_start = time.monotonic()
            self._last_row_time = self._fit_start
            self._last_row_step = int(trainer.global_step)

        def on_validation_end(self, trainer, pl_module) -> None:
            if trainer.sanity_checking or not trainer.is_global_zero:
                return
            metrics = numeric_callback_metrics(dict(trainer.callback_metrics))
            update_best_metrics(
                self.best,
                {name: value for name, value in metrics.items() if name.startswith("val")},
            )

            now = time.monotonic()
            step = int(trainer.global_step)
            interval = now - (self._last_row_time or now)
            steps_per_sec = (step - self._last_row_step) / interval if interval > 0 else 0.0
            row: dict[str, Any] = {
                "run": self.run_name,
                "step": step,
                "epoch": int(trainer.current_epoch),
                "wall_time_s": round(now - (self._fit_start or now), 3),
                "steps_per_sec": round(steps_per_sec, 4),
            }
            row.update({name: metrics[name] for name in sorted(metrics)})
            self._last_row_time = now
            self._last_row_step = step
            if self.csv_path is not None:
                append_wide_row(self.csv_path, row)

    return MetricsHistoryCallback()
