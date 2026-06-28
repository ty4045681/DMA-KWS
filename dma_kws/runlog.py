"""Training-run logging helpers (loss/metrics records on disk).

Mirrors the paper's PyTorch Lightning TensorBoard setup (qbyt/train.py uses
``TensorBoardLogger(save_dir, name=run_name)``) and additionally attaches a
dependency-free ``CSVLogger`` so loss/metrics are always recorded to a plain
``metrics.csv`` even when the ``tensorboard`` package is absent.

Logs land under ``{log_dir}/{run_name}/version_N/`` (Lightning auto-increments
the version), matching the paper's ``lightning_logs/{name}/version_0`` layout.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def build_loggers(
    log_dir: str | Path,
    run_name: str,
    *,
    config: dict[str, Any] | None = None,
) -> list:
    """Return Lightning loggers for a training run.

    When ``config`` is provided, reads ``stage2.logging.backends`` to choose
    CSV, TensorBoard, W&B, and/or Trackio loggers. Defaults to
    ``[csv, tensorboard]`` (backward compatible with Stage I call sites that
    omit ``config``).

    pytorch_lightning is imported lazily so this module stays importable in
    environments without torch installed.
    """
    from pytorch_lightning.loggers import CSVLogger

    log_dir = str(log_dir)
    stage2 = (config or {}).get("stage2", {}) or {}
    logging_cfg = stage2.get("logging", {}) or {}
    backends = [str(b).lower() for b in logging_cfg.get("backends", ["csv", "tensorboard"])]

    loggers: list = []

    if "csv" in backends:
        loggers.append(CSVLogger(save_dir=log_dir, name=run_name))

    if "tensorboard" in backends:
        try:
            from pytorch_lightning.loggers import TensorBoardLogger

            loggers.append(TensorBoardLogger(save_dir=log_dir, name=run_name))
        except (ImportError, ModuleNotFoundError):
            print(
                "tensorboard not installed; skipping TensorBoard logger. "
                "Install tensorboard for TensorBoard event logs."
            )

    if "wandb" in backends:
        try:
            from pytorch_lightning.loggers import WandbLogger

            wandb_cfg = logging_cfg.get("wandb", {}) or {}
            loggers.append(
                WandbLogger(
                    save_dir=log_dir,
                    name=run_name,
                    project=str(wandb_cfg.get("project", "dma-kws")),
                    mode=str(wandb_cfg.get("mode", "online")),
                )
            )
        except (ImportError, ModuleNotFoundError):
            print("wandb not installed; skipping W&B logger. Install wandb to enable.")

    if "trackio" in backends:
        trackio_logger = _build_trackio_logger(log_dir, run_name, logging_cfg.get("trackio", {}) or {})
        if trackio_logger is not None:
            loggers.append(trackio_logger)

    if not loggers:
        loggers.append(CSVLogger(save_dir=log_dir, name=run_name))

    return loggers


def _build_trackio_logger(log_dir: str | Path, run_name: str, trackio_cfg: dict[str, Any]):
    """Return a thin Lightning logger backed by Trackio, or None if unavailable."""
    try:
        import trackio
        from pytorch_lightning.loggers import Logger
    except (ImportError, ModuleNotFoundError):
        print("trackio not installed; skipping Trackio logger. Install trackio to enable.")
        return None

    project = str(trackio_cfg.get("project", "dma-kws"))

    class TrackioLogger(Logger):
        def __init__(self) -> None:
            super().__init__()
            self._run = trackio.init(project=project, name=run_name, dir=str(log_dir))

        @property
        def name(self) -> str:
            return "trackio"

        @property
        def version(self) -> str:
            return str(getattr(self._run, "id", run_name))

        def log_hyperparams(self, params: dict[str, Any]) -> None:
            trackio.config.update(params)

        def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
            payload = dict(metrics)
            if step is not None:
                payload["step"] = step
            trackio.log(payload)

    return TrackioLogger()
