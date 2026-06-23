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


def build_loggers(log_dir: str | Path, run_name: str) -> list:
    """Return Lightning loggers for a training run.

    Always includes a ``CSVLogger`` (no extra dependency) that writes
    ``metrics.csv`` + ``hparams.yaml``. Adds a ``TensorBoardLogger`` mirroring
    the paper when the ``tensorboard`` package is installed; if it is missing we
    skip TensorBoard rather than crash, so loss recording still works.

    pytorch_lightning is imported lazily so this module stays importable (and
    unit-testable) in environments without torch installed.
    """
    from pytorch_lightning.loggers import CSVLogger

    log_dir = str(log_dir)
    loggers: list = [CSVLogger(save_dir=log_dir, name=run_name)]
    try:
        from pytorch_lightning.loggers import TensorBoardLogger

        loggers.append(TensorBoardLogger(save_dir=log_dir, name=run_name))
    except (ImportError, ModuleNotFoundError):
        print(
            "tensorboard not installed; recording loss/metrics to CSV only. "
            "Install tensorboard for TensorBoard event logs."
        )
    return loggers
