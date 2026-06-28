"""Stage II Lightning callbacks and run-summary helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

def _import_weight_averaging():
    try:
        from pytorch_lightning.callbacks import WeightAveraging
    except ImportError:
        from lightning.pytorch.callbacks import WeightAveraging
    return WeightAveraging


class EMAWeightAveraging:
    """Exponential moving average via Lightning ``WeightAveraging``."""

    def __new__(cls, decay: float = 0.999, start_step: int = 0, **kwargs: Any):
        import torch
        from torch.optim.swa_utils import get_ema_avg_fn

        WeightAveraging = _import_weight_averaging()

        class _EMA(WeightAveraging):
            def __init__(self, **kw: Any) -> None:
                super().__init__(avg_fn=get_ema_avg_fn(decay=decay), **kw)
                self._start = start_step

            def should_update(self, step_idx=None, epoch_idx=None) -> bool:
                return step_idx is not None and step_idx >= self._start

        return _EMA(**kwargs)


def build_stage2_callbacks(config: dict[str, Any], recipe: str) -> list[Any]:
    """Build checkpoint, console, and optional EMA callbacks for Stage II."""
    from dma_kws.training.checkpoint_callback import build_stage2_checkpoint_callback

    stage2 = config.get("stage2", {})
    console_cfg = stage2.get("console", {}) or {}
    ema_cfg = stage2.get("ema", {}) or {}

    callbacks: list[Any] = [build_stage2_checkpoint_callback(config, recipe)]

    try:
        from pytorch_lightning.callbacks import LearningRateMonitor
    except ImportError:
        from lightning.pytorch.callbacks import LearningRateMonitor

    callbacks.append(LearningRateMonitor(logging_interval="step"))

    if console_cfg.get("device_stats", False):
        try:
            from pytorch_lightning.callbacks import DeviceStatsMonitor
        except ImportError:
            from lightning.pytorch.callbacks import DeviceStatsMonitor

        callbacks.append(DeviceStatsMonitor())

    if console_cfg.get("throughput", False):
        try:
            from pytorch_lightning.callbacks import ThroughputMonitor
        except ImportError:
            from lightning.pytorch.callbacks import ThroughputMonitor

        callbacks.append(ThroughputMonitor())

    if ema_cfg.get("enabled", False):
        callbacks.append(
            EMAWeightAveraging(
                decay=float(ema_cfg.get("decay", 0.999)),
                start_step=int(ema_cfg.get("start_step", 0)),
            )
        )

    if console_cfg.get("rich", True):
        try:
            from pytorch_lightning.callbacks import RichModelSummary, RichProgressBar
            from pytorch_lightning.callbacks.progress.rich_progress import RichProgressBarTheme
        except ImportError:
            try:
                from lightning.pytorch.callbacks import RichModelSummary, RichProgressBar
                from lightning.pytorch.callbacks.progress.rich_progress import RichProgressBarTheme
            except ImportError:
                RichProgressBar = None  # type: ignore[misc, assignment]

        if RichProgressBar is not None:
            callbacks.extend(
                [
                    RichProgressBar(
                        theme=RichProgressBarTheme(
                            metrics_format=".4f",
                            metrics_text_delimiter=" | ",
                        )
                    ),
                    RichModelSummary(max_depth=2),
                ]
            )
        else:
            print("rich not installed; using default TQDM progress bar.")
            try:
                from pytorch_lightning.callbacks import TQDMProgressBar
            except ImportError:
                from lightning.pytorch.callbacks import TQDMProgressBar

            callbacks.append(TQDMProgressBar())

    return callbacks


def print_run_summary(
    *,
    config: dict[str, Any],
    devices: int,
    accelerator: str,
    train_samples: int,
    val_samples: int,
    param_counts: dict[str, int] | None = None,
    paths: dict[str, str | Path] | None = None,
) -> None:
    """Print a one-screen summary of the Stage II training run."""
    from dma_kws.training.ddp import resolve_precision
    from dma_kws.training.scheduler import build_optimizer_config

    stage2 = config.get("stage2", {})
    dataloader_cfg = stage2.get("dataloader", {}) or {}
    ema_cfg = stage2.get("ema", {}) or {}
    opt_cfg = build_optimizer_config(stage2)

    batch_size = int(stage2.get("batch_size_per_gpu", 64))
    accumulate = int(stage2.get("accumulate_grad_batches", 1))
    effective_batch = batch_size * devices * accumulate
    num_workers = int(stage2.get("num_workers", 0))
    precision = resolve_precision(stage2, accelerator)

    rows = [
        ("recipe", str(config.get("training", {}).get("recipe", ""))),
        ("accelerator", accelerator),
        ("devices", str(devices)),
        ("precision", precision),
        ("effective_batch", str(effective_batch)),
        ("batch_size_per_gpu", str(batch_size)),
        ("accumulate_grad_batches", str(accumulate)),
        ("max_steps", str(stage2.get("max_steps", 50000))),
        ("learning_rate", str(opt_cfg["lr"])),
        ("warmup_steps", str(opt_cfg["warmup_steps"])),
        ("optimizer", opt_cfg["optimizer"]),
        ("weight_decay", str(opt_cfg["weight_decay"])),
        ("ema", "enabled" if ema_cfg.get("enabled", False) else "disabled"),
        ("train_samples", str(train_samples)),
        ("val_samples", str(val_samples)),
        ("num_workers", str(num_workers)),
        ("pin_memory", str(dataloader_cfg.get("pin_memory", True))),
    ]
    if num_workers > 0:
        rows.extend(
            [
                ("persistent_workers", str(dataloader_cfg.get("persistent_workers", True))),
                ("prefetch_factor", str(dataloader_cfg.get("prefetch_factor", 4))),
            ]
        )
    if param_counts:
        for name, count in param_counts.items():
            rows.append((f"params/{name}", f"{count:,}"))
    if paths:
        for name, path in paths.items():
            rows.append((name, str(path)))

    try:
        from rich.console import Console
        from rich.table import Table

        table = Table(title="Stage II QbyT Training Run", show_header=True, header_style="bold")
        table.add_column("Setting", style="cyan")
        table.add_column("Value")
        for key, value in rows:
            table.add_row(key, value)
        Console().print(table)
    except ImportError:
        print("=== Stage II QbyT Training Run ===")
        for key, value in rows:
            print(f"  {key}: {value}")
