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

    # No LearningRateMonitor: the modules already log ``train/lr`` each step,
    # so the monitor's ``lr-Adam`` column would duplicate it.
    callbacks: list[Any] = [build_stage2_checkpoint_callback(config, recipe)]

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


def build_stage1_callbacks(
    config: dict[str, Any],
    *,
    checkpoint_dir: Path,
    has_validation: bool,
) -> tuple[list[Any], Any | None]:
    """Build checkpoint callbacks for Stage I training."""
    from pytorch_lightning.callbacks import ModelCheckpoint

    stage1 = config.get("stage1", {})
    validation_cfg = stage1.get("validation", {}) or {}
    callbacks: list[Any] = []
    checkpoint_callback = None

    if has_validation:
        avg_cfg = stage1.get("checkpoint_avg", {}) or {}
        save_all = bool(avg_cfg.get("enabled", False))
        checkpoint_callback = ModelCheckpoint(
            dirpath=str(checkpoint_dir),
            monitor="val/per",
            mode="min",
            save_top_k=1 if not save_all else -1,
            filename="stage1_{epoch:03d}_{val_per:.4f}",
            save_last=True,
        )
        callbacks.append(checkpoint_callback)
    else:
        callbacks.append(
            ModelCheckpoint(
                dirpath=str(checkpoint_dir),
                save_last=True,
                save_top_k=0,
            )
        )

    return callbacks, checkpoint_callback


RUN_SUMMARY_TITLES = {
    "stage2": "Stage II QbyT Training Run",
    "adapt": "Stage II LoRA Adaptation Run",
}


def build_run_summary_rows(
    *,
    config: dict[str, Any],
    devices: int,
    accelerator: str,
    train_samples: int,
    val_samples: int,
    param_counts: dict[str, int] | None = None,
    paths: dict[str, str | Path] | None = None,
    section: str = "stage2",
    extra_rows: list[tuple[str, str]] | None = None,
) -> list[tuple[str, str]]:
    """Build ``(setting, value)`` rows describing a training run.

    ``section`` selects where hyperparameters come from. Trainer-level settings
    always come from ``stage2`` because ``build_trainer_kwargs`` and
    ``build_stage2_callbacks`` read that section regardless of the recipe.
    """
    from dma_kws.training.adapt_params import resolve_adapt_lr
    from dma_kws.training.ddp import resolve_precision
    from dma_kws.training.scheduler import build_optimizer_config

    stage2 = config.get("stage2", {}) or {}
    stage = stage2 if section == "stage2" else (config.get(section) or {})
    dataloader_cfg = stage2.get("dataloader", {}) or {}
    ema_cfg = stage2.get("ema", {}) or {}

    def setting(key: str, default: Any) -> Any:
        """Read from the run's section, falling back to ``stage2``."""
        value = stage.get(key)
        return stage2.get(key, default) if value is None else value

    if section == "adapt":
        lr, lr_source = resolve_adapt_lr(stage)
        opt_cfg = {
            "optimizer": str(stage.get("optimizer", "adam")).lower(),
            "lr": lr,
            "weight_decay": float(stage.get("weight_decay", 0.0)),
            "warmup_steps": int(stage.get("warmup_steps", 100)),
            "total_steps": int(stage.get("max_steps", 3000)),
        }
        lr_label = "learning_rate" if lr_source == "learning_rate" else f"learning_rate ({lr_source})"
    else:
        opt_cfg = build_optimizer_config(stage)
        lr_label = "learning_rate"

    batch_size = int(setting("batch_size_per_gpu", 64))
    accumulate = int(stage2.get("accumulate_grad_batches", 1))
    effective_batch = batch_size * devices * accumulate
    num_workers = int(setting("num_workers", 0))
    max_steps = int(setting("max_steps", 50000))

    rows = [
        ("recipe", str(config.get("training", {}).get("recipe", ""))),
        ("accelerator", accelerator),
        ("devices", str(devices)),
        ("precision", resolve_precision(stage2, accelerator)),
        ("effective_batch", str(effective_batch)),
        ("batch_size_per_gpu", str(batch_size)),
        ("accumulate_grad_batches", str(accumulate)),
        ("max_steps", str(max_steps)),
        (lr_label, str(opt_cfg["lr"])),
        ("warmup_steps", str(opt_cfg["warmup_steps"])),
        ("optimizer", opt_cfg["optimizer"]),
        ("weight_decay", str(opt_cfg["weight_decay"])),
    ]
    if int(opt_cfg["total_steps"]) != max_steps:
        rows.append(("scheduler_total_steps", str(opt_cfg["total_steps"])))
    rows.extend(
        [
            ("ema", "enabled" if ema_cfg.get("enabled", False) else "disabled"),
            ("train_samples", str(train_samples)),
            ("val_samples", str(val_samples)),
            ("num_workers", str(num_workers)),
            ("pin_memory", str(dataloader_cfg.get("pin_memory", True))),
        ]
    )
    if num_workers > 0:
        rows.extend(
            [
                ("persistent_workers", str(dataloader_cfg.get("persistent_workers", True))),
                ("prefetch_factor", str(dataloader_cfg.get("prefetch_factor", 4))),
            ]
        )
    if extra_rows:
        rows.extend(extra_rows)
    if param_counts:
        for name, count in param_counts.items():
            rows.append((f"params/{name}", f"{count:,}" if isinstance(count, int) else str(count)))
    if paths:
        for name, path in paths.items():
            rows.append((name, str(path)))
    return rows


def print_run_summary(
    *,
    config: dict[str, Any],
    devices: int,
    accelerator: str,
    train_samples: int,
    val_samples: int,
    param_counts: dict[str, int] | None = None,
    paths: dict[str, str | Path] | None = None,
    section: str = "stage2",
    title: str | None = None,
    extra_rows: list[tuple[str, str]] | None = None,
) -> None:
    """Print a one-screen summary of a training run."""
    rows = build_run_summary_rows(
        config=config,
        devices=devices,
        accelerator=accelerator,
        train_samples=train_samples,
        val_samples=val_samples,
        param_counts=param_counts,
        paths=paths,
        section=section,
        extra_rows=extra_rows,
    )
    heading = title or RUN_SUMMARY_TITLES.get(section, f"{section} Training Run")

    try:
        from rich.console import Console
        from rich.table import Table

        table = Table(title=heading, show_header=True, header_style="bold")
        table.add_column("Setting", style="cyan")
        table.add_column("Value")
        for key, value in rows:
            table.add_row(key, value)
        Console().print(table)
    except ImportError:
        print(f"=== {heading} ===")
        for key, value in rows:
            print(f"  {key}: {value}")
