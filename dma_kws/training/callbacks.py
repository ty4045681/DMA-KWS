"""Stage II Lightning callbacks and run-summary helpers."""

from __future__ import annotations

from collections.abc import Mapping
import math
from pathlib import Path
import sys
from typing import TYPE_CHECKING, Any, TextIO

if TYPE_CHECKING:
    from dma_kws.training.run_context import RunContext


def _stage2_batch_size(batch: Any) -> int:
    """Return the local sample count for a Stage II train/eval batch.

    Keeping this as a module-level callable makes the throughput callback safe
    to serialize under spawn-based distributed launchers. ``label`` is the
    canonical per-sample field; the feature/anchor fallbacks keep the callback
    usable with diagnostic loaders that omit labels.
    """
    if isinstance(batch, Mapping):
        for key in ("label", "feat", "feats", "anchor", "targets"):
            value = batch.get(key)
            if value is None:
                continue
            try:
                return int(len(value))
            except TypeError:
                continue
        keys = ", ".join(sorted(str(key) for key in batch))
        raise ValueError(
            "Cannot infer training batch size: expected a sized 'label', "
            f"'feat', 'feats', 'anchor', or 'targets' field; available fields: [{keys}]"
        )
    raise TypeError(
        "Cannot infer training batch size: expected a mapping with a "
        "'label', 'feat', 'feats', 'anchor', or 'targets' field"
    )


def _plain_console_callbacks(
    *,
    refresh_rate: int,
    leave: bool,
    max_depth: int = 2,
) -> list[Any]:
    """Build an explicit curated non-Rich progress bar and model summary."""
    try:
        from pytorch_lightning.callbacks import ModelSummary
    except ImportError:
        from lightning.pytorch.callbacks import ModelSummary

    from dma_kws.training.progress import CuratedTQDMProgressBar

    return [
        CuratedTQDMProgressBar(refresh_rate=refresh_rate, leave=leave),
        ModelSummary(max_depth=max_depth),
    ]


def build_console_callbacks(
    config: dict[str, Any],
    *,
    section: str,
) -> list[Any]:
    """Build one consistent console/system-metrics callback set per stage."""
    selected = config.get(section, {}) or {}
    trainer_stage = (
        (config.get("stage2", {}) or {}) if section == "adapt" else selected
    )
    console_cfg = selected.get("console", {}) or {}
    configured_refresh = console_cfg.get("refresh_rate")
    refresh_rate = max(
        1,
        int(
            configured_refresh
            if configured_refresh is not None
            else trainer_stage.get("log_interval", 10)
        ),
    )
    callbacks: list[Any] = []
    leave = bool(console_cfg.get("leave", True))

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

        callbacks.append(ThroughputMonitor(batch_size_fn=_stage2_batch_size))

    if console_cfg.get("rich", True):
        try:
            from pytorch_lightning.callbacks import RichModelSummary
            from dma_kws.training.progress import CuratedRichProgressBar
            rich_callbacks = [
                CuratedRichProgressBar(
                    refresh_rate=refresh_rate,
                    leave=leave,
                ),
                RichModelSummary(max_depth=2),
            ]
        except (ImportError, ModuleNotFoundError):
            callbacks.extend(
                _plain_console_callbacks(
                    refresh_rate=refresh_rate,
                    leave=leave,
                    max_depth=2,
                )
            )
        else:
            callbacks.extend(rich_callbacks)
    else:
        callbacks.extend(
            _plain_console_callbacks(
                refresh_rate=refresh_rate,
                leave=leave,
                max_depth=2,
            )
        )

    return callbacks


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


def build_stage2_callbacks(
    config: dict[str, Any],
    recipe: str,
    *,
    checkpoint_dir: str | Path | None = None,
    val_check_interval: int | None = None,
    monitor_override: str | None | object = None,
    filename_override: str | None = None,
    section: str = "stage2",
) -> list[Any]:
    """Build checkpoint, console, and optional EMA callbacks for Stage II."""
    from dma_kws.training.checkpoint_callback import build_stage2_checkpoint_callback

    stage2 = config.get("stage2", {})
    ema_cfg = stage2.get("ema", {}) or {}

    # No LearningRateMonitor: the modules already log ``train/lr`` each step,
    # so the monitor's ``lr-Adam`` column would duplicate it.
    checkpoint_kwargs: dict[str, Any] = {
        "checkpoint_dir": checkpoint_dir,
        "val_check_interval": val_check_interval,
        "filename_override": filename_override,
    }
    if monitor_override is not None:
        checkpoint_kwargs["monitor_override"] = monitor_override
    callbacks: list[Any] = [
        build_stage2_checkpoint_callback(config, recipe, **checkpoint_kwargs)
    ]

    if ema_cfg.get("enabled", False):
        callbacks.append(
            EMAWeightAveraging(
                decay=float(ema_cfg.get("decay", 0.999)),
                start_step=int(ema_cfg.get("start_step", 0)),
            )
        )

    callbacks.extend(build_console_callbacks(config, section=section))

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

    callbacks.extend(build_console_callbacks(config, section="stage1"))

    return callbacks, checkpoint_callback


RUN_SUMMARY_TITLES = {
    "stage1": "Stage I Phoneme CTC Training Run",
    "phoneme_adapter": "Phoneme Adapter Training Run",
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
    effective_max_steps: int | None = None,
    extra_rows: list[tuple[str, str]] | None = None,
    effective_logging_backends: list[str] | None = None,
) -> list[tuple[str, str]]:
    """Build ``(setting, value)`` rows describing a training run.

    ``section`` selects where data/optimizer/trainer settings come from. Keyword
    adaptation is the one exception: its optimizer/data values live in
    ``adapt``, while shared Trainer precision/accumulation live in ``stage2``.
    """
    from dma_kws.training.adapt_params import resolve_adapt_lr
    from dma_kws.training.ddp import resolve_precision
    from dma_kws.training.scheduler import build_optimizer_config

    stage2 = config.get("stage2", {}) or {}
    stage = stage2 if section == "stage2" else (config.get(section) or {})
    trainer_stage = stage2 if section == "adapt" else stage
    dataloader_cfg = trainer_stage.get("dataloader", {}) or {}
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
    elif section == "stage1":
        warmup_steps = int(stage.get("warmup_steps", 0))
        scheduler_steps = int(
            stage.get("total_scheduler_steps", stage.get("max_train_steps", 0))
        )
        scheduler_enabled = warmup_steps > 0 and scheduler_steps > 0
        opt_cfg = {
            "optimizer": "adam",
            "lr": float(stage.get("learning_rate", 1e-3)),
            "weight_decay": 0.0,
            "warmup_steps": warmup_steps,
            "total_steps": scheduler_steps if scheduler_enabled else -1,
        }
        lr_label = "learning_rate"
    else:
        opt_cfg = build_optimizer_config(stage)
        lr_label = "learning_rate"

    batch_size = int(setting("batch_size_per_gpu", 64))
    accumulate = int(trainer_stage.get("accumulate_grad_batches", 1))
    effective_batch = batch_size * devices * accumulate
    num_workers = int(setting("num_workers", 0))
    max_steps = (
        int(effective_max_steps)
        if effective_max_steps is not None
        else int(setting("max_steps", 50000))
    )

    rows = [
        ("recipe", str(config.get("training", {}).get("recipe", ""))),
        ("seed", str(config.get("training", {}).get("seed", 2025))),
        ("accelerator", accelerator),
        ("devices", str(devices)),
        ("precision", resolve_precision(trainer_stage, accelerator)),
        ("effective_batch", str(effective_batch)),
        ("batch_size_per_gpu", str(batch_size)),
        ("accumulate_grad_batches", str(accumulate)),
        ("max_steps", str(max_steps)),
        (lr_label, str(opt_cfg["lr"])),
        ("warmup_steps", str(opt_cfg["warmup_steps"])),
        ("optimizer", opt_cfg["optimizer"]),
        ("weight_decay", str(opt_cfg["weight_decay"])),
    ]
    if section == "adapt":
        from dma_kws.stage2.adapt_config import resolve_adapt_method

        method = resolve_adapt_method(stage)
        rows.append(("adapt_method", method))
        if method == "encoder_qbyt_full":
            from dma_kws.training.adapt_params import resolve_encoder_adapt_lr

            rows.append(("encoder_learning_rate", str(resolve_encoder_adapt_lr(stage))))
    if section == "stage1":
        max_epochs = int(stage.get("max_epochs", 1))
        rows.append(("max_epochs", str(max_epochs)))
        rows.append(
            (
                "stop_condition",
                f"max_epochs={max_epochs}" if max_steps < 0 else f"max_steps={max_steps}",
            )
        )
    if int(opt_cfg["total_steps"]) > 0 and int(opt_cfg["total_steps"]) != max_steps:
        rows.append(("scheduler_total_steps", str(opt_cfg["total_steps"])))
    if section in {"stage2", "adapt"}:
        rows.append(
            ("ema", "enabled" if ema_cfg.get("enabled", False) else "disabled")
        )
    rows.extend(
        [
            ("train_samples", str(train_samples)),
            ("val_samples", str(val_samples)),
            ("num_workers", str(num_workers)),
            (
                "pin_memory",
                str(
                    dataloader_cfg.get(
                        "pin_memory", False if section == "stage1" else True
                    )
                ),
            ),
        ]
    )
    validation_cfg = stage.get("validation", {}) or {}
    val_interval = validation_cfg.get(
        "val_check_interval",
        stage.get("val_check_interval", "epoch"),
    )
    checkpoint_cfg = (
        stage2.get("checkpoint", {}) or {}
        if section in {"stage2", "adapt"}
        else stage.get("checkpoint", {}) or {}
    )
    logging_cfg = stage.get("logging", {}) or {}
    rows.extend(
        [
            (
                "log_interval",
                str((stage2 if section == "adapt" else stage).get("log_interval", 10)),
            ),
            (
                "val_check_interval",
                (
                    f"{val_interval} train batches/rank "
                    f"(~{math.ceil(int(val_interval) / max(accumulate, 1))} optimizer steps)"
                    if isinstance(val_interval, int)
                    and not isinstance(val_interval, bool)
                    else str(val_interval)
                ),
            ),
            (
                "checkpoint_monitor",
                str(
                    stage.get("checkpoint_monitor")
                    if section == "adapt"
                    else (
                        (
                            "disabled (no validation)"
                            if section == "stage1" and val_samples == 0
                            else "val/per"
                        )
                        if section in {"stage1", "phoneme_adapter"}
                        else checkpoint_cfg.get("monitor", "val_auc")
                    )
                ),
            ),
            (
                "logging_backends",
                ", ".join(effective_logging_backends)
                if effective_logging_backends is not None
                else ", ".join(
                    str(item)
                    for item in logging_cfg.get("backends", ["csv", "tensorboard"])
                ),
            ),
        ]
    )
    if section in {"stage2", "adapt"}:
        from dma_kws.stage2.readout import resolve_qbyt_score_spec

        sequence_cfg = stage2.get("sequence_loss", {}) or {}
        adapter_cfg = stage2.get("phoneme_adapter", {}) or {}
        validation_diagnostics_cfg = stage2.get("validation", {}) or {}
        score = resolve_qbyt_score_spec(stage2)
        rows.append(("qbyt_readout_version", str(score.version)))
        if score.family == "pooling":
            rows.extend(
                [
                    ("qbyt_readout_mode", score.value.mode),
                    ("qbyt_readout_temperature", str(score.value.temperature)),
                ]
            )
        else:
            alignment = score.value
            rows.extend(
                [
                    ("qbyt_alignment_topology", alignment.topology),
                    (
                        "qbyt_weakest_phone_temperature",
                        str(getattr(alignment, "weakest_phone_temperature", "")),
                    ),
                    (
                        "qbyt_weakest_phone_weight",
                        str(getattr(alignment, "weakest_phone_weight", "")),
                    ),
                    (
                        "qbyt_min_phone_duration_frames",
                        str(alignment.min_phone_duration_frames),
                    ),
                    (
                        "qbyt_max_phone_duration_frames",
                        str(alignment.max_phone_duration_frames),
                    ),
                    (
                        "qbyt_max_inter_phone_gap_frames",
                        str(alignment.max_inter_phone_gap_frames),
                    ),
                    (
                        "qbyt_max_keyword_span_frames",
                        str(alignment.max_keyword_span_frames),
                    ),
                    ("qbyt_local_context_kernel", str(alignment.local_context_kernel)),
                ]
            )
        rows.extend(
            [
                (
                    "qbyt_deployment_threshold",
                    str(
                        float(
                            ((config.get("demo") or {}).get(
                                "qbyt_threshold", 0.5
                            ))
                        )
                    ),
                ),
                (
                    "score_ece_num_bins",
                    str(int(validation_diagnostics_cfg.get("ece_num_bins", 15))),
                ),
                (
                    "sequence_objective",
                    "target={target} progress={progress:g} "
                    "normalization={normalization}".format(
                        target=sequence_cfg.get(
                            "target_mode", "ordered_contiguous_prefix"
                        ),
                        progress=float(sequence_cfg.get("progress_weight", 0.3)),
                        normalization=sequence_cfg.get("normalization", "sample"),
                    ),
                ),
                (
                    "phoneme_adapter",
                    "enabled={enabled} freeze={freeze} ctc_weight={ctc:g}".format(
                        enabled=bool(adapter_cfg.get("enabled", False)),
                        freeze=True if section == "adapt" else bool(adapter_cfg.get("freeze", False)),
                        ctc=0.0 if section == "adapt" else float(adapter_cfg.get("ctc_weight", 0.0)),
                    ),
                ),
            ]
        )
    if num_workers > 0 and section != "stage1":
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
    effective_max_steps: int | None = None,
    title: str | None = None,
    extra_rows: list[tuple[str, str]] | None = None,
    effective_logging_backends: list[str] | None = None,
) -> None:
    """Print a one-screen summary of a training run."""
    from dma_kws.training.ddp import process_rank

    if process_rank() != 0:
        return
    rows = build_run_summary_rows(
        config=config,
        devices=devices,
        accelerator=accelerator,
        train_samples=train_samples,
        val_samples=val_samples,
        param_counts=param_counts,
        paths=paths,
        section=section,
        effective_max_steps=effective_max_steps,
        extra_rows=extra_rows,
        effective_logging_backends=effective_logging_backends,
    )
    heading = title or RUN_SUMMARY_TITLES.get(section, f"{section} Training Run")
    if title is None and section == "adapt":
        from dma_kws.stage2.adapt_config import adapt_method_label, resolve_adapt_method

        method = resolve_adapt_method(config.get("adapt") or {})
        heading = f"Stage II {adapt_method_label(method)} Adaptation Run"

    selected = config.get(section, {}) or {}
    console_cfg = selected.get("console", {}) or {}
    use_rich = bool(console_cfg.get("rich", True)) and sys.stdout.isatty()

    try:
        if not use_rich:
            raise ImportError
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


def _summary_value(value: Any, *, missing: str = "(not available)") -> str:
    """Convert scalar/path-like summary values without importing torch."""
    if value is None or (isinstance(value, str) and not value):
        return missing
    try:
        value = value.item()
    except (AttributeError, TypeError, ValueError):
        pass
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _include_result_metric(name: str) -> bool:
    """Keep the final console summary useful without dumping every diagnostic."""
    if not name.startswith("val/"):
        return False
    leaf = name.removeprefix("val/")
    return leaf.endswith(
        (
            "/per",
            "_utt_loss",
            "ctc_skip_rate",
            "auc",
            "eer",
            "eer_threshold",
            "deploy_tpr",
            "deploy_fpr",
            "tpr_at_fpr_1e_3",
            "score_neg_p95",
        )
    ) or leaf in {"per", "loss"}


def build_training_result_rows(
    *,
    run_context: RunContext,
    global_step: int,
    last_validation_step: int | None,
    best_checkpoint_monitor: str | None = None,
    best_checkpoint_path: str | Path | None = None,
    best_checkpoint_score: Any = None,
    final_metrics: Mapping[str, Any] | None = None,
    artifact_paths: Mapping[str, str | Path | None] | None = None,
    metrics_source: str = "last_trainer_state",
    artifact_sources: Mapping[str, str] | None = None,
) -> list[tuple[str, str]]:
    """Build stable rows for the final, post-fit training summary.

    ``validation_staleness_steps`` is the optimizer-step distance between the
    end of training and the last completed validation.  It makes it explicit
    when the best/last validation metrics do not describe the final weights.
    """
    final_step = int(global_step)
    if last_validation_step is None:
        validation_step = "(not validated)"
        staleness = "(not available)"
    else:
        completed_validation_step = int(last_validation_step)
        validation_step = str(completed_validation_step)
        staleness = str(max(0, final_step - completed_validation_step))

    rows = [
        ("run_id", run_context.run_id),
        ("run_dir", str(run_context.run_dir)),
        ("global_step", str(final_step)),
        ("effective_max_steps", str(run_context.effective_max_steps)),
        ("last_validation_step", validation_step),
        ("validation_staleness_steps", staleness),
        ("metrics_source", metrics_source),
        (
            "best_checkpoint_monitor",
            _summary_value(best_checkpoint_monitor),
        ),
        (
            "best_checkpoint_path",
            _summary_value(best_checkpoint_path),
        ),
        (
            "best_checkpoint_score",
            _summary_value(best_checkpoint_score),
        ),
    ]
    for name in sorted(final_metrics or {}):
        if _include_result_metric(name):
            rows.append((f"metric/{name}", _summary_value(final_metrics[name])))
    for name, path in (artifact_paths or {}).items():
        rows.append(
            (
                f"artifact/{name}",
                _summary_value(path, missing="(not produced)"),
            )
        )
        if artifact_sources and name in artifact_sources:
            rows.append((f"artifact_source/{name}", artifact_sources[name]))
    return rows


def print_training_result_summary(
    *,
    run_context: RunContext,
    global_step: int,
    last_validation_step: int | None,
    best_checkpoint_monitor: str | None = None,
    best_checkpoint_path: str | Path | None = None,
    best_checkpoint_score: Any = None,
    final_metrics: Mapping[str, Any] | None = None,
    artifact_paths: Mapping[str, str | Path | None] | None = None,
    metrics_source: str = "last_trainer_state",
    artifact_sources: Mapping[str, str] | None = None,
    title: str = "Training Result",
    rich: bool = True,
    stream: TextIO | None = None,
) -> None:
    """Print the final training result once, on rank zero.

    Rich output is used only for a TTY and falls back to deterministic plain
    text when Rich is unavailable or output is redirected to a log file.
    """
    from dma_kws.training.ddp import process_rank

    if process_rank() != 0:
        return

    rows = build_training_result_rows(
        run_context=run_context,
        global_step=global_step,
        last_validation_step=last_validation_step,
        best_checkpoint_monitor=best_checkpoint_monitor,
        best_checkpoint_path=best_checkpoint_path,
        best_checkpoint_score=best_checkpoint_score,
        final_metrics=final_metrics,
        artifact_paths=artifact_paths,
        metrics_source=metrics_source,
        artifact_sources=artifact_sources,
    )
    output = stream or sys.stdout
    is_tty = bool(getattr(output, "isatty", lambda: False)())
    use_rich = bool(rich) and is_tty

    try:
        if not use_rich:
            raise ImportError
        from rich.console import Console
        from rich.table import Table

        table = Table(title=title, show_header=True, header_style="bold")
        table.add_column("Result", style="cyan")
        table.add_column("Value")
        for key, value in rows:
            table.add_row(key, value)
        Console(file=output).print(table)
    except ImportError:
        print(f"=== {title} ===", file=output)
        for key, value in rows:
            print(f"  {key}: {value}", file=output)
