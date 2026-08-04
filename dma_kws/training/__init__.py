"""Shared training utilities for DMA-KWS."""

from dma_kws.training.adapt_params import normalize_adapt_params, resolve_adapt_lr
from dma_kws.training.ddp import build_trainer_kwargs, resolve_precision
from dma_kws.training.loaders import build_loader_kwargs
from dma_kws.training.resume import resolve_resume_path
from dma_kws.training.scheduler import build_cosine_warmup_optimizer, build_optimizer_config

__all__ = [
    "average_lightning_checkpoints",
    "build_cosine_warmup_optimizer",
    "build_loader_kwargs",
    "build_optimizer_config",
    "build_stage2_callbacks",
    "build_stage2_checkpoint_callback",
    "build_training_result_rows",
    "build_trainer_kwargs",
    "normalize_adapt_params",
    "print_run_summary",
    "print_training_result_summary",
    "resolve_adapt_lr",
    "resolve_precision",
    "resolve_resume_path",
]


def __getattr__(name: str):
    if name == "average_lightning_checkpoints":
        from dma_kws.training.checkpoint_avg import average_lightning_checkpoints

        return average_lightning_checkpoints
    if name == "build_stage2_callbacks":
        from dma_kws.training.callbacks import build_stage2_callbacks

        return build_stage2_callbacks
    if name == "print_run_summary":
        from dma_kws.training.callbacks import print_run_summary

        return print_run_summary
    if name == "build_training_result_rows":
        from dma_kws.training.callbacks import build_training_result_rows

        return build_training_result_rows
    if name == "print_training_result_summary":
        from dma_kws.training.callbacks import print_training_result_summary

        return print_training_result_summary
    if name == "build_stage2_checkpoint_callback":
        from dma_kws.training.checkpoint_callback import build_stage2_checkpoint_callback

        return build_stage2_checkpoint_callback
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
