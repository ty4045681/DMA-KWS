"""Curated Lightning progress bars backed by the canonical metric names."""

from __future__ import annotations

from typing import Any

from pytorch_lightning.callbacks import RichProgressBar, TQDMProgressBar
from pytorch_lightning.callbacks.progress.rich_progress import RichProgressBarTheme


_DISPLAY_NAMES = {
    "train/microbatch/loss_total": "loss",
    "train/microbatch/loss_utt_raw": "utt",
    "train/microbatch/loss_seq_weighted": "seq",
    "train/microbatch/loss_ctc_weighted": "ctc",
    "train/microbatch/ctc_skip_rate": "skip",
    "train/optimizer/lr": "lr",
    "val/utt_loss": "v_loss",
    "val/auc": "v_auc",
    "val/eer": "v_eer",
    "val/eer_threshold": "v_thr",
    "val/target_utt_loss": "t_loss",
    "val/target_auc": "t_auc",
    "val/target_eer": "t_eer",
    "val/target_eer_threshold": "t_thr",
    "val/lph_utt_loss": "lph_loss",
    "val/lph_auc": "lph_auc",
    "val/lph_eer": "lph_eer",
    "val/lph_eer_threshold": "lph_thr",
    "val/loss": "v_loss",
    "val/per": "v_per",
}


def _curate(metrics: dict[str, Any]) -> dict[str, Any]:
    curated: dict[str, Any] = {}
    if "v_num" in metrics:
        curated["v_num"] = metrics["v_num"]
    for source, display in _DISPLAY_NAMES.items():
        if source in metrics:
            curated[display] = metrics[source]
    return curated


class CuratedRichProgressBar(RichProgressBar):
    def __init__(self, *, refresh_rate: int, leave: bool) -> None:
        super().__init__(
            refresh_rate=refresh_rate,
            leave=leave,
            theme=RichProgressBarTheme(
                metrics_format=".4f",
                metrics_text_delimiter=" | ",
            ),
        )

    def get_metrics(self, trainer, pl_module) -> dict[str, Any]:
        return _curate(super().get_metrics(trainer, pl_module))


class CuratedTQDMProgressBar(TQDMProgressBar):
    def __init__(self, *, refresh_rate: int, leave: bool = False) -> None:
        super().__init__(refresh_rate=refresh_rate, leave=leave)

    def get_metrics(self, trainer, pl_module) -> dict[str, Any]:
        return _curate(super().get_metrics(trainer, pl_module))
