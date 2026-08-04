"""Lightning callback that embeds canonical run identity in every checkpoint."""

from __future__ import annotations

from typing import Any

import pytorch_lightning as pl

from dma_kws.training.run_context import RunContext, stamp_run_context


class RunContextCheckpointCallback(pl.Callback):
    def __init__(self, context: RunContext) -> None:
        super().__init__()
        self.context = context

    def on_save_checkpoint(
        self,
        trainer,
        pl_module,
        checkpoint: dict[str, Any],
    ) -> None:
        stamp_run_context(checkpoint, self.context)
