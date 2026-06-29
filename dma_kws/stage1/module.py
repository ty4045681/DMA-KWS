"""Stage I Wenet-aligned Conformer+CTC Lightning module."""

from __future__ import annotations

from typing import Any

import pytorch_lightning as pl
import torch

from dma_kws.metrics import collapse_ctc, edit_distance
from dma_kws.nn import build_encoder
from dma_kws.pathing import ensure_qbyt_on_path
from dma_kws.training.scheduler import build_cosine_warmup_optimizer

BLANK_ID = 0


def _load_ctc():
    ensure_qbyt_on_path()
    from models.ctc import CTC

    return CTC


class Stage1LightningModule(pl.LightningModule):
    """Lightning wrapper for ConformerEncoder + CTC Stage I training."""

    def __init__(
        self,
        config: dict[str, Any],
        *,
        vocab_size: int,
        blank_id: int = BLANK_ID,
        num_decode_batches: int = 0,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(
            {
                "vocab_size": vocab_size,
                "blank_id": blank_id,
                "num_decode_batches": num_decode_batches,
            }
        )

        stage1 = config["stage1"]
        encoder_dim = int(stage1.get("encoder_output_dim", 144))
        self.encoder = build_encoder(stage1, output_dim=encoder_dim)

        CTC = _load_ctc()
        self.ctc = CTC(
            odim=vocab_size,
            encoder_output_size=encoder_dim,
            dropout_rate=float(stage1.get("ctc_dropout", 0.0)),
            blank_id=blank_id,
        )

        self._stage1_cfg = stage1
        self.blank_id = blank_id
        self.num_decode_batches = num_decode_batches

    def forward(
        self,
        feats: torch.Tensor,
        feat_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        encoder_out, encoder_mask = self.encoder(feats, feat_lengths)
        encoder_lens = encoder_mask.squeeze(1).sum(dim=1).to(dtype=torch.long)
        return encoder_out, encoder_lens

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        encoder_out, encoder_lens = self(batch["feats"], batch["feat_lengths"])
        loss, _ = self.ctc(
            encoder_out,
            encoder_lens,
            batch["targets"],
            batch["target_lengths"],
        )
        batch_size = batch["feats"].size(0)
        self.log("train/loss", loss, on_step=True, prog_bar=True, batch_size=batch_size)

        optimizer = self.optimizers()
        lr = optimizer.param_groups[0]["lr"]
        self.log("train/lr", lr, on_step=True, prog_bar=True, batch_size=batch_size)
        return loss

    def on_validation_epoch_start(self) -> None:
        self._val_total_dist = 0
        self._val_total_ref = 0

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        if self.num_decode_batches and batch_idx >= self.num_decode_batches:
            return

        encoder_out, encoder_lens = self(batch["feats"], batch["feat_lengths"])
        log_probs = self.ctc.log_softmax(encoder_out)
        preds = log_probs.argmax(dim=2)
        targets = batch["targets"]
        target_lengths = batch["target_lengths"]

        for i in range(preds.size(0)):
            frames = int(encoder_lens[i].item())
            raw_ids = preds[i, :frames].tolist()
            hyp_ids = collapse_ctc(raw_ids, blank_id=self.blank_id)
            ref_len = int(target_lengths[i].item())
            ref_ids = targets[i, :ref_len].tolist()
            self._val_total_dist += edit_distance(ref_ids, hyp_ids)
            self._val_total_ref += ref_len

    def on_validation_epoch_end(self) -> None:
        if self._val_total_ref > 0:
            per = self._val_total_dist / self._val_total_ref
            self.log("val/per", per, prog_bar=True)

    def configure_optimizers(self) -> dict | torch.optim.Optimizer:
        stage1 = self._stage1_cfg
        lr = float(stage1.get("learning_rate", 1e-3))
        warmup_steps = int(stage1.get("warmup_steps", 0))
        total_steps = int(stage1.get("total_scheduler_steps", stage1.get("max_train_steps", 0)))

        if warmup_steps > 0 and total_steps > 0:
            return build_cosine_warmup_optimizer(self, lr, warmup_steps, total_steps)
        return torch.optim.Adam(self.parameters(), lr=lr)
