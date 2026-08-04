"""Stage I Wenet-aligned Conformer+CTC Lightning module."""

from __future__ import annotations

from typing import Any

import pytorch_lightning as pl
import torch
import torch.distributed as dist
import torchmetrics

from dma_kws.config import resolve_stream_policy
from dma_kws.metrics import collapse_ctc, edit_distance
from dma_kws.nn import build_encoder, run_encoder
from dma_kws.pathing import ensure_qbyt_on_path
from dma_kws.training.distributed_metrics import (
    gather_unique_sample_values,
    sum_across_processes,
)
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
        self.stream_policy = resolve_stream_policy(stage1)
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
        self._train_window_loss = torchmetrics.MeanMetric(sync_on_compute=True)
        self._train_window_microbatches = 0

    def forward(
        self,
        feats: torch.Tensor,
        feat_lengths: torch.Tensor,
        *,
        mode: str = "eval",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # ``mode`` defaults to "eval" so validation and inference always run at the
        # deployment operating point; only the training step opts into randomization.
        encoder_out, encoder_mask = run_encoder(
            self.encoder,
            feats,
            feat_lengths,
            policy=self.stream_policy,
            mode=mode,
        )
        encoder_lens = encoder_mask.squeeze(1).sum(dim=1).to(dtype=torch.long)
        return encoder_out, encoder_lens

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        encoder_out, encoder_lens = self(batch["feats"], batch["feat_lengths"], mode="train")
        loss, _ = self.ctc(
            encoder_out,
            encoder_lens,
            batch["targets"],
            batch["target_lengths"],
        )
        batch_size = batch["feats"].size(0)
        self._train_window_loss.update(loss.detach(), weight=batch_size)
        self._train_window_microbatches += 1
        self.log(
            "train/microbatch/loss_total",
            loss,
            on_step=True,
            prog_bar=True,
            batch_size=batch_size,
        )
        self.log(
            "train/loss",
            loss,
            on_step=True,
            prog_bar=False,
            logger=False,
            batch_size=batch_size,
        )

        optimizer = self.optimizers()
        lr = optimizer.param_groups[0]["lr"]
        self.log(
            "train/optimizer/lr",
            lr,
            on_step=True,
            prog_bar=True,
            batch_size=batch_size,
        )
        self.log(
            "train/lr",
            lr,
            on_step=True,
            prog_bar=False,
            logger=False,
            batch_size=batch_size,
        )
        return loss

    def _log_train_window(self) -> None:
        if self._train_window_microbatches <= 0:
            return
        self.log(
            "train/window/loss_total",
            self._train_window_loss.compute(),
            sync_dist=True,
        )
        self.log(
            "train/window/microbatches",
            float(self._train_window_microbatches),
            sync_dist=True,
        )
        self._train_window_loss.reset()
        self._train_window_microbatches = 0

    def on_validation_epoch_start(self) -> None:
        self._log_train_window()
        self._val_total_dist = 0
        self._val_total_ref = 0
        self._val_per_records: list[tuple[int, int, int]] = []
        self._val_synthetic_sample_index = 0

    def on_train_end(self) -> None:
        self._log_train_window()

    def _validation_sample_ids(
        self,
        batch: dict[str, torch.Tensor],
        batch_size: int,
    ) -> list[int]:
        raw_ids = batch.get("sample_id")
        if raw_ids is not None:
            ids = torch.as_tensor(raw_ids).detach().reshape(-1)
            if ids.numel() != batch_size:
                raise ValueError(
                    "sample_id must contain one value per validation sample; "
                    f"got {ids.numel()} ids for batch size {batch_size}"
                )
            return [int(value) for value in ids.cpu().tolist()]

        # Preserve compatibility with manually constructed batches.  Negative
        # ids are deliberately unique per rank and are never de-duplicated.
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        world_size = (
            dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
        )
        start = self._val_synthetic_sample_index
        self._val_synthetic_sample_index += batch_size
        return [-(1 + rank + world_size * (start + offset)) for offset in range(batch_size)]

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        if self.num_decode_batches and batch_idx >= self.num_decode_batches:
            return

        encoder_out, encoder_lens = self(batch["feats"], batch["feat_lengths"])
        log_probs = self.ctc.log_softmax(encoder_out)
        preds = log_probs.argmax(dim=2)
        targets = batch["targets"]
        target_lengths = batch["target_lengths"]
        sample_ids = self._validation_sample_ids(batch, preds.size(0))

        for i in range(preds.size(0)):
            frames = int(encoder_lens[i].item())
            raw_ids = preds[i, :frames].tolist()
            hyp_ids = collapse_ctc(raw_ids, blank_id=self.blank_id)
            ref_len = int(target_lengths[i].item())
            ref_ids = targets[i, :ref_len].tolist()
            distance = edit_distance(ref_ids, hyp_ids)
            self._val_total_dist += distance
            self._val_total_ref += ref_len
            self._val_per_records.append((sample_ids[i], distance, ref_len))

    def on_validation_epoch_end(self) -> None:
        records = getattr(self, "_val_per_records", [])
        sample_ids = torch.tensor(
            [record[0] for record in records],
            device=self.device,
            dtype=torch.long,
        )
        values = torch.tensor(
            [[record[1], record[2]] for record in records],
            device=self.device,
            dtype=torch.float64,
        ).reshape(-1, 2)
        unique_values = gather_unique_sample_values(sample_ids, values)
        if unique_values.size(0):
            totals = unique_values.sum(dim=0)
        else:
            # Compatibility for callers/tests that directly seed the legacy
            # epoch counters without running validation_step.
            totals = sum_across_processes(
                torch.tensor(
                    [self._val_total_dist, self._val_total_ref],
                    device=self.device,
                    dtype=torch.float64,
                )
            )
        total_dist, total_ref = totals.unbind()
        per = (
            total_dist / total_ref
            if total_ref.item() > 0
            else totals.new_tensor(float("nan"))
        )
        self.log("val/per_edit_distance", total_dist, sync_dist=True)
        self.log("val/per_reference_tokens", total_ref, sync_dist=True)
        self.log("val/per", per, prog_bar=True, sync_dist=True)
        # ModelCheckpoint filename placeholder cannot contain ``/``.
        self.log("val_per", per, sync_dist=True, logger=False)

    def configure_optimizers(self) -> dict | torch.optim.Optimizer:
        stage1 = self._stage1_cfg
        lr = float(stage1.get("learning_rate", 1e-3))
        warmup_steps = int(stage1.get("warmup_steps", 0))
        total_steps = int(stage1.get("total_scheduler_steps", stage1.get("max_train_steps", 0)))

        if warmup_steps > 0 and total_steps > 0:
            return build_cosine_warmup_optimizer(self, lr, warmup_steps, total_steps)
        return torch.optim.Adam(self.parameters(), lr=lr)
