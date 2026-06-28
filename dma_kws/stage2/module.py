"""Stage II QbyT Lightning module aligned with main ``qbyt/train.py`` Wrapper."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchmetrics

from dma_kws.nn import build_encoder
from dma_kws.stage2.losses import compute_stage2_losses
from dma_kws.training.scheduler import build_cosine_warmup_optimizer

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _ensure_qbyt_on_path() -> None:
    qbyt_root = PROJECT_ROOT / "qbyt"
    qbyt_path = str(qbyt_root)
    if qbyt_path not in sys.path:
        sys.path.insert(0, qbyt_path)


def _load_qbyt():
    _ensure_qbyt_on_path()
    from model import QbyT

    return QbyT


def _extract_state_dict(checkpoint: dict[str, Any]) -> dict[str, torch.Tensor]:
    if "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    if "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    if "model" in checkpoint and isinstance(checkpoint["model"], dict):
        return checkpoint["model"]
    return checkpoint


def _split_submodule_state(
    state: dict[str, torch.Tensor],
    prefix: str,
) -> dict[str, torch.Tensor]:
    prefix_with_dot = f"{prefix}."
    return {
        key[len(prefix_with_dot) :]: value
        for key, value in state.items()
        if key.startswith(prefix_with_dot)
    }


class Stage2LightningModule(pl.LightningModule):
    """Lightning wrapper for ConformerEncoder + QbyT Stage II training."""

    def __init__(
        self,
        config: dict[str, Any],
        *,
        vocab_size: int = 73,
        freeze_encoder: bool = False,
        init_checkpoint: str | Path | None = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(
            {
                "vocab_size": vocab_size,
                "freeze_encoder": freeze_encoder,
                "init_checkpoint": str(init_checkpoint) if init_checkpoint else None,
            }
        )

        stage1 = config["stage1"]
        stage2 = config["stage2"]
        encoder_dim = int(stage2.get("encoder_output_dim", stage1.get("encoder_output_dim", 144)))

        self.encoder = build_encoder(stage1, output_dim=encoder_dim)
        QbyT = _load_qbyt()
        self.qbyt = QbyT(
            encoder_output_size=encoder_dim,
            num_embeds=vocab_size,
            embed_dim=int(stage2.get("qbyt_embed_dim", 128)),
            post_num_layers=int(stage2.get("qbyt_layers", 2)),
        )

        self._stage2_cfg = stage2
        self.freeze_encoder = freeze_encoder

        if init_checkpoint:
            self._load_init_checkpoint(Path(init_checkpoint))

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.auc_metric = torchmetrics.AUROC(task="binary")
        self.eer_metric = torchmetrics.classification.EER(task="binary")

    def _load_init_checkpoint(self, checkpoint_path: Path) -> None:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state = _extract_state_dict(checkpoint)

        encoder_state = _split_submodule_state(state, "encoder")
        qbyt_state = _split_submodule_state(state, "qbyt")

        if encoder_state:
            missing, unexpected = self.encoder.load_state_dict(encoder_state, strict=False)
            print(
                f"Loaded encoder weights from {checkpoint_path}: "
                f"missing={len(missing)} unexpected={len(unexpected)}"
            )
        if qbyt_state:
            missing, unexpected = self.qbyt.load_state_dict(qbyt_state, strict=False)
            print(
                f"Loaded QbyT weights from {checkpoint_path}: "
                f"missing={len(missing)} unexpected={len(unexpected)}"
            )

    def forward(
        self,
        feat: torch.Tensor,
        feat_lengths: torch.Tensor,
        anchor: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        encoder_out, encoder_mask = self.encoder(feat, feat_lengths)
        encoder_lens = encoder_mask.squeeze(1).sum(dim=1).to(dtype=torch.long)
        anchor_lengths = anchor.ne(0).sum(dim=1).to(dtype=torch.long)
        logits, seq_logits = self.qbyt(
            encoder_out,
            anchor,
            speech_lengths=encoder_lens,
            text_lengths=anchor_lengths,
        )
        return logits, seq_logits

    def on_train_epoch_start(self) -> None:
        if self.freeze_encoder:
            self.encoder.eval()

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        if self.freeze_encoder:
            self.encoder.eval()

        logits, seq_logits = self(batch["feat"], batch["feat_lengths"], batch["anchor"])
        total_loss, losses = compute_stage2_losses(
            logits=logits,
            seq_logits=seq_logits,
            labels=batch["label"],
            seq_labels=batch["seq_label"],
            seq_label_mask=batch["seq_label_mask"],
        )

        self.log("train/loss", total_loss, on_step=True, prog_bar=True)
        self.log("train/utt_loss", losses["utt_loss"], on_step=True, prog_bar=True)
        self.log("train/seq_loss", losses["seq_loss"], on_step=True, prog_bar=True)

        optimizer = self.optimizers()
        lr = optimizer.param_groups[0]["lr"]
        self.log("train/lr", lr, on_step=True, prog_bar=True)
        return total_loss

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        logits, _ = self(batch["feat"], batch["feat_lengths"], batch["anchor"])
        preds = torch.sigmoid(logits)
        labels = batch["label"].int()
        utt_loss = F.binary_cross_entropy_with_logits(logits, labels.float())
        self.log("val/utt_loss", utt_loss, prog_bar=True, on_epoch=True)
        self.auc_metric.update(preds, labels)
        self.eer_metric.update(preds, labels)

    def on_validation_epoch_end(self) -> None:
        auc = self.auc_metric.compute()
        eer = self.eer_metric.compute()

        self.log("val/auc", auc, prog_bar=True)
        self.log("val/eer", eer, prog_bar=True)
        self.log("val_auc", auc, prog_bar=True)

        self.auc_metric.reset()
        self.eer_metric.reset()

    def test_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        logits, _ = self(batch["feat"], batch["feat_lengths"], batch["anchor"])
        preds = torch.sigmoid(logits)
        labels = batch["label"].int()
        self.auc_metric.update(preds, labels)
        self.eer_metric.update(preds, labels)

    def on_test_epoch_end(self) -> None:
        auc = self.auc_metric.compute()
        eer = self.eer_metric.compute()

        self.log("test/auc", auc, prog_bar=True)
        self.log("test/eer", eer, prog_bar=True)

        self.auc_metric.reset()
        self.eer_metric.reset()

    def configure_optimizers(self) -> dict:
        optim_module = self.qbyt if self.freeze_encoder else self
        return build_cosine_warmup_optimizer(optim_module, self._stage2_cfg)
