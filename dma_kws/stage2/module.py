"""Stage II QbyT Lightning module aligned with main ``qbyt/train.py`` Wrapper."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchmetrics

from dma_kws.nn import build_encoder
from dma_kws.pathing import load_qbyt_class
from dma_kws.stage2.losses import compute_stage2_losses
from dma_kws.training.checkpoint_io import extract_state_dict
from dma_kws.training.scheduler import build_cosine_warmup_optimizer


def _load_qbyt():
    return load_qbyt_class()


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
        self._log_grad_norm = bool((stage2.get("logging", {}) or {}).get("grad_norm", True))
        gradient_diagnostics = stage2.get("gradient_diagnostics", {}) or {}
        self._gradient_diagnostics_enabled = bool(gradient_diagnostics.get("enabled", False))
        self._gradient_diagnostics_max_steps = max(0, int(gradient_diagnostics.get("max_steps", 5)))
        self._gradient_diagnostics_checks = 0
        self._last_missing_gradients: tuple[str, ...] | None = None

        if init_checkpoint:
            self._load_init_checkpoint(Path(init_checkpoint))

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.auc_metric = torchmetrics.AUROC(task="binary")
        self.eer_metric = torchmetrics.classification.EER(task="binary")

    def _load_init_checkpoint(self, checkpoint_path: Path) -> None:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        
        # Check if this is an icefall checkpoint (has "model" key)
        # vs standard DMA-KWS checkpoint (has "state_dict", "model_state_dict", etc)
        from dma_kws.training.checkpoint_io import extract_icefall_encoder_state

        model_state = checkpoint.get("model")
        is_icefall_format = isinstance(model_state, dict) and any(
            key.startswith("encoder_embed.") or key.startswith("encoder.")
            for key in model_state
        )
        
        if is_icefall_format:
            # Load from icefall Zipformer checkpoint
            icefall_states = extract_icefall_encoder_state(checkpoint_path)
            
            # Check if encoder has submodules (icefall adapter has encoder_embed + encoder)
            encoder_embed_state = icefall_states.get("encoder_embed", {})
            encoder_state = icefall_states.get("encoder", {})
            
            if hasattr(self.encoder, "encoder_embed") and encoder_embed_state:
                missing, unexpected = self.encoder.encoder_embed.load_state_dict(
                    encoder_embed_state, strict=False
                )
                print(
                    f"Loaded encoder_embed weights from {checkpoint_path}: "
                    f"missing={len(missing)} unexpected={len(unexpected)}"
                )
            
            if hasattr(self.encoder, "encoder") and encoder_state:
                missing, unexpected = self.encoder.encoder.load_state_dict(encoder_state, strict=False)
                print(
                    f"Loaded encoder weights from {checkpoint_path}: "
                    f"missing={len(missing)} unexpected={len(unexpected)}"
                )
        else:
            # Standard DMA-KWS checkpoint (Lightning or custom format)
            state = extract_state_dict(checkpoint)
            
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

    def _forward_train_losses(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
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
        return total_loss, losses, logits

    def _log_train_losses(self, total_loss: torch.Tensor, losses: dict[str, torch.Tensor]) -> None:
        self.log("train/loss", total_loss, on_step=True, prog_bar=True)
        self.log("train/utt_loss", losses["utt_loss"], on_step=True, prog_bar=True)
        self.log("train/seq_loss", losses["seq_loss"], on_step=True, prog_bar=True)

        optimizer = self.optimizers()
        lr = optimizer.param_groups[0]["lr"]
        self.log("train/lr", lr, on_step=True, prog_bar=True)

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        total_loss, losses, _ = self._forward_train_losses(batch)
        self._log_train_losses(total_loss, losses)
        return total_loss

    def on_before_optimizer_step(self, optimizer) -> None:
        if not self._log_grad_norm:
            return
        from pytorch_lightning.utilities import grad_norm

        norms = grad_norm(self, norm_type=2)
        total = norms.get("grad_2.0_norm_total")
        if total is not None:
            self.log("train/grad_norm", total, on_step=True)

    def _parameters_without_gradients(self) -> tuple[str, ...]:
        return tuple(
            name
            for name, parameter in self.named_parameters()
            if parameter.requires_grad and parameter.grad is None
        )

    def on_after_backward(self) -> None:
        if (
            not self._gradient_diagnostics_enabled
            or self._gradient_diagnostics_checks >= self._gradient_diagnostics_max_steps
        ):
            return

        self._gradient_diagnostics_checks += 1
        missing = self._parameters_without_gradients()
        if missing == self._last_missing_gradients:
            return
        self._last_missing_gradients = missing

        step = int(self.global_step)
        if missing:
            names = "\n".join(f"  {name}" for name in missing)
            self.print(
                f"Gradient diagnostics at step {step}: "
                f"{len(missing)} trainable parameters have no gradient:\n{names}"
            )
        else:
            self.print(f"Gradient diagnostics at step {step}: all trainable parameters have gradients.")

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        logits, _ = self(batch["feat"], batch["feat_lengths"], batch["anchor"])
        preds = torch.sigmoid(logits)
        labels = batch["label"].int()
        utt_loss = F.binary_cross_entropy_with_logits(logits, labels.float())
        self.log("val/utt_loss", utt_loss, prog_bar=True, on_epoch=True, sync_dist=True)
        self.auc_metric.update(preds, labels)
        self.eer_metric.update(preds, labels)

    def on_validation_epoch_end(self) -> None:
        auc = self.auc_metric.compute()
        eer = self.eer_metric.compute()

        self.log("val/auc", auc, prog_bar=True, sync_dist=True)
        self.log("val/eer", eer, prog_bar=True, sync_dist=True)
        # Checkpoint-filename alias of val/auc; kept out of CSV/TensorBoard.
        self.log("val_auc", auc, sync_dist=True, logger=False)

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

        self.log("test/auc", auc, prog_bar=True, sync_dist=True)
        self.log("test/eer", eer, prog_bar=True, sync_dist=True)

        self.auc_metric.reset()
        self.eer_metric.reset()

    def configure_optimizers(self) -> dict:
        optim_module = self.qbyt if self.freeze_encoder else self
        return build_cosine_warmup_optimizer(optim_module, self._stage2_cfg)
