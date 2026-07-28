"""Stage II QbyT Lightning module aligned with main ``qbyt/train.py`` Wrapper."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchmetrics

from dma_kws.config import resolve_stream_policy
from dma_kws.nn import build_encoder, run_encoder
from dma_kws.pathing import load_qbyt_class
from dma_kws.stage2.losses import compute_stage2_losses
from dma_kws.training.checkpoint_io import (
    assert_qbyt_readout_version,
    assert_stream_policy_matches,
    extract_state_dict,
    stamp_qbyt_readout_version,
)
from dma_kws.training.scheduler import build_cosine_warmup_optimizer


def _load_qbyt():
    return load_qbyt_class()


def assert_adapter_weights_loaded(model, missing_keys) -> None:
    """Fail when an enabled phoneme adapter got no weights from a checkpoint.

    Evaluation loads Stage II weights with ``strict=False`` so older checkpoints
    keep working. That tolerance is dangerous here: a randomly initialized trunk
    still produces scores, just meaningless ones, and the only symptom is a
    "missing keys" line in the log.
    """
    if getattr(model, "adapter", None) is None:
        return
    missing_adapter = [key for key in missing_keys if key.startswith("adapter.")]
    if missing_adapter:
        raise SystemExit(
            f"stage2.phoneme_adapter is enabled but the checkpoint has no weights for "
            f"{len(missing_adapter)} adapter parameters (e.g. {missing_adapter[0]}). "
            "Scoring with a randomly initialized trunk is silently wrong; either load a "
            "checkpoint trained with the adapter or set stage2.phoneme_adapter.enabled=false."
        )


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
        vocab_size: int,
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

        self.stream_policy = resolve_stream_policy(stage1)
        self.encoder = build_encoder(stage1, output_dim=encoder_dim)

        # The phoneme adapter is the trunk the CTC loss and QbyT share. Without
        # it QbyT reads the encoder output directly, which for a BPE/transducer
        # encoder is not a space the phoneme text embedding can be compared
        # against. Disabled by default so pre-adapter checkpoints still load.
        adapter_cfg = stage2.get("phoneme_adapter", {}) or {}
        self.adapter_enabled = bool(adapter_cfg.get("enabled", False))
        self.freeze_adapter = self.adapter_enabled and bool(adapter_cfg.get("freeze", False))
        # There is no trunk to apply CTC to when the adapter is off, so keep the
        # reported weight honest instead of leaving a configured value dangling.
        self.ctc_weight = float(adapter_cfg.get("ctc_weight", 0.0)) if self.adapter_enabled else 0.0
        if self.adapter_enabled:
            from dma_kws.phoneme_adapter.module import build_phoneme_adapter

            self.adapter = build_phoneme_adapter(
                adapter_cfg,
                input_dim=encoder_dim,
                vocab_size=vocab_size,
                causal=bool(stage1.get("causal", False)),
            )
            qbyt_input_dim = self.adapter.output_dim
            if not self.ctc_weight:
                # With no auxiliary CTC loss the projection is dead weight in
                # this stage. It has to stay in the state dict for strict loads,
                # but leaving it trainable makes DDP abort: a parameter that
                # requires grad and never receives one fails the reduction check
                # unless find_unused_parameters happens to be on.
                for param in self.adapter.ctc.parameters():
                    param.requires_grad = False
        else:
            self.adapter = None
            qbyt_input_dim = encoder_dim

        QbyT = _load_qbyt()
        self.qbyt = QbyT(
            encoder_output_size=qbyt_input_dim,
            num_embeds=vocab_size,
            embed_dim=int(stage2.get("qbyt_embed_dim", 128)),
            post_num_layers=int(stage2.get("qbyt_layers", 2)),
        )

        self._stage2_cfg = stage2
        self._adapter_cfg = adapter_cfg
        self.freeze_encoder = freeze_encoder
        self._allow_legacy_qbyt_readout = bool(stage2.get("allow_legacy_qbyt_readout", False))
        self._log_grad_norm = bool((stage2.get("logging", {}) or {}).get("grad_norm", True))
        gradient_diagnostics = stage2.get("gradient_diagnostics", {}) or {}
        self._gradient_diagnostics_enabled = bool(gradient_diagnostics.get("enabled", False))
        self._gradient_diagnostics_max_steps = max(0, int(gradient_diagnostics.get("max_steps", 5)))
        self._gradient_diagnostics_checks = 0
        self._last_missing_gradients: tuple[str, ...] | None = None

        self._adapter_weights_loaded = False
        if init_checkpoint:
            self._load_init_checkpoint(Path(init_checkpoint))

        adapter_checkpoint = str(adapter_cfg.get("init_checkpoint", "")).strip()
        if self.adapter_enabled and adapter_checkpoint:
            self._load_adapter_checkpoint(Path(adapter_checkpoint))

        # A random trunk is a legitimate starting point only when no checkpoint
        # was supposed to provide one. When one was and it carried no adapter
        # weights, the trunk stays random, gets frozen for LoRA a moment later,
        # and every score comes from an unmapped space -- with nothing in the log
        # but a missing-keys count.
        if self.adapter is not None and not self._adapter_weights_loaded:
            source = adapter_checkpoint or init_checkpoint
            if source:
                raise SystemExit(
                    f"stage2.phoneme_adapter is enabled but {source} carries no adapter.* "
                    "weights, so the trunk would stay randomly initialized. Load a checkpoint "
                    "trained with the adapter, set stage2.phoneme_adapter.init_checkpoint to a "
                    "scripts/train_ctc_adapter.py export, or disable the adapter."
                )

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        if self.freeze_adapter:
            for param in self.adapter.parameters():
                param.requires_grad = False

        self.auc_metric = torchmetrics.AUROC(task="binary")
        self.eer_metric = torchmetrics.classification.EER(task="binary")

    def _load_init_checkpoint(self, checkpoint_path: Path) -> None:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        assert_stream_policy_matches(checkpoint, self.stream_policy, source=checkpoint_path)
        assert_qbyt_readout_version(
            checkpoint,
            source=checkpoint_path,
            allow_legacy=self._allow_legacy_qbyt_readout,
        )
        
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
            adapter_state = _split_submodule_state(state, "adapter")

            if encoder_state:
                missing, unexpected = self.encoder.load_state_dict(encoder_state, strict=False)
                print(
                    f"Loaded encoder weights from {checkpoint_path}: "
                    f"missing={len(missing)} unexpected={len(unexpected)}"
                )
            if adapter_state and self.adapter is not None:
                # Strict on purpose: a partially loaded trunk means QbyT reads a
                # space CTC only partly supervised, and the generic strict=False
                # path above would reduce that to a "missing=N" line.
                self.adapter.load_state_dict(adapter_state, strict=True)
                self._adapter_weights_loaded = True
                print(f"Loaded phoneme adapter weights from {checkpoint_path}")
            elif adapter_state and self.adapter is None:
                warnings.warn(
                    f"{checkpoint_path} carries phoneme adapter weights but "
                    "stage2.phoneme_adapter.enabled is false, so they are ignored.",
                    UserWarning,
                    stacklevel=2,
                )
            if qbyt_state:
                missing, unexpected = self.qbyt.load_state_dict(qbyt_state, strict=False)
                print(
                    f"Loaded QbyT weights from {checkpoint_path}: "
                    f"missing={len(missing)} unexpected={len(unexpected)}"
                )

    def _load_adapter_checkpoint(self, checkpoint_path: Path) -> None:
        """Load a Step A adapter export produced by ``scripts/train_ctc_adapter.py``.

        Strict: a trunk whose shape does not match the Stage II config is not a
        warning-level problem, it means QbyT would be reading a differently
        shaped space than the one CTC supervised.
        """
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        assert_stream_policy_matches(checkpoint, self.stream_policy, source=checkpoint_path)

        # The Step A export records the blank id it was trained with. Nothing
        # downstream would notice a mismatch: CTC would just be optimizing a
        # different symbol than the one collapse/search treat as blank.
        saved_blank_id = checkpoint.get("blank_id") if isinstance(checkpoint, dict) else None
        if saved_blank_id is not None and int(saved_blank_id) != self.adapter.blank_id:
            raise SystemExit(
                f"{checkpoint_path} was trained with blank_id={int(saved_blank_id)} but this "
                f"adapter uses blank_id={self.adapter.blank_id}."
            )

        state = extract_state_dict(checkpoint)
        adapter_state = _split_submodule_state(state, "adapter") or state
        self.adapter.load_state_dict(adapter_state, strict=True)
        self._adapter_weights_loaded = True
        print(f"Loaded phoneme adapter weights from {checkpoint_path}")

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Stamp the QbyT readout version onto every Lightning checkpoint.

        Checkpoint averaging keeps the first payload as its template and only
        replaces the state dict, so averaged checkpoints inherit this too.
        """
        stamp_qbyt_readout_version(checkpoint)

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Reject stale QbyT weights on both Lightning restore paths.

        Covers ``load_from_checkpoint`` and ``Trainer.fit(ckpt_path=...)``; the
        weights-only ``init_checkpoint`` path is guarded separately.
        """
        # Lightning does not hand the path to this hook, and touching
        # ``self.trainer`` raises when the module is not attached to one, which is
        # exactly the load_from_checkpoint case.
        assert_qbyt_readout_version(
            checkpoint,
            source="the Stage II checkpoint being restored",
            allow_legacy=self._allow_legacy_qbyt_readout,
        )

    def forward(
        self,
        feat: torch.Tensor,
        feat_lengths: torch.Tensor,
        anchor: torch.Tensor,
        *,
        mode: str = "eval",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits, seq_logits, _ = self.forward_with_encoder(
            feat, feat_lengths, anchor, mode=mode
        )
        return logits, seq_logits

    def forward_with_encoder(
        self,
        feat: torch.Tensor,
        feat_lengths: torch.Tensor,
        anchor: torch.Tensor,
        *,
        mode: str = "eval",
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor | None, torch.Tensor]]:
        """Like :meth:`forward` but also returns ``(ctc_log_probs, encoder_mask)``.

        The CTC log-probabilities come from the same adapter call that produced
        QbyT's input. Recomputing them would run the trunk twice per step and,
        with dropout on, would let the CTC loss supervise a different realization
        than the one the matcher saw, undoing the coupling the trunk exists for.
        """
        # ``mode`` defaults to "eval" so validation/test/inference always run at the
        # deployment operating point; only the training step opts into randomization.
        encoder_out, encoder_mask = run_encoder(
            self.encoder,
            feat,
            feat_lengths,
            policy=self.stream_policy,
            mode=mode,
        )
        encoder_lens = encoder_mask.squeeze(1).sum(dim=1).to(dtype=torch.long)
        anchor_lengths = anchor.ne(0).sum(dim=1).to(dtype=torch.long)

        speech = encoder_out
        ctc_log_probs = None
        if self.adapter is not None:
            speech, ctc_log_probs = self.adapter(
                encoder_out,
                encoder_mask,
                # Skip the CTC projection when no loss consumes it, otherwise
                # ``ctc.ctc_lo`` receives no gradient and DDP aborts unless
                # find_unused_parameters is on.
                with_log_probs=bool(self.ctc_weight),
            )

        logits, seq_logits = self.qbyt(
            speech,
            anchor,
            speech_lengths=encoder_lens,
            text_lengths=anchor_lengths,
        )
        return logits, seq_logits, (ctc_log_probs, encoder_mask)

    def on_train_start(self) -> None:
        # Only warn once a fit actually starts; eval scripts build this module with
        # the default freeze_encoder=False and must stay quiet.
        if not self.freeze_encoder and self.stream_policy.backend == "icefall_zipformer":
            warnings.warn(
                "Fine-tuning an icefall Zipformer encoder: icefall's set_batch_count() is never "
                "called here, so every ScheduledFloat (dropout, Balancer limits, layerdrop, "
                "whitening) stays pinned to its `default` and the icefall training recipe is not "
                "reproduced.",
                UserWarning,
                stacklevel=2,
            )

    def on_train_epoch_start(self) -> None:
        self._set_frozen_submodules_to_eval()

    def _set_frozen_submodules_to_eval(self) -> None:
        if self.freeze_encoder:
            self.encoder.eval()
        if self.freeze_adapter:
            self.adapter.eval()

    def _auxiliary_ctc_loss(
        self,
        batch: dict[str, torch.Tensor],
        ctc_log_probs: torch.Tensor | None,
        encoder_mask: torch.Tensor,
    ) -> torch.Tensor | None:
        """Phoneme CTC on the trunk, using each clip's *own* phoneme sequence.

        The target is ``query_seq``, not the anchor: for a negative pair the
        audio is a different phrase, so supervising it with the anchor's phonemes
        would teach the trunk the wrong transcript.
        """
        if ctc_log_probs is None or not self.ctc_weight:
            return None
        targets = batch.get("query_seq")
        target_lengths = batch.get("query_lengths")
        if targets is None or target_lengths is None:
            return None

        ctc_loss, num_skipped = self.adapter.ctc_loss(
            ctc_log_probs, encoder_mask, targets, target_lengths
        )
        # A high skip rate means the auxiliary loss only ever sees long clips.
        batch_size = int(target_lengths.numel())
        self.log(
            "train/ctc_skipped",
            float(num_skipped),
            on_step=True,
            batch_size=batch_size,
        )
        self.log(
            "train/ctc_skip_rate",
            num_skipped / max(batch_size, 1),
            on_step=True,
            on_epoch=True,
            batch_size=batch_size,
        )
        return ctc_loss

    def _forward_train_losses(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
        self._set_frozen_submodules_to_eval()

        logits, seq_logits, (ctc_log_probs, encoder_mask) = self.forward_with_encoder(
            batch["feat"], batch["feat_lengths"], batch["anchor"], mode="train"
        )
        total_loss, losses = compute_stage2_losses(
            logits=logits,
            seq_logits=seq_logits,
            labels=batch["label"],
            seq_labels=batch["seq_label"],
            seq_label_mask=batch["seq_label_mask"],
            ctc_loss=self._auxiliary_ctc_loss(batch, ctc_log_probs, encoder_mask),
            ctc_weight=self.ctc_weight,
        )
        return total_loss, losses, logits

    def _log_train_losses(self, total_loss: torch.Tensor, losses: dict[str, torch.Tensor]) -> None:
        self.log("train/loss", total_loss, on_step=True, prog_bar=True)
        self.log("train/utt_loss", losses["utt_loss"], on_step=True, prog_bar=True)
        self.log("train/seq_loss", losses["seq_loss"], on_step=True, prog_bar=True)
        if "ctc_loss" in losses:
            self.log("train/ctc_loss", losses["ctc_loss"], on_step=True, prog_bar=True)

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

    def trainable_module(self) -> torch.nn.Module:
        """Return the module whose parameters the optimizer should own.

        Enumerated explicitly rather than defaulting to ``self.qbyt`` whenever
        the encoder is frozen: that shortcut silently drops the phoneme adapter,
        which is exactly the module the auxiliary CTC loss is meant to train,
        and no loss curve would reveal the omission.
        """
        if not self.freeze_encoder:
            return self

        trainable = [self.qbyt]
        if self.adapter is not None and not self.freeze_adapter:
            trainable.append(self.adapter)
        return torch.nn.ModuleList(trainable)

    def configure_optimizers(self) -> dict:
        return build_cosine_warmup_optimizer(self.trainable_module(), self._stage2_cfg)
