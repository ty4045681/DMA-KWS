"""Stage II QbyT Lightning module aligned with main ``qbyt/train.py`` Wrapper."""

from __future__ import annotations

import copy
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
from dma_kws.stage2.objective import resolve_sequence_objective
from dma_kws.stage2.readout import resolve_qbyt_readout
from dma_kws.stage2.scoring import gather_completion_logits
from dma_kws.training.checkpoint_io import (
    assert_qbyt_readout_version,
    assert_stream_policy_matches,
    extract_state_dict,
    stamp_qbyt_readout_version,
)
from dma_kws.training.distributed_metrics import ddp_global_mean_loss, sum_across_processes
from dma_kws.training.ddp import process_rank, rank_zero_print
from dma_kws.training.scheduler import build_cosine_warmup_optimizer
from dma_kws.training.score_diagnostics import BinaryScoreDiagnostics


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
        require_full_qbyt_init: bool = False,
    ) -> None:
        super().__init__()
        # Lightning's hyper_parameters currently hold only constructor scalars.
        # Keep the resolved config separately so every future .ckpt is
        # self-describing enough to be converted to the inference .pt format.
        self._checkpoint_config = copy.deepcopy(config)
        self._checkpoint_vocab_size = int(vocab_size)
        self.save_hyperparameters(
            {
                "vocab_size": vocab_size,
                "freeze_encoder": freeze_encoder,
                "init_checkpoint": str(init_checkpoint) if init_checkpoint else None,
                "require_full_qbyt_init": require_full_qbyt_init,
            }
        )

        stage1 = config["stage1"]
        stage2 = config["stage2"]
        encoder_dim = int(stage2.get("encoder_output_dim", stage1.get("encoder_output_dim", 144)))

        sequence_objective = resolve_sequence_objective(stage2)
        qbyt_readout = resolve_qbyt_readout(stage2)
        self.qbyt_readout_mode = qbyt_readout.mode
        self.qbyt_readout_temperature = qbyt_readout.temperature
        self.seq_label_mode = sequence_objective.target_mode
        self.seq_progress_weight = sequence_objective.progress_weight
        self.seq_completion_weight = sequence_objective.completion_weight
        self.seq_normalization = sequence_objective.normalization
        # Minimal hand-written configs may omit this section. Stamp the resolved
        # objective into all new checkpoints so two same-shape QbyT models do not
        # become indistinguishable after being trained against different targets.
        self._checkpoint_config.setdefault("stage2", {})[
            "sequence_loss"
        ] = sequence_objective.as_dict()
        self._checkpoint_config["stage2"][
            "qbyt_readout"
        ] = qbyt_readout.as_dict()

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
            readout_mode=self.qbyt_readout_mode,
            readout_temperature=self.qbyt_readout_temperature,
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
            self._load_init_checkpoint(
                Path(init_checkpoint),
                require_full_qbyt=require_full_qbyt_init,
            )

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

        # All score diagnostics are non-decomposable. One shared accumulator
        # gathers probabilities/labels before computing AUC, EER/threshold,
        # low-FPR and calibration statistics, avoiding several duplicate state
        # buffers with subtly different semantics.
        self.deployment_threshold = float(
            (config.get("demo", {}) or {}).get("qbyt_threshold", 0.5)
        )
        validation_cfg = stage2.get("validation", {}) or {}
        self.ece_num_bins = int(validation_cfg.get("ece_num_bins", 15))
        self.sequence_diagnostic_threshold = float(
            validation_cfg.get("seq_diagnostic_threshold", 0.5)
        )
        self.sequence_diagnostic_namespace = (
            "completion_"
            if self.seq_label_mode == "ordered_contiguous_prefix"
            else "last_token_"
        )
        self.score_diagnostics = BinaryScoreDiagnostics(
            deployment_threshold=self.deployment_threshold,
            ece_num_bins=self.ece_num_bins,
            sync_on_compute=True,
        )
        self.completion_score_diagnostics = BinaryScoreDiagnostics(
            deployment_threshold=self.sequence_diagnostic_threshold,
            ece_num_bins=self.ece_num_bins,
            sync_on_compute=True,
        )
        self._train_window_metrics = torch.nn.ModuleDict(
            {
                name: torchmetrics.MeanMetric(sync_on_compute=True)
                for name in (
                    "loss_total",
                    "loss_utt_raw",
                    "loss_seq_weighted",
                    "loss_seq_progress_raw",
                    "loss_seq_progress_weighted",
                    "loss_seq_completion_raw",
                    "loss_seq_completion_weighted",
                    "loss_ctc_raw",
                    "loss_ctc_weighted",
                )
            }
        )
        self._train_window_updates = 0

    def _load_init_checkpoint(
        self,
        checkpoint_path: Path,
        *,
        require_full_qbyt: bool = False,
    ) -> None:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        assert_stream_policy_matches(checkpoint, self.stream_policy, source=checkpoint_path)
        assert_qbyt_readout_version(
            checkpoint,
            source=checkpoint_path,
            expected_mode=self.qbyt_readout_mode,
            expected_temperature=self.qbyt_readout_temperature,
            # A legacy readout is only useful as a warm start when QbyT will be
            # retrained. LoRA freezes QbyT, so its strict base path must reject it.
            allow_legacy=self._allow_legacy_qbyt_readout and not require_full_qbyt,
        )
        
        # Check if this is an icefall checkpoint (has "model" key)
        # vs standard DMA-KWS checkpoint (has "state_dict", "model_state_dict", etc)
        from dma_kws.training.checkpoint_io import extract_icefall_encoder_state

        model_state = checkpoint.get("model")
        has_qbyt_model_state = isinstance(model_state, dict) and any(
            key.startswith("qbyt.") for key in model_state
        )
        is_icefall_format = (
            isinstance(model_state, dict)
            and not has_qbyt_model_state
            and any(
                key.startswith("encoder_embed.") or key.startswith("encoder.")
                for key in model_state
            )
        )
        
        if is_icefall_format:
            if require_full_qbyt:
                raise SystemExit(
                    "LoRA adaptation requires a complete Stage II checkpoint with "
                    f"QbyT weights, but {checkpoint_path} is an encoder-only Icefall "
                    "checkpoint. Supply the trained Stage II .pt/.ckpt (or a merged "
                    "LoRA export) as the adaptation base."
                )
            # Load from icefall Zipformer checkpoint
            icefall_states = extract_icefall_encoder_state(checkpoint_path)
            
            # Check if encoder has submodules (icefall adapter has encoder_embed + encoder)
            encoder_embed_state = icefall_states.get("encoder_embed", {})
            encoder_state = icefall_states.get("encoder", {})
            
            if hasattr(self.encoder, "encoder_embed") and encoder_embed_state:
                missing, unexpected = self.encoder.encoder_embed.load_state_dict(
                    encoder_embed_state, strict=False
                )
                rank_zero_print(
                    f"Loaded encoder_embed weights from {checkpoint_path}: "
                    f"missing={len(missing)} unexpected={len(unexpected)}"
                )
            
            if hasattr(self.encoder, "encoder") and encoder_state:
                missing, unexpected = self.encoder.encoder.load_state_dict(encoder_state, strict=False)
                rank_zero_print(
                    f"Loaded encoder weights from {checkpoint_path}: "
                    f"missing={len(missing)} unexpected={len(unexpected)}"
                )
        else:
            # Standard DMA-KWS checkpoint (Lightning or custom format)
            state = extract_state_dict(checkpoint)
            
            encoder_state = _split_submodule_state(state, "encoder")
            qbyt_state = _split_submodule_state(state, "qbyt")
            adapter_state = _split_submodule_state(state, "adapter")

            if require_full_qbyt:
                missing_roots = [
                    root
                    for root, submodule_state in (
                        ("encoder.*", encoder_state),
                        ("qbyt.*", qbyt_state),
                    )
                    if not submodule_state
                ]
                if missing_roots:
                    raise SystemExit(
                        "LoRA adaptation requires a complete Stage II checkpoint, "
                        f"but {checkpoint_path} has no {', '.join(missing_roots)} "
                        "weights. The encoder and QbyT base are frozen during LoRA "
                        "training, so missing weights would remain random."
                    )

            if encoder_state:
                missing, unexpected = self.encoder.load_state_dict(
                    encoder_state,
                    strict=require_full_qbyt,
                )
                rank_zero_print(
                    f"Loaded encoder weights from {checkpoint_path}: "
                    f"missing={len(missing)} unexpected={len(unexpected)}"
                )
            if adapter_state and self.adapter is not None:
                # Strict on purpose: a partially loaded trunk means QbyT reads a
                # space CTC only partly supervised, and the generic strict=False
                # path above would reduce that to a "missing=N" line.
                self.adapter.load_state_dict(adapter_state, strict=True)
                self._adapter_weights_loaded = True
                rank_zero_print(f"Loaded phoneme adapter weights from {checkpoint_path}")
            elif adapter_state and self.adapter is None:
                if process_rank() == 0:
                    warnings.warn(
                        f"{checkpoint_path} carries phoneme adapter weights but "
                        "stage2.phoneme_adapter.enabled is false, so they are ignored.",
                        UserWarning,
                        stacklevel=2,
                    )
            if qbyt_state:
                missing, unexpected = self.qbyt.load_state_dict(
                    qbyt_state,
                    strict=require_full_qbyt,
                )
                rank_zero_print(
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
        rank_zero_print(f"Loaded phoneme adapter weights from {checkpoint_path}")

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Embed the metadata required for a safe ``.ckpt`` -> ``.pt`` export.

        Checkpoint averaging keeps the first payload as its template and only
        replaces the state dict, so averaged checkpoints inherit these fields.
        """
        stamp_qbyt_readout_version(checkpoint)
        checkpoint["checkpoint_kind"] = "stage2"
        checkpoint["config"] = copy.deepcopy(self._checkpoint_config)
        checkpoint["vocab_size"] = self._checkpoint_vocab_size

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Reject stale QbyT weights on both Lightning restore paths.

        Covers ``load_from_checkpoint`` and ``Trainer.fit(ckpt_path=...)``; the
        weights-only ``init_checkpoint`` path is guarded separately.
        """
        assert_stream_policy_matches(
            checkpoint,
            self.stream_policy,
            source="the Stage II checkpoint being restored",
        )
        # Lightning does not hand the path to this hook, and touching
        # ``self.trainer`` raises when the module is not attached to one, which is
        # exactly the load_from_checkpoint case.
        assert_qbyt_readout_version(
            checkpoint,
            source="the Stage II checkpoint being restored",
            expected_mode=self.qbyt_readout_mode,
            expected_temperature=self.qbyt_readout_temperature,
            # Full Lightning restore also restores optimizer/scheduler state.
            # The legacy escape hatch is weights-only and belongs exclusively
            # to _load_init_checkpoint; a v2 gru_last checkpoint is accepted by
            # the explicit compatibility rule without this flag.
            allow_legacy=False,
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
            if process_rank() == 0:
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
        local_valid = batch_size - num_skipped
        global_stats = sum_across_processes(
            torch.tensor(
                [float(ctc_loss.detach()) * local_valid, local_valid, num_skipped],
                device=ctc_loss.device,
                dtype=torch.float64,
            )
        )
        global_loss_sum, valid_total, skipped_total = global_stats.unbind()
        global_valid = int(valid_total.item())
        global_skipped = int(skipped_total.item())
        global_batch_size = global_valid + global_skipped
        ctc_loss = ddp_global_mean_loss(
            ctc_loss,
            local_count=local_valid,
            global_sum=global_loss_sum,
            global_count=global_valid,
        )
        self.log(
            "train/microbatch/ctc_valid",
            float(global_valid),
            on_step=True,
            batch_size=global_batch_size,
        )
        self.log(
            "train/microbatch/ctc_skipped",
            float(global_skipped),
            on_step=True,
            batch_size=global_batch_size,
        )
        self.log(
            "train/microbatch/ctc_skip_rate",
            global_skipped / max(global_batch_size, 1),
            on_step=True,
            batch_size=global_batch_size,
        )
        self.log(
            "train/epoch/ctc_skip_rate",
            global_skipped / max(global_batch_size, 1),
            on_step=False,
            on_epoch=True,
            batch_size=global_batch_size,
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
            seq_progress_weight=self.seq_progress_weight,
            seq_completion_weight=self.seq_completion_weight,
            seq_normalization=self.seq_normalization,
            ctc_loss=self._auxiliary_ctc_loss(batch, ctc_log_probs, encoder_mask),
            ctc_weight=self.ctc_weight,
        )
        return total_loss, losses, logits

    def _log_train_losses(self, total_loss: torch.Tensor, losses: dict[str, torch.Tensor]) -> None:
        metrics = {
            "loss_total": total_loss,
            "loss_utt_raw": losses["utt_loss"],
            "loss_seq_weighted": losses["seq_loss"],
            "loss_seq_progress_raw": losses["seq_progress_loss"],
            "loss_seq_progress_weighted": losses[
                "seq_progress_weighted_loss"
            ],
            "loss_seq_completion_raw": losses["seq_completion_loss"],
            "loss_seq_completion_weighted": losses[
                "seq_completion_weighted_loss"
            ],
        }
        if "ctc_loss" in losses:
            metrics["loss_ctc_raw"] = losses["ctc_loss"]
            metrics["loss_ctc_weighted"] = losses["ctc_weighted_loss"]

        progress_metrics = {
            "loss_total",
            "loss_utt_raw",
            "loss_seq_weighted",
            "loss_ctc_weighted",
        }
        for name, value in metrics.items():
            self.log(
                f"train/microbatch/{name}",
                value,
                on_step=True,
                prog_bar=name in progress_metrics,
            )
            self._train_window_metrics[name].update(value.detach())
        self._train_window_updates += 1

        # Keep the released top-level names for one compatibility cycle. They
        # are hidden from the progress bar; new dashboards should consume the
        # explicit microbatch/window hierarchy above.
        self.log(
            "train/loss", total_loss, on_step=True, prog_bar=False, logger=False
        )
        self.log(
            "train/utt_loss",
            losses["utt_loss"],
            on_step=True,
            prog_bar=False,
            logger=False,
        )
        self.log(
            "train/seq_loss",
            losses["seq_loss"],
            on_step=True,
            prog_bar=False,
            logger=False,
        )
        if self.seq_progress_weight:
            self.log(
                "train/seq_progress_loss",
                losses["seq_progress_loss"],
                on_step=True,
                logger=False,
            )
        if self.seq_completion_weight:
            self.log(
                "train/seq_completion_loss",
                losses["seq_completion_loss"],
                on_step=True,
                logger=False,
            )
        if "ctc_loss" in losses:
            self.log(
                "train/ctc_loss",
                losses["ctc_loss"],
                on_step=True,
                prog_bar=False,
                logger=False,
            )

        optimizer = self.optimizers()
        lr = optimizer.param_groups[0]["lr"]
        self.log("train/optimizer/lr", lr, on_step=True, prog_bar=True)
        self.log("train/lr", lr, on_step=True, prog_bar=False, logger=False)

    def _consume_train_window_metrics(self) -> dict[str, torch.Tensor]:
        if self._train_window_updates <= 0:
            return {}
        values: dict[str, torch.Tensor] = {}
        for name, metric in self._train_window_metrics.items():
            # Optional CTC metrics receive no updates when the adapter/auxiliary
            # loss is disabled; MeanMetric.compute() would otherwise emit NaN.
            if name.startswith("loss_ctc") and not self.ctc_weight:
                continue
            values[f"train/window/{name}"] = metric.compute()
            metric.reset()
        values["train/window/microbatches"] = torch.tensor(
            float(self._train_window_updates),
            device=self.device,
        )
        self._train_window_updates = 0
        return values

    def _log_train_window_metrics(self) -> None:
        for name, value in self._consume_train_window_metrics().items():
            self.log(name, value, sync_dist=True)

    def on_train_end(self) -> None:
        """Flush a trailing window directly because Lightning forbids self.log here."""
        values = self._consume_train_window_metrics()
        if not values or not self.trainer.is_global_zero:
            return
        payload = {
            name: float(value.detach().cpu()) for name, value in values.items()
        }
        for train_logger in self.trainer.loggers:
            train_logger.log_metrics(payload, step=int(self.global_step))

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
            self.log("train/optimizer/grad_norm_pre_clip", total, on_step=True)
            self.log("train/grad_norm", total, on_step=True, logger=False)

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

    @staticmethod
    def _completion_probabilities(
        seq_logits: torch.Tensor,
        anchor: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return the final valid anchor-position score and matching labels."""
        anchor_lengths = anchor.ne(0).sum(dim=1)
        completion_logits, valid = gather_completion_logits(seq_logits, anchor_lengths)
        return torch.sigmoid(completion_logits[valid]), labels[valid], valid

    def _update_score_diagnostics(
        self,
        score_metric: BinaryScoreDiagnostics,
        completion_metric: BinaryScoreDiagnostics,
        *,
        logits: torch.Tensor,
        seq_logits: torch.Tensor,
        anchor: torch.Tensor,
        labels: torch.Tensor,
        sample_ids: torch.Tensor | None = None,
    ) -> None:
        score_metric.update(torch.sigmoid(logits), labels, sample_ids)
        completion_scores, completion_labels, completion_valid = (
            self._completion_probabilities(
                seq_logits,
                anchor,
                labels,
            )
        )
        completion_metric.update(
            completion_scores,
            completion_labels,
            sample_ids[completion_valid] if sample_ids is not None else None,
        )

    def _log_score_diagnostics(
        self,
        metric: BinaryScoreDiagnostics,
        *,
        namespace: str,
        progress_bar: bool = False,
        deployment_metric: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Log one already-DDP-global diagnostic dictionary."""
        diagnostics = metric.compute()
        progress_names = {"auc", "eer", "eer_threshold"}
        for name, value in diagnostics.items():
            if name == "log_loss":
                # Callers give the utterance loss a dataset-specific canonical
                # name (val/utt_loss, val/target_utt_loss, ...).
                continue
            output_name = name
            if not deployment_metric and name.startswith("deploy_"):
                output_name = f"diagnostic_{name.removeprefix('deploy_')}"
            # Lightning reduces every tensor passed to ``self.log`` and warns
            # when integer/bool values need an implicit float conversion. Keep
            # the public diagnostics dictionary strongly typed, but make the
            # logging boundary explicit and quiet.
            logged_value = (
                value
                if torch.is_floating_point(value)
                else value.to(dtype=torch.float32)
            )
            self.log(
                f"{namespace}{output_name}",
                logged_value,
                prog_bar=progress_bar and name in progress_names,
                # BinaryScoreDiagnostics synchronized the raw sample state, so
                # every rank holds the same scalar. Lightning cannot be told
                # that explicitly; an identity mean silences its per-metric DDP
                # warning while preserving the value used by all callbacks.
                sync_dist=True,
            )
        return diagnostics

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        logits, seq_logits = self(
            batch["feat"], batch["feat_lengths"], batch["anchor"]
        )
        labels = batch["label"].int()
        self._update_score_diagnostics(
            self.score_diagnostics,
            self.completion_score_diagnostics,
            logits=logits,
            seq_logits=seq_logits,
            anchor=batch["anchor"],
            labels=labels,
            sample_ids=batch.get("sample_id"),
        )

    def on_validation_epoch_end(self) -> None:
        score_metrics = self._log_score_diagnostics(
            self.score_diagnostics,
            namespace="val/",
            progress_bar=True,
        )
        self.log(
            "val/utt_loss",
            score_metrics["log_loss"],
            prog_bar=True,
            sync_dist=True,
        )
        self._log_score_diagnostics(
            self.completion_score_diagnostics,
            namespace=f"val/{self.sequence_diagnostic_namespace}",
            deployment_metric=False,
        )
        # Checkpoint-filename alias of val/auc; kept out of CSV/TensorBoard.
        self.log("val_auc", score_metrics["auc"], sync_dist=True, logger=False)
        self._log_train_window_metrics()

        self.score_diagnostics.reset()
        self.completion_score_diagnostics.reset()

    def test_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        logits, seq_logits = self(
            batch["feat"], batch["feat_lengths"], batch["anchor"]
        )
        labels = batch["label"].int()
        self._update_score_diagnostics(
            self.score_diagnostics,
            self.completion_score_diagnostics,
            logits=logits,
            seq_logits=seq_logits,
            anchor=batch["anchor"],
            labels=labels,
            sample_ids=batch.get("sample_id"),
        )

    def on_test_epoch_end(self) -> None:
        self._log_score_diagnostics(
            self.score_diagnostics,
            namespace="test/",
            progress_bar=True,
        )
        self._log_score_diagnostics(
            self.completion_score_diagnostics,
            namespace=f"test/{self.sequence_diagnostic_namespace}",
            deployment_metric=False,
        )
        self.score_diagnostics.reset()
        self.completion_score_diagnostics.reset()

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
