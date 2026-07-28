"""Step A: train the phoneme CTC trunk on a frozen encoder."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytorch_lightning as pl
import torch

from dma_kws.config import resolve_stream_policy
from dma_kws.metrics import collapse_ctc, edit_distance
from dma_kws.nn import build_encoder, run_encoder
from dma_kws.phoneme_adapter.module import build_phoneme_adapter
from dma_kws.training.checkpoint_io import (
    assert_stream_policy_matches,
    extract_icefall_encoder_state,
    extract_state_dict,
)
from dma_kws.training.scheduler import build_cosine_warmup_optimizer

BLANK_ID = 0


class PhonemeAdapterCtcModule(pl.LightningModule):
    """Frozen encoder + trainable phoneme CTC adapter.

    The encoder is frozen and forced into ``eval()``: with
    ``use_icefall_dropout_schedule`` the Zipformer's ``ScheduledFloat`` dropout
    stays pinned at its ``default`` (0.3) because icefall's ``set_batch_count()``
    is never called here, so leaving it in train mode would train the trunk
    against a representation the deployed encoder never produces.
    """

    def __init__(
        self,
        config: dict[str, Any],
        *,
        vocab_size: int,
        blank_id: int = BLANK_ID,
        init_checkpoint: str | Path | None = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(
            {
                "vocab_size": vocab_size,
                "blank_id": blank_id,
                "init_checkpoint": str(init_checkpoint) if init_checkpoint else None,
            }
        )

        stage1 = config["stage1"]
        adapter_cfg = config["phoneme_adapter"]
        encoder_dim = int(stage1.get("encoder_output_dim", 144))

        self.stream_policy = resolve_stream_policy(stage1)
        self.encoder = build_encoder(stage1, output_dim=encoder_dim)
        self.adapter = build_phoneme_adapter(
            adapter_cfg,
            input_dim=encoder_dim,
            vocab_size=vocab_size,
            causal=bool(stage1.get("causal", False)),
            blank_id=blank_id,
        )

        self._adapter_cfg = adapter_cfg
        self.blank_id = blank_id
        self.num_decode_batches = int((adapter_cfg.get("validation", {}) or {}).get(
            "num_decode_batches", 0
        ))

        if init_checkpoint:
            load_encoder_weights(self.encoder, Path(init_checkpoint), self.stream_policy)

        for param in self.encoder.parameters():
            param.requires_grad = False
        self.encoder.eval()

    def train(self, mode: bool = True):
        """Keep the frozen encoder in ``eval()`` across every phase switch.

        Overridden rather than handled in ``on_train_epoch_start`` because
        Lightning calls ``train()`` on entry to each phase, which would otherwise
        put the encoder back into train mode and re-enable its dropout.
        """
        super().train(mode)
        self.encoder.eval()
        return self

    def forward(
        self,
        feats: torch.Tensor,
        feat_lengths: torch.Tensor,
        *,
        mode: str = "eval",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            encoder_out, encoder_mask = run_encoder(
                self.encoder,
                feats,
                feat_lengths,
                policy=self.stream_policy,
                mode=mode,
            )
        return encoder_out, encoder_mask

    def _ctc_step(self, batch: dict[str, torch.Tensor], *, mode: str):
        encoder_out, encoder_mask = self(batch["feats"], batch["feat_lengths"], mode=mode)
        _features, log_probs = self.adapter(encoder_out, encoder_mask)
        loss, num_skipped = self.adapter.ctc_loss(
            log_probs,
            encoder_mask,
            batch["targets"],
            batch["target_lengths"],
        )
        return loss, log_probs, encoder_mask, num_skipped

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        loss, _, _, num_skipped = self._ctc_step(batch, mode="train")
        batch_size = batch["feats"].size(0)
        self.log("train/loss", loss, on_step=True, prog_bar=True, batch_size=batch_size)
        self.log("train/ctc_skipped", float(num_skipped), on_step=True, batch_size=batch_size)
        self.log(
            "train/ctc_skip_rate",
            num_skipped / max(batch_size, 1),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
        )

        optimizer = self.optimizers()
        self.log(
            "train/lr",
            optimizer.param_groups[0]["lr"],
            on_step=True,
            prog_bar=True,
            batch_size=batch_size,
        )
        return loss

    def on_validation_epoch_start(self) -> None:
        self._val_total_dist = 0
        self._val_total_ref = 0

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        loss, log_probs, encoder_mask, _ = self._ctc_step(batch, mode="eval")
        self.log("val/loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)

        if self.num_decode_batches and batch_idx >= self.num_decode_batches:
            return

        encoder_lens = encoder_mask.squeeze(1).sum(dim=1).to(dtype=torch.long)
        preds = log_probs.argmax(dim=2)
        targets = batch["targets"]
        target_lengths = batch["target_lengths"]

        for i in range(preds.size(0)):
            frames = int(encoder_lens[i].item())
            hyp_ids = collapse_ctc(preds[i, :frames].tolist(), blank_id=self.blank_id)
            ref_len = int(target_lengths[i].item())
            ref_ids = targets[i, :ref_len].tolist()
            self._val_total_dist += edit_distance(ref_ids, hyp_ids)
            self._val_total_ref += ref_len

    def on_validation_epoch_end(self) -> None:
        if self._val_total_ref > 0:
            per = self._val_total_dist / self._val_total_ref
            self.log("val/per", per, prog_bar=True, sync_dist=True)
            # Checkpoint-filename alias, kept out of CSV/TensorBoard.
            self.log("val_per", per, sync_dist=True, logger=False)

    def configure_optimizers(self) -> dict:
        return build_cosine_warmup_optimizer(self.adapter, self._adapter_cfg)


def load_encoder_weights(encoder, checkpoint_path: Path, stream_policy) -> None:
    """Load encoder weights from an icefall or DMA-KWS checkpoint.

    Mirrors :meth:`dma_kws.stage2.module.Stage2LightningModule._load_init_checkpoint`
    so Step A and Stage II start from bit-identical encoder weights.
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    assert_stream_policy_matches(checkpoint, stream_policy, source=checkpoint_path)

    model_state = checkpoint.get("model")
    is_icefall_format = isinstance(model_state, dict) and any(
        key.startswith(("encoder_embed.", "encoder.")) for key in model_state
    )

    if is_icefall_format:
        icefall_states = extract_icefall_encoder_state(checkpoint_path)
        encoder_embed_state = icefall_states.get("encoder_embed", {})
        encoder_state = icefall_states.get("encoder", {})
        if hasattr(encoder, "encoder_embed") and encoder_embed_state:
            missing, unexpected = encoder.encoder_embed.load_state_dict(
                encoder_embed_state, strict=False
            )
            print(
                f"Loaded encoder_embed weights from {checkpoint_path}: "
                f"missing={len(missing)} unexpected={len(unexpected)}"
            )
        if hasattr(encoder, "encoder") and encoder_state:
            missing, unexpected = encoder.encoder.load_state_dict(encoder_state, strict=False)
            print(
                f"Loaded encoder weights from {checkpoint_path}: "
                f"missing={len(missing)} unexpected={len(unexpected)}"
            )
        return

    state = extract_state_dict(checkpoint)
    prefix = "encoder."
    encoder_state = {
        key[len(prefix) :]: value for key, value in state.items() if key.startswith(prefix)
    }
    if not encoder_state:
        raise SystemExit(
            f"{checkpoint_path} carries no encoder weights (looked for icefall 'model' keys "
            "and an 'encoder.' prefixed state dict)"
        )
    missing, unexpected = encoder.load_state_dict(encoder_state, strict=False)
    print(
        f"Loaded encoder weights from {checkpoint_path}: "
        f"missing={len(missing)} unexpected={len(unexpected)}"
    )
