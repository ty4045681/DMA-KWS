"""Step A: train the phoneme CTC trunk on a frozen encoder."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytorch_lightning as pl
import torch
import torch.distributed as dist
import torchmetrics

from dma_kws.config import resolve_stream_policy
from dma_kws.metrics import collapse_ctc, edit_distance
from dma_kws.nn import build_encoder, run_encoder
from dma_kws.phoneme_adapter.module import build_phoneme_adapter
from dma_kws.training.checkpoint_io import (
    assert_stream_policy_matches,
    extract_icefall_encoder_state,
    extract_state_dict,
)
from dma_kws.training.distributed_metrics import (
    ddp_global_mean_loss,
    gather_unique_sample_values,
    sum_across_processes,
)
from dma_kws.training.ddp import rank_zero_print
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
        adapter_init_checkpoint: str | Path | None = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(
            {
                "vocab_size": vocab_size,
                "blank_id": blank_id,
                "init_checkpoint": str(init_checkpoint) if init_checkpoint else None,
                "adapter_init_checkpoint": (
                    str(adapter_init_checkpoint) if adapter_init_checkpoint else None
                ),
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
        self._train_window_loss = torchmetrics.MeanMetric(sync_on_compute=True)
        self._train_window_microbatches = 0
        self._train_window_ctc_valid = 0
        self._train_window_ctc_skipped = 0

        if init_checkpoint:
            load_encoder_weights(self.encoder, Path(init_checkpoint), self.stream_policy)
        if adapter_init_checkpoint:
            load_adapter_weights(
                self.adapter,
                Path(adapter_init_checkpoint),
                self.stream_policy,
                blank_id=blank_id,
            )

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

    def _ctc_step(
        self,
        batch: dict[str, torch.Tensor],
        *,
        mode: str,
        return_details: bool = False,
    ):
        encoder_out, encoder_mask = self(batch["feats"], batch["feat_lengths"], mode=mode)
        _features, log_probs = self.adapter(encoder_out, encoder_mask)
        loss_args = (
            log_probs,
            encoder_mask,
            batch["targets"],
            batch["target_lengths"],
        )
        loss_result = (
            self.adapter.ctc_loss(*loss_args, return_details=True)
            if return_details
            else self.adapter.ctc_loss(*loss_args)
        )
        if return_details:
            loss, num_skipped, per_sample_losses, valid_mask = loss_result
            return (
                loss,
                log_probs,
                encoder_mask,
                num_skipped,
                per_sample_losses,
                valid_mask,
            )
        loss, num_skipped = loss_result
        return loss, log_probs, encoder_mask, num_skipped

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        loss, _, _, num_skipped = self._ctc_step(batch, mode="train")
        batch_size = batch["feats"].size(0)
        local_valid = batch_size - num_skipped
        global_stats = sum_across_processes(
            torch.tensor(
                [float(loss.detach()) * local_valid, local_valid, num_skipped],
                device=loss.device,
                dtype=torch.float64,
            )
        )
        global_loss_sum, valid_total, skipped_total = global_stats.unbind()
        global_valid = int(valid_total.item())
        global_skipped = int(skipped_total.item())
        global_batch_size = global_valid + global_skipped
        loss = ddp_global_mean_loss(
            loss,
            local_count=local_valid,
            global_sum=global_loss_sum,
            global_count=global_valid,
        )
        self._train_window_loss.update(loss.detach(), weight=global_valid)
        self._train_window_microbatches += 1
        self._train_window_ctc_valid += global_valid
        self._train_window_ctc_skipped += global_skipped

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
            prog_bar=True,
            batch_size=global_batch_size,
        )
        self.log(
            "train/epoch/ctc_skip_rate",
            global_skipped / max(global_batch_size, 1),
            on_step=False,
            on_epoch=True,
            batch_size=global_batch_size,
        )

        optimizer = self.optimizers()
        self.log(
            "train/optimizer/lr",
            optimizer.param_groups[0]["lr"],
            on_step=True,
            prog_bar=True,
            batch_size=batch_size,
        )
        self.log(
            "train/lr",
            optimizer.param_groups[0]["lr"],
            on_step=True,
            prog_bar=False,
            logger=False,
            batch_size=batch_size,
        )
        return loss

    def _log_train_window(self) -> None:
        if self._train_window_microbatches <= 0:
            return
        total = self._train_window_ctc_valid + self._train_window_ctc_skipped
        if self._train_window_ctc_valid > 0:
            self.log(
                "train/window/loss_total",
                self._train_window_loss.compute(),
                sync_dist=True,
            )
        self.log(
            "train/window/ctc_valid",
            float(self._train_window_ctc_valid),
            sync_dist=True,
        )
        self.log(
            "train/window/ctc_skipped",
            float(self._train_window_ctc_skipped),
            sync_dist=True,
        )
        self.log(
            "train/window/ctc_skip_rate",
            self._train_window_ctc_skipped / max(total, 1),
            sync_dist=True,
        )
        self.log(
            "train/window/microbatches",
            float(self._train_window_microbatches),
            sync_dist=True,
        )
        self._train_window_loss.reset()
        self._train_window_microbatches = 0
        self._train_window_ctc_valid = 0
        self._train_window_ctc_skipped = 0

    def on_validation_epoch_start(self) -> None:
        self._log_train_window()
        self._val_ctc_loss_sum = 0.0
        self._val_ctc_valid = 0
        self._val_ctc_skipped = 0
        self._val_total_dist = 0
        self._val_total_ref = 0
        self._val_ctc_records: list[tuple[int, float, int, int]] = []
        self._val_per_records: list[tuple[int, int, int]] = []
        self._val_ctc_records_complete = True
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

        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        world_size = (
            dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
        )
        start = self._val_synthetic_sample_index
        self._val_synthetic_sample_index += batch_size
        return [-(1 + rank + world_size * (start + offset)) for offset in range(batch_size)]

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        result = self._ctc_step(batch, mode="eval", return_details=True)
        if len(result) == 6:
            (
                loss,
                log_probs,
                encoder_mask,
                num_skipped,
                per_sample_losses,
                valid_mask,
            ) = result
        else:
            # Compatibility with tests/extensions overriding the private helper
            # before per-sample CTC details were available.
            loss, log_probs, encoder_mask, num_skipped = result
            per_sample_losses = None
            valid_mask = None
            self._val_ctc_records_complete = False
        batch_size = int(batch["feats"].size(0))
        sample_ids = self._validation_sample_ids(batch, batch_size)
        num_valid = batch_size - num_skipped
        # ``ctc_loss`` is a mean over retained samples. Recover its numerator so
        # batches/ranks with different skip counts get the correct weight.
        self._val_ctc_loss_sum += float(loss.detach()) * num_valid
        self._val_ctc_valid += num_valid
        self._val_ctc_skipped += num_skipped
        if per_sample_losses is not None and valid_mask is not None:
            losses = per_sample_losses.detach().reshape(-1).cpu()
            valid = valid_mask.detach().reshape(-1).cpu()
            if losses.numel() != batch_size or valid.numel() != batch_size:
                raise ValueError(
                    "Per-sample CTC details must contain one value per batch sample"
                )
            for sample_id, sample_loss, is_valid in zip(sample_ids, losses, valid):
                keep = bool(is_valid.item())
                self._val_ctc_records.append(
                    (sample_id, float(sample_loss.item()), int(keep), int(not keep))
                )

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
            distance = edit_distance(ref_ids, hyp_ids)
            self._val_total_dist += distance
            self._val_total_ref += ref_len
            self._val_per_records.append((sample_ids[i], distance, ref_len))

    def on_validation_epoch_end(self) -> None:
        ctc_records = getattr(self, "_val_ctc_records", [])
        ctc_ids = torch.tensor(
            [record[0] for record in ctc_records],
            device=self.device,
            dtype=torch.long,
        )
        ctc_values = torch.tensor(
            [[record[1], record[2], record[3]] for record in ctc_records],
            device=self.device,
            dtype=torch.float64,
        ).reshape(-1, 3)
        unique_ctc = gather_unique_sample_values(ctc_ids, ctc_values)

        per_records = getattr(self, "_val_per_records", [])
        per_ids = torch.tensor(
            [record[0] for record in per_records],
            device=self.device,
            dtype=torch.long,
        )
        per_values = torch.tensor(
            [[record[1], record[2]] for record in per_records],
            device=self.device,
            dtype=torch.float64,
        ).reshape(-1, 2)
        unique_per = gather_unique_sample_values(per_ids, per_values)

        records_complete = bool(getattr(self, "_val_ctc_records_complete", False))
        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            complete_flag = torch.tensor(
                int(records_complete), device=self.device, dtype=torch.long
            )
            dist.all_reduce(complete_flag, op=dist.ReduceOp.MIN)
            records_complete = bool(complete_flag.item())

        if unique_ctc.size(0) == 0 and unique_per.size(0) == 0:
            # Legacy/directly-seeded counters retain the original one-reduction
            # behavior and API.
            totals = sum_across_processes(
                torch.tensor(
                    [
                        self._val_ctc_loss_sum,
                        self._val_ctc_valid,
                        self._val_ctc_skipped,
                        self._val_total_dist,
                        self._val_total_ref,
                    ],
                    device=self.device,
                    dtype=torch.float64,
                )
            )
            loss_sum, num_valid, num_skipped, total_dist, total_ref = totals.unbind()
        else:
            if records_complete and unique_ctc.size(0):
                loss_sum, num_valid, num_skipped = unique_ctc.sum(dim=0).unbind()
            else:
                ctc_totals = sum_across_processes(
                    torch.tensor(
                        [
                            self._val_ctc_loss_sum,
                            self._val_ctc_valid,
                            self._val_ctc_skipped,
                        ],
                        device=self.device,
                        dtype=torch.float64,
                    )
                )
                loss_sum, num_valid, num_skipped = ctc_totals.unbind()

            if unique_per.size(0):
                total_dist, total_ref = unique_per.sum(dim=0).unbind()
            else:
                per_totals = sum_across_processes(
                    torch.tensor(
                        [self._val_total_dist, self._val_total_ref],
                        device=self.device,
                        dtype=torch.float64,
                    )
                )
                total_dist, total_ref = per_totals.unbind()

        no_value = loss_sum.new_tensor(float("nan"))
        val_loss = loss_sum / num_valid if num_valid.item() > 0 else no_value
        total_ctc_samples = num_valid + num_skipped
        skip_rate = (
            num_skipped / total_ctc_samples
            if total_ctc_samples.item() > 0
            else no_value
        )
        per = total_dist / total_ref if total_ref.item() > 0 else no_value

        # Values are already identical global sums/ratios. Lightning has no
        # "already synchronized" flag, so its mean reduction is an identity
        # operation used to avoid one warning per metric under DDP.
        self.log("val/loss", val_loss, prog_bar=True, sync_dist=True)
        self.log("val/ctc_valid", num_valid, sync_dist=True)
        self.log("val/ctc_skipped", num_skipped, sync_dist=True)
        self.log("val/ctc_skip_rate", skip_rate, sync_dist=True)
        self.log("val/per_edit_distance", total_dist, sync_dist=True)
        self.log("val/per_reference_tokens", total_ref, sync_dist=True)
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
            rank_zero_print(
                f"Loaded encoder_embed weights from {checkpoint_path}: "
                f"missing={len(missing)} unexpected={len(unexpected)}"
            )
        if hasattr(encoder, "encoder") and encoder_state:
            missing, unexpected = encoder.encoder.load_state_dict(encoder_state, strict=False)
            rank_zero_print(
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
    rank_zero_print(
        f"Loaded encoder weights from {checkpoint_path}: "
        f"missing={len(missing)} unexpected={len(unexpected)}"
    )


def load_adapter_weights(
    adapter,
    checkpoint_path: Path,
    stream_policy,
    *,
    blank_id: int,
) -> None:
    """Strictly initialize Step A from a previously exported adapter.

    Unlike Lightning resume, this path restores only adapter parameters and
    intentionally starts a fresh optimizer, scheduler, and global step for the
    new domain.  The encoder remains controlled independently by
    ``init_checkpoint``.
    """
    if not checkpoint_path.is_file():
        raise SystemExit(f"Phoneme adapter init checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise SystemExit(
            f"{checkpoint_path} is not a phoneme adapter export: expected a mapping payload"
        )

    checkpoint_kind = checkpoint.get("checkpoint_kind")
    if checkpoint_kind not in (None, "phoneme_adapter"):
        raise SystemExit(
            f"{checkpoint_path} has checkpoint_kind={checkpoint_kind!r}; expected "
            "a Step A phoneme_adapter export."
        )

    saved_config = checkpoint.get("config")
    if not isinstance(saved_config, dict) or "stage1" not in saved_config:
        raise SystemExit(
            f"{checkpoint_path} has no embedded stage1 config, so its streaming "
            "operating point cannot be verified. Use adapter_best_step*.pt or "
            "adapter_final_step*.pt exported by the Step A trainer."
        )
    assert_stream_policy_matches(checkpoint, stream_policy, source=checkpoint_path)

    saved_blank_id = checkpoint.get("blank_id")
    if saved_blank_id is None:
        raise SystemExit(
            f"{checkpoint_path} has no blank_id metadata, so it cannot be safely "
            "used for phoneme adapter warm-start."
        )
    if int(saved_blank_id) != int(blank_id):
        raise SystemExit(
            f"{checkpoint_path} was trained with blank_id={int(saved_blank_id)} but "
            f"the current tokenizer uses blank_id={int(blank_id)}."
        )

    saved_vocab_size = checkpoint.get("vocab_size")
    if saved_vocab_size is not None and int(saved_vocab_size) != int(adapter.vocab_size):
        raise SystemExit(
            f"{checkpoint_path} was trained with vocab_size={int(saved_vocab_size)} but "
            f"the current adapter uses vocab_size={int(adapter.vocab_size)}."
        )

    state = extract_state_dict(checkpoint)
    if not isinstance(state, dict) or not state:
        raise SystemExit(f"{checkpoint_path} carries no adapter model_state_dict")

    # Step A exports ``model.adapter`` directly, so accepting prefixed Stage II
    # or full Lightning states here would make it too easy to initialize from
    # the wrong artifact. strict=True also rejects missing tensors and every
    # trunk/CTC shape mismatch.
    adapter.load_state_dict(state, strict=True)
    rank_zero_print(f"Loaded phoneme adapter warm-start weights from {checkpoint_path}")
