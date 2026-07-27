"""Stage II utterance + sequence loss helpers."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def compute_stage2_losses(
    logits: torch.Tensor,
    seq_logits: torch.Tensor,
    labels: torch.Tensor,
    seq_labels: torch.Tensor,
    seq_label_mask: torch.Tensor,
    *,
    ctc_loss: torch.Tensor | None = None,
    ctc_weight: float = 0.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute utterance-level and sequence-level BCE losses for Stage II training.

    ``ctc_loss`` is the optional auxiliary phoneme CTC term computed on the
    adapter trunk. It exists because the two BCE terms are weak supervision for a
    cross-space mapping: ``seq_labels`` is per-anchor-phoneme *set membership*,
    with no ordering, position or count information, so on its own it lets the
    matcher settle for whole-word acoustic templates instead of compositional
    phoneme matching. ``ctc_weight=0.0`` reproduces the pre-adapter loss exactly.
    """
    utt_loss = F.binary_cross_entropy_with_logits(logits, labels.float())
    seq_loss = F.binary_cross_entropy_with_logits(
        seq_logits,
        seq_labels.float(),
        weight=seq_label_mask,
        reduction="sum",
    ) / (seq_label_mask.sum() + 1e-6)
    total_loss = utt_loss + seq_loss
    losses = {"utt_loss": utt_loss, "seq_loss": seq_loss}

    if ctc_loss is not None and ctc_weight:
        total_loss = total_loss + ctc_weight * ctc_loss
        losses["ctc_loss"] = ctc_loss

    return total_loss, losses
