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
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute utterance-level and sequence-level BCE losses for Stage II training."""
    utt_loss = F.binary_cross_entropy_with_logits(logits, labels.float())
    seq_loss = F.binary_cross_entropy_with_logits(
        seq_logits,
        seq_labels.float(),
        weight=seq_label_mask,
        reduction="sum",
    ) / (seq_label_mask.sum() + 1e-6)
    total_loss = utt_loss + seq_loss
    return total_loss, {"utt_loss": utt_loss, "seq_loss": seq_loss}
