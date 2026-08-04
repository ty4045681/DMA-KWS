"""Shared Stage II score extraction helpers.

The sequence head is padded to the batch's maximum anchor width.  Completion
must therefore be read from each sample's last *valid* anchor position, never
from the last padded column.
"""

from __future__ import annotations

import torch


def gather_last_valid_logits(
    seq_logits: torch.Tensor,
    valid_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather the final valid sequence logit and return its sample mask.

    Samples without a valid position receive a finite placeholder zero in the
    returned logits and are marked false in ``valid_samples``.  Callers must use
    the mask rather than treating that placeholder as a model score.
    """

    if seq_logits.ndim != 2 or valid_mask.ndim != 2:
        raise ValueError("seq_logits and valid_mask must both be rank-2 tensors")
    if seq_logits.shape != valid_mask.shape:
        raise ValueError(
            "seq_logits and valid_mask must have the same shape, got "
            f"{tuple(seq_logits.shape)} and {tuple(valid_mask.shape)}"
        )

    batch_size, width = seq_logits.shape
    if width == 0:
        return (
            seq_logits.new_zeros(batch_size),
            torch.zeros(batch_size, device=seq_logits.device, dtype=torch.bool),
        )

    valid = valid_mask.to(device=seq_logits.device, dtype=torch.bool)
    positions = torch.arange(width, device=seq_logits.device).unsqueeze(0)
    last_indices = torch.where(valid, positions, -1).amax(dim=1)
    valid_samples = last_indices.ge(0)
    gathered = seq_logits.gather(1, last_indices.clamp_min(0).unsqueeze(1)).squeeze(1)
    gathered = torch.where(valid_samples, gathered, torch.zeros_like(gathered))
    return gathered, valid_samples


def gather_completion_logits(
    seq_logits: torch.Tensor,
    anchor_lengths: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather completion logits using one valid prefix per anchor length."""

    if seq_logits.ndim != 2:
        raise ValueError("seq_logits must be a rank-2 tensor")
    lengths = anchor_lengths.to(device=seq_logits.device, dtype=torch.long).reshape(-1)
    if lengths.numel() != seq_logits.size(0):
        raise ValueError(
            "anchor_lengths must contain one value per sample, got "
            f"{lengths.numel()} for batch size {seq_logits.size(0)}"
        )
    if bool(lengths.lt(0).any()):
        raise ValueError("anchor_lengths must be non-negative")
    if bool(lengths.gt(seq_logits.size(1)).any()):
        raise ValueError(
            "anchor_lengths cannot exceed the sequence-logit width "
            f"{seq_logits.size(1)}"
        )

    positions = torch.arange(seq_logits.size(1), device=seq_logits.device).unsqueeze(0)
    valid_mask = positions < lengths.unsqueeze(1)
    return gather_last_valid_logits(seq_logits, valid_mask)


__all__ = ["gather_completion_logits", "gather_last_valid_logits"]
