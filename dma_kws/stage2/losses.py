"""Stage II utterance + sequence loss helpers."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

SEQ_LOSS_NORMALIZATIONS = frozenset({"sample", "token"})


def validate_seq_loss_weights(progress_weight: float) -> None:
    """Reject a negative or non-finite progress-loss coefficient."""
    if not math.isfinite(progress_weight) or progress_weight < 0:
        raise ValueError("Sequence loss progress_weight must be finite and non-negative")


def normalize_seq_loss_normalization(normalization: str) -> str:
    """Validate how the progress BCE is reduced across a batch."""
    normalized = str(normalization).strip().lower()
    if normalized not in SEQ_LOSS_NORMALIZATIONS:
        choices = ", ".join(sorted(SEQ_LOSS_NORMALIZATIONS))
        raise ValueError(
            f"Unsupported seq loss normalization {normalization!r}; "
            f"expected one of: {choices}"
        )
    return normalized


def _masked_progress_bce(
    seq_logits: torch.Tensor,
    seq_labels: torch.Tensor,
    seq_label_mask: torch.Tensor,
    *,
    normalization: str,
) -> torch.Tensor:
    """BCE on non-final valid prefixes, balanced by sample or token.

    The final valid position is the complete keyword path and is supervised only
    by utterance BCE. Removing it here avoids training the deployed scalar twice.
    A one-phone anchor therefore has no progress targets.
    """
    valid = seq_label_mask.bool()
    width = valid.size(1)
    if width:
        positions = torch.arange(width, device=valid.device).unsqueeze(0)
        last_valid = torch.where(valid, positions, -1).amax(dim=1)
        valid = valid & positions.ne(last_valid.unsqueeze(1))
    # Do not evaluate BCE on padded targets (-1). Besides making the target
    # contract explicit, replacing both operands prevents 0 * inf from becoming
    # NaN if a padded logit ever overflows.
    safe_logits = torch.where(valid, seq_logits, torch.zeros_like(seq_logits))
    safe_labels = torch.where(
        valid,
        seq_labels,
        torch.zeros_like(seq_labels),
    ).float()
    per_position = F.binary_cross_entropy_with_logits(
        safe_logits,
        safe_labels,
        reduction="none",
    )
    mask = valid.to(dtype=per_position.dtype)

    if normalization == "token":
        return (per_position * mask).sum() / mask.sum().clamp_min(1.0)

    valid_counts = mask.sum(dim=1)
    per_sample = (per_position * mask).sum(dim=1) / valid_counts.clamp_min(1.0)
    valid_samples = valid_counts.gt(0)
    return (per_sample * valid_samples).sum() / valid_samples.sum().clamp_min(1)


def compute_stage2_losses(
    logits: torch.Tensor,
    seq_logits: torch.Tensor,
    labels: torch.Tensor,
    seq_labels: torch.Tensor,
    seq_label_mask: torch.Tensor,
    *,
    seq_progress_weight: float = 0.3,
    seq_normalization: str = "sample",
    ctc_loss: torch.Tensor | None = None,
    ctc_weight: float = 0.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute utterance and non-final ordered-prefix losses for Stage II.

    ``ctc_loss`` is the optional auxiliary phoneme CTC term computed on the
    adapter trunk. The final valid prefix is deliberately excluded because the
    deployed full-path logit is already supervised by ``utt_loss``.
    ``ctc_weight=0.0`` keeps this loss independent of the phoneme adapter.
    """
    validate_seq_loss_weights(seq_progress_weight)
    seq_normalization = normalize_seq_loss_normalization(seq_normalization)

    utt_loss = F.binary_cross_entropy_with_logits(logits, labels.float())
    seq_progress_loss = _masked_progress_bce(
        seq_logits,
        seq_labels,
        seq_label_mask,
        normalization=seq_normalization,
    )
    seq_progress_weighted_loss = seq_progress_weight * seq_progress_loss
    seq_loss = seq_progress_weighted_loss
    total_loss = utt_loss + seq_loss
    losses = {
        "utt_loss": utt_loss,
        "seq_loss": seq_loss,
        "seq_progress_loss": seq_progress_loss,
        "seq_progress_weighted_loss": seq_progress_weighted_loss,
    }

    if ctc_loss is not None and ctc_weight:
        ctc_weighted_loss = ctc_weight * ctc_loss
        total_loss = total_loss + ctc_weighted_loss
        losses["ctc_loss"] = ctc_loss
        losses["ctc_weighted_loss"] = ctc_weighted_loss

    losses["total_loss"] = total_loss

    return total_loss, losses
