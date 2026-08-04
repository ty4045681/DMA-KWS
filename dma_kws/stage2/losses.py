"""Stage II utterance + sequence loss helpers."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


SEQ_LOSS_NORMALIZATIONS = frozenset({"sample", "token"})


def validate_seq_loss_weights(progress_weight: float, completion_weight: float) -> None:
    """Reject negative or non-finite sequence-loss coefficients."""
    for name, value in (
        ("progress_weight", progress_weight),
        ("completion_weight", completion_weight),
    ):
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"Sequence loss {name} must be finite and non-negative")


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
    """Masked position BCE, balanced either by sample or by valid token."""
    valid = seq_label_mask.bool()
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


def _completion_bce(
    seq_logits: torch.Tensor,
    seq_labels: torch.Tensor,
    seq_label_mask: torch.Tensor,
) -> torch.Tensor:
    """BCE on each sample's last valid anchor position.

    With cumulative prefix targets, that position is one exactly when the whole
    anchor occurs in order and contiguously in the query.
    """
    valid = seq_label_mask.bool()
    if seq_logits.size(1) == 0:
        return seq_logits.sum() * 0.0

    positions = torch.arange(
        seq_logits.size(1),
        device=seq_logits.device,
    ).unsqueeze(0)
    last_indices = torch.where(valid, positions, -1).amax(dim=1)
    valid_samples = last_indices.ge(0)
    gather_indices = last_indices.clamp_min(0).unsqueeze(1)
    completion_logits = seq_logits.gather(1, gather_indices).squeeze(1)
    completion_labels = seq_labels.gather(1, gather_indices).squeeze(1).float()
    safe_logits = torch.where(
        valid_samples,
        completion_logits,
        torch.zeros_like(completion_logits),
    )
    safe_labels = torch.where(
        valid_samples,
        completion_labels,
        torch.zeros_like(completion_labels),
    )
    per_sample = F.binary_cross_entropy_with_logits(
        safe_logits,
        safe_labels,
        reduction="none",
    )
    mask = valid_samples.to(dtype=per_sample.dtype)
    return (per_sample * mask).sum() / mask.sum().clamp_min(1.0)


def compute_stage2_losses(
    logits: torch.Tensor,
    seq_logits: torch.Tensor,
    labels: torch.Tensor,
    seq_labels: torch.Tensor,
    seq_label_mask: torch.Tensor,
    *,
    seq_progress_weight: float = 0.5,
    seq_completion_weight: float = 0.5,
    seq_normalization: str = "sample",
    ctc_loss: torch.Tensor | None = None,
    ctc_weight: float = 0.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute utterance, ordered-progress and completion losses for Stage II.

    ``ctc_loss`` is the optional auxiliary phoneme CTC term computed on the
    adapter trunk. When ``seq_completion_weight`` is non-zero, ``seq_labels``
    must be cumulative ordered-prefix targets (``111...000``), making the last
    valid target the full-keyword label. A progress-only membership target is
    retained for explicit legacy ablations. ``ctc_weight=0.0`` keeps this loss
    independent of the phoneme adapter.
    """
    validate_seq_loss_weights(seq_progress_weight, seq_completion_weight)
    seq_normalization = normalize_seq_loss_normalization(seq_normalization)

    utt_loss = F.binary_cross_entropy_with_logits(logits, labels.float())
    seq_progress_loss = _masked_progress_bce(
        seq_logits,
        seq_labels,
        seq_label_mask,
        normalization=seq_normalization,
    )
    seq_completion_loss = _completion_bce(
        seq_logits,
        seq_labels,
        seq_label_mask,
    )
    seq_loss = (
        seq_progress_weight * seq_progress_loss
        + seq_completion_weight * seq_completion_loss
    )
    total_loss = utt_loss + seq_loss
    losses = {
        "utt_loss": utt_loss,
        "seq_loss": seq_loss,
        "seq_progress_loss": seq_progress_loss,
        "seq_completion_loss": seq_completion_loss,
    }

    if ctc_loss is not None and ctc_weight:
        total_loss = total_loss + ctc_weight * ctc_loss
        losses["ctc_loss"] = ctc_loss

    return total_loss, losses
