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


def validate_negative_tail_loss(*, weight: float, fraction: float) -> None:
    """Validate the negative CVaR/top-k loss configuration."""
    if not math.isfinite(weight) or weight < 0:
        raise ValueError("negative_tail_weight must be finite and non-negative")
    if not math.isfinite(fraction) or not 0.0 < fraction <= 1.0:
        raise ValueError("negative_tail_fraction must be finite and in (0, 1]")


def negative_tail_cvar_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    fraction: float,
) -> torch.Tensor:
    """Mean BCE over the highest-scoring fraction of negative examples.

    For a negative target, BCE is monotonic in the logit, so selecting the
    largest per-example BCE values is exactly the same as selecting the most
    dangerous false alarms. ``ceil(fraction * num_negatives)`` samples are kept,
    with at least one sample whenever the batch contains a negative. An all-
    positive batch returns a differentiable zero.
    """
    validate_negative_tail_loss(weight=0.0, fraction=fraction)
    if logits.shape != labels.shape:
        raise ValueError(
            "negative tail loss expects logits and labels with identical shapes; "
            f"got {tuple(logits.shape)} and {tuple(labels.shape)}"
        )

    flat_logits = logits.reshape(-1)
    flat_labels = labels.reshape(-1)
    negative_logits = flat_logits[flat_labels.eq(0)]
    if negative_logits.numel() == 0:
        # An empty reduction retains a zero-gradient connection to ``logits``
        # without multiplying possibly non-finite positive logits by zero.
        return flat_logits[:0].sum()

    per_negative = F.softplus(negative_logits)
    keep = max(1, math.ceil(float(fraction) * per_negative.numel()))
    return per_negative.topk(keep, sorted=False).values.mean()


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
    valid_path_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """BCE on non-final valid prefixes, balanced by sample or token.

    The final valid position is the complete keyword path and is supervised only
    by utterance BCE. Removing it here avoids training the deployed scalar twice.
    A one-phone anchor therefore has no progress targets.
    """
    valid = seq_label_mask.bool()
    if valid_path_mask is not None:
        if valid_path_mask.ndim != 1 or valid_path_mask.size(0) != valid.size(0):
            raise ValueError(
                "valid_path_mask must have shape "
                f"[{valid.size(0)}], got {tuple(valid_path_mask.shape)}"
            )
        valid = valid & valid_path_mask.to(device=valid.device).bool().unsqueeze(1)
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
    negative_tail_weight: float = 0.0,
    negative_tail_fraction: float = 0.1,
    valid_path_mask: torch.Tensor | None = None,
    ctc_loss: torch.Tensor | None = None,
    ctc_weight: float = 0.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute utterance and non-final ordered-prefix losses for Stage II.

    ``negative_tail_weight`` optionally adds a CVaR/top-k penalty over the
    highest-scoring negative examples in the current batch. The default zero
    weight preserves the historical utterance BCE exactly. ``valid_path_mask``
    removes structurally impossible segmental examples from utterance, prefix,
    and negative-tail losses; those examples have a constant invalid score and
    therefore cannot supply a useful gradient. ``ctc_loss`` is the optional
    auxiliary phoneme CTC term computed on the
    adapter trunk. The final valid prefix is deliberately excluded because the
    deployed full-path logit is already supervised by ``utt_loss``.
    ``ctc_weight=0.0`` keeps this loss independent of the phoneme adapter.
    """
    validate_seq_loss_weights(seq_progress_weight)
    validate_negative_tail_loss(
        weight=negative_tail_weight,
        fraction=negative_tail_fraction,
    )
    seq_normalization = normalize_seq_loss_normalization(seq_normalization)

    if valid_path_mask is None:
        valid_samples = torch.ones_like(labels, dtype=torch.bool)
    else:
        if not isinstance(valid_path_mask, torch.Tensor):
            raise ValueError("valid_path_mask must be a tensor or None")
        if valid_path_mask.shape != labels.shape:
            raise ValueError(
                "valid_path_mask must match labels shape; "
                f"got {tuple(valid_path_mask.shape)} and {tuple(labels.shape)}"
            )
        valid_samples = valid_path_mask.to(device=logits.device).bool()
    safe_logits = torch.where(valid_samples, logits, torch.zeros_like(logits))
    safe_labels = torch.where(
        valid_samples,
        labels,
        torch.zeros_like(labels),
    ).float()
    per_sample_utt_loss = F.binary_cross_entropy_with_logits(
        safe_logits,
        safe_labels,
        reduction="none",
    )
    valid_weights = valid_samples.to(dtype=per_sample_utt_loss.dtype)
    utt_loss = (per_sample_utt_loss * valid_weights).sum() / valid_weights.sum().clamp_min(1.0)
    seq_progress_loss = _masked_progress_bce(
        seq_logits,
        seq_labels,
        seq_label_mask,
        normalization=seq_normalization,
        valid_path_mask=valid_samples,
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

    if negative_tail_weight:
        negative_tail_loss = negative_tail_cvar_loss(
            logits[valid_samples],
            labels[valid_samples],
            fraction=negative_tail_fraction,
        )
        negative_tail_weighted_loss = negative_tail_weight * negative_tail_loss
        total_loss = total_loss + negative_tail_weighted_loss
        losses["negative_tail_loss"] = negative_tail_loss
        losses["negative_tail_weighted_loss"] = negative_tail_weighted_loss

    if ctc_loss is not None and ctc_weight:
        ctc_weighted_loss = ctc_weight * ctc_loss
        total_loss = total_loss + ctc_weighted_loss
        losses["ctc_loss"] = ctc_loss
        losses["ctc_weighted_loss"] = ctc_weighted_loss

    losses["total_loss"] = total_loss

    return total_loss, losses
