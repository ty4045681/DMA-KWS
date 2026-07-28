"""Phoneme CTC adapter: the trunk QbyT and the CTC loss share.

The point of the shared trunk is that CTC supervision shapes the exact tensor
QbyT consumes. A CTC head bolted onto the frozen encoder as a separate branch
would only serve Stage I phoneme search and would leave Stage II's audio input
in the encoder's BPE/transducer space, which is what the phoneme text embedding
cannot be compared against.
"""

from __future__ import annotations

from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from dma_kws.pathing import ensure_qbyt_on_path
from dma_kws.phoneme_adapter.trunk import build_trunk


def _load_ctc():
    ensure_qbyt_on_path()
    from models.ctc import CTC

    return CTC


def ctc_min_input_lengths(
    targets: torch.Tensor,
    target_lengths: torch.Tensor,
) -> torch.Tensor:
    """Return the minimum number of frames needed to align each CTC target.

    CTC has to emit a blank between adjacent identical labels, so a target such
    as ``[AA, AA, K]`` needs four input frames rather than three. ``targets`` is
    the padded ``[batch, max_target_length]`` representation used throughout
    this project; padding beyond each entry in ``target_lengths`` is ignored.
    """
    if targets.ndim != 2:
        raise ValueError(
            f"Expected padded 2-D CTC targets [B, U], got shape {tuple(targets.shape)}"
        )
    if target_lengths.ndim != 1 or target_lengths.numel() != targets.size(0):
        raise ValueError(
            "target_lengths must be a 1-D tensor with one entry per target row; "
            f"got shape {tuple(target_lengths.shape)} for targets {tuple(targets.shape)}"
        )

    target_lengths = target_lengths.to(device=targets.device, dtype=torch.long)
    max_target_length = targets.size(1)
    invalid_lengths = (target_lengths < 0) | (target_lengths > max_target_length)
    if bool(invalid_lengths.any()):
        raise ValueError(
            f"target_lengths must be in [0, {max_target_length}], got "
            f"{target_lengths.detach().cpu().tolist()}"
        )

    if max_target_length < 2:
        return target_lengths

    # Pair position j compares targets[j - 1] with targets[j]. A pair is valid
    # only when its right-hand label is inside the unpadded target sequence.
    pair_positions = torch.arange(1, max_target_length, device=targets.device)
    valid_pairs = pair_positions.unsqueeze(0) < target_lengths.unsqueeze(1)
    adjacent_repeats = (
        (targets[:, 1:] == targets[:, :-1]) & valid_pairs
    ).sum(dim=1)
    return target_lengths + adjacent_repeats


class PhonemeAdapter(nn.Module):
    """Trunk + phoneme CTC output projection.

    ``forward`` returns both the representation Stage II reads and the CTC
    log-probabilities, so a single forward pass serves the phoneme loss, Stage I
    phoneme search and the QbyT matcher.
    """

    def __init__(
        self,
        *,
        input_dim: int,
        vocab_size: int,
        trunk_cfg: Mapping[str, Any],
        causal: bool = False,
        ctc_dropout: float = 0.0,
        expose_posterior: bool = False,
        blank_id: int = 0,
    ) -> None:
        super().__init__()
        self.trunk = build_trunk(trunk_cfg, input_dim=input_dim, causal=causal)
        self.vocab_size = vocab_size
        self.blank_id = blank_id
        self.expose_posterior = expose_posterior

        CTC = _load_ctc()
        self.ctc = CTC(
            odim=vocab_size,
            encoder_output_size=self.trunk.output_dim,
            dropout_rate=ctc_dropout,
            blank_id=blank_id,
        )

    @property
    def output_dim(self) -> int:
        """Width of the tensor handed to Stage II."""
        if self.expose_posterior:
            return self.trunk.output_dim + self.vocab_size
        return self.trunk.output_dim

    def forward(
        self,
        encoder_out: torch.Tensor,
        encoder_mask: torch.Tensor | None = None,
        *,
        with_log_probs: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Return ``(features_for_stage2, ctc_log_probs)``.

        ``ctc_log_probs`` is always computed from the trunk output, never from
        the raw encoder output, so the two consumers cannot drift apart.

        ``with_log_probs=False`` skips the CTC projection entirely for callers
        that do not use it (inference, or Stage II with ``ctc_weight=0``).
        Computing it and discarding the result leaves ``ctc.ctc_lo`` without a
        gradient, which makes DDP abort unless ``find_unused_parameters`` happens
        to be on.
        """
        hidden = self.trunk(encoder_out, encoder_mask)
        if not (with_log_probs or self.expose_posterior):
            return hidden, None

        log_probs = self.ctc.log_softmax(hidden)
        if self.expose_posterior:
            return torch.cat([hidden, log_probs], dim=-1), log_probs
        return hidden, log_probs

    def ctc_loss(
        self,
        log_probs: torch.Tensor,
        encoder_mask: torch.Tensor | None,
        targets: torch.Tensor,
        target_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, int]:
        """CTC loss over log-probabilities already produced by :meth:`forward`.

        Taking ``log_probs`` rather than the encoder output is deliberate: it
        keeps the trunk to one forward pass per step, and more importantly makes
        the CTC loss supervise the *same* stochastic realization QbyT reads.
        Running the trunk twice would let dropout hand the two consumers
        different tensors, which is precisely the coupling this module exists to
        create.

        Returns ``(loss, num_skipped)``. In addition to needing at least one
        frame per label, CTC needs an intervening blank frame for every adjacent
        repeated label. At 25 Hz a short phrase clip can genuinely fail either
        requirement. Those samples are dropped explicitly so the count stays
        visible instead of being silently zeroed by ``zero_infinity``.
        """
        batch, frames = log_probs.size(0), log_probs.size(1)
        if encoder_mask is None:
            input_lengths = torch.full(
                (batch,), frames, dtype=torch.long, device=log_probs.device
            )
        else:
            input_lengths = encoder_mask.squeeze(1).sum(dim=1).to(
                device=log_probs.device, dtype=torch.long
            )

        targets = targets.to(device=log_probs.device, dtype=torch.long)
        target_lengths = target_lengths.to(device=log_probs.device, dtype=torch.long)
        min_input_lengths = ctc_min_input_lengths(targets, target_lengths)

        target_positions = torch.arange(targets.size(1), device=targets.device).unsqueeze(0)
        valid_target_mask = target_positions < target_lengths.unsqueeze(1)
        if bool(((targets == self.blank_id) & valid_target_mask).any()):
            raise ValueError(
                f"CTC targets contain blank_id={self.blank_id} inside valid target positions"
            )

        keep = (target_lengths > 0) & (min_input_lengths <= input_lengths)
        num_skipped = int((~keep).sum().item())
        if not bool(keep.any()):
            return log_probs.sum() * 0.0, num_skipped

        # Batch-size average, matching qbyt/models/ctc.py.
        loss = F.ctc_loss(
            log_probs[keep].transpose(0, 1),
            targets[keep],
            input_lengths[keep],
            target_lengths[keep],
            blank=self.blank_id,
            reduction="sum",
            zero_infinity=False,
        )
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(
                "CTC loss became non-finite after infeasible targets were filtered; "
                "check target ids, lengths, and log-probabilities"
            )
        return loss / int(keep.sum()), num_skipped


def build_phoneme_adapter(
    adapter_cfg: Mapping[str, Any],
    *,
    input_dim: int,
    vocab_size: int,
    causal: bool = False,
    blank_id: int = 0,
) -> PhonemeAdapter:
    """Build a :class:`PhonemeAdapter` from a ``phoneme_adapter`` config section."""
    return PhonemeAdapter(
        input_dim=input_dim,
        vocab_size=vocab_size,
        trunk_cfg=adapter_cfg.get("trunk", {}) or {},
        causal=causal,
        ctc_dropout=float(adapter_cfg.get("ctc_dropout", 0.0)),
        expose_posterior=bool(adapter_cfg.get("expose_posterior", False)),
        blank_id=blank_id,
    )
