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

        Returns ``(loss, num_skipped)``. CTC cannot align a label sequence longer
        than the input, and at 25 Hz a short phrase clip can genuinely have fewer
        frames than phonemes. Those samples are dropped rather than left to
        ``zero_infinity`` so the count stays visible: a high skip rate means the
        auxiliary loss is quietly training on long clips only.
        """
        batch, frames = log_probs.size(0), log_probs.size(1)
        if encoder_mask is None:
            input_lengths = torch.full(
                (batch,), frames, dtype=torch.long, device=log_probs.device
            )
        else:
            input_lengths = encoder_mask.squeeze(1).sum(dim=1).to(dtype=torch.long)

        target_lengths = target_lengths.to(device=log_probs.device, dtype=torch.long)
        keep = target_lengths <= input_lengths
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
            zero_infinity=True,
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
