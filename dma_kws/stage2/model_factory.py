"""Single construction path for the QbyT v6 verifier."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from dma_kws.pathing import load_qbyt_class
from dma_kws.stage2.readout import QbyTAlignmentSpec, resolve_qbyt_alignment


def build_qbyt(
    stage2_cfg: Mapping[str, Any],
    *,
    input_dim: int,
    vocab_size: int,
):
    """Construct QbyT from the complete, canonical alignment specification."""

    alignment = resolve_qbyt_alignment(stage2_cfg)
    QbyT = load_qbyt_class()
    model = QbyT(
        encoder_output_size=int(input_dim),
        num_embeds=int(vocab_size),
        embed_dim=int(stage2_cfg.get("qbyt_embed_dim", 128)),
        post_num_layers=int(stage2_cfg.get("qbyt_layers", 2)),
        local_context_kernel=alignment.local_context_kernel,
        min_phone_duration_frames=alignment.min_phone_duration_frames,
        max_phone_duration_frames=alignment.max_phone_duration_frames,
        max_inter_phone_gap_frames=alignment.max_inter_phone_gap_frames,
        max_keyword_span_frames=alignment.max_keyword_span_frames,
        weakest_phone_temperature=alignment.weakest_phone_temperature,
        weakest_phone_weight=alignment.weakest_phone_weight,
    )
    return model


def qbyt_alignment_spec(stage2_cfg: Mapping[str, Any]) -> QbyTAlignmentSpec:
    """Return the exact score semantics used by :func:`build_qbyt`."""

    return resolve_qbyt_alignment(stage2_cfg)


__all__ = ["build_qbyt", "qbyt_alignment_spec"]
