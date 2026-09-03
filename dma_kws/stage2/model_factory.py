"""Single construction path for every supported QbyT readout family."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from dma_kws.pathing import load_qbyt_class
from dma_kws.stage2.readout import (
    QbyTAlignmentSpec,
    QbyTScoreSpec,
    resolve_qbyt_alignment,
    resolve_qbyt_score_spec,
)


def build_qbyt(
    stage2_cfg: Mapping[str, Any],
    *,
    input_dim: int,
    vocab_size: int,
):
    """Construct the QbyT family described by ``stage2_cfg``."""

    score = resolve_qbyt_score_spec(stage2_cfg)
    embed_dim = int(stage2_cfg.get("qbyt_embed_dim", 128))
    layers = int(stage2_cfg.get("qbyt_layers", 2))
    if score.family == "pooling":
        from qbyt.pooling import QbyT

        return QbyT(
            encoder_output_size=int(input_dim),
            num_embeds=int(vocab_size),
            embed_dim=embed_dim,
            post_num_layers=layers,
            readout_mode=score.value.mode,
            readout_temperature=score.value.temperature,
        )
    if score.family == "bounded":
        from qbyt.bounded import QbyT

        alignment = score.value
        return QbyT(
            encoder_output_size=int(input_dim),
            num_embeds=int(vocab_size),
            embed_dim=embed_dim,
            post_num_layers=layers,
            local_context_kernel=alignment.local_context_kernel,
            min_phone_duration_frames=alignment.min_phone_duration_frames,
            max_phone_duration_frames=alignment.max_phone_duration_frames,
            max_inter_phone_gap_frames=alignment.max_inter_phone_gap_frames,
            max_keyword_span_frames=alignment.max_keyword_span_frames,
            alignment_temperature=alignment.temperature,
        )

    QbyT = load_qbyt_class()
    alignment = score.value
    return QbyT(
        encoder_output_size=int(input_dim),
        num_embeds=int(vocab_size),
        embed_dim=embed_dim,
        post_num_layers=layers,
        local_context_kernel=alignment.local_context_kernel,
        min_phone_duration_frames=alignment.min_phone_duration_frames,
        max_phone_duration_frames=alignment.max_phone_duration_frames,
        max_inter_phone_gap_frames=alignment.max_inter_phone_gap_frames,
        max_keyword_span_frames=alignment.max_keyword_span_frames,
        weakest_phone_temperature=alignment.weakest_phone_temperature,
        weakest_phone_weight=alignment.weakest_phone_weight,
        emission=score.emission or "one_vs_rest",
    )


def qbyt_alignment_spec(stage2_cfg: Mapping[str, Any]) -> QbyTAlignmentSpec:
    """Return keyword-filler alignment used by :func:`build_qbyt`."""

    return resolve_qbyt_alignment(stage2_cfg)


def qbyt_score_spec(stage2_cfg: Mapping[str, Any]) -> QbyTScoreSpec:
    """Return the complete score semantics used by :func:`build_qbyt`."""

    return resolve_qbyt_score_spec(stage2_cfg)


__all__ = ["build_qbyt", "qbyt_alignment_spec", "qbyt_score_spec"]
