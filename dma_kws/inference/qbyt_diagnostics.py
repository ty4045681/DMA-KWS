"""Emission- and alignment-level diagnostics for a deployed QbyT v6 score.

The deployed score is a bounded segmental graph LLR over one tensor of frame
posteriors. ``results.jsonl`` records only the scalar at the end of that chain,
which cannot distinguish the two ways it can be small:

* the emission model has no evidence for the keyword's phones, or
* the evidence exists but no legal path can collect it, because the keyword's
  own span, a phone duration, or an inter-phone gap exceeds its bound.

Every quantity here is derived from the *same* ``frame_class_log_probs`` tensor
the deployed score used, so the ablations below re-score the identical acoustic
evidence under a different readout. Nothing retrains and nothing mutates the
model, so an ablation logit is directly comparable with ``qbyt_raw_logit``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import torch
import torch.nn.functional as F

__all__ = [
    "AblationSpec",
    "DEFAULT_ABLATIONS",
    "clip_emission_diagnostics",
    "summarize_emission_diagnostics",
]


@dataclass(frozen=True)
class AblationSpec:
    """One alternative readout scored on unchanged frame posteriors.

    ``filler`` selects how the per-frame denominator is formed:

    ``one_vs_rest``
        The deployed definition: ``log p(phone) - log(1 - p(phone))``. The
        competing phone always stays in the denominator.
    ``query_relative``
        The legacy (readout-version-6) definition, kept for comparison: blank,
        non-speech and every inventory phone the query does not use. Phones
        that *are* in the query are removed from the denominator, so a
        substitution onto another query phone is charged against an
        artificially small filler.

    ``max_inter_phone_gap_frames`` and ``max_keyword_span_frames`` of ``None``
    keep the deployed bound. ``max_phone_duration_frames`` is deliberately not
    ablatable: it sizes the learned per-phone duration table, so changing it
    would require inventing duration potentials the checkpoint never trained.
    """

    name: str
    filler: str = "one_vs_rest"
    max_inter_phone_gap_frames: int | None = None
    max_keyword_span_frames: int | None = None

    def __post_init__(self) -> None:
        if self.filler not in {"query_relative", "one_vs_rest"}:
            raise ValueError(
                "AblationSpec.filler must be 'query_relative' or 'one_vs_rest', "
                f"got {self.filler!r}"
            )


#: Deployed readout first, then each readout-version-6 rule reinstated on its
#: own, then both together, so a score shift can be attributed to one of them.
DEFAULT_ABLATIONS: tuple[AblationSpec, ...] = (
    AblationSpec("deployed"),
    AblationSpec("query_relative_filler", filler="query_relative"),
    AblationSpec("gap_1", max_inter_phone_gap_frames=1),
    AblationSpec(
        "legacy_v6_readout",
        filler="query_relative",
        max_inter_phone_gap_frames=1,
    ),
)


def _target_llr(
    class_log_probs: torch.Tensor,
    anchors: torch.Tensor,
    phone_mask: torch.Tensor,
    frame_mask: torch.Tensor,
    phone_class_count: int,
    *,
    filler: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(target_llr [B,U,T], filler_log_probs [B,T])`` for one filler rule."""

    batch_size, frame_width, _ = class_log_probs.shape
    safe_indices = (anchors - 1).clamp(min=0, max=phone_class_count - 1)
    target_log_probs = class_log_probs[:, :, :phone_class_count].transpose(1, 2)
    target_log_probs = target_log_probs.gather(
        1, safe_indices.unsqueeze(-1).expand(-1, -1, frame_width)
    )

    if filler == "query_relative":
        counts = torch.zeros(
            batch_size, phone_class_count, device=anchors.device, dtype=torch.long
        )
        counts.scatter_add_(1, safe_indices, phone_mask.to(dtype=torch.long))
        class_mask = torch.cat(
            (
                counts.eq(0),
                torch.ones(batch_size, 2, device=anchors.device, dtype=torch.bool),
            ),
            dim=1,
        )
        filler_log_probs = torch.logsumexp(
            class_log_probs.masked_fill(~class_mask.unsqueeze(1), -torch.inf), dim=-1
        )
        target_llr = target_log_probs - filler_log_probs.unsqueeze(1)
    else:
        # log(1 - p) computed from log p without leaving log space.
        rest = torch.log1p(-target_log_probs.exp().clamp(max=1.0 - 1e-6))
        target_llr = target_log_probs - rest
        # Only the graph LLR is reported, and the filler baseline cancels there,
        # so a per-phone denominator needs no separate [B,T] baseline.
        filler_log_probs = torch.zeros_like(class_log_probs[:, :, 0])

    valid = phone_mask.unsqueeze(2) & frame_mask.unsqueeze(1)
    target_llr = torch.where(valid, target_llr, torch.zeros_like(target_llr))
    filler_log_probs = torch.where(
        frame_mask, filler_log_probs, torch.zeros_like(filler_log_probs)
    )
    return target_llr, filler_log_probs


def _sentinel_to_none(value: float, invalid_score: float) -> float | None:
    """Map the aligner's finite "no legal path" sentinel to ``None``.

    The sentinel is a large negative constant, not a score: leaving it in a
    numeric field would drag any aggregate mean towards it and make a handful of
    structurally unscoreable clips look like a uniformly terrible model.
    """
    return None if value == invalid_score else value


def _duration_potentials(qbyt, anchors, phone_mask) -> torch.Tensor:
    count = qbyt.duration_logits.embedding_dim
    potentials = F.log_softmax(
        qbyt.duration_logits(anchors).float(), dim=-1
    ) + math.log(float(count))
    return torch.where(
        phone_mask.unsqueeze(-1), potentials, torch.zeros_like(potentials)
    )


def _aligner_for(qbyt, spec: AblationSpec):
    from qbyt.monotonic_alignment import BoundedSegmentalAligner

    base = qbyt.aligner
    if (
        spec.max_inter_phone_gap_frames is None
        and spec.max_keyword_span_frames is None
    ):
        return base
    return BoundedSegmentalAligner(
        min_phone_duration_frames=base.min_phone_duration_frames,
        max_phone_duration_frames=base.max_phone_duration_frames,
        max_inter_phone_gap_frames=(
            base.max_inter_phone_gap_frames
            if spec.max_inter_phone_gap_frames is None
            else int(spec.max_inter_phone_gap_frames)
        ),
        max_keyword_span_frames=(
            base.max_keyword_span_frames
            if spec.max_keyword_span_frames is None
            else int(spec.max_keyword_span_frames)
        ),
        temperature=base.temperature,
        invalid_score=base.invalid_score,
    )


def _utterance_logit(qbyt, aligner, target_llr, filler_log_probs, duration, lengths):
    text_lengths, speech_lengths = lengths
    alignment = aligner(
        target_llr,
        filler_log_probs,
        text_lengths,
        speech_lengths,
        duration_log_probs=duration,
    )
    phone_mask = qbyt._length_mask(text_lengths, target_llr.size(1))
    legal = phone_mask & alignment.has_legal_path
    veto = torch.where(
        legal,
        qbyt.weakest_phone_weight * F.logsigmoid(alignment.weakest_phone_evidence),
        torch.zeros_like(alignment.weakest_phone_evidence),
    )
    seq_logits = torch.where(
        legal,
        alignment.prefix_llr + veto,
        torch.full_like(alignment.prefix_llr, aligner.invalid_score),
    )
    last = (text_lengths - 1).clamp_min(0)
    return seq_logits.gather(1, last.unsqueeze(1)).squeeze(1), alignment, last


@torch.no_grad()
def clip_emission_diagnostics(
    qbyt,
    speech: torch.Tensor,
    anchors: torch.Tensor,
    speech_lengths: torch.Tensor,
    anchor_lengths: torch.Tensor,
    *,
    ablations: Sequence[AblationSpec] = DEFAULT_ABLATIONS,
) -> list[dict[str, Any]]:
    """Per-clip emission statistics and readout ablations for one batch."""

    if anchors.size(1) == 0 or int(anchor_lengths.min()) <= 0:
        # QbyT.forward returns the invalid sentinel for an empty query instead of
        # indexing the graph. Rather than reproduce that branch and risk the two
        # drifting, refuse the input: no caller has a use for it.
        raise ValueError("clip emission diagnostics require a non-empty query")

    frame_width = speech.size(1)
    phone_width = anchors.size(1)
    frame_mask = qbyt._length_mask(speech_lengths, frame_width)
    phone_mask = qbyt._length_mask(anchor_lengths, phone_width)
    phone_class_count = qbyt.text_projection.num_embeddings - 1

    class_log_probs = qbyt.frame_class_log_probs(speech, frame_mask).float()
    duration = _duration_potentials(qbyt, anchors, phone_mask)

    # Probability mass the legacy query-relative filler removes from its
    # denominator, per frame. ``filler_masked = 1 - this`` while the deployed
    # ``filler_one_vs_rest = 1 - p_i``, so this is precisely the size of the
    # discount the query-relative ablation reinstates.
    #
    # Counted over the *set* of query phones, exactly like that ablation's class
    # mask: a phone the query uses twice is still removed from the denominator
    # once.
    safe_indices = (anchors - 1).clamp(min=0, max=phone_class_count - 1)
    counts = torch.zeros(
        anchors.size(0), phone_class_count, device=anchors.device, dtype=torch.long
    )
    counts.scatter_add_(1, safe_indices, phone_mask.to(dtype=torch.long))
    in_query = counts.gt(0).unsqueeze(1)
    masked_mass = (
        class_log_probs[:, :, :phone_class_count].exp() * in_query
    ).sum(dim=-1)

    per_ablation: dict[str, Any] = {}
    for spec in ablations:
        target_llr, filler_log_probs = _target_llr(
            class_log_probs,
            anchors,
            phone_mask,
            frame_mask,
            phone_class_count,
            filler=spec.filler,
        )
        aligner = _aligner_for(qbyt, spec)
        logit, alignment, last = _utterance_logit(
            qbyt,
            aligner,
            target_llr,
            filler_log_probs,
            duration,
            (anchor_lengths, speech_lengths),
        )
        per_ablation[spec.name] = {
            "logit": logit,
            "alignment": alignment,
            "last": last,
            "target_llr": target_llr,
        }

    records: list[dict[str, Any]] = []
    for index in range(speech.size(0)):
        frames = int(speech_lengths[index])
        phones = int(anchor_lengths[index])
        record: dict[str, Any] = {
            "encoder_frames": frames,
            "phone_count": phones,
            "masked_query_mass_mean": float(
                masked_mass[index, :frames].mean() if frames else 0.0
            ),
            "masked_query_mass_max": float(
                masked_mass[index, :frames].max() if frames else 0.0
            ),
        }
        for spec in ablations:
            bundle = per_ablation[spec.name]
            alignment = bundle["alignment"]
            last = int(bundle["last"][index])
            llr = bundle["target_llr"][index, :phones, :frames]
            best_frames = llr.argmax(dim=1) if phones and frames else llr.new_zeros(0)
            per_phone_best = llr.max(dim=1).values if phones and frames else llr.new_zeros(0)
            top = min(3, frames) if frames else 0
            invalid = _aligner_for(qbyt, spec).invalid_score
            # Absolute log partitions are not reported: they carry the pure-filler
            # baseline, which a per-phone (one-vs-rest) denominator does not
            # define, so they would not be comparable across ablations. This
            # difference is, and it is the informative part -- how much better the
            # one-deletion route is than explaining the whole clip as filler.
            one_edit = float(alignment.one_edit_log_partition[index, last])
            entry = {
                "logit": float(bundle["logit"][index]),
                "has_legal_path": bool(alignment.has_legal_path[index, last]),
                "prefix_llr": _sentinel_to_none(
                    float(alignment.prefix_llr[index, last]), invalid
                ),
                "one_edit_vs_filler": (
                    None
                    if one_edit == invalid
                    else one_edit - float(alignment.filler_log_partition[index])
                ),
                "weakest_phone_evidence": _sentinel_to_none(
                    float(alignment.weakest_phone_evidence[index, last]), invalid
                ),
                "phone_evidence": [
                    _sentinel_to_none(float(value), invalid)
                    for value in alignment.phone_evidence[index, :phones]
                ],
                # Per-frame evidence for each query phone at its single best
                # frame, and averaged over its best few. Independent of the
                # graph, so it separates "no evidence" from "unreachable".
                "phone_best_frame_llr": [float(value) for value in per_phone_best],
                "phone_best_frame_index": [int(value) for value in best_frames],
                "phone_top_frames_llr_mean": (
                    float(llr.topk(top, dim=1).values.mean()) if top and phones else 0.0
                ),
            }
            if phones and frames:
                order = best_frames.tolist()
                entry["best_frame_span"] = int(max(order) - min(order))
                entry["best_frames_monotonic"] = all(
                    left <= right for left, right in zip(order, order[1:])
                )
            record[spec.name] = entry
        records.append(record)
    return records


def _quantiles(values: Sequence[float | None]) -> dict[str, float] | None:
    """Quantiles over the clips where the quantity is defined.

    ``None`` marks a clip whose graph had no legal path, so the quantity does not
    exist there. Dropping those keeps ``count`` an honest denominator instead of
    mixing a sentinel into the distribution.
    """
    present = [float(value) for value in values if value is not None]
    if not present:
        return None
    tensor = torch.tensor(present, dtype=torch.float64)
    return {
        "count": len(present),
        "p05": float(tensor.quantile(0.05)),
        "p50": float(tensor.quantile(0.50)),
        "p95": float(tensor.quantile(0.95)),
        "mean": float(tensor.mean()),
    }


def summarize_emission_diagnostics(
    records: Sequence[dict[str, Any]],
    *,
    ablations: Sequence[AblationSpec] = DEFAULT_ABLATIONS,
    group_key: str = "group",
) -> dict[str, Any]:
    """Aggregate per-clip diagnostics by ``group_key`` for a summary report."""

    groups: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        groups.setdefault(str(record.get(group_key, "all")), []).append(record)

    summary: dict[str, Any] = {}
    for name, rows in sorted(groups.items()):
        entry: dict[str, Any] = {
            "num_clips": len(rows),
            "encoder_frames": _quantiles([float(r["encoder_frames"]) for r in rows]),
            "masked_query_mass_mean": _quantiles(
                [float(r["masked_query_mass_mean"]) for r in rows]
            ),
        }
        for spec in ablations:
            rows_with = [r for r in rows if spec.name in r]
            if not rows_with:
                continue
            # ``get`` throughout: a field is absent for a clip where the quantity
            # is undefined, and ``_quantiles`` already treats that as "not
            # measured here" rather than as a value.
            aggregate: dict[str, Any] = {
                name: _quantiles([r[spec.name].get(name) for r in rows_with])
                for name in (
                    "logit",
                    "prefix_llr",
                    "one_edit_vs_filler",
                    "weakest_phone_evidence",
                    "phone_top_frames_llr_mean",
                    "best_frame_span",
                )
            }
            aggregate["illegal_path_rate"] = sum(
                not bool(r[spec.name].get("has_legal_path")) for r in rows_with
            ) / len(rows_with)
            aggregate["monotonic_best_frame_rate"] = sum(
                bool(r[spec.name].get("best_frames_monotonic", False))
                for r in rows_with
            ) / len(rows_with)
            # Paired per-clip delta, not a difference of two medians: the
            # ablations score the same clips, so pairing is both available and
            # strictly more informative about whether the readout change helps.
            if spec.name != "deployed":
                aggregate["logit_delta_vs_deployed"] = _quantiles(
                    [
                        r[spec.name]["logit"] - r["deployed"]["logit"]
                        for r in rows_with
                        if "deployed" in r
                        and r[spec.name].get("logit") is not None
                        and r["deployed"].get("logit") is not None
                    ]
                )
            entry[spec.name] = aggregate
        summary[name] = entry
    return summary
