"""Resolved Stage II sequence-objective metadata and compatibility checks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from dma_kws.stage2.losses import (
    normalize_seq_loss_normalization,
    validate_seq_loss_weights,
)
from dma_kws.tokenizer import (
    DEFAULT_SEQ_LABEL_MODE,
    SEQ_LABEL_MEMBERSHIP,
    normalize_seq_label_mode,
)


@dataclass(frozen=True)
class SequenceObjective:
    target_mode: str
    progress_weight: float
    completion_weight: float
    normalization: str

    def as_dict(self) -> dict[str, str | float]:
        return {
            "target_mode": self.target_mode,
            "progress_weight": self.progress_weight,
            "completion_weight": self.completion_weight,
            "normalization": self.normalization,
        }

    def describe(self) -> str:
        return (
            f"{self.target_mode}/progress={self.progress_weight:g}/"
            f"completion={self.completion_weight:g}/{self.normalization}"
        )


CURRENT_SEQUENCE_OBJECTIVE = SequenceObjective(
    target_mode=DEFAULT_SEQ_LABEL_MODE,
    progress_weight=0.5,
    completion_weight=0.5,
    normalization="sample",
)

# Checkpoints written before ``stage2.sequence_loss`` existed used exactly this
# target/reduction. Treating a missing section as current would erase provenance
# and make an old optimizer state appear safe to resume.
LEGACY_SEQUENCE_OBJECTIVE = SequenceObjective(
    target_mode=SEQ_LABEL_MEMBERSHIP,
    progress_weight=1.0,
    completion_weight=0.0,
    normalization="token",
)


def resolve_sequence_objective(
    stage2_cfg: Mapping[str, Any],
    *,
    missing_is_legacy: bool = False,
) -> SequenceObjective:
    """Resolve and validate ``stage2.sequence_loss`` from a config mapping."""
    raw = stage2_cfg.get("sequence_loss")
    if raw is None:
        return (
            LEGACY_SEQUENCE_OBJECTIVE
            if missing_is_legacy
            else CURRENT_SEQUENCE_OBJECTIVE
        )
    if not isinstance(raw, Mapping):
        raise ValueError("stage2.sequence_loss must be a mapping")

    objective = SequenceObjective(
        target_mode=normalize_seq_label_mode(
            raw.get("target_mode", CURRENT_SEQUENCE_OBJECTIVE.target_mode)
        ),
        progress_weight=float(
            raw.get("progress_weight", CURRENT_SEQUENCE_OBJECTIVE.progress_weight)
        ),
        completion_weight=float(
            raw.get("completion_weight", CURRENT_SEQUENCE_OBJECTIVE.completion_weight)
        ),
        normalization=normalize_seq_loss_normalization(
            raw.get("normalization", CURRENT_SEQUENCE_OBJECTIVE.normalization)
        ),
    )
    validate_seq_loss_weights(
        objective.progress_weight,
        objective.completion_weight,
    )
    if objective.target_mode == SEQ_LABEL_MEMBERSHIP and objective.completion_weight:
        raise ValueError(
            "sequence_loss.completion_weight requires cumulative ordered-prefix "
            "targets; set target_mode=ordered_contiguous_prefix, or set "
            "completion_weight=0 for a membership ablation"
        )
    return objective


def checkpoint_sequence_objective(checkpoint: Any) -> SequenceObjective:
    """Read saved objective metadata; missing metadata means released legacy."""
    if not isinstance(checkpoint, Mapping):
        return LEGACY_SEQUENCE_OBJECTIVE
    config = checkpoint.get("config")
    if not isinstance(config, Mapping):
        return LEGACY_SEQUENCE_OBJECTIVE
    stage2 = config.get("stage2")
    if not isinstance(stage2, Mapping):
        return LEGACY_SEQUENCE_OBJECTIVE
    return resolve_sequence_objective(stage2, missing_is_legacy=True)


def assert_sequence_objective_matches(
    checkpoint: Any,
    current_stage2_cfg: Mapping[str, Any],
    *,
    source: Any,
) -> None:
    """Reject full-state resume across different same-shape objectives."""
    saved = checkpoint_sequence_objective(checkpoint)
    current = resolve_sequence_objective(current_stage2_cfg)
    if saved != current:
        raise ValueError(
            f"Sequence objective mismatch for {source}: checkpoint used "
            f"[{saved.describe()}], current config uses [{current.describe()}]. "
            "Do not resume optimizer state across objectives; use a weights-only "
            "init/resume checkpoint for an explicit warm start instead."
        )
