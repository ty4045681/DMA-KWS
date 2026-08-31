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
    normalize_seq_label_mode,
)


@dataclass(frozen=True)
class SequenceObjective:
    target_mode: str
    progress_weight: float
    normalization: str

    def as_dict(self) -> dict[str, str | float]:
        return {
            "target_mode": self.target_mode,
            "progress_weight": self.progress_weight,
            "normalization": self.normalization,
        }

    def describe(self) -> str:
        return (
            f"{self.target_mode}/progress={self.progress_weight:g}/"
            f"{self.normalization}"
        )


CURRENT_SEQUENCE_OBJECTIVE = SequenceObjective(
    target_mode=DEFAULT_SEQ_LABEL_MODE,
    progress_weight=0.3,
    normalization="sample",
)

def resolve_sequence_objective(
    stage2_cfg: Mapping[str, Any],
) -> SequenceObjective:
    """Resolve and validate ``stage2.sequence_loss`` from a config mapping."""
    raw = stage2_cfg.get("sequence_loss")
    if raw is None:
        return CURRENT_SEQUENCE_OBJECTIVE
    if not isinstance(raw, Mapping):
        raise ValueError("stage2.sequence_loss must be a mapping")
    supported = {"target_mode", "progress_weight", "normalization"}
    unknown = sorted(set(raw) - supported)
    if unknown:
        raise ValueError(f"stage2.sequence_loss has unknown fields: {unknown}")

    objective = SequenceObjective(
        target_mode=normalize_seq_label_mode(
            raw.get("target_mode", CURRENT_SEQUENCE_OBJECTIVE.target_mode)
        ),
        progress_weight=float(
            raw.get("progress_weight", CURRENT_SEQUENCE_OBJECTIVE.progress_weight)
        ),
        normalization=normalize_seq_loss_normalization(
            raw.get("normalization", CURRENT_SEQUENCE_OBJECTIVE.normalization)
        ),
    )
    validate_seq_loss_weights(objective.progress_weight)
    return objective


def checkpoint_sequence_objective(checkpoint: Any) -> SequenceObjective:
    """Read the objective metadata required in every QbyT v6 checkpoint."""
    if not isinstance(checkpoint, Mapping):
        raise ValueError("QbyT v6 checkpoint must be a mapping")
    config = checkpoint.get("config")
    if not isinstance(config, Mapping):
        raise ValueError("QbyT v6 checkpoint must record config.stage2.sequence_loss")
    stage2 = config.get("stage2")
    if not isinstance(stage2, Mapping):
        raise ValueError("QbyT v6 checkpoint must record config.stage2.sequence_loss")
    if "sequence_loss" not in stage2:
        raise ValueError("QbyT v6 checkpoint must record config.stage2.sequence_loss")
    return resolve_sequence_objective(stage2)


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
