"""Canonical QbyT v5 bounded-segmental score configuration."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
from numbers import Integral, Real
from typing import Any


QBYT_ALIGNMENT_TOPOLOGY = "bounded_segmental_v1"
DEFAULT_MIN_PHONE_DURATION_FRAMES = 1
DEFAULT_MAX_PHONE_DURATION_FRAMES = 8
DEFAULT_MAX_INTER_PHONE_GAP_FRAMES = 2
DEFAULT_MAX_KEYWORD_SPAN_FRAMES = 30
DEFAULT_QBYT_ALIGNMENT_TEMPERATURE = 0.2
DEFAULT_LOCAL_CONTEXT_KERNEL = 5

def normalize_qbyt_alignment_topology(value: Any) -> str:
    """Return the sole v5 topology, rejecting every historical readout mode."""

    if isinstance(value, Mapping):
        value = value.get("topology", value.get("mode", QBYT_ALIGNMENT_TOPOLOGY))
    if value is None:
        value = QBYT_ALIGNMENT_TOPOLOGY
    topology = str(value).strip().lower()
    if topology != QBYT_ALIGNMENT_TOPOLOGY:
        raise ValueError(
            f"Unsupported QbyT alignment topology {value!r}; expected "
            f"{QBYT_ALIGNMENT_TOPOLOGY!r}. Historical GRU/EPS readouts are not "
            "loadable by the v5 model."
        )
    return topology


def _normalize_integer(value: Any, *, field: str, minimum: int) -> int:
    if isinstance(value, bool):
        raise ValueError(f"QbyT alignment {field} must be an integer >= {minimum}")
    if isinstance(value, Integral):
        parsed = int(value)
    elif isinstance(value, Real):
        numeric = float(value)
        if not math.isfinite(numeric) or not numeric.is_integer():
            raise ValueError(
                f"QbyT alignment {field} must be an integer >= {minimum}, got {value!r}"
            )
        parsed = int(numeric)
    else:
        text = str(value).strip()
        try:
            parsed = int(text)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"QbyT alignment {field} must be an integer >= {minimum}, got {value!r}"
            ) from exc
        if str(parsed) != text.lstrip("+"):
            raise ValueError(
                f"QbyT alignment {field} must be an integer >= {minimum}, got {value!r}"
            )
    if parsed < minimum:
        raise ValueError(
            f"QbyT alignment {field} must be an integer >= {minimum}, got {value!r}"
        )
    return parsed


def _normalize_temperature(value: Any) -> float:
    if isinstance(value, bool):
        raise ValueError(
            "QbyT alignment temperature must be a finite number greater than 0, "
            f"got {value!r}"
        )
    try:
        temperature = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "QbyT alignment temperature must be a finite number greater than 0, "
            f"got {value!r}"
        ) from exc
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError(
            "QbyT alignment temperature must be a finite number greater than 0, "
            f"got {value!r}"
        )
    return temperature


@dataclass(frozen=True)
class QbyTAlignmentSpec:
    """Complete semantics of the single deployed v5 QbyT score.

    Every field affects either the bounded segmental path set or the emission
    scores consumed by it. The complete mapping is therefore checkpoint
    metadata even though the aligner itself may have no trainable parameters.
    """

    topology: str = QBYT_ALIGNMENT_TOPOLOGY
    min_phone_duration_frames: int = DEFAULT_MIN_PHONE_DURATION_FRAMES
    max_phone_duration_frames: int = DEFAULT_MAX_PHONE_DURATION_FRAMES
    max_inter_phone_gap_frames: int = DEFAULT_MAX_INTER_PHONE_GAP_FRAMES
    max_keyword_span_frames: int = DEFAULT_MAX_KEYWORD_SPAN_FRAMES
    temperature: float = DEFAULT_QBYT_ALIGNMENT_TEMPERATURE
    local_context_kernel: int = DEFAULT_LOCAL_CONTEXT_KERNEL

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "topology",
            normalize_qbyt_alignment_topology(self.topology),
        )
        object.__setattr__(
            self,
            "min_phone_duration_frames",
            _normalize_integer(
                self.min_phone_duration_frames,
                field="min_phone_duration_frames",
                minimum=1,
            ),
        )
        object.__setattr__(
            self,
            "max_phone_duration_frames",
            _normalize_integer(
                self.max_phone_duration_frames,
                field="max_phone_duration_frames",
                minimum=1,
            ),
        )
        object.__setattr__(
            self,
            "max_inter_phone_gap_frames",
            _normalize_integer(
                self.max_inter_phone_gap_frames,
                field="max_inter_phone_gap_frames",
                minimum=0,
            ),
        )
        object.__setattr__(
            self,
            "max_keyword_span_frames",
            _normalize_integer(
                self.max_keyword_span_frames,
                field="max_keyword_span_frames",
                minimum=1,
            ),
        )
        object.__setattr__(
            self,
            "temperature",
            _normalize_temperature(self.temperature),
        )
        object.__setattr__(
            self,
            "local_context_kernel",
            _normalize_integer(
                self.local_context_kernel,
                field="local_context_kernel",
                minimum=1,
            ),
        )
        if self.max_phone_duration_frames < self.min_phone_duration_frames:
            raise ValueError(
                "QbyT alignment max_phone_duration_frames must be >= "
                "min_phone_duration_frames"
            )
        if self.max_keyword_span_frames < self.max_phone_duration_frames:
            raise ValueError(
                "QbyT alignment max_keyword_span_frames must be >= "
                "max_phone_duration_frames"
            )
        if self.local_context_kernel % 2 == 0:
            raise ValueError("QbyT alignment local_context_kernel must be odd")

    def as_dict(self) -> dict[str, str | int | float]:
        """Return the stable mapping embedded in checkpoints and provenance."""

        return {
            "topology": self.topology,
            "min_phone_duration_frames": self.min_phone_duration_frames,
            "max_phone_duration_frames": self.max_phone_duration_frames,
            "max_inter_phone_gap_frames": self.max_inter_phone_gap_frames,
            "max_keyword_span_frames": self.max_keyword_span_frames,
            "temperature": self.temperature,
            "local_context_kernel": self.local_context_kernel,
        }


def resolve_qbyt_alignment(
    stage2_config: Mapping[str, Any],
) -> QbyTAlignmentSpec:
    """Resolve and validate ``stage2.qbyt_alignment``."""

    if not isinstance(stage2_config, Mapping):
        raise ValueError("stage2 config must be a mapping")
    raw = stage2_config.get("qbyt_alignment")
    if raw is None:
        if stage2_config.get("qbyt_readout") is not None:
            raise ValueError(
                "stage2.qbyt_readout is a legacy GRU/EPS switch and is unsupported "
                "by QbyT v5; replace it with stage2.qbyt_alignment"
            )
        raw = {}
    if not isinstance(raw, Mapping):
        raise ValueError("stage2.qbyt_alignment must be a mapping")

    supported = set(QbyTAlignmentSpec.__dataclass_fields__)
    unknown = sorted(set(raw) - supported)
    if unknown:
        raise ValueError(f"stage2.qbyt_alignment has unknown fields: {unknown}")
    return QbyTAlignmentSpec(**dict(raw))


def assert_qbyt_alignment_state_loaded(
    missing: list[str] | tuple[str, ...],
    unexpected: list[str] | tuple[str, ...],
    *,
    source: Any,
    expected_topology: str = QBYT_ALIGNMENT_TOPOLOGY,
) -> None:
    """Reject any non-strict load that did not reproduce the complete v5 QbyT."""

    topology = normalize_qbyt_alignment_topology(expected_topology)
    mismatches = [
        str(key)
        for key in (*missing, *unexpected)
        if str(key).startswith("qbyt.")
    ]
    if mismatches:
        raise SystemExit(
            f"Checkpoint {source} does not carry the complete {topology!r} "
            f"QbyT v5 state: {mismatches}"
        )


__all__ = [
    "DEFAULT_LOCAL_CONTEXT_KERNEL",
    "DEFAULT_MAX_INTER_PHONE_GAP_FRAMES",
    "DEFAULT_MAX_KEYWORD_SPAN_FRAMES",
    "DEFAULT_MAX_PHONE_DURATION_FRAMES",
    "DEFAULT_MIN_PHONE_DURATION_FRAMES",
    "DEFAULT_QBYT_ALIGNMENT_TEMPERATURE",
    "QBYT_ALIGNMENT_TOPOLOGY",
    "QbyTAlignmentSpec",
    "assert_qbyt_alignment_state_loaded",
    "normalize_qbyt_alignment_topology",
    "resolve_qbyt_alignment",
]
