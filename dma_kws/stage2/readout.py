"""Canonical QbyT v6 keyword-vs-filler score configuration."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
from numbers import Integral, Real
from typing import Any


QBYT_ALIGNMENT_TOPOLOGY = "keyword_filler_segmental_crf_v1"
DEFAULT_MIN_PHONE_DURATION_FRAMES = 1
DEFAULT_MAX_PHONE_DURATION_FRAMES = 8
DEFAULT_MAX_INTER_PHONE_GAP_FRAMES = 1
DEFAULT_MAX_KEYWORD_SPAN_FRAMES = 30
DEFAULT_LOCAL_CONTEXT_KERNEL = 5
DEFAULT_WEAKEST_PHONE_TEMPERATURE = 0.2
DEFAULT_WEAKEST_PHONE_WEIGHT = 1.0


def normalize_qbyt_alignment_topology(value: Any) -> str:
    """Return the sole v6 topology, rejecting every historical readout mode."""

    if isinstance(value, Mapping):
        value = value.get("topology", value.get("mode", QBYT_ALIGNMENT_TOPOLOGY))
    if value is None:
        value = QBYT_ALIGNMENT_TOPOLOGY
    topology = str(value).strip().lower()
    if topology != QBYT_ALIGNMENT_TOPOLOGY:
        raise ValueError(
            f"Unsupported QbyT alignment topology {value!r}; expected "
            f"{QBYT_ALIGNMENT_TOPOLOGY!r}. Historical GRU/EPS and target-only "
            "bounded-segmental readouts are not loadable by the v6 model."
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


def _normalize_non_negative_float(value: Any, *, field: str) -> float:
    if isinstance(value, bool):
        raise ValueError(
            f"QbyT alignment {field} must be a finite number greater than or equal to 0"
        )
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"QbyT alignment {field} must be a finite number greater than or equal to 0"
        ) from exc
    if not math.isfinite(parsed) or parsed < 0.0:
        raise ValueError(
            f"QbyT alignment {field} must be a finite number greater than or equal to 0"
        )
    return parsed


@dataclass(frozen=True)
class QbyTAlignmentSpec:
    """Complete semantics of the single deployed v6 QbyT score.

    Every field affects either the bounded segmental path set or the emission
    scores consumed by it. The complete mapping is therefore checkpoint
    metadata even though the aligner itself may have no trainable parameters.
    """

    topology: str = QBYT_ALIGNMENT_TOPOLOGY
    min_phone_duration_frames: int = DEFAULT_MIN_PHONE_DURATION_FRAMES
    max_phone_duration_frames: int = DEFAULT_MAX_PHONE_DURATION_FRAMES
    max_inter_phone_gap_frames: int = DEFAULT_MAX_INTER_PHONE_GAP_FRAMES
    max_keyword_span_frames: int = DEFAULT_MAX_KEYWORD_SPAN_FRAMES
    local_context_kernel: int = DEFAULT_LOCAL_CONTEXT_KERNEL
    weakest_phone_temperature: float = DEFAULT_WEAKEST_PHONE_TEMPERATURE
    weakest_phone_weight: float = DEFAULT_WEAKEST_PHONE_WEIGHT

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
            "local_context_kernel",
            _normalize_integer(
                self.local_context_kernel,
                field="local_context_kernel",
                minimum=1,
            ),
        )
        object.__setattr__(
            self,
            "weakest_phone_temperature",
            _normalize_temperature(self.weakest_phone_temperature),
        )
        object.__setattr__(
            self,
            "weakest_phone_weight",
            _normalize_non_negative_float(
                self.weakest_phone_weight,
                field="weakest_phone_weight",
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
            "local_context_kernel": self.local_context_kernel,
            "weakest_phone_temperature": self.weakest_phone_temperature,
            "weakest_phone_weight": self.weakest_phone_weight,
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
                "by QbyT v6; replace it with stage2.qbyt_alignment"
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
    """Reject any non-strict load that did not reproduce the complete v6 QbyT."""

    topology = normalize_qbyt_alignment_topology(expected_topology)
    mismatches = [
        str(key)
        for key in (*missing, *unexpected)
        if str(key).startswith("qbyt.")
    ]
    if mismatches:
        raise SystemExit(
            f"Checkpoint {source} does not carry the complete {topology!r} "
            f"QbyT v6 state: {mismatches}"
        )


__all__ = [
    "DEFAULT_LOCAL_CONTEXT_KERNEL",
    "DEFAULT_MAX_INTER_PHONE_GAP_FRAMES",
    "DEFAULT_MAX_KEYWORD_SPAN_FRAMES",
    "DEFAULT_MAX_PHONE_DURATION_FRAMES",
    "DEFAULT_MIN_PHONE_DURATION_FRAMES",
    "DEFAULT_WEAKEST_PHONE_TEMPERATURE",
    "DEFAULT_WEAKEST_PHONE_WEIGHT",
    "QBYT_ALIGNMENT_TOPOLOGY",
    "QbyTAlignmentSpec",
    "assert_qbyt_alignment_state_loaded",
    "normalize_qbyt_alignment_topology",
    "resolve_qbyt_alignment",
]
