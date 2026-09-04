"""QbyT score configuration across readout versions 2-7.

Version 6/7 keyword-filler specs live in this module. Pooling (v2-v4) and
bounded-segmental (v5) specs are restored from git in sibling modules; the
functions below dispatch to them.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
from numbers import Integral, Real
from typing import Any


QBYT_ALIGNMENT_TOPOLOGY = "keyword_filler_segmental_crf_v1"
DEFAULT_MIN_PHONE_DURATION_FRAMES = 1
DEFAULT_MAX_PHONE_DURATION_FRAMES = 8
DEFAULT_MAX_INTER_PHONE_GAP_FRAMES = 3
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
    """Resolve v6/v7 ``stage2.qbyt_alignment``.

    Historical pooling and bounded configs must go through
    :func:`resolve_qbyt_score_spec`.
    """

    score = resolve_qbyt_score_spec(stage2_config)
    if score.version in (2, 3, 4):
        raise ValueError(
            "stage2.qbyt_readout is a legacy GRU/EPS switch and is unsupported "
            "by QbyT v6; replace it with stage2.qbyt_alignment"
        )
    if score.version == 5:
        raise ValueError(
            "Unsupported QbyT alignment topology 'bounded_segmental_v1'; expected "
            f"{QBYT_ALIGNMENT_TOPOLOGY!r}. Historical GRU/EPS and target-only "
            "bounded-segmental readouts are not loadable by the v6 model."
        )
    if not isinstance(score.value, QbyTAlignmentSpec):
        raise TypeError("keyword-filler spec must be QbyTAlignmentSpec")
    return score.value


_KEYWORD_FILLER_ONLY_ALIGNMENT_FIELDS = frozenset(
    {"weakest_phone_temperature", "weakest_phone_weight"}
)
_BOUNDED_ONLY_ALIGNMENT_FIELDS = frozenset({"temperature"})
SUPPORTED_QBYT_READOUT_VERSIONS = frozenset({2, 3, 4, 5, 6, 7})
CURRENT_QBYT_READOUT_VERSION = 7


@dataclass(frozen=True)
class QbyTScoreSpec:
    """Versioned score semantics carried by a checkpoint or Hydra run."""

    version: int
    value: Any

    @property
    def family(self) -> str:
        if self.version in (2, 3, 4):
            return "pooling"
        if self.version == 5:
            return "bounded"
        return "keyword_filler"

    @property
    def emission(self) -> str | None:
        if self.version == 6:
            return "query_relative"
        if self.version == 7:
            return "one_vs_rest"
        return None

    def as_dict(self) -> dict[str, Any]:
        payload = {"version": self.version}
        if hasattr(self.value, "as_dict"):
            payload.update(self.value.as_dict())
        return payload


def _plain_mapping(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, Mapping):
        return {
            str(key): item
            for key, item in value.items()
            if item is not None
        }
    raise ValueError("expected a mapping")


def _alignment_topology(stage2_config: Mapping[str, Any]) -> str | None:
    raw = stage2_config.get("qbyt_alignment")
    if not isinstance(raw, Mapping):
        return None
    topology = raw.get("topology", raw.get("mode"))
    if topology is None:
        return None
    return str(topology).strip().lower()


def default_clip_padding_ms(stage2_config: Mapping[str, Any] | None) -> int:
    """Clip-eval zero padding. Version 6 defaulted to 160 ms; others are unpadded."""

    if not isinstance(stage2_config, Mapping):
        return 0
    try:
        version = resolve_qbyt_readout_version(stage2_config)
    except ValueError:
        return 0
    return 160 if version == 6 else 0


def resolve_qbyt_readout_version(stage2_config: Mapping[str, Any]) -> int:
    """Return the readout version declared or inferred from a Stage-II mapping."""

    if not isinstance(stage2_config, Mapping):
        raise ValueError("stage2 config must be a mapping")
    raw = stage2_config.get("qbyt_readout_version")
    if raw is not None:
        if isinstance(raw, bool) or not isinstance(raw, (int, str)):
            raise ValueError(
                f"stage2.qbyt_readout_version must be an integer, got {raw!r}"
            )
        try:
            version = int(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"stage2.qbyt_readout_version must be an integer, got {raw!r}"
            ) from exc
        if str(raw).strip().lstrip("+") != str(version) and not isinstance(raw, int):
            raise ValueError(
                f"stage2.qbyt_readout_version must be an integer, got {raw!r}"
            )
        if version not in SUPPORTED_QBYT_READOUT_VERSIONS:
            raise ValueError(
                f"Unsupported QbyT readout version {version!r}; expected one of "
                f"{sorted(SUPPORTED_QBYT_READOUT_VERSIONS)}"
            )
        return version

    topology = _alignment_topology(stage2_config)
    has_readout = stage2_config.get("qbyt_readout") is not None
    if topology == "bounded_segmental_v1":
        return 5
    if has_readout and topology not in {
        QBYT_ALIGNMENT_TOPOLOGY,
        "bounded_segmental_v1",
    }:
        return 4
    if has_readout and topology == QBYT_ALIGNMENT_TOPOLOGY:
        raise ValueError(
            "stage2.qbyt_readout and keyword-filler qbyt_alignment both present; "
            "set stage2.qbyt_readout_version to choose a family"
        )
    if has_readout:
        return 4
    return CURRENT_QBYT_READOUT_VERSION


def _alignment_fields_for_version(
    raw: Mapping[str, Any] | None, version: int
) -> dict[str, Any]:
    mapping = dict(raw or {})
    if version == 5:
        drop = _KEYWORD_FILLER_ONLY_ALIGNMENT_FIELDS
        mapping = {key: value for key, value in mapping.items() if key not in drop}
    elif version in (6, 7):
        drop = _BOUNDED_ONLY_ALIGNMENT_FIELDS
        mapping = {key: value for key, value in mapping.items() if key not in drop}
    return mapping


def resolve_qbyt_score_spec(stage2_config: Mapping[str, Any]) -> QbyTScoreSpec:
    """Resolve the complete QbyT score for any supported readout version."""

    if not isinstance(stage2_config, Mapping):
        raise ValueError("stage2 config must be a mapping")
    version = resolve_qbyt_readout_version(stage2_config)
    readout = stage2_config.get("qbyt_readout")
    alignment = stage2_config.get("qbyt_alignment")

    if version in (2, 3, 4):
        from dma_kws.stage2.readout_pooling import (
            pooling_extension_offenders,
            resolve_qbyt_readout,
        )

        pooling = resolve_qbyt_readout({"qbyt_readout": readout})
        if version == 2 and pooling.mode != "gru_last":
            raise ValueError(
                f"QbyT readout version 2 cannot carry mode {pooling.mode!r}"
            )
        if version == 3 and pooling.mode == "eps_softmin":
            raise ValueError(
                f"QbyT readout version 3 cannot carry mode {pooling.mode!r}"
            )
        if version in (2, 3):
            extras = pooling_extension_offenders(pooling)
            if extras:
                raise ValueError(
                    f"QbyT readout version {version} cannot carry "
                    + ", ".join(extras)
                )
        return QbyTScoreSpec(version=version, value=pooling)

    if version == 5:
        from dma_kws.stage2.readout_bounded import resolve_qbyt_alignment as resolve_bounded

        if readout is not None:
            raise ValueError(
                "stage2.qbyt_readout is a legacy GRU/EPS switch and is unsupported "
                "by QbyT v5; replace it with stage2.qbyt_alignment"
            )
        filtered = _alignment_fields_for_version(_plain_mapping(alignment), 5)
        return QbyTScoreSpec(
            version=5,
            value=resolve_bounded({"qbyt_alignment": filtered}),
        )

    if readout is not None:
        raise ValueError(
            "stage2.qbyt_readout is a legacy GRU/EPS switch and is unsupported "
            "by QbyT v6; replace it with stage2.qbyt_alignment"
        )
    filtered = _alignment_fields_for_version(_plain_mapping(alignment), version)
    if filtered is None:
        filtered = {}
    supported = set(QbyTAlignmentSpec.__dataclass_fields__)
    unknown = sorted(set(filtered) - supported)
    if unknown:
        raise ValueError(f"stage2.qbyt_alignment has unknown fields: {unknown}")
    return QbyTScoreSpec(version=version, value=QbyTAlignmentSpec(**dict(filtered)))


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
    "CURRENT_QBYT_READOUT_VERSION",
    "DEFAULT_LOCAL_CONTEXT_KERNEL",
    "DEFAULT_MAX_INTER_PHONE_GAP_FRAMES",
    "DEFAULT_MAX_KEYWORD_SPAN_FRAMES",
    "DEFAULT_MAX_PHONE_DURATION_FRAMES",
    "DEFAULT_MIN_PHONE_DURATION_FRAMES",
    "DEFAULT_WEAKEST_PHONE_TEMPERATURE",
    "DEFAULT_WEAKEST_PHONE_WEIGHT",
    "QBYT_ALIGNMENT_TOPOLOGY",
    "QbyTAlignmentSpec",
    "QbyTScoreSpec",
    "SUPPORTED_QBYT_READOUT_VERSIONS",
    "assert_qbyt_alignment_state_loaded",
    "default_clip_padding_ms",
    "normalize_qbyt_alignment_topology",
    "resolve_qbyt_alignment",
    "resolve_qbyt_readout_version",
    "resolve_qbyt_score_spec",
]
