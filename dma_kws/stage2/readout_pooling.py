"""Canonical QbyT utterance-readout configuration."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
from numbers import Integral, Real
from typing import Any


GRU_LAST_READOUT = "gru_last"
EPS_MEAN_READOUT = "eps_mean"
EPS_SOFTMIN_READOUT = "eps_softmin"
DEFAULT_QBYT_READOUT_TEMPERATURE = 1.0
DEFAULT_QBYT_SINK_TOKEN = False
DEFAULT_QBYT_TEXT_POSITION = "sinusoidal"
DEFAULT_QBYT_AUDIO_POSITION = "sinusoidal"
DEFAULT_QBYT_RELATIVE_NUM_BUCKETS = 32
DEFAULT_QBYT_RELATIVE_MAX_DISTANCE = 64
DEFAULT_QBYT_SINK_READOUT = "none"
DEFAULT_QBYT_SINK_IDENTITY = False
DEFAULT_QBYT_SINK_ZERO_INIT = False
DEFAULT_QBYT_TEMPERATURE_LEARNABLE = False
_MIN_RELATIVE_NUM_BUCKETS = 4
QBYT_READOUT_MODES = frozenset(
    {GRU_LAST_READOUT, EPS_MEAN_READOUT, EPS_SOFTMIN_READOUT}
)
QBYT_TEXT_POSITIONS = frozenset({"sinusoidal", "learned"})
QBYT_AUDIO_POSITIONS = frozenset({"sinusoidal", "relative_bias"})
QBYT_SINK_READOUTS = frozenset({"none", "additive", "mixture"})
QBYT_POOLING_EXTENSION_FIELDS = (
    "sink_token",
    "text_position",
    "audio_position",
    "relative_num_buckets",
    "relative_max_distance",
    "sink_readout",
    "sink_identity",
    "sink_zero_init",
    "temperature_learnable",
    "score_temperature",
)


def normalize_qbyt_sink_readout(value: Any) -> str:
    """Validate the v4.1+ sink-state readout switch."""

    normalized = str(value).strip().lower()
    if normalized not in QBYT_SINK_READOUTS:
        choices = ", ".join(sorted(QBYT_SINK_READOUTS))
        raise ValueError(
            f"Unsupported QbyT sink_readout {value!r}; expected one of: {choices}"
        )
    return normalized


def _normalize_qbyt_bool(value: Any, *, field: str) -> bool:
    """Accept real booleans plus the string/int forms Hydra can produce."""

    if isinstance(value, bool):
        return value
    if isinstance(value, (int, str)):
        text = str(value).strip().lower()
        if text in {"1", "true", "yes", "on"}:
            return True
        if text in {"0", "false", "no", "off"}:
            return False
    raise ValueError(f"QbyT readout {field} must be a boolean, got {value!r}")


def normalize_qbyt_readout_mode(value: Any) -> str:
    """Resolve a scalar or ``{mode: ...}`` readout config."""

    if isinstance(value, Mapping):
        value = value.get("mode", GRU_LAST_READOUT)
    if value is None:
        value = GRU_LAST_READOUT
    mode = str(value).strip().lower()
    if mode not in QBYT_READOUT_MODES:
        choices = ", ".join(sorted(QBYT_READOUT_MODES))
        raise ValueError(
            f"Unsupported QbyT readout mode {value!r}; expected one of: {choices}"
        )
    return mode


def _normalize_qbyt_readout_temperature(value: Any) -> float:
    if isinstance(value, bool):
        raise ValueError(
            "QbyT readout temperature must be a finite number greater than 0, "
            f"got {value!r}"
        )
    try:
        temperature = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "QbyT readout temperature must be a finite number greater than 0, "
            f"got {value!r}"
        ) from exc
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError(
            "QbyT readout temperature must be a finite number greater than 0, "
            f"got {value!r}"
        )
    return temperature


def _normalize_qbyt_sink_token(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, Integral) and value in (0, 1):
        return bool(value)
    raise ValueError(f"QbyT readout sink_token must be a bool, got {value!r}")


def _normalize_qbyt_position_choice(
    value: Any, *, field: str, allowed: frozenset[str]
) -> str:
    choice = str(value).strip().lower()
    if choice not in allowed:
        choices = ", ".join(sorted(allowed))
        raise ValueError(
            f"QbyT readout {field} must be one of: {choices}; got {value!r}"
        )
    return choice


def _normalize_qbyt_positive_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"QbyT readout {field} must be a positive integer")
    if isinstance(value, Integral):
        parsed = int(value)
    elif isinstance(value, Real):
        numeric = float(value)
        if not math.isfinite(numeric) or not numeric.is_integer():
            raise ValueError(
                f"QbyT readout {field} must be a positive integer, got {value!r}"
            )
        parsed = int(numeric)
    else:
        text = str(value).strip()
        try:
            parsed = int(text)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"QbyT readout {field} must be a positive integer, got {value!r}"
            ) from exc
        if str(parsed) != text.lstrip("+"):
            raise ValueError(
                f"QbyT readout {field} must be a positive integer, got {value!r}"
            )
    if parsed < 1:
        raise ValueError(
            f"QbyT readout {field} must be a positive integer, got {value!r}"
        )
    return parsed


def _validate_relative_attention_buckets(
    num_buckets: int, max_distance: int
) -> None:
    """Reject T5 bucket sizes that divide by zero or index past the table."""
    if num_buckets < _MIN_RELATIVE_NUM_BUCKETS:
        raise ValueError(
            "QbyT readout relative_num_buckets must be >= "
            f"{_MIN_RELATIVE_NUM_BUCKETS} so bidirectional T5 buckets have a "
            f"non-zero exact range; got {num_buckets}"
        )
    max_exact = num_buckets // 4
    if max_distance <= max_exact:
        raise ValueError(
            "QbyT readout relative_max_distance must be greater than "
            f"relative_num_buckets // 4 ({max_exact}); got "
            f"relative_num_buckets={num_buckets}, "
            f"relative_max_distance={max_distance}"
        )


@dataclass(frozen=True)
class QbyTReadoutConfig:
    """Canonical final-score readout configuration.

    ``temperature`` controls ``eps_softmin`` pooling. It remains canonicalized
    for the other modes as well so checkpoints and evaluation provenance can
    carry one stable configuration shape. Pooling v4.1 knobs default to the
    legacy v4 embedding path.
    """

    mode: str = GRU_LAST_READOUT
    temperature: float = DEFAULT_QBYT_READOUT_TEMPERATURE
    sink_token: bool = DEFAULT_QBYT_SINK_TOKEN
    text_position: str = DEFAULT_QBYT_TEXT_POSITION
    audio_position: str = DEFAULT_QBYT_AUDIO_POSITION
    relative_num_buckets: int = DEFAULT_QBYT_RELATIVE_NUM_BUCKETS
    relative_max_distance: int = DEFAULT_QBYT_RELATIVE_MAX_DISTANCE
    sink_readout: str = DEFAULT_QBYT_SINK_READOUT
    sink_identity: bool = DEFAULT_QBYT_SINK_IDENTITY
    sink_zero_init: bool = DEFAULT_QBYT_SINK_ZERO_INIT
    temperature_learnable: bool = DEFAULT_QBYT_TEMPERATURE_LEARNABLE
    #: Scoring temperature used at eval/inference; None = the training
    #: temperature (S8b: training and scoring temperatures are decoupled).
    score_temperature: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "mode", normalize_qbyt_readout_mode(self.mode))
        object.__setattr__(
            self,
            "temperature",
            _normalize_qbyt_readout_temperature(self.temperature),
        )
        object.__setattr__(
            self, "sink_token", _normalize_qbyt_sink_token(self.sink_token)
        )
        object.__setattr__(
            self,
            "text_position",
            _normalize_qbyt_position_choice(
                self.text_position,
                field="text_position",
                allowed=QBYT_TEXT_POSITIONS,
            ),
        )
        object.__setattr__(
            self,
            "audio_position",
            _normalize_qbyt_position_choice(
                self.audio_position,
                field="audio_position",
                allowed=QBYT_AUDIO_POSITIONS,
            ),
        )
        object.__setattr__(
            self,
            "relative_num_buckets",
            _normalize_qbyt_positive_int(
                self.relative_num_buckets, field="relative_num_buckets"
            ),
        )
        object.__setattr__(
            self,
            "relative_max_distance",
            _normalize_qbyt_positive_int(
                self.relative_max_distance, field="relative_max_distance"
            ),
        )
        _validate_relative_attention_buckets(
            self.relative_num_buckets, self.relative_max_distance
        )
        object.__setattr__(
            self, "sink_readout", normalize_qbyt_sink_readout(self.sink_readout)
        )
        object.__setattr__(
            self,
            "sink_identity",
            _normalize_qbyt_bool(self.sink_identity, field="sink_identity"),
        )
        object.__setattr__(
            self,
            "sink_zero_init",
            _normalize_qbyt_bool(self.sink_zero_init, field="sink_zero_init"),
        )
        object.__setattr__(
            self,
            "temperature_learnable",
            _normalize_qbyt_bool(
                self.temperature_learnable, field="temperature_learnable"
            ),
        )
        if self.sink_zero_init and self.sink_readout == DEFAULT_QBYT_SINK_READOUT:
            raise ValueError("QbyT sink_zero_init requires a sink_readout")
        if self.score_temperature is not None:
            object.__setattr__(
                self,
                "score_temperature",
                _normalize_qbyt_readout_temperature(self.score_temperature),
            )
        if self.sink_readout != DEFAULT_QBYT_SINK_READOUT and not self.sink_token:
            raise ValueError("QbyT sink_readout requires sink_token=True")
        if self.sink_identity and not self.sink_token:
            raise ValueError("QbyT sink_identity requires sink_token=True")
        if self.temperature_learnable and self.mode != EPS_SOFTMIN_READOUT:
            raise ValueError(
                "QbyT temperature_learnable requires the eps_softmin readout"
            )
        if self.sink_readout != DEFAULT_QBYT_SINK_READOUT and self.mode != EPS_SOFTMIN_READOUT:
            raise ValueError("QbyT sink_readout requires the eps_softmin readout")

    def as_dict(self) -> dict[str, str | float | bool | int]:
        """Return the stable mapping embedded in checkpoints and provenance."""

        return {
            "mode": self.mode,
            "temperature": self.temperature,
            "sink_token": self.sink_token,
            "text_position": self.text_position,
            "audio_position": self.audio_position,
            "relative_num_buckets": self.relative_num_buckets,
            "relative_max_distance": self.relative_max_distance,
            "sink_readout": self.sink_readout,
            "sink_identity": self.sink_identity,
            "sink_zero_init": self.sink_zero_init,
            "temperature_learnable": self.temperature_learnable,
            "score_temperature": self.score_temperature,
        }


def pooling_extension_offenders(config: QbyTReadoutConfig) -> list[str]:
    """Names of v4.1 fields that differ from the legacy v4 defaults."""

    defaults = QbyTReadoutConfig()
    return [
        name
        for name in QBYT_POOLING_EXTENSION_FIELDS
        if getattr(config, name) != getattr(defaults, name)
    ]


def resolve_qbyt_readout(
    stage2_config: Mapping[str, Any],
) -> QbyTReadoutConfig:
    """Resolve and validate ``stage2.qbyt_readout`` in canonical form."""

    value = stage2_config.get("qbyt_readout")
    if isinstance(value, Mapping):
        mode = value.get("mode", GRU_LAST_READOUT)
        temperature = value.get(
            "temperature", DEFAULT_QBYT_READOUT_TEMPERATURE
        )
        sink_token = value.get("sink_token", DEFAULT_QBYT_SINK_TOKEN)
        text_position = value.get("text_position", DEFAULT_QBYT_TEXT_POSITION)
        audio_position = value.get(
            "audio_position", DEFAULT_QBYT_AUDIO_POSITION
        )
        relative_num_buckets = value.get(
            "relative_num_buckets", DEFAULT_QBYT_RELATIVE_NUM_BUCKETS
        )
        relative_max_distance = value.get(
            "relative_max_distance", DEFAULT_QBYT_RELATIVE_MAX_DISTANCE
        )
        sink_readout = value.get("sink_readout", DEFAULT_QBYT_SINK_READOUT)
        sink_identity = value.get("sink_identity", DEFAULT_QBYT_SINK_IDENTITY)
        sink_zero_init = value.get("sink_zero_init", DEFAULT_QBYT_SINK_ZERO_INIT)
        score_temperature = value.get("score_temperature", None)
        temperature_learnable = value.get(
            "temperature_learnable", DEFAULT_QBYT_TEMPERATURE_LEARNABLE
        )
    else:
        # Preserve the historical scalar/None shorthand accepted by the
        # mode-only resolver. Extension knobs stay at the legacy v4 defaults.
        mode = value
        temperature = DEFAULT_QBYT_READOUT_TEMPERATURE
        sink_token = DEFAULT_QBYT_SINK_TOKEN
        text_position = DEFAULT_QBYT_TEXT_POSITION
        audio_position = DEFAULT_QBYT_AUDIO_POSITION
        relative_num_buckets = DEFAULT_QBYT_RELATIVE_NUM_BUCKETS
        relative_max_distance = DEFAULT_QBYT_RELATIVE_MAX_DISTANCE
        sink_readout = DEFAULT_QBYT_SINK_READOUT
        sink_identity = DEFAULT_QBYT_SINK_IDENTITY
        sink_zero_init = DEFAULT_QBYT_SINK_ZERO_INIT
        temperature_learnable = DEFAULT_QBYT_TEMPERATURE_LEARNABLE
        score_temperature = None
    return QbyTReadoutConfig(
        mode=mode,
        temperature=temperature,
        sink_token=sink_token,
        text_position=text_position,
        audio_position=audio_position,
        relative_num_buckets=relative_num_buckets,
        relative_max_distance=relative_max_distance,
        sink_readout=sink_readout,
        sink_identity=sink_identity,
        sink_zero_init=sink_zero_init,
        temperature_learnable=temperature_learnable,
        score_temperature=score_temperature,
    )


def resolve_qbyt_readout_mode(stage2_config: Mapping[str, Any]) -> str:
    """Resolve ``stage2.qbyt_readout`` with the legacy GRU default."""

    return resolve_qbyt_readout(stage2_config).mode


def resolve_qbyt_readout_temperature(stage2_config: Mapping[str, Any]) -> float:
    """Return the validated readout temperature, defaulting to ``1.0``."""

    return resolve_qbyt_readout(stage2_config).temperature


def assert_qbyt_readout_state_loaded(
    missing: list[str] | tuple[str, ...],
    unexpected: list[str] | tuple[str, ...],
    *,
    source: Any,
    expected_mode: str,
) -> None:
    """Reject a non-strict eval load that omitted or replaced the final head."""

    mode = normalize_qbyt_readout_mode(expected_mode)
    prefixes = (
        "qbyt.gru.",
        "qbyt.fc.",
        "qbyt.final_pos_fc.",
        "qbyt.sink_token",
        "qbyt.text_pos_emb.",
        "qbyt.relative_bias.",
    )
    mismatches = [
        key for key in (*missing, *unexpected) if str(key).startswith(prefixes)
    ]
    if mismatches:
        raise SystemExit(
            f"Checkpoint {source} does not carry the configured {mode!r} "
            f"QbyT readout weights: {mismatches}"
        )


__all__ = [
    "DEFAULT_QBYT_AUDIO_POSITION",
    "DEFAULT_QBYT_READOUT_TEMPERATURE",
    "DEFAULT_QBYT_RELATIVE_MAX_DISTANCE",
    "DEFAULT_QBYT_RELATIVE_NUM_BUCKETS",
    "DEFAULT_QBYT_SINK_TOKEN",
    "DEFAULT_QBYT_TEXT_POSITION",
    "EPS_MEAN_READOUT",
    "EPS_SOFTMIN_READOUT",
    "GRU_LAST_READOUT",
    "QbyTReadoutConfig",
    "QBYT_AUDIO_POSITIONS",
    "QBYT_POOLING_EXTENSION_FIELDS",
    "QBYT_READOUT_MODES",
    "QBYT_TEXT_POSITIONS",
    "assert_qbyt_readout_state_loaded",
    "normalize_qbyt_readout_mode",
    "pooling_extension_offenders",
    "resolve_qbyt_readout",
    "resolve_qbyt_readout_mode",
    "resolve_qbyt_readout_temperature",
]
