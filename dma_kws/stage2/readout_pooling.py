"""Canonical QbyT utterance-readout configuration."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
from typing import Any


GRU_LAST_READOUT = "gru_last"
EPS_MEAN_READOUT = "eps_mean"
EPS_SOFTMIN_READOUT = "eps_softmin"
DEFAULT_QBYT_READOUT_TEMPERATURE = 1.0
QBYT_READOUT_MODES = frozenset(
    {GRU_LAST_READOUT, EPS_MEAN_READOUT, EPS_SOFTMIN_READOUT}
)


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


@dataclass(frozen=True)
class QbyTReadoutConfig:
    """Canonical final-score readout configuration.

    ``temperature`` controls ``eps_softmin`` pooling. It remains canonicalized
    for the other modes as well so checkpoints and evaluation provenance can
    carry one stable configuration shape.
    """

    mode: str = GRU_LAST_READOUT
    temperature: float = DEFAULT_QBYT_READOUT_TEMPERATURE

    def __post_init__(self) -> None:
        object.__setattr__(self, "mode", normalize_qbyt_readout_mode(self.mode))
        object.__setattr__(
            self,
            "temperature",
            _normalize_qbyt_readout_temperature(self.temperature),
        )

    def as_dict(self) -> dict[str, str | float]:
        """Return the stable mapping embedded in checkpoints and provenance."""

        return {
            "mode": self.mode,
            "temperature": self.temperature,
        }


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
    else:
        # Preserve the historical scalar/None shorthand accepted by the
        # mode-only resolver.
        mode = value
        temperature = DEFAULT_QBYT_READOUT_TEMPERATURE
    return QbyTReadoutConfig(mode=mode, temperature=temperature)


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
    prefixes = ("qbyt.gru.", "qbyt.fc.", "qbyt.final_pos_fc.")
    mismatches = [
        key for key in (*missing, *unexpected) if str(key).startswith(prefixes)
    ]
    if mismatches:
        raise SystemExit(
            f"Checkpoint {source} does not carry the configured {mode!r} "
            f"QbyT readout weights: {mismatches}"
        )


__all__ = [
    "DEFAULT_QBYT_READOUT_TEMPERATURE",
    "EPS_MEAN_READOUT",
    "EPS_SOFTMIN_READOUT",
    "GRU_LAST_READOUT",
    "QbyTReadoutConfig",
    "QBYT_READOUT_MODES",
    "assert_qbyt_readout_state_loaded",
    "normalize_qbyt_readout_mode",
    "resolve_qbyt_readout",
    "resolve_qbyt_readout_mode",
    "resolve_qbyt_readout_temperature",
]
