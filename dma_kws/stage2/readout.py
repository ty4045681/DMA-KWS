"""Canonical QbyT utterance-readout configuration."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


GRU_LAST_READOUT = "gru_last"
EPS_MEAN_READOUT = "eps_mean"
QBYT_READOUT_MODES = frozenset({GRU_LAST_READOUT, EPS_MEAN_READOUT})


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


def resolve_qbyt_readout_mode(stage2_config: Mapping[str, Any]) -> str:
    """Resolve ``stage2.qbyt_readout`` with the legacy GRU default."""

    return normalize_qbyt_readout_mode(stage2_config.get("qbyt_readout"))


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
    "EPS_MEAN_READOUT",
    "GRU_LAST_READOUT",
    "QBYT_READOUT_MODES",
    "assert_qbyt_readout_state_loaded",
    "normalize_qbyt_readout_mode",
    "resolve_qbyt_readout_mode",
]
