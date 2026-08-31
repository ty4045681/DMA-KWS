"""Validation and semantic comparison for Stage-II score provenance."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping


PROVENANCE_SCHEMA_VERSION = 2

_REQUIRED_FIELDS = {
    "schema_version",
    "checkpoint",
    "qbyt_alignment",
    "stream",
    "audio_padding_ms",
    "fbank",
    "tokenizer",
    "sequence_objective",
}

_REQUIRED_SECTION_FIELDS = {
    "checkpoint": {"path", "size_bytes", "sha256"},
    "tokenizer": {"path", "size_bytes", "sha256", "split_with_space"},
}


def validate_score_provenance(
    provenance: object,
    *,
    source: object = "score provenance",
) -> dict[str, Any]:
    """Validate the versioned identity needed to compare Stage-II scores."""

    if not isinstance(provenance, Mapping):
        raise ValueError(f"{source} provenance must be a mapping")

    missing = sorted(_REQUIRED_FIELDS - set(provenance))
    if missing:
        raise ValueError(f"{source} provenance is missing fields: {missing}")

    schema_version = provenance["schema_version"]
    if (
        isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version != PROVENANCE_SCHEMA_VERSION
    ):
        raise ValueError(
            f"unsupported score provenance version "
            f"{provenance['schema_version']!r} in {source}"
        )

    for section, required_fields in _REQUIRED_SECTION_FIELDS.items():
        value = provenance.get(section)
        if not isinstance(value, Mapping):
            raise ValueError(f"{source} provenance {section!r} must be a mapping")
        missing_section = sorted(required_fields - set(value))
        if missing_section:
            raise ValueError(
                f"{source} provenance {section!r} is missing: {missing_section}"
            )

    return dict(provenance)


def semantic_score_provenance(provenance: Mapping[str, Any]) -> dict[str, Any]:
    """Drop location-only paths while retaining content and score semantics."""

    value = deepcopy(dict(provenance))
    checkpoint = value.get("checkpoint")
    if isinstance(checkpoint, dict):
        checkpoint.pop("path", None)
    tokenizer = value.get("tokenizer")
    if isinstance(tokenizer, dict):
        tokenizer.pop("path", None)
    return value


__all__ = [
    "PROVENANCE_SCHEMA_VERSION",
    "semantic_score_provenance",
    "validate_score_provenance",
]
