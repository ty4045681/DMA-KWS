"""Validation and semantic comparison for Stage-II score provenance."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping


LEGACY_PROVENANCE_SCHEMA_VERSION = 3
PROVENANCE_SCHEMA_VERSION = 4
SUPPORTED_PROVENANCE_SCHEMA_VERSIONS = frozenset(
    {LEGACY_PROVENANCE_SCHEMA_VERSION, PROVENANCE_SCHEMA_VERSION}
)

_REQUIRED_FIELDS = {
    "schema_version",
    "checkpoint",
    "calibration",
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
        or schema_version not in SUPPORTED_PROVENANCE_SCHEMA_VERSIONS
    ):
        raise ValueError(
            f"unsupported score provenance version "
            f"{provenance['schema_version']!r} in {source}"
        )
    if schema_version >= PROVENANCE_SCHEMA_VERSION:
        keyword_eval = provenance.get("keyword_eval")
        if not isinstance(keyword_eval, Mapping):
            raise ValueError(f"{source} provenance 'keyword_eval' must be a mapping")
        mode = keyword_eval.get("mode")
        if mode not in {"per_row", "any"}:
            raise ValueError(
                f"{source} provenance keyword_eval.mode must be 'per_row' or 'any'"
            )
        if mode == "any":
            missing_eval = [
                key
                for key in (
                    "aggregation",
                    "keyword_set_id",
                    "texts",
                    "token_sequences",
                )
                if key not in keyword_eval
            ]
            if missing_eval:
                raise ValueError(
                    f"{source} provenance keyword_eval is missing: {missing_eval}"
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

    calibration = provenance.get("calibration")
    if not isinstance(calibration, Mapping):
        raise ValueError(f"{source} provenance 'calibration' must be a mapping")
    if calibration.get("type") == "identity_logit_sigmoid":
        if set(calibration) != {"type"}:
            raise ValueError(
                f"{source} identity calibration has unexpected fields"
            )
    else:
        missing_calibration = sorted(
            {"path", "size_bytes", "sha256"} - set(calibration)
        )
        if missing_calibration:
            raise ValueError(
                f"{source} provenance 'calibration' is missing: "
                f"{missing_calibration}"
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
    calibration = value.get("calibration")
    if isinstance(calibration, dict):
        calibration.pop("path", None)
    from dma_kws.inference.keyword_set import semantic_keyword_eval

    schema_version = value.get("schema_version", LEGACY_PROVENANCE_SCHEMA_VERSION)
    try:
        schema_version = int(schema_version)
    except (TypeError, ValueError):
        schema_version = LEGACY_PROVENANCE_SCHEMA_VERSION
    value["keyword_eval"] = semantic_keyword_eval(
        value.get("keyword_eval"),
        schema_version=schema_version,
    )
    if schema_version in SUPPORTED_PROVENANCE_SCHEMA_VERSIONS:
        value["schema_version"] = PROVENANCE_SCHEMA_VERSION
    return value


__all__ = [
    "LEGACY_PROVENANCE_SCHEMA_VERSION",
    "PROVENANCE_SCHEMA_VERSION",
    "SUPPORTED_PROVENANCE_SCHEMA_VERSIONS",
    "semantic_score_provenance",
    "validate_score_provenance",
]
