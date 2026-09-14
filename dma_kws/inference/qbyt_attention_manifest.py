"""Diagnostic-column validation for pooling QbyT attention manifests.

Call :func:`validate_attention_manifest_rows` after the shared
:func:`dma_kws.inference.manifest.load_manifest`. Extra columns stay on the
loader; this module only applies the diagnostics-specific rules.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


__all__ = [
    "AttentionManifestRow",
    "validate_attention_manifest_rows",
]


_KNOWN_COLUMNS = frozenset(
    {
        "audio_path",
        "keyword",
        "label",
        "keyword_phonemes",
        "sample_id",
        "condition",
        "pair_id",
        "keyword_spans",
        "noise_spans",
        "pronunciation_id",
    }
)
_SAFE_ID = re.compile(r"[^A-Za-z0-9_]+")

PAIR_OK = "ok"
PAIR_MISSING_CLEAN = "missing_clean_baseline"
PAIR_MULTIPLE_CLEAN = "multiple_clean_baselines"
PAIR_INCONSISTENT = "inconsistent_keyword_or_phonemes"


@dataclass(frozen=True)
class AttentionManifestRow:
    """One diagnostics-ready manifest row after extra-column validation."""

    audio_path: str
    keyword: str
    label: int | None
    keyword_phonemes: object | None
    phoneme_override: object | None
    sample_id: str
    internal_sample_id: str
    condition: str
    pair_id: str | None
    keyword_spans: tuple[tuple[float, float], ...] | None
    noise_spans: tuple[tuple[float, float], ...] | None
    pronunciation_id: str | None
    pair_status: str
    pair_reason: str | None
    record_number: int
    extra_fields: dict[str, Any]


def validate_attention_manifest_rows(
    rows: list[dict],
    *,
    source_durations_sec: Sequence[float | None] | None = None,
    first_record_number: int = 2,
) -> list[AttentionManifestRow]:
    """Validate diagnostics columns on rows already loaded by ``load_manifest``."""

    if isinstance(first_record_number, bool) or not isinstance(first_record_number, int):
        raise TypeError(
            f"first_record_number must be an integer, got {first_record_number!r}"
        )
    if source_durations_sec is not None and len(source_durations_sec) != len(rows):
        raise ValueError(
            "source_durations_sec has "
            f"{len(source_durations_sec)} entries for {len(rows)} rows"
        )

    parsed: list[dict[str, Any]] = []
    seen_ids: dict[str, int] = {}
    for offset, row in enumerate(rows):
        record = first_record_number + offset
        duration = None if source_durations_sec is None else source_durations_sec[offset]
        item = _parse_row(row, record_number=record, duration_sec=duration)
        sample_id = item["sample_id"]
        previous = seen_ids.get(sample_id)
        if previous is not None:
            raise ValueError(
                f"Manifest row {record} duplicates sample_id {sample_id!r} "
                f"from row {previous}"
            )
        seen_ids[sample_id] = record
        parsed.append(item)

    used_internal: set[str] = set()
    for item in parsed:
        item["internal_sample_id"] = _internal_sample_id(
            item["sample_id"], used_internal
        )

    _assign_pair_status(parsed)

    return [
        AttentionManifestRow(
            audio_path=item["audio_path"],
            keyword=item["keyword"],
            label=item["label"],
            keyword_phonemes=item["keyword_phonemes"],
            phoneme_override=item["phoneme_override"],
            sample_id=item["sample_id"],
            internal_sample_id=item["internal_sample_id"],
            condition=item["condition"],
            pair_id=item["pair_id"],
            keyword_spans=item["keyword_spans"],
            noise_spans=item["noise_spans"],
            pronunciation_id=item["pronunciation_id"],
            pair_status=item["pair_status"],
            pair_reason=item["pair_reason"],
            record_number=item["record_number"],
            extra_fields=item["extra_fields"],
        )
        for item in parsed
    ]


def _parse_row(
    row: Mapping[str, Any],
    *,
    record_number: int,
    duration_sec: float | None,
) -> dict[str, Any]:
    audio_path = row.get("audio_path", "")
    keyword = row.get("keyword", "")
    if not audio_path:
        raise ValueError(f"Manifest row {record_number} is missing audio_path")
    if not keyword:
        raise ValueError(f"Manifest row {record_number} is missing keyword")

    label: int | None
    if "label" in row:
        label = _binary_label(row["label"], record_number=record_number)
    else:
        label = None

    if "keyword_phonemes" in row:
        keyword_phonemes: object | None = row["keyword_phonemes"]
    else:
        keyword_phonemes = None
    phoneme_override = _phoneme_override(keyword_phonemes, present="keyword_phonemes" in row)

    raw_sample_id = row.get("sample_id", None)
    if raw_sample_id is None or raw_sample_id == "":
        sample_id = f"row_{record_number:06d}"
    else:
        sample_id = str(raw_sample_id)
        if sample_id == "":
            raise ValueError(f"Manifest row {record_number} sample_id must be non-empty")

    raw_condition = row.get("condition", None)
    if raw_condition is None or raw_condition == "":
        condition = "unknown"
    else:
        condition = str(raw_condition)

    raw_pair = row.get("pair_id", None)
    pair_id = None if raw_pair is None or raw_pair == "" else str(raw_pair)

    raw_pronunciation = row.get("pronunciation_id", None)
    pronunciation_id = (
        None
        if raw_pronunciation is None or raw_pronunciation == ""
        else str(raw_pronunciation)
    )

    duration = _optional_duration(duration_sec, record_number=record_number)
    keyword_spans = _parse_span_cell(
        row.get("keyword_spans") if "keyword_spans" in row else None,
        field="keyword_spans",
        record_number=record_number,
        duration_sec=duration,
        present="keyword_spans" in row,
    )
    noise_spans = _parse_span_cell(
        row.get("noise_spans") if "noise_spans" in row else None,
        field="noise_spans",
        record_number=record_number,
        duration_sec=duration,
        present="noise_spans" in row,
    )

    extra_fields = {
        key: value for key, value in row.items() if key not in _KNOWN_COLUMNS
    }
    return {
        "audio_path": str(audio_path),
        "keyword": str(keyword),
        "label": label,
        "keyword_phonemes": keyword_phonemes,
        "phoneme_override": phoneme_override,
        "sample_id": sample_id,
        "condition": condition,
        "pair_id": pair_id,
        "keyword_spans": keyword_spans,
        "noise_spans": noise_spans,
        "pronunciation_id": pronunciation_id,
        "record_number": record_number,
        "extra_fields": extra_fields,
        "pair_status": PAIR_OK,
        "pair_reason": None,
    }


def _binary_label(value: object, *, record_number: int) -> int:
    if isinstance(value, bool) or type(value) is not int or value not in (0, 1):
        raise ValueError(
            f"Manifest row {record_number} label must be integer 0 or 1, got {value!r}"
        )
    return value


def _phoneme_override(value: object | None, *, present: bool) -> object | None:
    if not present or value is None:
        return None
    if isinstance(value, str) and value == "":
        return None
    return value


def _optional_duration(value: float | None, *, record_number: int) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            f"Manifest row {record_number} source duration must be a finite number, "
            f"got {value!r}"
        )
    duration = float(value)
    if not math.isfinite(duration) or duration < 0.0:
        raise ValueError(
            f"Manifest row {record_number} source duration must be a finite "
            f"non-negative number, got {value!r}"
        )
    return duration


def _parse_span_cell(
    value: object,
    *,
    field: str,
    record_number: int,
    duration_sec: float | None,
    present: bool,
) -> tuple[tuple[float, float], ...] | None:
    if not present:
        return None
    if value is None:
        return None
    parsed: object
    if isinstance(value, str):
        if value == "":
            return None
        try:
            parsed = json.loads(value, parse_constant=_reject_json_constant)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Manifest row {record_number} {field} is not valid JSON: {exc.msg}"
            ) from exc
        except ValueError as exc:
            raise ValueError(
                f"Manifest row {record_number} {field} {exc}"
            ) from exc
    else:
        parsed = value
    if not isinstance(parsed, list):
        raise ValueError(
            f"Manifest row {record_number} {field} must be a JSON array of "
            f"[start, end) pairs, got {value!r}"
        )
    spans: list[tuple[float, float]] = []
    for item in parsed:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError(
                f"Manifest row {record_number} {field} entries must be [start, end) "
                f"pairs, got {item!r}"
            )
        start = _finite_number(
            item[0], field=field, record_number=record_number, role="start"
        )
        end = _finite_number(
            item[1], field=field, record_number=record_number, role="end"
        )
        if not 0.0 <= start < end:
            raise ValueError(
                f"Manifest row {record_number} {field} requires 0 <= start < end, "
                f"got [{start}, {end})"
            )
        if duration_sec is not None and end > duration_sec:
            raise ValueError(
                f"Manifest row {record_number} {field} end {end} exceeds source "
                f"duration {duration_sec}"
            )
        spans.append((start, end))
    return tuple(spans)


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"rejects non-finite JSON constant {value}")


def _finite_number(
    value: object,
    *,
    field: str,
    record_number: int,
    role: str,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            f"Manifest row {record_number} {field} {role} must be a finite number, "
            f"got {value!r}"
        )
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(
            f"Manifest row {record_number} {field} {role} must be a finite number, "
            f"got {value!r}"
        )
    return number


def _internal_sample_id(sample_id: str, used: set[str]) -> str:
    digest = hashlib.sha1(sample_id.encode("utf-8")).hexdigest()[:10]
    cleaned = _SAFE_ID.sub("_", sample_id).strip("_")
    if not cleaned:
        cleaned = "sample"
    candidate = f"{cleaned}_{digest}"
    suffix = 2
    while candidate in used:
        candidate = f"{cleaned}_{digest}_{suffix}"
        suffix += 1
    used.add(candidate)
    return candidate


def _override_key(value: object | None) -> object | None:
    if isinstance(value, list):
        return tuple(value)
    return value


def _assign_pair_status(parsed: list[dict[str, Any]]) -> None:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, item in enumerate(parsed):
        pair_id = item["pair_id"]
        if pair_id:
            groups[pair_id].append(index)

    for pair_id, indices in groups.items():
        keys = {
            (parsed[index]["keyword"], _override_key(parsed[index]["phoneme_override"]))
            for index in indices
        }
        if len(keys) > 1:
            status = PAIR_INCONSISTENT
            reason = (
                f"pair_id {pair_id!r} has inconsistent keyword or phoneme override"
            )
        else:
            n_clean = sum(parsed[index]["condition"] == "clean" for index in indices)
            if n_clean == 0:
                status = PAIR_MISSING_CLEAN
                reason = f"pair_id {pair_id!r} has no clean baseline"
            elif n_clean > 1:
                status = PAIR_MULTIPLE_CLEAN
                reason = f"pair_id {pair_id!r} has {n_clean} clean baselines"
            else:
                status = PAIR_OK
                reason = None
        for index in indices:
            parsed[index]["pair_status"] = status
            parsed[index]["pair_reason"] = reason
