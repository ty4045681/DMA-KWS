"""Shared background recording catalog types, hashes, and isolation checks."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

from dma_kws.jsonl import read_jsonl


BACKGROUND_MANIFEST_SCHEMA_VERSION = 1
RECORDINGS_JSONL_NAME = "recordings.jsonl"
CATALOG_JSON_NAME = "catalog.json"
ALLOWED_SPLITS = frozenset({"train", "val", "test"})
PROVENANCE_CAPABILITY_LIMIT = (
    "only verifiable identity and exact-byte duplicate checks were performed; "
    "transcode near-duplicates were not fully detected"
)

RECORD_FIELD_ORDER = (
    "schema_version",
    "dataset_id",
    "recording_id",
    "relative_path",
    "audio_path",
    "group_id",
    "origin_ids",
    "split",
    "categories",
    "background_eligible",
    "eligibility_basis",
    "duration_seconds",
    "sample_rate",
    "channels",
    "audio_sha256",
    "license_id",
    "provenance_complete",
)
SEMANTIC_RECORD_FIELDS = (
    "schema_version",
    "dataset_id",
    "recording_id",
    "relative_path",
    "group_id",
    "origin_ids",
    "split",
    "categories",
    "background_eligible",
    "eligibility_basis",
    "duration_seconds",
    "sample_rate",
    "channels",
    "audio_sha256",
    "license_id",
    "provenance_complete",
)


@dataclass(frozen=True)
class BackgroundRecord:
    """One catalog recording. Locators are not part of semantic identity."""

    schema_version: int
    dataset_id: str
    recording_id: str
    relative_path: str
    audio_path: str
    group_id: str
    origin_ids: tuple[str, ...]
    split: str
    categories: tuple[str, ...]
    background_eligible: bool
    eligibility_basis: str
    duration_seconds: float
    sample_rate: int
    channels: int
    audio_sha256: str
    license_id: str
    provenance_complete: bool = False


@dataclass(frozen=True)
class BackgroundCatalog:
    """catalog.json beside recordings.jsonl."""

    dataset_id: str
    schema_version: int
    root: str
    filter_policy_version: str
    filter_policy_hash: str
    raw_metadata_hashes: Mapping[str, str]
    split_seed: int | None
    split_rules: Mapping[str, Any]
    stats: Mapping[str, Any]
    recordings_byte_hash: str
    semantic_catalog_hash: str


@dataclass(frozen=True)
class IsolationAuditReport:
    provenance_complete: bool
    capability_limit: str = ""


def canonical_json_dumps(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _as_str_tuple(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        items = (value,)
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        items = tuple(str(item) for item in value)
    else:
        raise ValueError(f"expected a string list, got {value!r}")
    return tuple(item for item in items if item.strip())


def background_record_from_mapping(payload: Mapping[str, Any]) -> BackgroundRecord:
    if not isinstance(payload, Mapping):
        raise ValueError("background record must be a mapping")
    schema_version = int(
        payload.get("schema_version", BACKGROUND_MANIFEST_SCHEMA_VERSION)
    )
    if schema_version != BACKGROUND_MANIFEST_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported background record schema_version {schema_version}"
        )
    split = str(payload.get("split", "")).strip()
    if split not in ALLOWED_SPLITS:
        raise ValueError(f"background record split must be train/val/test, got {split!r}")
    origin_ids = _as_str_tuple(payload.get("origin_ids"))
    if not origin_ids:
        provenance_complete = False
    elif "provenance_complete" in payload:
        provenance_complete = payload.get("provenance_complete") is True
    else:
        provenance_complete = True
    return BackgroundRecord(
        schema_version=schema_version,
        dataset_id=str(payload.get("dataset_id", "")),
        recording_id=str(payload.get("recording_id", "")),
        relative_path=str(payload.get("relative_path", "")),
        audio_path=str(payload.get("audio_path", "")),
        group_id=str(payload.get("group_id", "")),
        origin_ids=origin_ids,
        split=split,
        categories=_as_str_tuple(payload.get("categories")),
        background_eligible=payload.get("background_eligible") is True,
        eligibility_basis=str(payload.get("eligibility_basis", "")),
        duration_seconds=float(payload.get("duration_seconds", 0.0)),
        sample_rate=int(payload.get("sample_rate", 0)),
        channels=int(payload.get("channels", 0)),
        audio_sha256=str(payload.get("audio_sha256", "")),
        license_id=str(payload.get("license_id", "")),
        provenance_complete=provenance_complete,
    )


def background_record_to_mapping(record: BackgroundRecord) -> dict[str, Any]:
    payload = {
        "schema_version": record.schema_version,
        "dataset_id": record.dataset_id,
        "recording_id": record.recording_id,
        "relative_path": record.relative_path,
        "audio_path": record.audio_path,
        "group_id": record.group_id,
        "origin_ids": list(record.origin_ids),
        "split": record.split,
        "categories": list(record.categories),
        "background_eligible": record.background_eligible,
        "eligibility_basis": record.eligibility_basis,
        "duration_seconds": record.duration_seconds,
        "sample_rate": record.sample_rate,
        "channels": record.channels,
        "audio_sha256": record.audio_sha256,
        "license_id": record.license_id,
        "provenance_complete": record.provenance_complete,
    }
    return {key: payload[key] for key in RECORD_FIELD_ORDER}


def semantic_record_payload(record: BackgroundRecord) -> dict[str, Any]:
    mapping = background_record_to_mapping(record)
    return {key: mapping[key] for key in SEMANTIC_RECORD_FIELDS}


def write_recordings_jsonl(
    path: str | Path, records: Sequence[BackgroundRecord]
) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(canonical_json_dumps(background_record_to_mapping(record)))
            handle.write("\n")


def read_recordings_jsonl(path: str | Path) -> list[BackgroundRecord]:
    return [background_record_from_mapping(row) for row in read_jsonl(Path(path))]


def semantic_catalog_hash(
    records: Sequence[BackgroundRecord],
    *,
    filter_policy_version: str = "",
    filter_policy_hash: str = "",
    split_seed: int | None = None,
    split_rules: Mapping[str, Any] | None = None,
) -> str:
    ordered = sorted(records, key=lambda item: (item.split, item.recording_id))
    payload = {
        "filter_policy_hash": filter_policy_hash,
        "filter_policy_version": filter_policy_version,
        "records": [semantic_record_payload(record) for record in ordered],
        "split_rules": dict(split_rules or {}),
        "split_seed": split_seed,
    }
    return hashlib.sha256(canonical_json_dumps(payload).encode("utf-8")).hexdigest()


def catalog_stats(records: Sequence[BackgroundRecord]) -> dict[str, int]:
    return {
        "num_recordings": len(records),
        "num_train": sum(record.split == "train" for record in records),
        "num_val": sum(record.split == "val" for record in records),
        "num_test": sum(record.split == "test" for record in records),
        "num_eligible_train": len(eligible_train_records(records)),
    }


def build_catalog(
    *,
    dataset_id: str,
    root: str,
    records: Sequence[BackgroundRecord],
    recordings_path: str | Path,
    filter_policy_version: str = "",
    filter_policy_hash: str = "",
    split_seed: int | None = None,
    split_rules: Mapping[str, Any] | None = None,
    raw_metadata_hashes: Mapping[str, str] | None = None,
    stats: Mapping[str, Any] | None = None,
) -> BackgroundCatalog:
    recordings_path = Path(recordings_path)
    return BackgroundCatalog(
        dataset_id=dataset_id,
        schema_version=BACKGROUND_MANIFEST_SCHEMA_VERSION,
        root=str(root),
        filter_policy_version=filter_policy_version,
        filter_policy_hash=filter_policy_hash,
        raw_metadata_hashes=dict(raw_metadata_hashes or {}),
        split_seed=split_seed,
        split_rules=dict(split_rules or {}),
        stats=dict(stats) if stats is not None else catalog_stats(records),
        recordings_byte_hash=sha256_file(recordings_path),
        semantic_catalog_hash=semantic_catalog_hash(
            records,
            filter_policy_version=filter_policy_version,
            filter_policy_hash=filter_policy_hash,
            split_seed=split_seed,
            split_rules=split_rules,
        ),
    )


def catalog_to_mapping(catalog: BackgroundCatalog) -> dict[str, Any]:
    return {
        "dataset_id": catalog.dataset_id,
        "filter_policy_hash": catalog.filter_policy_hash,
        "filter_policy_version": catalog.filter_policy_version,
        "raw_metadata_hashes": dict(catalog.raw_metadata_hashes),
        "recordings_byte_hash": catalog.recordings_byte_hash,
        "root": catalog.root,
        "schema_version": catalog.schema_version,
        "semantic_catalog_hash": catalog.semantic_catalog_hash,
        "split_rules": dict(catalog.split_rules),
        "split_seed": catalog.split_seed,
        "stats": dict(catalog.stats),
    }


def write_catalog(path: str | Path, catalog: BackgroundCatalog) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(catalog_to_mapping(catalog), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def read_catalog(path: str | Path) -> BackgroundCatalog:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"catalog must be a mapping: {path}")
    return BackgroundCatalog(
        dataset_id=str(payload.get("dataset_id", "")),
        schema_version=int(
            payload.get("schema_version", BACKGROUND_MANIFEST_SCHEMA_VERSION)
        ),
        root=str(payload.get("root", "")),
        filter_policy_version=str(payload.get("filter_policy_version", "")),
        filter_policy_hash=str(payload.get("filter_policy_hash", "")),
        raw_metadata_hashes=dict(payload.get("raw_metadata_hashes") or {}),
        split_seed=payload.get("split_seed"),
        split_rules=dict(payload.get("split_rules") or {}),
        stats=dict(payload.get("stats") or {}),
        recordings_byte_hash=str(payload.get("recordings_byte_hash", "")),
        semantic_catalog_hash=str(payload.get("semantic_catalog_hash", "")),
    )


def assert_catalog_source_id(catalog: BackgroundCatalog, source_id: str) -> None:
    if catalog.dataset_id != source_id:
        raise ValueError(
            f"catalog dataset_id {catalog.dataset_id!r} does not match source id "
            f"{source_id!r}"
        )


def resolve_relative_path(relative_path: str, *, root: str | Path) -> Path:
    if not isinstance(relative_path, str) or not relative_path.strip():
        raise ValueError(f"relative_path {relative_path!r} escapes catalog root")
    root_path = Path(root).resolve()
    candidate = Path(relative_path)
    if candidate.is_absolute():
        resolved = candidate.resolve()
    else:
        if ".." in candidate.parts:
            raise ValueError(f"relative_path {relative_path!r} escapes catalog root")
        resolved = (root_path / candidate).resolve()
    try:
        resolved.relative_to(root_path)
    except ValueError as exc:
        raise ValueError(
            f"relative_path {relative_path!r} escapes catalog root"
        ) from exc
    return resolved


def _record_content(record: BackgroundRecord) -> tuple[Any, ...]:
    return (
        record.dataset_id,
        record.relative_path,
        record.group_id,
        record.origin_ids,
        record.categories,
        record.background_eligible,
        record.eligibility_basis,
        record.duration_seconds,
        record.sample_rate,
        record.channels,
        record.audio_sha256,
        record.license_id,
    )


def _effective_group_id(record: BackgroundRecord) -> str:
    group_id = str(record.group_id or "").strip()
    if group_id:
        return group_id
    return f"recording:{record.recording_id}"


def _format_dataset_ids(records: Sequence[BackgroundRecord]) -> str:
    dataset_ids = sorted(
        {str(record.dataset_id) for record in records if str(record.dataset_id).strip()}
    )
    if not dataset_ids:
        return ""
    if len(dataset_ids) == 1:
        return f"dataset_id {dataset_ids[0]!r} "
    return f"dataset_id {dataset_ids!r} "


def audit_split_isolation(
    records: Sequence[BackgroundRecord],
) -> IsolationAuditReport:
    by_recording_id: dict[str, list[BackgroundRecord]] = {}
    for record in records:
        by_recording_id.setdefault(record.recording_id, []).append(record)
    for recording_id, group in by_recording_id.items():
        prefix = _format_dataset_ids(group)
        contents = {_record_content(item) for item in group}
        if len(contents) > 1:
            raise ValueError(
                f"{prefix}recording_id {recording_id!r} has different content"
            )
        splits = {item.split for item in group}
        if len(group) > 1 and len(splits) == 1:
            raise ValueError(
                f"{prefix}recording_id {recording_id!r} is not unique within split "
                f"{next(iter(splits))!r}"
            )
        if len(splits) > 1:
            raise ValueError(
                f"{prefix}recording_id {recording_id!r} appears in multiple splits; "
                f"expected one split, got {sorted(splits)}"
            )

    group_records: dict[str, list[BackgroundRecord]] = {}
    origin_records: dict[str, list[BackgroundRecord]] = {}
    sha_records: dict[str, list[BackgroundRecord]] = {}
    for record in records:
        group_records.setdefault(_effective_group_id(record), []).append(record)
        for origin_id in record.origin_ids:
            origin_records.setdefault(origin_id, []).append(record)
        if record.audio_sha256:
            sha_records.setdefault(record.audio_sha256, []).append(record)

    for group_id, group in group_records.items():
        splits = {item.split for item in group}
        if len(splits) > 1:
            raise ValueError(
                f"{_format_dataset_ids(group)}group_id {group_id!r} appears in "
                f"multiple splits; expected one split, got {sorted(splits)}"
            )
    for origin_id, group in origin_records.items():
        splits = {item.split for item in group}
        if len(splits) > 1:
            raise ValueError(
                f"{_format_dataset_ids(group)}origin_id {origin_id!r} appears in "
                f"multiple splits; expected one split, got {sorted(splits)}"
            )
    for digest, group in sha_records.items():
        splits = {item.split for item in group}
        if len(splits) > 1:
            raise ValueError(
                f"{_format_dataset_ids(group)}identical audio bytes appear in "
                f"multiple splits (audio_sha256={digest!r}); expected one split, "
                f"got {sorted(splits)}"
            )

    provenance_complete = all(record.provenance_complete for record in records)
    return IsolationAuditReport(
        provenance_complete=provenance_complete,
        capability_limit="" if provenance_complete else PROVENANCE_CAPABILITY_LIMIT,
    )


def eligible_train_records(
    records: Sequence[BackgroundRecord],
) -> tuple[BackgroundRecord, ...]:
    return tuple(
        record
        for record in records
        if record.split == "train" and record.background_eligible is True
    )


def require_eligible_train_records(
    records: Sequence[BackgroundRecord], *, source_id: str
) -> tuple[BackgroundRecord, ...]:
    eligible = eligible_train_records(records)
    if not eligible:
        raise ValueError(f"no eligible train recordings for source {source_id}")
    return eligible
