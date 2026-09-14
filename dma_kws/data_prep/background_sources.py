"""Facade for discovering, filtering, splitting, auditing, and publishing catalogs."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

from dma_kws.data_prep.background_adapters import (
    SourceImportConfig,
    get_source_adapter,
)
from dma_kws.data_prep.background_manifest import (
    CATALOG_JSON_NAME,
    RECORDINGS_JSONL_NAME,
    BackgroundCatalog,
    BackgroundRecord,
    IsolationAuditReport,
    assert_catalog_source_id,
    audit_split_isolation,
    build_catalog,
    canonical_json_dumps,
    require_eligible_train_records,
    resolve_relative_path,
    sha256_file,
    write_catalog,
    write_recordings_jsonl,
)


FILTER_POLICY_VERSION = "curated_allowlist_v1"
SPLIT_RATIOS = (0.8, 0.1, 0.1)
ALLOWED_SPLIT_POLICIES = frozenset({"preserve", "group_random"})


@dataclass(frozen=True)
class PrepareBackgroundConfig:
    output_dir: str | Path
    seed: int = 2025
    sources: tuple[SourceImportConfig, ...] = ()


@dataclass(frozen=True)
class PrepareResult:
    output_dir: Path
    catalogs: Mapping[str, BackgroundCatalog]
    audit: IsolationAuditReport


def prepare_config_from_mapping(
    payload: Mapping[str, Any], *, base_dir: str | Path | None = None
) -> PrepareBackgroundConfig:
    if not isinstance(payload, Mapping):
        raise ValueError("prepare config must be a mapping")
    base = Path(base_dir).expanduser() if base_dir is not None else Path(".")
    sources = tuple(
        source_import_config_from_mapping(item, base_dir=base)
        for item in payload.get("sources") or ()
    )
    return PrepareBackgroundConfig(
        output_dir=_resolve_config_path(payload.get("output_dir", ""), base),
        seed=_require_seed(payload.get("seed", 2025)),
        sources=sources,
    )


def source_import_config_from_mapping(
    payload: Mapping[str, Any], *, base_dir: str | Path | None = None
) -> SourceImportConfig:
    if not isinstance(payload, Mapping):
        raise ValueError("source import config must be a mapping")
    base = Path(base_dir).expanduser() if base_dir is not None else Path(".")
    return SourceImportConfig(
        adapter=str(payload.get("adapter", "")).strip(),
        id=str(payload.get("id", "")).strip(),
        root=_resolve_config_path(payload.get("root", ""), base),
        split_dir=_resolve_config_path(payload.get("split_dir", ""), base),
        metadata=_resolve_config_path(payload.get("metadata", ""), base),
        eligible_ids_file=_resolve_config_path(
            payload.get("eligible_ids_file", ""), base
        ),
        split_policy=str(payload.get("split_policy", "group_random")).strip()
        or "group_random",
        category_allow=_as_str_tuple(payload.get("category_allow")),
        category_exclude=_as_str_tuple(payload.get("category_exclude")),
    )


def prepare_background_sources(config: PrepareBackgroundConfig) -> PrepareResult:
    _validate_prepare_config(config)
    destination = Path(config.output_dir).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(
            f"Refusing to overwrite an existing background source directory: {destination}"
        )

    prepared: list[tuple[SourceImportConfig, list[BackgroundRecord], dict[str, Any]]] = []
    for source in config.sources:
        adapter = get_source_adapter(source.adapter)
        discovered = list(adapter.discover(source))
        records = _normalize_records(discovered, source)
        records, dropped = _apply_category_prefilter(records, source)
        eligible_ids, eligible_digest = _load_eligible_ids(source.eligible_ids_file)
        records = _apply_eligibility(records, eligible_ids)
        records = _assign_splits(records, source=source, seed=config.seed)
        require_eligible_train_records(records, source_id=source.id)
        audit_split_isolation(records)
        prepared.append(
            (
                source,
                records,
                {
                    "category_prefilter_dropped": dropped,
                    "eligible_ids_sha256": eligible_digest,
                    "allowlisted_eligible": sum(
                        record.background_eligible is True for record in records
                    ),
                    "not_allowlisted": sum(
                        record.background_eligible is not True for record in records
                    ),
                    "per_recording_groups": _uses_per_recording_groups(records, source),
                },
            )
        )

    all_records = [record for _source, records, _meta in prepared for record in records]
    joint_audit = audit_split_isolation(all_records)

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(dir=destination.parent, prefix=f".{destination.name}.")
    )
    catalogs: dict[str, BackgroundCatalog] = {}
    try:
        source_reports: dict[str, Any] = {}
        for source, records, filter_info in prepared:
            source_dir = staging / source.id
            source_dir.mkdir(parents=True, exist_ok=True)
            recordings_path = source_dir / RECORDINGS_JSONL_NAME
            ordered = sorted(records, key=lambda item: item.recording_id)
            write_recordings_jsonl(recordings_path, ordered)
            _fsync_file(recordings_path)
            raw_metadata_hashes = _raw_metadata_hashes(source, filter_info)
            catalog = build_catalog(
                dataset_id=source.id,
                root=str(Path(source.root).expanduser().resolve()),
                records=ordered,
                recordings_path=recordings_path,
                filter_policy_version=FILTER_POLICY_VERSION,
                filter_policy_hash=_filter_policy_hash(source, filter_info),
                split_seed=config.seed,
                split_rules=_split_rules(source),
                raw_metadata_hashes=raw_metadata_hashes,
            )
            assert_catalog_source_id(catalog, source.id)
            catalog_path = source_dir / CATALOG_JSON_NAME
            write_catalog(catalog_path, catalog)
            _fsync_file(catalog_path)
            for split in ("train", "val", "test"):
                list_path = source_dir / f"{split}.list"
                _write_split_list(list_path, ordered, split)
                _fsync_file(list_path)
            catalogs[source.id] = catalog
            source_reports[source.id] = {
                "stats": dict(catalog.stats),
                "filter": {
                    "category_prefilter_dropped": filter_info["category_prefilter_dropped"],
                    "allowlisted_eligible": filter_info["allowlisted_eligible"],
                    "not_allowlisted": filter_info["not_allowlisted"],
                },
                "grouping": (
                    "per_recording"
                    if filter_info["per_recording_groups"]
                    else "native"
                ),
                "isolation_limits": (
                    joint_audit.capability_limit
                    if filter_info["per_recording_groups"] or not joint_audit.provenance_complete
                    else ""
                ),
            }
        audit_payload = {
            "capability_limit": joint_audit.capability_limit,
            "provenance_complete": joint_audit.provenance_complete,
            "sources": source_reports,
        }
        audit_path = staging / "audit.json"
        audit_path.write_text(
            json.dumps(audit_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _fsync_file(audit_path)
        _fsync_tree(staging)
        os.replace(staging, destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)

    return PrepareResult(output_dir=destination, catalogs=catalogs, audit=joint_audit)


def _validate_prepare_config(config: PrepareBackgroundConfig) -> None:
    _require_seed(config.seed)
    if not config.sources:
        raise ValueError("prepare config sources must not be empty")
    seen: set[str] = set()
    for source in config.sources:
        if not source.adapter:
            raise ValueError("source adapter must not be empty")
        get_source_adapter(source.adapter)
        if not source.id:
            raise ValueError("source id must not be empty")
        if source.id in seen:
            raise ValueError(f"duplicate source id {source.id!r}")
        seen.add(source.id)
        if source.split_policy not in ALLOWED_SPLIT_POLICIES:
            raise ValueError(
                f"source id={source.id!r} field=split_policy: unknown split_policy "
                f"{source.split_policy!r}; expected one of {sorted(ALLOWED_SPLIT_POLICIES)}"
            )
        if source.split_policy == "preserve" and source.adapter == "musan" and not source.split_dir:
            raise ValueError(
                f"source id={source.id!r} field=split_dir: preserve requires an old "
                "MUSAN split_dir; expected a train_background.list directory, got ''"
            )


def _require_seed(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"seed must be a non-negative integer, got {value!r}")
    return value


def _as_str_tuple(value: object) -> tuple[str, ...]:
    if value is None or value == "":
        return ()
    if isinstance(value, str):
        items = tuple(part.strip() for part in value.split(",") if part.strip())
        return items
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return tuple(str(item).strip() for item in value if str(item).strip())
    raise ValueError(f"expected a string list, got {value!r}")


def _resolve_config_path(value: object, base: Path) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = base / path
    return str(path)


def _normalize_records(
    records: Sequence[BackgroundRecord], source: SourceImportConfig
) -> list[BackgroundRecord]:
    normalized: list[BackgroundRecord] = []
    root = Path(source.root).expanduser().resolve()
    for record in records:
        if record.dataset_id != source.id:
            raise ValueError(
                f"adapter {source.adapter!r} recording {record.recording_id!r} "
                f"dataset_id {record.dataset_id!r} does not match source id {source.id!r}"
            )
        if not record.audio_sha256 or len(record.audio_sha256) != 64:
            raise ValueError(
                f"empty audio_sha256 from adapter {source.adapter} recording "
                f"{record.recording_id}"
            )
        resolve_relative_path(record.relative_path, root=root)
        group_id = record.group_id.strip() or f"{source.id}:{record.recording_id}"
        if group_id != record.group_id:
            record = replace(record, group_id=group_id)
        normalized.append(record)
    return normalized


def _apply_category_prefilter(
    records: Sequence[BackgroundRecord], source: SourceImportConfig
) -> tuple[list[BackgroundRecord], int]:
    allow = set(source.category_allow)
    exclude = set(source.category_exclude)
    if not allow and not exclude:
        return list(records), 0
    kept: list[BackgroundRecord] = []
    dropped = 0
    for record in records:
        categories = set(record.categories)
        if exclude and categories & exclude:
            dropped += 1
            continue
        if allow and not (categories & allow):
            dropped += 1
            continue
        kept.append(record)
    return kept, dropped


def _load_eligible_ids(path: str) -> tuple[set[str], str]:
    if not path:
        return set(), ""
    eligible_path = Path(path).expanduser()
    if not eligible_path.is_file():
        raise FileNotFoundError(f"eligible_ids_file not found: {eligible_path}")
    digest = sha256_file(eligible_path)
    ids: set[str] = set()
    for line in eligible_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        ids.add(stripped)
    return ids, digest


def _eligibility_keys(record: BackgroundRecord) -> set[str]:
    prefix = f"{record.dataset_id}:"
    native_id = (
        record.recording_id[len(prefix) :]
        if record.recording_id.startswith(prefix)
        else record.recording_id
    )
    return {
        record.recording_id,
        native_id,
        record.relative_path,
    }


def _apply_eligibility(
    records: Sequence[BackgroundRecord], eligible_ids: set[str]
) -> list[BackgroundRecord]:
    updated: list[BackgroundRecord] = []
    for record in records:
        matched = bool(eligible_ids) and bool(_eligibility_keys(record) & eligible_ids)
        updated.append(
            replace(
                record,
                background_eligible=matched,
                eligibility_basis=FILTER_POLICY_VERSION if matched else "",
            )
        )
    return updated


def _group_id(record: BackgroundRecord) -> str:
    group_id = str(record.group_id or "").strip()
    if group_id:
        return group_id
    return f"recording:{record.recording_id}"


def _stable_digest(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}\0{value}".encode("utf-8")).hexdigest()


def _assign_splits(
    records: Sequence[BackgroundRecord],
    *,
    source: SourceImportConfig,
    seed: int,
) -> list[BackgroundRecord]:
    if source.split_policy == "preserve":
        splits = {record.split for record in records}
        if (
            source.adapter == "musan"
            and "train" in splits
            and "val" not in splits
        ):
            return _carve_val_from_train(records, seed=seed)
        return list(records)
    if source.split_policy == "group_random":
        return _group_random_split(records, seed=seed, source_id=source.id)
    raise ValueError(
        f"source id={source.id!r} field=split_policy: unknown split_policy "
        f"{source.split_policy!r}"
    )


def _carve_val_from_train(
    records: Sequence[BackgroundRecord], *, seed: int
) -> list[BackgroundRecord]:
    grouped: dict[str, list[BackgroundRecord]] = defaultdict(list)
    for record in records:
        grouped[_group_id(record)].append(record)
    train_groups = [
        group_id
        for group_id, items in grouped.items()
        if {item.split for item in items} == {"train"}
    ]
    if len(train_groups) < 2:
        return list(records)
    ordered = sorted(
        train_groups, key=lambda group_id: (_stable_digest(seed, group_id), group_id)
    )
    n_val = max(1, int(round(len(ordered) * (SPLIT_RATIOS[1] / (1.0 - SPLIT_RATIOS[2])))))
    if n_val >= len(ordered):
        n_val = 1
    val_groups = set(ordered[-n_val:])
    return [
        replace(record, split="val")
        if _group_id(record) in val_groups
        else record
        for record in records
    ]


def _ratio_split_groups(group_ids: Sequence[str], *, seed: int) -> dict[str, str]:
    ordered = sorted(
        group_ids, key=lambda group_id: (_stable_digest(seed, group_id), group_id)
    )
    count = len(ordered)
    if count < 3:
        raise ValueError(
            f"too few independent eligible groups to form train/val/test without "
            f"splitting groups; got {count}"
        )
    n_test = max(1, int(round(count * SPLIT_RATIOS[2])))
    n_val = max(1, int(round(count * SPLIT_RATIOS[1])))
    if n_test + n_val >= count:
        n_test = 1
        n_val = 1
    n_train = count - n_val - n_test
    if n_train < 1:
        raise ValueError(
            f"too few independent eligible groups to form train/val/test without "
            f"splitting groups; got {count}"
        )
    mapping: dict[str, str] = {}
    for group_id in ordered[:n_train]:
        mapping[group_id] = "train"
    for group_id in ordered[n_train : n_train + n_val]:
        mapping[group_id] = "val"
    for group_id in ordered[n_train + n_val :]:
        mapping[group_id] = "test"
    return mapping


def _bucket_split(seed: int, group_id: str) -> str:
    digest = _stable_digest(seed, group_id)
    fraction = int(digest[:8], 16) / 0xFFFFFFFF
    if fraction < SPLIT_RATIOS[0]:
        return "train"
    if fraction < SPLIT_RATIOS[0] + SPLIT_RATIOS[1]:
        return "val"
    return "test"


def _group_random_split(
    records: Sequence[BackgroundRecord], *, seed: int, source_id: str
) -> list[BackgroundRecord]:
    grouped: dict[str, list[BackgroundRecord]] = defaultdict(list)
    for record in records:
        grouped[_group_id(record)].append(record)
    locked_test = {
        group_id
        for group_id, items in grouped.items()
        if any(item.split == "test" for item in items)
    }
    eligible_groups = [
        group_id
        for group_id, items in grouped.items()
        if any(item.background_eligible is True for item in items)
    ]
    eligible_unlocked = [
        group_id for group_id in eligible_groups if group_id not in locked_test
    ]
    if not locked_test and len(eligible_groups) < 3:
        raise ValueError(
            f"too few independent eligible groups for source {source_id} to form "
            f"train/val/test without splitting groups; got {len(eligible_groups)}"
        )
    if locked_test and len(eligible_unlocked) < 2:
        raise ValueError(
            f"too few independent eligible groups for source {source_id} to form "
            f"train/val/test without splitting groups; got {len(eligible_unlocked)}"
        )

    if not locked_test:
        assignment = _ratio_split_groups(eligible_unlocked, seed=seed)
    elif len(eligible_unlocked) >= 3:
        assignment = _ratio_split_groups(eligible_unlocked, seed=seed)
    else:
        ordered = sorted(
            eligible_unlocked,
            key=lambda group_id: (_stable_digest(seed, group_id), group_id),
        )
        assignment = {ordered[0]: "train", ordered[1]: "val"}

    updated: list[BackgroundRecord] = []
    for record in records:
        group_id = _group_id(record)
        if group_id in locked_test:
            split = "test"
        elif group_id in assignment:
            split = assignment[group_id]
        else:
            split = _bucket_split(seed, group_id)
        updated.append(replace(record, split=split))
    return updated


def _uses_per_recording_groups(
    records: Sequence[BackgroundRecord], source: SourceImportConfig
) -> bool:
    if source.adapter != "dns":
        return False
    return all(
        record.group_id in {f"{source.id}:{record.relative_path}", record.recording_id}
        for record in records
    )


def _filter_policy_hash(
    source: SourceImportConfig, filter_info: Mapping[str, Any]
) -> str:
    payload = {
        "category_allow": list(source.category_allow),
        "category_exclude": list(source.category_exclude),
        "eligible_ids_sha256": filter_info.get("eligible_ids_sha256", ""),
        "version": FILTER_POLICY_VERSION,
    }
    return hashlib.sha256(canonical_json_dumps(payload).encode("utf-8")).hexdigest()


def _split_rules(source: SourceImportConfig) -> dict[str, Any]:
    return {
        "official_eval_mapped_to": "test",
        "policy": source.split_policy,
        "ratios": list(SPLIT_RATIOS),
        "unit": "group",
    }


def _raw_metadata_hashes(
    source: SourceImportConfig, filter_info: Mapping[str, Any]
) -> dict[str, str]:
    hashes: dict[str, str] = {}
    eligible_digest = str(filter_info.get("eligible_ids_sha256") or "")
    if eligible_digest:
        hashes["eligible_ids_file"] = eligible_digest
    candidates: list[tuple[str, Path]] = []
    if source.split_dir:
        split_json = Path(source.split_dir).expanduser() / "split.json"
        candidates.append(("split.json", split_json))
    if source.metadata:
        meta = Path(source.metadata).expanduser()
        if meta.is_file():
            candidates.append((meta.name, meta))
        elif meta.is_dir():
            for name in ("dev.csv", "eval.csv", "dev_clips.csv", "eval_clips.csv"):
                candidates.append((name, meta / name))
    root = Path(source.root).expanduser()
    for name, rel in (
        ("dev.csv", Path("FSD50K.ground_truth") / "dev.csv"),
        ("eval.csv", Path("FSD50K.ground_truth") / "eval.csv"),
        ("dev_clips.csv", Path("FSD50K.metadata") / "dev_clips.csv"),
        ("eval_clips.csv", Path("FSD50K.metadata") / "eval_clips.csv"),
    ):
        candidates.append((name, root / rel))
    for key, path in candidates:
        if path.is_file() and key not in hashes:
            hashes[key] = sha256_file(path)
    return hashes


def _write_split_list(
    path: Path, records: Sequence[BackgroundRecord], split: str
) -> None:
    lines = sorted(
        record.audio_path for record in records if record.split == split
    )
    path.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")


def _fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _fsync_dir(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_tree(root: Path) -> None:
    for dirpath, dirnames, filenames in os.walk(root):
        current = Path(dirpath)
        for name in filenames:
            _fsync_file(current / name)
        for name in dirnames:
            _fsync_dir(current / name)
        _fsync_dir(current)


__all__ = [
    "FILTER_POLICY_VERSION",
    "PrepareBackgroundConfig",
    "PrepareResult",
    "prepare_background_sources",
    "prepare_config_from_mapping",
    "source_import_config_from_mapping",
]
