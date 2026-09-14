"""Background data signatures, resume identity, and metric-alias policy."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
from typing import Any

from dma_kws.configs.schema import (
    active_background_sources,
    background_source_batch_index,
)
from dma_kws.data_prep.background_manifest import (
    CATALOG_JSON_NAME,
    canonical_json_dumps,
    read_catalog,
    read_recordings_jsonl,
    semantic_catalog_hash,
    semantic_record_payload,
    train_catalog_hash,
)

BACKGROUND_DATA_SIGNATURE_VERSION = 2
SAMPLING_POLICY_VERSION = 1
CROP_POLICY_VERSION = 1
LEGACY_BACKGROUND_KEYS = (
    "enabled",
    "probability",
    "audio_list_path",
    "duration_seconds_min",
    "duration_seconds_max",
    "mode",
    "cache_manifest",
    "max_open_shards",
)


def is_multisource_background(config: Mapping[str, Any] | None) -> bool:
    """Return True when ``sources`` is a non-empty list."""
    sources = (config or {}).get("sources") or []
    return bool(sources)


def audit_background_sources_at_train_start(
    background_cfg: Mapping[str, Any] | None,
) -> None:
    """Merge-audit active catalogs when multi-source background training is on.

    ``enabled=false`` and empty ``sources`` (legacy) skip catalog I/O. Overlaps
    raise; they are not dropped.
    """
    payload = dict(background_cfg or {})
    if not bool(payload.get("enabled", False)):
        return
    if not is_multisource_background(payload):
        return
    from dma_kws.stage2.joint_manifest import validate_background_sources_identity

    validate_background_sources_identity(payload)


def musan_metric_alias_allowed(source_ids: Sequence[str]) -> bool:
    """Legacy ``musan`` aliases apply only to empty/legacy or single-MUSAN runs."""
    names = [str(item) for item in source_ids]
    return (not names) or names == ["musan"]


def legacy_background_payload(background: Mapping[str, Any] | None) -> dict[str, Any]:
    """Old joint-signature background mapping: no ``sources``/``validation`` keys."""
    payload = dict(background or {})
    return {key: payload[key] for key in LEGACY_BACKGROUND_KEYS if key in payload}


def joint_background_validation_disabled_message(
    background_cfg: Mapping[str, Any] | None,
) -> str:
    """Hint for joint runs that skipped held-out background validation."""
    if is_multisource_background(background_cfg):
        return (
            "Joint background validation is disabled: set "
            "stage2.background_negative.validation.enabled=true for per-source "
            "held-out crops"
        )
    return (
        "Joint MUSAN validation is disabled: set "
        "adapt.joint.background_eval_list to a held-out list"
    )


def assert_background_eval_list_compatible(
    background_cfg: Mapping[str, Any] | None,
    joint_cfg: Mapping[str, Any] | None,
) -> None:
    """New-mode non-empty ``adapt.joint.background_eval_list`` is a conflict."""
    if not is_multisource_background(background_cfg):
        return
    eval_list = str((joint_cfg or {}).get("background_eval_list") or "").strip()
    if eval_list:
        raise ValueError(
            "adapt.joint.background_eval_list must be empty when "
            "stage2.background_negative.sources is non-empty; "
            f"expected '', got {eval_list!r}. Use "
            "stage2.background_negative.validation instead."
        )


def _split_identity(records: Sequence[Any], split: str) -> dict[str, Any]:
    selected = sorted(
        (
            record
            for record in records
            if record.split == split and record.background_eligible is True
        ),
        key=lambda item: item.recording_id,
    )
    payload = [semantic_record_payload(record) for record in selected]
    return {
        "count": len(selected),
        "hash": hashlib.sha256(canonical_json_dumps(payload).encode("utf-8")).hexdigest(),
    }


def _optional_cache_id(path: str) -> str | None:
    manifest = Path(str(path or "")).expanduser()
    if not str(path or "").strip() or not manifest.is_file():
        return None
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if isinstance(payload, dict) and payload.get("cache_id"):
        return str(payload["cache_id"])
    return None


def _canonical_fbank(fbank: Mapping[str, Any] | None) -> dict[str, Any]:
    payload = dict(fbank or {})
    return {key: payload[key] for key in sorted(payload)}


def background_data_signature_payload(
    background_cfg: Mapping[str, Any],
    *,
    fbank: Mapping[str, Any] | None = None,
    seed: int = 2025,
) -> dict[str, Any]:
    """Version-2 semantic payload. Locators and mount points are omitted."""
    payload = dict(background_cfg or {})
    active = active_background_sources(payload.get("sources") or [])
    index = background_source_batch_index([source.id for source in active])
    mode = str(payload.get("mode") or "online").strip() or "online"
    validation = dict(payload.get("validation") or {})
    sources: list[dict[str, Any]] = []
    for source in active:
        records = read_recordings_jsonl(source.manifest)
        catalog_path = Path(str(source.manifest)).expanduser().parent / CATALOG_JSON_NAME
        if catalog_path.is_file():
            catalog = read_catalog(catalog_path)
            catalog_hash = catalog.semantic_catalog_hash
            filter_policy_version = catalog.filter_policy_version
            filter_policy_hash = catalog.filter_policy_hash
        else:
            catalog_hash = semantic_catalog_hash(records)
            filter_policy_version = ""
            filter_policy_hash = ""
        entry = {
            "id": source.id,
            "normalized_weight": float(source.weight)
            / sum(float(item.weight) for item in active),
            "semantic_catalog_hash": catalog_hash,
            "filter_policy_version": filter_policy_version,
            "filter_policy_hash": filter_policy_hash,
            "train_identity": {
                "count": len(
                    [
                        record
                        for record in records
                        if record.split == "train" and record.background_eligible is True
                    ]
                ),
                "hash": train_catalog_hash(records),
            },
            "val_identity": _split_identity(records, "val"),
            "test_identity": _split_identity(records, "test"),
            "cache_id": _optional_cache_id(source.cache_manifest)
            if mode == "fbank_cache"
            else None,
        }
        sources.append(entry)
    return {
        "background_data_signature_version": BACKGROUND_DATA_SIGNATURE_VERSION,
        "mode": mode,
        "probability": float(payload.get("probability", 0.25)),
        "duration_seconds_min": float(payload.get("duration_seconds_min", 1.0)),
        "duration_seconds_max": float(payload.get("duration_seconds_max", 3.0)),
        "crop_policy_version": int(CROP_POLICY_VERSION),
        "sampling_policy_version": SAMPLING_POLICY_VERSION,
        "fbank": _canonical_fbank(fbank),
        "seed": int(seed),
        "source_id_index": dict(index),
        "validation": {
            "enabled": bool(validation.get("enabled", False)),
            "samples_per_source": int(validation.get("samples_per_source", 256)),
            "seed": int(validation.get("seed", 2025)),
        },
        "sources": sources,
    }


def background_data_signature_hash(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json_dumps(dict(payload)).encode("utf-8")).hexdigest()


def build_background_data_signature(config: Mapping[str, Any]) -> str | None:
    """Return the v2 hash for multi-source runs; ``None`` keeps legacy resume."""
    background = (config.get("stage2") or {}).get("background_negative") or {}
    if not is_multisource_background(background):
        return None
    seed = int((config.get("training") or {}).get("seed", 2025))
    return background_data_signature_hash(
        background_data_signature_payload(
            background,
            fbank=config.get("fbank") or {},
            seed=seed,
        )
    )


def assert_background_resume_identity(
    checkpoint: Mapping[str, Any] | None,
    *,
    current_signature: str | None,
    current_config: Mapping[str, Any] | None = None,
) -> None:
    """Strict resume: source/weight/policy/content changes are a new run."""
    checkpoint = checkpoint or {}
    background = ((current_config or {}).get("stage2") or {}).get(
        "background_negative"
    ) or {}
    current_is_multi = bool(current_signature) or is_multisource_background(background)
    saved_version = checkpoint.get("background_data_signature_version")
    saved_hash = checkpoint.get("background_data_signature")
    if current_is_multi:
        if saved_version != BACKGROUND_DATA_SIGNATURE_VERSION or not saved_hash:
            raise SystemExit(
                "Cannot resume a legacy background checkpoint as a multi-source "
                "run. Switching to multi-source is a new run; use "
                "run.init_checkpoint for weight initialization."
            )
        if current_signature is None:
            current_signature = build_background_data_signature(current_config or {})
        if saved_hash != current_signature:
            raise SystemExit(
                "Cannot resume: background data signature mismatch; "
                "field=background_data_signature; "
                f"expected {current_signature!r}, got {saved_hash!r}. "
                "Source, weight, policy, or content changed. Start a new run "
                "from complete model weights with run.init_checkpoint."
            )
        return
    if saved_version == BACKGROUND_DATA_SIGNATURE_VERSION and saved_hash:
        raise SystemExit(
            "Cannot resume a multi-source background checkpoint as a legacy "
            "run. Start a new run from complete model weights with "
            "run.init_checkpoint."
        )


def background_source_ids_from_sampler(sampler: Any) -> list[str]:
    if sampler is None:
        return []
    record = getattr(sampler, "run_record_fields", None)
    if not callable(record):
        return []
    fields = record() or {}
    ids = fields.get("background_source_ids") or []
    return [str(item) for item in ids]
