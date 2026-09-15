from __future__ import annotations

import hashlib
from typing import Any

import pytest

from dma_kws.data_prep.background_manifest import (
    BACKGROUND_MANIFEST_SCHEMA_VERSION,
    CATALOG_JSON_NAME,
    RECORDINGS_JSONL_NAME,
    assert_catalog_source_id,
    audit_split_isolation,
    background_record_from_mapping,
    build_catalog,
    eligible_train_records,
    read_catalog,
    read_recordings_jsonl,
    require_eligible_train_records,
    resolve_relative_path,
    semantic_catalog_hash,
    sha256_file,
    write_catalog,
    write_recordings_jsonl,
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _payload(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": BACKGROUND_MANIFEST_SCHEMA_VERSION,
        "dataset_id": "musan",
        "recording_id": "musan:noise-0001",
        "relative_path": "noise/free-sound/noise-0001.wav",
        "audio_path": "/data/musan/noise/free-sound/noise-0001.wav",
        "group_id": "musan:source:free-sound",
        "origin_ids": ["musan:noise-0001"],
        "split": "train",
        "categories": ["noise"],
        "background_eligible": True,
        "eligibility_basis": "curated_allowlist_v1",
        "duration_seconds": 1.5,
        "sample_rate": 16000,
        "channels": 1,
        "audio_sha256": _sha("noise-0001"),
        "license_id": "cc-by-4.0",
    }
    payload.update(overrides)
    return payload


def _record(**overrides: Any):
    return background_record_from_mapping(_payload(**overrides))


def test_same_recording_id_different_content_rejected():
    records = [
        _record(audio_sha256=_sha("a"), relative_path="noise/a.wav"),
        _record(audio_sha256=_sha("b"), relative_path="noise/b.wav"),
    ]
    with pytest.raises(ValueError, match=r"dataset_id 'musan'.+recording_id.+different content"):
        audit_split_isolation(records)


def test_same_group_id_across_splits_rejected():
    records = [
        _record(
            recording_id="musan:r1",
            split="train",
            group_id="uploader:1",
            origin_ids=["o1"],
            audio_sha256=_sha("r1"),
            relative_path="train/r1.wav",
        ),
        _record(
            recording_id="musan:r2",
            split="val",
            group_id="uploader:1",
            origin_ids=["o2"],
            audio_sha256=_sha("r2"),
            relative_path="val/r2.wav",
        ),
    ]
    with pytest.raises(ValueError, match=r"dataset_id 'musan'.+group_id"):
        audit_split_isolation(records)


def test_shared_origin_id_across_splits_rejected():
    records = [
        _record(
            recording_id="fsd50k:1",
            dataset_id="fsd50k",
            split="train",
            group_id="fsd50k:uploader:1",
            origin_ids=["freesound:12345"],
            audio_sha256=_sha("1"),
            relative_path="dev/1.wav",
        ),
        _record(
            recording_id="fsd50k:2",
            dataset_id="fsd50k",
            split="test",
            group_id="fsd50k:uploader:2",
            origin_ids=["freesound:999", "freesound:12345"],
            audio_sha256=_sha("2"),
            relative_path="eval/2.wav",
        ),
    ]
    with pytest.raises(ValueError, match=r"dataset_id 'fsd50k'.+origin_id"):
        audit_split_isolation(records)


def test_identical_audio_bytes_across_splits_rejected(tmp_path):
    payload = b"tiny-wav-bytes"
    train_audio = tmp_path / "train.wav"
    val_audio = tmp_path / "val.wav"
    train_audio.write_bytes(payload)
    val_audio.write_bytes(payload)
    digest = sha256_file(train_audio)
    assert digest == hashlib.sha256(payload).hexdigest()
    assert digest == sha256_file(val_audio)

    records = [
        _record(
            recording_id="musan:train",
            split="train",
            group_id="g-train",
            origin_ids=["origin-train"],
            audio_path=str(train_audio),
            relative_path="train.wav",
            audio_sha256=digest,
        ),
        _record(
            recording_id="musan:val",
            split="val",
            group_id="g-val",
            origin_ids=["origin-val"],
            audio_path=str(val_audio),
            relative_path="val.wav",
            audio_sha256=digest,
        ),
    ]
    with pytest.raises(ValueError, match=r"dataset_id 'musan'.+identical audio bytes"):
        audit_split_isolation(records)


def test_same_identity_with_relocated_mount_is_still_detected(tmp_path):
    audio_bytes = b"shared-identity"
    digest = hashlib.sha256(audio_bytes).hexdigest()
    left = [
        _record(
            audio_path="/mnt/a/musan/noise/free-sound/noise-0001.wav",
            audio_sha256=digest,
        ),
        _record(
            recording_id="musan:noise-0002",
            split="val",
            relative_path="noise/free-sound/noise-0002.wav",
            audio_path="/mnt/a/musan/noise/free-sound/noise-0002.wav",
            group_id="musan:source:free-sound",
            origin_ids=["musan:noise-0002"],
            audio_sha256=_sha("other"),
        ),
    ]
    right = [
        _record(
            audio_path="/mnt/b/musan/noise/free-sound/noise-0001.wav",
            audio_sha256=digest,
        ),
        _record(
            recording_id="musan:noise-0002",
            split="val",
            relative_path="noise/free-sound/noise-0002.wav",
            audio_path="/mnt/b/musan/noise/free-sound/noise-0002.wav",
            group_id="musan:source:free-sound",
            origin_ids=["musan:noise-0002"],
            audio_sha256=_sha("other"),
        ),
    ]
    with pytest.raises(ValueError, match=r"dataset_id 'musan'.+group_id"):
        audit_split_isolation(left)
    with pytest.raises(ValueError, match=r"dataset_id 'musan'.+group_id"):
        audit_split_isolation(right)

    filter_kwargs = {
        "filter_policy_version": "allowlist_v1",
        "filter_policy_hash": _sha("policy"),
        "split_seed": 2025,
        "split_rules": {"strategy": "group_holdout"},
    }
    assert semantic_catalog_hash(left, **filter_kwargs) == semantic_catalog_hash(
        right, **filter_kwargs
    )

    recordings_a = tmp_path / "a" / RECORDINGS_JSONL_NAME
    recordings_b = tmp_path / "b" / RECORDINGS_JSONL_NAME
    recordings_a.parent.mkdir()
    recordings_b.parent.mkdir()
    write_recordings_jsonl(recordings_a, left)
    write_recordings_jsonl(recordings_b, right)
    catalog_a = build_catalog(
        dataset_id="musan",
        root="/mnt/a/musan",
        records=left,
        recordings_path=recordings_a,
        **filter_kwargs,
    )
    catalog_b = build_catalog(
        dataset_id="musan",
        root="/mnt/b/musan",
        records=right,
        recordings_path=recordings_b,
        **filter_kwargs,
    )
    assert catalog_a.semantic_catalog_hash == catalog_b.semantic_catalog_hash
    assert catalog_a.recordings_byte_hash != catalog_b.recordings_byte_hash
    assert catalog_a.root != catalog_b.root


def test_unknown_provenance_sets_capability_limit():
    records = [
        _record(
            recording_id="dns:1",
            dataset_id="dns",
            origin_ids=[],
            split="train",
            group_id="dns:1",
            relative_path="noise/1.wav",
            audio_path="/data/dns/noise/1.wav",
            audio_sha256=_sha("dns-1"),
        ),
        _record(
            recording_id="dns:2",
            dataset_id="dns",
            origin_ids=[],
            split="val",
            group_id="dns:2",
            relative_path="noise/2.wav",
            audio_path="/data/dns/noise/2.wav",
            audio_sha256=_sha("dns-2"),
        ),
    ]
    assert records[0].provenance_complete is False
    assert records[1].provenance_complete is False
    report = audit_split_isolation(records)
    assert report.provenance_complete is False
    assert "verifiable identity" in report.capability_limit
    assert "exact" in report.capability_limit.lower()
    assert "transcode" in report.capability_limit.lower()


def test_path_escape_rejected(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    with pytest.raises(ValueError, match="escapes"):
        resolve_relative_path("../secret.wav", root=root)
    with pytest.raises(ValueError, match="escapes"):
        resolve_relative_path("ok/../../outside.wav", root=root)
    with pytest.raises(ValueError, match="escapes"):
        resolve_relative_path(str(tmp_path / "outside.wav"), root=root)
    resolved = resolve_relative_path("noise/a.wav", root=root)
    assert resolved == (root / "noise" / "a.wav").resolve()


def test_missing_eligibility_defaults_false_and_is_excluded_from_train_filter():
    payload = _payload()
    payload.pop("background_eligible")
    record = background_record_from_mapping(payload)
    assert record.background_eligible is False
    assert eligible_train_records([record]) == ()

    ineligible = _record(background_eligible=False)
    val_only = _record(
        recording_id="musan:val",
        split="val",
        relative_path="noise/val.wav",
        audio_path="/data/musan/noise/val.wav",
        audio_sha256=_sha("val"),
        origin_ids=["musan:val"],
        group_id="musan:val",
    )
    assert eligible_train_records([ineligible, val_only, _record()])[0].recording_id == (
        "musan:noise-0001"
    )


def test_empty_eligible_train_set_for_active_source_raises():
    records = [
        _record(background_eligible=False),
        _record(
            recording_id="musan:val",
            split="val",
            background_eligible=True,
            relative_path="noise/val.wav",
            audio_path="/data/musan/noise/val.wav",
            audio_sha256=_sha("val"),
            origin_ids=["musan:val"],
            group_id="musan:val",
        ),
    ]
    with pytest.raises(ValueError, match="no eligible train recordings for source musan"):
        require_eligible_train_records(records, source_id="musan")


def test_jsonl_roundtrip_and_catalog_hashes(tmp_path):
    records = [
        _record(),
        _record(
            recording_id="musan:noise-0002",
            relative_path="noise/free-sound/noise-0002.wav",
            audio_path="/data/musan/noise/free-sound/noise-0002.wav",
            origin_ids=["musan:noise-0002"],
            audio_sha256=_sha("noise-0002"),
            group_id="musan:source:free-sound",
        ),
    ]
    recordings_path = tmp_path / RECORDINGS_JSONL_NAME
    write_recordings_jsonl(recordings_path, records)
    loaded = read_recordings_jsonl(recordings_path)
    assert loaded == records
    assert sha256_file(recordings_path) == hashlib.sha256(
        recordings_path.read_bytes()
    ).hexdigest()

    catalog = build_catalog(
        dataset_id="musan",
        root="/data/musan",
        records=records,
        recordings_path=recordings_path,
        filter_policy_version="allowlist_v1",
        filter_policy_hash=_sha("policy"),
        split_seed=2025,
        split_rules={"strategy": "group_holdout"},
        raw_metadata_hashes={"annotations": _sha("ann")},
    )
    catalog_path = tmp_path / CATALOG_JSON_NAME
    write_catalog(catalog_path, catalog)
    assert catalog_path.name == "catalog.json"
    reloaded = read_catalog(catalog_path)
    assert reloaded.semantic_catalog_hash == catalog.semantic_catalog_hash
    assert reloaded.recordings_byte_hash == catalog.recordings_byte_hash
    assert_catalog_source_id(reloaded, "musan")
    with pytest.raises(ValueError, match="dataset_id"):
        assert_catalog_source_id(reloaded, "dns")
