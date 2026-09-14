"""T13 resume identity, v2 background signature, overlays, eval-list conflict."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import soundfile as sf

from dma_kws.config import compose_config, config_to_dict
from dma_kws.data_prep.background_manifest import (
    CATALOG_JSON_NAME,
    background_record_from_mapping,
    build_catalog,
    write_catalog,
    write_recordings_jsonl,
)
from dma_kws.stage2.adapt import _joint_data_signature
from dma_kws.stage2.background_identity import (
    BACKGROUND_DATA_SIGNATURE_VERSION,
    assert_background_eval_list_compatible,
    assert_background_resume_identity,
    background_data_signature_hash,
    background_data_signature_payload,
    is_multisource_background,
)
from tests.test_stage2_background_sources import _online_config, _source_config


def _write_wav(path: Path, *, scale: float = 0.2) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(
        str(path),
        np.linspace(-scale, scale, 16_000, dtype=np.float32),
        16_000,
        subtype="FLOAT",
    )
    return path


def _record(
    *,
    dataset_id: str,
    recording_id: str,
    audio_path: Path,
    split: str = "train",
    audio_sha256: str = "a" * 64,
    group_id: str | None = None,
) -> Any:
    return background_record_from_mapping(
        {
            "schema_version": 1,
            "dataset_id": dataset_id,
            "recording_id": recording_id,
            "relative_path": audio_path.name,
            "audio_path": str(audio_path),
            "group_id": group_id or f"{dataset_id}:group:{recording_id}",
            "origin_ids": [recording_id],
            "split": split,
            "categories": ["noise"],
            "background_eligible": True,
            "eligibility_basis": "curated_allowlist_v1",
            "duration_seconds": 1.0,
            "sample_rate": 16000,
            "channels": 1,
            "audio_sha256": audio_sha256,
            "license_id": "cc0",
            "provenance_complete": True,
        }
    )


def _write_source(
    root: Path,
    source_id: str,
    *,
    audio_root: Path | None = None,
    audio_sha256: str = "a" * 64,
) -> Path:
    audio_root = audio_root or (root / "audio" / source_id)
    records = []
    for split, name in (("train", "train"), ("val", "val"), ("test", "test")):
        wav = _write_wav(
            audio_root / f"{name}.wav",
            scale=0.15 if source_id == "dns" else 0.35,
        )
        records.append(
            _record(
                dataset_id=source_id,
                recording_id=f"{source_id}:{name}",
                audio_path=wav,
                split=split,
                audio_sha256=audio_sha256[:-1] + {"train": "1", "val": "2", "test": "3"}[split],
                group_id=f"{source_id}:group:{split}",
            )
        )
    catalog_dir = root / "catalog" / source_id
    catalog_dir.mkdir(parents=True, exist_ok=True)
    manifest = catalog_dir / "recordings.jsonl"
    write_recordings_jsonl(manifest, records)
    catalog = build_catalog(
        dataset_id=source_id,
        root=str(audio_root),
        records=records,
        recordings_path=manifest,
        filter_policy_version="curated_allowlist_v1",
        filter_policy_hash="b" * 64,
        split_seed=2025,
        split_rules={"policy": "preserve"},
    )
    write_catalog(catalog_dir / CATALOG_JSON_NAME, catalog)
    return manifest


def _legacy_bg(list_path: str) -> dict[str, Any]:
    return {
        "enabled": True,
        "probability": 0.25,
        "audio_list_path": list_path,
        "duration_seconds_min": 1.0,
        "duration_seconds_max": 3.0,
        "mode": "online",
        "cache_manifest": "",
        "max_open_shards": 8,
    }


def _joint_config(background: dict[str, Any], *, wav_dir: Path) -> dict[str, Any]:
    return {
        "adapt": {
            "keyword": "hello",
            "joint": {},
            "mix_ratio": 0.5,
            "sample_lens": 64,
        },
        "stage2": {
            "background_negative": background,
            "parquet_file": "",
            "wav_dir": str(wav_dir),
            "negative_ratio": 1,
            "hard_negative_ratio": 1,
            "accumulate_grad_batches": 1,
        },
        "training": {"seed": 2025},
        "fbank": {"num_mel_bins": 80, "dither": 0.0},
    }


def _signature_args(tmp_path: Path) -> dict[str, Any]:
    parquet = tmp_path / "pairs.parquet"
    dictionary = tmp_path / "dict.txt"
    wav_dir = tmp_path / "features"
    parquet.write_bytes(b"pairs-v1")
    dictionary.write_text("a 1\n")
    wav_dir.mkdir()
    return {
        "manifests": {},
        "parquet_file": parquet,
        "dict_path": dictionary,
        "wav_dir": wav_dir,
    }


def test_legacy_joint_signature_ignores_empty_sources_and_validation_defaults(tmp_path):
    args = _signature_args(tmp_path)
    list_path = tmp_path / "train_background.list"
    list_path.write_text("noise.wav\n")
    old_bg = _legacy_bg(str(list_path))
    new_bg = {
        **old_bg,
        "sources": [],
        "validation": {"enabled": False, "samples_per_source": 256, "seed": 2025},
    }
    old = _joint_data_signature(_joint_config(old_bg, wav_dir=args["wav_dir"]), **args)
    new = _joint_data_signature(_joint_config(new_bg, wav_dir=args["wav_dir"]), **args)
    assert old == new


def test_legacy_joint_signature_payload_is_byte_identical_to_pre_sources_hash(tmp_path):
    args = _signature_args(tmp_path)
    list_path = tmp_path / "train_background.list"
    list_path.write_text("noise.wav\n")
    bg = {
        **_legacy_bg(str(list_path)),
        "sources": [],
        "validation": {"enabled": False, "samples_per_source": 256, "seed": 2025},
    }
    config = _joint_config(bg, wav_dir=args["wav_dir"])
    files = {
        "libri_parquet": Path(args["parquet_file"]),
        "tokenizer": Path(args["dict_path"]),
        "background_train": list_path,
    }
    digests = {}
    for key, path in files.items():
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        digests[key] = {"path": str(path.resolve()), "sha256": digest.hexdigest()}
    payload = {
        "files": digests,
        "joint": config["adapt"].get("joint", {}),
        "mix_ratio": config["adapt"].get("mix_ratio", 0.5),
        "seed": config.get("training", {}).get("seed", 2025),
        "sample_lens": config["adapt"].get("sample_lens"),
        "background": _legacy_bg(str(list_path)),
        "fbank": config.get("fbank"),
        "wav_dir": str(Path(args["wav_dir"]).resolve()),
        "keyword": config["adapt"].get("keyword"),
        "replay": {
            key: config["stage2"].get(key)
            for key in ("parquet_file", "wav_dir", "negative_ratio", "hard_negative_ratio")
        },
        "accumulate_grad_batches": config["stage2"].get("accumulate_grad_batches", 1),
    }
    expected = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    assert _joint_data_signature(config, **args) == expected


def test_is_multisource_background_false_for_empty_sources():
    assert is_multisource_background({}) is False
    assert is_multisource_background({"sources": []}) is False
    assert is_multisource_background({"enabled": True, "audio_list_path": "x.list"}) is False


def test_multisource_signature_is_version_2_and_ignores_mount_paths(tmp_path):
    first_root = tmp_path / "mount_a"
    second_root = tmp_path / "mount_b"
    sources = []
    relocated = []
    for source_id, weight in (("dns", 0.4), ("musan", 0.6)):
        manifest = _write_source(first_root, source_id, audio_sha256="c" * 64)
        sources.append(_source_config(source_id, manifest, weight=weight))
        relocated.append(
            _source_config(
                source_id,
                _write_source(
                    second_root,
                    source_id,
                    audio_root=second_root / "audio" / source_id,
                    audio_sha256="c" * 64,
                ),
                weight=weight,
            )
        )
    fbank = {"num_mel_bins": 80, "dither": 0.0, "frame_length": 25, "frame_shift": 10}
    first = background_data_signature_payload(
        _online_config(sources),
        fbank=fbank,
        seed=2025,
    )
    second = background_data_signature_payload(
        _online_config(relocated),
        fbank=fbank,
        seed=2025,
    )
    assert first["background_data_signature_version"] == BACKGROUND_DATA_SIGNATURE_VERSION == 2
    assert first["sources"][0]["id"] == "dns"
    assert "audio_path" not in json.dumps(first)
    assert str(first_root) not in json.dumps(first)
    assert str(second_root) not in json.dumps(second)
    assert background_data_signature_hash(first) == background_data_signature_hash(second)


def test_multisource_signature_changes_with_weight_content_and_policy(tmp_path):
    manifests = {
        source_id: _write_source(tmp_path, source_id)
        for source_id in ("dns", "musan")
    }
    fbank = {"num_mel_bins": 80, "dither": 0.0}
    base = _online_config(
        [
            _source_config("dns", manifests["dns"], weight=0.4),
            _source_config("musan", manifests["musan"], weight=0.6),
        ]
    )
    base_hash = background_data_signature_hash(
        background_data_signature_payload(base, fbank=fbank, seed=2025)
    )
    heavier = _online_config(
        [
            _source_config("dns", manifests["dns"], weight=0.5),
            _source_config("musan", manifests["musan"], weight=0.5),
        ]
    )
    assert (
        background_data_signature_hash(
            background_data_signature_payload(heavier, fbank=fbank, seed=2025)
        )
        != base_hash
    )
    other_content = _write_source(tmp_path / "other", "dns", audio_sha256="d" * 64)
    mutated = _online_config(
        [
            _source_config("dns", other_content, weight=0.4),
            _source_config("musan", manifests["musan"], weight=0.6),
        ]
    )
    assert (
        background_data_signature_hash(
            background_data_signature_payload(mutated, fbank=fbank, seed=2025)
        )
        != base_hash
    )
    policy = dict(base)
    policy["duration_seconds_max"] = 2.0
    assert (
        background_data_signature_hash(
            background_data_signature_payload(policy, fbank=fbank, seed=2025)
        )
        != base_hash
    )


def test_strict_resume_rejects_source_weight_and_legacy_to_multisource(tmp_path):
    manifests = {
        source_id: _write_source(tmp_path, source_id)
        for source_id in ("dns", "musan")
    }
    fbank = {"num_mel_bins": 80, "dither": 0.0}
    current = background_data_signature_hash(
        background_data_signature_payload(
            _online_config(
                [
                    _source_config("dns", manifests["dns"], weight=0.4),
                    _source_config("musan", manifests["musan"], weight=0.6),
                ]
            ),
            fbank=fbank,
            seed=2025,
        )
    )
    with pytest.raises((SystemExit, ValueError), match="signature|multi-source|new run"):
        assert_background_resume_identity(
            {},
            current_signature=current,
            current_config={"stage2": {"background_negative": {"sources": [{"id": "dns"}]}}},
        )
    with pytest.raises((SystemExit, ValueError), match="signature"):
        assert_background_resume_identity(
            {
                "background_data_signature_version": 2,
                "background_data_signature": "0" * 64,
            },
            current_signature=current,
            current_config={"stage2": {"background_negative": {"sources": [{"id": "dns"}]}}},
        )
    assert_background_resume_identity(
        {"background_data_signature_version": 2, "background_data_signature": current},
        current_signature=current,
        current_config={"stage2": {"background_negative": {"sources": [{"id": "dns"}]}}},
    )
    assert_background_resume_identity(
        {},
        current_signature=None,
        current_config={"stage2": {"background_negative": {"sources": []}}},
    )


def test_new_mode_joint_val_warning_does_not_mention_background_eval_list():
    from dma_kws.stage2.background_identity import (
        joint_background_validation_disabled_message,
    )

    message = joint_background_validation_disabled_message(
        {"sources": [{"id": "dns", "weight": 1.0, "manifest": "x.jsonl"}]}
    )
    assert "background_eval_list" not in message
    assert "stage2.background_negative.validation" in message
    legacy = joint_background_validation_disabled_message({"sources": []})
    assert "background_eval_list" in legacy


def test_background_eval_list_conflicts_with_nonempty_sources():
    with pytest.raises(ValueError, match="background_eval_list"):
        assert_background_eval_list_compatible(
            {"sources": [{"id": "dns", "weight": 1.0, "manifest": "x.jsonl"}]},
            {"background_eval_list": "val.list"},
        )
    assert_background_eval_list_compatible(
        {"sources": []},
        {"background_eval_list": "val.list"},
    )
    assert_background_eval_list_compatible(
        {"sources": [{"id": "dns", "weight": 1.0, "manifest": "x.jsonl"}]},
        {"background_eval_list": ""},
    )


def test_multisource_overlays_empty_inherited_paths_and_isolate_outputs():
    v41 = config_to_dict(
        compose_config("icefall_zipformer_stage2_eps_softmin_v41")
    )
    online = config_to_dict(
        compose_config("icefall_zipformer_stage2_eps_softmin_v41_multisource")
    )
    cached = config_to_dict(
        compose_config("icefall_zipformer_stage2_eps_softmin_v41_multisource_cached")
    )
    for cfg in (online, cached):
        bg = cfg["stage2"]["background_negative"]
        assert bg["audio_list_path"] == ""
        assert bg["cache_manifest"] == ""
        assert [source["id"] for source in bg["sources"]] == ["musan", "dns", "fsd50k"]
        assert [source["weight"] for source in bg["sources"]] == [0.4, 0.4, 0.2]
        assert cfg["stage2"]["run_name"] != v41["stage2"]["run_name"]
        assert cfg["stage2"]["checkpoint_dir"] != v41["stage2"]["checkpoint_dir"]
        assert cfg["stage2"]["log_dir"] != v41["stage2"]["log_dir"]
        assert cfg["stage2"]["qbyt_readout_version"] == 4
        assert cfg["stage2"]["qbyt_readout"]["mode"] == "eps_softmin"
    assert online["stage2"]["background_negative"]["mode"] == "online"
    assert cached["stage2"]["background_negative"]["mode"] == "fbank_cache"
    cached_sources = cached["stage2"]["background_negative"]["sources"]
    assert all(source["manifest"] for source in cached_sources)
    assert all(source["cache_manifest"] for source in cached_sources)
    assert "multisource-cached" in cached["stage2"]["run_name"]
    assert cached["stage2"]["run_name"] != online["stage2"]["run_name"]
    assert all(
        not source["cache_manifest"]
        for source in online["stage2"]["background_negative"]["sources"]
    )
