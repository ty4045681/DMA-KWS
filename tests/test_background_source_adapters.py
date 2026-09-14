from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import struct
import wave

import pytest
import yaml

from dma_kws.data_prep.background_adapters import (
    DnsSourceAdapter,
    Fsd50kSourceAdapter,
    MusanSourceAdapter,
    SourceImportConfig,
    get_source_adapter,
)
from dma_kws.data_prep.background_manifest import (
    CATALOG_JSON_NAME,
    RECORDINGS_JSONL_NAME,
    read_catalog,
    read_recordings_jsonl,
    sha256_file,
)
from dma_kws.data_prep.background_sources import (
    PrepareBackgroundConfig,
    prepare_background_sources,
)
from dma_kws.data_prep.musan_split import discover_musan_recordings
from scripts.prepare_background_sources import build_parser, main


def _write_wav(
    path: Path,
    *,
    frames: int = 160,
    sample_rate: int = 8000,
    marker: int = 1,
    channels: int = 1,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        frame = struct.pack("<" + "h" * channels, *([marker & 0x7FFF] + [0] * (channels - 1)))
        silence = b"\x00\x00" * channels
        handle.writeframes(frame + silence * (frames - 1))
    return path


def _write_list(path: Path, lines: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")
    return path


def _write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def _write_json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _fsd_clip_info(*, uploader: str, license_id: str) -> dict[str, object]:
    return {
        "description": "fixture clip",
        "license": license_id,
        "tags": ["fixture"],
        "title": "fixture",
        "uploader": uploader,
    }


def _assert_source_error(message: str, source_id: str, field: str) -> None:
    assert f"source id={source_id!r}" in message
    assert f"field={field}" in message
    assert "expected" in message
    assert "got" in message


def _build_musan_tree(root: Path) -> dict[str, Path]:
    files = {
        "music_a": _write_wav(root / "music" / "fma" / "music-a.wav", marker=11),
        "music_b": _write_wav(root / "music" / "fma" / "music-b.wav", marker=12),
        "noise_0": _write_wav(root / "noise" / "free-sound" / "noise-0.wav", marker=21),
        "noise_1": _write_wav(root / "noise" / "free-sound" / "noise-1.wav", marker=22),
        "noise_2": _write_wav(root / "noise" / "free-sound" / "noise-2.wav", marker=23),
        "speech_0": _write_wav(root / "speech" / "us-gov" / "speech-0.wav", marker=31),
        "speech_1": _write_wav(root / "speech" / "us-gov" / "speech-1.wav", marker=32),
    }
    (root / "music" / "fma" / "ANNOTATIONS").write_text(
        f"{files['music_a'].name} genre N artist-a\n"
        f"{files['music_b'].name} genre N artist-b\n",
        encoding="utf-8",
    )
    return files


def _musan_split_dir(root: Path, files: dict[str, Path], dest: Path) -> Path:
    train = [files["noise_0"], files["noise_1"], files["music_a"]]
    evaluation = [files["noise_2"], files["music_b"], files["speech_0"], files["speech_1"]]
    _write_list(dest / "train_background.list", [str(path.resolve()) for path in train])
    _write_list(dest / "eval_musan.list", [str(path.resolve()) for path in evaluation])
    (dest / "split.json").write_text("{}\n", encoding="utf-8")
    return dest


def _build_dns_tree(root: Path) -> dict[str, Path]:
    noise = root / "noise"
    files = {
        "a": _write_wav(noise / "a.wav", marker=41),
        "b": _write_wav(noise / "office" / "b.wav", marker=42),
        "c": _write_wav(noise / "office" / "c.wav", marker=43),
        "d": _write_wav(noise / "street" / "d.wav", marker=44),
    }
    _write_wav(root / "clean" / "speech.wav", marker=99)
    return files


def _build_fsd50k_tree(root: Path, *, eval_fields: list[str] | None = None) -> dict[str, Path]:
    audio = {
        "1001": _write_wav(root / "FSD50K.dev_audio" / "1001.wav", marker=51, sample_rate=44100),
        "1002": _write_wav(root / "FSD50K.dev_audio" / "1002.wav", marker=52, sample_rate=44100),
        "1003": _write_wav(root / "FSD50K.dev_audio" / "1003.wav", marker=53, sample_rate=44100),
        "2001": _write_wav(root / "FSD50K.eval_audio" / "2001.wav", marker=54, sample_rate=44100),
    }
    gt = root / "FSD50K.ground_truth"
    meta = root / "FSD50K.metadata"
    _write_csv(
        gt / "dev.csv",
        [
            {
                "fname": "1001",
                "labels": "Mechanical_fan",
                "mids": "/m/fan",
                "split": "train",
            },
            {
                "fname": "1002",
                "labels": "Mechanical_fan,Domestic_sounds_and_home_sounds",
                "mids": "/m/fan,/m/home",
                "split": "train",
            },
            {
                "fname": "1003",
                "labels": "Bark,Dog",
                "mids": "/m/bark,/m/dog",
                "split": "val",
            },
        ],
        ["fname", "labels", "mids", "split"],
    )
    eval_fields = eval_fields or ["fname", "labels", "mids"]
    eval_row = {"fname": "2001", "labels": "Rain", "mids": "/m/rain", "split": "train"}
    _write_csv(gt / "eval.csv", [{key: eval_row[key] for key in eval_fields}], eval_fields)
    _write_json(
        meta / "dev_clips_info_FSD50K.json",
        {
            "1001": _fsd_clip_info(uploader="alice", license_id="CC-BY-4.0"),
            "1002": _fsd_clip_info(uploader="bob", license_id="CC0"),
            "1003": _fsd_clip_info(uploader="carol", license_id="CC-BY-4.0"),
        },
    )
    _write_json(
        meta / "eval_clips_info_FSD50K.json",
        {"2001": _fsd_clip_info(uploader="dave", license_id="CC-BY-NC-4.0")},
    )
    return audio


def _by_id(records):
    return {record.recording_id: record for record in records}


def test_unknown_adapter_lists_allowed_names():
    with pytest.raises(ValueError, match=r"unknown .+ adapter") as caught:
        get_source_adapter("not-a-source")
    message = str(caught.value)
    for name in ("musan", "dns", "fsd50k"):
        assert name in message


def test_musan_adapter_reads_real_tree_and_reuses_existing_grouping(tmp_path):
    root = tmp_path / "musan"
    files = _build_musan_tree(root)
    split_dir = _musan_split_dir(root, files, tmp_path / "musan_split")
    records = list(
        MusanSourceAdapter().discover(
            SourceImportConfig(
                adapter="musan",
                id="musan",
                root=str(root),
                split_dir=str(split_dir),
                split_policy="preserve",
            )
        )
    )
    discovered = {item.relative_path: item for item in discover_musan_recordings(root)}
    assert len(records) == len(discovered) == 7
    noise0 = _by_id(records)["musan:noise/free-sound/noise-0.wav"]
    assert noise0.categories == ("noise",)
    assert noise0.audio_sha256 == sha256_file(files["noise_0"])
    assert noise0.audio_sha256 == hashlib.sha256(files["noise_0"].read_bytes()).hexdigest()
    assert noise0.split == "train"
    eval_noise = _by_id(records)["musan:noise/free-sound/noise-2.wav"]
    assert eval_noise.split == "test"
    for record in records:
        native = discovered[record.relative_path]
        assert record.group_id == f"musan:{native.group_id}"
        assert record.categories == (native.category,)
        assert record.sample_rate == native.sample_rate
        assert record.audio_sha256
        assert record.background_eligible is False


def test_dns_adapter_walks_noise_root_only_and_groups_per_recording(tmp_path):
    tree = tmp_path / "dns"
    files = _build_dns_tree(tree)
    records = list(
        DnsSourceAdapter().discover(
            SourceImportConfig(
                adapter="dns",
                id="dns",
                root=str(tree / "noise"),
                split_policy="group_random",
            )
        )
    )
    by_id = _by_id(records)
    assert set(by_id) == {
        "dns:a.wav",
        "dns:office/b.wav",
        "dns:office/c.wav",
        "dns:street/d.wav",
    }
    assert "speech" not in " ".join(record.relative_path for record in records)
    assert by_id["dns:a.wav"].audio_sha256 == sha256_file(files["a"])
    assert by_id["dns:a.wav"].group_id == "dns:a.wav"
    assert by_id["dns:office/b.wav"].group_id == "dns:office/b.wav"
    assert by_id["dns:a.wav"].origin_ids == ()
    assert by_id["dns:a.wav"].provenance_complete is False
    assert all(record.background_eligible is False for record in records)


def test_fsd50k_adapter_reads_official_clip_info_json(tmp_path):
    root = tmp_path / "fsd50k"
    audio = _build_fsd50k_tree(root)
    records = list(
        Fsd50kSourceAdapter().discover(
            SourceImportConfig(
                adapter="fsd50k",
                id="fsd50k",
                root=str(root),
                metadata=str(root / "FSD50K.metadata"),
                split_policy="preserve",
            )
        )
    )
    by_id = _by_id(records)
    train = by_id["fsd50k:1001"]
    assert train.split == "train"
    assert train.categories == ("Mechanical_fan",)
    assert train.group_id == "fsd50k:uploader:alice"
    assert train.origin_ids == ("freesound:1001",)
    assert train.license_id == "CC-BY-4.0"
    assert train.relative_path == "FSD50K.dev_audio/1001.wav"
    assert train.audio_sha256 == sha256_file(audio["1001"])
    assert train.sample_rate == 44100
    assert train.provenance_complete is True
    assert by_id["fsd50k:1002"].categories == (
        "Mechanical_fan",
        "Domestic_sounds_and_home_sounds",
    )
    assert by_id["fsd50k:1003"].split == "val"
    eval_clip = by_id["fsd50k:2001"]
    assert eval_clip.split == "test"
    assert eval_clip.license_id == "CC-BY-NC-4.0"
    assert eval_clip.group_id == "fsd50k:uploader:dave"
    assert eval_clip.relative_path == "FSD50K.eval_audio/2001.wav"


def test_fsd50k_missing_required_columns_lists_names(tmp_path):
    root = tmp_path / "fsd50k"
    _build_fsd50k_tree(root, eval_fields=["fname", "mids"])
    with pytest.raises(ValueError, match=r"missing required columns:.*labels") as caught:
        list(
            Fsd50kSourceAdapter().discover(
                SourceImportConfig(
                    adapter="fsd50k",
                    id="fsd50k",
                    root=str(root),
                    metadata=str(root / "FSD50K.metadata"),
                )
            )
        )
    _assert_source_error(str(caught.value), "fsd50k", "labels")
    assert "labels" in str(caught.value)


def test_prepare_raw_fixtures_to_catalogs_and_eligibility(tmp_path):
    musan_root = tmp_path / "musan"
    musan_files = _build_musan_tree(musan_root)
    musan_split = _musan_split_dir(musan_root, musan_files, tmp_path / "musan_split")
    dns_tree = tmp_path / "dns"
    dns_files = _build_dns_tree(dns_tree)
    fsd_root = tmp_path / "fsd50k"
    _build_fsd50k_tree(fsd_root)

    musan_allow = _write_list(
        tmp_path / "curation" / "musan_eligible.list",
        [
            "noise/free-sound/noise-0.wav",
            "noise/free-sound/noise-1.wav",
            "noise/free-sound/noise-2.wav",
        ],
    )
    dns_allow = _write_list(
        tmp_path / "curation" / "dns_eligible.list",
        ["a.wav", "office/b.wav", "office/c.wav", "street/d.wav"],
    )
    fsd_allow = _write_list(
        tmp_path / "curation" / "fsd50k_eligible.list",
        ["1001", "1002"],
    )
    output_dir = tmp_path / "background"
    result = prepare_background_sources(
        PrepareBackgroundConfig(
            output_dir=output_dir,
            seed=2025,
            sources=(
                SourceImportConfig(
                    adapter="musan",
                    id="musan",
                    root=str(musan_root),
                    split_dir=str(musan_split),
                    eligible_ids_file=str(musan_allow),
                    split_policy="preserve",
                    category_exclude=("speech",),
                ),
                SourceImportConfig(
                    adapter="dns",
                    id="dns",
                    root=str(dns_tree / "noise"),
                    eligible_ids_file=str(dns_allow),
                    split_policy="group_random",
                ),
                SourceImportConfig(
                    adapter="fsd50k",
                    id="fsd50k",
                    root=str(fsd_root),
                    metadata=str(fsd_root / "FSD50K.metadata"),
                    eligible_ids_file=str(fsd_allow),
                    split_policy="preserve",
                    category_allow=("Mechanical_fan", "Rain"),
                ),
            ),
        )
    )
    assert (output_dir / "audit.json").is_file()
    musan_records = read_recordings_jsonl(output_dir / "musan" / RECORDINGS_JSONL_NAME)
    musan_by_id = _by_id(musan_records)
    assert "musan:speech/us-gov/speech-0.wav" not in musan_by_id
    assert musan_by_id["musan:noise/free-sound/noise-0.wav"].background_eligible is True
    assert musan_by_id["musan:music/fma/music-a.wav"].background_eligible is False
    assert musan_by_id["musan:noise/free-sound/noise-2.wav"].split == "test"
    train_or_val = {
        musan_by_id["musan:noise/free-sound/noise-0.wav"].split,
        musan_by_id["musan:noise/free-sound/noise-1.wav"].split,
        musan_by_id["musan:music/fma/music-a.wav"].split,
    }
    assert train_or_val <= {"train", "val"}
    assert "train" in train_or_val
    assert musan_by_id["musan:noise/free-sound/noise-2.wav"].split != "train"
    assert all(
        record.split != "test"
        or record.recording_id
        in {
            "musan:noise/free-sound/noise-2.wav",
            "musan:music/fma/music-b.wav",
        }
        for record in musan_records
    )
    groups = {}
    for record in musan_records:
        groups.setdefault(record.group_id, set()).add(record.split)
    assert all(len(splits) == 1 for splits in groups.values())

    dns_records = read_recordings_jsonl(output_dir / "dns" / RECORDINGS_JSONL_NAME)
    assert {record.relative_path for record in dns_records} == {
        "a.wav",
        "office/b.wav",
        "office/c.wav",
        "street/d.wav",
    }
    assert {record.split for record in dns_records} == {"train", "val", "test"}
    assert all(record.background_eligible is True for record in dns_records)
    dns_groups = {}
    for record in dns_records:
        dns_groups.setdefault(record.group_id, set()).add(record.split)
    assert all(len(splits) == 1 for splits in dns_groups.values())

    fsd_records = read_recordings_jsonl(output_dir / "fsd50k" / RECORDINGS_JSONL_NAME)
    fsd_by_id = _by_id(fsd_records)
    assert "fsd50k:1003" not in fsd_by_id
    assert fsd_by_id["fsd50k:1001"].background_eligible is True
    assert fsd_by_id["fsd50k:1001"].split == "train"
    assert fsd_by_id["fsd50k:1002"].split == "train"
    assert fsd_by_id["fsd50k:2001"].split == "test"
    assert fsd_by_id["fsd50k:2001"].background_eligible is False
    fsd_catalog = read_catalog(output_dir / "fsd50k" / CATALOG_JSON_NAME)
    assert fsd_catalog.dataset_id == "fsd50k"
    assert fsd_catalog.raw_metadata_hashes["eligible_ids_file"] == sha256_file(fsd_allow)
    assert result.audit.capability_limit
    assert (output_dir / "musan" / "train.list").is_file()
    assert (output_dir / "dns" / "val.list").is_file()
    assert (output_dir / "fsd50k" / "test.list").is_file()
    for record in (*musan_records, *dns_records, *fsd_records):
        audio = Path(record.audio_path)
        assert record.audio_sha256 == sha256_file(audio)
        assert len(record.audio_sha256) == 64


def test_category_allow_does_not_mark_unknown_clips_eligible(tmp_path):
    root = tmp_path / "fsd50k"
    _build_fsd50k_tree(root)
    output_dir = tmp_path / "out"
    with pytest.raises(ValueError, match="no eligible train recordings for source fsd50k"):
        prepare_background_sources(
            PrepareBackgroundConfig(
                output_dir=output_dir,
                seed=2025,
                sources=(
                    SourceImportConfig(
                        adapter="fsd50k",
                        id="fsd50k",
                        root=str(root),
                        metadata=str(root / "FSD50K.metadata"),
                        split_policy="preserve",
                        category_allow=("Mechanical_fan",),
                    ),
                ),
            )
        )
    assert not output_dir.exists()


def test_group_random_errors_when_too_few_eligible_groups(tmp_path):
    noise = tmp_path / "noise"
    _write_wav(noise / "only-a.wav", marker=1)
    _write_wav(noise / "only-b.wav", marker=2)
    allow = _write_list(tmp_path / "allow.list", ["only-a.wav", "only-b.wav"])
    with pytest.raises(ValueError, match="too few independent eligible groups"):
        prepare_background_sources(
            PrepareBackgroundConfig(
                output_dir=tmp_path / "out",
                seed=2025,
                sources=(
                    SourceImportConfig(
                        adapter="dns",
                        id="dns",
                        root=str(noise),
                        eligible_ids_file=str(allow),
                        split_policy="group_random",
                    ),
                ),
            )
        )


def test_joint_overlap_audit_rejects_shared_bytes_across_sources(tmp_path):
    musan_root = tmp_path / "musan"
    musan_files = _build_musan_tree(musan_root)
    musan_split = _musan_split_dir(musan_root, musan_files, tmp_path / "musan_split")
    fsd_root = tmp_path / "fsd50k"
    fsd_audio = _build_fsd50k_tree(fsd_root)
    musan_files["noise_0"].write_bytes(fsd_audio["2001"].read_bytes())
    musan_allow = _write_list(
        tmp_path / "musan.list",
        [
            "noise/free-sound/noise-0.wav",
            "noise/free-sound/noise-1.wav",
            "noise/free-sound/noise-2.wav",
        ],
    )
    fsd_allow = _write_list(tmp_path / "fsd.list", ["1001", "1002"])
    with pytest.raises(ValueError, match="identical audio bytes"):
        prepare_background_sources(
            PrepareBackgroundConfig(
                output_dir=tmp_path / "out",
                seed=2025,
                sources=(
                    SourceImportConfig(
                        adapter="musan",
                        id="musan",
                        root=str(musan_root),
                        split_dir=str(musan_split),
                        eligible_ids_file=str(musan_allow),
                        split_policy="preserve",
                    ),
                    SourceImportConfig(
                        adapter="fsd50k",
                        id="fsd50k",
                        root=str(fsd_root),
                        metadata=str(fsd_root / "FSD50K.metadata"),
                        eligible_ids_file=str(fsd_allow),
                        split_policy="preserve",
                    ),
                ),
            )
        )


def test_publish_refuses_overwrite_and_failed_prepare_leaves_no_destination(
    tmp_path, monkeypatch
):
    noise = tmp_path / "noise"
    for index, name in enumerate(("a.wav", "b.wav", "c.wav", "d.wav")):
        _write_wav(noise / name, marker=90 + index)
    allow = _write_list(tmp_path / "allow.list", ["a.wav", "b.wav", "c.wav", "d.wav"])
    output_dir = tmp_path / "background"
    config = PrepareBackgroundConfig(
        output_dir=output_dir,
        seed=2025,
        sources=(
            SourceImportConfig(
                adapter="dns",
                id="dns",
                root=str(noise),
                eligible_ids_file=str(allow),
                split_policy="group_random",
            ),
        ),
    )
    prepare_background_sources(config)
    assert (output_dir / "dns" / RECORDINGS_JSONL_NAME).is_file()
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        prepare_background_sources(config)

    from dma_kws.data_prep import background_sources as sources_mod

    def _boom(*_args, **_kwargs):
        raise RuntimeError("injected publish failure")

    monkeypatch.setattr(sources_mod, "write_catalog", _boom)
    failed_dir = tmp_path / "failed-background"
    with pytest.raises(RuntimeError, match="injected publish failure"):
        prepare_background_sources(
            PrepareBackgroundConfig(
                output_dir=failed_dir,
                seed=2025,
                sources=config.sources,
            )
        )
    assert not failed_dir.exists()
    leftovers = list(tmp_path.glob(".failed-background*"))
    assert leftovers == []


def test_cli_help_documents_metadata_layouts():
    help_text = build_parser().format_help()
    for token in (
        "FSD50K.ground_truth",
        "dev.csv",
        "eval.csv",
        "fname",
        "labels",
        "split",
        "FSD50K.metadata",
        "dev_clips_info_FSD50K.json",
        "eval_clips_info_FSD50K.json",
        "uploader",
        "license",
        "FSD50K.dev_audio",
        "music/",
        "noise root",
    ):
        assert token in help_text
    assert "dev_clips.csv" not in help_text
    assert "username" not in help_text


def test_cli_safe_yaml_prepares_sources(tmp_path):
    noise = tmp_path / "noise"
    for index, name in enumerate(("a.wav", "b.wav", "c.wav", "d.wav")):
        _write_wav(noise / name, marker=60 + index)
    allow = _write_list(tmp_path / "allow.list", ["a.wav", "b.wav", "c.wav", "d.wav"])
    output_dir = tmp_path / "out"
    config_path = tmp_path / "prepare.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "output_dir": str(output_dir),
                "seed": 2025,
                "sources": [
                    {
                        "adapter": "dns",
                        "id": "dns",
                        "root": str(noise),
                        "eligible_ids_file": str(allow),
                        "split_policy": "group_random",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    main(["--config", str(config_path)])
    records = read_recordings_jsonl(output_dir / "dns" / RECORDINGS_JSONL_NAME)
    assert len(records) == 4
    assert {record.split for record in records} == {"train", "val", "test"}


def test_dns_metadata_non_file_is_not_a_silent_fallback(tmp_path):
    tree = tmp_path / "dns"
    _build_dns_tree(tree)
    missing = tmp_path / "missing-sidecar.csv"
    with pytest.raises(ValueError) as caught:
        list(
            DnsSourceAdapter().discover(
                SourceImportConfig(
                    adapter="dns",
                    id="dns",
                    root=str(tree / "noise"),
                    metadata=str(missing),
                    split_policy="group_random",
                )
            )
        )
    _assert_source_error(str(caught.value), "dns", "metadata")
    assert str(missing) in str(caught.value)

    directory = tmp_path / "not-a-file"
    directory.mkdir()
    with pytest.raises(ValueError) as caught_dir:
        list(
            DnsSourceAdapter().discover(
                SourceImportConfig(
                    adapter="dns",
                    id="dns",
                    root=str(tree / "noise"),
                    metadata=str(directory),
                    split_policy="group_random",
                )
            )
        )
    _assert_source_error(str(caught_dir.value), "dns", "metadata")


def test_prepare_errors_include_source_id_field_expected_got(tmp_path, monkeypatch):
    musan_root = tmp_path / "musan"
    files = _build_musan_tree(musan_root)
    split_dir = _musan_split_dir(musan_root, files, tmp_path / "musan_split")
    (split_dir / "train_background.list").write_text(
        f"{files['noise_0'].resolve()}\n{files['noise_1'].resolve()}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError) as musan_caught:
        list(
            MusanSourceAdapter().discover(
                SourceImportConfig(
                    adapter="musan",
                    id="musan",
                    root=str(musan_root),
                    split_dir=str(split_dir),
                    split_policy="preserve",
                )
            )
        )
    _assert_source_error(str(musan_caught.value), "musan", "split_dir")

    empty_noise = tmp_path / "empty-noise"
    empty_noise.mkdir()
    with pytest.raises(ValueError) as dns_caught:
        list(
            DnsSourceAdapter().discover(
                SourceImportConfig(
                    adapter="dns",
                    id="dns",
                    root=str(empty_noise),
                    split_policy="group_random",
                )
            )
        )
    _assert_source_error(str(dns_caught.value), "dns", "root")

    fsd_root = tmp_path / "fsd-ok"
    _build_fsd50k_tree(fsd_root)
    with pytest.raises(ValueError) as allow_caught:
        prepare_background_sources(
            PrepareBackgroundConfig(
                output_dir=tmp_path / "missing-allow",
                seed=2025,
                sources=(
                    SourceImportConfig(
                        adapter="fsd50k",
                        id="fsd50k",
                        root=str(fsd_root),
                        metadata=str(fsd_root / "FSD50K.metadata"),
                        eligible_ids_file=str(tmp_path / "no-such-allow.list"),
                        split_policy="preserve",
                    ),
                ),
            )
        )
    _assert_source_error(str(allow_caught.value), "fsd50k", "eligible_ids_file")

    from dma_kws.data_prep import background_adapters as adapters_mod

    monkeypatch.setattr(adapters_mod, "sha256_file", lambda _path: "")
    hash_root = tmp_path / "dns-hash"
    _write_wav(hash_root / "a.wav", marker=1)
    with pytest.raises(ValueError) as hash_caught:
        list(
            DnsSourceAdapter().discover(
                SourceImportConfig(
                    adapter="dns",
                    id="dns",
                    root=str(hash_root),
                    split_policy="group_random",
                )
            )
        )
    _assert_source_error(str(hash_caught.value), "dns", "audio_sha256")


def test_musan_preserve_carve_keeps_only_eligible_train_group(tmp_path):
    root = tmp_path / "musan"
    music = _write_wav(root / "music" / "fma" / "music-a.wav", marker=11)
    noise = _write_wav(root / "noise" / "free-sound" / "noise-0.wav", marker=21)
    speech = _write_wav(root / "speech" / "us-gov" / "speech-0.wav", marker=31)
    (root / "music" / "fma" / "ANNOTATIONS").write_text(
        f"{music.name} genre N artist-a\n",
        encoding="utf-8",
    )
    split_dir = tmp_path / "split"
    _write_list(
        split_dir / "train_background.list",
        [str(music.resolve()), str(noise.resolve())],
    )
    _write_list(split_dir / "eval_musan.list", [str(speech.resolve())])
    allow = _write_list(tmp_path / "allow.list", ["noise/free-sound/noise-0.wav"])
    output_dir = tmp_path / "out"
    prepare_background_sources(
        PrepareBackgroundConfig(
            output_dir=output_dir,
            seed=2025,
            sources=(
                SourceImportConfig(
                    adapter="musan",
                    id="musan",
                    root=str(root),
                    split_dir=str(split_dir),
                    eligible_ids_file=str(allow),
                    split_policy="preserve",
                ),
            ),
        )
    )
    records = _by_id(read_recordings_jsonl(output_dir / "musan" / RECORDINGS_JSONL_NAME))
    eligible = records["musan:noise/free-sound/noise-0.wav"]
    ineligible = records["musan:music/fma/music-a.wav"]
    assert eligible.background_eligible is True
    assert eligible.split == "train"
    assert ineligible.background_eligible is False
    assert ineligible.split == "val"
    assert records["musan:speech/us-gov/speech-0.wav"].split == "test"


def test_example_yaml_matches_plan_contract():
    path = Path("configs/background_sources/example.yaml")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert payload["output_dir"] == "/data/background"
    assert payload["seed"] == 2025
    adapters = [source["adapter"] for source in payload["sources"]]
    assert adapters == ["musan", "dns", "fsd50k"]
    assert payload["sources"][0]["split_policy"] == "preserve"
    assert payload["sources"][1]["split_policy"] == "group_random"
    assert payload["sources"][2]["metadata"] == "/data/fsd50k/metadata"
