from __future__ import annotations

import json
import logging
from pathlib import Path, PurePosixPath
import csv
from collections import Counter
import wave
import zipfile

import pytest

from dma_kws.data_prep.chinese_accent_english import (
    DEFAULT_REMOTE_DATASET_ROOT,
    L2_MANDARIN_SPLITS,
    ManifestPathMapper,
    _deterministic_sample,
    build_chinese_accent_manifests,
    extract_l2_arctic_mandarin,
    hard_negative_candidate_records,
    materialize_adapter_mixtures,
    normalize_l2_arctic_audio,
    prepare_edacc_records,
    prepare_l2_arctic_records,
    prepare_real_recording_records,
    prepare_speechocean_records,
)


class _FakeG2P:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def __call__(self, text: str) -> list[str]:
        self.calls.append(text)
        return ["HH", "EH1", "L", "OW1"]


def test_deterministic_sample_logs_oversampling_factor(caplog):
    with caplog.at_level(logging.WARNING):
        sampled = _deterministic_sample(
            [{"utt_id": "one", "wav_path": "/remote/one.wav"}],
            3,
            seed=7,
            source="tiny",
            partition="train",
        )

    assert len(sampled) == 3
    assert len({record["mixture_instance_id"] for record in sampled}) == 3
    assert (
        "Oversampling source 'tiny' partition=train: requested=3 available=1 "
        "mean_reuse=3.00x max_reuse=3x"
    ) in caplog.text


def _write_kaldi_map(path: Path, rows: list[tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{key}\t{value}\n" for key, value in rows), encoding="utf-8")


def _make_speechocean(root: Path) -> None:
    train_rows = []
    train_utt2spk = []
    train_text = []
    train_ages = []
    train_genders = []
    for index in range(4):
        speaker = f"{index + 1:04d}"
        utt_id = f"{index + 1:04d}001"
        relative = f"WAVE/SPEAKER{speaker}/{utt_id}.WAV"
        (root / relative).parent.mkdir(parents=True, exist_ok=True)
        (root / relative).write_bytes(b"wav")
        train_rows.append((utt_id, relative))
        train_utt2spk.append((utt_id, speaker))
        train_text.append((utt_id, f"TRAIN TEXT {index}"))
        train_ages.append((speaker, "12" if index < 2 else "30"))
        train_genders.append((speaker, "m" if index % 2 == 0 else "f"))
    _write_kaldi_map(root / "train" / "wav.scp", train_rows)
    _write_kaldi_map(root / "train" / "utt2spk", train_utt2spk)
    _write_kaldi_map(root / "train" / "text", train_text)
    _write_kaldi_map(root / "train" / "spk2age", train_ages)
    _write_kaldi_map(root / "train" / "spk2gender", train_genders)

    test_utt = "9001001"
    test_speaker = "9001"
    test_relative = f"WAVE/SPEAKER{test_speaker}/{test_utt}.WAV"
    (root / test_relative).parent.mkdir(parents=True, exist_ok=True)
    (root / test_relative).write_bytes(b"wav")
    _write_kaldi_map(root / "test" / "wav.scp", [(test_utt, test_relative)])
    _write_kaldi_map(root / "test" / "utt2spk", [(test_utt, test_speaker)])
    _write_kaldi_map(root / "test" / "text", [(test_utt, "TEST TEXT")])
    _write_kaldi_map(root / "test" / "spk2age", [(test_speaker, "25")])
    _write_kaldi_map(root / "test" / "spk2gender", [(test_speaker, "f")])


def _make_l2_arctic(root: Path) -> None:
    for speaker in L2_MANDARIN_SPLITS:
        (root / speaker / "transcript").mkdir(parents=True)
        (root / speaker / "wav").mkdir(parents=True)
        (root / speaker / "annotation").mkdir(parents=True)
        (root / speaker / "transcript" / "arctic_a0001.txt").write_text(
            f"Canonical words from {speaker}.", encoding="utf-8"
        )
        (root / speaker / "wav" / "arctic_a0001.wav").write_bytes(b"wav")
        # A conflicting TextGrid token must never become the CTC target.
        (root / speaker / "annotation" / "arctic_a0001.TextGrid").write_text(
            "PPL WRONG_PHONE <unk>", encoding="utf-8"
        )


def _make_edacc(root: Path) -> None:
    rows = [
        {
            "relative_audio_path": "audio/validation/EDACC-C16-A/keep.wav",
            "audio_path": "/stale/mac/path/keep.wav",
            "speaker_id": "EDACC-C16-A",
            "text": "HELLO <LAUGH> WORLD",
            "source_split": "validation",
            "accent": "Chinese",
            "l1": "Mandarin",
            "gender": "female",
            "duration_seconds": 1.0,
            "sample_rate": 32000,
        },
        {
            "relative_audio_path": "audio/validation/EDACC-C16-B/reject.wav",
            "audio_path": "/stale/mac/path/reject.wav",
            "speaker_id": "EDACC-C16-B",
            "text": "HELLO <OVERLAP> WORLD",
            "source_split": "validation",
            "accent": "Chinese",
            "l1": "Mandarin",
        },
        {
            "relative_audio_path": "audio/test/EDACC-C19-A/test.wav",
            "speaker_id": "EDACC-C19-A",
            "text": "A VALID TEST",
            "source_split": "test",
            "accent": "Chinese",
            "l1": "Mandarin",
        },
        {
            "relative_audio_path": "audio/test/EDACC-C42-A/test.wav",
            "speaker_id": "EDACC-C42-A",
            "text": "ANOTHER TEST",
            "source_split": "test",
            "accent": "Chinese",
            "l1": "Mandarin",
        },
        {
            "relative_audio_path": "audio/validation/EDACC-C04-B/control.wav",
            "speaker_id": "EDACC-C04-B",
            "text": "CONTROL SAMPLE",
            "source_split": "validation",
            "accent": "Southern British",
            "l1": "Mandarin",
        },
    ]
    for row in rows:
        audio_path = root / row["relative_audio_path"]
        audio_path.parent.mkdir(parents=True, exist_ok=True)
        audio_path.write_bytes(b"wav")
    root.mkdir(parents=True, exist_ok=True)
    (root / "manifest.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def _write_silent_wav(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(b"\x00\x00" * 160)


def _make_real_recordings(root: Path) -> list[str]:
    fieldnames = ["姓名", "性别", "年龄段", "录音环境", "类别", "唤醒词", "录音组ID"]
    rows: list[dict[str, str]] = []

    def append_speaker(name: str, gender: str, count: int) -> None:
        for index in range(count):
            positive = index < count // 2
            group_id = f"group_{len(rows):04d}"
            phrase = "Hey Eva" if positive else ("Hey Ava" if index % 2 else "Hey Eve")
            category = "positive" if positive else "near_negative"
            rows.append(
                {
                    "姓名": name,
                    "性别": gender,
                    "年龄段": "26~35",
                    "录音环境": "office",
                    "类别": category,
                    "唤醒词": phrase,
                    "录音组ID": group_id,
                }
            )
            _write_silent_wav(root / category / phrase / group_id / "contains-name.wav")

    # The two raw aliases normalize to one 40-row train-only speaker.
    append_speaker("alias", "male", 20)
    append_speaker("alias2", "male", 20)
    complete_names: list[str] = []
    for index in range(8):
        name = f"male_{index}x"
        complete_names.append(name)
        append_speaker(name, "male", 20)
    for index in range(4):
        name = f"female_{index}x"
        complete_names.append(name)
        append_speaker(name, "female", 20)
    append_speaker("secretperson", "male", 1)

    root.mkdir(parents=True, exist_ok=True)
    with (root / "recording_info.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return ["alias", "alias2", "secretperson", *complete_names]


def test_manifest_path_mapper_writes_requested_remote_absolute_prefix(tmp_path):
    source_root = tmp_path / "chinese_accent_english_datasets"
    audio = source_root / "raw" / "corpus" / "speaker" / "sample.wav"
    audio.parent.mkdir(parents=True)
    audio.write_bytes(b"wav")
    mapper = ManifestPathMapper(source_root, DEFAULT_REMOTE_DATASET_ROOT)

    assert mapper.manifest_path(audio) == (
        "/home/q00931063/DMA-KWS/data/dma-kws/"
        "chinese_accent_english_datasets/raw/corpus/speaker/sample.wav"
    )
    with pytest.raises(ValueError, match="outside local dataset root"):
        mapper.manifest_path(tmp_path / "outside.wav")
    with pytest.raises(ValueError, match="absolute POSIX"):
        ManifestPathMapper(source_root, PurePosixPath("relative/root"))


def test_speechocean_uses_only_official_rows_and_speaker_disjoint_split(tmp_path):
    dataset_root = tmp_path / "dataset"
    root = dataset_root / "raw" / "speechocean762"
    _make_speechocean(root)
    # Extra physical audio is not referenced by wav.scp and must stay out.
    extra = root / "WAVE" / "SPEAKER0001" / "extra.WAV"
    extra.write_bytes(b"wav")
    records = prepare_speechocean_records(
        root,
        mapper=ManifestPathMapper(dataset_root),
        g2p=_FakeG2P(),
        dev_speaker_count=1,
    )

    assert len(records) == 5
    assert {record["split"] for record in records} == {"train", "dev", "test"}
    assert all(PurePosixPath(record["wav_path"]).is_absolute() for record in records)
    speakers_by_split = {
        split: {record["speaker_id"] for record in records if record["split"] == split}
        for split in ("train", "dev", "test")
    }
    assert speakers_by_split["train"].isdisjoint(speakers_by_split["dev"])
    assert speakers_by_split["test"].isdisjoint(
        speakers_by_split["train"] | speakers_by_split["dev"]
    )
    assert not any(record["source_utt_id"] == "extra" for record in records)


def test_l2_arctic_uses_transcripts_not_textgrids_and_fixed_splits(tmp_path):
    dataset_root = tmp_path / "dataset"
    root = dataset_root / "raw" / "l2_arctic_v5"
    _make_l2_arctic(root)
    g2p = _FakeG2P()

    records = prepare_l2_arctic_records(
        root, mapper=ManifestPathMapper(dataset_root), g2p=g2p
    )

    assert len(records) == 4
    assert {
        record["speaker_id"].removeprefix("l2_arctic-"): record["split"]
        for record in records
    } == dict(L2_MANDARIN_SPLITS)
    assert all("wrong_phone" not in call for call in g2p.calls)
    assert all(record["phonemes_g2p"] == "HH EH1 L OW1" for record in records)


def test_edacc_uses_relative_audio_path_and_quarantines_special_segments(tmp_path):
    dataset_root = tmp_path / "dataset"
    root = dataset_root / "raw" / "edacc_mandarin"
    _make_edacc(root)

    records, quarantine = prepare_edacc_records(
        root, mapper=ManifestPathMapper(dataset_root), g2p=_FakeG2P()
    )

    assert len(records) == 4
    assert len(quarantine) == 1
    assert quarantine[0]["reason"] == "<OVERLAP>"
    keep = next(record for record in records if record["source_utt_id"] == "keep")
    assert keep["text"] == "HELLO WORLD"
    assert "/stale/mac/path" not in keep["wav_path"]
    assert {record["split"] for record in records} == {"dev", "test", "control"}


def test_full_build_writes_source_manifests_without_fake_librispeech_mix(tmp_path):
    source_root = tmp_path / "dataset"
    _make_speechocean(source_root / "raw" / "speechocean762")
    _make_l2_arctic(source_root / "raw" / "l2_arctic_v5")
    _make_edacc(source_root / "raw" / "edacc_mandarin")

    outputs = build_chinese_accent_manifests(
        source_root=source_root,
        g2p=_FakeG2P(),
        speechocean_dev_speakers=1,
        normalize_l2_audio=False,
    )

    assert (outputs.manifest_dir / "adapter" / "accent_train.jsonl").is_file()
    assert not (outputs.manifest_dir / "adapter" / "train_mix.jsonl").exists()
    assert (outputs.manifest_dir / "eval" / "edacc_control.jsonl").is_file()
    catalog = [
        json.loads(line)
        for line in (outputs.manifest_dir / "catalog.jsonl").read_text().splitlines()
    ]
    assert catalog
    assert all(
        record["wav_path"].startswith(str(DEFAULT_REMOTE_DATASET_ROOT) + "/")
        for record in catalog
    )
    source_rows = [
        json.loads(line)
        for line in (source_root / "raw" / "edacc_mandarin" / "manifest.jsonl")
        .read_text()
        .splitlines()
    ]
    assert all(
        row["audio_path"].startswith(str(DEFAULT_REMOTE_DATASET_ROOT) + "/")
        for row in source_rows
    )
    split_audit = json.loads((outputs.reports_dir / "split_audit.json").read_text())
    assert split_audit["speaker_split_overlap"] == 0


def test_extract_l2_arctic_selects_only_mandarin_nested_archives(tmp_path):
    archive_path = tmp_path / "release.zip"
    nested_archives: dict[str, bytes] = {}
    for speaker in (*L2_MANDARIN_SPLITS, "OTHER"):
        nested_path = tmp_path / f"{speaker}.zip"
        with zipfile.ZipFile(nested_path, "w") as nested:
            nested.writestr(f"{speaker}/transcript/utt.txt", "HELLO")
            nested.writestr(f"{speaker}/wav/utt.wav", b"wav")
        nested_archives[f"{speaker}.zip"] = nested_path.read_bytes()
    with zipfile.ZipFile(archive_path, "w") as outer:
        outer.writestr("LICENSE", "license")
        for name, content in nested_archives.items():
            outer.writestr(name, content)

    destination = tmp_path / "l2_arctic_v5"
    extract_l2_arctic_mandarin(archive_path, destination)

    assert all((destination / speaker / "wav" / "utt.wav").is_file() for speaker in L2_MANDARIN_SPLITS)
    assert not (destination / "OTHER").exists()
    assert (destination / "LICENSE").read_text() == "license"


def test_l2_normalization_reuses_only_verified_16k_pcm16_outputs(tmp_path, monkeypatch):
    raw_root = tmp_path / "raw_l2"
    output_root = tmp_path / "derived"
    _make_l2_arctic(raw_root)
    for speaker in L2_MANDARIN_SPLITS:
        _write_silent_wav(output_root / speaker / "arctic_a0001.wav")
    monkeypatch.setattr(
        "dma_kws.data_prep.chinese_accent_english.shutil.which", lambda _: "/fake/ffmpeg"
    )

    stats = normalize_l2_arctic_audio(raw_root, output_root, num_workers=2)
    records = prepare_l2_arctic_records(
        raw_root,
        audio_root=output_root,
        mapper=ManifestPathMapper(tmp_path),
        g2p=_FakeG2P(),
    )

    assert stats == {"total": 4, "written": 0, "skipped": 4}
    assert all("/derived/" in record["wav_path"] for record in records)


def test_real_recordings_are_anonymized_and_split_200_40_40_1(tmp_path):
    recordings_root = tmp_path / "recordings"
    source_names = _make_real_recordings(recordings_root)
    dataset_root = tmp_path / "dataset"

    records, quarantine = prepare_real_recording_records(
        recordings_root,
        destination_root=dataset_root / "raw" / "hey_eva_real" / "pc",
        mapper=ManifestPathMapper(dataset_root),
        g2p=_FakeG2P(),
    )

    assert Counter(record["split"] for record in records) == {
        "train": 200,
        "dev": 40,
        "test": 40,
    }
    assert len(quarantine) == 1
    assert Counter(record["label"] for record in records) == {1: 140, 0: 140}
    serialized = json.dumps(records + quarantine, ensure_ascii=False)
    assert not any(name in serialized for name in source_names)
    assert all(record["speaker_id"].startswith("real_pc_spk_") for record in records)
    assert all(PurePosixPath(record["wav_path"]).is_absolute() for record in records)


def test_materialize_adapter_mixtures_has_exact_ratios_and_requires_absolute_ls(tmp_path):
    manifest_dir = tmp_path / "manifests"
    adapter_dir = manifest_dir / "adapter"
    adapter_dir.mkdir(parents=True)
    source_files = {
        "speechocean762": "speechocean762",
        "l2_arctic_mandarin": "l2_arctic_mandarin",
        "hey_eva_real_pc": "real",
    }
    for source, prefix in source_files.items():
        for split in ("train", "dev"):
            row = {
                "utt_id": f"{source}-{split}",
                "source": source,
                "wav_path": f"/remote/{source}/{split}.wav",
                "phonemes_g2p": "HH EH1 L OW1",
            }
            (adapter_dir / f"{prefix}_{split}.jsonl").write_text(json.dumps(row) + "\n")
    libri_train = tmp_path / "ls_train.jsonl"
    libri_dev = tmp_path / "ls_dev.jsonl"
    libri_train.write_text(
        json.dumps(
            {
                "utt_id": "ls-train",
                "speaker_id": "ls-train-speaker",
                "wav_path": "/remote/ls/train.flac",
                "phonemes": ["HH"],
            }
        )
        + "\n"
    )
    libri_dev.write_text(
        json.dumps(
            {
                "utt_id": "ls-dev",
                "speaker_id": "ls-dev-speaker",
                "wav_path": "/remote/ls/dev.flac",
                "phonemes": ["HH"],
            }
        )
        + "\n"
    )

    materialize_adapter_mixtures(
        manifest_dir=manifest_dir,
        librispeech_train_manifest=libri_train,
        librispeech_dev_manifest=libri_dev,
        train_size=20,
        dev_size=10,
    )
    train = [json.loads(line) for line in (adapter_dir / "train_mix.jsonl").read_text().splitlines()]
    dev = [json.loads(line) for line in (adapter_dir / "dev_select.jsonl").read_text().splitlines()]

    assert Counter(record["source"] for record in train) == {
        "librispeech": 12,
        "speechocean762": 5,
        "l2_arctic_mandarin": 2,
        "hey_eva_real_pc": 1,
    }
    assert Counter(record["source"] for record in dev) == {
        "librispeech": 4,
        "speechocean762": 3,
        "l2_arctic_mandarin": 2,
        "hey_eva_real_pc": 1,
    }

    libri_train.write_text(
        json.dumps({"utt_id": "bad", "wav_path": "relative.wav", "phonemes": ["HH"]}) + "\n"
    )
    with pytest.raises(ValueError, match="absolute Linux"):
        materialize_adapter_mixtures(
            manifest_dir=manifest_dir,
            librispeech_train_manifest=libri_train,
            librispeech_dev_manifest=libri_dev,
            train_size=20,
            dev_size=10,
        )

    duplicate = {
        "utt_id": "same-utt",
        "wav_path": "/remote/ls/same.flac",
        "phonemes": ["HH"],
    }
    payload = json.dumps(duplicate) + "\n"
    libri_train.write_text(payload, encoding="utf-8")
    libri_dev.write_text(payload, encoding="utf-8")
    with pytest.raises(ValueError, match="overlap by wav_path"):
        materialize_adapter_mixtures(
            manifest_dir=manifest_dir,
            librispeech_train_manifest=libri_train,
            librispeech_dev_manifest=libri_dev,
            train_size=20,
            dev_size=10,
        )

    libri_train.write_text(
        json.dumps(
            {
                "utt_id": "19-100-0001",
                "wav_path": "/remote/ls/train-speaker.flac",
                "phonemes": ["HH"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    libri_dev.write_text(
        json.dumps(
            {
                "utt_id": "19-200-0002",
                "wav_path": "/remote/ls/dev-speaker.flac",
                "phonemes": ["HH"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="overlap by speaker_id"):
        materialize_adapter_mixtures(
            manifest_dir=manifest_dir,
            librispeech_train_manifest=libri_train,
            librispeech_dev_manifest=libri_dev,
            train_size=20,
            dev_size=10,
        )

    libri_train.write_text(
        json.dumps(
            {
                "utt_id": "custom-train-id",
                "wav_path": "/remote/ls/custom-train.flac",
                "phonemes": ["HH"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    libri_dev.write_text(
        json.dumps(
            {
                "utt_id": "custom-dev-id",
                "wav_path": "/remote/ls/custom-dev.flac",
                "phonemes": ["HH"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="must provide speaker_id"):
        materialize_adapter_mixtures(
            manifest_dir=manifest_dir,
            librispeech_train_manifest=libri_train,
            librispeech_dev_manifest=libri_dev,
            train_size=20,
            dev_size=10,
        )


def test_hard_negative_pool_excludes_keyword_and_nontrain_speakers():
    base = {
        "wav_path": "/remote/sample.wav",
        "speaker_id": "speaker",
        "utt_id": "utt",
    }
    records = [
        {**base, "source": "speechocean762", "split": "train", "text": "hello world"},
        {**base, "utt_id": "dev", "source": "speechocean762", "split": "dev", "text": "other"},
        {
            **base,
            "utt_id": "real-neg",
            "source": "hey_eva_real_pc",
            "split": "train",
            "text": "Hey Ava",
            "label": 0,
        },
        {
            **base,
            "utt_id": "real-pos",
            "source": "hey_eva_real_pc",
            "split": "train",
            "text": "Hey Eva",
            "label": 1,
        },
    ]

    candidates = hard_negative_candidate_records(records)

    assert {candidate["utt_id"] for candidate in candidates} == {"utt", "real-neg"}
    assert all(candidate["label"] == 0 and candidate["split"] == "train" for candidate in candidates)
