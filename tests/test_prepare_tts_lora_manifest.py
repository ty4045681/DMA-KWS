import csv
from pathlib import Path

import pytest

from dma_kws.stage2.prepare_adapt import (
    load_manifest_csv,
    split_explicit_manifest,
    validate_manifest_speaker_splits,
)
from scripts.prepare_tts_lora_manifest import (
    _provider_from_audio_path,
    build_tts_lora_manifest,
)


FIELDS = [
    "audio_path",
    "keyword",
    "label",
    "voice_id",
    "voice_name",
    "text_variant",
    "sha256",
    "keyword_phonemes",
    "text_variant_phonemes",
]


def _write_source(
    tmp_path: Path,
    speakers: dict[str, list[tuple[int, str]]],
) -> Path:
    source = tmp_path / "tts_manifest.csv"
    rows = []
    index = 0
    for speaker_id, examples in speakers.items():
        provider_dir = (
            "elevenlabs_output" if speaker_id.startswith("eleven-") else "googletts_output"
        )
        for label, text in examples:
            relative = Path(provider_dir) / speaker_id / f"sample_{index}.wav"
            audio = tmp_path / relative
            audio.parent.mkdir(parents=True, exist_ok=True)
            audio.touch()
            rows.append(
                {
                    "audio_path": relative.as_posix(),
                    "keyword": "Hey Eva",
                    "label": label,
                    "voice_id": speaker_id,
                    "voice_name": f"voice-{speaker_id}",
                    "text_variant": text,
                    "sha256": f"{index:064x}",
                    "keyword_phonemes": "HH EY1 IY1 V AH0",
                    "text_variant_phonemes": (
                        "HH EY1 IY1 V AH0" if label else "HH EY1 AY1 V AH0 N"
                    ),
                }
            )
            index += 1
    with source.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return source


def _read_output(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_build_tts_lora_manifest_maps_text_and_isolates_speakers(tmp_path: Path):
    source = _write_source(
        tmp_path,
        {
            "eleven-a": [(1, "Hey Eva"), (0, "Hey Ivan")],
            "eleven-b": [(1, "Hey Eva"), (0, "Hey Ava")],
            "google-a": [(1, "Hey Eva"), (0, "Hey Evil")],
            "google-b": [(1, "Hey Eva"), (0, "Hey Ever")],
        },
    )
    output = tmp_path / "prepared" / "tts_lora_source.csv"

    summary = build_tts_lora_manifest(
        source,
        output,
        audio_root=tmp_path,
        eval_fraction=0.5,
        seed=7,
    )
    rows = _read_output(output)

    assert summary["exact_1_to_1"] is True
    assert summary["splits"]["train"]["rows"] == 4
    assert summary["splits"]["eval"]["rows"] == 4
    assert summary["speaker_overlap"] == []
    assert {row["phase"] for row in rows} == {"tts"}
    assert {row["split"] for row in rows} == {"train", "eval"}
    assert all(Path(row["audio_path"]).is_absolute() for row in rows)
    assert all(Path(row["audio_path"]).is_file() for row in rows)
    assert all(row["text"] == row["text_variant"] for row in rows)
    assert all(row["keyword_phonemes"] == "HH EY1 IY1 V AH0" for row in rows)

    splits_by_speaker: dict[str, set[str]] = {}
    labels_by_split: dict[str, set[str]] = {}
    for row in rows:
        splits_by_speaker.setdefault(row["speaker_id"], set()).add(row["split"])
        labels_by_split.setdefault(row["split"], set()).add(row["label"])
    assert all(len(splits) == 1 for splits in splits_by_speaker.values())
    assert labels_by_split == {"train": {"0", "1"}, "eval": {"0", "1"}}
    assert any(row["speaker_id"].startswith("elevenlabs:") for row in rows)
    assert any(row["speaker_id"].startswith("googletts:") for row in rows)

    prepared_rows = load_manifest_csv(output)
    validate_manifest_speaker_splits(prepared_rows)
    explicit = split_explicit_manifest(prepared_rows, phase="tts")
    assert explicit is not None
    assert tuple(map(len, explicit)) == (4, 4)


def test_speaker_isolation_reports_unavoidable_row_gap(tmp_path: Path):
    source = _write_source(
        tmp_path,
        {
            "eleven-a": [(1, "Hey Eva"), (0, "Hey Ivan"), (0, "Hey Ava")],
            "google-a": [(1, "Hey Eva"), (0, "Hey Evil"), (1, "Hey Eva")],
            "google-b": [(1, "Hey Eva"), (0, "Hey Ever")],
        },
    )
    output = tmp_path / "tts_lora_source.csv"

    summary = build_tts_lora_manifest(source, output, audio_root=tmp_path)

    assert summary["output_rows"] == 8
    assert summary["row_gap"] == 2
    assert summary["exact_1_to_1"] is False
    assert summary["speaker_overlap"] == []
    assert all(
        summary["splits"][split][label] > 0
        for split in ("train", "eval")
        for label in ("positive", "negative")
    )


def test_rejects_when_positive_class_exists_for_only_one_speaker(tmp_path: Path):
    source = _write_source(
        tmp_path,
        {
            "eleven-a": [(1, "Hey Eva"), (0, "Hey Ivan")],
            "google-a": [(0, "Hey Ava"), (0, "Hey Evil")],
        },
    )

    with pytest.raises(ValueError, match="label=1 occurs in only 1 speaker"):
        build_tts_lora_manifest(
            source,
            tmp_path / "tts_lora_source.csv",
            audio_root=tmp_path,
        )


def test_rejects_missing_audio_by_default(tmp_path: Path):
    source = _write_source(
        tmp_path,
        {
            "eleven-a": [(1, "Hey Eva"), (0, "Hey Ivan")],
            "google-a": [(1, "Hey Eva"), (0, "Hey Ava")],
        },
    )
    missing = tmp_path / "elevenlabs_output" / "eleven-a" / "sample_0.wav"
    missing.unlink()

    with pytest.raises(FileNotFoundError, match="audio file not found"):
        build_tts_lora_manifest(
            source,
            tmp_path / "tts_lora_source.csv",
            audio_root=tmp_path,
        )


def test_speaker_identity_is_case_insensitive(tmp_path: Path):
    source = _write_source(
        tmp_path,
        {
            "VoiceA": [(1, "Hey Eva"), (0, "Hey Ivan")],
            "voicea": [(1, "Hey Eva"), (0, "Hey Ava")],
            "VoiceB": [(1, "Hey Eva"), (0, "Hey Evil")],
            "voiceb": [(1, "Hey Eva"), (0, "Hey Ever")],
        },
    )
    output = tmp_path / "tts_lora_source.csv"

    summary = build_tts_lora_manifest(source, output, audio_root=tmp_path)
    rows = _read_output(output)

    assert summary["unique_speakers"] == 2
    assert {row["speaker_id"] for row in rows} == {
        "googletts:voicea",
        "googletts:voiceb",
    }
    validate_manifest_speaker_splits(load_manifest_csv(output))


def test_rejects_duplicate_audio_with_conflicting_phonemes(tmp_path: Path):
    source = _write_source(
        tmp_path,
        {
            "eleven-a": [
                (1, "Hey Eva"),
                (1, "Hey Eva"),
                (0, "Hey Ivan"),
            ],
            "google-a": [(1, "Hey Eva"), (0, "Hey Ava")],
        },
    )
    with source.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows[1]["sha256"] = rows[0]["sha256"]
    rows[1]["keyword_phonemes"] = "HH EH1 IY1 V AH0"
    with source.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(ValueError, match="duplicate audio key.*conflicts"):
        build_tts_lora_manifest(
            source,
            tmp_path / "tts_lora_source.csv",
            audio_root=tmp_path,
        )


def test_provider_detection_prefers_specific_nested_provider():
    assert (
        _provider_from_audio_path("tts_output/elevenlabs_output/voice/sample.wav")
        == "elevenlabs"
    )
    assert _provider_from_audio_path("root_output/acme_output/sample.wav") == "acme"
