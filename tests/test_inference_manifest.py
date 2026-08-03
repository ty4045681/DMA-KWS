from pathlib import Path

import pytest

from dma_kws.inference.manifest import (
    build_manifest_rows_by_filename,
    build_manifest_rows,
    filename_keyword_candidate,
    iter_audio_files,
    load_manifest,
    match_keyword_from_filename,
    normalize_keyword,
    write_manifest,
)


def _touch_audio(directory: Path, name: str) -> Path:
    path = directory / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")
    return path


def test_iter_audio_files_filters_and_sorts(tmp_path):
    _touch_audio(tmp_path, "b.wav")
    _touch_audio(tmp_path, "a.flac")
    _touch_audio(tmp_path, "notes.txt")
    _touch_audio(tmp_path, "nested/c.mp3")

    files = iter_audio_files(tmp_path)

    assert [p.name for p in files] == ["a.flac", "b.wav", "c.mp3"]


def test_iter_audio_files_non_recursive_skips_subdirs(tmp_path):
    _touch_audio(tmp_path, "a.wav")
    _touch_audio(tmp_path, "nested/b.wav")

    files = iter_audio_files(tmp_path, recursive=False)

    assert [p.name for p in files] == ["a.wav"]


def test_iter_audio_files_missing_dir(tmp_path):
    with pytest.raises(NotADirectoryError):
        iter_audio_files(tmp_path / "missing")


def test_build_write_load_roundtrip_relative(tmp_path):
    audio_dir = tmp_path / "audio"
    audio_a = _touch_audio(audio_dir, "a.wav")
    audio_b = _touch_audio(audio_dir, "b.wav")
    manifest_path = tmp_path / "manifest.csv"

    rows = build_manifest_rows(
        [audio_a, audio_b],
        "hello world",
        manifest_dir=manifest_path.parent,
    )
    assert rows[0]["audio_path"] == str(Path("audio") / "a.wav")
    assert "label" not in rows[0]

    write_manifest(manifest_path, rows)
    loaded = load_manifest(manifest_path)

    assert len(loaded) == 2
    assert loaded[0]["keyword"] == "hello world"
    assert Path(loaded[0]["audio_path"]) == audio_a.resolve()


def test_build_manifest_rows_falls_back_to_absolute_when_not_relative(tmp_path):
    audio_dir = tmp_path / "audio"
    audio_a = _touch_audio(audio_dir, "a.wav")
    manifest_path = tmp_path / "out" / "manifest.csv"

    rows = build_manifest_rows([audio_a], "hello world", manifest_dir=manifest_path.parent)

    assert Path(rows[0]["audio_path"]) == audio_a.resolve()


def test_build_manifest_rows_with_label(tmp_path):
    audio = _touch_audio(tmp_path, "a.wav")
    manifest_path = tmp_path / "manifest.csv"

    rows = build_manifest_rows([audio], "hey eva", label=1, manifest_dir=manifest_path.parent)
    write_manifest(manifest_path, rows)
    loaded = load_manifest(manifest_path)

    assert loaded[0]["label"] == 1
    assert manifest_path.read_text(encoding="utf-8").splitlines()[0] == "audio_path,keyword,label"


def test_build_manifest_rows_requires_keyword(tmp_path):
    audio = _touch_audio(tmp_path, "a.wav")

    with pytest.raises(ValueError, match="keyword is required"):
        build_manifest_rows([audio], "")


def test_write_manifest_jsonl_roundtrip(tmp_path):
    audio = _touch_audio(tmp_path, "a.wav")
    manifest_path = tmp_path / "manifest.jsonl"

    rows = build_manifest_rows([audio], "hey eva", label=0, manifest_dir=manifest_path.parent)
    write_manifest(manifest_path, rows)
    loaded = load_manifest(manifest_path)

    assert loaded[0]["keyword"] == "hey eva"
    assert loaded[0]["label"] == 0


def test_load_manifest_csv_invalid_label_reports_file_row(tmp_path):
    manifest_path = tmp_path / "manifest.csv"
    manifest_path.write_text(
        "audio_path,keyword,label\n"
        "a.wav,hey eva,1\n"
        "b.wav,hey eva,pos\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Manifest row 3 has invalid label 'pos'"):
        load_manifest(manifest_path)


def test_load_manifest_jsonl_invalid_label_reports_file_row(tmp_path):
    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.write_text(
        '{"audio_path":"a.wav","keyword":"hey eva","label":1}\n'
        '{"audio_path":"b.wav","keyword":"hey eva","label":"pos"}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Manifest row 2 has invalid label 'pos'"):
        load_manifest(manifest_path)


def test_write_manifest_empty_rows(tmp_path):
    with pytest.raises(ValueError, match="empty manifest"):
        write_manifest(tmp_path / "manifest.csv", [])


def test_write_manifest_unknown_suffix(tmp_path):
    audio = _touch_audio(tmp_path, "a.wav")
    rows = build_manifest_rows([audio], "hey eva")

    with pytest.raises(ValueError, match="infer manifest format"):
        write_manifest(tmp_path / "manifest.txt", rows)


def test_filename_keyword_candidate_rule():
    assert filename_keyword_candidate("hey_eva_001.wav") == "heyeva"
    assert filename_keyword_candidate("wake_word_neg_a.flac") == "wakewordneg"
    assert filename_keyword_candidate("single.wav") == ""


def test_normalize_keyword_removes_spaces_and_underscores():
    assert normalize_keyword("Hey Eva") == "heyeva"
    assert normalize_keyword("hey_eva") == "heyeva"
    assert normalize_keyword("Hi Lamp", casefold=False) == "HiLamp"


def test_match_keyword_from_filename_casefold_and_spacing():
    matched = match_keyword_from_filename("HEY_EVA_002.wav", ["hey eva", "ok lamp"])
    assert matched == "hey eva"


def test_build_manifest_rows_by_filename_assigns_labels(tmp_path):
    audio_dir = tmp_path / "audio"
    audio_a = _touch_audio(audio_dir, "hey_eva_001.wav")
    audio_b = _touch_audio(audio_dir, "ok_lamp_010.wav")
    manifest_path = tmp_path / "manifest.csv"

    rows, unmatched = build_manifest_rows_by_filename(
        [audio_a, audio_b],
        keywords=["hey eva", "ok lamp"],
        keyword_labels={"hey eva": 1, "ok lamp": 0},
        manifest_dir=manifest_path.parent,
    )

    assert unmatched == []
    assert len(rows) == 2
    assert rows[0]["keyword"] == "hey eva"
    assert rows[0]["label"] == 1
    assert rows[1]["keyword"] == "ok lamp"
    assert rows[1]["label"] == 0
    assert rows[0]["audio_path"] == str(Path("audio") / "hey_eva_001.wav")


def test_build_manifest_rows_by_filename_skips_unmatched(tmp_path):
    audio_dir = tmp_path / "audio"
    audio_a = _touch_audio(audio_dir, "hey_eva_001.wav")
    audio_b = _touch_audio(audio_dir, "unknown_001.wav")

    rows, unmatched = build_manifest_rows_by_filename(
        [audio_a, audio_b],
        keywords=["hey eva"],
        keyword_labels={"hey eva": 1},
        skip_unmatched=True,
    )

    assert len(rows) == 1
    assert len(unmatched) == 1
    assert unmatched[0].endswith("unknown_001.wav")


def test_build_manifest_rows_by_filename_raises_without_labels(tmp_path):
    audio = _touch_audio(tmp_path, "hey_eva_001.wav")

    with pytest.raises(ValueError, match="keyword_labels is required"):
        build_manifest_rows_by_filename([audio], keywords=["hey eva"])


def test_build_manifest_rows_by_filename_raises_for_ambiguous_keywords(tmp_path):
    audio = _touch_audio(tmp_path, "hey_eva_001.wav")

    with pytest.raises(ValueError, match="Ambiguous keywords"):
        build_manifest_rows_by_filename(
            [audio],
            keywords=["hey eva", "hey_eva"],
            keyword_labels={"hey eva": 1, "hey_eva": 1},
        )
