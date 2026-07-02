from pathlib import Path

import pytest

from dma_kws.inference.manifest import (
    build_manifest_rows,
    iter_audio_files,
    load_manifest,
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


def test_write_manifest_empty_rows(tmp_path):
    with pytest.raises(ValueError, match="empty manifest"):
        write_manifest(tmp_path / "manifest.csv", [])


def test_write_manifest_unknown_suffix(tmp_path):
    audio = _touch_audio(tmp_path, "a.wav")
    rows = build_manifest_rows([audio], "hey eva")

    with pytest.raises(ValueError, match="infer manifest format"):
        write_manifest(tmp_path / "manifest.txt", rows)
