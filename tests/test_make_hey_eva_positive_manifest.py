from __future__ import annotations

import csv

from dma_kws.inference.manifest import load_manifest
from scripts.make_hey_eva_positive_manifest import (
    KEYWORD_PHONEMES,
    write_manifest,
)


def test_writes_sibling_csv_with_relative_positive_rows(tmp_path):
    audio_dir = tmp_path / "positive_clips"
    nested_dir = audio_dir / "nested"
    nested_dir.mkdir(parents=True)
    (audio_dir / "b.WAV").touch()
    (nested_dir / "a.flac").touch()
    (audio_dir / "ignored.txt").touch()

    output_path, count = write_manifest(audio_dir)

    assert output_path == tmp_path / "positive_clips.csv"
    assert count == 2
    with output_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert list(rows[0]) == [
        "audio_path",
        "keyword",
        "label",
        "keyword_phonemes",
    ]
    assert {row["audio_path"] for row in rows} == {
        "positive_clips/b.WAV",
        "positive_clips/nested/a.flac",
    }
    assert all(row["keyword"] == "hey eva" for row in rows)
    assert all(row["label"] == "1" for row in rows)
    assert all(row["keyword_phonemes"] == KEYWORD_PHONEMES for row in rows)

    loaded_rows = load_manifest(output_path)
    assert {row["audio_path"] for row in loaded_rows} == {
        str((audio_dir / "b.WAV").resolve()),
        str((nested_dir / "a.flac").resolve()),
    }
    assert all(row["label"] == 1 for row in loaded_rows)
