from dma_kws.stage1 import librispeech
from dma_kws.stage1.librispeech import iter_librispeech_utterances, strip_stress_marker


def test_iter_librispeech_utterances_parses_transcript_and_audio_path(tmp_path):
    speaker_dir = tmp_path / "LibriSpeech" / "train-clean-100" / "19" / "198"
    speaker_dir.mkdir(parents=True)
    (speaker_dir / "19-198-0001.flac").write_bytes(b"fake")
    (speaker_dir / "19-198.trans.txt").write_text("19-198-0001 HELLO WORLD\n", encoding="utf-8")

    utterances = list(iter_librispeech_utterances(tmp_path / "LibriSpeech", ["train-clean-100"]))

    assert len(utterances) == 1
    assert utterances[0].utt_id == "19-198-0001"
    assert utterances[0].text == "HELLO WORLD"
    assert utterances[0].audio_path == speaker_dir / "19-198-0001.flac"
    assert strip_stress_marker("AH0") == "AH"


def test_iter_librispeech_parquet_utterances_reads_hf_audio_bytes(tmp_path, monkeypatch):
    parquet_dir = tmp_path / "clean" / "train.360"
    parquet_dir.mkdir(parents=True)
    parquet_file = parquet_dir / "0000.parquet"
    parquet_file.write_bytes(b"fake parquet placeholder")

    def fake_records(path):
        assert path == parquet_file
        yield {
            "id": "19-198-0001",
            "text": "HELLO WORLD",
            "speaker_id": 19,
            "chapter_id": 198,
            "file": "19/198/19-198-0001.flac",
            "audio": {"bytes": b"fake-flac-bytes", "path": "19-198-0001.flac"},
        }

    monkeypatch.setattr(librispeech, "_iter_parquet_records", fake_records)

    utterances = list(
        librispeech.iter_librispeech_parquet_utterances(
            parquet_dir,
            split="train-clean-360",
        )
    )

    assert len(utterances) == 1
    assert utterances[0].utt_id == "19-198-0001"
    assert utterances[0].text == "HELLO WORLD"
    assert utterances[0].split == "train-clean-360"
    assert utterances[0].speaker_id == "19"
    assert utterances[0].chapter_id == "198"
    assert utterances[0].audio_bytes == b"fake-flac-bytes"
    assert utterances[0].audio_path is None
    assert utterances[0].audio_extension == ".flac"
