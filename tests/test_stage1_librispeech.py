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
