import json

from dma_kws.stage1.librispeech import ParquetAudioUtterance
from scripts import prepare_stage1_librispeech as prepare_stage1


class FakeG2P:
    def __call__(self, text):
        assert text == "hello world"
        return ["HH", "AH0", "L", "OW1"]


def test_prepare_parquet_split_writes_manifest_and_audio_cache(tmp_path, monkeypatch):
    parquet_root = tmp_path / "parquet"
    parquet_root.mkdir()
    output_path = tmp_path / "processed" / "train.jsonl"
    audio_output_dir = tmp_path / "processed" / "audio" / "train-clean-360"
    utterance = ParquetAudioUtterance(
        utt_id="19-198-0001",
        text="HELLO WORLD",
        split="train-clean-360",
        speaker_id="19",
        chapter_id="198",
        audio_bytes=b"fake-flac-bytes",
        audio_path=None,
        audio_extension=".flac",
    )

    monkeypatch.setattr(
        prepare_stage1,
        "iter_librispeech_parquet_utterances",
        lambda root, split: iter([utterance]),
    )

    phones = prepare_stage1.prepare_parquet_split(
        g2p=FakeG2P(),
        parquet_root=parquet_root,
        split="train-clean-360",
        output_path=output_path,
        audio_output_dir=audio_output_dir,
        limit=0,
    )

    extracted_audio = audio_output_dir / "19" / "198" / "19-198-0001.flac"
    assert extracted_audio.read_bytes() == b"fake-flac-bytes"
    assert phones == ["HH", "AH", "L", "OW"]
    records = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert records == [
        {
            "utt_id": "19-198-0001",
            "split": "train-clean-360",
            "wav_path": str(extracted_audio),
            "text": "HELLO WORLD",
            "normalized_text": "hello world",
            "phonemes": ["HH", "AH", "L", "OW"],
            "phonemes_g2p": "HH AH L OW",
        }
    ]
