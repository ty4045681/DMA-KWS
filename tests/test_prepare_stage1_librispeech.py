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
    (parquet_root / "0000.parquet").write_bytes(b"placeholder")
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
    assert phones == ["HH", "AH0", "L", "OW1"]
    records = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert records == [
        {
            "utt_id": "19-198-0001",
            "split": "train-clean-360",
            "wav_path": str(extracted_audio),
            "text": "HELLO WORLD",
            "normalized_text": "hello world",
            "phonemes": ["HH", "AH0", "L", "OW1"],
            "phonemes_g2p": "HH AH0 L OW1",
        }
    ]


def test_prepare_parquet_split_parallel_merges_by_sorted_shard_name(tmp_path, monkeypatch):
    parquet_root = tmp_path / "parquet"
    parquet_root.mkdir()
    shard_b = parquet_root / "0002.parquet"
    shard_a = parquet_root / "0001.parquet"
    shard_b.write_bytes(b"placeholder")
    shard_a.write_bytes(b"placeholder")

    output_path = tmp_path / "processed" / "train.jsonl"
    audio_output_dir = tmp_path / "processed" / "audio" / "train-clean-360"

    utterances = {
        shard_a: [
            ParquetAudioUtterance(
                utt_id="a-utt",
                text="A",
                split="train-clean-360",
                speaker_id="1",
                chapter_id="1",
                audio_bytes=b"a",
                audio_path=None,
                audio_extension=".flac",
            )
        ],
        shard_b: [
            ParquetAudioUtterance(
                utt_id="b-utt",
                text="B",
                split="train-clean-360",
                speaker_id="2",
                chapter_id="2",
                audio_bytes=b"b",
                audio_path=None,
                audio_extension=".flac",
            )
        ],
    }

    class _FakeG2P:
        def __call__(self, text):
            return [text]

    class _FakeFuture:
        def __init__(self, value):
            self._value = value

        def result(self):
            return self._value

    class _FakeProcessPoolExecutor:
        def __init__(self, *, max_workers):
            self.max_workers = max_workers

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def submit(self, fn, task):
            return _FakeFuture(fn(task))

    monkeypatch.setattr(prepare_stage1, "make_g2p", lambda: _FakeG2P())
    monkeypatch.setattr(
        prepare_stage1,
        "iter_librispeech_parquet_shard_utterances",
        lambda shard_path, split: iter(utterances[shard_path]),
    )
    monkeypatch.setattr(prepare_stage1, "ProcessPoolExecutor", _FakeProcessPoolExecutor)
    monkeypatch.setattr(prepare_stage1, "as_completed", lambda futures: list(reversed(list(futures))))

    phones = prepare_stage1.prepare_parquet_split(
        g2p=_FakeG2P(),
        parquet_root=parquet_root,
        split="train-clean-360",
        output_path=output_path,
        audio_output_dir=audio_output_dir,
        limit=0,
        num_workers=2,
    )

    rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert [row["utt_id"] for row in rows] == ["a-utt", "b-utt"]
    assert phones == ["a", "b"]
