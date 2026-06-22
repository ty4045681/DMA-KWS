from pathlib import Path

from dma_kws.stage2.pairs import AnchorExample, PairRecord, clip_to_audio_rel, iter_decoded_audio_rows, make_pair_records


def test_clip_to_audio_rel_strips_lp100_prefix():
    assert clip_to_audio_rel("LP-100/missus_rachel/103-1240-0000_000.wav") == "missus_rachel/103-1240-0000_000.wav"


def test_clip_to_audio_rel_without_prefix_is_identity():
    assert clip_to_audio_rel("missus_rachel/103-1240-0000_000.wav") == "missus_rachel/103-1240-0000_000.wav"


def test_pair_record_carries_sample_rate():
    record = PairRecord(
        anchor_text="hi",
        anchor_phonemes=["HH", "AY"],
        wav_path="audio/a.npy",
        label=1,
        sample_rate=16000,
    )
    assert record.to_json_dict()["sample_rate"] == 16000


def test_make_pair_records_creates_positive_and_negative_pair():
    anchors = [
        AnchorExample(text="hello", phonemes=["HH", "AH", "L", "OW"], clips=["a.wav"]),
        AnchorExample(text="world", phonemes=["W", "ER", "L", "D"], clips=["b.wav"]),
    ]

    pairs = make_pair_records(anchors, negatives_per_anchor=1, seed=7)

    assert len(pairs) == 4
    assert pairs[0].label == 1
    assert pairs[0].wav_path == "a.wav"
    assert any(pair.label == 0 and pair.anchor_text == "hello" and pair.wav_path == "b.wav" for pair in pairs)


class _FakeDF:
    def __init__(self, rows):
        self._rows = rows

    def iterrows(self):
        for i, row in enumerate(self._rows):
            yield i, row


def test_iter_decoded_audio_rows_filters_to_needed_keys():
    shard_rows = {
        Path("0000.parquet"): [
            {"audio_rel": "a/x.wav", "audio": [0.1, 0.2], "sampling_rate": 16000},
            {"audio_rel": "a/y.wav", "audio": [0.3], "sampling_rate": 16000},
        ],
        Path("0001.parquet"): [
            {"audio_rel": "b/z.wav", "audio": [0.4, 0.5], "sampling_rate": 16000},
        ],
    }

    def fake_read(path):
        return _FakeDF(shard_rows[path])

    needed = {"a/x.wav", "b/z.wav"}
    got = list(
        iter_decoded_audio_rows(
            [Path("0000.parquet"), Path("0001.parquet")],
            needed,
            read_parquet=fake_read,
        )
    )

    assert [(rel, sr) for rel, _, sr in got] == [("a/x.wav", 16000), ("b/z.wav", 16000)]
    assert got[0][1] == [0.1, 0.2]
