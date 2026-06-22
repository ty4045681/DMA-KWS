from pathlib import Path

import numpy as np
import pytest

from dma_kws.stage2.pairs import PairRecord
from scripts import prepare_stage2_libriphrase as prep


class _FakeDF:
    def __init__(self, rows):
        self._rows = rows

    def iterrows(self):
        for i, row in enumerate(self._rows):
            yield i, row


def test_materialize_pairs_writes_npy_and_rewrites_paths(tmp_path):
    audio_dir = tmp_path / "stage2_qbyt" / "audio"
    pairs = [
        PairRecord(
            anchor_text="hi",
            anchor_phonemes=["HH", "AY"],
            wav_path="LP-100/hi/1-2-3_000.wav",
            label=1,
            sample_rate=16000,
        ),
        PairRecord(
            anchor_text="hi",
            anchor_phonemes=["HH", "AY"],
            wav_path="LP-100/bye/4-5-6_000.wav",
            label=0,
            sample_rate=16000,
        ),
    ]
    rows = [
        {"audio_rel": "hi/1-2-3_000.wav", "audio": [0.1, 0.2, 0.3], "sampling_rate": 16000},
        {"audio_rel": "bye/4-5-6_000.wav", "audio": [0.4, 0.5], "sampling_rate": 16000},
    ]

    def fake_read(path):
        return _FakeDF(rows)

    rewritten, unmatched = prep.materialize_pairs(
        pairs,
        decoded_parquet_paths=[Path("0000.parquet")],
        audio_dir=audio_dir,
        read_parquet=fake_read,
    )

    assert unmatched == 0
    assert rewritten[0].wav_path == "audio/hi/1-2-3_000.wav.npy"
    saved = np.load(audio_dir / "hi" / "1-2-3_000.wav.npy")
    assert saved.dtype == np.float32
    np.testing.assert_allclose(saved, np.array([0.1, 0.2, 0.3], dtype=np.float32))


def test_materialize_pairs_counts_unmatched(tmp_path):
    audio_dir = tmp_path / "stage2_qbyt" / "audio"
    pairs = [
        PairRecord(
            anchor_text="hi",
            anchor_phonemes=["HH", "AY"],
            wav_path="LP-100/missing/9-9-9_000.wav",
            label=1,
            sample_rate=16000,
        )
    ]

    def fake_read(path):
        return _FakeDF([])

    rewritten, unmatched = prep.materialize_pairs(
        pairs,
        decoded_parquet_paths=[Path("0000.parquet")],
        audio_dir=audio_dir,
        read_parquet=fake_read,
    )

    assert unmatched == 1
    assert rewritten == []


def test_find_decoded_parquets_raises_when_none(tmp_path):
    with pytest.raises(SystemExit):
        prep.find_decoded_parquets(tmp_path)
