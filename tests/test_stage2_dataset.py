from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from dma_kws.stage2.collate import train_collate_fn
from dma_kws.stage2.dataset import LibriPhraseTrainDataset, _resolve_fbank_path
from dma_kws.tokenizer import load_char_tokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
DICT_PATH = REPO_ROOT / "data" / "dict" / "lang_char.txt"


def _mock_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ngram": ["hello", "world"],
            "ngram_g2p": ["HH AH0 L OW1", "W ER1 L D"],
            "clips_file": ["clips-2-a.npy", "clips-2-b.npy"],
            "distances_file": ["dist-0-a.npy", "dist-2-b.npy"],
        }
    )


class _FakeTokenizer:
    def tokenize(self, text: str):
        tokens = text.split()
        ids = [hash(token) % 100 + 1 for token in tokens]
        return tokens, ids


@pytest.fixture
def mock_npy_loader(monkeypatch):
    clips = {
        "clips-2-a.npy": np.array(
            [{"audio_path": "LP-460/hello/a.wav"}, {"audio_path": "LP-460/hello/b.wav"}],
            dtype=object,
        ),
        "clips-2-b.npy": np.array(
            [{"audio_path": "LP-460/world/c.wav"}, {"audio_path": "LP-460/world/d.wav"}],
            dtype=object,
        ),
    }
    distances = {
        "dist-0-a.npy": np.array([], dtype=object),
        "dist-2-b.npy": np.array([{"ngram": "hello"}, {"ngram": "hello"}], dtype=object),
    }
    fbank = np.ones((5, 80), dtype=np.float32)

    def fake_load(path, allow_pickle=False):
        name = Path(path).name
        if name in clips:
            return clips[name]
        if name in distances:
            return distances[name]
        if name.endswith(".npy") and "fbank" in str(path):
            return fbank
        raise FileNotFoundError(path)

    monkeypatch.setattr("dma_kws.stage2.dataset.np.load", fake_load)
    return fbank


def test_resolve_fbank_path_replaces_prefix_and_extension():
    path = _resolve_fbank_path("/data/segments", "LP-460/hello/a.wav")
    assert path == "/data/segments/LP-460-fbank/hello/a.npy"

    path = _resolve_fbank_path("/data/segments", "GP-1000/world/b.wav")
    assert path == "/data/segments/GP-1000-fbank/world/b.npy"


def test_dataset_getitem_returns_expected_keys_and_seq_label_length(mock_npy_loader):
    dataset = LibriPhraseTrainDataset(
        wav_dir="/data/segments",
        tokenizer=_FakeTokenizer(),
        df=_mock_dataframe(),
        sample_lens=4,
        seed=0,
    )

    sample = dataset[0]

    assert set(sample.keys()) == {"anchor_seq", "feat", "label", "seq_label"}
    assert sample["anchor_seq"].dtype == torch.long
    assert sample["feat"].shape == (5, 80)
    assert sample["label"].dtype == torch.long
    assert sample["seq_label"].dtype == torch.long
    assert sample["seq_label"].numel() == sample["anchor_seq"].numel()


def test_dataset_positive_sample_has_matching_seq_label(mock_npy_loader):
    dataset = LibriPhraseTrainDataset(
        wav_dir="/data/segments",
        tokenizer=_FakeTokenizer(),
        df=_mock_dataframe(),
        sample_lens=1,
        seed=1,
    )

    sample = dataset[0]

    assert sample["label"].item() == 1
    assert torch.all(sample["seq_label"] == 1)


def test_dataset_collate_batch_shapes(mock_npy_loader):
    dataset = LibriPhraseTrainDataset(
        wav_dir="/data/segments",
        tokenizer=_FakeTokenizer(),
        df=_mock_dataframe(),
        sample_lens=4,
        seed=0,
    )
    batch = [dataset[i] for i in range(2)]
    collated = train_collate_fn(batch)

    assert collated["anchor"].shape[0] == 2
    assert collated["feat"].shape[0] == 2
    assert collated["feat_lengths"].shape == (2,)
    assert collated["label"].shape == (2,)
    assert collated["seq_label"].shape[0] == 2
    assert collated["seq_label_mask"].shape == collated["seq_label"].shape


def test_dataset_loads_tokenizer_from_dict_path(mock_npy_loader):
    dataset = LibriPhraseTrainDataset(
        wav_dir="/data/segments",
        dict_path=DICT_PATH,
        df=_mock_dataframe(),
        sample_lens=1,
        seed=0,
    )

    sample = dataset[0]
    anchor_ids = load_char_tokenizer(DICT_PATH)
    _, expected = anchor_ids.tokenize("HH AH0 L OW1")

    assert sample["anchor_seq"].tolist() == expected
