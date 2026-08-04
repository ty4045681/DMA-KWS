from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dma_kws.stage2.train import _build_val_dataloader


def _mock_eval_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "anchor_text": ["hello", "hello"],
            "anchor": ["a.wav", "b.wav"],
            "anchor_dur": [1.0, 1.0],
            "comparison_text": ["hello", "world"],
            "comparison": ["pos.wav", "hard.wav"],
            "comparison_dur": [1.0, 1.0],
            "target": [1, 0],
            "type": ["diffspk_positive", "diffspk_hardneg"],
        }
    )


class _FakeTokenizer:
    def tokenize(self, text: str):
        tokens = text.split()
        ids = [hash(token) % 100 + 1 for token in tokens]
        return tokens, ids


@pytest.fixture
def mock_eval_npy(monkeypatch):
    fbank = np.ones((4, 80), dtype=np.float32)

    def fake_load(path, allow_pickle=False):
        if str(path).endswith(".npy"):
            return fbank
        raise FileNotFoundError(path)

    monkeypatch.setattr("dma_kws.stage2.dataset.np.load", fake_load)
    return fbank


def _fake_g2p(text: str):
    """Mimic g2p_en: upper-case ARPAbet with stress digits on vowels."""
    phones = {"hello": ["HH", "AH0", "L", "OW1"], "world": ["W", "ER1", "L", "D"]}
    return phones.get(text, text.upper().split())


@pytest.fixture
def mock_g2p(monkeypatch):
    monkeypatch.setattr("dma_kws.stage2.dataset.make_g2p", lambda: _fake_g2p)


def _write_eval_aggregate(test_dir: Path) -> None:
    aggregate = test_dir / "evaluation_set" / "test_all_phrase.csv"
    aggregate.parent.mkdir(parents=True, exist_ok=True)
    _mock_eval_df().to_csv(aggregate, index=False)


def _base_config(test_dir: Path) -> dict:
    return {
        "paths": {},
        "tokenizer": {"dict_path": "data/dict/lang_char.txt", "split_with_space": " "},
        "stage2": {
            "eval": {
                "test_dir": str(test_dir),
                "split": "hard",
                "aggregate_csv": "evaluation_set/test_all_phrase.csv",
                "batch_size": 2,
                "num_workers": 0,
            },
        },
    }


def test_build_val_dataloader_uses_test_collate_fn(mock_eval_npy, mock_g2p, tmp_path):
    test_dir = tmp_path / "eval"
    test_dir.mkdir()
    _write_eval_aggregate(test_dir)

    dataloader = _build_val_dataloader(_base_config(test_dir), _FakeTokenizer())
    batch = next(iter(dataloader))

    assert dataloader.drop_last is False
    assert set(batch.keys()) == {
        "sample_id",
        "anchor",
        "feat",
        "feat_lengths",
        "label",
    }
    assert "seq_label" not in batch
    assert batch["label"].tolist() == [1, 0]


def test_build_val_dataloader_missing_test_dir_raises_system_exit(tmp_path):
    missing_dir = tmp_path / "missing_eval"
    config = _base_config(missing_dir)

    with pytest.raises(SystemExit, match="LibriPhrase eval data is required"):
        _build_val_dataloader(config, _FakeTokenizer())


def test_build_val_dataloader_unresolved_test_dir_raises_system_exit():
    config = {
        "paths": {},
        "tokenizer": {"dict_path": "data/dict/lang_char.txt"},
        "stage2": {"eval": {}},
    }

    with pytest.raises(SystemExit, match="LibriPhrase eval data is required"):
        _build_val_dataloader(config, _FakeTokenizer())


def test_build_val_dataloader_empty_split_raises_system_exit(mock_eval_npy, mock_g2p, tmp_path):
    test_dir = tmp_path / "eval"
    test_dir.mkdir()
    aggregate = test_dir / "evaluation_set" / "test_all_phrase.csv"
    aggregate.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "anchor_text": ["hello"],
            "anchor": ["a.wav"],
            "anchor_dur": [1.0],
            "comparison_text": ["world"],
            "comparison": ["hard.wav"],
            "comparison_dur": [1.0],
            "target": [0],
            "type": ["diffspk_hardneg"],
        }
    ).to_csv(aggregate, index=False)

    config = _base_config(test_dir)
    config["stage2"]["eval"]["split"] = "easy"

    with pytest.raises(SystemExit, match="Eval split 'easy' is empty"):
        _build_val_dataloader(config, _FakeTokenizer())
