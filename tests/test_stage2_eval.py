from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
import torch

from dma_kws.stage2.collate import test_collate_fn
from dma_kws.stage2.dataset import (
    LibriPhraseEvalDataset,
    _filter_eval_split,
    _resolve_eval_fbank_path,
)


def _mock_eval_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "anchor_text": ["hello", "hello", "hello"],
            "anchor": ["a.wav", "b.wav", "c.wav"],
            "anchor_dur": [1.0, 1.0, 1.0],
            "comparison_text": ["hello", "world", "hello"],
            "comparison": ["pos.wav", "easy.wav", "hard.wav"],
            "comparison_dur": [1.0, 1.0, 1.0],
            "target": [1, 0, 0],
            "type": ["diffspk_positive", "diffspk_easyneg", "diffspk_hardneg"],
        }
    )


class _FakeTokenizer:
    def tokenize(self, text: str):
        tokens = text.split()
        ids = [hash(token) % 100 + 1 for token in tokens]
        return tokens, ids


def _fake_g2p(text: str):
    """Mimic g2p_en: upper-case ARPAbet with stress digits on vowels."""
    phones = {"hello": ["HH", "AH0", "L", "OW1"], "world": ["W", "ER1", "L", "D"]}
    return phones.get(text, text.upper().split())


@pytest.fixture
def mock_eval_npy(monkeypatch):
    fbank = np.ones((4, 80), dtype=np.float32)

    def fake_load(path, allow_pickle=False):
        if str(path).endswith(".npy"):
            return fbank
        raise FileNotFoundError(path)

    monkeypatch.setattr("dma_kws.stage2.dataset.np.load", fake_load)
    return fbank


def test_resolve_eval_fbank_path_replaces_extension():
    path = _resolve_eval_fbank_path("/data/eval", "clips/sample.wav")
    assert path == "/data/eval/clips/sample.npy"


def test_resolve_eval_fbank_path_uses_independent_root():
    path = _resolve_eval_fbank_path(
        "/data/eval",
        "clips/sample.wav",
        fbank_dir="/features/eval",
    )
    assert path == "/features/eval/clips/sample.npy"


def test_filter_eval_split_easy_and_hard():
    df = _mock_eval_df()

    easy = _filter_eval_split(df, "easy")
    hard = _filter_eval_split(df, "hard")
    all_rows = _filter_eval_split(df, "all")

    assert set(easy["type"]) == {"diffspk_positive", "diffspk_easyneg"}
    assert set(hard["type"]) == {"diffspk_positive", "diffspk_hardneg"}
    assert len(all_rows) == 3


def test_eval_dataset_easy_split_length(mock_eval_npy):
    dataset = LibriPhraseEvalDataset(
        test_dir="/data/eval",
        split="easy",
        df=_mock_eval_df(),
        tokenizer=_FakeTokenizer(),
        g2p=_fake_g2p,
    )

    assert len(dataset) == 2


def test_eval_dataset_getitem_keys_and_shapes(mock_eval_npy):
    dataset = LibriPhraseEvalDataset(
        test_dir="/data/eval",
        split="all",
        df=_mock_eval_df(),
        tokenizer=_FakeTokenizer(),
        g2p=_fake_g2p,
    )

    sample = dataset[0]

    assert set(sample.keys()) == {"sample_id", "anchor_seq", "feat", "label"}
    assert sample["sample_id"].item() == 0
    assert sample["anchor_seq"].dtype == torch.long
    assert sample["feat"].shape == (4, 80)
    assert sample["label"].item() == 1


def test_test_collate_fn_shapes():
    batch = [
        {
            "sample_id": torch.tensor(10),
            "anchor_seq": torch.tensor([1, 2, 3], dtype=torch.long),
            "feat": torch.ones(4, 80),
            "label": torch.tensor(1, dtype=torch.long),
        },
        {
            "sample_id": torch.tensor(11),
            "anchor_seq": torch.tensor([4, 5], dtype=torch.long),
            "feat": torch.ones(6, 80),
            "label": torch.tensor(0, dtype=torch.long),
        },
    ]

    collated = test_collate_fn(batch)

    assert collated["feat"].shape == (2, 6, 80)
    assert collated["anchor"].shape == (2, 3)
    assert collated["feat_lengths"].tolist() == [4, 6]
    assert collated["label"].tolist() == [1, 0]
    assert collated["sample_id"].tolist() == [10, 11]
    assert "seq_label" not in collated


def test_eval_metrics_smoke(monkeypatch, mock_eval_npy):
    pytest.importorskip("pytorch_lightning")

    from dma_kws.stage2.module import Stage2LightningModule

    dataset = LibriPhraseEvalDataset(
        test_dir="/data/eval",
        split="all",
        df=_mock_eval_df(),
        tokenizer=_FakeTokenizer(),
        g2p=_fake_g2p,
    )

    def _mock_encoder_output(feat: torch.Tensor, feat_lengths: torch.Tensor):
        encoded = torch.randn(feat.size(0), feat.size(1), 144)
        mask = torch.arange(feat.size(1)).unsqueeze(0) < feat_lengths.unsqueeze(1)
        return encoded, mask.unsqueeze(1)

    fake_encoder = MagicMock(side_effect=_mock_encoder_output)

    class _FakeQbyT(torch.nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            self.dummy = torch.nn.Parameter(torch.zeros(1))

        def forward(self, speech, text, speech_lengths=None, text_lengths=None):
            batch_size = speech.size(0)
            anchor_len = text.size(1)
            logits = self.dummy.expand(batch_size) + text.float().mean(dim=1)
            seq_logits = self.dummy.expand(batch_size, anchor_len)
            return logits, seq_logits

    config = {
        "stage1": {
            "input_dim": 80,
            "encoder_output_dim": 144,
            "attention_heads": 4,
            "linear_units": 576,
            "num_blocks": 2,
            "dropout_rate": 0.1,
            "positional_dropout_rate": 0.1,
            "attention_dropout_rate": 0.0,
            "cnn_module_kernel": 3,
        },
        "stage2": {
            "encoder_output_dim": 144,
            "qbyt_embed_dim": 128,
            "qbyt_layers": 2,
        },
    }

    monkeypatch.setattr("dma_kws.stage2.module.build_encoder", lambda *_args, **_kwargs: fake_encoder)
    monkeypatch.setattr(
        "dma_kws.stage2.module.build_qbyt",
        lambda *_args, **_kwargs: _FakeQbyT(),
    )

    module = Stage2LightningModule(config, vocab_size=71)
    module.eval()

    batch = test_collate_fn([dataset[i] for i in range(len(dataset))])
    with torch.no_grad():
        module.test_step(batch, 0)

    diagnostics = module.score_diagnostics.compute()
    auc = float(diagnostics["auc"])
    eer = float(diagnostics["eer"])
    assert 0.0 <= auc <= 1.0
    assert 0.0 <= eer <= 1.0
