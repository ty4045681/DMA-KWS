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
        g2p=lambda text: text.upper().split(),
    )

    assert len(dataset) == 2


def test_eval_dataset_getitem_keys_and_shapes(mock_eval_npy):
    dataset = LibriPhraseEvalDataset(
        test_dir="/data/eval",
        split="all",
        df=_mock_eval_df(),
        tokenizer=_FakeTokenizer(),
        g2p=lambda text: text.upper().split(),
    )

    sample = dataset[0]

    assert set(sample.keys()) == {"anchor_seq", "feat", "label"}
    assert sample["anchor_seq"].dtype == torch.long
    assert sample["feat"].shape == (4, 80)
    assert sample["label"].item() == 1


def test_test_collate_fn_shapes():
    batch = [
        {
            "anchor_seq": torch.tensor([1, 2, 3], dtype=torch.long),
            "feat": torch.ones(4, 80),
            "label": torch.tensor(1, dtype=torch.long),
        },
        {
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
    assert "seq_label" not in collated


def test_eval_metrics_smoke(monkeypatch, mock_eval_npy):
    pytest.importorskip("pytorch_lightning")

    from dma_kws.stage2.module import Stage2LightningModule

    def fake_g2p(text: str):
        return text.upper().split()

    dataset = LibriPhraseEvalDataset(
        test_dir="/data/eval",
        split="all",
        df=_mock_eval_df(),
        tokenizer=_FakeTokenizer(),
        g2p=fake_g2p,
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
    monkeypatch.setattr("dma_kws.stage2.module._load_qbyt", lambda: _FakeQbyT)

    module = Stage2LightningModule(config, vocab_size=73)
    module.eval()

    batch = test_collate_fn([dataset[i] for i in range(len(dataset))])
    with torch.no_grad():
        module.test_step(batch, 0)

    auc = float(module.auc_metric.compute())
    eer = float(module.eer_metric.compute())
    assert 0.0 <= auc <= 1.0
    assert 0.0 <= eer <= 1.0
