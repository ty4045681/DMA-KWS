from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

pytest.importorskip("pytorch_lightning")

from dma_kws.stage1.wenet_ctc import (
    Stage1LightningModule,
    encode_manifest_target,
    phonemes_to_g2p_string,
    stage1_collate_fn,
)
from dma_kws.tokenizer import load_char_tokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
DICT_PATH = REPO_ROOT / "data" / "dict" / "lang_char.txt"


def _minimal_config() -> dict:
    return {
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
            "learning_rate": 1e-3,
            "warmup_steps": 2,
            "total_scheduler_steps": 10,
        },
    }


class _FakeCTC(nn.Module):
    def __init__(self, odim: int, encoder_output_size: int, **kwargs):
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(1))

    def forward(self, encoder_out, encoder_lens, targets, target_lengths):
        return self.dummy.sum() * 0 + encoder_out.mean(), None

    def log_softmax(self, encoder_out):
        batch, time, vocab = encoder_out.size(0), encoder_out.size(1), 73
        return torch.zeros(batch, time, vocab)


def _mock_encoder_output(feat: torch.Tensor, feat_lengths: torch.Tensor):
    encoded = torch.randn(feat.size(0), feat.size(1), 144)
    mask = torch.arange(feat.size(1)).unsqueeze(0) < feat_lengths.unsqueeze(1)
    return encoded, mask.unsqueeze(1)


@pytest.fixture
def patched_module(monkeypatch):
    fake_encoder = MagicMock(side_effect=_mock_encoder_output)
    monkeypatch.setattr("dma_kws.stage1.module.build_encoder", lambda *_args, **_kwargs: fake_encoder)
    monkeypatch.setattr("dma_kws.stage1.module._load_ctc", lambda: _FakeCTC)
    return Stage1LightningModule(_minimal_config(), vocab_size=73)


def test_phonemes_to_g2p_string_and_encode_manifest_target():
    tok = load_char_tokenizer(DICT_PATH)
    assert phonemes_to_g2p_string(["HH", "AH", "L", "OW"]) == "HH AH L OW"

    list_ids = encode_manifest_target({"phonemes": ["HH", "AH", "L", "OW"]}, tok)
    str_ids = encode_manifest_target({"phonemes_g2p": "HH AH L OW"}, tok)
    assert list_ids == str_ids
    assert all(isinstance(token_id, int) for token_id in list_ids)


def test_forward_and_training_step_smoke(patched_module):
    module = patched_module
    module.eval()

    batch = stage1_collate_fn(
        [
            {"feat": torch.randn(4, 80), "target": torch.tensor([17, 4, 22, 26], dtype=torch.long)},
            {"feat": torch.randn(6, 80), "target": torch.tensor([17, 4], dtype=torch.long)},
        ]
    )

    encoder_out, encoder_lens = module(batch["feats"], batch["feat_lengths"])
    assert encoder_out.shape[0] == 2
    assert encoder_lens.shape == (2,)

    fake_optimizer = MagicMock()
    fake_optimizer.param_groups = [{"lr": 1e-3}]
    module.optimizers = MagicMock(return_value=fake_optimizer)

    module.train()
    loss = module.training_step(batch, 0)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_export_stage1_encoder_pt_roundtrip(tmp_path, monkeypatch):
    pytest.importorskip("pytorch_lightning")

    from dma_kws.stage2.module import Stage2LightningModule

    fake_encoder = MagicMock()
    fake_encoder.parameters.return_value = []

    class _FakeCTC(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.dummy = nn.Parameter(torch.zeros(1))

    monkeypatch.setattr("dma_kws.stage1.module.build_encoder", lambda *_a, **_k: fake_encoder)
    monkeypatch.setattr("dma_kws.stage1.module._load_ctc", lambda: _FakeCTC)

    from dma_kws.stage1.wenet_ctc import Stage1LightningModule, export_stage1_encoder_pt

    config = _minimal_config()
    module = Stage1LightningModule(config, vocab_size=73)
    out = export_stage1_encoder_pt(
        module,
        tmp_path / "stage1.pt",
        config=config,
        dict_path=DICT_PATH,
        vocab_size=73,
        blank_id=0,
        step=42,
    )

    stage2 = Stage2LightningModule({"stage1": config["stage1"], "stage2": {}}, vocab_size=73)
    stage2._load_init_checkpoint(out)
    assert any(p.requires_grad for p in stage2.qbyt.parameters())


def test_validation_step_accumulates_per(patched_module):
    module = patched_module
    module.on_validation_epoch_start()

    batch = stage1_collate_fn(
        [{"feat": torch.randn(3, 80), "target": torch.tensor([17, 4, 22], dtype=torch.long)}]
    )
    module.validation_step(batch, 0)
    module.on_validation_epoch_end()

    assert module._val_total_ref > 0
