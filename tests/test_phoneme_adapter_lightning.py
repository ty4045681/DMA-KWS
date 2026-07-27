"""Step A module: frozen encoder + trainable phoneme CTC adapter."""

from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

pytest.importorskip("pytorch_lightning")

from dma_kws.phoneme_adapter.lightning import PhonemeAdapterCtcModule

ENCODER_DIM = 64
TRUNK_DIM = 32
VOCAB_SIZE = 71


class _FakeEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(80, ENCODER_DIM)
        self.forward_modes: list[bool] = []

    def forward(self, feat, feat_lengths):
        self.forward_modes.append(self.training)
        encoded = self.proj(feat)
        mask = torch.arange(feat.size(1)).unsqueeze(0) < feat_lengths.unsqueeze(1)
        return encoded, mask.unsqueeze(1)


def _config() -> dict:
    return {
        "stage1": {"input_dim": 80, "encoder_output_dim": ENCODER_DIM, "causal": False},
        "phoneme_adapter": {
            "trunk": {"type": "conv", "output_dim": TRUNK_DIM, "num_layers": 1, "kernel_size": 3},
            "learning_rate": 1e-3,
            "warmup_steps": 2,
            "total_scheduler_steps": 10,
            "validation": {"num_decode_batches": 2},
        },
    }


def _batch(frames: int = 20) -> dict:
    return {
        "feats": torch.randn(2, frames, 80),
        "feat_lengths": torch.tensor([frames, frames - 5], dtype=torch.long),
        "targets": torch.tensor([[3, 4, 5], [6, 7, 0]], dtype=torch.long),
        "target_lengths": torch.tensor([3, 2], dtype=torch.long),
    }


@pytest.fixture
def module(monkeypatch):
    encoder = _FakeEncoder()
    monkeypatch.setattr("dma_kws.phoneme_adapter.lightning.build_encoder", lambda *_a, **_k: encoder)
    built = PhonemeAdapterCtcModule(_config(), vocab_size=VOCAB_SIZE)
    built.log = MagicMock()
    return built


def test_encoder_is_frozen_and_never_leaves_eval(module):
    """icefall's ScheduledFloat dropout stays pinned at its 0.3 default because
    set_batch_count() is never called here, so a train-mode encoder would train
    the trunk against a representation the deployed encoder never produces."""
    assert all(not param.requires_grad for param in module.encoder.parameters())

    module.train()
    assert module.encoder.training is False
    assert module.adapter.training is True

    fake_optimizer = MagicMock()
    fake_optimizer.param_groups = [{"lr": 1e-3}]
    module.optimizers = MagicMock(return_value=fake_optimizer)
    module.training_step(_batch(), 0)

    assert module.encoder.forward_modes == [False]


def test_training_step_returns_a_finite_ctc_loss(module):
    fake_optimizer = MagicMock()
    fake_optimizer.param_groups = [{"lr": 1e-3}]
    module.optimizers = MagicMock(return_value=fake_optimizer)

    loss = module.training_step(_batch(), 0)

    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_encoder_receives_no_gradients(module):
    fake_optimizer = MagicMock()
    fake_optimizer.param_groups = [{"lr": 1e-3}]
    module.optimizers = MagicMock(return_value=fake_optimizer)

    module.training_step(_batch(), 0).backward()

    assert all(param.grad is None for param in module.encoder.parameters())
    assert any(param.grad is not None for param in module.adapter.parameters())


def test_validation_accumulates_per(module):
    module.on_validation_epoch_start()
    module.validation_step(_batch(), 0)

    assert module._val_total_ref == 5
    assert module._val_total_dist >= 0

    module.on_validation_epoch_end()
    logged = {call.args[0] for call in module.log.call_args_list}
    assert "val/per" in logged
