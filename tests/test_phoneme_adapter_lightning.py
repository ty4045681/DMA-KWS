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


def test_validation_boundary_logs_valid_weighted_training_window(module):
    module.optimizers = MagicMock(
        return_value=MagicMock(param_groups=[{"lr": 1e-3}])
    )
    module._ctc_step = MagicMock(
        side_effect=[
            (torch.tensor(2.0, requires_grad=True), None, None, 0),
            (torch.tensor(8.0, requires_grad=True), None, None, 1),
        ]
    )

    module.training_step(_batch(), 0)
    module.training_step(_batch(), 1)
    module.on_validation_epoch_start()

    values = {call.args[0]: call.args[1] for call in module.log.call_args_list}
    assert float(values["train/window/loss_total"]) == pytest.approx(4.0)
    assert values["train/window/ctc_valid"] == 3.0
    assert values["train/window/ctc_skipped"] == 1.0
    assert values["train/window/ctc_skip_rate"] == pytest.approx(0.25)
    assert values["train/window/microbatches"] == 2.0


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


def test_validation_metrics_are_ratios_of_global_sums(module, monkeypatch):
    """Global PER/loss must weight ranks by their token/valid-sample counts.

    Local PERs would be 1/2 and 9/10, whose mean is 0.7. The correct global
    result is (1 + 9) / (2 + 10) = 5/6.
    """
    module.on_validation_epoch_start()
    module._val_ctc_loss_sum = 4.0
    module._val_ctc_valid = 2
    module._val_ctc_skipped = 0
    module._val_total_dist = 1
    module._val_total_ref = 2

    peer_stats = torch.tensor([18.0, 3.0, 1.0, 9.0, 10.0], dtype=torch.float64)

    def fake_sum(values):
        return values + peer_stats.to(values)

    monkeypatch.setattr(
        "dma_kws.phoneme_adapter.lightning.sum_across_processes",
        fake_sum,
    )
    module.on_validation_epoch_end()

    values = {call.args[0]: float(call.args[1]) for call in module.log.call_args_list}
    assert values["val/loss"] == pytest.approx((4.0 + 18.0) / (2 + 3))
    assert values["val/ctc_valid"] == 5
    assert values["val/ctc_skipped"] == 1
    assert values["val/ctc_skip_rate"] == pytest.approx(1 / 6)
    assert values["val/per"] == pytest.approx((1 + 9) / (2 + 10))
    assert values["val/per"] != pytest.approx((1 / 2 + 9 / 10) / 2)


def test_validation_loss_is_weighted_by_valid_ctc_samples(module):
    module.on_validation_epoch_start()
    module._ctc_step = MagicMock(
        side_effect=[
            (torch.tensor(2.0), None, None, 0),
            (torch.tensor(10.0), None, None, 1),
        ]
    )

    # ``batch_idx`` is beyond the configured decode budget, so only CTC loss
    # accounting runs. The two local means represent 2 and 1 valid samples.
    module.validation_step(_batch(), 2)
    module.validation_step(_batch(), 3)
    module.on_validation_epoch_end()

    values = {call.args[0]: float(call.args[1]) for call in module.log.call_args_list}
    assert values["val/loss"] == pytest.approx((2.0 * 2 + 10.0 * 1) / 3)
    assert values["val/ctc_valid"] == 3
    assert values["val/ctc_skipped"] == 1


def test_validation_metrics_deduplicate_distributed_sampler_padding(module, monkeypatch):
    module.on_validation_epoch_start()
    module._val_ctc_records = [
        (0, 2.0, 1, 0),
        (2, 0.0, 0, 1),
    ]
    module._val_per_records = [
        (0, 1, 2),
        (2, 0, 3),
    ]

    def fake_gather(local_rows):
        if local_rows.size(1) == 4:
            peer_rows = torch.tensor(
                [[1.0, 4.0, 1.0, 0.0], [0.0, 99.0, 1.0, 0.0]],
                dtype=local_rows.dtype,
                device=local_rows.device,
            )
        else:
            peer_rows = torch.tensor(
                [[1.0, 2.0, 5.0], [0.0, 99.0, 99.0]],
                dtype=local_rows.dtype,
                device=local_rows.device,
            )
        return torch.cat([local_rows, peer_rows], dim=0)

    monkeypatch.setattr(
        "dma_kws.training.distributed_metrics.gather_variable_rows",
        fake_gather,
    )
    module.on_validation_epoch_end()

    values = {call.args[0]: float(call.args[1]) for call in module.log.call_args_list}
    assert values["val/loss"] == pytest.approx(3.0)
    assert values["val/ctc_valid"] == 2
    assert values["val/ctc_skipped"] == 1
    assert values["val/ctc_skip_rate"] == pytest.approx(1 / 3)
    assert values["val/per_edit_distance"] == 3
    assert values["val/per_reference_tokens"] == 10
    assert values["val/per"] == pytest.approx(0.3)


def test_training_ctc_counts_are_global_per_step(module, monkeypatch):
    loss = torch.tensor(2.0, requires_grad=True)
    module._ctc_step = MagicMock(return_value=(loss, None, None, 1))
    module.optimizers = MagicMock(
        return_value=MagicMock(param_groups=[{"lr": 1e-3}])
    )

    # Local batch: loss sum=2, valid=1, skipped=1. Peer: the same loss sum,
    # valid=1, skipped=3.
    monkeypatch.setattr(
        "dma_kws.phoneme_adapter.lightning.sum_across_processes",
        lambda values: values
        + torch.tensor([2.0, 1.0, 3.0], device=values.device),
    )
    returned = module.training_step(_batch(), 0)

    values = {
        call.args[0]: float(
            call.args[1].detach() if torch.is_tensor(call.args[1]) else call.args[1]
        )
        for call in module.log.call_args_list
    }
    assert float(returned.detach()) == pytest.approx(2.0)
    assert values["train/microbatch/ctc_valid"] == 2
    assert values["train/microbatch/ctc_skipped"] == 4
    assert values["train/microbatch/ctc_skip_rate"] == pytest.approx(4 / 6)
    assert values["train/epoch/ctc_skip_rate"] == pytest.approx(4 / 6)
