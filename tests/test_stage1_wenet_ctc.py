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
    return Stage1LightningModule(_minimal_config(), vocab_size=71)


class _StreamSpyEncoder(nn.Module):
    """Wenet-style encoder that records the chunk kwargs it is called with."""

    def __init__(self) -> None:
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(1))
        self.kwargs: list[dict] = []

    def forward(self, feat, feat_lengths, **kwargs):
        self.kwargs.append(kwargs)
        return _mock_encoder_output(feat, feat_lengths)


@pytest.fixture
def stream_spy_module(monkeypatch):
    config = _minimal_config()
    config["stage1"].update(
        {
            "use_dynamic_chunk": True,
            "stream": {"chunk_size": 8, "left_context_frames": 32, "train_policy": "multi"},
        }
    )
    encoder = _StreamSpyEncoder()
    monkeypatch.setattr("dma_kws.stage1.module.build_encoder", lambda *_a, **_k: encoder)
    monkeypatch.setattr("dma_kws.stage1.module._load_ctc", lambda: _FakeCTC)
    return Stage1LightningModule(config, vocab_size=71), encoder


def _stage1_batch():
    return stage1_collate_fn(
        [
            {"feat": torch.randn(4, 80), "target": torch.tensor([17, 4, 22, 26], dtype=torch.long)},
            {"feat": torch.randn(6, 80), "target": torch.tensor([17, 4], dtype=torch.long)},
        ]
    )


def test_collate_propagates_optional_stable_sample_ids():
    batch = stage1_collate_fn(
        [
            {
                "feat": torch.randn(4, 80),
                "target": torch.tensor([3, 4]),
                "sample_id": torch.tensor(10),
            },
            {
                "feat": torch.randn(5, 80),
                "target": torch.tensor([5]),
                "sample_id": torch.tensor(11),
            },
        ]
    )

    assert batch["sample_id"].tolist() == [10, 11]


def test_training_step_delegates_to_the_dynamic_chunk_sampler(stream_spy_module):
    module, encoder = stream_spy_module
    fake_optimizer = MagicMock()
    fake_optimizer.param_groups = [{"lr": 1e-3}]
    module.optimizers = MagicMock(return_value=fake_optimizer)
    module.log = MagicMock()

    module.training_step(_stage1_batch(), 0)

    assert encoder.kwargs == [{"decoding_chunk_size": 0, "num_decoding_left_chunks": -1}]


def test_validation_step_pins_the_deployment_point(stream_spy_module):
    module, encoder = stream_spy_module
    module.on_validation_epoch_start()

    module.validation_step(_stage1_batch(), 0)

    assert encoder.kwargs == [{"decoding_chunk_size": 8, "num_decoding_left_chunks": 4}]


def test_phonemes_to_g2p_string_and_encode_manifest_target():
    tok = load_char_tokenizer(DICT_PATH)
    assert phonemes_to_g2p_string(["HH", "AH0", "L", "OW1"]) == "HH AH0 L OW1"

    list_ids = encode_manifest_target({"phonemes": ["HH", "AH0", "L", "OW1"]}, tok)
    str_ids = encode_manifest_target({"phonemes_g2p": "HH AH0 L OW1"}, tok)
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


def test_validation_boundary_logs_mean_training_window(patched_module):
    module = patched_module
    module.log = MagicMock()
    module.optimizers = MagicMock(
        return_value=MagicMock(param_groups=[{"lr": 1e-3}])
    )
    module.ctc.forward = MagicMock(
        side_effect=[(torch.tensor(2.0), None), (torch.tensor(4.0), None)]
    )
    batch = _stage1_batch()

    module.training_step(batch, 0)
    module.training_step(batch, 1)
    module.on_validation_epoch_start()

    values = {call.args[0]: call.args[1] for call in module.log.call_args_list}
    assert float(values["train/window/loss_total"]) == pytest.approx(3.0)
    assert values["train/window/microbatches"] == 2.0


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
    module = Stage1LightningModule(config, vocab_size=71)
    out = export_stage1_encoder_pt(
        module,
        tmp_path / "stage1.pt",
        config=config,
        dict_path=DICT_PATH,
        vocab_size=71,
        blank_id=0,
        step=42,
    )

    stage2 = Stage2LightningModule({"stage1": config["stage1"], "stage2": {}}, vocab_size=71)
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


def test_validation_per_is_global_edit_sum_over_global_reference_sum(
    patched_module, monkeypatch
):
    """Do not average rank-local PERs when reference-token counts differ."""
    module = patched_module
    module.log = MagicMock()
    module.on_validation_epoch_start()
    module._val_total_dist = 1
    module._val_total_ref = 2

    # Peer PER is 9/10. Mean(local PERs)=0.7, while global PER=10/12.
    monkeypatch.setattr(
        "dma_kws.stage1.module.sum_across_processes",
        lambda values: values + torch.tensor([9, 10], device=values.device),
    )
    module.on_validation_epoch_end()

    values = {call.args[0]: float(call.args[1]) for call in module.log.call_args_list}
    assert values["val/per_edit_distance"] == 10
    assert values["val/per_reference_tokens"] == 12
    assert values["val/per"] == pytest.approx(10 / 12)
    assert values["val/per"] != pytest.approx((1 / 2 + 9 / 10) / 2)
    alias_call = next(call for call in module.log.call_args_list if call.args[0] == "val_per")
    assert alias_call.kwargs["logger"] is False


def test_validation_per_deduplicates_distributed_sampler_padding(
    patched_module, monkeypatch
):
    module = patched_module
    module.log = MagicMock()
    module.on_validation_epoch_start()
    module._val_per_records = [(0, 1, 2), (2, 0, 3)]

    def fake_gather(local_rows):
        peer_rows = torch.tensor(
            [
                [1.0, 2.0, 5.0],
                # Sampler padding repeats sample 0 on the peer rank.  Deliberately
                # different values make a failure to de-duplicate unambiguous.
                [0.0, 99.0, 99.0],
            ],
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
    assert values["val/per_edit_distance"] == 3
    assert values["val/per_reference_tokens"] == 10
    assert values["val/per"] == pytest.approx(0.3)
