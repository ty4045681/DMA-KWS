from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

pytest.importorskip("pytorch_lightning")

from dma_kws.stage2.module import Stage2LightningModule


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
        },
        "stage2": {
            "encoder_output_dim": 144,
            "qbyt_embed_dim": 128,
            "qbyt_layers": 2,
            "learning_rate": 1e-3,
            "warmup_steps": 2,
            "total_scheduler_steps": 10,
            "max_steps": 10,
        },
    }


class _FakeQbyT(nn.Module):
    def __init__(self, encoder_output_size: int = 144, num_embeds: int = 73, **kwargs):
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(1))

    def forward(self, speech, text, speech_lengths=None, text_lengths=None):
        batch_size = speech.size(0)
        anchor_len = text.size(1)
        logits = self.dummy.expand(batch_size)
        seq_logits = self.dummy.expand(batch_size, anchor_len)
        return logits, seq_logits


def _random_batch(batch_size: int = 2) -> dict:
    feat_lengths = torch.tensor([4, 6], dtype=torch.long)
    max_feat = int(feat_lengths.max())

    feat = torch.randn(batch_size, max_feat, 80)
    anchor = torch.tensor([[10, 11, 12], [13, 14, 0]], dtype=torch.long)
    seq_label = torch.tensor([[1, 0, 1], [0, 1, -1]], dtype=torch.long)
    seq_label_mask = (seq_label != -1).float()

    return {
        "feat": feat,
        "feat_lengths": feat_lengths,
        "anchor": anchor,
        "label": torch.tensor([1, 0], dtype=torch.long),
        "seq_label": seq_label,
        "seq_label_mask": seq_label_mask,
    }


def _mock_encoder_output(feat: torch.Tensor, feat_lengths: torch.Tensor):
    encoded = torch.randn(feat.size(0), feat.size(1), 144)
    mask = torch.arange(feat.size(1)).unsqueeze(0) < feat_lengths.unsqueeze(1)
    return encoded, mask.unsqueeze(1)


@pytest.fixture
def patched_module(monkeypatch):
    fake_encoder = MagicMock(side_effect=_mock_encoder_output)
    monkeypatch.setattr("dma_kws.stage2.module.build_encoder", lambda *_args, **_kwargs: fake_encoder)
    monkeypatch.setattr("dma_kws.stage2.module._load_qbyt", lambda: _FakeQbyT)
    return Stage2LightningModule(_minimal_config(), vocab_size=71)


def test_forward_and_training_step_smoke(patched_module):
    module = patched_module
    module.eval()
    batch = _random_batch()

    logits, seq_logits = module(batch["feat"], batch["feat_lengths"], batch["anchor"])
    assert logits.shape == (2,)
    assert seq_logits.shape == (2, 3)

    fake_optimizer = MagicMock()
    fake_optimizer.param_groups = [{"lr": 1e-3}]
    module.optimizers = MagicMock(return_value=fake_optimizer)

    module.train()
    loss = module.training_step(batch, 0)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_gradient_diagnostics_reports_missing_gradient_changes(patched_module):
    module = patched_module
    module._gradient_diagnostics_enabled = True
    module._gradient_diagnostics_max_steps = 2
    module._trainer = MagicMock(global_step=3)
    module.print = MagicMock()

    module.on_after_backward()
    module.qbyt.dummy.grad = torch.ones_like(module.qbyt.dummy)
    module.on_after_backward()
    module.on_after_backward()

    assert module.print.call_count == 2
    assert "qbyt.dummy" in module.print.call_args_list[0].args[0]
    assert "all trainable parameters have gradients" in module.print.call_args_list[1].args[0]


def test_validation_logs_are_synchronized(patched_module):
    module = patched_module
    module.log = MagicMock()
    module.auc_metric.update = MagicMock()
    module.eer_metric.update = MagicMock()

    module.validation_step(_random_batch(), 0)

    loss_call = next(call for call in module.log.call_args_list if call.args[0] == "val/utt_loss")
    assert loss_call.kwargs["sync_dist"] is True

    module.auc_metric.compute = MagicMock(return_value=torch.tensor(0.75))
    module.eer_metric.compute = MagicMock(return_value=torch.tensor(0.25))
    module.log.reset_mock()

    module.on_validation_epoch_end()

    calls = {call.args[0]: call.kwargs for call in module.log.call_args_list}
    assert calls["val/auc"]["sync_dist"] is True
    assert calls["val/eer"]["sync_dist"] is True
    assert calls["val_auc"]["sync_dist"] is True


def test_freeze_encoder_disables_encoder_gradients(monkeypatch):
    encoder = nn.Linear(80, 144)
    fake_encoder = MagicMock(side_effect=_mock_encoder_output)
    fake_encoder.parameters = encoder.parameters

    monkeypatch.setattr("dma_kws.stage2.module.build_encoder", lambda *_args, **_kwargs: fake_encoder)
    monkeypatch.setattr("dma_kws.stage2.module._load_qbyt", lambda: _FakeQbyT)

    module = Stage2LightningModule(_minimal_config(), vocab_size=71, freeze_encoder=True)

    assert all(not param.requires_grad for param in encoder.parameters())
    assert any(param.requires_grad for param in module.qbyt.parameters())

    optim_cfg = module.configure_optimizers()
    optim_param_ids = {id(param) for group in optim_cfg["optimizer"].param_groups for param in group["params"]}
    encoder_param_ids = {id(param) for param in encoder.parameters()}
    assert encoder_param_ids.isdisjoint(optim_param_ids)


def _icefall_config(chunk_size: int = 16, left_context_frames: int = 64) -> dict:
    config = _minimal_config()
    config["stage1"].update(
        {
            "encoder_type": "icefall_zipformer",
            "causal": True,
            "downsampling_factor": "1,2,4,8,4,2",
            "cnn_module_kernel": "31,31,15,15,15,31",
            "stream": {
                "chunk_size": chunk_size,
                "left_context_frames": left_context_frames,
                "train_policy": "multi",
                "train_chunk_size": "16,32,64,-1",
                "train_left_context_frames": "64,128,256,-1",
            },
        }
    )
    return config


class _StreamSpyEncoder(nn.Module):
    """Records the chunk config applied before each forward pass."""

    def __init__(self) -> None:
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(1))
        self.applied: list[tuple[int, ...]] = []

    def apply_stream_config(self, chunk_sizes, left_context_frames) -> None:
        del left_context_frames
        self.applied.append(tuple(chunk_sizes))

    def forward(self, feat, feat_lengths):
        return _mock_encoder_output(feat, feat_lengths)


@pytest.fixture
def stream_spy_module(monkeypatch):
    encoder = _StreamSpyEncoder()
    monkeypatch.setattr("dma_kws.stage2.module.build_encoder", lambda *_a, **_k: encoder)
    monkeypatch.setattr("dma_kws.stage2.module._load_qbyt", lambda: _FakeQbyT)
    module = Stage2LightningModule(_icefall_config(), vocab_size=71, freeze_encoder=True)
    return module, encoder


def test_training_step_opts_into_multi_latency_chunks(stream_spy_module):
    module, encoder = stream_spy_module
    fake_optimizer = MagicMock()
    fake_optimizer.param_groups = [{"lr": 1e-3}]
    module.optimizers = MagicMock(return_value=fake_optimizer)
    module.log = MagicMock()

    module.training_step(_random_batch(), 0)

    assert encoder.applied == [(16, 32, 64, -1)]


@pytest.mark.parametrize("step_name", ["validation_step", "test_step"])
def test_validation_and_test_steps_pin_the_deployment_point(stream_spy_module, step_name):
    module, encoder = stream_spy_module
    module.log = MagicMock()

    getattr(module, step_name)(_random_batch(), 0)

    assert encoder.applied == [(16,)]


def test_forward_defaults_to_the_deployment_point(stream_spy_module):
    """Callers such as the sweep's target-AUC module never pass a mode."""
    module, encoder = stream_spy_module
    batch = _random_batch()

    module(batch["feat"], batch["feat_lengths"], batch["anchor"])

    assert encoder.applied == [(16,)]


def test_init_checkpoint_from_another_operating_point_is_rejected(monkeypatch, tmp_path):
    encoder = _StreamSpyEncoder()
    monkeypatch.setattr("dma_kws.stage2.module.build_encoder", lambda *_a, **_k: encoder)
    monkeypatch.setattr("dma_kws.stage2.module._load_qbyt", lambda: _FakeQbyT)

    ckpt_path = tmp_path / "stage2_step000010.pt"
    torch.save({"model_state_dict": {}, "config": _icefall_config(chunk_size=32, left_context_frames=128)}, ckpt_path)

    module = Stage2LightningModule(_icefall_config(), vocab_size=71, freeze_encoder=True)

    with pytest.raises(ValueError, match="Streaming operating point mismatch"):
        module._load_init_checkpoint(ckpt_path)


def test_init_checkpoint_predating_the_stream_policy_warns(monkeypatch, tmp_path):
    """A pre-migration checkpoint has no operating point; that must not pass silently."""
    encoder = _StreamSpyEncoder()
    monkeypatch.setattr("dma_kws.stage2.module.build_encoder", lambda *_a, **_k: encoder)
    monkeypatch.setattr("dma_kws.stage2.module._load_qbyt", lambda: _LoraReadyQbyT)

    legacy = _minimal_config()
    legacy["stage1"].update(
        {"encoder_type": "icefall_zipformer", "causal": True, "chunk_size": "16,32,64,-1"}
    )
    ckpt_path = tmp_path / "legacy.pt"
    torch.save({"model_state_dict": {}, "config": legacy}, ckpt_path)

    module = Stage2LightningModule(_icefall_config(), vocab_size=71, freeze_encoder=True)

    with pytest.warns(UserWarning, match="randomized chunk config"):
        module._load_init_checkpoint(ckpt_path)


def test_init_checkpoint_from_the_same_operating_point_is_accepted(monkeypatch, tmp_path):
    encoder = _StreamSpyEncoder()
    monkeypatch.setattr("dma_kws.stage2.module.build_encoder", lambda *_a, **_k: encoder)
    monkeypatch.setattr("dma_kws.stage2.module._load_qbyt", lambda: _FakeQbyT)

    ckpt_path = tmp_path / "stage2_step000010.pt"
    torch.save({"model_state_dict": {}, "config": _icefall_config()}, ckpt_path)

    module = Stage2LightningModule(_icefall_config(), vocab_size=71, freeze_encoder=True)
    module._load_init_checkpoint(ckpt_path)


class _LoraReadyQbyT(nn.Module):
    """QbyT stub exposing the phone_matchor attention layers LoRA hooks into."""

    def __init__(self, encoder_output_size: int = 144, num_embeds: int = 73, **kwargs):
        super().__init__()
        attn_layer = nn.Module()
        attn_layer.self_attn = nn.MultiheadAttention(embed_dim=8, num_heads=1)
        self.phone_matchor = nn.Module()
        self.phone_matchor.layers = nn.ModuleList([attn_layer])

    def forward(self, speech, text, speech_lengths=None, text_lengths=None):
        del speech_lengths, text_lengths
        batch_size = speech.size(0)
        zero = speech.sum() * 0.0
        return zero.expand(batch_size), zero.expand(batch_size, text.size(1))


def test_lora_validation_step_pins_the_deployment_point(monkeypatch):
    """The LoRA module overrides validation_step; it must still land on eval."""
    from dma_kws.stage2.adapt import Stage2LoraAdaptationModule

    encoder = _StreamSpyEncoder()
    monkeypatch.setattr("dma_kws.stage2.module.build_encoder", lambda *_a, **_k: encoder)
    monkeypatch.setattr("dma_kws.stage2.module._load_qbyt", lambda: _LoraReadyQbyT)

    config = _icefall_config()
    config["adapt"] = {"keyword": "hey eva"}
    module = Stage2LoraAdaptationModule(config, vocab_size=71, lora_rank=2, lora_alpha=4.0)
    module.log = MagicMock()

    module.validation_step(_random_batch(), 0, dataloader_idx=0)
    module.validation_step(_random_batch(), 0, dataloader_idx=1)

    assert encoder.applied == [(16,), (16,)]


def test_load_init_checkpoint_detects_icefall_model_keys(monkeypatch, tmp_path):
    class _FakeIcefallWrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder_embed = nn.Linear(4, 4)
            self.encoder = nn.Linear(4, 4)

        def forward(self, feat, feat_lengths):
            batch, time = feat.shape[0], feat.shape[1]
            encoded = torch.zeros(batch, time, 4)
            mask = torch.ones(batch, 1, time, dtype=torch.bool)
            return encoded, mask

    fake_encoder = _FakeIcefallWrapper()
    monkeypatch.setattr("dma_kws.stage2.module.build_encoder", lambda *_args, **_kwargs: fake_encoder)
    monkeypatch.setattr("dma_kws.stage2.module._load_qbyt", lambda: _FakeQbyT)

    module = Stage2LightningModule(_minimal_config(), vocab_size=71)

    # If generic branch is used, this would be called and fail the test.
    module.encoder.load_state_dict = MagicMock(side_effect=AssertionError("wrong checkpoint branch used"))

    checkpoint = {
        "model": {
            "encoder_embed.weight": torch.randn_like(module.encoder.encoder_embed.weight),
            "encoder_embed.bias": torch.randn_like(module.encoder.encoder_embed.bias),
            "encoder.weight": torch.randn_like(module.encoder.encoder.weight),
            "encoder.bias": torch.randn_like(module.encoder.encoder.bias),
        }
    }
    ckpt_path = tmp_path / "icefall.pt"
    torch.save(checkpoint, ckpt_path)

    module._load_init_checkpoint(ckpt_path)

    module.encoder.load_state_dict.assert_not_called()
