import copy
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
            "sequence_loss": {
                "target_mode": "ordered_contiguous_prefix",
                "progress_weight": 0.5,
                "completion_weight": 0.5,
                "normalization": "sample",
            },
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
    seq_label = torch.tensor([[1, 1, 1], [1, 0, -1]], dtype=torch.long)
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


def test_train_end_flushes_partial_window_once(patched_module):
    module = patched_module
    module.log = MagicMock()
    train_logger = MagicMock()
    module._trainer = MagicMock(
        is_global_zero=True,
        loggers=[train_logger],
        global_step=7,
    )
    module.optimizers = MagicMock(
        return_value=MagicMock(param_groups=[{"lr": 1e-3}])
    )

    module.training_step(_random_batch(), 0)
    module.log.reset_mock()

    module.on_train_end()

    train_logger.log_metrics.assert_called_once()
    payload = train_logger.log_metrics.call_args.args[0]
    assert "train/window/loss_total" in payload
    assert payload["train/window/microbatches"] == 1.0
    assert train_logger.log_metrics.call_args.kwargs["step"] == 7

    train_logger.log_metrics.reset_mock()
    module.on_train_end()
    assert not train_logger.log_metrics.called

    # A validation-end flush immediately before train end is likewise not
    # emitted a second time.
    module.training_step(_random_batch(), 1)
    module.log.reset_mock()
    module._log_train_window_metrics()
    assert any(
        call.args[0] == "train/window/loss_total"
        for call in module.log.call_args_list
    )
    train_logger.log_metrics.reset_mock()
    module.on_train_end()
    assert not train_logger.log_metrics.called


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

    module.validation_step(_random_batch(), 0)
    assert not module.log.called

    module.on_validation_epoch_end()

    calls = {call.args[0]: call.kwargs for call in module.log.call_args_list}
    # The custom metric synchronizes raw scores first. Every rank therefore has
    # the same scalar and Lightning's second mean is an identity operation that
    # avoids noisy DDP sync warnings.
    assert calls["val/auc"]["sync_dist"] is True
    assert calls["val/eer"]["sync_dist"] is True
    assert calls["val/eer_threshold"]["sync_dist"] is True
    assert calls["val/completion_auc"]["sync_dist"] is True
    assert "val/completion_diagnostic_threshold" in calls
    assert "val/completion_deploy_threshold" not in calls
    assert calls["val/utt_loss"]["sync_dist"] is True
    assert calls["val_auc"]["sync_dist"] is True
    logged_values = {
        call.args[0]: call.args[1] for call in module.log.call_args_list
    }
    for name in (
        "val/num_samples",
        "val/num_pos",
        "val/num_neg",
        "val/has_both_classes",
    ):
        assert torch.is_floating_point(logged_values[name])


def test_membership_objective_names_endpoint_as_last_token(monkeypatch):
    fake_encoder = MagicMock(side_effect=_mock_encoder_output)
    monkeypatch.setattr(
        "dma_kws.stage2.module.build_encoder", lambda *_args, **_kwargs: fake_encoder
    )
    monkeypatch.setattr("dma_kws.stage2.module._load_qbyt", lambda: _FakeQbyT)
    config = _minimal_config()
    config["stage2"]["sequence_loss"].update(
        {"target_mode": "membership", "progress_weight": 1.0, "completion_weight": 0.0}
    )
    module = Stage2LightningModule(config, vocab_size=71)
    module.log = MagicMock()

    module.validation_step(_random_batch(), 0)
    module.on_validation_epoch_end()

    names = {call.args[0] for call in module.log.call_args_list}
    assert "val/last_token_auc" in names
    assert "val/completion_auc" not in names


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


def _stage2_module(monkeypatch, config: dict | None = None) -> Stage2LightningModule:
    monkeypatch.setattr(
        "dma_kws.stage2.module.build_encoder", lambda *_a, **_k: _StreamSpyEncoder()
    )
    monkeypatch.setattr("dma_kws.stage2.module._load_qbyt", lambda: _FakeQbyT)
    return Stage2LightningModule(config or _minimal_config(), vocab_size=71)


def test_init_checkpoint_with_a_stale_qbyt_readout_is_rejected(monkeypatch, tmp_path):
    """Shapes still match after the readout fix, so nothing else would notice."""
    module = _stage2_module(monkeypatch)
    ckpt_path = tmp_path / "legacy_readout.pt"
    torch.save({"model_state_dict": {"qbyt.dummy": torch.zeros(1)}}, ckpt_path)

    with pytest.raises(SystemExit, match="readout unversioned"):
        module._load_init_checkpoint(ckpt_path)


def test_init_checkpoint_without_qbyt_weights_skips_the_readout_check(monkeypatch, tmp_path):
    """Stage I exports and icefall checkpoints encode no readout convention."""
    module = _stage2_module(monkeypatch)
    ckpt_path = tmp_path / "stage1.pt"
    torch.save({"model_state_dict": {"encoder.dummy": torch.zeros(1)}}, ckpt_path)

    module._load_init_checkpoint(ckpt_path)


def test_lora_base_requires_complete_encoder_and_qbyt_state(monkeypatch, tmp_path):
    from dma_kws.training.checkpoint_io import stamp_qbyt_readout_version

    module = _stage2_module(monkeypatch)
    complete_path = tmp_path / "complete_stage2.pt"
    torch.save(
        stamp_qbyt_readout_version(
            {
                "model_state_dict": {
                    "encoder.dummy": torch.ones_like(module.encoder.dummy),
                    "qbyt.dummy": torch.ones_like(module.qbyt.dummy),
                }
            }
        ),
        complete_path,
    )
    module._load_init_checkpoint(complete_path, require_full_qbyt=True)
    torch.testing.assert_close(
        module.encoder.dummy,
        torch.ones_like(module.encoder.dummy),
    )
    torch.testing.assert_close(module.qbyt.dummy, torch.ones_like(module.qbyt.dummy))

    missing_qbyt_path = tmp_path / "encoder_only.pt"
    torch.save(
        {"model_state_dict": {"encoder.dummy": torch.zeros(1)}},
        missing_qbyt_path,
    )
    with pytest.raises(SystemExit, match="qbyt"):
        module._load_init_checkpoint(missing_qbyt_path, require_full_qbyt=True)

    missing_encoder_path = tmp_path / "qbyt_only.pt"
    torch.save(
        stamp_qbyt_readout_version(
            {"model_state_dict": {"qbyt.dummy": torch.zeros(1)}}
        ),
        missing_encoder_path,
    )
    with pytest.raises(SystemExit, match="encoder"):
        module._load_init_checkpoint(missing_encoder_path, require_full_qbyt=True)

    partial_encoder_path = tmp_path / "partial_encoder.pt"
    torch.save(
        stamp_qbyt_readout_version(
            {
                "model_state_dict": {
                    "encoder.unexpected": torch.zeros(1),
                    "qbyt.dummy": torch.zeros(1),
                }
            }
        ),
        partial_encoder_path,
    )
    with pytest.raises(RuntimeError, match="state_dict"):
        module._load_init_checkpoint(partial_encoder_path, require_full_qbyt=True)

    partial_qbyt_path = tmp_path / "partial_qbyt.pt"
    torch.save(
        stamp_qbyt_readout_version(
            {
                "model_state_dict": {
                    "encoder.dummy": torch.zeros(1),
                    "qbyt.unexpected": torch.zeros(1),
                }
            }
        ),
        partial_qbyt_path,
    )
    with pytest.raises(RuntimeError, match="state_dict"):
        module._load_init_checkpoint(partial_qbyt_path, require_full_qbyt=True)


def test_full_stage2_model_container_is_not_misclassified_as_icefall(
    monkeypatch,
    tmp_path,
):
    from dma_kws.training.checkpoint_io import stamp_qbyt_readout_version

    module = _stage2_module(monkeypatch)
    checkpoint_path = tmp_path / "full_model_container.pt"
    torch.save(
        stamp_qbyt_readout_version(
            {
                "model": {
                    "encoder.dummy": torch.ones_like(module.encoder.dummy),
                    "qbyt.dummy": torch.ones_like(module.qbyt.dummy),
                }
            }
        ),
        checkpoint_path,
    )

    module._load_init_checkpoint(checkpoint_path, require_full_qbyt=True)

    torch.testing.assert_close(
        module.encoder.dummy,
        torch.ones_like(module.encoder.dummy),
    )
    torch.testing.assert_close(module.qbyt.dummy, torch.ones_like(module.qbyt.dummy))


def test_lora_base_rejects_encoder_only_icefall_checkpoint(monkeypatch, tmp_path):
    module = _stage2_module(monkeypatch)
    checkpoint_path = tmp_path / "icefall_encoder.pt"
    torch.save({"model": {"encoder.layer.weight": torch.zeros(1)}}, checkpoint_path)

    with pytest.raises(SystemExit, match="encoder-only Icefall"):
        module._load_init_checkpoint(checkpoint_path, require_full_qbyt=True)


def test_stale_qbyt_readout_can_be_opted_into_with_a_warning(monkeypatch, tmp_path):
    config = _minimal_config()
    config["stage2"]["allow_legacy_qbyt_readout"] = True
    module = _stage2_module(monkeypatch, config)
    ckpt_path = tmp_path / "legacy_readout.pt"
    torch.save({"model_state_dict": {"qbyt.dummy": torch.zeros(1)}}, ckpt_path)

    with pytest.warns(UserWarning, match="not comparable"):
        module._load_init_checkpoint(ckpt_path)


def test_lora_base_cannot_opt_into_a_stale_qbyt_readout(monkeypatch, tmp_path):
    config = _minimal_config()
    config["stage2"]["allow_legacy_qbyt_readout"] = True
    module = _stage2_module(monkeypatch, config)
    checkpoint_path = tmp_path / "legacy_readout.pt"
    torch.save(
        {"model_state_dict": {"qbyt.dummy": torch.zeros(1)}},
        checkpoint_path,
    )

    with pytest.raises(SystemExit, match="readout unversioned"):
        module._load_init_checkpoint(
            checkpoint_path,
            require_full_qbyt=True,
        )


def test_saved_checkpoints_carry_the_readout_version(monkeypatch):
    """Without the stamp, every checkpoint this build writes looks stale."""
    from dma_kws.training.checkpoint_io import (
        QBYT_READOUT_VERSION,
        QBYT_READOUT_VERSION_KEY,
    )

    module = _stage2_module(monkeypatch)
    checkpoint: dict = {"state_dict": module.state_dict()}
    module.on_save_checkpoint(checkpoint)

    assert checkpoint[QBYT_READOUT_VERSION_KEY] == QBYT_READOUT_VERSION
    assert checkpoint["checkpoint_kind"] == "stage2"
    assert checkpoint["config"] == _minimal_config()
    assert checkpoint["vocab_size"] == 71
    module.on_load_checkpoint(checkpoint)


def test_lightning_restore_rejects_a_stale_readout(monkeypatch):
    """Covers load_from_checkpoint and Trainer.fit(ckpt_path=...) alike."""
    module = _stage2_module(monkeypatch)

    with pytest.raises(SystemExit, match="readout unversioned"):
        module.on_load_checkpoint({"state_dict": {"qbyt.dummy": torch.zeros(1)}})


def test_lightning_restore_rejects_a_different_stream_point(monkeypatch):
    from dma_kws.training.checkpoint_io import stamp_qbyt_readout_version

    module = _stage2_module(monkeypatch, _icefall_config())
    checkpoint = stamp_qbyt_readout_version(
        {
            "state_dict": module.state_dict(),
            "config": _icefall_config(chunk_size=32, left_context_frames=128),
        }
    )

    with pytest.raises(ValueError, match="Streaming operating point mismatch"):
        module.on_load_checkpoint(checkpoint)


def test_lora_adapter_payloads_are_readout_checked(monkeypatch, tmp_path):
    """A LoRA payload holds no ``qbyt.``-prefixed keys, so it needs its own probe."""
    from dma_kws.training.checkpoint_io import assert_qbyt_readout_version

    ckpt_path = tmp_path / "adapter.pt"
    torch.save({"lora_state_dict": {"phone_matchor.layers.0.self_attn": torch.zeros(1)}}, ckpt_path)

    with pytest.raises(SystemExit, match="readout unversioned"):
        assert_qbyt_readout_version(torch.load(ckpt_path), source=ckpt_path)


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


def test_lora_source_metrics_are_global_and_fields_are_stable(monkeypatch):
    from dma_kws.stage2.adapt import Stage2LoraAdaptationModule

    encoder = _StreamSpyEncoder()
    monkeypatch.setattr("dma_kws.stage2.module.build_encoder", lambda *_a, **_k: encoder)
    monkeypatch.setattr("dma_kws.stage2.module._load_qbyt", lambda: _LoraReadyQbyT)

    config = _icefall_config()
    config["adapt"] = {"keyword": "hey eva"}
    module = Stage2LoraAdaptationModule(
        config,
        vocab_size=71,
        lora_rank=2,
        lora_alpha=4.0,
    )
    logits = torch.tensor([0.0, 1.0])
    labels = torch.tensor([1, 0])
    module._forward_train_losses = MagicMock(
        return_value=(torch.tensor(1.0, requires_grad=True), {}, logits)
    )
    module._log_train_losses = MagicMock()
    module.log = MagicMock()

    # This rank contains only keyword examples. The peer contributes one LPh
    # example with loss sum 3, so all three fields must still exist globally.
    monkeypatch.setattr(
        "dma_kws.stage2.adapt.sum_across_processes",
        lambda values: values
        + torch.tensor([0.0, 0.0, 3.0, 1.0], device=values.device),
    )
    module.training_step(
        {"label": labels, "source": torch.tensor([1, 1])},
        0,
    )

    calls = {call.args[0]: call for call in module.log.call_args_list}
    expected_keyword_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        logits,
        labels.float(),
        reduction="none",
    ).mean()
    assert float(calls["train/microbatch/source_keyword_fraction"].args[1]) == pytest.approx(
        2 / 3
    )
    assert float(calls["train/microbatch/loss_keyword_utt_raw"].args[1]) == pytest.approx(
        float(expected_keyword_loss)
    )
    assert float(calls["train/microbatch/loss_lph_utt_raw"].args[1]) == pytest.approx(3.0)
    for name in (
        "train/microbatch/source_keyword_fraction",
        "train/microbatch/loss_keyword_utt_raw",
        "train/microbatch/loss_lph_utt_raw",
    ):
        assert calls[name].kwargs["sync_dist"] is True


def test_lora_adam_honors_weight_decay(monkeypatch):
    import sys
    from types import SimpleNamespace

    from dma_kws.stage2.adapt import Stage2LoraAdaptationModule

    encoder = _StreamSpyEncoder()
    monkeypatch.setattr("dma_kws.stage2.module.build_encoder", lambda *_a, **_k: encoder)
    monkeypatch.setattr("dma_kws.stage2.module._load_qbyt", lambda: _LoraReadyQbyT)
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            get_cosine_schedule_with_warmup=lambda optimizer, **_kwargs: (
                torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _step: 1.0)
            )
        ),
    )

    config = _icefall_config()
    config["adapt"] = {
        "keyword": "hey eva",
        "optimizer": "adam",
        "learning_rate": 1e-3,
        "weight_decay": 0.125,
        "warmup_steps": 1,
        "max_steps": 10,
    }
    module = Stage2LoraAdaptationModule(
        config,
        vocab_size=71,
        lora_rank=2,
        lora_alpha=4.0,
    )
    optimizer = module.configure_optimizers()["optimizer"]

    assert optimizer.param_groups[0]["weight_decay"] == pytest.approx(0.125)
    optimized = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    expected = {id(parameter) for parameter in module.parameters() if parameter.requires_grad}
    assert optimized == expected

    module.train()
    assert module.training is True
    assert module.qbyt.training is True
    assert module.encoder.training is False


def test_lora_validation_step_pins_the_deployment_point(monkeypatch):
    """The LoRA module overrides validation_step; it must still land on eval."""
    from dma_kws.stage2.adapt import Stage2LoraAdaptationModule

    encoder = _StreamSpyEncoder()
    monkeypatch.setattr("dma_kws.stage2.module.build_encoder", lambda *_a, **_k: encoder)
    monkeypatch.setattr("dma_kws.stage2.module._load_qbyt", lambda: _LoraReadyQbyT)

    config = _icefall_config()
    config["adapt"] = {"keyword": "hey eva", "slug": ""}
    module = Stage2LoraAdaptationModule(config, vocab_size=71, lora_rank=2, lora_alpha=4.0)
    module.log = MagicMock()

    assert module.target_score_diagnostics.sync_on_compute is True
    assert module.target_completion_score_diagnostics.sync_on_compute is True
    assert module.score_diagnostics.sync_on_compute is True
    assert module.completion_score_diagnostics.sync_on_compute is True

    module.validation_step(_random_batch(), 0, dataloader_idx=0)
    module.validation_step(_random_batch(), 0, dataloader_idx=1)

    assert encoder.applied == [(16,), (16,)]
    assert not module.log.called

    module.on_validation_epoch_end()
    validation_logs = {
        call.args[0]: call.kwargs for call in module.log.call_args_list
    }
    assert validation_logs["val/target_eer_threshold"]["sync_dist"] is True
    assert validation_logs["val/lph_eer_threshold"]["sync_dist"] is True
    assert validation_logs["val/target_utt_loss"]["sync_dist"] is True
    assert validation_logs["val/lph_utt_loss"]["sync_dist"] is True
    assert validation_logs["val/target_completion_auc"]["sync_dist"] is True
    assert validation_logs["val/lph_completion_auc"]["sync_dist"] is True

    checkpoint = {"state_dict": module.state_dict()}
    module.on_save_checkpoint(checkpoint)
    assert checkpoint["checkpoint_kind"] == "stage2_lora"
    assert checkpoint["keyword"] == "hey eva"
    assert checkpoint["slug"] == "hey_eva"
    assert checkpoint["phase"] == "tts"
    assert checkpoint["rank"] == 2
    assert checkpoint["alpha"] == 4.0
    assert checkpoint["lora_targets"] == ["in_proj_weight", "out_proj.weight"]
    assert checkpoint["config"]["adapt"]["rank"] == 2
    assert checkpoint["config"]["adapt"]["alpha"] == 4.0
    assert checkpoint["config"]["adapt"]["lora_targets"] == [
        "in_proj_weight",
        "out_proj.weight",
    ]
    module.on_load_checkpoint(checkpoint)

    wrong_alpha = dict(checkpoint)
    wrong_alpha["alpha"] = 8.0
    with pytest.raises(SystemExit, match="alpha=8.0"):
        module.on_load_checkpoint(wrong_alpha)

    conflicting_alpha = copy.deepcopy(checkpoint)
    conflicting_alpha["config"]["adapt"]["alpha"] = 8.0
    with pytest.raises(SystemExit, match="config.adapt.alpha"):
        module.on_load_checkpoint(conflicting_alpha)

    missing_alpha = copy.deepcopy(checkpoint)
    missing_alpha.pop("alpha")
    missing_alpha["config"]["adapt"].pop("alpha")
    with pytest.raises(SystemExit, match="required metadata.*alpha"):
        module.on_load_checkpoint(missing_alpha)


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
