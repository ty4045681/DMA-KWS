"""Stage II wiring of the phoneme CTC adapter."""

from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

pytest.importorskip("pytorch_lightning")

from dma_kws.stage2.losses import compute_stage2_losses
from dma_kws.stage2.module import Stage2LightningModule, assert_adapter_weights_loaded
from dma_kws.training.checkpoint_io import stamp_qbyt_readout_version

ENCODER_DIM = 144
TRUNK_DIM = 32
VOCAB_SIZE = 71


class _RecordingQbyT(nn.Module):
    """Records the width of the audio tensor it is handed."""

    def __init__(self, encoder_output_size: int = ENCODER_DIM, num_embeds: int = 73, **kwargs):
        super().__init__()
        self.encoder_output_size = encoder_output_size
        self.proj = nn.Linear(encoder_output_size, 1)
        self.seen_widths: list[int] = []

    def forward(self, speech, text, speech_lengths=None, text_lengths=None):
        del speech_lengths, text_lengths
        self.seen_widths.append(speech.size(-1))
        pooled = self.proj(speech).mean(dim=1).squeeze(-1)
        return pooled, pooled.unsqueeze(1).expand(-1, text.size(1))


def _config(**adapter_overrides) -> dict:
    adapter = {
        "enabled": True,
        "trunk": {"type": "conv", "output_dim": TRUNK_DIM, "num_layers": 1, "kernel_size": 3},
    }
    adapter.update(adapter_overrides)
    return {
        "stage1": {"input_dim": 80, "encoder_output_dim": ENCODER_DIM, "causal": False},
        "stage2": {
            "encoder_output_dim": ENCODER_DIM,
            "qbyt_embed_dim": 128,
            "qbyt_layers": 2,
            "qbyt_alignment": {
                "topology": "bounded_segmental_v1",
                "min_phone_duration_frames": 1,
                "max_phone_duration_frames": 8,
                "max_inter_phone_gap_frames": 2,
                "max_keyword_span_frames": 30,
                "temperature": 0.2,
                "local_context_kernel": 5,
            },
            "sequence_loss": {
                "target_mode": "ordered_contiguous_prefix",
                "progress_weight": 0.3,
                "normalization": "sample",
            },
            "learning_rate": 1e-3,
            "warmup_steps": 2,
            "total_scheduler_steps": 10,
            "max_steps": 10,
            "phoneme_adapter": adapter,
        },
    }


def _mock_encoder_output(feat: torch.Tensor, feat_lengths: torch.Tensor):
    encoded = torch.randn(feat.size(0), feat.size(1), ENCODER_DIM)
    mask = torch.arange(feat.size(1)).unsqueeze(0) < feat_lengths.unsqueeze(1)
    return encoded, mask.unsqueeze(1)


def _batch(batch_size: int = 2, frames: int = 12) -> dict:
    feat_lengths = torch.tensor([frames, frames - 4], dtype=torch.long)
    seq_label = torch.tensor([[1, 1, 1], [1, 0, -1]], dtype=torch.long)
    return {
        "feat": torch.randn(batch_size, frames, 80),
        "feat_lengths": feat_lengths,
        "anchor": torch.tensor([[10, 11, 12], [13, 14, 0]], dtype=torch.long),
        "label": torch.tensor([1, 0], dtype=torch.long),
        "seq_label": seq_label,
        "seq_label_mask": (seq_label != -1).float(),
        "query_seq": torch.tensor([[10, 11, 12], [13, 14, 0]], dtype=torch.long),
        "query_lengths": torch.tensor([3, 2], dtype=torch.long),
    }


def _build(
    monkeypatch,
    config: dict,
    *,
    freeze_encoder: bool = True,
    init_checkpoint=None,
) -> Stage2LightningModule:
    encoder = nn.Linear(80, ENCODER_DIM)
    fake_encoder = MagicMock(side_effect=_mock_encoder_output)
    fake_encoder.parameters = encoder.parameters
    fake_encoder.eval = MagicMock()
    fake_encoder.load_state_dict = MagicMock(return_value=([], []))
    monkeypatch.setattr("dma_kws.stage2.module.build_encoder", lambda *_a, **_k: fake_encoder)
    monkeypatch.setattr(
        "dma_kws.stage2.module.build_qbyt",
        lambda *_a, **kwargs: _RecordingQbyT(
            encoder_output_size=kwargs["input_dim"],
            num_embeds=kwargs["vocab_size"],
        ),
    )
    return Stage2LightningModule(
        config,
        vocab_size=VOCAB_SIZE,
        freeze_encoder=freeze_encoder,
        init_checkpoint=init_checkpoint,
    )


def _build_with_init(monkeypatch, config: dict, *, init_checkpoint) -> Stage2LightningModule:
    return _build(monkeypatch, config, init_checkpoint=init_checkpoint)


def test_disabled_adapter_leaves_qbyt_reading_the_encoder(monkeypatch):
    config = _config()
    config["stage2"]["phoneme_adapter"]["enabled"] = False
    module = _build(monkeypatch, config)

    assert module.adapter is None
    assert module.qbyt.encoder_output_size == ENCODER_DIM


def test_enabled_adapter_feeds_qbyt_the_trunk_output(monkeypatch):
    """QbyT must read the CTC-supervised trunk, not the raw encoder output.

    If it read the encoder directly the CTC head would be a dead branch that
    only serves Stage I, which is the whole failure mode this module exists to
    prevent.
    """
    module = _build(monkeypatch, _config())
    batch = _batch()

    module(batch["feat"], batch["feat_lengths"], batch["anchor"])

    assert module.qbyt.encoder_output_size == TRUNK_DIM
    assert module.qbyt.seen_widths == [TRUNK_DIM]


def test_expose_posterior_widens_the_qbyt_input(monkeypatch):
    module = _build(monkeypatch, _config(expose_posterior=True))

    assert module.qbyt.encoder_output_size == TRUNK_DIM + VOCAB_SIZE


def test_frozen_encoder_optimizer_still_includes_the_adapter(monkeypatch):
    """Optimizing only ``qbyt`` when the encoder is frozen silently drops the
    trunk, which is precisely the module the auxiliary CTC loss trains."""
    module = _build(monkeypatch, _config(ctc_weight=0.2))

    optimized = {id(param) for param in module.trainable_module().parameters()}

    assert {id(p) for p in module.adapter.parameters()} <= optimized
    assert {id(p) for p in module.qbyt.parameters()} <= optimized


def test_frozen_adapter_is_excluded_from_the_optimizer(monkeypatch):
    module = _build(monkeypatch, _config(freeze=True))

    assert all(not param.requires_grad for param in module.adapter.parameters())

    optimized = {id(param) for param in module.trainable_module().parameters()}
    assert {id(p) for p in module.adapter.parameters()}.isdisjoint(optimized)


def test_frozen_encoder_optimizer_excludes_the_encoder(monkeypatch):
    module = _build(monkeypatch, _config())

    optimized = {id(param) for param in module.trainable_module().parameters()}
    encoder_params = {id(param) for param in module.encoder.parameters()}

    assert encoder_params
    assert encoder_params.isdisjoint(optimized)


def test_auxiliary_ctc_loss_is_reported_when_weighted(monkeypatch):
    module = _build(monkeypatch, _config(ctc_weight=0.3))
    module.log = MagicMock()

    total_loss, losses, _ = module._forward_train_losses(_batch())

    assert "ctc_loss" in losses
    assert torch.isfinite(total_loss)


def test_auxiliary_ctc_skip_metrics_use_global_counts(monkeypatch):
    module = _build(monkeypatch, _config(ctc_weight=0.3))
    module.log = MagicMock()
    module.adapter.ctc_loss = MagicMock(
        return_value=(torch.tensor(2.0, requires_grad=True), 1)
    )

    # Local batch: loss sum=2, valid=1, skipped=1. Peer: the same loss sum,
    # valid=1, skipped=3.
    monkeypatch.setattr(
        "dma_kws.stage2.module.sum_across_processes",
        lambda values: values
        + torch.tensor([2.0, 1.0, 3.0], device=values.device),
    )
    batch = _batch()
    loss = module._auxiliary_ctc_loss(
        batch,
        torch.randn(2, 12, VOCAB_SIZE),
        torch.ones(2, 1, 12, dtype=torch.bool),
    )

    values = {call.args[0]: float(call.args[1]) for call in module.log.call_args_list}
    assert float(loss.detach()) == 2.0
    assert values["train/microbatch/ctc_valid"] == 2
    assert values["train/microbatch/ctc_skipped"] == 4
    assert values["train/microbatch/ctc_skip_rate"] == pytest.approx(4 / 6)
    assert values["train/epoch/ctc_skip_rate"] == pytest.approx(4 / 6)


def test_auxiliary_ctc_loss_is_off_by_default(monkeypatch):
    module = _build(monkeypatch, _config())
    module.log = MagicMock()

    _total, losses, _ = module._forward_train_losses(_batch())

    assert "ctc_loss" not in losses


def test_train_logs_raw_weighted_and_windowed_loss_contract(monkeypatch):
    module = _build(monkeypatch, _config())
    module.log = MagicMock()
    module.optimizers = MagicMock(
        return_value=MagicMock(param_groups=[{"lr": 1e-3}])
    )
    total, losses = compute_stage2_losses(
        logits=torch.tensor([0.2, -0.4]),
        seq_logits=torch.tensor([[0.3, -0.2], [0.1, 0.6]]),
        labels=torch.tensor([1, 0]),
        seq_labels=torch.tensor([[1, 0], [1, 1]]),
        seq_label_mask=torch.ones(2, 2),
        seq_progress_weight=0.25,
    )

    module._log_train_losses(total, losses)
    microbatch_names = {call.args[0] for call in module.log.call_args_list}

    assert "train/microbatch/loss_total" in microbatch_names
    assert "train/microbatch/loss_seq_progress_raw" in microbatch_names
    assert "train/microbatch/loss_seq_progress_weighted" in microbatch_names

    module.log.reset_mock()
    module._log_train_window_metrics()
    window_values = {
        call.args[0]: float(call.args[1]) for call in module.log.call_args_list
    }

    assert window_values["train/window/loss_total"] == pytest.approx(float(total))
    assert window_values["train/window/loss_seq_weighted"] == pytest.approx(
        float(losses["seq_loss"])
    )
    assert window_values["train/window/microbatches"] == 1


@pytest.mark.parametrize("ctc_weight", [0.0, 0.3])
def test_every_trainable_parameter_receives_a_gradient(monkeypatch, ctc_weight):
    """This is DDP's precondition, not a nicety.

    With ctc_weight=0 the CTC projection produces nothing any loss consumes. A
    parameter that requires grad and never receives one makes DDP abort the
    reduction, and single-GPU runs give no warning at all.
    """
    module = _build(monkeypatch, _config(ctc_weight=ctc_weight))
    module.log = MagicMock()

    total_loss, _losses, _ = module._forward_train_losses(_batch())
    total_loss.backward()

    starved = [
        name
        for name, param in module.named_parameters()
        if param.requires_grad and param.grad is None
    ]
    assert starved == []


def test_ctc_projection_is_frozen_when_unused_and_trainable_when_weighted(monkeypatch):
    unused = _build(monkeypatch, _config(ctc_weight=0.0))
    weighted = _build(monkeypatch, _config(ctc_weight=0.3))

    assert all(not p.requires_grad for p in unused.adapter.ctc.parameters())
    assert all(p.requires_grad for p in weighted.adapter.ctc.parameters())
    # Either way it stays in the state dict so checkpoints load strictly.
    assert any(key.startswith("ctc.") for key in unused.adapter.state_dict())


def test_ctc_weight_zero_reproduces_the_pre_adapter_loss():
    """Backward compatibility: the adapter must not perturb existing runs."""
    torch.manual_seed(0)
    logits = torch.randn(4)
    seq_logits = torch.randn(4, 3)
    labels = torch.tensor([1, 0, 1, 0])
    seq_labels = torch.tensor([[1, 1, 1], [1, 0, -1], [1, 1, 1], [0, 0, -1]])
    mask = (seq_labels != -1).float()

    baseline, _ = compute_stage2_losses(logits, seq_logits, labels, seq_labels, mask)
    with_ctc, losses = compute_stage2_losses(
        logits,
        seq_logits,
        labels,
        seq_labels,
        mask,
        ctc_loss=torch.tensor(7.0),
        ctc_weight=0.0,
    )

    torch.testing.assert_close(baseline, with_ctc)
    assert "ctc_loss" not in losses


def test_adapter_checkpoint_roundtrip(monkeypatch, tmp_path):
    module = _build(monkeypatch, _config())
    checkpoint = tmp_path / "adapter.pt"
    torch.save({"model_state_dict": module.adapter.state_dict()}, checkpoint)

    config = _config(init_checkpoint=str(checkpoint))
    reloaded = _build(monkeypatch, config)

    for (name, before), after in zip(
        module.adapter.state_dict().items(), reloaded.adapter.state_dict().values()
    ):
        torch.testing.assert_close(before, after, msg=name)


class _LoraReadyQbyT(_RecordingQbyT):
    """Adds the alignment projections LoRA injects into."""

    def __init__(self, encoder_output_size: int = ENCODER_DIM, num_embeds: int = 73, **kwargs):
        super().__init__(encoder_output_size=encoder_output_size, num_embeds=num_embeds)
        self.audio_projection = nn.Linear(encoder_output_size, 8)
        self.audio_key = nn.Linear(8, 8)
        self.text_query = nn.Linear(8, 8)


def test_lora_adaptation_freezes_the_trunk_and_drops_the_ctc_loss(monkeypatch):
    """The trunk is part of the forward pass Stage I and Stage II share, so LoRA
    must not move it; otherwise the two stages stop agreeing on one encoder pass."""
    from dma_kws.stage2.adapt import Stage2LoraAdaptationModule

    class _FakeEncoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj = nn.Linear(80, ENCODER_DIM)

        def forward(self, feat, feat_lengths):
            return _mock_encoder_output(feat, feat_lengths)

    monkeypatch.setattr(
        "dma_kws.stage2.module.build_encoder",
        lambda *_a, **_k: _FakeEncoder(),
    )
    monkeypatch.setattr(
        "dma_kws.stage2.module.build_qbyt",
        lambda *_a, **kwargs: _LoraReadyQbyT(
            encoder_output_size=kwargs["input_dim"],
            num_embeds=kwargs["vocab_size"],
        ),
    )

    config = _config(ctc_weight=0.5)
    config["adapt"] = {"keyword": "hey eva"}
    module = Stage2LoraAdaptationModule(config, vocab_size=VOCAB_SIZE, lora_rank=2, lora_alpha=4.0)

    assert module.ctc_weight == 0.0
    assert module.freeze_adapter is True
    assert all(not param.requires_grad for param in module.adapter.parameters())

    module.log = MagicMock()
    _total, losses, _ = module._forward_train_losses(_batch())
    assert "ctc_loss" not in losses


def test_lora_full_resume_does_not_require_external_adapter_init(monkeypatch):
    from dma_kws.stage2.adapt import Stage2LoraAdaptationModule

    class _FakeEncoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj = nn.Linear(80, ENCODER_DIM)

        def forward(self, feat, feat_lengths):
            return _mock_encoder_output(feat, feat_lengths)

    monkeypatch.setattr(
        "dma_kws.stage2.module.build_encoder",
        lambda *_a, **_k: _FakeEncoder(),
    )
    monkeypatch.setattr(
        "dma_kws.stage2.module.build_qbyt",
        lambda *_a, **kwargs: _LoraReadyQbyT(
            encoder_output_size=kwargs["input_dim"],
            num_embeds=kwargs["vocab_size"],
        ),
    )

    config = _config(
        ctc_weight=0.0,
        init_checkpoint="/checkpoint/that/no/longer/exists.pt",
    )
    config["adapt"] = {"keyword": "hey eva"}
    module = Stage2LoraAdaptationModule(
        config,
        vocab_size=VOCAB_SIZE,
        lora_rank=2,
        lora_alpha=4.0,
        restoring_full_checkpoint=True,
    )

    assert module.adapter is not None
    assert module._checkpoint_config["stage2"]["phoneme_adapter"]["init_checkpoint"] == ""


def test_verifier_architecture_matches_the_training_module(monkeypatch, tmp_path):
    """The verifier loads Stage II weights with strict=True, so its module tree
    has to track Stage2LightningModule's. Without this the adapter would be
    dropped at inference and every score would come from an unmapped space."""
    from dma_kws.config import FbankConfig
    from dma_kws.inference.stage2_verifier import Stage2Verifier

    class _FakeEncoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj = nn.Linear(80, ENCODER_DIM)

        def output_frames(self, num_input_frames: int) -> int:
            return num_input_frames

        def forward(self, feat, feat_lengths):
            return _mock_encoder_output(feat, feat_lengths)

    config = _config()
    monkeypatch.setattr("dma_kws.stage2.module.build_encoder", lambda *_a, **_k: _FakeEncoder())
    monkeypatch.setattr(
        "dma_kws.stage2.module.build_qbyt",
        lambda *_a, **kwargs: _RecordingQbyT(
            encoder_output_size=kwargs["input_dim"],
            num_embeds=kwargs["vocab_size"],
        ),
    )
    trained = Stage2LightningModule(config, vocab_size=VOCAB_SIZE, freeze_encoder=True)

    checkpoint = tmp_path / "stage2.pt"
    torch.save(
        stamp_qbyt_readout_version(
            {"model_state_dict": trained.state_dict()},
            alignment=config["stage2"]["qbyt_alignment"],
        ),
        checkpoint,
    )

    monkeypatch.setattr(
        "dma_kws.inference.stage2_verifier.build_encoder", lambda *_a, **_k: _FakeEncoder()
    )
    monkeypatch.setattr(
        "dma_kws.stage2.model_factory.build_qbyt",
        lambda *_a, **kwargs: _RecordingQbyT(
            encoder_output_size=kwargs["input_dim"],
            num_embeds=kwargs["vocab_size"],
        ),
    )

    verifier = Stage2Verifier(
        stage1_cfg=config["stage1"],
        stage2_cfg=config["stage2"],
        demo_cfg={},
        fbank_cfg=FbankConfig(),
        stage2_ckpt=str(checkpoint),
        vocab_size=VOCAB_SIZE,
        device=torch.device("cpu"),
    )

    assert verifier._model.adapter is not None
    assert verifier._model.qbyt.encoder_output_size == TRUNK_DIM


def test_eval_rejects_a_checkpoint_without_adapter_weights(monkeypatch):
    """``strict=False`` at eval time keeps old checkpoints loadable, but a
    randomly initialized trunk still produces scores, just meaningless ones."""
    module = _build(monkeypatch, _config())

    with pytest.raises(SystemExit, match="randomly initialized trunk"):
        assert_adapter_weights_loaded(module, ["adapter.trunk.proj.weight", "qbyt.proj.bias"])


def test_eval_check_is_a_no_op_without_an_adapter(monkeypatch):
    config = _config()
    config["stage2"]["phoneme_adapter"]["enabled"] = False
    module = _build(monkeypatch, config)

    assert_adapter_weights_loaded(module, ["qbyt.proj.bias"])


def test_init_checkpoint_without_adapter_weights_fails(monkeypatch, tmp_path):
    """The LoRA runner loads its base checkpoint through this path and then
    freezes the trunk immediately, so a checkpoint with no adapter.* would leave
    LoRA adapting on top of a random projection."""
    config = _config()
    qbyt_state = {
        f"qbyt.{name}": value
        for name, value in _RecordingQbyT(
            encoder_output_size=TRUNK_DIM,
            num_embeds=VOCAB_SIZE,
        ).state_dict().items()
    }
    checkpoint = tmp_path / "stage2_no_adapter.pt"
    torch.save(
        stamp_qbyt_readout_version(
            {"model_state_dict": qbyt_state},
            alignment=config["stage2"]["qbyt_alignment"],
        ),
        checkpoint,
    )

    with pytest.raises(SystemExit, match="carries no adapter"):
        _build_with_init(monkeypatch, config, init_checkpoint=checkpoint)


def test_init_checkpoint_with_adapter_weights_loads(monkeypatch, tmp_path):
    trained = _build(monkeypatch, _config())
    state = {f"adapter.{k}": v for k, v in trained.adapter.state_dict().items()}
    checkpoint = tmp_path / "stage2_with_adapter.pt"
    torch.save({"model_state_dict": state}, checkpoint)

    reloaded = _build_with_init(monkeypatch, _config(), init_checkpoint=checkpoint)

    torch.testing.assert_close(
        trained.adapter.trunk.proj.weight, reloaded.adapter.trunk.proj.weight
    )


def test_partial_adapter_state_in_init_checkpoint_fails(monkeypatch, tmp_path):
    trained = _build(monkeypatch, _config())
    state = {f"adapter.{k}": v for k, v in trained.adapter.state_dict().items()}
    state.pop("adapter.trunk.proj.weight")
    checkpoint = tmp_path / "stage2_partial.pt"
    torch.save({"model_state_dict": state}, checkpoint)

    with pytest.raises(RuntimeError):
        _build_with_init(monkeypatch, _config(), init_checkpoint=checkpoint)


def test_random_trunk_is_allowed_when_no_checkpoint_is_given(monkeypatch):
    """Training the trunk from scratch inside Stage II is a legitimate variant;
    only a checkpoint that was supposed to supply one and did not is an error."""
    module = _build(monkeypatch, _config(ctc_weight=0.2))

    assert module.adapter is not None


def test_adapter_checkpoint_with_a_different_blank_id_is_rejected(monkeypatch, tmp_path):
    """Nothing downstream would notice: CTC would optimize one symbol while
    collapse/search treat another as blank."""
    module = _build(monkeypatch, _config())
    checkpoint = tmp_path / "adapter.pt"
    torch.save(
        {"model_state_dict": module.adapter.state_dict(), "blank_id": 3}, checkpoint
    )

    config = _config(init_checkpoint=str(checkpoint))
    with pytest.raises(SystemExit, match="blank_id"):
        _build(monkeypatch, config)


def test_mismatched_adapter_checkpoint_fails_loudly(monkeypatch, tmp_path):
    """A trunk of the wrong shape means QbyT would read a different space than
    the one CTC supervised, so this must not degrade to a warning."""
    module = _build(monkeypatch, _config())
    checkpoint = tmp_path / "adapter.pt"
    torch.save({"model_state_dict": module.adapter.state_dict()}, checkpoint)

    config = _config(init_checkpoint=str(checkpoint))
    config["stage2"]["phoneme_adapter"]["trunk"]["output_dim"] = TRUNK_DIM * 2

    with pytest.raises(RuntimeError):
        _build(monkeypatch, config)
