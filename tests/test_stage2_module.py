"""Stage-II integration tests for the single QbyT v6 alignment path."""

from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

pytest.importorskip("pytorch_lightning")

from dma_kws.inference.score_calibration import PositiveAffineCalibrator
from dma_kws.stage2.module import Stage2LightningModule


def _config(*, span: int = 30) -> dict:
    return {
        "stage1": {
            "input_dim": 80,
            "encoder_output_dim": 16,
            "attention_heads": 2,
            "linear_units": 32,
            "num_blocks": 1,
            "dropout_rate": 0.0,
            "positional_dropout_rate": 0.0,
            "attention_dropout_rate": 0.0,
            "cnn_module_kernel": 3,
        },
        "stage2": {
            "encoder_output_dim": 16,
            "qbyt_embed_dim": 8,
            "qbyt_layers": 1,
            "learning_rate": 1e-3,
            "warmup_steps": 2,
            "total_scheduler_steps": 10,
            "max_steps": 10,
            "qbyt_alignment": {
                "topology": "keyword_filler_segmental_crf_v1",
                "min_phone_duration_frames": 1,
                "max_phone_duration_frames": 4,
                "max_inter_phone_gap_frames": 1,
                "max_keyword_span_frames": span,
                "weakest_phone_temperature": 0.2,
                "weakest_phone_weight": 1.0,
                "local_context_kernel": 5,
            },
            "sequence_loss": {
                "target_mode": "ordered_contiguous_prefix",
                "progress_weight": 0.3,
                "normalization": "sample",
            },
        },
    }


def _pooling_config() -> dict:
    cfg = _config()
    stage2 = cfg["stage2"]
    stage2.pop("qbyt_alignment", None)
    stage2["qbyt_readout_version"] = 4
    stage2["qbyt_readout"] = {"mode": "eps_softmin", "temperature": 1.0}
    stage2["background_negative"] = {
        "enabled": True,
        "probability": 0.25,
        "audio_list_path": "/tmp/x.list",
    }
    stage2["negative_tail_loss"] = {
        "enabled": True,
        "weight": 0.5,
        "fraction": 0.1,
    }
    return cfg


def _bounded_config() -> dict:
    cfg = _config()
    stage2 = cfg["stage2"]
    stage2["qbyt_readout_version"] = 5
    stage2["qbyt_alignment"] = {
        "topology": "bounded_segmental_v1",
        "temperature": 0.2,
        "min_phone_duration_frames": 1,
        "max_phone_duration_frames": 4,
        "max_inter_phone_gap_frames": 1,
        "max_keyword_span_frames": 30,
        "local_context_kernel": 5,
    }
    return cfg


class _Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(80, 16)

    def forward(self, feat, feat_lengths):
        encoded = self.projection(feat)
        mask = torch.arange(feat.size(1)).unsqueeze(0) < feat_lengths.unsqueeze(1)
        return encoded, mask.unsqueeze(1)


class _FakeQbyT(nn.Module):
    def __init__(self):
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(()))

    def forward(self, speech, text, speech_lengths=None, text_lengths=None):
        del speech_lengths
        batch = speech.size(0)
        width = text.size(1)
        prefixes = self.dummy.expand(batch, width)
        lengths = text.ne(0).sum(1) if text_lengths is None else text_lengths
        last = (lengths - 1).clamp_min(0)
        logits = prefixes.gather(1, last.unsqueeze(1)).squeeze(1)
        return logits, prefixes


def _patch_model(monkeypatch):
    monkeypatch.setattr("dma_kws.stage2.module.build_encoder", lambda *_a, **_k: _Encoder())
    monkeypatch.setattr(
        "dma_kws.stage2.module.build_qbyt", lambda *_a, **_k: _FakeQbyT()
    )


def _batch():
    return {
        "feat": torch.randn(2, 8, 80),
        "feat_lengths": torch.tensor([8, 6]),
        "anchor": torch.tensor([[3, 4, 5], [6, 7, 0]]),
        "label": torch.tensor([1, 0]),
        "seq_label": torch.tensor([[1, 1, 1], [1, 0, -1]]),
        "seq_label_mask": torch.tensor([[1, 1, 1], [1, 1, 0]], dtype=torch.float32),
    }


@pytest.fixture
def module(monkeypatch):
    _patch_model(monkeypatch)
    return Stage2LightningModule(_config(), vocab_size=20)


def test_forward_and_training_step_smoke(module):
    batch = _batch()
    logits, prefixes = module(batch["feat"], batch["feat_lengths"], batch["anchor"])
    assert logits.shape == (2,)
    assert prefixes.shape == (2, 3)

    module.optimizers = MagicMock(
        return_value=MagicMock(param_groups=[{"lr": 1e-3}])
    )
    loss = module.training_step(batch, 0)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_training_masks_structurally_illegal_paths(monkeypatch):
    import dma_kws.stage2.module as stage2_module

    _patch_model(monkeypatch)
    module = Stage2LightningModule(_config(span=4), vocab_size=20)
    real_compute_losses = stage2_module.compute_stage2_losses
    captured = {}

    def _capture_valid_path_mask(**kwargs):
        captured["valid_path_mask"] = kwargs.get("valid_path_mask")
        return real_compute_losses(**kwargs)

    monkeypatch.setattr(
        stage2_module,
        "compute_stage2_losses",
        _capture_valid_path_mask,
    )
    batch = {
        "feat": torch.randn(3, 8, 80),
        "feat_lengths": torch.tensor([8, 2, 8]),
        "anchor": torch.tensor(
            [
                [3, 4, 5, 0, 0],
                [6, 7, 8, 0, 0],
                [9, 10, 11, 12, 13],
            ]
        ),
        "label": torch.tensor([1, 0, 1]),
        "seq_label": torch.tensor(
            [
                [1, 1, 1, -1, -1],
                [1, 0, 0, -1, -1],
                [1, 1, 1, 1, 1],
            ]
        ),
        "seq_label_mask": torch.tensor(
            [
                [1, 1, 1, 0, 0],
                [1, 1, 1, 0, 0],
                [1, 1, 1, 1, 1],
            ],
            dtype=torch.float32,
        ),
    }
    # min duration is one frame: sample 1 is valid, sample 2 has too few
    # encoder frames, and sample 3 exceeds max_keyword_span_frames=4.

    _, losses, _ = module._forward_train_losses(batch)

    assert captured["valid_path_mask"].tolist() == [True, False, False]
    assert losses["utt_sample_mask"].tolist() == [True, False, False]
    assert torch.allclose(losses["illegal_path_rate"], torch.tensor(2.0 / 3.0))


def test_module_stores_only_the_canonical_alignment_spec(module):
    assert module.qbyt_alignment.topology == "keyword_filler_segmental_crf_v1"
    assert module.qbyt_alignment.max_keyword_span_frames == 30
    stage2 = module._checkpoint_config["stage2"]
    assert stage2["qbyt_alignment"] == module.qbyt_alignment.as_dict()
    assert "qbyt_readout" not in stage2
    assert "allow_legacy_qbyt_readout" not in stage2


def test_external_calibration_applies_only_at_score_diagnostics(monkeypatch, tmp_path):
    _patch_model(monkeypatch)
    calibration_path = tmp_path / "calibration.json"
    PositiveAffineCalibrator(slope=2.0, bias=-1.0).save_json(calibration_path)
    config = _config()
    config["prep"] = {"stage2_calibration": str(calibration_path)}
    module = Stage2LightningModule(config, vocab_size=20)

    class _CaptureMetric:
        def update(self, scores, labels, sample_ids):
            self.scores = scores
            self.labels = labels
            self.sample_ids = sample_ids

    metric = _CaptureMetric()
    logits = torch.tensor([-1.0, 2.0])
    labels = torch.tensor([0, 1])
    module._update_score_diagnostics(metric, logits=logits, labels=labels)

    assert torch.allclose(metric.scores, torch.sigmoid(2.0 * logits - 1.0))
    assert torch.equal(metric.labels, labels)
    assert metric.sample_ids is None


def test_freeze_encoder_disables_encoder_gradients(monkeypatch):
    _patch_model(monkeypatch)
    module = Stage2LightningModule(_config(), vocab_size=20, freeze_encoder=True)
    assert all(not parameter.requires_grad for parameter in module.encoder.parameters())


def test_encoder_only_checkpoint_is_a_valid_warm_start(monkeypatch, tmp_path):
    _patch_model(monkeypatch)
    module = Stage2LightningModule(_config(), vocab_size=20)
    state = {
        f"encoder.{name}": torch.ones_like(value)
        for name, value in module.encoder.state_dict().items()
    }
    path = tmp_path / "encoder_only.pt"
    torch.save({"model_state_dict": state}, path)
    module._load_init_checkpoint(path)
    assert all(torch.equal(value, torch.ones_like(value)) for value in module.encoder.state_dict().values())


def test_pre_v6_qbyt_checkpoint_is_rejected(monkeypatch, tmp_path):
    _patch_model(monkeypatch)
    module = Stage2LightningModule(_config(), vocab_size=20)
    path = tmp_path / "v4.pt"
    torch.save(
        {
            "qbyt_readout_version": 4,
            "model_state_dict": {"qbyt.dummy": torch.zeros(())},
        },
        path,
    )
    with pytest.raises(SystemExit, match="version 4"):
        module._load_init_checkpoint(path)


def test_full_v6_checkpoint_loads_strictly(monkeypatch, tmp_path):
    from dma_kws.training.checkpoint_io import stamp_qbyt_readout_version

    _patch_model(monkeypatch)
    module = Stage2LightningModule(_config(), vocab_size=20)
    state = module.state_dict()
    state["qbyt.dummy"] = torch.ones(())
    payload = {
        "model_state_dict": state,
        "config": module._checkpoint_config,
    }
    stamp_qbyt_readout_version(payload, alignment=module.qbyt_alignment)
    path = tmp_path / "v6.pt"
    torch.save(payload, path)
    module._load_init_checkpoint(path, require_full_qbyt=True)
    assert module.qbyt.dummy.item() == 1.0


def test_full_init_requires_encoder_and_qbyt(monkeypatch, tmp_path):
    _patch_model(monkeypatch)
    module = Stage2LightningModule(_config(), vocab_size=20)
    path = tmp_path / "encoder_only.pt"
    torch.save({"model_state_dict": {"encoder.projection.weight": torch.zeros_like(module.encoder.projection.weight)}}, path)
    with pytest.raises(SystemExit, match="qbyt"):
        module._load_init_checkpoint(path, require_full_qbyt=True)


def test_saved_checkpoint_stamps_complete_alignment(module):
    from dma_kws.training.checkpoint_io import (
        QBYT_ALIGNMENT_SPEC_KEY,
        QBYT_READOUT_VERSION,
        QBYT_READOUT_VERSION_KEY,
    )

    checkpoint = {"state_dict": module.state_dict()}
    module.on_save_checkpoint(checkpoint)
    assert checkpoint[QBYT_READOUT_VERSION_KEY] == QBYT_READOUT_VERSION == 7
    assert checkpoint[QBYT_ALIGNMENT_SPEC_KEY] == module.qbyt_alignment.as_dict()
    module.on_load_checkpoint(checkpoint)


def test_restore_with_different_alignment_is_rejected(monkeypatch):
    from dma_kws.training.checkpoint_io import stamp_qbyt_readout_version

    _patch_model(monkeypatch)
    module = Stage2LightningModule(_config(span=30), vocab_size=20)
    other = Stage2LightningModule(_config(span=20), vocab_size=20)
    checkpoint = {"state_dict": other.state_dict(), "config": other._checkpoint_config}
    stamp_qbyt_readout_version(checkpoint, alignment=other.qbyt_alignment)
    with pytest.raises(SystemExit, match="current config expects"):
        module.on_load_checkpoint(checkpoint)


@pytest.mark.parametrize(
    ("config_fn", "family"),
    [(_pooling_config, "pooling"), (_bounded_config, "bounded")],
)
def test_forward_train_losses_utt_sample_mask_is_all_valid_without_path_filter(
    monkeypatch, config_fn, family
):
    _patch_model(monkeypatch)
    module = Stage2LightningModule(config_fn(), vocab_size=20)
    assert module.qbyt_score.family == family

    _, losses, _ = module._forward_train_losses(_batch())

    assert losses["utt_sample_mask"].tolist() == [True, True]
    assert "valid_path_mask" not in losses


def test_pooling_allows_background_negative_and_negative_tail(monkeypatch):
    _patch_model(monkeypatch)
    module = Stage2LightningModule(_pooling_config(), vocab_size=20)
    assert module.qbyt_score.family == "pooling"
    assert module.negative_tail_weight == 0.5
    assert module.negative_tail_fraction == 0.1


def test_keyword_filler_still_allows_background_negative_and_negative_tail(
    monkeypatch,
):
    _patch_model(monkeypatch)
    cfg = _config()
    cfg["stage2"]["background_negative"] = {
        "enabled": True,
        "probability": 0.25,
        "audio_list_path": "/tmp/x.list",
    }
    cfg["stage2"]["negative_tail_loss"] = {
        "enabled": True,
        "weight": 0.5,
        "fraction": 0.1,
    }
    module = Stage2LightningModule(cfg, vocab_size=20)
    assert module.qbyt_score.family == "keyword_filler"
    assert module.negative_tail_weight == 0.5


@pytest.mark.parametrize(
    "knob",
    [
        {
            "negative_tail_loss": {
                "enabled": True,
                "weight": 0.5,
                "fraction": 0.1,
            }
        },
        {
            "background_negative": {
                "enabled": True,
                "probability": 0.25,
                "audio_list_path": "/tmp/x.list",
            }
        },
    ],
)
def test_bounded_rejects_background_negative_and_negative_tail(monkeypatch, knob):
    _patch_model(monkeypatch)
    cfg = _bounded_config()
    cfg["stage2"].update(knob)
    with pytest.raises(ValueError, match="not supported for bounded"):
        Stage2LightningModule(cfg, vocab_size=20)
