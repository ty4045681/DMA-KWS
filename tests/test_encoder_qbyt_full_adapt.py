"""Encoder + QbyT optimization and exact stochastic continuation contracts."""

from __future__ import annotations

import copy
import os
import random
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from dma_kws.stage2 import adapt as adapt_module
from dma_kws.stage2.collate import train_collate_fn
from dma_kws.stage2.joint_dataset import JointBatchSampler
from dma_kws.stage2.joint_loader import JointDataLoader
from test_qbyt_full_adapt import (
    _Encoder,
    _JointFeatures,
    _checkpoint,
    full_fixture as frozen_full_fixture,
)


class _ScheduledEncoder(_Encoder):
    """Exercise both torch dropout and Python randomness used by Zipformer."""

    def __init__(self, input_dim, output_dim):
        super().__init__(input_dim, output_dim)
        self.batch_count = None
        self.schedule_history = []
        self.random_draws = []

    def set_batch_count(self, batch_count):
        self.batch_count = float(batch_count)
        self.schedule_history.append(self.batch_count)
        return 1

    def forward(self, feat, lengths, **kwargs):
        encoded, mask = super().forward(feat, lengths, **kwargs)
        if self.training:
            gain = 0.9 + 0.2 * random.random()
            self.random_draws.append(gain)
            encoded = encoded * gain
        return encoded, mask


@pytest.fixture
def encoder_fixture(frozen_full_fixture, monkeypatch):
    config, base_path, base = frozen_full_fixture
    config["stage2"]["accumulate_grad_batches"] = 2
    config["adapt"].update({
        "method": "encoder_qbyt_full", "learning_rate": 3e-5,
        "encoder_learning_rate": 3e-6,
        "encoder_schedule": {"start_batch_count": 100000.0, "reference_duration": 600.0},
    })
    monkeypatch.setattr(
        "dma_kws.stage2.module.build_encoder",
        lambda stage1, *, output_dim: _ScheduledEncoder(stage1["input_dim"], output_dim),
    )
    return config, base_path, base


def _model(encoder_fixture, **kwargs):
    config, base_path, _ = encoder_fixture
    return adapt_module.Stage2EncoderQbytFullAdaptationModule(
        config, vocab_size=16, init_checkpoint=base_path, **kwargs,
    )


def _assert_optimizer_equal(actual, expected):
    # Parameter-group names are strings, which torch.assert_close cannot compare.
    assert actual["param_groups"] == expected["param_groups"]
    torch.testing.assert_close(actual["state"], expected["state"], rtol=0, atol=0)


def _batch():
    dataset = _JointFeatures()
    return train_collate_fn([dataset[(domain, 101 + index)] for index, domain in enumerate(dataset.weights)])


def _loader(*, accumulation=2, signature="encoder-qbyt-fixture"):
    data = _JointFeatures()
    return JointDataLoader(
        data, batch_sampler=JointBatchSampler(data, batch_size=4, seed=11),
        data_signature=signature, num_workers=0, accumulation_steps=accumulation,
        collate_fn=train_collate_fn,
    )


@pytest.mark.parametrize("adapter_enabled", [False, True])
def test_encoder_full_step_updates_encoder_and_qbyt_through_frozen_adapter(encoder_fixture, adapter_enabled):
    config, _, _ = encoder_fixture
    config["stage2"]["phoneme_adapter"]["enabled"] = adapter_enabled
    model = _model(encoder_fixture)
    model.train()
    assert model.encoder.training and model.qbyt.training
    assert not model.freeze_encoder
    assert not model._checkpoint_config["stage2"]["freeze_encoder"]
    assert model.ctc_weight == 0.0
    if adapter_enabled:
        assert not model.adapter.training
        assert not any(param.requires_grad for param in model.adapter.parameters())
    assert all(param.requires_grad for param in model.encoder.parameters())
    assert all(param.requires_grad for param in model.qbyt.parameters())
    assert not any("lora_" in name or "parametrizations" in name for name in model.state_dict())

    optimizer = model.configure_optimizers()["optimizer"]
    before = copy.deepcopy(model.state_dict())
    adapter_outputs = []
    if adapter_enabled:
        model.adapter.register_forward_hook(lambda module, args, output: adapter_outputs.append(output[0]))
    model.log = lambda *args, **kwargs: None
    model._log_train_losses = lambda *args, **kwargs: None
    loss = model.training_step(_batch(), 0)
    assert torch.isfinite(loss)
    loss.backward()
    if adapter_enabled:
        assert adapter_outputs and adapter_outputs[-1].requires_grad
        assert all(param.grad is None for param in model.adapter.parameters())
    for weight in (model.encoder.projection.weight, model.qbyt.audio_projection.weight):
        assert weight.grad is not None and torch.isfinite(weight.grad).all()
        assert torch.count_nonzero(weight.grad) > 0
    optimizer.step()
    assert not torch.equal(model.encoder.projection.weight, before["encoder.projection.weight"])
    assert not torch.equal(model.qbyt.audio_projection.weight, before["qbyt.audio_projection.weight"])
    for name, value in model.state_dict().items():
        if name.startswith("adapter."):
            torch.testing.assert_close(value, before[name], rtol=0, atol=0)

    model.eval()
    assert not model.encoder.training and not model.qbyt.training
    model.train()
    model.on_train_epoch_start()
    assert model.encoder.training and model.qbyt.training
    if model.adapter is not None:
        assert not model.adapter.training


def test_encoder_full_optimizer_groups_have_distinct_rates_and_exact_parameter_membership(encoder_fixture):
    model = _model(encoder_fixture)
    result = model.configure_optimizers()
    groups = result["optimizer"].param_groups
    assert [group["name"] for group in groups] == ["qbyt", "encoder"]
    assert [group["lr"] for group in groups] == pytest.approx([3e-5, 3e-6])
    assert {id(param) for param in groups[0]["params"]} == {id(param) for param in model.qbyt.parameters()}
    assert {id(param) for param in groups[1]["params"]} == {id(param) for param in model.encoder.parameters()}
    assert not ({id(param) for param in groups[0]["params"]} & {id(param) for param in groups[1]["params"]})
    result["optimizer"].step()
    result["lr_scheduler"]["scheduler"].step()
    assert groups[0]["lr"] / groups[1]["lr"] == pytest.approx(10.0)


def test_encoder_schedule_uses_consumed_unpadded_frames_and_validation_does_not_advance_it(encoder_fixture):
    config, _, _ = encoder_fixture
    model = _model(encoder_fixture)
    model.train()
    model.log = lambda *args, **kwargs: None
    model._log_train_losses = lambda *args, **kwargs: None
    batch = _batch()
    # Distinct valid lengths ensure padded batch size cannot stand in for duration.
    batch["feat_lengths"] = torch.tensor([8, 7, 6, 5])
    start = config["adapt"]["encoder_schedule"]["start_batch_count"]
    reference = config["adapt"]["encoder_schedule"]["reference_duration"]
    optimizer = model.configure_optimizers()["optimizer"]
    model.training_step(batch, 0).backward()
    with pytest.raises(RuntimeError, match="gradient accumulation|optimizer step"):
        _checkpoint(model)
    model.on_before_optimizer_step(optimizer)
    optimizer.step()
    optimizer.zero_grad()
    assert model.encoder.batch_count == pytest.approx(start)
    assert model._encoder_schedule.consumed_frames == 26
    model.training_step(batch, 1).backward()
    model.on_before_optimizer_step(optimizer)
    optimizer.step()
    optimizer.zero_grad()
    assert model.encoder.batch_count == pytest.approx(start + 26 * 0.01 / reference)
    checkpoint = _checkpoint(model)
    assert checkpoint["encoder_schedule"]["consumed_frames"] == 52
    model.eval()
    model.validation_step(batch, 0)
    assert model._encoder_schedule.consumed_frames == 52
    assert _checkpoint(model)["encoder_schedule"] == checkpoint["encoder_schedule"]


def test_encoder_schedule_advances_by_all_ranks_frame_sum(encoder_fixture, monkeypatch):
    model = _model(encoder_fixture)
    model.train()
    model.log = lambda *args, **kwargs: None
    model._log_train_losses = lambda *args, **kwargs: None
    batch = _batch()
    batch["feat_lengths"] = torch.tensor([8, 7, 6, 5])
    reduced_frames = []

    def reduce(value):
        if value.ndim == 0 and value.dtype == torch.int64:
            reduced_frames.append(int(value))
            # A second rank has a different valid-duration total; a local mean
            # or equal-rank assumption would advance this clock incorrectly.
            return value + 37
        return value

    monkeypatch.setattr(adapt_module, "sum_across_processes", reduce)
    model.training_step(batch, 0)
    assert reduced_frames == [26]
    assert model._encoder_schedule.consumed_frames == 63


@pytest.mark.parametrize("other_method", ["lora", "qbyt_full"])
def test_encoder_full_and_frozen_methods_reject_each_others_full_state_resume(encoder_fixture, other_method):
    config, base_path, _ = encoder_fixture
    encoder_model = _model(encoder_fixture)
    other_config = copy.deepcopy(config)
    other_config["adapt"]["method"] = other_method
    cls = {
        "lora": adapt_module.Stage2LoraAdaptationModule,
        "qbyt_full": adapt_module.Stage2QbytFullAdaptationModule,
    }[other_method]
    other = cls(other_config, vocab_size=16, init_checkpoint=base_path)
    with pytest.raises((SystemExit, ValueError), match="method|checkpoint_kind"):
        encoder_model.on_load_checkpoint(_checkpoint(other))
    with pytest.raises((SystemExit, ValueError), match="method|checkpoint_kind"):
        other.on_load_checkpoint(_checkpoint(encoder_model))


@pytest.mark.parametrize("change", ["encoder_learning_rate", "accumulation", "schedule_start", "reference_duration", "frames", "adapter"])
def test_encoder_full_resume_rejects_changed_optimizer_schedule_or_frozen_adapter(encoder_fixture, change):
    config, base_path, _ = encoder_fixture
    model = _model(encoder_fixture)
    checkpoint = _checkpoint(model)
    model.on_load_checkpoint(checkpoint)
    changed_config = copy.deepcopy(config)
    if change == "encoder_learning_rate":
        changed_config["adapt"]["encoder_learning_rate"] *= 2
    elif change == "accumulation":
        changed_config["stage2"]["accumulate_grad_batches"] *= 2
    elif change == "schedule_start":
        changed_config["adapt"]["encoder_schedule"]["start_batch_count"] += 100
    elif change == "reference_duration":
        changed_config["adapt"]["encoder_schedule"]["reference_duration"] *= 2
    elif change == "frames":
        checkpoint["encoder_schedule"]["consumed_frames"] = -1
    else:
        key = next(key for key in checkpoint["state_dict"] if key.startswith("adapter."))
        checkpoint["state_dict"][key] += 1
    restored = adapt_module.Stage2EncoderQbytFullAdaptationModule(
        changed_config, vocab_size=16, init_checkpoint=base_path,
    )
    with pytest.raises((SystemExit, ValueError), match="optimizer|learning rate|schedule|frames|adapter|frozen"):
        restored.on_load_checkpoint(checkpoint)


def test_encoder_full_stochastic_lightning_resume_matches_uninterrupted_training(encoder_fixture, tmp_path):
    import pytorch_lightning as pl
    config, _, base = encoder_fixture
    observations = {}

    class Observe(pl.Callback):
        def __init__(self, label):
            self.label = label
            observations[label] = {"batches": [], "schedules": [], "start": None}

        def on_train_batch_start(self, trainer, model, batch, batch_idx):
            if observations[self.label]["start"] is None:
                observations[self.label]["start"] = {
                    "model": copy.deepcopy(model.state_dict()),
                    "optimizer": copy.deepcopy(trainer.optimizers[0].state_dict()),
                    "scheduler": copy.deepcopy(trainer.lr_scheduler_configs[0].scheduler.state_dict()),
                    "consumed_frames": model._encoder_schedule.consumed_frames,
                }

        def on_train_batch_end(self, trainer, model, outputs, batch, batch_idx):
            observations[self.label]["batches"].append(batch["feat"].clone())
            observations[self.label]["schedules"].append(model.encoder.batch_count)

    def trainer(steps, label):
        return pl.Trainer(
            accelerator="cpu", devices=1, max_steps=steps, max_epochs=-1,
            callbacks=[Observe(label)], logger=False, enable_checkpointing=False,
            enable_progress_bar=False, enable_model_summary=False,
            use_distributed_sampler=False, accumulate_grad_batches=2,
            limit_val_batches=0, num_sanity_val_steps=0, default_root_dir=tmp_path,
        )

    def seed(value):
        torch.manual_seed(value)
        random.seed(value)
        np.random.seed(value)

    seed(991)
    baseline = _model(encoder_fixture)
    baseline_trainer = trainer(4, "baseline")
    baseline_trainer.fit(baseline, train_dataloaders=_loader())

    seed(991)
    first = _model(encoder_fixture)
    interrupted = trainer(2, "first")
    interrupted.fit(first, train_dataloaders=_loader())
    path = tmp_path / "encoder-resume.ckpt"
    interrupted.save_checkpoint(path)
    saved = torch.load(path, map_location="cpu", weights_only=False)
    assert saved["checkpoint_kind"] == "stage2_encoder_qbyt_full"
    assert saved["encoder_schedule"]["consumed_frames"] == 4 * 4 * 8

    # Rebuilding modules and loaders consumes RNG. The resumed training must use
    # the checkpoint's random state, including encoder dropout and Python draws.
    seed(482)
    resumed = adapt_module.Stage2EncoderQbytFullAdaptationModule(
        config, vocab_size=16, restoring_full_checkpoint=True,
    )
    restored = trainer(4, "resumed")
    restored.fit(resumed, train_dataloaders=_loader(), ckpt_path=path)
    at_resume = observations["resumed"]["start"]
    torch.testing.assert_close(at_resume["model"], saved["state_dict"], rtol=0, atol=0)
    _assert_optimizer_equal(at_resume["optimizer"], saved["optimizer_states"][0])
    assert at_resume["scheduler"] == saved["lr_schedulers"][0]
    assert at_resume["consumed_frames"] == saved["encoder_schedule"]["consumed_frames"]

    combined_batches = observations["first"]["batches"] + observations["resumed"]["batches"]
    assert len(combined_batches) == len(observations["baseline"]["batches"]) == 8
    for actual, expected in zip(combined_batches, observations["baseline"]["batches"]):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert first.encoder.random_draws + resumed.encoder.random_draws == baseline.encoder.random_draws
    assert observations["first"]["schedules"] + observations["resumed"]["schedules"] == observations["baseline"]["schedules"]
    torch.testing.assert_close(resumed.state_dict(), baseline.state_dict(), rtol=0, atol=0)
    _assert_optimizer_equal(restored.optimizers[0].state_dict(), baseline_trainer.optimizers[0].state_dict())
    assert restored.lr_scheduler_configs[0].scheduler.state_dict() == baseline_trainer.lr_scheduler_configs[0].scheduler.state_dict()
    assert resumed._encoder_schedule.consumed_frames == baseline._encoder_schedule.consumed_frames == 8 * 4 * 8
    assert restored.train_dataloader.consumed_batches == 8
    assert not torch.equal(resumed.encoder.projection.weight, base.encoder.projection.weight)
    assert not torch.equal(resumed.qbyt.audio_projection.weight, base.qbyt.audio_projection.weight)
    torch.testing.assert_close(resumed.adapter.state_dict(), base.adapter.state_dict(), rtol=0, atol=0)


def test_real_icefall_zipformer_encoder_qbyt_forward_backward_when_dependencies_available(tmp_path, monkeypatch):
    # This intentionally loads upstream kernels. Do not replace missing k2 with
    # a fake module: it supplies custom activations used by real backward passes.
    pytest.importorskip("k2", reason="Real Zipformer backward requires the optional k2 package")
    local_root = Path(os.environ.get("ICEFALL_ROOT", Path(__file__).resolve().parents[2] / "icefall"))
    if not (local_root / "egs/gigaspeech/KWS/zipformer/zipformer.py").is_file():
        pytest.skip("Local ICEFALL_ROOT Zipformer source is unavailable")
    monkeypatch.setenv("ICEFALL_ROOT", str(local_root))
    from dma_kws.stage2.icefall_encoder import _load_icefall_modules
    try:
        _load_icefall_modules()
    except SystemExit as exc:
        if isinstance(exc.__cause__, ModuleNotFoundError):
            pytest.skip(f"Real Zipformer optional dependency missing: {exc.__cause__.name}")
        raise
    from dma_kws.config import compose_config, config_to_dict
    config = config_to_dict(compose_config(overrides=[
        "+experiment=[icefall_zipformer_stage2_eps_softmin_v41,adapt_joint,adapt_encoder_qbyt_full]",
    ]))
    config["stage1"].update({
        "input_dim": 80, "encoder_dim": "32", "encoder_unmasked_dim": "32",
        "num_encoder_layers": "1", "downsampling_factor": "1", "feedforward_dim": "64",
        "num_heads": "2", "query_head_dim": "8", "value_head_dim": "8",
        "pos_head_dim": "4", "pos_dim": 16, "cnn_module_kernel": "7", "causal": True,
    })
    config["stage2"].update({"encoder_output_dim": 32, "qbyt_embed_dim": 32, "qbyt_layers": 1})
    config["stage2"]["phoneme_adapter"]["enabled"] = False
    config["adapt"]["keyword"] = "hey"
    model = adapt_module.Stage2EncoderQbytFullAdaptationModule(config, vocab_size=16)
    model.train()
    model.log = lambda *args, **kwargs: None
    model._log_train_losses = lambda *args, **kwargs: None
    batch = _batch()
    batch["feat"] = torch.randn(4, 63, 80)
    batch["feat_lengths"] = torch.tensor([63, 59, 55, 51])
    loss = model.training_step(batch, 0)
    assert torch.isfinite(loss)
    loss.backward()
    encoder_grads = [param.grad for param in model.encoder.parameters() if param.grad is not None]
    assert encoder_grads and all(torch.isfinite(grad).all() for grad in encoder_grads)
    assert sum(int(torch.count_nonzero(grad)) for grad in encoder_grads) > 0
    assert model.qbyt.audio_projection.weight.grad is not None
    assert model._encoder_schedule.consumed_frames == int(batch["feat_lengths"].sum())


@pytest.mark.parametrize("saved_device", ["cuda", "mps"])
def test_encoder_full_resume_rejects_changed_accelerator_before_training(encoder_fixture, saved_device):
    model = _model(encoder_fixture)
    checkpoint = _checkpoint(model)
    assert checkpoint["adapt_training_device"] == "cpu"
    checkpoint["adapt_training_device"] = saved_device
    model.on_load_checkpoint(checkpoint)
    random_state = torch.get_rng_state().clone()
    with pytest.raises(SystemExit, match="device|accelerator|weights-only"):
        model.on_train_start()
    torch.testing.assert_close(torch.get_rng_state(), random_state, rtol=0, atol=0)
