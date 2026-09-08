"""Numerical full-QbyT training, checkpoint identity and continuation contracts."""

from __future__ import annotations

import copy
import sys
from types import ModuleType

import pytest
import torch
from torch import nn
from torch.utils.data import Dataset

from dma_kws.config import compose_config, config_to_dict
from dma_kws.stage2.adapt import Stage2LoraAdaptationModule, Stage2QbytFullAdaptationModule
from dma_kws.stage2.collate import train_collate_fn
from dma_kws.stage2.joint_dataset import JointBatchSampler
from dma_kws.stage2.joint_loader import JointDataLoader
from dma_kws.stage2.module import Stage2LightningModule
from dma_kws.training.checkpoint_io import stamp_qbyt_readout_version


class _Encoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.projection = nn.Linear(input_dim, output_dim)
        self.dropout = nn.Dropout(0.5)

    def forward(self, feat, lengths, **kwargs):
        mask = torch.arange(feat.shape[1], device=feat.device)[None] < lengths[:, None]
        return self.dropout(self.projection(feat)), mask[:, None]

    def apply_stream_config(self, chunk_sizes, left_context_frames):
        pass  # The fixture has no time-dependent receptive field.


@pytest.fixture
def full_fixture(tmp_path, monkeypatch):
    config = config_to_dict(compose_config(overrides=[
        "+experiment=[icefall_zipformer_stage2_eps_softmin_v41,adapt_joint,adapt_qbyt_full]",
    ]))
    config["stage1"].update({"input_dim": 8, "causal": True})
    config["stage2"].update({
        "encoder_output_dim": 8, "qbyt_embed_dim": 8, "qbyt_layers": 1,
        "precision": "32-true", "log_interval": 1,
    })
    # Deliberately request a trainable phoneme adapter in the inherited Stage II
    # config: adaptation must still freeze it and disable auxiliary CTC.
    config["stage2"]["phoneme_adapter"].update({
        "enabled": True, "freeze": False, "init_checkpoint": "", "ctc_weight": 1.0,
        "trunk": {"type": "linear", "output_dim": 8, "dropout": 0.5},
    })
    config["adapt"].update({
        "keyword": "hey", "learning_rate": 1e-3, "max_steps": 4,
        "warmup_steps": 0, "optimizer": "adamw", "weight_decay": 0.01,
    })
    monkeypatch.setattr(
        "dma_kws.stage2.module.build_encoder",
        lambda stage1, *, output_dim: _Encoder(stage1["input_dim"], output_dim),
    )
    transformers = ModuleType("transformers")
    transformers.get_cosine_schedule_with_warmup = (
        lambda optimizer, **kwargs: torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda step: 1.0 / (1 + step),
        )
    )
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    torch.manual_seed(19)
    base = Stage2LightningModule(config, vocab_size=16, freeze_encoder=True)
    base_path = tmp_path / "base.pt"
    torch.save(stamp_qbyt_readout_version({
        "model_state_dict": base.state_dict(), "config": base._checkpoint_config,
    }, alignment=base.qbyt_score), base_path)
    return config, base_path, base


def _full_model(full_fixture):
    config, path, _ = full_fixture
    return Stage2QbytFullAdaptationModule(config, vocab_size=16, init_checkpoint=path)


def _checkpoint(model):
    checkpoint = {"state_dict": copy.deepcopy(model.state_dict())}
    model.on_save_checkpoint(checkpoint)
    return checkpoint


def test_full_qbyt_optimizes_every_qbyt_parameter_and_keeps_feature_trunk_in_eval(full_fixture):
    model = _full_model(full_fixture)
    model.train()
    assert model.qbyt.training
    assert not model.encoder.training
    assert not model.adapter.training
    assert model.ctc_weight == 0.0
    assert all(param.requires_grad for param in model.qbyt.parameters())
    assert not any(param.requires_grad for param in model.encoder.parameters())
    assert not any(param.requires_grad for param in model.adapter.parameters())
    assert not any("lora_" in name or "parametrizations" in name for name in model.state_dict())
    optimizer = model.configure_optimizers()["optimizer"]
    optimized = {id(param) for group in optimizer.param_groups for param in group["params"]}
    assert optimized == {id(param) for param in model.qbyt.parameters()}
    feature = torch.randn(2, 8, 8)
    lengths = torch.tensor([8, 6])
    torch.testing.assert_close(model.encoder(feature, lengths)[0], model.encoder(feature, lengths)[0], rtol=0, atol=0)


@pytest.mark.parametrize("missing", ["encoder", "qbyt", "one_encoder_parameter", "one_qbyt_parameter"])
def test_full_qbyt_initialization_rejects_incomplete_base(full_fixture, tmp_path, missing):
    config, _, base = full_fixture
    state = copy.deepcopy(base.state_dict())
    if missing in {"encoder", "qbyt"}:
        state = {key: value for key, value in state.items() if not key.startswith(missing + ".")}
    else:
        del state["encoder.projection.weight" if missing == "one_encoder_parameter" else "qbyt.audio_projection.weight"]
    path = tmp_path / "partial.pt"
    torch.save(stamp_qbyt_readout_version({
        "model_state_dict": state, "config": base._checkpoint_config,
    }, alignment=base.qbyt_score), path)
    with pytest.raises((SystemExit, RuntimeError), match="complete|Missing key"):
        Stage2QbytFullAdaptationModule(config, vocab_size=16, init_checkpoint=path)


@pytest.mark.parametrize("adapter_only", [False, True])
def test_full_qbyt_initialization_rejects_unmerged_lora(full_fixture, tmp_path, adapter_only):
    config, path, _ = full_fixture
    lora_config = copy.deepcopy(config)
    lora_config["adapt"]["method"] = "lora"
    lora = Stage2LoraAdaptationModule(lora_config, vocab_size=16, init_checkpoint=path, lora_rank=2, lora_alpha=4)
    payload = _checkpoint(lora)
    if adapter_only:
        from dma_kws.training.lora import lora_state_dict
        payload.pop("state_dict")
        payload["lora_state_dict"] = lora_state_dict(lora.qbyt)
    path = tmp_path / "unmerged.pt"
    torch.save(payload, path)
    with pytest.raises(SystemExit, match="unmerged LoRA|Merge LoRA"):
        Stage2QbytFullAdaptationModule(config, vocab_size=16, init_checkpoint=path)


def test_full_qbyt_accepts_a_merged_lora_model_as_weights_only_initialization(full_fixture, tmp_path):
    from dma_kws.training.lora import merge_lora
    config, path, _ = full_fixture
    lora_config = copy.deepcopy(config)
    lora_config["adapt"]["method"] = "lora"
    lora = Stage2LoraAdaptationModule(lora_config, vocab_size=16, init_checkpoint=path, lora_rank=2, lora_alpha=4)
    with torch.no_grad():
        for name, param in lora.named_parameters():
            if name.endswith(".lora_B"):
                param.fill_(0.1)
    merge_lora(lora.qbyt)
    payload = stamp_qbyt_readout_version({
        "model_state_dict": lora.state_dict(), "config": lora._checkpoint_config,
        "method": "lora", "phase": "joint",
    }, alignment=lora.qbyt_score)
    merged_path = tmp_path / "merged.pt"
    torch.save(payload, merged_path)
    model = Stage2QbytFullAdaptationModule(config, vocab_size=16, init_checkpoint=merged_path)
    torch.testing.assert_close(model.state_dict(), lora.state_dict(), rtol=0, atol=0)
    assert _checkpoint(model)["method"] == "qbyt_full"


@pytest.mark.parametrize("external_exists", [False, True])
def test_full_qbyt_uses_selected_model_adapter_despite_stale_external_init(full_fixture, tmp_path, external_exists):
    config, base_path, base = full_fixture
    external = tmp_path / "external-adapter.pt"
    if external_exists:
        different = {
            name: value + 10 for name, value in base.state_dict().items()
            if name.startswith("adapter.")
        }
        torch.save({"model_state_dict": different, "config": base._checkpoint_config}, external)
    config["stage2"]["phoneme_adapter"]["init_checkpoint"] = str(external)
    model = Stage2QbytFullAdaptationModule(config, vocab_size=16, init_checkpoint=base_path)
    torch.testing.assert_close(model.adapter.state_dict(), base.adapter.state_dict(), rtol=0, atol=0)


def test_full_qbyt_rejects_model_missing_adapter_even_with_external_adapter(full_fixture, tmp_path):
    config, _, base = full_fixture
    external = tmp_path / "external-adapter.pt"
    torch.save({"model_state_dict": {
        name: value for name, value in base.state_dict().items() if name.startswith("adapter.")
    }, "config": base._checkpoint_config}, external)
    config["stage2"]["phoneme_adapter"]["init_checkpoint"] = str(external)
    incomplete = tmp_path / "without-adapter.pt"
    torch.save(stamp_qbyt_readout_version({
        "model_state_dict": {
            name: value for name, value in base.state_dict().items() if not name.startswith("adapter.")
        }, "config": base._checkpoint_config,
    }, alignment=base.qbyt_score), incomplete)
    with pytest.raises(SystemExit, match="no adapter|adapter.*weights"):
        Stage2QbytFullAdaptationModule(config, vocab_size=16, init_checkpoint=incomplete)


def test_full_qbyt_and_lora_reject_each_others_training_checkpoint(full_fixture):
    config, path, _ = full_fixture
    full = _full_model(full_fixture)
    lora_config = copy.deepcopy(config)
    lora_config["adapt"]["method"] = "lora"
    lora = Stage2LoraAdaptationModule(lora_config, vocab_size=16, init_checkpoint=path, lora_rank=2, lora_alpha=4)
    with pytest.raises(SystemExit, match="method|checkpoint_kind"):
        full.on_load_checkpoint(_checkpoint(lora))
    with pytest.raises(SystemExit, match="method|checkpoint_kind"):
        lora.on_load_checkpoint(_checkpoint(full))


@pytest.mark.parametrize("change", ["method", "keyword", "phase", "trunk", "learning_rate", "readout"])
def test_full_qbyt_resume_rejects_changed_training_identity(full_fixture, change):
    model = _full_model(full_fixture)
    checkpoint = _checkpoint(model)
    model.on_load_checkpoint(checkpoint)
    if change in {"method", "keyword", "phase"}:
        checkpoint[change] = {"method": "lora", "keyword": "different", "phase": "tts"}[change]
    elif change == "trunk":
        checkpoint["state_dict"]["encoder.projection.weight"] += 1
    elif change == "learning_rate":
        model._adapt_cfg["learning_rate"] *= 2
    else:
        checkpoint["config"]["stage2"]["qbyt_readout"]["temperature"] = 0.5
    with pytest.raises((SystemExit, ValueError), match="method|keyword|phase|trunk|learning rate|readout"):
        model.on_load_checkpoint(checkpoint)


class _JointFeatures(Dataset):
    weights = {"real": 0.25, "tts": 0.25, "libri": 0.25, "musan": 0.25}

    def __len__(self):
        return 32

    def __getitem__(self, ticket):
        domain, seed = ticket
        domain_id = list(self.weights).index(domain)
        generator = torch.Generator().manual_seed(seed)
        return {
            "feat": torch.randn(8, 8, generator=generator),
            "anchor_seq": torch.tensor([1, 2]), "query_seq": torch.tensor([1, 2]),
            "seq_label": torch.tensor([0, 1]), "label": int(domain_id < 2),
            "source": int(domain_id < 2), "domain_source": domain_id,
        }


def test_full_qbyt_lightning_resume_restores_optimizer_scheduler_and_joint_suffix(full_fixture, tmp_path):
    import pytorch_lightning as pl
    config, _, base = full_fixture
    accumulation = 2

    def loader():
        data = _JointFeatures()
        return JointDataLoader(
            data, batch_sampler=JointBatchSampler(data, batch_size=4, seed=11),
            data_signature="full-qbyt-fixture", num_workers=0,
            accumulation_steps=accumulation, collate_fn=train_collate_fn,
        )

    snapshots = []
    seen = []

    class Observe(pl.Callback):
        def on_train_start(self, trainer, model):
            snapshots.append({
                "model": copy.deepcopy(model.state_dict()),
                "optimizer": copy.deepcopy(trainer.optimizers[0].state_dict()),
                "scheduler": copy.deepcopy(trainer.lr_scheduler_configs[0].scheduler.state_dict()),
            })

        def on_train_batch_end(self, trainer, model, outputs, batch, batch_idx):
            seen.append(batch["feat"].clone())

    def trainer(steps):
        return pl.Trainer(
            accelerator="cpu", devices=1, max_steps=steps, max_epochs=-1,
            callbacks=[Observe()], logger=False, enable_checkpointing=False,
            enable_progress_bar=False, enable_model_summary=False,
            use_distributed_sampler=False, accumulate_grad_batches=accumulation,
            limit_val_batches=0, num_sanity_val_steps=0, default_root_dir=tmp_path,
        )

    first = _full_model(full_fixture)
    interrupted = trainer(2)
    interrupted.fit(first, train_dataloaders=loader())
    path = tmp_path / "resume.ckpt"
    interrupted.save_checkpoint(path)
    saved = torch.load(path, map_location="cpu", weights_only=False)
    assert saved["checkpoint_kind"] == "stage2_qbyt_full"
    assert saved["optimizer_states"][0]["state"]
    resumed = Stage2QbytFullAdaptationModule(config, vocab_size=16, restoring_full_checkpoint=True)
    restored = trainer(4)
    restored.fit(resumed, train_dataloaders=loader(), ckpt_path=path)

    torch.testing.assert_close(snapshots[1]["model"], saved["state_dict"], rtol=0, atol=0)
    torch.testing.assert_close(snapshots[1]["optimizer"], saved["optimizer_states"][0], rtol=0, atol=0)
    assert snapshots[1]["scheduler"] == saved["lr_schedulers"][0]
    expected = list(loader())
    assert len(seen) == len(expected) == 8
    for actual, batch in zip(seen, expected):
        torch.testing.assert_close(actual, batch["feat"], rtol=0, atol=0)
    assert restored.global_step == 4
    assert restored.train_dataloader.consumed_batches == 8
    assert not torch.equal(resumed.qbyt.audio_projection.weight, base.qbyt.audio_projection.weight)
    torch.testing.assert_close(resumed.encoder.state_dict(), base.encoder.state_dict(), rtol=0, atol=0)
    torch.testing.assert_close(resumed.adapter.state_dict(), base.adapter.state_dict(), rtol=0, atol=0)
