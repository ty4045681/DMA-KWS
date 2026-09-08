"""Joint adaptation integration: diagnostics and real Lightning loader resumes."""

import pytest
import torch
from torch.utils.data import Dataset

from dma_kws.stage2.adapt import (
    Stage2LoraAdaptationModule,
    Stage2QbytFullAdaptationModule,
    grouped_domain_bce_totals,
)
from dma_kws.stage2.joint_dataset import JointBatchSampler
from dma_kws.stage2.joint_loader import JointDataLoader, JointEvalSampler
from dma_kws.training.score_diagnostics import BinaryScoreDiagnostics


class _Tickets(Dataset):
    weights = {"real": 0.3, "tts": 0.2, "libri": 0.4, "musan": 0.1}

    def __len__(self):
        return 80

    def __getitem__(self, ticket):
        kind, seed = ticket
        return torch.tensor([list(self.weights).index(kind), seed % 100000], dtype=torch.long)


def _serialize_tickets(batch):
    # Exercise real worker prefetch without requiring macOS shared-memory IPC.
    return torch.stack(batch).numpy()


def _loader(signature="fixture-v1", num_workers=0, accumulation_steps=1):
    dataset = _Tickets()
    return JointDataLoader(
        dataset, batch_sampler=JointBatchSampler(dataset, batch_size=8, seed=11),
        data_signature=signature, num_workers=num_workers,
        accumulation_steps=accumulation_steps,
        **({"collate_fn": _serialize_tickets} if num_workers else {}),
    )


def test_joint_loader_checkpoints_consumed_not_prefetched_batches():
    loader = _loader(num_workers=2)
    iterator = iter(loader)
    first = torch.as_tensor(next(iterator))
    loader.mark_consumed()
    state = loader.state_dict()
    expected = torch.as_tensor(next(iterator))
    # Prefetch may have issued the entire tiny epoch. Only one draw is consumed.
    assert state["consumed_batches"] == 1
    restored = _loader()
    restored.load_state_dict(state)
    assert torch.equal(next(iter(restored)), expected)
    assert not torch.equal(first, expected)
    assert len(restored) == 10


def test_joint_loader_refuses_changed_resume_policy():
    state = _loader().state_dict()
    with pytest.raises(ValueError, match="data_signature"):
        _loader(signature="different-data").load_state_dict(state)
    state["world_size"] = 2
    with pytest.raises(ValueError, match="world_size"):
        _loader().load_state_dict(state)


def test_joint_data_signature_binds_resolved_replay_and_tokenizer(tmp_path):
    from dma_kws.stage2.adapt import _joint_data_signature
    config = {"adapt": {"keyword": "hello", "joint": {}}, "stage2": {}}
    parquet, dictionary = tmp_path / "pairs.parquet", tmp_path / "dict.txt"
    parquet.write_bytes(b"pairs-v1")
    dictionary.write_text("a 1\n")

    def signature(wav_dir="features"):
        return _joint_data_signature(
            config, {}, parquet_file=parquet, dict_path=dictionary, wav_dir=tmp_path / wav_dir,
        )

    initial = signature()
    assert initial != signature("another-feature-root")
    parquet.write_bytes(b"pairs-v2")
    assert initial != signature()
    parquet.write_bytes(b"pairs-v1")
    dictionary.write_text("a 2\n")
    assert initial != signature()


@pytest.mark.parametrize("accumulation_steps", [1, 2])
def test_lightning_joint_resume_replays_exact_suffix_and_optimizer(tmp_path, accumulation_steps):
    import pytorch_lightning as pl

    class Model(pl.LightningModule):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(0.2))
            self.seen = []

        def training_step(self, batch, batch_idx):
            self.seen.append(batch.clone())
            self.trainer.train_dataloader.mark_consumed()
            x = batch[:, 1].float().mean() / 100000
            return (self.weight - x).square()

        def configure_optimizers(self):
            return torch.optim.Adam(self.parameters(), lr=0.01)

    def trainer(steps):
        return pl.Trainer(
            accelerator="cpu", devices=1, max_steps=steps, max_epochs=-1,
            logger=False, enable_checkpointing=False, enable_progress_bar=False,
            enable_model_summary=False, use_distributed_sampler=False,
            accumulate_grad_batches=accumulation_steps,
            num_sanity_val_steps=0, default_root_dir=tmp_path,
        )

    baseline = Model()
    trainer(13).fit(baseline, train_dataloaders=_loader(accumulation_steps=accumulation_steps))
    first = Model()
    interrupted = trainer(3)
    interrupted.fit(first, train_dataloaders=_loader(accumulation_steps=accumulation_steps))
    path = tmp_path / "resume.ckpt"
    interrupted.save_checkpoint(path)
    resumed = Model()
    restored_trainer = trainer(13)
    restored_trainer.fit(resumed, train_dataloaders=_loader(accumulation_steps=accumulation_steps), ckpt_path=path)

    assert len(first.seen) == 3 * accumulation_steps
    assert len(resumed.seen) == 10 * accumulation_steps
    for actual, expected in zip(first.seen + resumed.seen, baseline.seen):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(resumed.weight, baseline.weight, rtol=0, atol=0)
    assert restored_trainer.train_dataloader.consumed_batches == 13 * accumulation_steps


def test_joint_loader_rejects_incomplete_accumulation_checkpoint():
    loader = _loader(accumulation_steps=2)
    state = loader.state_dict()
    state["consumed_batches"] = 1
    with pytest.raises(ValueError, match="optimizer boundaries"):
        loader.load_state_dict(state)
    loader.mark_consumed()
    with pytest.raises(ValueError, match="optimizer boundaries"):
        loader.state_dict()
    with pytest.raises(ValueError, match="divisible"):
        len(_loader(accumulation_steps=3))


def test_joint_eval_rank_padding_has_stable_ids(monkeypatch):
    import dma_kws.stage2.joint_loader as module
    monkeypatch.setattr(module, "world_size", lambda: 2)
    monkeypatch.setattr(module, "process_rank", lambda: 0)
    assert list(JointEvalSampler(range(5))) == [0, 2, 4]
    monkeypatch.setattr(module, "process_rank", lambda: 1)
    assert list(JointEvalSampler(range(5))) == [1, 3, 0]


def test_four_domain_losses_use_audio_source_and_valid_sample_mask():
    logits = torch.tensor([0., 1., 2., 3., 50.])
    labels = torch.tensor([1, 0, 1, 0, 0])
    domains = torch.tensor([0, 1, 2, 3, 3])
    valid = torch.tensor([True, True, True, True, False])
    totals = grouped_domain_bce_totals(logits, labels, domains, valid)
    per_sample = torch.nn.functional.binary_cross_entropy_with_logits(
        logits[:4], labels[:4].float(), reduction="none"
    )
    torch.testing.assert_close(totals[:, 0], per_sample.double())
    assert totals[:, 1].tolist() == [1, 1, 1, 1]
    assert totals[:, 2].tolist() == [1, 1, 1, 2]
    assert totals[:, 3].tolist() == [1, 0, 1, 0]


@pytest.mark.parametrize("module_type", [Stage2LoraAdaptationModule, Stage2QbytFullAdaptationModule])
def test_joint_validation_keeps_real_tts_lph_and_background_separate(module_type):
    module = module_type.__new__(module_type)
    torch.nn.Module.__init__(module)
    module._adapt_cfg = {"phase": "joint", "joint": {"background_eval_list": "val.list"}}
    module._score_calibration_slope = 1.0
    module._score_calibration_bias = 0.0
    module.target_score_diagnostics = BinaryScoreDiagnostics()
    module.score_diagnostics = BinaryScoreDiagnostics()
    module.joint_score_diagnostics = torch.nn.ModuleDict({
        name: BinaryScoreDiagnostics() for name in ("tts", "musan")
    })
    module.forward = lambda feat, *args: (feat[:, 0], None)
    module._log_train_window_metrics = lambda: None
    logged = {}
    module.log = lambda key, value, **kwargs: logged.__setitem__(key, value)
    for loader_idx, (scores, labels) in enumerate((
        ([3., -3.], [1, 0]), ([1., -1.], [1, 0]),
        ([-1., 2.], [1, 0]), ([-5., 5.], [0, 0]),
    )):
        module.validation_step({
            "feat": torch.tensor(scores).unsqueeze(1), "feat_lengths": torch.tensor([1, 1]),
            "anchor": torch.ones(2, 1, dtype=torch.long), "label": torch.tensor(labels),
            "sample_id": torch.tensor([0, 1]),
        }, 0, loader_idx)
    module.on_validation_epoch_end()
    assert logged["val/real_auc"] == logged["val/target_auc"] == 1.0
    assert logged["val/tts_auc"] == 0.0
    assert logged["val/lph_auc"] == 1.0
    assert logged["val/musan_deploy_fpr"] == 0.5
    assert torch.isnan(logged["val/musan_auc"])


def test_background_validation_is_fixed_and_preserves_cpu_rng(monkeypatch):
    from dma_kws.stage2.joint_validation import BackgroundValidationDataset

    class Background:
        def __init__(self, **kwargs):
            pass

        def extract(self, *, rng):
            return torch.rand(3, 80) + rng.random()

    monkeypatch.setattr("dma_kws.stage2.joint_validation.TrainingBackgroundSampler", Background)
    dataset = BackgroundValidationDataset(audio_list_path="val.list", anchor_seq=[1, 2])
    before = torch.random.get_rng_state().clone()
    a = dataset[0]
    assert torch.equal(before, torch.random.get_rng_state())
    torch.testing.assert_close(a["feat"], dataset[0]["feat"], rtol=0, atol=0)
    assert not torch.equal(a["feat"], dataset[1]["feat"])
    assert a["label"] == 0


@pytest.mark.parametrize("module_type", [Stage2LoraAdaptationModule, Stage2QbytFullAdaptationModule])
def test_joint_training_logs_four_sources_and_acknowledges_once(module_type):
    from types import SimpleNamespace
    from dma_kws.stage2.collate import train_collate_fn
    sample = {"feat": torch.zeros(3, 80), "anchor_seq": torch.tensor([1]),
              "query_seq": torch.tensor([]).long(), "seq_label": torch.tensor([0]),
              "label": torch.tensor(0), "source": 0, "domain_source": 3}
    batch = train_collate_fn([sample])
    module = module_type.__new__(module_type)
    torch.nn.Module.__init__(module)
    logits = torch.tensor([0.])
    module._forward_train_losses = lambda b: (logits.sum(), {"utt_sample_mask": torch.tensor([True])}, logits)
    module._log_train_losses = lambda *args: None
    module._trainer = SimpleNamespace(train_dataloader=_loader())
    logged = {}
    module.log = lambda key, value, **kwargs: logged.__setitem__(key, value)
    module.training_step(batch, 0)
    assert logged["train/microbatch/source_musan_fraction"] == 1.0
    assert logged["train/microbatch/source_real_fraction"] == 0.0
    assert logged["train/microbatch/source_keyword_fraction"] == 0.0
    assert module._trainer.train_dataloader.consumed_batches == 1


def test_joint_console_displays_tts_and_continuous_false_accept_rate():
    from dma_kws.stage2.adapt_console import eval_comparison_table
    _, rows = eval_comparison_table({
        "tts_base": {"auc": 0.7}, "tts_adapted": {"auc": 0.8},
        "musan_base": {"metrics": {"fa_per_hour": 2.0}},
        "musan_adapted": {"metrics": {"fa_per_hour": 1.0}},
    })
    assert {row[0] for row in rows} == {"tts/auc", "musan/fa_per_hour"}
    assert next(row[-1] for row in rows if row[0] == "musan/fa_per_hour") == "-1.0000"
