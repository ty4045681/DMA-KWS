"""T12/T15 train background metrics, empty-source NaN, CPU smokes."""

from __future__ import annotations

import math
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

from dma_kws.stage2.collate import train_collate_fn
from dma_kws.stage2.module import (
    Stage2LightningModule,
    grouped_background_source_totals,
)
from tests.test_stage2_background_identity import _write_source
from tests.test_stage2_background_sources import (
    _FakeTokenizer,
    _mock_dataframe,
    _online_config,
    _patch_online_fbank,
    _source_config,
)


@pytest.fixture
def mock_npy_loader(monkeypatch):
    import numpy as np
    from pathlib import Path

    clips = {
        "clips-2-a.npy": np.array(
            [{"audio_path": "LP-460/hello/a.wav"}, {"audio_path": "LP-460/hello/b.wav"}],
            dtype=object,
        ),
        "clips-2-b.npy": np.array(
            [{"audio_path": "LP-460/world/c.wav"}, {"audio_path": "LP-460/world/d.wav"}],
            dtype=object,
        ),
    }
    distances = {
        "dist-0-a.npy": np.array([], dtype=object),
        "dist-2-b.npy": np.array([{"ngram": "hello"}, {"ngram": "hello"}], dtype=object),
    }
    fbank = np.ones((5, 80), dtype=np.float32)

    def fake_load(path, allow_pickle=False):
        name = Path(path).name
        if name in clips:
            return clips[name]
        if name in distances:
            return distances[name]
        if name.endswith(".npy") and "fbank" in str(path):
            return fbank
        raise FileNotFoundError(path)

    monkeypatch.setattr("dma_kws.stage2.dataset.np.load", fake_load)
    return fbank


def test_grouped_background_totals_skip_non_background_and_invalid():
    logits = torch.tensor([0.0, 1.0, 2.0, 3.0])
    labels = torch.tensor([0, 0, 0, 1])
    source_ids = torch.tensor([0, 1, -1, 0])
    valid = torch.tensor([True, True, True, False])
    totals = grouped_background_source_totals(
        logits, labels, source_ids, valid, num_sources=2
    )
    per_sample = torch.nn.functional.binary_cross_entropy_with_logits(
        logits[:2], labels[:2].float(), reduction="none"
    )
    torch.testing.assert_close(totals[0, 0], per_sample[0].double())
    torch.testing.assert_close(totals[1, 0], per_sample[1].double())
    assert totals[:2, 1].tolist() == [1.0, 1.0]
    assert totals[:2, 2].tolist() == [2.0, 1.0]
    assert totals[2, 0].item() == 4
    assert totals[2, 1].item() == 3


def test_train_background_metrics_nan_when_count_zero_and_sum_to_consumed_batch(monkeypatch):
    class _Encoder(nn.Module):
        def __init__(self, input_dim, output_dim):
            super().__init__()
            self.projection = nn.Linear(input_dim, output_dim)

        def forward(self, feat, lengths, **kwargs):
            mask = torch.arange(feat.shape[1], device=feat.device)[None] < lengths[:, None]
            return self.projection(feat), mask[:, None]

        def apply_stream_config(self, chunk_sizes, left_context_frames):
            pass

    monkeypatch.setattr(
        "dma_kws.stage2.module.build_encoder",
        lambda stage1, *, output_dim: _Encoder(stage1["input_dim"], output_dim),
    )
    config = {
        "stage1": {
            "input_dim": 8,
            "encoder_output_dim": 8,
            "causal": True,
        },
        "stage2": {
            "encoder_output_dim": 8,
            "qbyt_embed_dim": 8,
            "qbyt_layers": 1,
            "qbyt_readout_version": 4,
            "qbyt_readout": {"mode": "eps_softmin", "temperature": 1.0},
            "learning_rate": 1e-3,
            "warmup_steps": 1,
            "total_scheduler_steps": 10,
            "max_steps": 10,
            "freeze_encoder": True,
            "sequence_loss": {
                "target_mode": "ordered_contiguous_prefix",
                "progress_weight": 0.3,
                "normalization": "sample",
            },
            "background_negative": {
                "enabled": True,
                "probability": 0.25,
                "sources": [
                    {"id": "dns", "weight": 0.5, "manifest": "x"},
                    {"id": "musan", "weight": 0.5, "manifest": "y"},
                ],
            },
        },
        "demo": {"qbyt_threshold": 0.5},
        "prep": {},
    }
    model = Stage2LightningModule(config, vocab_size=16, freeze_encoder=True)
    model.configure_background_sources(["dns", "musan"])
    model.optimizers = MagicMock(return_value=MagicMock(param_groups=[{"lr": 1e-3}]))
    logged = {}
    model.log = lambda key, value, **kwargs: logged.__setitem__(
        key,
        float(value.detach()) if torch.is_tensor(value) else value,
    )
    samples = []
    for source_id, feat in ((0, 0.0), (0, 1.0), (-1, 2.0)):
        samples.append(
            {
                "anchor_seq": torch.tensor([1, 2], dtype=torch.long),
                "query_seq": torch.tensor([] if source_id >= 0 else [3], dtype=torch.long),
                "feat": torch.ones(4, 8) * feat,
                "label": torch.tensor(0 if source_id >= 0 else 1, dtype=torch.long),
                "seq_label": torch.tensor([0, 0], dtype=torch.long),
                "background_source_id": source_id,
            }
        )
    batch = train_collate_fn(samples)
    loss = model.training_step(batch, 0)
    assert torch.isfinite(loss)
    assert logged["train/background/dns/count"] == 2
    assert logged["train/background/musan/count"] == 0
    assert math.isnan(logged["train/background/musan/utt_bce"])
    assert math.isnan(logged["train/background/musan/score_mean"])
    assert math.isnan(logged["train/background/musan/fraction_background"])
    assert logged["train/background/dns/fraction_all"] == pytest.approx(2 / 3)
    assert logged["train/background/dns/fraction_background"] == pytest.approx(1.0)
    assert not any(p.requires_grad for p in model.encoder.parameters())


def test_cpu_smoke_dataset_consumes_both_sources(tmp_path, monkeypatch, mock_npy_loader):
    from dma_kws.stage2.dataset import LibriPhraseTrainDataset

    _patch_online_fbank(monkeypatch)
    sources = []
    for source_id in ("dns", "musan"):
        manifest = _write_source(tmp_path, source_id)
        sources.append(_source_config(source_id, manifest, weight=1.0))
    dataset = LibriPhraseTrainDataset(
        wav_dir="/data/segments",
        tokenizer=_FakeTokenizer(),
        df=_mock_dataframe(),
        sample_lens=1,
        seed=3,
        background_negative=_online_config(sources, probability=1.0),
        fbank_kwargs={"dither": 0.0},
    )
    seen = set()
    for _ in range(40):
        item = dataset[0]
        seen.add(int(item["background_source_id"]))
        if item["background_source_id"] >= 0:
            assert item["label"].item() == 0
            assert item["query_seq"].numel() == 0
    assert seen >= {0, 1, -1} or seen >= {0, 1}


def test_cpu_smoke_lora_and_qbyt_full_keep_trainability(tmp_path, monkeypatch):
    pytest.importorskip("pytorch_lightning")
    from dma_kws.stage2.adapt import Stage2LoraAdaptationModule, Stage2QbytFullAdaptationModule
    from dma_kws.training.checkpoint_io import stamp_qbyt_readout_version

    class _Encoder(nn.Module):
        def __init__(self, input_dim, output_dim):
            super().__init__()
            self.projection = nn.Linear(input_dim, output_dim)

        def forward(self, feat, lengths, **kwargs):
            mask = torch.arange(feat.shape[1], device=feat.device)[None] < lengths[:, None]
            return self.projection(feat), mask[:, None]

        def apply_stream_config(self, chunk_sizes, left_context_frames):
            pass

    monkeypatch.setattr(
        "dma_kws.stage2.module.build_encoder",
        lambda stage1, *, output_dim: _Encoder(stage1["input_dim"], output_dim),
    )
    transformers = ModuleType("transformers")
    transformers.get_cosine_schedule_with_warmup = (
        lambda optimizer, **kwargs: torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda step: 1.0
        )
    )
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    config = {
        "stage1": {"input_dim": 8, "encoder_output_dim": 8, "causal": True},
        "stage2": {
            "encoder_output_dim": 8,
            "qbyt_embed_dim": 8,
            "qbyt_layers": 1,
            "qbyt_readout_version": 4,
            "qbyt_readout": {"mode": "eps_softmin", "temperature": 1.0},
            "learning_rate": 1e-3,
            "warmup_steps": 1,
            "total_scheduler_steps": 10,
            "max_steps": 10,
            "freeze_encoder": True,
            "phoneme_adapter": {"enabled": False},
            "sequence_loss": {
                "target_mode": "ordered_contiguous_prefix",
                "progress_weight": 0.3,
                "normalization": "sample",
            },
            "background_negative": {
                "enabled": True,
                "probability": 0.25,
                "sources": [
                    {"id": "dns", "weight": 0.5, "manifest": "x"},
                    {"id": "fsd50k", "weight": 0.5, "manifest": "y"},
                ],
            },
        },
        "adapt": {
            "method": "lora",
            "keyword": "hey",
            "phase": "tts",
            "learning_rate": 1e-3,
            "max_steps": 4,
            "warmup_steps": 0,
            "optimizer": "adamw",
            "rank": 2,
            "alpha": 4,
        },
        "demo": {"qbyt_threshold": 0.5},
        "prep": {},
    }
    base = Stage2LightningModule(config, vocab_size=16, freeze_encoder=True)
    ckpt = tmp_path / "base.pt"
    torch.save(
        stamp_qbyt_readout_version(
            {"model_state_dict": base.state_dict(), "config": base._checkpoint_config},
            alignment=base.qbyt_score,
        ),
        ckpt,
    )
    lora = Stage2LoraAdaptationModule(
        config, vocab_size=16, init_checkpoint=ckpt, lora_rank=2, lora_alpha=4
    )
    lora.configure_background_sources(["dns", "fsd50k"])
    assert not any(p.requires_grad for p in lora.encoder.parameters())
    assert any("lora_" in name for name, param in lora.named_parameters() if param.requires_grad)
    full_cfg = dict(config)
    full_cfg["adapt"] = dict(config["adapt"], method="qbyt_full")
    full = Stage2QbytFullAdaptationModule(full_cfg, vocab_size=16, init_checkpoint=ckpt)
    assert all(param.requires_grad for param in full.qbyt.parameters())
    assert not any(param.requires_grad for param in full.encoder.parameters())
    sample = {
        "feat": torch.zeros(3, 8),
        "anchor_seq": torch.tensor([1]),
        "query_seq": torch.tensor([]).long(),
        "seq_label": torch.tensor([0]),
        "label": torch.tensor(0),
        "source": 0,
        "domain_source": 3,
        "background_source_id": 0,
    }
    batch = train_collate_fn([sample])
    logged: dict[str, object] = {}
    lora._forward_train_losses = lambda _batch: (
        torch.tensor(0.1),
        {"utt_sample_mask": torch.tensor([True])},
        torch.tensor([0.0]),
    )
    lora._log_train_losses = lambda *_args: None
    lora._trainer = SimpleNamespace(
        train_dataloader=SimpleNamespace(mark_consumed=lambda: None, consumed_batches=0)
    )
    lora.log = lambda key, value, **kwargs: logged.__setitem__(key, value)
    loss = lora.training_step(batch, 0)
    assert torch.isfinite(loss)
    assert "train/microbatch/source_musan_fraction" not in logged
    assert logged.get("train/background/dns/count") == 1 or logged.get(
        "train/microbatch/source_background_fraction"
    ) == 1.0
