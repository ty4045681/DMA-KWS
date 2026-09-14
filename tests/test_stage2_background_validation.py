"""T14 per-source validation crops, clip_fpr, DDP padding, loader roles."""

from __future__ import annotations

import random

import pytest
import torch

from dma_kws.stage2.collate import test_collate_fn
from dma_kws.stage2.joint_validation import (
    background_clip_stats,
    build_background_validation_datasets,
)
from tests.test_stage2_background_identity import _write_source
from tests.test_stage2_background_sources import (
    _IdentityFbank,
    _online_config,
    _patch_online_fbank,
    _source_config,
)


def _two_source_config(tmp_path):
    sources = []
    for source_id, weight in (("dns", 0.4), ("musan", 0.6)):
        manifest = _write_source(tmp_path, source_id)
        sources.append(_source_config(source_id, manifest, weight=weight))
    return _online_config(
        sources,
        duration_seconds_min=0.25,
        duration_seconds_max=0.8,
        validation={"enabled": True, "samples_per_source": 4, "seed": 2025},
    )


def _patch_val_fbank(monkeypatch) -> None:
    _patch_online_fbank(monkeypatch)
    monkeypatch.setattr("dma_kws.stage2.joint_validation.FbankExtractor", _IdentityFbank)


def test_per_source_val_crops_are_fixed_and_do_not_consume_train_rng(tmp_path, monkeypatch):
    _patch_val_fbank(monkeypatch)
    config = _two_source_config(tmp_path)
    train_rng = random.Random(11)
    before_py = random.getstate()
    before_torch = torch.random.get_rng_state().clone()
    before_train = train_rng.getstate()
    datasets = build_background_validation_datasets(
        config,
        anchor_seq=[7, 8, 9],
        fbank_kwargs={"dither": 0.1},
    )
    assert [role for role, _dataset in datasets] == ["background:dns", "background:musan"]
    role, dataset = datasets[0]
    assert role == "background:dns"
    assert len(dataset) == 4
    first = dataset[0]
    second = dataset[1]
    assert first["label"].item() == 0
    assert int(first["background_source_id"]) == 0
    assert first["sample_id"].item() == 0
    torch.testing.assert_close(first["feat"], dataset[0]["feat"], rtol=0, atol=0)
    assert not torch.equal(first["feat"], second["feat"])
    assert random.getstate() == before_py
    assert torch.equal(torch.random.get_rng_state(), before_torch)
    assert train_rng.getstate() == before_train
    train_rng.random()
    torch.testing.assert_close(dataset[0]["feat"], first["feat"], rtol=0, atol=0)


def test_val_crop_changes_with_source_recording_ordinal_and_seed(tmp_path, monkeypatch):
    _patch_val_fbank(monkeypatch)
    config = _two_source_config(tmp_path)
    a = build_background_validation_datasets(
        config, anchor_seq=[1], fbank_kwargs={"dither": 0.0}
    )
    other_seed = dict(config)
    other_seed["validation"] = {"enabled": True, "samples_per_source": 4, "seed": 7}
    b = build_background_validation_datasets(
        other_seed, anchor_seq=[1], fbank_kwargs={"dither": 0.0}
    )
    dns_a = a[0][1][0]["feat"]
    dns_b = b[0][1][0]["feat"]
    musan_a = a[1][1][0]["feat"]
    assert not torch.equal(dns_a, dns_b)
    assert not torch.equal(dns_a, musan_a)
    assert a[0][1][0]["recording_id"]
    assert a[0][1][0]["recording_id"] != a[1][1][0]["recording_id"]


def test_clip_fpr_is_threshold_hit_rate_not_fa_per_hour():
    scores = torch.tensor([0.1, 0.6, 0.9, 0.4])
    stats = background_clip_stats(scores, threshold=0.5, sample_ids=torch.arange(4))
    assert stats["count"] == 4
    assert stats["clip_fpr"] == pytest.approx(0.5)
    assert stats["score_mean"] == pytest.approx(float(scores.mean()))
    assert stats["score_max"] == pytest.approx(0.9)
    assert "fa_per_hour" not in stats
    assert "fa/h" not in stats


def test_ddp_padding_does_not_double_count_val_samples():
    scores = torch.tensor([0.2, 0.8, 0.2])
    sample_ids = torch.tensor([0, 1, 0])
    stats = background_clip_stats(scores, threshold=0.5, sample_ids=sample_ids)
    assert stats["count"] == 2
    assert stats["clip_fpr"] == pytest.approx(0.5)
    assert stats["score_mean"] == pytest.approx(0.5)


def test_background_val_collate_keeps_sample_id_for_dedup(tmp_path, monkeypatch):
    _patch_val_fbank(monkeypatch)
    config = _two_source_config(tmp_path)
    _role, dataset = build_background_validation_datasets(
        config, anchor_seq=[3, 4], fbank_kwargs={"dither": 0.0}
    )[0]
    batch = test_collate_fn([dataset[0], dataset[1]])
    assert batch["sample_id"].tolist() == [0, 1]
    assert batch["label"].tolist() == [0, 0]
    assert "background_source_id" in batch


def test_update_background_val_does_not_copy_scores_to_cpu(monkeypatch):
    from dma_kws.stage2.module import Stage2LightningModule

    cpu_calls: list[tuple[int, ...]] = []
    real_cpu = torch.Tensor.cpu

    def tracking_cpu(self):
        cpu_calls.append(tuple(self.shape))
        return real_cpu(self)

    monkeypatch.setattr(torch.Tensor, "cpu", tracking_cpu)
    captured: dict[str, torch.device] = {}

    def fake_gather(rows):
        captured["device"] = rows.device
        return rows.clone()

    monkeypatch.setattr("dma_kws.stage2.module.gather_variable_rows", fake_gather)

    module = Stage2LightningModule.__new__(Stage2LightningModule)
    torch.nn.Module.__init__(module)
    module._score_calibration_slope = 1.0
    module._score_calibration_bias = 0.0
    module._background_val_scores = {}
    module._background_val_ids = {}
    module.register_parameter("_probe", torch.nn.Parameter(torch.zeros(1)))
    module._device = torch.device("cpu")
    logits = torch.tensor([0.0, 1.5])
    module._update_background_val(
        "dns",
        logits,
        {"sample_id": torch.tensor([0, 1])},
    )
    assert cpu_calls == []
    stored = module._background_val_scores["dns"][0]
    assert stored.device == logits.device
    module._gather_background_val_scores("dns")
    assert captured["device"] == module.device


def test_legacy_single_loader_validation_step_still_uses_integer_dispatch():
    from dma_kws.stage2.adapt import Stage2LoraAdaptationModule
    from dma_kws.training.score_diagnostics import BinaryScoreDiagnostics

    module = Stage2LoraAdaptationModule.__new__(Stage2LoraAdaptationModule)
    torch.nn.Module.__init__(module)
    module._adapt_cfg = {"phase": "joint", "joint": {"background_eval_list": "val.list"}}
    module._score_calibration_slope = 1.0
    module._score_calibration_bias = 0.0
    module._val_loader_roles = None
    module.target_score_diagnostics = BinaryScoreDiagnostics()
    module.score_diagnostics = BinaryScoreDiagnostics()
    module.joint_score_diagnostics = torch.nn.ModuleDict(
        {name: BinaryScoreDiagnostics() for name in ("tts", "musan")}
    )
    module.forward = lambda feat, *args: (feat[:, 0], None)
    module._log_train_window_metrics = lambda: None
    logged = {}
    module.log = lambda key, value, **kwargs: logged.__setitem__(key, value)
    for loader_idx, (scores, labels) in enumerate(
        (
            ([3.0, -3.0], [1, 0]),
            ([1.0, -1.0], [1, 0]),
            ([-1.0, 2.0], [1, 0]),
            ([-5.0, 5.0], [0, 0]),
        )
    ):
        module.validation_step(
            {
                "feat": torch.tensor(scores).unsqueeze(1),
                "feat_lengths": torch.tensor([1, 1]),
                "anchor": torch.ones(2, 1, dtype=torch.long),
                "label": torch.tensor(labels),
                "sample_id": torch.tensor([0, 1]),
            },
            0,
            loader_idx,
        )
    module.on_validation_epoch_end()
    assert logged["val/musan_deploy_fpr"] == 0.5
    assert "val/background/overall/clip_fpr" not in logged


def test_multisource_overall_metrics_are_not_named_musan():
    from dma_kws.stage2.background_identity import musan_metric_alias_allowed
    from dma_kws.stage2.joint_validation import format_background_val_metrics

    assert musan_metric_alias_allowed([])
    assert musan_metric_alias_allowed(["musan"])
    assert not musan_metric_alias_allowed(["dns", "musan"])
    assert not musan_metric_alias_allowed(["dns"])
    metrics = format_background_val_metrics(
        {
            "dns": {
                "count": 2,
                "clip_fpr": 0.5,
                "score_mean": 0.2,
                "score_p95": 0.8,
                "score_p99": 0.85,
                "score_max": 0.9,
            },
            "musan": {
                "count": 2,
                "clip_fpr": 0.0,
                "score_mean": 0.1,
                "score_p95": 0.2,
                "score_p99": 0.2,
                "score_max": 0.2,
            },
        }
    )
    assert metrics["val/background/dns/clip_fpr"] == 0.5
    assert metrics["val/background/musan/count"] == 2
    assert metrics["val/background/overall/count"] == 4
    assert metrics["val/background/overall/clip_fpr"] == pytest.approx(0.25)
    assert "val/musan_deploy_fpr" not in metrics
    assert not any(name.startswith("val/musan_") for name in metrics)

    single = format_background_val_metrics(
        {
            "musan": {
                "count": 4,
                "clip_fpr": 0.25,
                "score_mean": 0.1,
                "score_p95": 0.2,
                "score_p99": 0.3,
                "score_max": 0.4,
            }
        }
    )
    assert single["val/musan_deploy_fpr"] == pytest.approx(0.25)
    assert single["val/background/musan/clip_fpr"] == pytest.approx(0.25)
