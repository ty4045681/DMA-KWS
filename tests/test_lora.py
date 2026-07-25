import copy

import pytest
import torch
import torch.nn as nn

pytest.importorskip("torch")

from dma_kws.training.lora import (
    LoRAParametrization,
    count_lora_params,
    inject_qbyt_lora,
    lora_state_dict,
    load_lora_state_dict,
    merge_lora,
)


def _build_qbyt() -> nn.Module:
    layer = nn.TransformerEncoderLayer(
        d_model=128,
        nhead=4,
        dim_feedforward=512,
        batch_first=True,
    )
    qbyt = nn.Module()
    qbyt.phone_matchor = nn.TransformerEncoder(layer, num_layers=2)
    return qbyt


def test_lora_initial_delta_is_zero():
    lora = LoRAParametrization(128, 384, rank=8, alpha=16.0)
    weight = torch.randn(384, 128)
    merged = lora(weight)
    assert torch.allclose(merged, weight)


def test_inject_qbyt_lora_only_lora_trainable():
    qbyt = _build_qbyt()
    injected = inject_qbyt_lora(qbyt, rank=4, alpha=8.0)
    assert injected
    counts = count_lora_params(qbyt)
    assert counts["trainable"] == counts["lora_trainable"]
    assert counts["trainable"] > 0
    assert counts["trainable"] < counts["total"]
    for name, param in qbyt.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            assert param.requires_grad
        else:
            assert not param.requires_grad


def test_merge_lora_removes_parametrizations():
    qbyt = _build_qbyt()
    inject_qbyt_lora(qbyt, rank=4, alpha=8.0)
    assert lora_state_dict(qbyt)
    merge_lora(qbyt)
    assert not lora_state_dict(qbyt)


def test_lora_state_dict_roundtrip():
    qbyt = _build_qbyt()
    inject_qbyt_lora(qbyt, rank=4, alpha=8.0)
    saved = lora_state_dict(qbyt)
    assert saved

    qbyt2 = _build_qbyt()
    inject_qbyt_lora(qbyt2, rank=4, alpha=8.0)
    with torch.no_grad():
        for param in qbyt2.parameters():
            if param.requires_grad:
                param.zero_()
    load_lora_state_dict(qbyt2, saved)
    assert all(torch.allclose(saved[key], lora_state_dict(qbyt2)[key]) for key in saved)


def test_sweep_objective_stub(monkeypatch):
    from dma_kws.stage2 import sweep_adapt
    from dma_kws.stage2.adapt import Stage2AdaptArgs

    def fake_run(config, args):
        phase = config["adapt"]["phase"]
        return {"merged": f"/tmp/{phase}/stage2_adapted.pt"}

    # run_adaptation_trial imports this lazily, so patch it at the source module.
    monkeypatch.setattr("dma_kws.stage2.adapt.run_stage2_adaptation", fake_run)
    config = {"adapt": {"keyword": "hey eva", "max_steps": 10}}
    metrics = sweep_adapt.run_adaptation_trial(
        config,
        params={"rank": 8, "alpha": 16, "learning_rate": 1e-4, "max_steps": 10, "_trial_number": 0},
        base_args=Stage2AdaptArgs(init_checkpoint="/tmp/base.pt"),
        single_phase=True,
        eval_lph_fn=lambda _ckpt: 0.85,
        eval_target_fn=lambda _cfg, _ckpt, _kw: 0.92,
    )
    score = sweep_adapt.compute_sweep_score(
        target_auc=metrics["target_auc"],
        lph_auc_adapted=metrics["lph_auc"],
        lph_auc_base=0.88,
    )
    assert metrics["target_auc"] == 0.92
    assert score == pytest.approx(0.89)
