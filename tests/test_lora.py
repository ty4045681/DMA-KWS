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
    normalize_lora_targets,
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


@pytest.mark.parametrize("alpha", [0.0, -1.0, float("inf"), float("nan")])
def test_lora_rejects_non_positive_or_non_finite_alpha(alpha):
    with pytest.raises(ValueError, match="alpha"):
        LoRAParametrization(128, 384, rank=8, alpha=alpha)


def test_lora_targets_reject_empty_and_misspelled_entries():
    with pytest.raises(ValueError, match="At least one"):
        normalize_lora_targets([])
    with pytest.raises(ValueError, match="Unsupported"):
        normalize_lora_targets(["in_proj_weight", "out_project.weight"])
    assert normalize_lora_targets(["out_proj"]) == ("out_proj.weight",)


def test_lora_runtime_rejects_base_mutating_ema_and_true_half_precision():
    from dma_kws.stage2.adapt import _validate_lora_runtime_config

    with pytest.raises(ValueError, match="ema"):
        _validate_lora_runtime_config({"ema": {"enabled": True}})
    with pytest.raises(ValueError, match="precision"):
        _validate_lora_runtime_config({"precision": "bf16-true"})

    _validate_lora_runtime_config({"precision": "bf16-mixed"})
    _validate_lora_runtime_config({"precision": "32-true"})


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


def test_lora_state_loader_never_writes_base_parameters():
    qbyt = _build_qbyt()
    inject_qbyt_lora(qbyt, rank=4, alpha=8.0)
    saved = lora_state_dict(qbyt)
    base_name = next(name for name in qbyt.state_dict() if "lora_" not in name)
    base_before = qbyt.state_dict()[base_name].clone()
    mixed = {**saved, base_name: torch.full_like(base_before, 123.0)}

    with pytest.raises(KeyError, match="non-adapter"):
        load_lora_state_dict(qbyt, mixed, strict=True)

    load_lora_state_dict(qbyt, mixed, strict=False)
    torch.testing.assert_close(qbyt.state_dict()[base_name], base_before)


def _adapter_payload(**overrides):
    payload = {
        "keyword": "hey eva",
        "rank": 4,
        "alpha": 8.0,
        "lora_targets": ["in_proj_weight", "out_proj.weight"],
        "lora_state_dict": {"layer.lora_A": torch.ones(4, 8)},
    }
    payload.update(overrides)
    return payload


def test_adapter_resume_metadata_must_match_exactly():
    from dma_kws.stage2.adapt import _validate_adapter_checkpoint

    expected = {
        "source": "adapter.pt",
        "keyword": "hey eva",
        "rank": 4,
        "alpha": 8.0,
        "targets": ("in_proj_weight", "out_proj.weight"),
    }
    state = _validate_adapter_checkpoint(_adapter_payload(), **expected)
    assert state
    assert _validate_adapter_checkpoint(
        _adapter_payload(keyword="hey-eva"),
        **expected,
    )

    with pytest.raises(SystemExit, match="keyword"):
        _validate_adapter_checkpoint(_adapter_payload(keyword="hey siri"), **expected)
    with pytest.raises(SystemExit, match="keyword"):
        _validate_adapter_checkpoint(
            _adapter_payload(keyword="hey'eva"),
            **{**expected, "keyword": "hey eva"},
        )
    with pytest.raises(SystemExit, match="rank"):
        _validate_adapter_checkpoint(_adapter_payload(rank=8), **expected)
    with pytest.raises(SystemExit, match="alpha"):
        _validate_adapter_checkpoint(_adapter_payload(alpha=16.0), **expected)
    with pytest.raises(SystemExit, match="lora_targets"):
        _validate_adapter_checkpoint(
            _adapter_payload(lora_targets=["in_proj_weight"]),
            **expected,
        )


def test_adapter_resume_requires_metadata_and_nonempty_state():
    from dma_kws.stage2.adapt import _validate_adapter_checkpoint

    expected = {
        "source": "adapter.pt",
        "keyword": "hey eva",
        "rank": 4,
        "alpha": 8.0,
        "targets": ("in_proj_weight", "out_proj.weight"),
    }
    with pytest.raises(SystemExit, match="required metadata"):
        _validate_adapter_checkpoint({"lora_state_dict": {"x": torch.ones(1)}}, **expected)
    with pytest.raises(SystemExit, match="missing or empty"):
        _validate_adapter_checkpoint(_adapter_payload(lora_state_dict={}), **expected)


def test_adapter_resume_rejects_conflicting_nested_metadata_and_wrong_kind():
    from dma_kws.stage2.adapt import _validate_adapter_checkpoint

    expected = {
        "source": "adapter.pt",
        "keyword": "hey eva",
        "rank": 4,
        "alpha": 8.0,
        "targets": ("in_proj_weight", "out_proj.weight"),
    }
    conflicting = _adapter_payload(
        config={"adapt": {"keyword": "hey eva", "rank": 4, "alpha": 16.0}}
    )
    with pytest.raises(SystemExit, match="config.adapt.alpha"):
        _validate_adapter_checkpoint(conflicting, **expected)

    with pytest.raises(SystemExit, match="checkpoint_kind"):
        _validate_adapter_checkpoint(
            _adapter_payload(checkpoint_kind="stage2"),
            **expected,
        )


def test_adapter_resume_checks_base_fingerprint_and_requires_legacy_opt_in():
    from dma_kws.stage2.adapt import _validate_adapter_checkpoint

    expected = {
        "source": "adapter.pt",
        "keyword": "hey eva",
        "rank": 4,
        "alpha": 8.0,
        "targets": ("in_proj_weight", "out_proj.weight"),
        "base_model_sha256": "a" * 64,
    }
    with pytest.raises(SystemExit, match="base_model_sha256"):
        _validate_adapter_checkpoint(
            _adapter_payload(base_model_sha256="b" * 64),
            **expected,
        )
    with pytest.raises(SystemExit, match="allow_legacy_adapter"):
        _validate_adapter_checkpoint(_adapter_payload(), **expected)
    with pytest.warns(UserWarning, match="explicitly enabled"):
        _validate_adapter_checkpoint(
            _adapter_payload(),
            **expected,
            allow_legacy_adapter=True,
        )


def test_adapter_resume_is_explicit_except_for_tts_to_real_handoff(tmp_path):
    from dma_kws.stage2.adapt import _resolve_adapter_resume

    phase_dir = tmp_path / "trial" / "tts"
    phase_dir.mkdir(parents=True)
    stale_tts = phase_dir / "adapter_hey_eva.pt"
    stale_tts.write_bytes(b"adapter")
    paths = {"phase_dir": phase_dir, "slug_str": "hey_eva"}

    # Re-running TTS is fresh by default; it must not silently stack another run
    # on an old final adapter with a reset optimizer.
    assert _resolve_adapter_resume(paths, "tts") is None
    assert _resolve_adapter_resume(paths, "tts", str(stale_tts)) == str(stale_tts)

    real_paths = {"phase_dir": tmp_path / "trial" / "real", "slug_str": "hey_eva"}
    assert _resolve_adapter_resume(real_paths, "real") == str(stale_tts)


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
