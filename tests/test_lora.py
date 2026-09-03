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
    qbyt = nn.Module()
    qbyt.audio_projection = nn.Linear(144, 128)
    qbyt.audio_key = nn.Linear(128, 96)
    qbyt.text_query = nn.Linear(128, 96)
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


def test_pooling_lora_targets_are_phone_matchor_attention_weights():
    from qbyt.pooling import QbyT
    from dma_kws.training.lora import lora_targets_for_qbyt_family

    assert normalize_lora_targets(None, family="pooling") == (
        "in_proj_weight",
        "out_proj.weight",
    )
    assert lora_targets_for_qbyt_family(
        ["audio_key.weight", "text_query.weight"],
        family="pooling",
    ) == ("in_proj_weight", "out_proj.weight")
    assert lora_targets_for_qbyt_family(
        None, family="keyword_filler"
    ) == ("audio_key.weight", "text_query.weight")
    qbyt = QbyT(
        encoder_output_size=24,
        num_embeds=16,
        embed_dim=32,
        post_num_layers=1,
        readout_mode="gru_last",
    )
    injected = inject_qbyt_lora(qbyt, rank=2, alpha=4.0)
    assert injected
    assert all("phone_matchor" in name for name in injected)


def test_lora_targets_reject_empty_and_misspelled_entries():
    assert normalize_lora_targets(None) == (
        "audio_key.weight",
        "text_query.weight",
    )
    assert normalize_lora_targets(
        ["audio_projection.weight", "audio_key.weight", "audio_key.weight"]
    ) == ("audio_projection.weight", "audio_key.weight")
    with pytest.raises(ValueError, match="At least one"):
        normalize_lora_targets([])
    with pytest.raises(ValueError, match="Unsupported"):
        normalize_lora_targets(
            ["in_proj_weight", "out_proj.weight"], family="keyword_filler"
        )
    with pytest.raises(ValueError, match="Unsupported"):
        normalize_lora_targets(["out_proj"], family="keyword_filler")


def test_lora_runtime_rejects_base_mutating_ema_and_true_half_precision():
    from dma_kws.stage2.adapt import _validate_lora_runtime_config

    with pytest.raises(ValueError, match="noise_augmentation.*not supported"):
        _validate_lora_runtime_config({"noise_augmentation": {"enabled": True}})
    with pytest.raises(ValueError, match="ema"):
        _validate_lora_runtime_config({"ema": {"enabled": True}})
    with pytest.raises(ValueError, match="precision"):
        _validate_lora_runtime_config({"precision": "bf16-true"})

    _validate_lora_runtime_config({"precision": "bf16-mixed"})
    _validate_lora_runtime_config({"precision": "32-true"})


def test_inject_qbyt_lora_only_lora_trainable():
    qbyt = _build_qbyt()
    injected = inject_qbyt_lora(qbyt, rank=4, alpha=8.0)
    assert injected == ["audio_key.weight", "text_query.weight"]
    counts = count_lora_params(qbyt)
    assert counts["trainable"] == counts["lora_trainable"]
    assert counts["trainable"] > 0
    assert counts["trainable"] < counts["total"]
    for name, param in qbyt.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            assert param.requires_grad
        else:
            assert not param.requires_grad


def test_inject_qbyt_lora_can_target_audio_projection_only():
    qbyt = _build_qbyt()

    injected = inject_qbyt_lora(
        qbyt,
        rank=4,
        alpha=8.0,
        targets=["audio_projection.weight"],
    )

    assert injected == ["audio_projection.weight"]
    state = lora_state_dict(qbyt)
    assert state
    assert all(name.startswith("audio_projection.parametrizations.weight") for name in state)


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
        "lora_targets": ["audio_key.weight", "text_query.weight"],
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
        "targets": ("audio_key.weight", "text_query.weight"),
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
            _adapter_payload(lora_targets=["audio_key.weight"]),
            **expected,
        )


def test_adapter_resume_requires_metadata_and_nonempty_state():
    from dma_kws.stage2.adapt import _validate_adapter_checkpoint

    expected = {
        "source": "adapter.pt",
        "keyword": "hey eva",
        "rank": 4,
        "alpha": 8.0,
        "targets": ("audio_key.weight", "text_query.weight"),
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
        "targets": ("audio_key.weight", "text_query.weight"),
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


def test_adapter_resume_requires_exact_base_fingerprint():
    from dma_kws.stage2.adapt import _validate_adapter_checkpoint

    expected = {
        "source": "adapter.pt",
        "keyword": "hey eva",
        "rank": 4,
        "alpha": 8.0,
        "targets": ("audio_key.weight", "text_query.weight"),
        "base_model_sha256": "a" * 64,
    }
    with pytest.raises(SystemExit, match="base_model_sha256"):
        _validate_adapter_checkpoint(
            _adapter_payload(base_model_sha256="b" * 64),
            **expected,
        )
    with pytest.raises(SystemExit, match="base_model_sha256"):
        _validate_adapter_checkpoint(_adapter_payload(), **expected)


def test_full_lora_checkpoint_restore_validates_embedded_base(monkeypatch):
    from dma_kws.stage2.adapt import Stage2LoraAdaptationModule
    from dma_kws.stage2.module import Stage2LightningModule
    from dma_kws.stage2.readout import QbyTAlignmentSpec, QbyTScoreSpec
    from dma_kws.training.checkpoint_io import (
        STAGE2_BASE_FINGERPRINT_KEY,
        fingerprint_stage2_base,
        stamp_qbyt_readout_version,
    )

    monkeypatch.setattr(
        Stage2LightningModule,
        "on_load_checkpoint",
        lambda self, checkpoint: None,
    )
    module = Stage2LoraAdaptationModule.__new__(Stage2LoraAdaptationModule)
    nn.Module.__init__(module)
    module.qbyt_alignment = QbyTAlignmentSpec()
    module.qbyt_score = QbyTScoreSpec(version=7, value=module.qbyt_alignment)
    module._checkpoint_config = {
        "adapt": {"keyword": "hey eva", "phase": "tts"}
    }
    module._lora_rank = 4
    module._lora_alpha = 8.0
    module._lora_targets = ("audio_key.weight", "text_query.weight")

    state = {
        "encoder.weight": torch.arange(4, dtype=torch.float32),
        "qbyt.audio_key.parametrizations.weight.original": torch.arange(
            6, dtype=torch.float32
        ),
        "qbyt.audio_key.parametrizations.weight.0.lora_A": torch.ones(1),
    }
    fingerprint = fingerprint_stage2_base(state)
    checkpoint = stamp_qbyt_readout_version(
        {
            "checkpoint_kind": "stage2_lora",
            "keyword": "hey eva",
            "phase": "tts",
            "rank": 4,
            "alpha": 8.0,
            "lora_targets": list(module._lora_targets),
            STAGE2_BASE_FINGERPRINT_KEY: fingerprint,
            "state_dict": state,
        },
        alignment=module.qbyt_alignment,
    )

    module.on_load_checkpoint(checkpoint)

    assert module._base_model_sha256 == fingerprint


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


def test_sweep_single_phase_honors_real_only_configuration(monkeypatch):
    from dma_kws.stage2 import sweep_adapt
    from dma_kws.stage2.adapt import Stage2AdaptArgs

    phases: list[str] = []

    def fake_run(config, args):
        del args
        phase = config["adapt"]["phase"]
        phases.append(phase)
        return {
            "adapter": f"/tmp/{phase}/adapter.pt",
            "merged": f"/tmp/{phase}/stage2_adapted.pt",
        }

    monkeypatch.setattr("dma_kws.stage2.adapt.run_stage2_adaptation", fake_run)
    result = sweep_adapt.run_adaptation_training_trial(
        {
            "adapt": {
                "keyword": "hey eva",
                "max_steps": 10,
                "train_phases": ["real"],
            }
        },
        params={"rank": 8, "alpha": 16, "learning_rate": 1e-4, "max_steps": 10},
        base_args=Stage2AdaptArgs(init_checkpoint="/tmp/base.pt"),
        single_phase=True,
    )

    assert phases == ["real"]
    assert result["tts"] is None
    assert result["real"] is not None
