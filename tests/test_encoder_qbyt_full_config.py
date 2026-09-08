"""Public configuration, artifact isolation and orchestration for encoder tuning."""

import copy
import sys
from types import SimpleNamespace

import pytest

from dma_kws.config import compose_config, config_to_dict
from dma_kws.stage2.adapt import Stage2AdaptArgs
from dma_kws.stage2.adapt_config import (
    adapt_method_label,
    is_full_adapt_method,
    resolve_adapt_method,
)
from dma_kws.stage2.adapt_paths import adapt_data_root, adapt_exp_root
from dma_kws.stage2.sweep_adapt import (
    bind_study_adapt_method,
    run_adaptation_training_trial,
    save_best_params,
    suggest_adapt_params,
)
from dma_kws.training.adapt_params import (
    load_adapt_params_file,
    merge_adapt_params,
    resolve_adapt_lr,
    resolve_encoder_adapt_lr,
)
from dma_kws.training.callbacks import build_run_summary_rows
from dma_kws.training.metrics_history import collect_hparams


METHOD = "encoder_qbyt_full"


def test_encoder_overlay_preserves_joint_data_and_pooling_architecture():
    base = config_to_dict(compose_config(overrides=[
        "+experiment=[icefall_zipformer_stage2_eps_softmin_v41,adapt_joint]",
    ]))
    actual = config_to_dict(compose_config(overrides=[
        "+experiment=[icefall_zipformer_stage2_eps_softmin_v41,adapt_joint,adapt_encoder_qbyt_full]",
    ]))
    assert actual["stage2"] == base["stage2"]
    assert actual["stage1"] == base["stage1"]
    assert actual["adapt"]["joint"] == base["adapt"]["joint"]
    assert actual["adapt"]["train_phases"] == ["joint"]
    assert resolve_adapt_method(actual["adapt"]) == METHOD
    assert is_full_adapt_method(METHOD)
    assert adapt_method_label(METHOD) == "Encoder + QbyT"
    assert resolve_adapt_lr(actual["adapt"])[0] == 3e-5
    assert resolve_encoder_adapt_lr(actual["adapt"]) == 3e-6
    assert actual["adapt"]["encoder_schedule"] == {
        "start_batch_count": 100000.0, "reference_duration": 600.0,
    }
    assert actual["adapt"]["params_file"] == ""
    assert actual["adapt"]["sweep"]["storage"] == ""


@pytest.mark.parametrize("joint", [False, True])
def test_all_methods_isolate_outputs_and_reuse_fbank_corpus(tmp_path, joint):
    config = {
        "paths": {"exp_root": str(tmp_path / "exp"), "processed_root": str(tmp_path / "data")},
        "adapt": {"train_phases": ["joint"] if joint else ["tts", "real"]},
    }
    roots, corpora = [], []
    for method in ("lora", "qbyt_full", METHOD):
        config["adapt"]["method"] = method
        roots.append(adapt_exp_root(config, "hey eva"))
        corpora.append(adapt_data_root(config, "hey eva"))
    assert len(set(roots)) == 3
    assert len(set(corpora)) == 1
    namespace = "stage2_adapt_joint" if joint else "stage2_adapt"
    assert roots[-1] == tmp_path / "exp" / f"{namespace}_{METHOD}" / "hey_eva"
    config["adapt"]["exp_root"] = str(tmp_path / "custom")
    assert adapt_exp_root(config, "hey eva") == tmp_path / "custom"


@pytest.mark.parametrize("invalid", [None, "", "bad", 0, -1e-6, float("nan"), float("inf"), True])
def test_encoder_lr_rejects_invalid_values(invalid):
    with pytest.raises(ValueError, match="encoder_learning_rate.*finite and positive"):
        resolve_encoder_adapt_lr({"encoder_learning_rate": invalid})


def test_encoder_lr_is_independent_from_qbyt_alias_and_logged_consistently():
    config = config_to_dict(compose_config(overrides=[
        "+experiment=adapt_encoder_qbyt_full",
        "adapt.lr=8e-5", "adapt.encoder_learning_rate=2e-6",
    ]))
    assert resolve_adapt_lr(config["adapt"]) == (8e-5, "lr")
    assert resolve_encoder_adapt_lr(config["adapt"]) == 2e-6
    hparams = collect_hparams(config, section="adapt")
    assert hparams["learning_rate"] == 8e-5
    assert hparams["encoder_learning_rate"] == 2e-6
    assert hparams["adapt_method"] == METHOD
    assert hparams["encoder_schedule_start_batch_count"] == 100000.0
    assert not {"rank", "alpha"} & hparams.keys()
    rows = dict(build_run_summary_rows(
        config=config, section="adapt", devices=1, accelerator="cpu",
        train_samples=10, val_samples=4,
    ))
    assert float(rows["learning_rate (lr)"]) == 8e-5
    assert float(rows["encoder_learning_rate"]) == 2e-6


class _Trial:
    def __init__(self):
        self.calls = {}

    def suggest_float(self, name, low, high, **kwargs):
        self.calls[name] = (low, high, kwargs)
        # Distinct interior values catch accidentally coupling the two groups.
        return 4e-6 if name == "encoder_learning_rate" else 7e-5

    def suggest_categorical(self, name, choices):
        self.calls[name] = choices
        return choices[1]


def test_encoder_sweep_roundtrips_two_independent_lrs_and_method_identity(tmp_path):
    trial = _Trial()
    params = suggest_adapt_params(trial, method=METHOD)
    assert set(trial.calls) == {"learning_rate", "encoder_learning_rate", "max_steps"}
    assert trial.calls["encoder_learning_rate"] == (1e-6, 1e-5, {"log": True})
    path = tmp_path / "best.yaml"
    save_best_params(path, params, score=0.91, method=METHOD)
    saved = load_adapt_params_file(path)
    target = {"method": METHOD, "lr": 1e-3}
    merge_adapt_params(target, saved)
    assert resolve_adapt_lr(target)[0] == 7e-5
    assert resolve_encoder_adapt_lr(target) == 4e-6
    assert target["method"] == METHOD
    assert "score" not in target
    for other in ("lora", "qbyt_full"):
        with pytest.raises(ValueError, match="same training method"):
            merge_adapt_params({"method": other}, saved)


@pytest.mark.parametrize("params", [
    {"rank": 4}, {"alpha": 8}, {"alpha_ratio": 2},
    {"lora_targets": ["in_proj_weight"]}, {"method": "qbyt_full"},
    {"encoder_learning_rate": -1},
])
def test_incompatible_params_fail_before_mutation(params):
    adapt = {"method": METHOD, "learning_rate": 3e-5, "lr": 1e-4}
    before = copy.deepcopy(adapt)
    with pytest.raises(ValueError):
        merge_adapt_params(adapt, {"learning_rate": 8e-5, **params})
    assert adapt == before


@pytest.mark.parametrize("other", ["lora", "qbyt_full"])
def test_frozen_encoder_modes_reject_encoder_lr_from_untagged_params(other):
    with pytest.raises(ValueError, match="requires adapt.method=encoder_qbyt_full"):
        merge_adapt_params({"method": other}, {"encoder_learning_rate": 3e-6})


@pytest.mark.parametrize("previous", [None, "lora", "qbyt_full"])
def test_encoder_study_rejects_incompatible_existing_trials(previous):
    attrs = {} if previous is None else {"adapt_method": previous}
    study = SimpleNamespace(user_attrs=attrs, trials=[object()], set_user_attr=attrs.__setitem__)
    before = attrs.copy()
    with pytest.raises(ValueError, match="different adapt.sweep.study_name"):
        bind_study_adapt_method(study, METHOD)
    assert attrs == before


def test_encoder_sequential_sweep_hands_off_full_weights_with_both_lrs(monkeypatch, tmp_path):
    seen = []

    def run(config, args):
        phase = config["adapt"]["phase"]
        seen.append((phase, args.init_checkpoint, args.resume_checkpoint,
                     resolve_adapt_lr(config["adapt"])[0],
                     resolve_encoder_adapt_lr(config["adapt"])))
        return {"merged": tmp_path / phase / "stage2_adapted.pt"}

    monkeypatch.setattr("dma_kws.stage2.adapt.run_stage2_adaptation", run)
    run_adaptation_training_trial(
        {"adapt": {"keyword": "hey eva", "method": METHOD,
                   "train_phases": ["tts", "real"], "exp_root": str(tmp_path / "exp")}},
        params={"learning_rate": 7e-5, "encoder_learning_rate": 4e-6},
        base_args=Stage2AdaptArgs(init_checkpoint=str(tmp_path / "base.pt")),
    )
    assert seen == [
        ("tts", str(tmp_path / "base.pt"), "", 7e-5, 4e-6),
        ("real", str(tmp_path / "tts" / "stage2_adapted.pt"), "", 7e-5, 4e-6),
    ]


def _capture_commands(monkeypatch):
    from scripts import run_keyword_adaptation

    commands = []
    monkeypatch.setattr(run_keyword_adaptation.subprocess, "run",
                        lambda cmd, **kwargs: commands.append(cmd))
    reporter = SimpleNamespace(section=lambda *_: None, print_plan=lambda *_, **__: None,
                               info=lambda *_: None, use_rich=False)
    monkeypatch.setattr(run_keyword_adaptation.adapt_console, "adapt_reporter", lambda *_: reporter)
    return run_keyword_adaptation, commands


def test_encoder_joint_orchestrator_resumes_without_external_base(monkeypatch, tmp_path):
    script, commands = _capture_commands(monkeypatch)
    overrides = [
        "+experiment=[adapt_joint,adapt_encoder_qbyt_full]", "adapt.stage=train",
        "run.resume_from=/checkpoints/resume.ckpt", f"paths.exp_root={tmp_path}",
    ]
    monkeypatch.setattr(sys, "argv", ["run_keyword_adaptation.py", *overrides])
    script.main.__wrapped__(compose_config(overrides=overrides))
    assert len(commands) == 1
    assert "run.resume_from=/checkpoints/resume.ckpt" in commands[0]
    assert f"adapt.method={METHOD}" in commands[0]
    assert "adapt.phase=joint" in commands[0]
    assert not any(arg.startswith("prep.stage2_ckpt=") for arg in commands[0])


def test_encoder_orchestrator_sequential_handoff_overrides_original_base(monkeypatch, tmp_path):
    script, commands = _capture_commands(monkeypatch)
    overrides = [
        "+experiment=adapt_encoder_qbyt_full", "adapt.stage=train",
        "run.init_checkpoint=/checkpoints/base.pt", f"paths.exp_root={tmp_path}",
        "adapt.encoder_learning_rate=2e-6",
    ]
    monkeypatch.setattr(sys, "argv", ["run_keyword_adaptation.py", *overrides])
    script.main.__wrapped__(compose_config(overrides=overrides))
    assert len(commands) == 2
    second = dict(arg.split("=", 1) for arg in commands[1] if "=" in arg)
    assert second["run.init_checkpoint"] == str(
        tmp_path / f"stage2_adapt_{METHOD}" / "hey_eva" / "tts" / "stage2_adapted.pt"
    )
    assert second["run.resume_checkpoint"] == ""
    assert second["adapt.encoder_learning_rate"] == "2e-6"


def test_encoder_resume_rejects_multi_phase_orchestration_before_launch(monkeypatch):
    script, commands = _capture_commands(monkeypatch)
    overrides = [
        "+experiment=adapt_encoder_qbyt_full", "adapt.stage=train",
        "run.resume_from=/checkpoints/resume.ckpt",
    ]
    with pytest.raises(SystemExit, match="restores one phase"):
        script.main.__wrapped__(compose_config(overrides=overrides))
    assert commands == []
