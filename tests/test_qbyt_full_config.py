"""Full-QbyT adaptation must not inherit LoRA artifacts or sweep parameters."""

import copy
import sys
from types import SimpleNamespace

import pytest

from dma_kws.config import compose_config, config_to_dict
from dma_kws.stage2.adapt import Stage2AdaptArgs
from dma_kws.stage2.adapt_config import resolve_adapt_method
from dma_kws.stage2.adapt_paths import adapt_data_root, adapt_exp_root
from dma_kws.stage2.sweep_adapt import (
    bind_study_adapt_method,
    run_adaptation_training_trial,
    suggest_adapt_params,
)
from dma_kws.training.adapt_params import merge_adapt_params


def test_adapt_method_defaults_to_lora_and_rejects_unknown_method():
    assert resolve_adapt_method({}) == "lora"
    assert resolve_adapt_method({"method": "qbyt_full"}) == "qbyt_full"
    for value in ("full", "encoder", "", None):
        with pytest.raises(ValueError, match="adapt.method"):
            resolve_adapt_method({"method": value})


@pytest.mark.parametrize("joint", [False, True])
def test_full_qbyt_default_paths_are_isolated_but_reuse_the_corpus(tmp_path, joint):
    config = {
        "paths": {"exp_root": str(tmp_path / "exp"), "processed_root": str(tmp_path / "data")},
        "adapt": {"keyword": "hey eva", "train_phases": ["joint"] if joint else ["tts", "real"]},
    }
    lora_exp = adapt_exp_root(config, "hey eva")
    data = adapt_data_root(config, "hey eva")
    config["adapt"]["method"] = "qbyt_full"
    full_exp = adapt_exp_root(config, "hey eva")
    assert full_exp != lora_exp
    assert full_exp.parent.name == ("stage2_adapt_joint_qbyt_full" if joint else "stage2_adapt_qbyt_full")
    assert adapt_data_root(config, "hey eva") == data
    config["adapt"]["exp_root"] = str(tmp_path / "custom")
    assert adapt_exp_root(config, "hey eva") == tmp_path / "custom"


def test_full_qbyt_overlay_composes_with_the_user_pooling_joint_experiment():
    config = config_to_dict(compose_config(overrides=[
        "+experiment=[icefall_zipformer_stage2_eps_softmin_v41,adapt_joint,adapt_qbyt_full]",
        "adapt.keyword=hey eva",
    ]))
    assert config["adapt"]["method"] == "qbyt_full"
    assert config["adapt"]["phase"] == "joint"
    assert config["adapt"]["train_phases"] == ["joint"]
    assert config["adapt"]["learning_rate"] == 3e-5
    assert config["stage2"]["qbyt_readout_version"] == 4
    assert config["stage2"]["qbyt_readout"]["sink_token"] is True
    assert config["stage2"]["qbyt_readout"]["text_position"] == "learned"
    assert config["stage2"]["qbyt_readout"]["audio_position"] == "relative_bias"
    assert "qbyt_full" in str(adapt_exp_root(config, "hey eva"))


class _RecordingTrial:
    def __init__(self):
        self.calls = {}

    def suggest_categorical(self, name, choices):
        self.calls[name] = choices
        return choices[0]

    def suggest_float(self, name, low, high, **kwargs):
        self.calls[name] = (low, high, kwargs)
        return low


def test_full_qbyt_sweep_searches_full_tuning_budget_without_lora_parameters():
    trial = _RecordingTrial()
    params = suggest_adapt_params(trial, method="qbyt_full", joint=True, search_mix=True)
    assert set(trial.calls) == {"learning_rate", "max_steps", "mix_ratio"}
    assert trial.calls["learning_rate"] == (1e-5, 1e-4, {"log": True})
    assert trial.calls["max_steps"] == [500, 1000, 2000]
    assert not {"rank", "alpha", "alpha_ratio", "lora_targets"} & params.keys()


@pytest.mark.parametrize("incompatible", [
    {"rank": 4}, {"alpha": 8}, {"alpha_ratio": 2},
    {"lora_targets": ["in_proj_weight"]}, {"method": "lora", "learning_rate": 1e-3},
])
def test_full_qbyt_rejects_lora_tuned_params_without_partially_mutating_config(incompatible):
    adapt = {"method": "qbyt_full", "learning_rate": 3e-5, "max_steps": 1000}
    before = copy.deepcopy(adapt)
    with pytest.raises(ValueError):
        merge_adapt_params(adapt, {"learning_rate": 1e-4, **incompatible})
    assert adapt == before


def test_full_qbyt_sequential_trial_hands_off_complete_weights(monkeypatch, tmp_path):
    seen = []

    def run(config, args):
        phase = config["adapt"]["phase"]
        seen.append((phase, args.init_checkpoint, args.resume_checkpoint))
        return {"merged": tmp_path / phase / "stage2_adapted.pt"}

    monkeypatch.setattr("dma_kws.stage2.adapt.run_stage2_adaptation", run)
    result = run_adaptation_training_trial(
        {"adapt": {"keyword": "hey eva", "method": "qbyt_full", "train_phases": ["tts", "real"],
                   "exp_root": str(tmp_path / "exp")}},
        params={"learning_rate": 3e-5, "max_steps": 1000},
        base_args=Stage2AdaptArgs(init_checkpoint=str(tmp_path / "base.pt")),
    )
    assert seen == [
        ("tts", str(tmp_path / "base.pt"), ""),
        ("real", str(tmp_path / "tts" / "stage2_adapted.pt"), ""),
    ]
    assert result["real"]["merged"] == tmp_path / "real" / "stage2_adapted.pt"


def test_full_qbyt_study_rejects_old_lora_trials_and_tracks_its_method():
    attrs = {}
    study = SimpleNamespace(user_attrs=attrs, trials=[object()], set_user_attr=attrs.__setitem__)
    with pytest.raises(ValueError):
        bind_study_adapt_method(study, "qbyt_full")
    assert attrs == {}
    study.trials = []
    bind_study_adapt_method(study, "qbyt_full")
    assert "qbyt_full" in attrs.values()
    with pytest.raises(ValueError):
        bind_study_adapt_method(study, "lora")


def _capture_orchestrator_commands(monkeypatch):
    from scripts import run_keyword_adaptation
    commands = []
    monkeypatch.setattr(
        run_keyword_adaptation.subprocess, "run",
        lambda cmd, **kwargs: commands.append(cmd),
    )
    reporter = SimpleNamespace(
        section=lambda *_: None, print_plan=lambda *_, **__: None,
        info=lambda *_: None, use_rich=False,
    )
    monkeypatch.setattr(run_keyword_adaptation.adapt_console, "adapt_reporter", lambda *_: reporter)
    return run_keyword_adaptation, commands


def test_full_qbyt_orchestrator_resumes_joint_without_an_external_base(monkeypatch, tmp_path):
    script, commands = _capture_orchestrator_commands(monkeypatch)
    overrides = [
        "+experiment=[adapt_joint,adapt_qbyt_full]", "adapt.stage=train",
        "run.resume_from=/checkpoints/resume.ckpt", f"paths.exp_root={tmp_path}",
    ]
    monkeypatch.setattr(sys, "argv", ["run_keyword_adaptation.py", *overrides])
    script.main.__wrapped__(compose_config(overrides=overrides))
    assert len(commands) == 1
    assert "run.resume_from=/checkpoints/resume.ckpt" in commands[0]
    assert "adapt.phase=joint" in commands[0]
    assert not any(arg.startswith("prep.stage2_ckpt=") for arg in commands[0])


def test_full_qbyt_orchestrator_second_phase_overrides_original_base_and_adapter(monkeypatch, tmp_path):
    script, commands = _capture_orchestrator_commands(monkeypatch)
    overrides = [
        "+experiment=adapt_qbyt_full", "adapt.stage=train", "adapt.keyword=hey eva",
        "run.init_checkpoint=/checkpoints/base.pt", f"paths.exp_root={tmp_path}",
    ]
    # Simulate the forwarding surface explicitly: a former adapter override
    # must not win over the full-model handoff appended for the second child.
    monkeypatch.setattr(sys, "argv", [
        "run_keyword_adaptation.py", *overrides, "run.resume_checkpoint=/stale/adapter.pt",
    ])
    script.main.__wrapped__(compose_config(overrides=overrides))
    assert len(commands) == 2
    second = dict(arg.split("=", 1) for arg in commands[1] if "=" in arg)
    assert second["adapt.phase"] == "real"
    assert second["run.init_checkpoint"] == str(tmp_path / "stage2_adapt_qbyt_full" / "hey_eva" / "tts" / "stage2_adapted.pt")
    assert second["run.resume_checkpoint"] == ""
