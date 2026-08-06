import json
import sys
from types import SimpleNamespace

import pytest

from scripts import run_keyword_adaptation
from dma_kws.stage2.adapt_paths import resolve_adapt_train_phases


def test_resolve_adapt_train_phases_keeps_default_and_supports_real_only():
    assert resolve_adapt_train_phases({}) == ("tts", "real")
    assert resolve_adapt_train_phases({"train_phases": ["real"]}) == ("real",)
    assert resolve_adapt_train_phases({"train_phases": "tts,real"}) == ("tts", "real")


@pytest.mark.parametrize("value", [[], ["real", "real"], ["blind_test"]])
def test_resolve_adapt_train_phases_rejects_invalid_values(value):
    with pytest.raises(ValueError, match="train_phases"):
        resolve_adapt_train_phases({"train_phases": value})


def test_libriphrase_eval_forwards_experiment_overrides(tmp_path, monkeypatch):
    commands: list[list[str]] = []

    def fake_run(command, **_kwargs):
        commands.append([str(item) for item in command])
        return SimpleNamespace(stdout=json.dumps({"auc": 0.9, "eer": 0.1}) + "\n")

    monkeypatch.setattr(run_keyword_adaptation.subprocess, "run", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_keyword_adaptation.py",
            "+experiment=adapt_hey_eva_icefall_adapter_v2",
            "stage2.eval.batch_size=32",
        ],
    )

    run_keyword_adaptation._build_eval_report(
        {
            "adapt": {"exp_root": str(tmp_path / "exp")},
            "paths": {"exp_root": str(tmp_path / "fallback")},
        },
        "hey eva",
        "/checkpoints/base.pt",
        "/checkpoints/adapted.pt",
        tmp_path / "data",
    )

    assert len(commands) == 2
    for command in commands:
        assert "+experiment=adapt_hey_eva_icefall_adapter_v2" in command
        assert "stage2.eval.batch_size=32" in command
        assert any(arg.startswith("prep.checkpoint=") for arg in command)
