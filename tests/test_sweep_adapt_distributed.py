"""Regression coverage for Optuna-controller/DDP-worker isolation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dma_kws.stage2.adapt import Stage2AdaptArgs
from dma_kws.stage2 import sweep_adapt


def _config(tmp_path: Path) -> dict:
    return {
        "adapt": {
            "keyword": "hey eva",
            "exp_root": str(tmp_path),
            "max_steps": 10,
        },
        "stage2": {"dataloader": {"persistent_workers": True}},
    }


def test_distributed_trial_launches_fixed_request_under_torchrun(monkeypatch, tmp_path):
    params = {
        "rank": 32,
        "alpha": 64,
        "learning_rate": 1.2e-4,
        "max_steps": 1000,
        "mix_ratio": 0.37,
        "_trial_number": 2,
    }
    seen = {}

    def fake_run(command, *, check, cwd, env):
        assert check is True
        seen["command"] = command
        request_path = Path(command[-1])
        request = json.loads(request_path.read_text(encoding="utf-8"))
        seen["request"] = request
        result_path = Path(request["result_path"])
        result_path.write_text(
            json.dumps(
                {
                    "config": request["config"],
                    "keyword": "hey eva",
                    "params": {
                        key: value
                        for key, value in request["params"].items()
                        if not key.startswith("_")
                    },
                    "tts": {
                        "adapter": str(tmp_path / "adapter.pt"),
                        "merged": str(tmp_path / "stage2_adapted.pt"),
                    },
                    "real": None,
                    "merged_checkpoint": str(tmp_path / "stage2_adapted.pt"),
                }
            ),
            encoding="utf-8",
        )

    monkeypatch.setattr(sweep_adapt.subprocess, "run", fake_run)
    monkeypatch.setattr(sweep_adapt, "_find_free_local_port", lambda: 29400)
    result = sweep_adapt.launch_distributed_adaptation_trial(
        _config(tmp_path),
        params=params,
        base_args=Stage2AdaptArgs(
            init_checkpoint="/checkpoints/base.pt",
            device="cuda",
            devices=2,
        ),
        single_phase=True,
    )

    command = seen["command"]
    assert command[1:3] == ["-m", "torch.distributed.run"]
    assert "--standalone" not in command
    assert "--master_addr=127.0.0.1" in command
    assert any(argument.startswith("--master_port=") for argument in command)
    assert "--nproc_per_node=2" in command
    assert command[-3].endswith("scripts/run_adapt_lora_trial.py")
    assert Path(command[-1]).is_absolute()
    assert Path(seen["request"]["result_path"]).is_absolute()
    assert Path(seen["request"]["config"]["adapt"]["exp_root"]).is_absolute()
    assert seen["request"]["params"] == params
    assert seen["request"]["base_args"]["devices"] == 2
    assert result["params"]["rank"] == 32
    assert isinstance(result["tts"]["adapter"], Path)


def test_multidevice_objective_never_trains_inside_optuna_controller(monkeypatch, tmp_path):
    training = {
        "config": _config(tmp_path),
        "keyword": "hey eva",
        "params": {"rank": 8, "alpha": 16},
        "tts": {"adapter": tmp_path / "adapter.pt"},
        "real": None,
        "merged_checkpoint": tmp_path / "stage2_adapted.pt",
    }
    monkeypatch.setattr(
        sweep_adapt,
        "launch_distributed_adaptation_trial",
        lambda *_args, **_kwargs: training,
    )

    def fail_in_process(*_args, **_kwargs):
        raise AssertionError("Optuna controller must not enter Lightning DDP")

    monkeypatch.setattr(sweep_adapt, "run_adaptation_training_trial", fail_in_process)
    metrics = sweep_adapt.run_adaptation_trial(
        _config(tmp_path),
        params={"rank": 8, "alpha": 16, "_trial_number": 0},
        base_args=Stage2AdaptArgs(devices=2),
        single_phase=True,
        eval_target_fn=lambda _cfg, _ckpt, _keyword: 0.9,
        eval_lph_fn=lambda _ckpt: 0.8,
    )

    assert metrics["target_auc"] == 0.9
    assert metrics["lph_auc"] == 0.8


def test_sweep_rejects_runtime_options_it_cannot_honor(tmp_path):
    for field in ("resume_from", "params_file"):
        kwargs = {field: "unexpected"}
        with pytest.raises(ValueError, match=field):
            sweep_adapt.run_adaptation_training_trial(
                _config(tmp_path),
                params={"rank": 8, "alpha": 16},
                base_args=Stage2AdaptArgs(**kwargs),
                single_phase=True,
            )
