"""Joint adaptation keeps one training run and separate evaluation sources."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from dma_kws.config import compose_config, config_to_dict
from dma_kws.stage2 import adapt as adapt_module
from dma_kws.stage2 import sweep_adapt
from dma_kws.stage2.adapt_paths import (
    adapt_exp_root,
    resolve_adapt_train_phases,
    target_eval_manifest,
)
from scripts import run_keyword_adaptation, sweep_adapt_lora


def test_joint_phase_resolution_requires_one_explicit_phase():
    assert resolve_adapt_train_phases({"train_phases": ["joint"]}) == ("joint",)
    assert resolve_adapt_train_phases({"train_phases": " JOINT "}) == ("joint",)
    for adapt in (
        {"train_phases": ["joint", "real"]},
        {"train_phases": ["tts", "joint"]},
        {"phase": "joint"},
        {"phase": "joint", "train_phases": ["real"]},
    ):
        with pytest.raises(ValueError, match="train_phases"):
            resolve_adapt_train_phases(adapt)


@pytest.mark.parametrize("joint,expected", [(False, [1000, 2000, 3000]), (True, [2000, 4000, 6000])])
def test_joint_sweep_uses_the_same_total_update_budget(joint, expected):
    choices = {}

    class Trial:
        def suggest_categorical(self, name, values):
            choices[name] = values
            return values[0]

        def suggest_float(self, name, low, high, **kwargs):
            return low

    params = sweep_adapt.suggest_adapt_params(Trial(), joint=joint)
    assert choices["max_steps"] == expected
    assert params["max_steps"] == expected[0]


def test_joint_overlay_preserves_checkpoint_architecture_and_isolates_outputs(tmp_path):
    base = config_to_dict(
        compose_config("adapt_hey_eva_icefall_adapter_v2", overrides=[f"paths.exp_root={tmp_path}"])
    )
    joint = config_to_dict(
        compose_config(
            overrides=[
                "+experiment=[adapt_hey_eva_icefall_adapter_v2,adapt_joint]",
                f"paths.exp_root={tmp_path}",
            ]
        )
    )
    assert joint["adapt"]["phase"] == "joint"
    assert joint["adapt"]["train_phases"] == ["joint"]
    assert joint["adapt"]["max_steps"] == 6000
    assert joint["adapt"]["mix_ratio"] == 0.5
    assert joint["adapt"]["joint"]["real_fraction"] == 0.6
    assert joint["adapt"]["params_file"] == ""
    assert joint["adapt"]["sweep"]["storage"] == ""
    for section in ("stage1", "tokenizer", "fbank"):
        assert joint[section] == base[section]
    for key in base["stage2"]:
        if key != "background_negative":
            assert joint["stage2"][key] == base["stage2"][key]
    assert joint["stage2"]["background_negative"]["enabled"] is True
    assert joint["stage2"]["background_negative"]["probability"] == 0.4
    assert adapt_exp_root(joint, "hey eva") == tmp_path / "stage2_adapt_joint" / "hey_eva"


def test_joint_orchestrator_dispatches_training_once(tmp_path, monkeypatch):
    cfg = compose_config(
        overrides=[
            "+experiment=adapt_joint",
            "adapt.stage=train",
            "prep.stage2_ckpt=/checkpoints/base.pt",
            f"paths.exp_root={tmp_path}",
        ]
    )
    legacy_best = tmp_path / "stage2_adapt" / "hey_eva" / "sweep" / "best_params.yaml"
    legacy_best.parent.mkdir(parents=True)
    legacy_best.write_text("max_steps: 1000\n", encoding="utf-8")
    commands = []
    monkeypatch.setattr(
        run_keyword_adaptation,
        "_run_script",
        lambda script, overrides, reporter: commands.append((script, overrides)),
    )
    reporter = SimpleNamespace(section=lambda *_: None, print_plan=lambda *_, **__: None)
    monkeypatch.setattr(run_keyword_adaptation.adapt_console, "adapt_reporter", lambda *_: reporter)

    run_keyword_adaptation.main.__wrapped__(cfg)

    assert len(commands) == 1
    script, overrides = commands[0]
    assert script == "adapt_stage2_keyword.py"
    assert "adapt.phase=joint" in overrides
    assert not any(value.startswith("adapt.params_file=") for value in overrides)


def test_joint_sweep_trains_once_and_roundtrips_joint_artifacts(tmp_path, monkeypatch):
    calls = []

    def fake_train(config, args):
        calls.append((config["adapt"]["phase"], args.resume_checkpoint))
        return {"adapter": tmp_path / "joint.pt", "merged": tmp_path / "stage2_adapted.pt"}

    monkeypatch.setattr(adapt_module, "run_stage2_adaptation", fake_train)
    result = sweep_adapt.run_adaptation_training_trial(
        {
            "adapt": {
                "keyword": "hey eva",
                "phase": "joint",
                "train_phases": ["joint"],
                "exp_root": str(tmp_path),
            },
        },
        params={"rank": 8, "alpha": 16, "max_steps": 6000},
        base_args=adapt_module.Stage2AdaptArgs(init_checkpoint="/checkpoints/base.pt"),
    )
    assert calls == [("joint", "")]
    assert result["tts"] is None
    assert result["real"] is None
    assert result["joint"]["adapter"] == tmp_path / "joint.pt"
    payload = json.loads(json.dumps(sweep_adapt.serialize_training_result(result)))
    restored = sweep_adapt._deserialize_training_result(payload)
    assert restored["joint"] == result["joint"]


def test_joint_sweep_target_is_real_manifest(tmp_path, monkeypatch):
    import dma_kws.config
    import dma_kws.pathing
    import dma_kws.stage2.adapt_dataset
    import dma_kws.tokenizer

    class ManifestObserved(Exception):
        pass

    seen = []

    def record_dataset(**kwargs):
        seen.append(kwargs["manifest_path"])
        raise ManifestObserved

    monkeypatch.setattr(dma_kws.config, "get_tokenizer_config", lambda _: {})
    monkeypatch.setattr(dma_kws.pathing, "resolve_dict_path", lambda _: "unused")
    monkeypatch.setattr(
        dma_kws.tokenizer, "load_char_tokenizer", lambda *_, **__: SimpleNamespace(_symbol_table=["a"])
    )
    monkeypatch.setattr(dma_kws.stage2.adapt_dataset, "TargetKeywordValDataset", record_dataset)
    with pytest.raises(ManifestObserved):
        sweep_adapt_lora._eval_target_auc(
            {"adapt": {"phase": "joint", "data_root": str(tmp_path)}},
            "/checkpoints/unused.pt",
            "hey eva",
        )
    assert seen == [tmp_path / "manifests" / "real_eval.csv"]
    assert target_eval_manifest(tmp_path, "tts").name == "tts_eval.csv"


@pytest.mark.parametrize("configured_phonemes", ["", "HH EY IY V AH"])
def test_joint_report_keeps_real_tts_lph_and_musan_separate(tmp_path, monkeypatch, configured_phonemes):
    data_root = tmp_path / "data"
    manifests = data_root / "manifests"
    manifests.mkdir(parents=True)
    for source in ("real", "tts"):
        (manifests / f"{source}_eval.csv").touch()
    (manifests / "real_eval.csv").write_text(
        "audio_path,keyword_phonemes\npositive.wav, HH EY  IY V AH \nnegative.wav,\n"
    )
    musan_root = tmp_path / "musan"
    musan_root.mkdir()
    background_list = tmp_path / "val_background.list"
    background_list.touch()
    config = {
        "adapt": {
            "phase": "joint",
            "train_phases": ["joint"],
            "exp_root": str(tmp_path / "exp"),
            "joint": {"background_eval_list": str(background_list)},
        },
        "prep": {"musan_root": str(musan_root), "keyword_phonemes": configured_phonemes},
    }
    commands = []

    def fake_run(command, **kwargs):
        commands.append(command)
        if command[1].endswith("eval_stage2_libriphrase.py"):
            return SimpleNamespace(stdout=json.dumps({"auc": 0.8}) + "\n")
        output_arg = next(
            value for value in command
            if value.startswith(("prep.stage2_clip_output_dir=", "prep.output_dir="))
        )
        output_dir = Path(output_arg.split("=", 1)[1])
        output_dir.mkdir(parents=True)
        summary = {"metrics": {"fa_per_hour": 0.2}}
        if command[1].endswith("eval_stage2_clips.py"):
            summary = {"metrics": {"auc": 0.9 if "tts" in str(output_dir) else 0.85}}
        (output_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
        return SimpleNamespace(stdout="")

    monkeypatch.setattr(run_keyword_adaptation.subprocess, "run", fake_run)

    def fake_clip_manifest(source, keyword, output, *, manifest_root):
        assert manifest_root == data_root
        output.touch()

    monkeypatch.setattr(
        run_keyword_adaptation,
        "clips_eval_manifest_from_adapt",
        fake_clip_manifest,
    )
    model_override = "+experiment=[adapt_hey_eva_icefall_adapter_v2,adapt_joint]"
    phonemes_override = "prep.keyword_phonemes='HH EY IY V AH'"
    forwarded = [phonemes_override] if configured_phonemes else []
    monkeypatch.setattr(sys, "argv", ["run_keyword_adaptation.py", model_override, *forwarded])
    report = run_keyword_adaptation._build_eval_report(
        config, "hey eva", "/base.pt", "/adapted.pt", data_root
    )
    assert report["target_base"] == {"auc": 0.85}
    assert report["tts_adapted"] == {"auc": 0.9}
    assert report["lph_adapted"] == {"auc": 0.8}
    assert report["musan_adapted"]["metrics"]["fa_per_hour"] == 0.2
    assert len(commands) == 8
    for command in commands:
        assert model_override in command
        if configured_phonemes:
            assert phonemes_override in command
    musan_commands = [command for command in commands if command[1].endswith("eval_musan_fa.py")]
    for command in musan_commands:
        assert f"prep.musan_audio_list_path={background_list}" in command
        assert f"prep.musan_root={musan_root}" in command
        assert "prep.keyword='hey eva'" in command
        assert phonemes_override in command


@pytest.mark.parametrize("configured,manifest_values,match", [
    ("HH EY EH V AH", ["HH EY IY V AH", ""], "prep.keyword_phonemes differs"),
    ("", ["HH EY IY V AH", "HH EY EH V AH"], "inconsistent keyword_phonemes"),
])
def test_joint_musan_report_rejects_pronunciation_conflicts_before_scoring(
    tmp_path, monkeypatch, configured, manifest_values, match,
):
    root = tmp_path / "data"
    manifest = root / "manifests/real_eval.csv"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        "audio_path,keyword_phonemes\n"
        + "".join(f"clip_{index}.wav,{value}\n" for index, value in enumerate(manifest_values))
    )
    musan_root = tmp_path / "musan"
    musan_root.mkdir()
    background_list = tmp_path / "eval.list"
    background_list.touch()

    def unexpected_scoring(*_args, **_kwargs):
        raise AssertionError("Pronunciation conflicts must fail before evaluating any clips")

    monkeypatch.setattr(run_keyword_adaptation.subprocess, "run", unexpected_scoring)
    with pytest.raises(ValueError, match=match):
        run_keyword_adaptation._build_eval_report(
            {
                "adapt": {
                    "phase": "joint", "train_phases": ["joint"],
                    "exp_root": str(tmp_path / "exp"),
                    "joint": {"background_eval_list": str(background_list)},
                },
                "prep": {"musan_root": str(musan_root), "keyword_phonemes": configured},
            },
            "hey eva", "/base.pt", "/adapted.pt", root,
        )


def test_joint_musan_report_requires_catalog_root(tmp_path):
    background_list = tmp_path / "background.list"
    background_list.touch()
    with pytest.raises(SystemExit, match="prep.musan_root"):
        run_keyword_adaptation._build_eval_report(
            {
                "adapt": {
                    "phase": "joint",
                    "train_phases": ["joint"],
                    "exp_root": str(tmp_path / "exp"),
                    "joint": {"background_eval_list": str(background_list)},
                }
            },
            "hey eva", "/base.pt", "/adapted.pt", tmp_path / "data",
        )


@pytest.mark.parametrize("mode", ["online", "fbank_cache"])
def test_joint_eval_only_rejects_musan_training_sources_before_scoring(tmp_path, monkeypatch, mode):
    root = tmp_path / "musan"
    audio = root / "noise/shared.wav"
    audio.parent.mkdir(parents=True)
    audio.touch()
    train_list = tmp_path / "train.list"
    eval_list = tmp_path / "eval.list"
    train_list.write_text(str(audio) + "\n")
    eval_list.write_text(str(audio) + "\n")
    background = dict(enabled=True, mode=mode, audio_list_path=str(train_list))
    if mode == "fbank_cache":
        manifest = tmp_path / "cache.json"
        manifest.write_text(json.dumps(dict(
            split_role="train", split=dict(musan_root=str(root)),
            recordings=dict(path="recordings.jsonl"),
        )))
        (tmp_path / "recordings.jsonl").write_text(json.dumps(dict(
            source_id="noise/shared.wav", relative_path="noise/shared.wav",
        )) + "\n")
        background["cache_manifest"] = str(manifest)
    config = {
        "adapt": {
            "stage": "eval", "phase": "joint", "train_phases": ["joint"],
            "exp_root": str(tmp_path / "exp"),
            "joint": {"background_eval_list": str(eval_list)},
        },
        "stage2": {"background_negative": background},
        "prep": {"musan_root": str(root)},
    }

    def unexpected_scoring(*_args, **_kwargs):
        raise AssertionError("The split audit must run before any evaluation subprocess")

    monkeypatch.setattr(run_keyword_adaptation.subprocess, "run", unexpected_scoring)
    with pytest.raises(ValueError, match="MUSAN train/eval source leakage"):
        run_keyword_adaptation._build_eval_report(
            config, "hey eva", "/base.pt", "/adapted.pt", tmp_path / "data",
        )


def test_clip_manifest_copy_resolves_raw_paths_from_adaptation_root(tmp_path):
    import csv

    from dma_kws.inference.manifest import load_manifest
    from dma_kws.stage2.adapt_paths import clips_eval_manifest_from_adapt

    root = tmp_path / "data"
    source = root / "manifests/real_eval.csv"
    source.parent.mkdir(parents=True)
    relative_audio = "raw/real/eval/positive/sample.wav"
    absolute_audio = tmp_path / "external/negative.wav"
    with source.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["audio_path", "text", "label", "speaker_id"])
        writer.writeheader()
        writer.writerows([
            dict(audio_path=relative_audio, text="hey eva", label=1, speaker_id="positive"),
            dict(audio_path=str(absolute_audio), text="hey ava", label=0, speaker_id="negative"),
        ])
    output = tmp_path / "experiment/reports/real_eval_clips.csv"
    clips_eval_manifest_from_adapt(source, "hey eva", output, manifest_root=root)
    copied = load_manifest(output)
    assert copied[0]["audio_path"] == str(root / relative_audio)
    assert copied[1]["audio_path"] == str(absolute_audio)
    assert copied[0]["speaker_id"] == "positive"
