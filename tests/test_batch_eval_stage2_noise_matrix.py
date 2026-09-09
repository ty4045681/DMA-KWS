import argparse
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "batch_eval_stage2_noise_matrix.py"
)
SPEC = importlib.util.spec_from_file_location(
    "batch_eval_stage2_noise_matrix", SCRIPT_PATH
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _make_inputs(tmp_path: Path, *, model_count: int = 1):
    audio_negative = tmp_path / "negative.wav"
    audio_positive = tmp_path / "positive.wav"
    audio_negative.write_bytes(b"negative")
    audio_positive.write_bytes(b"positive")
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "audio_path,keyword,label\n"
        f"{audio_negative.name},hey eva,0\n"
        f"{audio_positive.name},hey eva,1\n",
        encoding="utf-8",
    )
    checkpoints = []
    for index in range(model_count):
        checkpoint = tmp_path / f"model_{index}.pt"
        torch.save(
            {
                "model_state_dict": {
                    "encoder.weight": torch.tensor([float(index)]),
                    "qbyt.weight": torch.tensor([float(index)]),
                }
            },
            checkpoint,
        )
        checkpoints.append(checkpoint)
    musan_root = tmp_path / "musan"
    (musan_root / "noise").mkdir(parents=True)
    (musan_root / "music").mkdir()
    (musan_root / "speech").mkdir()
    (musan_root / "noise" / "noise.wav").write_bytes(b"noise")
    (musan_root / "music" / "music.wav").write_bytes(b"music")
    (musan_root / "speech" / "speech.wav").write_bytes(b"speech")
    return manifest, checkpoints, musan_root


def test_condition_catalog_covers_requested_15_runs_per_model():
    assert [condition.name for condition in MODULE.CONDITIONS] == [
        "clean",
        "volume_variation",
        "stationary_snr10",
        "stationary_snr20",
        "burst_snr10",
        "burst_snr20",
        "musan_noise_snr10",
        "musan_noise_snr20",
        "musan_noise_snr10_music_snr10",
        "musan_noise_snr20_music_snr20",
        "musan_noise_snr10_speech_equal",
        "musan_noise_snr20_speech_equal",
        "musan_speech_quieter",
        "musan_speech_equal",
        "musan_speech_louder",
    ]
    assert MODULE.CONDITION_BY_NAME["musan_speech_quieter"].speech_relative_db == -6
    assert MODULE.CONDITION_BY_NAME["musan_speech_equal"].speech_relative_db == 0
    assert MODULE.CONDITION_BY_NAME["musan_speech_louder"].speech_relative_db == 6
    for level in (10, 20):
        condition = MODULE.CONDITION_BY_NAME[
            f"musan_noise_snr{level}_music_snr{level}"
        ]
        assert condition.snr_db == level
        assert condition.music_snr_db == level
        assert condition.requires_noise is True
        assert condition.requires_music is True


def test_four_model_dry_run_expands_to_60_jobs_and_exports_five_wavs(
    tmp_path, capsys
):
    manifest, checkpoints, musan_root = _make_inputs(tmp_path, model_count=4)
    argv = [
        "--manifest",
        str(manifest),
        "--musan-root",
        str(musan_root),
        "--output-root",
        str(tmp_path / "out"),
        "--dry-run",
    ]
    for index, checkpoint in enumerate(checkpoints):
        argv.extend(["--model", f"candidate_{index}={checkpoint}"])

    args = MODULE.build_parser().parse_args(argv)
    payload = MODULE.run_batch(args)

    assert payload["counts"] == {"total": 60, "planned": 60}
    assert payload["audio_export"] == {
        "mode": "random",
        "count": 5,
        "seed": 2025,
    }
    output = capsys.readouterr().out
    assert output.count("eval_stage2_clips.py") == 60
    assert "+eval_condition=clean" in output
    assert "+eval_condition=volume_variation" in output
    assert "+eval_condition=musan_noise_snr10_music_snr10" in output
    assert "+eval_condition=musan_speech_louder" in output
    assert "prep.audio_export.mode=random" in output
    assert "prep.audio_export.count=5" in output
    assert "prep.audio_export.seed=2025" in output
    assert not (tmp_path / "out").exists()


def test_model_specific_experiment_and_overrides_are_isolated(tmp_path):
    _, checkpoints, _ = _make_inputs(tmp_path, model_count=2)
    args = MODULE.build_parser().parse_args(
        [
            "--model",
            f"base={checkpoints[0]}::paper_ls460",
            "--model",
            f"adapted={checkpoints[1]}",
            "--model-experiment",
            "adapted=icefall_zipformer_stage2_adapter_v2",
            "--model-override",
            "adapted=stage2.phoneme_adapter.enabled=true",
        ]
    )

    models = MODULE.resolve_models(args)

    assert models[0].experiment == "paper_ls460"
    assert models[0].overrides == ()
    assert models[1].experiment == "icefall_zipformer_stage2_adapter_v2"
    assert models[1].overrides == ("stage2.phoneme_adapter.enabled=true",)


def test_models_file_resolves_relative_checkpoints_from_its_own_directory(tmp_path):
    _, checkpoints, _ = _make_inputs(tmp_path)
    spec_dir = tmp_path / "specs"
    spec_dir.mkdir()
    models_file = spec_dir / "models.txt"
    models_file.write_text(
        f"candidate=../{checkpoints[0].name}\n",
        encoding="utf-8",
    )
    args = MODULE.build_parser().parse_args(["--models-file", str(models_file)])

    models = MODULE.resolve_models(args)

    assert models[0].checkpoint == checkpoints[0].resolve()


def test_augmentation_overrides_cannot_make_condition_metadata_lie():
    parser = MODULE.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            ["--override", "prep.musan_mix.noise.snr_db=3"]
        )
    with pytest.raises(SystemExit):
        parser.parse_args(
            ["--override", "+prep.audio_export.mode=all"]
        )

    args = parser.parse_args(
        [
            "--model",
            "candidate=model.pt",
            "--model-override",
            "candidate=prep.audio_aug.transforms.noise_mix.enabled=true",
        ]
    )
    with pytest.raises(MODULE.BatchConfigError, match="runner-owned key"):
        MODULE.resolve_models(args)


def test_build_eval_command_forwards_condition_seed_and_paths(tmp_path):
    manifest, checkpoints, musan_root = _make_inputs(tmp_path)
    model = MODULE.ModelSpec(
        "candidate",
        checkpoints[0],
        "icefall_zipformer_stage2",
        ("stage1.stream.chunk_size=32",),
    )
    output_dir = tmp_path / "output with spaces"

    command = MODULE.build_eval_command(
        model=model,
        condition=MODULE.CONDITION_BY_NAME["musan_noise_snr10"],
        manifest=manifest,
        musan_root=musan_root,
        output_dir=output_dir,
        device="cuda",
        batch_size=16,
        num_workers=2,
        seed=99,
        common_overrides=("prep.plot_min_recall=0.9",),
    )

    assert command[:4] == (
        sys.executable,
        str(MODULE.EVAL_SCRIPT),
        "+experiment=icefall_zipformer_stage2",
        "+eval_condition=musan_noise_snr10",
    )
    assert "stage1.stream.chunk_size=32" in command
    assert "prep.plot_min_recall=0.9" in command
    assert f"prep.output_dir={json.dumps(str(output_dir))}" in command
    assert "prep.musan_mix.seed=99" in command
    assert "prep.audio_aug.seed=99" in command
    assert "prep.audio_export.mode=random" in command
    assert "prep.audio_export.count=5" in command
    assert "prep.audio_export.seed=99" in command
    assert "prep.batch_size=16" in command
    assert "prep.num_workers=2" in command


def test_audio_export_cli_supports_all_none_and_independent_seed():
    parser = MODULE.build_parser()

    defaults = parser.parse_args([])
    assert MODULE.resolve_audio_export(
        defaults.export_audio,
        seed=7,
        export_seed=defaults.export_audio_seed,
    ) == MODULE.AudioExportSpec(mode="random", count=5, seed=7)

    all_args = parser.parse_args(
        ["--export-audio", "all", "--export-audio-seed", "19"]
    )
    assert MODULE.resolve_audio_export(
        all_args.export_audio,
        seed=7,
        export_seed=all_args.export_audio_seed,
    ) == MODULE.AudioExportSpec(mode="all", count=0, seed=19)

    none_args = parser.parse_args(["--export-audio", "none"])
    assert MODULE.resolve_audio_export(
        none_args.export_audio,
        seed=7,
        export_seed=none_args.export_audio_seed,
    ) == MODULE.AudioExportSpec(mode="disabled", count=0, seed=7)

    with pytest.raises(SystemExit):
        parser.parse_args(["--export-audio", "-1"])
    with pytest.raises(SystemExit):
        parser.parse_args(["--export-audio", "sometimes"])


def test_job_fingerprint_does_not_depend_on_other_selected_conditions(tmp_path):
    manifest, checkpoints, musan_root = _make_inputs(tmp_path)
    model = MODULE.ModelSpec(
        "candidate", checkpoints[0].resolve(), "icefall_zipformer_stage2"
    )
    common = dict(
        models=[model],
        manifest=manifest.resolve(),
        musan_root=musan_root.resolve(),
        output_root=(tmp_path / "out").resolve(),
        device="cuda",
        batch_size=0,
        num_workers=0,
        seed=2025,
        common_overrides=(),
    )

    matrix_jobs, _ = MODULE.build_jobs(
        conditions=[
            MODULE.CONDITION_BY_NAME["clean"],
            MODULE.CONDITION_BY_NAME["musan_noise_snr10"],
            MODULE.CONDITION_BY_NAME["musan_speech_equal"],
        ],
        **common,
    )
    single_jobs, _ = MODULE.build_jobs(
        conditions=[MODULE.CONDITION_BY_NAME["musan_noise_snr10"]],
        **common,
    )

    matrix_noise = next(
        job for job in matrix_jobs if job.condition.name == "musan_noise_snr10"
    )
    assert matrix_noise.fingerprint == single_jobs[0].fingerprint

    (tmp_path / "negative.wav").write_bytes(b"changed-audio")
    changed_jobs, _ = MODULE.build_jobs(
        conditions=[MODULE.CONDITION_BY_NAME["musan_noise_snr10"]],
        **common,
    )
    assert changed_jobs[0].fingerprint != single_jobs[0].fingerprint

    music_jobs_before, _ = MODULE.build_jobs(
        conditions=[
            MODULE.CONDITION_BY_NAME["musan_noise_snr10_music_snr10"]
        ],
        **common,
    )
    (musan_root / "music" / "music.wav").write_bytes(b"changed-music")
    music_jobs_after, _ = MODULE.build_jobs(
        conditions=[
            MODULE.CONDITION_BY_NAME["musan_noise_snr10_music_snr10"]
        ],
        **common,
    )
    assert music_jobs_after[0].fingerprint != music_jobs_before[0].fingerprint
    noise_jobs_after_music_change, _ = MODULE.build_jobs(
        conditions=[MODULE.CONDITION_BY_NAME["musan_noise_snr10"]],
        **common,
    )
    assert (
        noise_jobs_after_music_change[0].fingerprint
        == changed_jobs[0].fingerprint
    )

    all_export_jobs, _ = MODULE.build_jobs(
        conditions=[MODULE.CONDITION_BY_NAME["musan_noise_snr10"]],
        audio_export=MODULE.AudioExportSpec(mode="all", count=0, seed=2025),
        **common,
    )
    different_export_seed_jobs, _ = MODULE.build_jobs(
        conditions=[MODULE.CONDITION_BY_NAME["musan_noise_snr10"]],
        audio_export=MODULE.AudioExportSpec(mode="random", count=5, seed=99),
        **common,
    )
    assert all_export_jobs[0].fingerprint != changed_jobs[0].fingerprint
    assert different_export_seed_jobs[0].fingerprint != changed_jobs[0].fingerprint


def test_validate_inputs_requires_both_binary_labels_and_needed_musan_subset(tmp_path):
    manifest, checkpoints, musan_root = _make_inputs(tmp_path)
    model = MODULE.ModelSpec(
        "candidate", checkpoints[0], "icefall_zipformer_stage2"
    )
    one_class = tmp_path / "one_class.csv"
    one_class.write_text(
        "audio_path,keyword,label\nnegative.wav,hey eva,0\n",
        encoding="utf-8",
    )

    with pytest.raises(MODULE.BatchConfigError, match="both label=0 and label=1"):
        MODULE.validate_inputs(
            manifest=one_class,
            models=[model],
            conditions=[MODULE.CONDITION_BY_NAME["clean"]],
            musan_root=None,
        )

    validated = MODULE.validate_inputs(
        manifest=manifest,
        models=[model],
        conditions=[MODULE.CONDITION_BY_NAME["volume_variation"]],
        musan_root=None,
    )
    assert len(validated) == 2

    (musan_root / "music" / "music.wav").unlink()
    (musan_root / "music").rmdir()
    with pytest.raises(MODULE.BatchConfigError, match="MUSAN subset not found"):
        MODULE.validate_inputs(
            manifest=manifest,
            models=[model],
            conditions=[
                MODULE.CONDITION_BY_NAME["musan_noise_snr10_music_snr10"]
            ],
            musan_root=musan_root,
        )

    (musan_root / "speech" / "speech.wav").unlink()
    (musan_root / "speech").rmdir()
    with pytest.raises(MODULE.BatchConfigError, match="MUSAN subset not found"):
        MODULE.validate_inputs(
            manifest=manifest,
            models=[model],
            conditions=[MODULE.CONDITION_BY_NAME["musan_speech_equal"]],
            musan_root=musan_root,
        )


def test_validate_inputs_rejects_adapter_only_lora_content(tmp_path):
    manifest, _, _ = _make_inputs(tmp_path)
    adapter = tmp_path / "adapter_hey_eva.pt"
    torch.save(
        {
            "checkpoint_kind": "stage2_lora_adapter",
            "lora_state_dict": {"qbyt.layer.lora_A": torch.zeros(1)},
        },
        adapter,
    )
    model = MODULE.ModelSpec(
        "candidate", adapter, "icefall_zipformer_stage2"
    )

    with pytest.raises(MODULE.BatchConfigError, match="adapter-only LoRA"):
        MODULE.validate_inputs(
            manifest=manifest,
            models=[model],
            conditions=[MODULE.CONDITION_BY_NAME["clean"]],
            musan_root=None,
        )


def test_run_job_writes_status_then_reuses_matching_success(tmp_path, monkeypatch):
    manifest, checkpoints, _ = _make_inputs(tmp_path)
    checkpoint_sha256 = MODULE._sha256_file(checkpoints[0])
    model = MODULE.ModelSpec(
        "candidate", checkpoints[0], "icefall_zipformer_stage2"
    )
    output_dir = tmp_path / "outputs" / "candidate" / "clean"
    job = MODULE.JobSpec(
        model=model,
        condition=MODULE.CONDITION_BY_NAME["clean"],
        output_dir=output_dir,
        fingerprint="fingerprint",
        command=(sys.executable, str(MODULE.EVAL_SCRIPT)),
    )
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / MODULE.RESULTS_FILENAME).write_text("{}\n{}\n", encoding="utf-8")
        (output_dir / MODULE.SUMMARY_FILENAME).write_text(
            json.dumps(
                {
                    "num_samples": 2,
                    "num_skipped": 0,
                    "output_dir": str(output_dir.resolve()),
                    "provenance": {
                        "checkpoint": {"sha256": checkpoint_sha256}
                    },
                    "metrics": {
                        "accuracy": 0.75,
                        "recall": 0.5,
                        "fpr": 0.0,
                    },
                }
            ),
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(MODULE.subprocess, "run", fake_run)

    first = MODULE.run_job(
        job, checkpoint_sha256=checkpoint_sha256, force=False
    )
    second = MODULE.run_job(
        job, checkpoint_sha256=checkpoint_sha256, force=False
    )

    assert first["status"] == "succeeded"
    assert first["accuracy"] == 0.75
    assert second["status"] == "cached"
    assert len(calls) == 1
    status = json.loads((output_dir / MODULE.STATUS_FILENAME).read_text())
    assert status["status"] == "succeeded"
    assert status["fingerprint"] == "fingerprint"


def test_cached_summary_rejects_missing_exported_wav(tmp_path):
    _, checkpoints, _ = _make_inputs(tmp_path)
    checkpoint_sha256 = MODULE._sha256_file(checkpoints[0])
    output_dir = tmp_path / "outputs" / "candidate" / "clean"
    export_dir = output_dir / "exported_audio"
    export_dir.mkdir(parents=True)
    wav_path = export_dir / "row_000000.wav"
    wav_path.write_bytes(b"wav")
    manifest_path = export_dir / "index.jsonl"
    manifest_path.write_text(
        json.dumps(
            {
                "row_index": 0,
                "exported_audio_path": str(wav_path),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    job = MODULE.JobSpec(
        model=MODULE.ModelSpec(
            "candidate", checkpoints[0], "icefall_zipformer_stage2"
        ),
        condition=MODULE.CONDITION_BY_NAME["clean"],
        output_dir=output_dir,
        fingerprint="fingerprint",
        command=(sys.executable, str(MODULE.EVAL_SCRIPT)),
        audio_export=MODULE.AudioExportSpec(mode="random", count=1, seed=7),
    )
    (output_dir / MODULE.RESULTS_FILENAME).write_text("{}\n{}\n", encoding="utf-8")
    MODULE._atomic_json(
        output_dir / MODULE.SUMMARY_FILENAME,
        {
            "num_samples": 2,
            "output_dir": str(output_dir.resolve()),
            "provenance": {"checkpoint": {"sha256": checkpoint_sha256}},
            "audio_exports": {
                "status": "generated",
                "mode": "random",
                "requested_count": 1,
                "seed": 7,
                "num_exported": 1,
                "directory": str(export_dir),
                "manifest": str(manifest_path),
            },
        },
    )
    MODULE._atomic_json(
        output_dir / MODULE.STATUS_FILENAME,
        {
            "status": "succeeded",
            "attempt": 1,
            "fingerprint": job.fingerprint,
        },
    )

    assert MODULE.cached_summary(job, checkpoint_sha256=checkpoint_sha256) is not None
    wav_path.unlink()
    assert MODULE.cached_summary(job, checkpoint_sha256=checkpoint_sha256) is None


def test_fail_fast_preserves_later_matching_cache_status(tmp_path, monkeypatch):
    manifest, checkpoints, _ = _make_inputs(tmp_path)
    output_root = tmp_path / "matrix"
    args = MODULE.build_parser().parse_args(
        [
            "--manifest",
            str(manifest),
            "--model",
            f"candidate={checkpoints[0]}",
            "--output-root",
            str(output_root),
            "--condition",
            "clean",
            "--condition",
            "stationary_snr10",
            "--export-audio",
            "none",
            "--fail-fast",
        ]
    )
    models = MODULE.resolve_models(args)
    conditions = MODULE.resolve_conditions(args.condition)
    jobs, identities = MODULE.build_jobs(
        models=models,
        conditions=conditions,
        manifest=manifest.resolve(),
        musan_root=None,
        output_root=output_root.resolve(),
        device="cuda",
        batch_size=0,
        num_workers=0,
        seed=2025,
        common_overrides=(),
        audio_export=MODULE.AudioExportSpec(
            mode="disabled", count=0, seed=2025
        ),
    )
    cached_job = jobs[1]
    checkpoint_sha256 = identities["models"]["candidate"]["sha256"]
    cached_job.output_dir.mkdir(parents=True)
    (cached_job.output_dir / MODULE.RESULTS_FILENAME).write_text(
        "{}\n{}\n", encoding="utf-8"
    )
    MODULE._atomic_json(
        cached_job.output_dir / MODULE.SUMMARY_FILENAME,
        {
            "num_samples": 2,
            "num_skipped": 0,
            "output_dir": str(cached_job.output_dir),
            "provenance": {"checkpoint": {"sha256": checkpoint_sha256}},
            "metrics": {"accuracy": 1.0},
        },
    )
    MODULE._atomic_json(
        cached_job.output_dir / MODULE.STATUS_FILENAME,
        {
            "status": "succeeded",
            "attempt": 1,
            "fingerprint": cached_job.fingerprint,
        },
    )

    def fail_first(job, *, checkpoint_sha256, force):
        return MODULE._failure_row(
            job,
            status="failed",
            attempt=1,
            checkpoint_sha256=checkpoint_sha256,
            returncode=1,
            elapsed_seconds=0.1,
            error="boom",
        )

    monkeypatch.setattr(MODULE, "run_job", fail_first)

    payload = MODULE.run_batch(args)

    assert payload["counts"]["failed"] == 1
    assert payload["counts"]["cached"] == 1
    assert payload["counts"]["pending"] == 0
    assert payload["runs"][1]["condition"] == "stationary_snr10"
    assert payload["runs"][1]["status"] == "cached"


def test_matrix_summary_keeps_failed_and_successful_rows(tmp_path):
    rows = [
        {field: None for field in MODULE.CSV_FIELDS},
        {field: None for field in MODULE.CSV_FIELDS},
    ]
    rows[0].update({"model": "a", "condition": "clean", "status": "succeeded"})
    rows[1].update(
        {"model": "b", "condition": "clean", "status": "failed", "error": "boom"}
    )

    payload = MODULE.write_matrix_summary(
        output_root=tmp_path,
        manifest=tmp_path / "manifest.csv",
        musan_root=None,
        seed=2025,
        rows=rows,
    )

    assert payload["counts"]["total"] == 2
    assert payload["counts"]["succeeded"] == 1
    assert payload["counts"]["failed"] == 1
    saved = json.loads((tmp_path / MODULE.MATRIX_JSON_FILENAME).read_text())
    assert saved["runs"][1]["error"] == "boom"
    csv_text = (tmp_path / MODULE.MATRIX_CSV_FILENAME).read_text()
    assert "model,condition,family" in csv_text
    assert "b,clean" in csv_text


def test_any_mode_dry_run_accepts_keyword_labels_without_top_level_label(tmp_path, capsys):
    manifest, checkpoints, musan_root = _make_inputs(tmp_path)
    any_manifest = tmp_path / "keyword_set.jsonl"
    any_manifest.write_text(
        json.dumps(
            {
                "audio_path": str((tmp_path / "negative.wav").resolve()),
                "keyword_labels": {"hey eva": 0},
            }
        )
        + "\n"
        + json.dumps(
            {
                "audio_path": str((tmp_path / "positive.wav").resolve()),
                "keyword_labels": {"hey eva": 1},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    argv = [
        "--manifest",
        str(any_manifest),
        "--musan-root",
        str(musan_root),
        "--output-root",
        str(tmp_path / "out"),
        "--model",
        f"candidate={checkpoints[0]}::icefall_zipformer_stage2_eps_softmin_v41",
        "--condition",
        "clean",
        "--override",
        "+keyword_eval=hey_eva_variants",
        "--dry-run",
    ]
    args = MODULE.build_parser().parse_args(argv)
    payload = MODULE.run_batch(args)
    assert payload["counts"]["planned"] == 1
    output = capsys.readouterr().out
    assert "+keyword_eval=hey_eva_variants" in output
    assert "eval_stage2_clips.py" in output


def test_fingerprint_changes_when_a_pronunciation_is_added(tmp_path):
    manifest, checkpoints, musan_root = _make_inputs(tmp_path)
    any_manifest = tmp_path / "keyword_set.jsonl"
    any_manifest.write_text(
        json.dumps(
            {
                "audio_path": str((tmp_path / "negative.wav").resolve()),
                "keyword_labels": {"hey eva": 0},
            }
        )
        + "\n"
        + json.dumps(
            {
                "audio_path": str((tmp_path / "positive.wav").resolve()),
                "keyword_labels": {"hey eva": 1},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    model = MODULE.ModelSpec(
        "candidate",
        checkpoints[0],
        "icefall_zipformer_stage2_eps_softmin_v41",
    )
    common = dict(
        models=[model],
        conditions=[MODULE.CONDITION_BY_NAME["clean"]],
        manifest=any_manifest,
        musan_root=musan_root,
        output_root=tmp_path / "out",
        device="cpu",
        batch_size=0,
        num_workers=0,
        seed=2025,
    )
    two_prons = ["+keyword_eval=hey_eva_variants"]
    one_pron = [
        "+keyword_eval=hey_eva_variants",
        'prep.keyword_eval.targets=[{text: "hey eva", pronunciations: ["HH EY1 IY1 V AH0"]}]',
    ]
    jobs_two, _ = MODULE.build_jobs(common_overrides=two_prons, **common)
    jobs_two_again, _ = MODULE.build_jobs(common_overrides=two_prons, **common)
    jobs_one, _ = MODULE.build_jobs(
        common_overrides=one_pron,
        **{**common, "output_root": tmp_path / "out_one"},
    )
    assert jobs_two[0].fingerprint == jobs_two_again[0].fingerprint
    assert jobs_two[0].fingerprint != jobs_one[0].fingerprint


def test_precheck_override_order_matches_executed_command(tmp_path):
    from omegaconf import OmegaConf

    manifest, checkpoints, musan_root = _make_inputs(tmp_path)
    lamp_only = tmp_path / "lamp.jsonl"
    lamp_only.write_text(
        json.dumps(
            {
                "audio_path": str((tmp_path / "negative.wav").resolve()),
                "keyword_labels": {"ok lamp": 0},
            }
        )
        + "\n"
        + json.dumps(
            {
                "audio_path": str((tmp_path / "positive.wav").resolve()),
                "keyword_labels": {"ok lamp": 1},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    model = MODULE.ModelSpec(
        "candidate",
        checkpoints[0],
        "icefall_zipformer_stage2_eps_softmin_v41",
        overrides=(
            'prep.keyword_eval.targets=[{text: "ok lamp", pronunciations: ["OW1 K EY1 L AE1 M P"]}]',
        ),
    )
    common_overrides = ["+keyword_eval=hey_eva_variants"]
    command = MODULE.build_eval_command(
        model=model,
        condition=MODULE.CONDITION_BY_NAME["clean"],
        manifest=lamp_only,
        musan_root=musan_root,
        output_dir=tmp_path / "out",
        device="cpu",
        batch_size=0,
        num_workers=0,
        seed=2025,
        common_overrides=common_overrides,
    )
    executed = OmegaConf.to_container(
        MODULE.compose_config(model.experiment, overrides=command[3:]).prep.keyword_eval,
        resolve=True,
    )
    precheck = MODULE._compose_model_config(model, common_overrides)["prep"][
        "keyword_eval"
    ]
    assert executed["targets"] == precheck["targets"]
    assert executed["targets"][0]["text"] == "ok lamp"
    rows, keyword_set = MODULE._load_matrix_manifest(
        lamp_only,
        MODULE._compose_model_config(model, common_overrides),
    )
    assert keyword_set.texts == ("ok lamp",)
    assert {int(row["label"]) for row in rows} == {0, 1}
