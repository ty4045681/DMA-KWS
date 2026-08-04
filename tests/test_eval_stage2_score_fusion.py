from __future__ import annotations

import json
from pathlib import Path

import pytest

from dma_kws.inference.score_calibration import sigmoid
from scripts.eval_stage2_score_fusion import run_analysis


def _record(
    tmp_path: Path,
    name: str,
    *,
    label: int,
    qbyt_logit: float,
    completion_logit: float,
    voice_id: str,
    skipped: bool = False,
) -> dict:
    return {
        "audio_path": str(tmp_path / f"{name}.wav"),
        "keyword": "Hey Google",
        "label": label,
        "qbyt_logit": None if skipped else qbyt_logit,
        "qbyt_score": 0.0 if skipped else float(sigmoid([qbyt_logit])[0]),
        "completion_logit": None if skipped else completion_logit,
        "completion_score": (
            None if skipped else float(sigmoid([completion_logit])[0])
        ),
        "skipped": skipped,
        "manifest_meta": {
            "tts_provider": "google",
            "voice_id": voice_id,
        },
    }


def _provenance(*, checkpoint_sha256: str = "a" * 64, completion_weight: float = 0.5) -> dict:
    return {
        "schema_version": 1,
        "checkpoint": {
            "path": "/models/stage2.pt",
            "size_bytes": 123,
            "sha256": checkpoint_sha256,
        },
        "qbyt_readout_mode": "gru_last",
        "stream": "backend=zipformer chunking=off",
        "audio_padding_ms": {"left": 160, "right": 160},
        "fbank": {"num_mel_bins": 80, "dither": 0.0},
        "tokenizer": {
            "path": "/data/lang_char.txt",
            "size_bytes": 456,
            "sha256": "b" * 64,
            "split_with_space": " ",
        },
        "sequence_objective": {
            "target_mode": "ordered_contiguous_prefix",
            "progress_weight": 0.5,
            "completion_weight": completion_weight,
            "normalization": "sample",
        },
    }


def _write_jsonl(
    path: Path,
    records: list[dict],
    *,
    provenance: dict | None = None,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    summary_path = path.parent / "summary.json"
    if provenance is not None or not summary_path.exists():
        summary_path.write_text(
            json.dumps({"provenance": provenance or _provenance()}),
            encoding="utf-8",
        )
    return path


def _balanced_rows(tmp_path: Path, prefix: str, voice_id: str) -> list[dict]:
    rows = []
    for index, label in enumerate((0, 1, 0, 1, 0, 1)):
        direction = 1.0 if label else -1.0
        rows.append(
            _record(
                tmp_path,
                f"{prefix}_{index}",
                label=label,
                qbyt_logit=direction * (0.2 + index * 0.05) + 3.0,
                completion_logit=direction * (0.8 + index * 0.05),
                voice_id=voice_id,
            )
        )
    return rows


def test_run_analysis_fits_only_calibration_and_writes_json_outputs(tmp_path) -> None:
    calibration_rows = _balanced_rows(tmp_path, "cal", "cal-voice")
    calibration_rows.append(
        _record(
            tmp_path,
            "cal_skipped",
            label=1,
            qbyt_logit=0.0,
            completion_logit=0.0,
            voice_id="cal-voice",
            skipped=True,
        )
    )
    evaluation_rows = _balanced_rows(tmp_path, "eval", "eval-voice")
    evaluation_rows.append(
        _record(
            tmp_path,
            "eval_skipped",
            label=0,
            qbyt_logit=0.0,
            completion_logit=0.0,
            voice_id="eval-voice",
            skipped=True,
        )
    )
    calibration_path = _write_jsonl(tmp_path / "calibration.jsonl", calibration_rows)
    evaluation_path = _write_jsonl(tmp_path / "evaluation.jsonl", evaluation_rows)

    summary = run_analysis(
        calibration_results=calibration_path,
        evaluation_results=evaluation_path,
        output_dir=tmp_path / "analysis",
        l2=0.05,
        group_fields=("manifest_meta.tts_provider", "manifest_meta.voice_id"),
    )

    assert summary["calibration_population"]["num_samples"] == 6
    assert summary["calibration_population"]["num_skipped_excluded"] == 1
    assert summary["evaluation_population"]["num_samples"] == 6
    assert summary["evaluation_population"]["num_skipped_excluded"] == 1
    assert set(summary["evaluation_metrics"]) == {
        "utterance_raw",
        "completion_raw",
        "utterance_platt",
        "completion_platt",
        "utterance_completion_fusion",
    }
    # All three models are fitted on calibration only. Evaluation labels are
    # used below solely by the diagnostics report.
    assert {
        model["fit"]["num_samples"] for model in summary["models"].values()
    } == {6}

    outputs = {name: Path(path) for name, path in summary["outputs"].items()}
    assert all(path.is_file() for path in outputs.values())
    saved_summary = json.loads(outputs["summary"].read_text(encoding="utf-8"))
    assert saved_summary == summary
    calibrators = json.loads(outputs["calibrators"].read_text(encoding="utf-8"))
    assert calibrators["fit_source"] == str(calibration_path.resolve())

    enriched = [
        json.loads(line)
        for line in outputs["evaluation_results"].read_text(encoding="utf-8").splitlines()
    ]
    assert len(enriched) == len(evaluation_rows)
    assert enriched[0]["fused_logit"] is not None
    assert 0.0 <= enriched[0]["fused_score"] <= 1.0
    assert enriched[-1]["fused_logit"] is None
    assert enriched[-1]["calibrated_qbyt_score"] is None


def test_run_analysis_rejects_the_same_results_path(tmp_path) -> None:
    results = _write_jsonl(
        tmp_path / "results.jsonl",
        _balanced_rows(tmp_path, "same", "voice"),
    )

    with pytest.raises(ValueError, match="must be different files"):
        run_analysis(
            calibration_results=results,
            evaluation_results=results,
            output_dir=tmp_path / "out",
            group_fields=("manifest_meta.tts_provider", "manifest_meta.voice_id"),
        )


def test_run_analysis_rejects_exact_sample_overlap_across_files(tmp_path) -> None:
    calibration_rows = _balanced_rows(tmp_path, "cal", "cal-voice")
    evaluation_rows = _balanced_rows(tmp_path, "eval", "eval-voice")
    evaluation_rows[0]["audio_path"] = calibration_rows[0]["audio_path"]
    evaluation_rows[0]["keyword"] = "Different Query"
    calibration_path = _write_jsonl(tmp_path / "cal.jsonl", calibration_rows)
    evaluation_path = _write_jsonl(tmp_path / "eval.jsonl", evaluation_rows)

    with pytest.raises(ValueError, match="sample leakage"):
        run_analysis(
            calibration_results=calibration_path,
            evaluation_results=evaluation_path,
            output_dir=tmp_path / "out",
            group_fields=("manifest_meta.tts_provider", "manifest_meta.voice_id"),
        )


def test_run_analysis_requires_group_check_by_default(tmp_path) -> None:
    calibration_path = _write_jsonl(
        tmp_path / "cal.jsonl",
        _balanced_rows(tmp_path, "cal", "cal-voice"),
    )
    evaluation_path = _write_jsonl(
        tmp_path / "eval.jsonl",
        _balanced_rows(tmp_path, "eval", "eval-voice"),
    )

    with pytest.raises(ValueError, match="at least one --group-field"):
        run_analysis(
            calibration_results=calibration_path,
            evaluation_results=evaluation_path,
            output_dir=tmp_path / "out",
        )


def test_run_analysis_rejects_score_provenance_mismatch(tmp_path) -> None:
    calibration_path = _write_jsonl(
        tmp_path / "cal" / "results.jsonl",
        _balanced_rows(tmp_path, "cal", "cal-voice"),
        provenance=_provenance(checkpoint_sha256="a" * 64),
    )
    evaluation_path = _write_jsonl(
        tmp_path / "eval" / "results.jsonl",
        _balanced_rows(tmp_path, "eval", "eval-voice"),
        provenance=_provenance(checkpoint_sha256="c" * 64),
    )

    with pytest.raises(ValueError, match="provenance mismatch"):
        run_analysis(
            calibration_results=calibration_path,
            evaluation_results=evaluation_path,
            output_dir=tmp_path / "out",
            group_fields=("manifest_meta.tts_provider", "manifest_meta.voice_id"),
        )


def test_run_analysis_rejects_unsafe_completion_objective(tmp_path) -> None:
    provenance = _provenance(completion_weight=0.0)
    provenance["sequence_objective"]["target_mode"] = "membership"
    calibration_path = _write_jsonl(
        tmp_path / "cal" / "results.jsonl",
        _balanced_rows(tmp_path, "cal", "cal-voice"),
        provenance=provenance,
    )
    evaluation_path = _write_jsonl(
        tmp_path / "eval" / "results.jsonl",
        _balanced_rows(tmp_path, "eval", "eval-voice"),
        provenance=provenance,
    )

    with pytest.raises(ValueError, match="Completion fusion is unsafe"):
        run_analysis(
            calibration_results=calibration_path,
            evaluation_results=evaluation_path,
            output_dir=tmp_path / "out",
            group_fields=("manifest_meta.tts_provider", "manifest_meta.voice_id"),
        )


def test_run_analysis_validates_completion_probability_pair(tmp_path) -> None:
    calibration_rows = _balanced_rows(tmp_path, "cal", "cal-voice")
    evaluation_rows = _balanced_rows(tmp_path, "eval", "eval-voice")
    evaluation_rows[0]["completion_score"] = 0.99
    calibration_path = _write_jsonl(tmp_path / "cal.jsonl", calibration_rows)
    evaluation_path = _write_jsonl(tmp_path / "eval.jsonl", evaluation_rows)

    with pytest.raises(ValueError, match="completion_score is inconsistent"):
        run_analysis(
            calibration_results=calibration_path,
            evaluation_results=evaluation_path,
            output_dir=tmp_path / "out",
            group_fields=("manifest_meta.tts_provider", "manifest_meta.voice_id"),
        )


def test_run_analysis_rejects_composite_voice_group_overlap(tmp_path) -> None:
    calibration_path = _write_jsonl(
        tmp_path / "cal.jsonl",
        _balanced_rows(tmp_path, "cal", "shared-voice"),
    )
    evaluation_path = _write_jsonl(
        tmp_path / "eval.jsonl",
        _balanced_rows(tmp_path, "eval", "shared-voice"),
    )

    with pytest.raises(ValueError, match="group leakage"):
        run_analysis(
            calibration_results=calibration_path,
            evaluation_results=evaluation_path,
            output_dir=tmp_path / "out",
            group_fields=("manifest_meta.tts_provider", "manifest_meta.voice_id"),
        )


def test_run_analysis_requires_raw_logits_on_non_skipped_rows(tmp_path) -> None:
    calibration_rows = _balanced_rows(tmp_path, "cal", "cal-voice")
    evaluation_rows = _balanced_rows(tmp_path, "eval", "eval-voice")
    del evaluation_rows[0]["completion_logit"]
    calibration_path = _write_jsonl(tmp_path / "cal.jsonl", calibration_rows)
    evaluation_path = _write_jsonl(tmp_path / "eval.jsonl", evaluation_rows)

    with pytest.raises(ValueError, match="completion_logit"):
        run_analysis(
            calibration_results=calibration_path,
            evaluation_results=evaluation_path,
            output_dir=tmp_path / "out",
            group_fields=("manifest_meta.tts_provider", "manifest_meta.voice_id"),
        )
