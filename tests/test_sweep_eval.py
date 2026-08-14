import json
from pathlib import Path

import pytest

from dma_kws.stage2.sweep_eval import (
    resolve_target_eval_output_dir,
    write_target_eval_report,
)
from dma_kws.training.run_context import RUN_CONTEXT_KEY


def test_target_eval_output_uses_checkpoint_versioned_log_directory(tmp_path: Path):
    run_dir = tmp_path / "real" / "logs" / "adapt_hey_eva_real" / "version_3"
    checkpoint = {RUN_CONTEXT_KEY: {"run_dir": str(run_dir)}}

    output_dir = resolve_target_eval_output_dir(
        tmp_path / "sweep" / "trial_4" / "stage2_adapted.pt",
        checkpoint,
    )

    assert output_dir == run_dir / "target_eval"


def test_target_eval_output_falls_back_to_checkpoint_directory(tmp_path: Path):
    checkpoint_path = tmp_path / "sweep" / "trial_4" / "stage2_adapted.pt"

    output_dir = resolve_target_eval_output_dir(checkpoint_path, {})

    assert output_dir == checkpoint_path.resolve().parent / "target_eval"


def test_target_eval_report_writes_scores_summary_and_curves(tmp_path: Path):
    pytest.importorskip("matplotlib")
    records = [
        {
            "sample_id": 0,
            "audio_path": "positive.wav",
            "keyword": "hey eva",
            "label": 1,
            "qbyt_score": 0.9,
        },
        {
            "sample_id": 1,
            "audio_path": "negative.wav",
            "keyword": "hey eva",
            "label": 0,
            "qbyt_score": 0.1,
        },
    ]
    output_dir = tmp_path / "logs" / "version_0" / "target_eval"

    report = write_target_eval_report(
        records,
        output_dir=output_dir,
        manifest_path=tmp_path / "manifests" / "real_eval.csv",
        checkpoint_path=tmp_path / "stage2_adapted.pt",
        threshold=0.5,
        plot_dpi=72,
    )

    assert report["metrics"]["auc"] == pytest.approx(1.0)
    assert report["plots"]["status"] == "generated"
    assert (output_dir / "roc_curve.png").read_bytes().startswith(
        b"\x89PNG\r\n\x1a\n"
    )
    assert (output_dir / "det_curve.png").read_bytes().startswith(
        b"\x89PNG\r\n\x1a\n"
    )
    predictions = [
        json.loads(line)
        for line in (output_dir / "predictions.jsonl").read_text().splitlines()
    ]
    assert predictions == records
    saved_summary = json.loads((output_dir / "summary.json").read_text())
    assert saved_summary["plots"] == report["plots"]
    assert saved_summary["predictions"] == str(
        (output_dir / "predictions.jsonl").resolve()
    )


def test_target_eval_report_can_disable_curves(tmp_path: Path):
    records = [
        {
            "sample_id": 0,
            "audio_path": "positive.wav",
            "keyword": "hey eva",
            "label": 1,
            "qbyt_score": 0.9,
        },
        {
            "sample_id": 1,
            "audio_path": "negative.wav",
            "keyword": "hey eva",
            "label": 0,
            "qbyt_score": 0.1,
        },
    ]
    output_dir = tmp_path / "target_eval"

    report = write_target_eval_report(
        records,
        output_dir=output_dir,
        manifest_path=tmp_path / "eval.csv",
        checkpoint_path=tmp_path / "stage2_adapted.pt",
        threshold=0.5,
        plot_curves=False,
    )

    assert report["plots"] == {
        "status": "disabled",
        "score_field": "qbyt_score",
    }
    assert not (output_dir / "roc_curve.png").exists()
    assert not (output_dir / "det_curve.png").exists()
