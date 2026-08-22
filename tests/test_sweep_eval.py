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
    csv_rows = (output_dir / "roc_curve.csv").read_text(encoding="utf-8").strip().splitlines()
    assert csv_rows[0] == "threshold,tpr,fpr"
    assert csv_rows[1] == "inf,0.0,0.0"
    assert report["plots"]["roc_curve_csv"] == str(
        (output_dir / "roc_curve.csv").resolve()
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


@pytest.mark.parametrize(
    ("plot_min_recall", "plot_max_fpr"),
    [
        (0.8, None),
        (None, 0.05),
    ],
)
def test_target_eval_report_forwards_plot_constraint_into_summary(
    tmp_path: Path,
    monkeypatch,
    plot_min_recall: float | None,
    plot_max_fpr: float | None,
):
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
    captured: dict = {}
    constraint_kind = (
        "min_recall" if plot_min_recall is not None else "max_fpr"
    )
    expected_plots = {
        "status": "generated",
        "score_field": "qbyt_score",
        "constraint": {
            "kind": constraint_kind,
            "metric": "recall" if plot_min_recall is not None else "fpr",
            "requested": (
                plot_min_recall
                if plot_min_recall is not None
                else plot_max_fpr
            ),
            "exact": False,
            "actual_recall": 0.9,
            "actual_fpr": 0.01,
            "guide_recall": (
                plot_min_recall if plot_min_recall is not None else 0.9
            ),
            "guide_fpr": (
                0.01 if plot_min_recall is not None else plot_max_fpr
            ),
            "threshold": 0.7,
        },
    }

    def fake_write_detection_plots(
        plot_records,
        *,
        output_dir,
        threshold,
        metrics,
        dpi,
        min_recall,
        max_fpr,
    ):
        captured.update(
            {
                "records": plot_records,
                "output_dir": output_dir,
                "threshold": threshold,
                "metrics": metrics,
                "dpi": dpi,
                "min_recall": min_recall,
                "max_fpr": max_fpr,
            }
        )
        return expected_plots

    monkeypatch.setattr(
        "dma_kws.stage2.sweep_eval.write_detection_plots",
        fake_write_detection_plots,
    )
    output_dir = tmp_path / "target_eval"

    report = write_target_eval_report(
        records,
        output_dir=output_dir,
        manifest_path=tmp_path / "eval.csv",
        checkpoint_path=tmp_path / "stage2_adapted.pt",
        threshold=0.5,
        plot_dpi=72,
        plot_min_recall=plot_min_recall,
        plot_max_fpr=plot_max_fpr,
    )

    assert captured["records"] == records
    assert captured["output_dir"] == output_dir
    assert captured["threshold"] == pytest.approx(0.5)
    assert captured["dpi"] == 72
    assert captured["min_recall"] == plot_min_recall
    assert captured["max_fpr"] == plot_max_fpr
    assert captured["metrics"] == report["metrics"]
    assert report["plots"] == expected_plots
    saved_summary = json.loads((output_dir / "summary.json").read_text())
    assert saved_summary["plots"] == expected_plots


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
    assert not (output_dir / "roc_curve.csv").exists()
