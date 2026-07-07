from pathlib import Path

import pytest

from dma_kws.inference.manifest import load_manifest
from dma_kws.inference.metrics import summarize_labeled_results
from scripts.eval_stage2_clips import _result_record as stage2_clip_result_record
from scripts.eval_two_stage_kws import _result_record as two_stage_result_record


FIXTURES = Path(__file__).resolve().parent / "fixtures"


def test_load_manifest_csv_smoke():
    rows = load_manifest(FIXTURES / "manifest_smoke.csv")

    assert len(rows) == 3
    assert rows[0]["keyword"] == "hello world"
    assert rows[0]["label"] == 1
    assert Path(rows[0]["audio_path"]).name == "audio_a.wav"


def test_load_manifest_requires_audio_and_keyword(tmp_path):
    manifest = tmp_path / "bad.csv"
    manifest.write_text("audio_path,keyword\n,hello\n", encoding="utf-8")

    with pytest.raises(ValueError, match="missing required columns"):
        load_manifest(manifest)


def test_summarize_labeled_results_metrics():
    results = [
        {"label": 1, "best_qbyt_score": 0.9, "detected": True},
        {"label": 0, "best_qbyt_score": 0.1, "detected": False},
        {"label": 1, "best_qbyt_score": 0.4, "detected": False},
    ]

    summary = summarize_labeled_results(results, threshold=0.5)

    assert summary["num_samples"] == 3.0
    assert summary["tp"] == 1.0
    assert summary["tn"] == 1.0
    assert summary["fp"] == 0.0
    assert summary["fn"] == 1.0
    assert summary["accuracy"] == pytest.approx(2 / 3)
    assert summary["precision"] == pytest.approx(1.0)
    assert summary["recall"] == pytest.approx(0.5)
    assert 0.0 <= summary["auc"] <= 1.0
    assert 0.0 <= summary["eer"] <= 1.0


def test_summarize_labeled_results_empty_without_labels():
    results = [{"best_qbyt_score": 0.5, "detected": True}]

    assert summarize_labeled_results(results, threshold=0.5) == {}


def test_two_stage_result_record_includes_manifest_meta_for_extra_columns():
    manifest_row = {
        "audio_path": "/tmp/audio.wav",
        "keyword": "hello",
        "label": "1",
        "speaker_id": "spk-001",
        "source_split": "dev",
    }
    pipeline_result = {
        "detected": True,
        "best_qbyt_score": 0.91,
        "threshold": 0.5,
        "stage1_candidates": [],
        "stage2_scores": [],
    }

    record = two_stage_result_record(manifest_row, pipeline_result)

    assert record["audio_path"] == "/tmp/audio.wav"
    assert record["keyword"] == "hello"
    assert record["label"] == 1
    assert record["manifest_meta"] == {
        "speaker_id": "spk-001",
        "source_split": "dev",
    }


def test_stage2_clip_result_record_includes_manifest_meta_for_extra_columns():
    manifest_row = {
        "audio_path": "/tmp/audio.wav",
        "keyword": "hello",
        "label": "0",
        "speaker_id": "spk-002",
        "utterance_id": "utt-77",
    }
    runner_result = {
        "qbyt_score": 0.12,
        "detected": False,
        "threshold": 0.5,
        "skipped": False,
    }

    record = stage2_clip_result_record(manifest_row, runner_result)

    assert record["audio_path"] == "/tmp/audio.wav"
    assert record["keyword"] == "hello"
    assert record["label"] == 0
    assert record["manifest_meta"] == {
        "speaker_id": "spk-002",
        "utterance_id": "utt-77",
    }
