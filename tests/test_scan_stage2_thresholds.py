from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.scan_stage2_thresholds import (
    build_thresholds,
    load_scan_input,
    run_scan,
    scan_thresholds,
    select_operating_point,
    write_single_class_plot,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _clip_rows() -> list[dict]:
    return [
        {"audio_path": "p1.wav", "keyword": "hey eva", "label": 1, "qbyt_score": 0.9},
        {"audio_path": "n1.wav", "keyword": "hey eva", "label": 0, "qbyt_score": 0.8},
        {"audio_path": "p2.wav", "keyword": "hey eva", "label": 1, "qbyt_score": 0.7},
        {"audio_path": "n2.wav", "keyword": "hey eva", "label": 0, "qbyt_score": 0.2},
    ]


def _musan_rows() -> list[dict]:
    return [
        {
            "audio_path": "speech/a.wav",
            "keyword": "hey eva",
            "label": 0,
            "qbyt_score": 0.9,
            "manifest_meta": {"subset": "speech", "start_sec": 0.0, "end_sec": 3.0},
        },
        {
            "audio_path": "noise/b.wav",
            "keyword": "hey eva",
            "label": 0,
            "qbyt_score": 0.4,
            "manifest_meta": {"subset": "noise", "start_sec": 0.0, "end_sec": 3.0},
        },
        {
            "audio_path": "speech/a.wav",
            "keyword": "hey eva",
            "label": 0,
            "qbyt_score": 0.8,
            "manifest_meta": {"subset": "speech", "start_sec": 1.0, "end_sec": 4.0},
        },
        {
            "audio_path": "noise/b.wav",
            "keyword": "hey eva",
            "label": 0,
            "qbyt_score": 0.2,
            "manifest_meta": {"subset": "noise", "start_sec": 1.0, "end_sec": 4.0},
        },
    ]


def test_clip_scan_matches_confusion_counts_and_selects_under_fpr(tmp_path):
    results = tmp_path / "clips" / "results.jsonl"
    _write_jsonl(results, _clip_rows())

    scan_input = load_scan_input(results)
    thresholds = build_thresholds(scan_input.scores)
    arrays, workers = scan_thresholds(scan_input, thresholds, workers=2)

    assert scan_input.mode == "clips"
    assert workers == 2
    index = list(arrays["threshold"]).index(0.9)
    assert arrays["tp"][index] == 1
    assert arrays["fp"][index] == 0
    assert arrays["fn"][index] == 1
    assert arrays["tn"][index] == 2
    assert arrays["recall"][index] == pytest.approx(0.5)
    assert arrays["fpr"][index] == pytest.approx(0.0)

    selection = select_operating_point(arrays, mode="clips", max_fpr=0.0)
    assert selection is not None
    assert selection["found"] is True
    assert selection["operating_point"]["threshold"] == pytest.approx(0.9)
    assert selection["operating_point"]["recall"] == pytest.approx(0.5)


def test_parallel_and_single_thread_scans_are_identical(tmp_path):
    results = tmp_path / "clips" / "results.jsonl"
    _write_jsonl(results, _clip_rows())
    scan_input = load_scan_input(results)
    thresholds = build_thresholds(scan_input.scores, step=0.05)

    serial, serial_workers = scan_thresholds(scan_input, thresholds, workers=1)
    parallel, parallel_workers = scan_thresholds(scan_input, thresholds, workers=4)

    assert serial_workers == 1
    assert parallel_workers == 4
    assert serial.keys() == parallel.keys()
    for key in serial:
        assert parallel[key] == pytest.approx(serial[key])


def test_threshold_scan_excludes_and_reports_skipped_rows(tmp_path):
    output_dir = tmp_path / "clips"
    results = output_dir / "results.jsonl"
    rows = _clip_rows() + [
        {
            "audio_path": "skipped-positive.wav",
            "keyword": "hey eva",
            "label": 1,
            "qbyt_score": 0.0,
            "skipped": True,
        }
    ]
    _write_jsonl(results, rows)

    scan_input = load_scan_input(results, mode="clips")

    assert scan_input.scores.size == 4
    assert scan_input.num_skipped == 1

    summary = run_scan(
        argparse.Namespace(
            results=results,
            mode="clips",
            summary=None,
            total_hours=None,
            workers=1,
            threshold_step=None,
            max_fpr=None,
            min_recall=None,
            max_fa_per_hour=None,
            no_subsets=False,
            out_csv=tmp_path / "curve.csv",
            out_summary=tmp_path / "scan.json",
        )
    )
    assert summary["num_samples"] == 4
    assert summary["num_input_rows"] == 5
    assert summary["num_skipped_excluded"] == 1


def test_musan_scan_uses_summary_hours_and_writes_subset_metrics(tmp_path):
    output_dir = tmp_path / "musan"
    results = output_dir / "results.jsonl"
    _write_jsonl(results, _musan_rows())
    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                "musan_root": "/data/musan",
                "total_hours": 2.0,
                "subsets": {
                    "speech": {"total_hours": 1.0},
                    "noise": {"total_hours": 1.0},
                },
            }
        ),
        encoding="utf-8",
    )

    scan_input = load_scan_input(output_dir)
    thresholds = build_thresholds(scan_input.scores)
    arrays, _ = scan_thresholds(scan_input, thresholds, workers=3)

    assert scan_input.mode == "musan"
    assert scan_input.total_hours == pytest.approx(2.0)
    index = list(arrays["threshold"]).index(0.8)
    assert arrays["fp"][index] == 2
    assert arrays["fpr"][index] == pytest.approx(0.5)
    assert arrays["fa_per_hour"][index] == pytest.approx(1.0)
    assert arrays["subset_speech_fp"][index] == 2
    assert arrays["subset_speech_fa_per_hour"][index] == pytest.approx(2.0)
    assert arrays["subset_noise_fp"][index] == 0

    selection = select_operating_point(
        arrays,
        mode="musan",
        max_fa_per_hour=0.5,
    )
    assert selection is not None
    assert selection["found"] is True
    assert selection["operating_point"]["threshold"] == pytest.approx(0.9)
    assert selection["operating_point"]["fa_per_hour"] == pytest.approx(0.5)


def test_run_scan_writes_curve_and_summary(tmp_path):
    output_dir = tmp_path / "clips"
    results = output_dir / "results.jsonl"
    _write_jsonl(results, _clip_rows())
    out_csv = tmp_path / "artifacts" / "curve.csv"
    out_summary = tmp_path / "artifacts" / "scan.json"
    args = argparse.Namespace(
        results=results,
        mode="clips",
        summary=None,
        total_hours=None,
        workers=2,
        threshold_step=None,
        max_fpr=0.0,
        min_recall=None,
        max_fa_per_hour=None,
        no_subsets=False,
        out_csv=out_csv,
        out_summary=out_summary,
    )

    summary = run_scan(args)

    assert summary["mode"] == "clips"
    assert summary["selection"]["operating_point"]["threshold"] == pytest.approx(0.9)
    assert out_csv.is_file()
    assert out_summary.is_file()
    with out_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == summary["num_thresholds"]
    assert {"threshold", "recall", "fpr", "tp", "tn", "fp", "fn"} <= set(rows[0])


def test_positive_only_clip_scan_writes_recall_plot(tmp_path):
    pytest.importorskip("matplotlib")
    results = tmp_path / "clips" / "results.jsonl"
    _write_jsonl(
        results,
        [
            {"audio_path": "p1.wav", "keyword": "hey eva", "label": 1, "qbyt_score": 0.9},
            {"audio_path": "p2.wav", "keyword": "hey eva", "label": 1, "qbyt_score": 0.4},
        ],
    )

    scan_input = load_scan_input(tmp_path / "clips")
    thresholds = build_thresholds(scan_input.scores)
    arrays, _ = scan_thresholds(scan_input, thresholds, workers=1)

    assert scan_input.mode == "clips"
    assert int(np.sum(scan_input.labels == 1)) == 2
    assert int(np.sum(scan_input.labels == 0)) == 0
    index = list(arrays["threshold"]).index(0.9)
    assert arrays["tp"][index] == 1
    assert arrays["fn"][index] == 1
    assert arrays["recall"][index] == pytest.approx(0.5)
    assert arrays["fp"][index] == 0
    assert arrays["fpr"][index] == pytest.approx(0.0)

    out_plot = tmp_path / "recall.png"
    plot = write_single_class_plot(arrays, metric="recall", output_path=out_plot)
    assert plot["status"] == "generated"
    assert plot["metric"] == "recall"
    assert out_plot.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")

    summary = run_scan(
        argparse.Namespace(
            results=tmp_path / "clips",
            mode="auto",
            summary=None,
            total_hours=None,
            workers=1,
            threshold_step=None,
            max_fpr=None,
            min_recall=None,
            max_fa_per_hour=None,
            no_subsets=False,
            out_csv=tmp_path / "pos_curve.csv",
            out_summary=tmp_path / "pos_scan.json",
            out_plot=out_plot,
        )
    )
    assert summary["positives"] == 2
    assert summary["negatives"] == 0
    assert "auc" not in summary
    assert summary["plot"]["metric"] == "recall"
    assert summary["plot"]["path"] == str(out_plot.resolve())


def test_negative_only_clip_scan_writes_fpr_plot(tmp_path):
    pytest.importorskip("matplotlib")
    results = tmp_path / "clips" / "results.jsonl"
    _write_jsonl(
        results,
        [
            {"audio_path": "n1.wav", "keyword": "hey eva", "label": 0, "qbyt_score": 0.8},
            {"audio_path": "n2.wav", "keyword": "hey eva", "label": 0, "qbyt_score": 0.2},
        ],
    )

    scan_input = load_scan_input(tmp_path / "clips")
    thresholds = build_thresholds(scan_input.scores)
    arrays, _ = scan_thresholds(scan_input, thresholds, workers=1)

    assert scan_input.mode == "clips"
    index = list(arrays["threshold"]).index(0.8)
    assert arrays["fp"][index] == 1
    assert arrays["tn"][index] == 1
    assert arrays["fpr"][index] == pytest.approx(0.5)
    assert arrays["tp"][index] == 0
    assert arrays["recall"][index] == pytest.approx(0.0)

    out_plot = tmp_path / "fpr.png"
    summary = run_scan(
        argparse.Namespace(
            results=tmp_path / "clips",
            mode="clips",
            summary=None,
            total_hours=None,
            workers=1,
            threshold_step=None,
            max_fpr=0.0,
            min_recall=None,
            max_fa_per_hour=None,
            no_subsets=False,
            out_csv=tmp_path / "neg_curve.csv",
            out_summary=tmp_path / "neg_scan.json",
            out_plot=out_plot,
        )
    )
    assert summary["positives"] == 0
    assert summary["negatives"] == 2
    assert summary["plot"]["metric"] == "fpr"
    assert out_plot.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert summary["selection"]["found"] is True
    assert summary["selection"]["operating_point"]["fpr"] == pytest.approx(0.0)


def test_musan_scan_requires_total_hours(tmp_path):
    results = tmp_path / "results.jsonl"
    _write_jsonl(results, _musan_rows())

    with pytest.raises(SystemExit, match="total audio duration"):
        load_scan_input(results, mode="musan")
