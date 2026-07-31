from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import pytest

from scripts.scan_stage2_thresholds import (
    build_thresholds,
    load_scan_input,
    run_scan,
    scan_thresholds,
    select_operating_point,
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


def test_clip_scan_requires_both_classes(tmp_path):
    results = tmp_path / "results.jsonl"
    _write_jsonl(
        results,
        [{"label": 0, "qbyt_score": 0.2}, {"label": 0, "qbyt_score": 0.8}],
    )

    with pytest.raises(SystemExit, match="both positive and negative"):
        load_scan_input(results, mode="clips")


def test_musan_scan_requires_total_hours(tmp_path):
    results = tmp_path / "results.jsonl"
    _write_jsonl(results, _musan_rows())

    with pytest.raises(SystemExit, match="total audio duration"):
        load_scan_input(results, mode="musan")
