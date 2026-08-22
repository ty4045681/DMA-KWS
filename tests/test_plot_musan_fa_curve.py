from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.plot_musan_fa_curve import plot_musan_fa_curve


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


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
            "skipped": True,
            "manifest_meta": {"subset": "speech", "start_sec": 1.0, "end_sec": 4.0},
        },
    ]


def test_plot_musan_fa_curve_reads_eval_directory(tmp_path):
    pytest.importorskip("matplotlib")
    eval_dir = tmp_path / "musan_test"
    _write_jsonl(eval_dir / "results.jsonl", _musan_rows())
    (eval_dir / "summary.json").write_text(
        json.dumps(
            {
                "total_hours": 2.0,
                "metrics": {"threshold": 0.5},
            }
        ),
        encoding="utf-8",
    )

    plot_summary = plot_musan_fa_curve(eval_dir, dpi=72)

    plot_path = eval_dir / "fa_per_hour_curve.png"
    csv_path = eval_dir / "fa_per_hour_curve.csv"
    assert plot_summary["status"] == "generated"
    assert plot_summary["num_samples"] == 2
    assert plot_summary["total_hours"] == pytest.approx(2.0)
    assert plot_summary["deployment_threshold"] == pytest.approx(0.5)
    assert plot_summary["deployment_false_accepts"] == 1
    assert plot_summary["deployment_fa_per_hour"] == pytest.approx(0.5)
    assert plot_summary["fa_per_hour_curve"] == str(plot_path.resolve())
    assert plot_summary["fa_per_hour_curve_csv"] == str(csv_path.resolve())
    assert plot_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    csv_rows = csv_path.read_text(encoding="utf-8").strip().splitlines()
    assert csv_rows[0] == (
        "threshold,false_accepts,fa_per_hour,fa_per_24_hours,fa_per_1000_hours"
    )
    thresholds = [row.split(",", 1)[0] for row in csv_rows[1:]]
    assert "0.0" in thresholds
    assert "0.4" in thresholds
    assert "0.9" in thresholds
    assert "1.0" in thresholds


def test_plot_musan_fa_curve_requires_hours_without_summary(tmp_path):
    results = tmp_path / "results.jsonl"
    _write_jsonl(results, _musan_rows())

    with pytest.raises(SystemExit, match="total audio duration"):
        plot_musan_fa_curve(results)
