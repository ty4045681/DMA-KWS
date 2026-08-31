from __future__ import annotations

import csv
import json

import pytest

from dma_kws.inference.score_calibration import PositiveAffineCalibrator
from scripts.fit_stage2_calibration import build_parser, load_labeled_scores, run


def _write_jsonl(path, rows) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_supervised_jsonl_fit_writes_loadable_versioned_artifact(tmp_path) -> None:
    source = tmp_path / "held_out.jsonl"
    destination = tmp_path / "artifacts" / "calibration.json"
    _write_jsonl(
        source,
        [
            {"raw_logit": -2.0, "label": 0},
            {"raw_logit": -1.0, "label": 0},
            {"raw_logit": 1.0, "label": 1},
            {"raw_logit": 2.0, "label": 1},
        ],
    )
    args = build_parser().parse_args([str(source), "--output", str(destination)])

    fitted = run(args)
    loaded = PositiveAffineCalibrator.load_json(destination)

    assert loaded == fitted
    assert loaded.score_name == "raw_logit"
    assert loaded.slope > 0.0
    assert loaded.fit is not None
    assert loaded.fit.num_samples == 4
    assert loaded.fit.num_positive == 2


def test_supervised_csv_auto_detects_qbyt_raw_logit(tmp_path) -> None:
    source = tmp_path / "held_out.csv"
    with source.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["qbyt_raw_logit", "label"])
        writer.writeheader()
        writer.writerows(
            [
                {"qbyt_raw_logit": -1.0, "label": 0},
                {"qbyt_raw_logit": -0.5, "label": 0},
                {"qbyt_raw_logit": 0.5, "label": 1},
                {"qbyt_raw_logit": 1.0, "label": 1},
            ]
        )

    scores, labels, field = load_labeled_scores(source)

    assert field == "qbyt_raw_logit"
    assert scores.tolist() == [-1.0, -0.5, 0.5, 1.0]
    assert labels.tolist() == [0, 0, 1, 1]


def test_operating_threshold_mode_needs_no_input_and_maps_to_half(tmp_path) -> None:
    destination = tmp_path / "calibration.json"
    args = build_parser().parse_args(
        [
            "--output",
            str(destination),
            "--operating-threshold",
            "0.95",
            "--slope",
            "4.0",
        ]
    )

    calibrator = run(args)

    assert calibrator.fit is None
    assert calibrator.slope == pytest.approx(4.0)
    assert calibrator.predict_one(0.95) == pytest.approx(0.5)
    assert PositiveAffineCalibrator.load_json(destination) == calibrator


def test_supervised_fit_requires_both_classes(tmp_path) -> None:
    source = tmp_path / "held_out.jsonl"
    _write_jsonl(
        source,
        [
            {"raw_logit": -2.0, "label": 0},
            {"raw_logit": -1.0, "label": 0},
        ],
    )
    args = build_parser().parse_args(
        [str(source), "--output", str(tmp_path / "calibration.json")]
    )

    with pytest.raises(SystemExit, match="both positive and negative"):
        run(args)


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        ([{"raw_logit": 0.0}], "Missing label"),
        ([{"label": 0}], "consistent score field"),
        ([{"raw_logit": "nan", "label": 0}], "must be finite"),
        ([{"raw_logit": 0.0, "label": 2}], "expected 0 or 1"),
    ],
)
def test_loader_reports_invalid_rows(tmp_path, rows, message) -> None:
    source = tmp_path / "held_out.jsonl"
    _write_jsonl(source, rows)

    with pytest.raises(SystemExit, match=message):
        load_labeled_scores(source)


def test_modes_are_mutually_exclusive(tmp_path) -> None:
    source = tmp_path / "held_out.jsonl"
    _write_jsonl(
        source,
        [
            {"raw_logit": -1.0, "label": 0},
            {"raw_logit": 1.0, "label": 1},
        ],
    )
    args = build_parser().parse_args(
        [
            str(source),
            "--output",
            str(tmp_path / "calibration.json"),
            "--operating-threshold",
            "0.5",
        ]
    )

    with pytest.raises(SystemExit, match="Do not pass an input file"):
        run(args)
