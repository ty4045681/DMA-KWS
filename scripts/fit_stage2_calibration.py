#!/usr/bin/env python3
"""Fit or define a versioned monotone Stage-II score calibrator.

Supervised Platt fit from held-out scores::

    python scripts/fit_stage2_calibration.py held_out.jsonl \
      --output stage2_calibration.json

Define a deployment threshold that should become calibrated score 0.5::

    python scripts/fit_stage2_calibration.py \
      --operating-threshold 0.95 --output stage2_calibration.json

The input may be JSONL or CSV and must contain ``label`` plus either
``raw_logit`` or ``qbyt_raw_logit``.  This script never runs inference.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from dma_kws.inference.score_calibration import (
    PositiveAffineCalibrator,
    fit_positive_affine_calibrator,
)


_SCORE_FIELDS = ("raw_logit", "qbyt_raw_logit")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise SystemExit(
                        f"Invalid JSON at {path}:{line_number}: {exc.msg}"
                    ) from exc
                if not isinstance(value, dict):
                    raise SystemExit(
                        f"Expected a JSON object at {path}:{line_number}"
                    )
                rows.append(value)
    except OSError as exc:
        raise SystemExit(f"Failed to read {path}: {exc}") from exc
    return rows


def _read_csv(path: Path) -> list[dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise SystemExit(f"CSV has no header: {path}")
            return [dict(row) for row in reader]
    except OSError as exc:
        raise SystemExit(f"Failed to read {path}: {exc}") from exc


def _read_rows(path: Path) -> list[dict[str, Any]]:
    source = path.expanduser()
    if not source.is_file():
        raise SystemExit(f"Calibration input not found: {source}")
    suffix = source.suffix.lower()
    if suffix == ".jsonl":
        rows = _read_jsonl(source)
    elif suffix == ".csv":
        rows = _read_csv(source)
    else:
        raise SystemExit("Calibration input must have .jsonl or .csv extension")
    if not rows:
        raise SystemExit(f"No calibration rows found in {source}")
    return rows


def _has_value(row: Mapping[str, Any], field: str) -> bool:
    value = row.get(field)
    return value is not None and (not isinstance(value, str) or bool(value.strip()))


def _resolve_score_field(
    rows: Sequence[Mapping[str, Any]],
    requested: str | None,
) -> str:
    if requested is not None:
        if all(_has_value(row, requested) for row in rows):
            return requested
        raise SystemExit(f"Every calibration row must contain {requested!r}")
    for field in _SCORE_FIELDS:
        if all(_has_value(row, field) for row in rows):
            return field
    raise SystemExit(
        "Every calibration row must contain one consistent score field: "
        "raw_logit or qbyt_raw_logit"
    )


def _parse_float(value: object, *, field: str, row_number: int) -> float:
    if isinstance(value, bool):
        raise SystemExit(f"Invalid {field} at row {row_number}: {value!r}")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise SystemExit(
            f"Invalid {field} at row {row_number}: {value!r}"
        ) from exc
    if not np.isfinite(result):
        raise SystemExit(f"Invalid {field} at row {row_number}: must be finite")
    return result


def _parse_label(value: object, *, row_number: int) -> int:
    if isinstance(value, bool):
        raise SystemExit(f"Invalid label at row {row_number}: {value!r}")
    if value in (0, 1, "0", "1"):
        return int(value)
    raise SystemExit(f"Invalid label at row {row_number}: expected 0 or 1")


def load_labeled_scores(
    path: Path,
    *,
    score_field: str | None = None,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Load finite scores, binary labels, and the resolved score field."""

    rows = _read_rows(path)
    resolved_field = _resolve_score_field(rows, score_field)
    scores: list[float] = []
    labels: list[int] = []
    for row_number, row in enumerate(rows, start=1):
        scores.append(
            _parse_float(
                row[resolved_field],
                field=resolved_field,
                row_number=row_number,
            )
        )
        if not _has_value(row, "label"):
            raise SystemExit(f"Missing label at row {row_number}")
        labels.append(_parse_label(row["label"], row_number=row_number))
    return (
        np.asarray(scores, dtype=np.float64),
        np.asarray(labels, dtype=np.int64),
        resolved_field,
    )


def run(args: argparse.Namespace) -> PositiveAffineCalibrator:
    """Build the requested calibrator and write its JSON artifact."""

    if args.operating_threshold is not None:
        if args.input is not None:
            raise SystemExit(
                "Do not pass an input file with --operating-threshold; "
                "this mode does not fit labels"
            )
        calibrator = PositiveAffineCalibrator.from_operating_threshold(
            args.operating_threshold,
            slope=args.slope,
            score_name=args.score_field or "qbyt_raw_logit",
        )
    else:
        if args.input is None:
            raise SystemExit(
                "Supervised calibration requires a JSONL/CSV input, or pass "
                "--operating-threshold"
            )
        if args.slope != 1.0:
            raise SystemExit("--slope is only valid with --operating-threshold")
        scores, labels, score_field = load_labeled_scores(
            args.input,
            score_field=args.score_field,
        )
        try:
            calibrator = fit_positive_affine_calibrator(
                scores,
                labels,
                score_name=score_field,
                l2=args.l2,
                max_iterations=args.max_iterations,
            )
        except ValueError as exc:
            raise SystemExit(f"Cannot fit calibration: {exc}") from exc

    calibrator.save_json(args.output)
    return calibrator


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fit positive-affine Platt calibration from held-out scores, or "
            "map one raw operating threshold to calibrated score 0.5."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        help="Held-out .jsonl/.csv containing label and a raw score field",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination versioned calibrator JSON",
    )
    parser.add_argument(
        "--score-field",
        choices=_SCORE_FIELDS,
        help="Score column; auto-detected for supervised fitting",
    )
    parser.add_argument(
        "--operating-threshold",
        type=float,
        help="Skip fitting and map this raw threshold exactly to 0.5",
    )
    parser.add_argument(
        "--slope",
        type=float,
        default=1.0,
        help="Positive slope for --operating-threshold mode",
    )
    parser.add_argument(
        "--l2",
        type=float,
        default=1.0e-2,
        help="Slope L2 regularization for supervised fitting",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=100,
        help="Maximum projected-Newton iterations for supervised fitting",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        calibrator = run(args)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    print(
        json.dumps(
            {
                "output": str(args.output.expanduser().resolve()),
                "slope": calibrator.slope,
                "bias": calibrator.bias,
                "raw_operating_threshold": calibrator.raw_operating_threshold,
            },
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main(sys.argv[1:])
