"""Persist final target-keyword evaluation artifacts for one sweep trial."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from dma_kws.inference.detection_plots import (
    DEFAULT_PLOT_DPI,
    write_detection_plots,
)
from dma_kws.inference.metrics import summarize_labeled_results
from dma_kws.training.run_context import checkpoint_run_context


TARGET_EVAL_DIRNAME = "target_eval"


def resolve_target_eval_output_dir(
    checkpoint_path: str | Path,
    checkpoint: Mapping[str, Any],
) -> Path:
    """Place sweep eval artifacts below the checkpoint's versioned log directory."""

    context = checkpoint_run_context(checkpoint)
    run_dir = str((context or {}).get("run_dir", "")).strip()
    root = Path(run_dir) if run_dir else Path(checkpoint_path).resolve().parent
    return root / TARGET_EVAL_DIRNAME


def write_target_eval_report(
    records: list[dict[str, Any]],
    *,
    output_dir: str | Path,
    manifest_path: str | Path,
    checkpoint_path: str | Path,
    threshold: float,
    plot_curves: bool = True,
    plot_dpi: int = DEFAULT_PLOT_DPI,
    plot_min_recall: float | None = None,
    plot_max_fpr: float | None = None,
) -> dict[str, Any]:
    """Write predictions, metrics, and optional ROC/DET plots for a sweep trial."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    prediction_path = destination / "predictions.jsonl"
    with prediction_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(
                json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n"
            )

    metric_rows = [
        {
            "label": int(record["label"]),
            "best_qbyt_score": float(record["qbyt_score"]),
        }
        for record in records
    ]
    metrics = summarize_labeled_results(metric_rows, threshold=threshold)
    if plot_curves:
        plots = write_detection_plots(
            records,
            output_dir=destination,
            threshold=threshold,
            metrics=metrics,
            dpi=plot_dpi,
            min_recall=plot_min_recall,
            max_fpr=plot_max_fpr,
        )
    else:
        plots = {
            "status": "disabled",
            "score_field": "qbyt_score",
        }

    summary = {
        "manifest": str(Path(manifest_path).resolve()),
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "num_samples": len(records),
        "threshold": float(threshold),
        "metrics": metrics,
        "plots": plots,
        "predictions": str(prediction_path.resolve()),
    }
    summary_path = destination / "summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(
            summary,
            handle,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        handle.write("\n")
    return {**summary, "summary": str(summary_path.resolve())}
