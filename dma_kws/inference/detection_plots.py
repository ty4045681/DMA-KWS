"""Reusable ROC and DET plot generation for binary detection scores."""

from __future__ import annotations

from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np


DEFAULT_PLOT_DPI = 160


def binary_roc_points(
    records: list[dict],
    *,
    score_field: str,
) -> dict[str, Any] | None:
    """Build exact ROC operating points, keeping tied scores together."""

    usable = [
        record
        for record in records
        if "label" in record
        and not bool(record.get("skipped", False))
        and record.get(score_field) is not None
    ]
    if not usable:
        return None

    labels = np.asarray([int(record["label"]) for record in usable], dtype=np.int64)
    scores = np.asarray(
        [float(record[score_field]) for record in usable],
        dtype=np.float64,
    )
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("labels used for ROC/DET plots must be binary (0 or 1)")
    if not np.isfinite(scores).all():
        raise ValueError(f"{score_field} used for ROC/DET plots must be finite")

    num_positive = int(np.sum(labels == 1))
    num_negative = int(np.sum(labels == 0))
    if num_positive == 0 or num_negative == 0:
        return None

    order = np.argsort(-scores, kind="stable")
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    distinct_score_ends = np.flatnonzero(
        np.r_[sorted_scores[1:] != sorted_scores[:-1], True]
    )
    true_positives = np.cumsum(sorted_labels == 1)[distinct_score_ends]
    false_positives = np.cumsum(sorted_labels == 0)[distinct_score_ends]

    return {
        "fpr": np.r_[0.0, false_positives / num_negative],
        "tpr": np.r_[0.0, true_positives / num_positive],
        "thresholds": np.r_[np.inf, sorted_scores[distinct_score_ends]],
        "num_samples": len(usable),
        "num_positive": num_positive,
        "num_negative": num_negative,
    }


def _threshold_point(
    records: list[dict],
    *,
    score_field: str,
    threshold: float,
) -> tuple[float, float]:
    usable = [
        record
        for record in records
        if "label" in record
        and not bool(record.get("skipped", False))
        and record.get(score_field) is not None
    ]
    labels = np.asarray([int(record["label"]) for record in usable], dtype=np.int64)
    scores = np.asarray(
        [float(record[score_field]) for record in usable],
        dtype=np.float64,
    )
    accepted = scores >= threshold
    fpr = float(np.mean(accepted[labels == 0]))
    tpr = float(np.mean(accepted[labels == 1]))
    return fpr, tpr


def _probit(values: np.ndarray, *, clip: float = 1.0e-4) -> np.ndarray:
    """Map probabilities to normal-deviate coordinates for a DET plot."""

    normal = NormalDist()
    clipped = np.clip(np.asarray(values, dtype=np.float64), clip, 1.0 - clip)
    return np.asarray([normal.inv_cdf(float(value)) for value in clipped])


def write_detection_plots(
    records: list[dict],
    *,
    output_dir: Path,
    threshold: float,
    metrics: dict[str, float],
    dpi: int = DEFAULT_PLOT_DPI,
) -> dict[str, Any]:
    """Write ROC and normal-deviate DET plots for the utterance score."""

    score_field = "qbyt_score"
    curve = binary_roc_points(records, score_field=score_field)
    if curve is None:
        return {
            "status": "skipped",
            "score_field": score_field,
            "reason": "ROC/DET plots require at least one valid positive and negative sample",
        }
    if dpi <= 0:
        raise ValueError("plot_dpi must be positive")

    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ImportError:
        return {
            "status": "skipped",
            "score_field": score_field,
            "reason": "matplotlib is not installed",
        }

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fpr = curve["fpr"]
    tpr = curve["tpr"]
    fnr = 1.0 - tpr
    deploy_fpr, deploy_tpr = _threshold_point(
        records,
        score_field=score_field,
        threshold=threshold,
    )
    eer_threshold = float(metrics["eer_threshold"])
    eer_fpr, eer_tpr = _threshold_point(
        records,
        score_field=score_field,
        threshold=eer_threshold,
    )

    roc_path = output_dir / "roc_curve.png"
    figure, axis = plt.subplots(figsize=(6.4, 5.2))
    try:
        axis.step(
            fpr,
            tpr,
            where="post",
            linewidth=2.0,
            label=f"QbyT (AUC={float(metrics['auc']):.4f})",
        )
        axis.plot([0.0, 1.0], [0.0, 1.0], "--", color="0.6", label="Chance")
        axis.scatter(
            [deploy_fpr],
            [deploy_tpr],
            color="tab:orange",
            marker="o",
            zorder=3,
            label=f"Deploy threshold={threshold:.4g}",
        )
        axis.scatter(
            [eer_fpr],
            [eer_tpr],
            color="tab:green",
            marker="x",
            s=60,
            zorder=3,
            label=f"EER={float(metrics['eer']):.4f}",
        )
        axis.set(
            xlim=(0.0, 1.0),
            ylim=(0.0, 1.0),
            xlabel="False Positive Rate",
            ylabel="True Positive Rate",
            title="Stage II ROC Curve",
        )
        axis.grid(True, alpha=0.25)
        axis.legend(loc="lower right")
        figure.tight_layout()
        figure.savefig(roc_path, dpi=dpi, bbox_inches="tight")
    finally:
        plt.close(figure)

    det_path = output_dir / "det_curve.png"
    det_tick_probabilities = np.asarray(
        [
            0.0001,
            0.001,
            0.01,
            0.05,
            0.1,
            0.2,
            0.4,
            0.6,
            0.8,
            0.9,
            0.95,
            0.99,
            0.999,
        ],
        dtype=np.float64,
    )
    det_tick_labels = [
        "0.01",
        "0.1",
        "1",
        "5",
        "10",
        "20",
        "40",
        "60",
        "80",
        "90",
        "95",
        "99",
        "99.9",
    ]
    det_ticks = _probit(det_tick_probabilities)
    figure, axis = plt.subplots(figsize=(6.4, 5.2))
    try:
        axis.plot(_probit(fpr), _probit(fnr), linewidth=2.0, label="QbyT")
        axis.scatter(
            _probit(np.asarray([deploy_fpr])),
            _probit(np.asarray([1.0 - deploy_tpr])),
            color="tab:orange",
            marker="o",
            zorder=3,
            label=f"Deploy threshold={threshold:.4g}",
        )
        axis.scatter(
            _probit(np.asarray([eer_fpr])),
            _probit(np.asarray([1.0 - eer_tpr])),
            color="tab:green",
            marker="x",
            s=60,
            zorder=3,
            label=f"EER={float(metrics['eer']):.4f}",
        )
        axis.set_xticks(det_ticks, det_tick_labels)
        axis.set_yticks(det_ticks, det_tick_labels)
        axis.set(
            xlim=(det_ticks[0], det_ticks[-1]),
            ylim=(det_ticks[0], det_ticks[-1]),
            xlabel="False Positive Rate (%)",
            ylabel="False Negative Rate (%)",
            title="Stage II DET Curve",
        )
        axis.grid(True, alpha=0.25)
        axis.legend(loc="upper right")
        figure.tight_layout()
        figure.savefig(det_path, dpi=dpi, bbox_inches="tight")
    finally:
        plt.close(figure)

    return {
        "status": "generated",
        "score_field": score_field,
        "num_samples": curve["num_samples"],
        "num_positive": curve["num_positive"],
        "num_negative": curve["num_negative"],
        "roc": str(roc_path.resolve()),
        "det": str(det_path.resolve()),
    }
