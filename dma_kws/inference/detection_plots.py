"""Reusable detection plot generation for Stage-II scores."""

from __future__ import annotations

import csv
from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np


DEFAULT_PLOT_DPI = 160
_CONSTRAINT_COLOR = "red"


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


def false_accept_rate_points(
    records: list[dict],
    *,
    score_field: str,
    total_hours: float,
) -> dict[str, Any] | None:
    """Build exact threshold/FA-hour points for negative-only evaluation."""

    hours = float(total_hours)
    if not np.isfinite(hours) or hours <= 0.0:
        raise ValueError("total_hours used for an FA/hour plot must be positive")

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
    if not np.all(labels == 0):
        raise ValueError("FA/hour plots require negative-only records with label=0")

    scores = np.asarray(
        [float(record[score_field]) for record in usable],
        dtype=np.float64,
    )
    if not np.isfinite(scores).all():
        raise ValueError(f"{score_field} used for an FA/hour plot must be finite")
    if np.any((scores < 0.0) | (scores > 1.0)):
        raise ValueError(
            f"{score_field} used for an FA/hour plot must be in [0, 1]"
        )

    thresholds = np.unique(np.r_[0.0, scores, 1.0])
    sorted_scores = np.sort(scores)
    false_accepts = scores.size - np.searchsorted(
        sorted_scores,
        thresholds,
        side="left",
    )
    return {
        "thresholds": thresholds,
        "false_accepts": false_accepts,
        "fa_per_hour": false_accepts.astype(np.float64) / hours,
        "num_samples": len(usable),
        "total_hours": hours,
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


def _constraint_rate(name: str, value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number in [0, 1]")
    try:
        rate = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number in [0, 1]") from exc
    if not np.isfinite(rate) or not 0.0 <= rate <= 1.0:
        raise ValueError(f"{name} must be a finite number in [0, 1]")
    return rate


def _best_threshold_index(
    candidates: np.ndarray,
    thresholds: np.ndarray,
) -> int:
    return int(candidates[int(np.argmax(thresholds[candidates]))])


def select_roc_constraint_point(
    curve: dict[str, Any],
    *,
    min_recall: float | None = None,
    max_fpr: float | None = None,
) -> dict[str, Any] | None:
    """Select one real ROC operating point under an optional plot constraint.

    Empirical ROC points are discrete.  The returned ``actual_*`` values are a
    conservative threshold that can really be deployed.  ``guide_*`` keeps the
    requested axis value and projects it onto the empirical staircase so the
    plot can show an unavailable bound explicitly without inventing a threshold.
    """

    requested_recall = _constraint_rate("plot_min_recall", min_recall)
    requested_fpr = _constraint_rate("plot_max_fpr", max_fpr)
    if requested_recall is not None and requested_fpr is not None:
        raise ValueError(
            "plot_min_recall and plot_max_fpr are mutually exclusive; configure only one"
        )
    if requested_recall is None and requested_fpr is None:
        return None

    fpr = np.asarray(curve["fpr"], dtype=np.float64)
    recall = np.asarray(curve["tpr"], dtype=np.float64)
    thresholds = np.asarray(curve["thresholds"], dtype=np.float64)
    if not (fpr.shape == recall.shape == thresholds.shape) or fpr.ndim != 1:
        raise ValueError("ROC curve fpr, tpr, and thresholds must be aligned 1-D arrays")

    if requested_recall is not None:
        candidates = np.flatnonzero(recall >= requested_recall)
        best_fpr = np.min(fpr[candidates])
        candidates = candidates[fpr[candidates] == best_fpr]
        best_recall = np.max(recall[candidates])
        candidates = candidates[recall[candidates] == best_recall]
        index = _best_threshold_index(candidates, thresholds)
        kind = "min_recall"
        metric = "recall"
        requested = requested_recall
    else:
        assert requested_fpr is not None
        candidates = np.flatnonzero(fpr <= requested_fpr)
        best_recall = np.max(recall[candidates])
        candidates = candidates[recall[candidates] == best_recall]
        best_fpr = np.min(fpr[candidates])
        candidates = candidates[fpr[candidates] == best_fpr]
        index = _best_threshold_index(candidates, thresholds)
        kind = "max_fpr"
        metric = "fpr"
        requested = requested_fpr

    threshold = float(thresholds[index])
    guide_recall = (
        float(requested) if kind == "min_recall" else float(recall[index])
    )
    guide_fpr = float(fpr[index]) if kind == "min_recall" else float(requested)
    exact = bool(
        np.any(
            np.isclose(fpr, guide_fpr, rtol=0.0, atol=1.0e-12)
            & np.isclose(recall, guide_recall, rtol=0.0, atol=1.0e-12)
        )
    )
    return {
        "kind": kind,
        "metric": metric,
        "requested": float(requested),
        "exact": exact,
        "actual_recall": float(recall[index]),
        "actual_fpr": float(fpr[index]),
        "guide_recall": guide_recall,
        "guide_fpr": guide_fpr,
        "threshold": threshold if np.isfinite(threshold) else None,
    }


def _constraint_legend_label(constraint: dict[str, Any]) -> str:
    if constraint["kind"] == "min_recall":
        label = f"Min Recall={constraint['requested']:.4g}"
    else:
        label = f"Max FPR={constraint['requested']:.4g}"
    if not constraint["exact"]:
        label += " (interpolated)"
    return label


def _draw_constraint_guides(
    axis: Any,
    *,
    x: float,
    y: float,
    x_origin: float,
    y_origin: float,
    x_label: str,
    y_label: str,
    legend_label: str,
) -> None:
    line_style = (0, (5, 3))
    axis.plot(
        [x_origin, x],
        [y, y],
        color=_CONSTRAINT_COLOR,
        linestyle=line_style,
        linewidth=1.4,
        zorder=3,
    )
    axis.plot(
        [x, x],
        [y_origin, y],
        color=_CONSTRAINT_COLOR,
        linestyle=line_style,
        linewidth=1.4,
        zorder=3,
    )
    axis.scatter(
        [x],
        [y],
        color=_CONSTRAINT_COLOR,
        marker="o",
        s=38,
        zorder=5,
        clip_on=False,
        label=legend_label,
    )
    text_box = {
        "facecolor": "white",
        "edgecolor": "none",
        "alpha": 0.85,
        "pad": 0.8,
    }
    axis.annotate(
        x_label,
        xy=(x, 0.0),
        xycoords=("data", "axes fraction"),
        xytext=(0, -18),
        textcoords="offset points",
        color=_CONSTRAINT_COLOR,
        fontsize=8,
        ha="center",
        va="top",
        annotation_clip=False,
        bbox=text_box,
    )
    axis.annotate(
        y_label,
        xy=(0.0, y),
        xycoords=("axes fraction", "data"),
        xytext=(5, 0),
        textcoords="offset points",
        color=_CONSTRAINT_COLOR,
        fontsize=8,
        ha="left",
        va="center",
        annotation_clip=False,
        bbox=text_box,
    )


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
    min_recall: float | None = None,
    max_fpr: float | None = None,
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
    constraint = select_roc_constraint_point(
        curve,
        min_recall=min_recall,
        max_fpr=max_fpr,
    )

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
        if constraint is not None:
            constraint_fpr = float(constraint["guide_fpr"])
            constraint_recall = float(constraint["guide_recall"])
            _draw_constraint_guides(
                axis,
                x=constraint_fpr,
                y=constraint_recall,
                x_origin=0.0,
                y_origin=0.0,
                x_label=f"FPR={constraint_fpr:.4f}",
                y_label=f"Recall={constraint_recall:.4f}",
                legend_label=_constraint_legend_label(constraint),
            )
        axis.set(
            xlim=(0.0, 1.0),
            ylim=(0.0, 1.0),
            xlabel="False Positive Rate",
            ylabel="Recall (True Positive Rate)",
            title="Stage II ROC Curve",
        )
        axis.xaxis.labelpad = 16
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
            0.9999,
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
        "99.99",
    ]
    det_ticks = _probit(det_tick_probabilities)
    figure, axis = plt.subplots(figsize=(6.4, 5.2))
    try:
        axis.step(
            _probit(fpr),
            _probit(fnr),
            where="post",
            linewidth=2.0,
            label="QbyT",
        )
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
        if constraint is not None:
            constraint_fpr = float(constraint["guide_fpr"])
            constraint_recall = float(constraint["guide_recall"])
            constraint_fnr = 1.0 - constraint_recall
            constraint_det_x = float(
                _probit(np.asarray([constraint_fpr], dtype=np.float64))[0]
            )
            constraint_det_y = float(
                _probit(np.asarray([constraint_fnr], dtype=np.float64))[0]
            )
            _draw_constraint_guides(
                axis,
                x=constraint_det_x,
                y=constraint_det_y,
                x_origin=float(det_ticks[0]),
                y_origin=float(det_ticks[0]),
                x_label=f"FPR={constraint_fpr:.2%}",
                y_label=(
                    f"FNR={constraint_fnr:.2%}\n"
                    f"Recall={constraint_recall:.2%}"
                ),
                legend_label=_constraint_legend_label(constraint),
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
        axis.xaxis.labelpad = 16
        axis.grid(True, alpha=0.25)
        axis.legend(loc="upper right")
        figure.tight_layout()
        figure.savefig(det_path, dpi=dpi, bbox_inches="tight")
    finally:
        plt.close(figure)

    result = {
        "status": "generated",
        "score_field": score_field,
        "num_samples": curve["num_samples"],
        "num_positive": curve["num_positive"],
        "num_negative": curve["num_negative"],
        "roc": str(roc_path.resolve()),
        "det": str(det_path.resolve()),
    }
    if constraint is not None:
        result["constraint"] = constraint
    return result


def write_false_accept_rate_plot(
    records: list[dict],
    *,
    output_dir: Path,
    threshold: float,
    total_hours: float,
    dpi: int = DEFAULT_PLOT_DPI,
) -> dict[str, Any]:
    """Write a threshold-versus-FA/hour plot for negative-only MUSAN scores."""

    score_field = "qbyt_score"
    hours = float(total_hours)
    if not np.isfinite(hours) or hours <= 0.0:
        return {
            "status": "skipped",
            "score_field": score_field,
            "reason": "FA/hour plots require positive total audio duration",
        }
    curve = false_accept_rate_points(
        records,
        score_field=score_field,
        total_hours=hours,
    )
    if curve is None:
        return {
            "status": "skipped",
            "score_field": score_field,
            "reason": "FA/hour plots require at least one valid negative sample",
        }
    if dpi <= 0:
        raise ValueError("plot_dpi must be positive")

    deployment_threshold = float(threshold)
    if not np.isfinite(deployment_threshold):
        raise ValueError("threshold used for an FA/hour plot must be finite")

    thresholds = np.asarray(curve["thresholds"], dtype=np.float64)
    fa_per_hour = np.asarray(curve["fa_per_hour"], dtype=np.float64)
    false_accepts = np.asarray(curve["false_accepts"], dtype=np.int64)
    deploy_index = int(
        np.searchsorted(thresholds, deployment_threshold, side="left")
    )
    if deploy_index >= thresholds.size:
        deploy_false_accepts = 0
    else:
        deploy_false_accepts = int(false_accepts[deploy_index])
    deploy_fa_per_hour = deploy_false_accepts / hours

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "fa_per_hour_curve.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "threshold",
                "false_accepts",
                "fa_per_hour",
                "fa_per_24_hours",
                "fa_per_1000_hours",
            ]
        )
        for threshold_value, count, rate in zip(
            thresholds, false_accepts, fa_per_hour
        ):
            writer.writerow(
                [
                    float(threshold_value),
                    int(count),
                    float(rate),
                    float(rate) * 24.0,
                    float(rate) * 1000.0,
                ]
            )

    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ImportError:
        return {
            "status": "skipped",
            "score_field": score_field,
            "reason": "matplotlib is not installed",
            "fa_per_hour_curve_csv": str(csv_path.resolve()),
        }

    plot_path = output_dir / "fa_per_hour_curve.png"
    figure, axis = plt.subplots(figsize=(6.4, 5.2))
    try:
        axis.step(
            thresholds,
            fa_per_hour,
            where="pre",
            linewidth=2.0,
            label="QbyT",
        )
        axis.scatter(
            [deployment_threshold],
            [deploy_fa_per_hour],
            color="tab:orange",
            marker="o",
            zorder=3,
            label=(
                f"Deploy threshold={deployment_threshold:.4g} "
                f"(FA/hour={deploy_fa_per_hour:.4g})"
            ),
        )
        axis.set(
            xlim=(min(0.0, deployment_threshold), max(1.0, deployment_threshold)),
            xlabel="QbyT Threshold",
            ylabel="False Accepts per Hour",
            title="Stage II MUSAN False Accept Rate",
        )
        axis.set_ylim(bottom=0.0)
        axis.grid(True, alpha=0.25)
        axis.legend(loc="upper right")
        figure.tight_layout()
        figure.savefig(plot_path, dpi=dpi, bbox_inches="tight")
    finally:
        plt.close(figure)

    return {
        "status": "generated",
        "score_field": score_field,
        "num_samples": curve["num_samples"],
        "total_hours": hours,
        "deployment_threshold": deployment_threshold,
        "deployment_false_accepts": deploy_false_accepts,
        "deployment_fa_per_hour": deploy_fa_per_hour,
        "fa_per_hour_curve": str(plot_path.resolve()),
        "fa_per_hour_curve_csv": str(csv_path.resolve()),
    }
