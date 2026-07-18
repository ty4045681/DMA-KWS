"""Classification metrics for two-stage manifest evaluation."""

from __future__ import annotations

from typing import Any

import numpy as np


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _binary_predictions(labels: np.ndarray, scores: np.ndarray, threshold: float) -> np.ndarray:
    return scores >= threshold


def _confusion_counts(labels: np.ndarray, predictions: np.ndarray) -> tuple[int, int, int, int]:
    labels = labels.astype(np.int64)
    predictions = predictions.astype(np.int64)
    tp = int(np.sum((labels == 1) & (predictions == 1)))
    tn = int(np.sum((labels == 0) & (predictions == 0)))
    fp = int(np.sum((labels == 0) & (predictions == 1)))
    fn = int(np.sum((labels == 1) & (predictions == 0)))
    return tp, tn, fp, fn


def binary_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Compute ROC AUC with a rank-based estimator."""
    labels = labels.astype(np.int64)
    scores = scores.astype(np.float64)
    positives = int(np.sum(labels == 1))
    negatives = int(np.sum(labels == 0))
    if positives == 0 or negatives == 0:
        return 0.0

    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)

    # Average ranks for tied scores.
    sorted_scores = scores[order]
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        avg_rank = ranks[order[start:end]].mean()
        ranks[order[start:end]] = avg_rank
        start = end

    positive_ranks = ranks[labels == 1].sum()
    auc = (positive_ranks - positives * (positives + 1) / 2.0) / (positives * negatives)
    return float(np.clip(auc, 0.0, 1.0))


def binary_eer(labels: np.ndarray, scores: np.ndarray) -> float:
    """Compute equal error rate over score thresholds."""
    labels = labels.astype(np.int64)
    scores = scores.astype(np.float64)
    positives = int(np.sum(labels == 1))
    negatives = int(np.sum(labels == 0))
    if positives == 0 or negatives == 0:
        return 0.0

    thresholds = np.unique(scores)
    best_eer = 1.0
    for threshold in thresholds:
        predictions = _binary_predictions(labels, scores, float(threshold))
        tp, tn, fp, fn = _confusion_counts(labels, predictions)
        fpr = fp / negatives
        fnr = fn / positives
        best_eer = min(best_eer, abs(fpr - fnr))

    return float(best_eer)


def summarize_labeled_results(
    results: list[dict[str, Any]],
    *,
    threshold: float,
) -> dict[str, float]:
    """Aggregate accuracy, precision, recall, f1, auc, and eer from labeled rows."""
    labeled = [row for row in results if "label" in row]
    if not labeled:
        return {}

    labels = np.array([int(row["label"]) for row in labeled], dtype=np.int64)
    scores = np.array([float(row.get("best_qbyt_score", 0.0)) for row in labeled], dtype=np.float64)
    predictions = _binary_predictions(labels, scores, threshold).astype(np.int64)

    tp, tn, fp, fn = _confusion_counts(labels, predictions)
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    fpr = _safe_div(fp, fp + tn)
    fnr = _safe_div(fn, fn + tp)

    return {
        "num_samples": float(len(labeled)),
        "accuracy": _safe_div(tp + tn, len(labeled)),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": fpr,
        "fnr": fnr,
        "auc": binary_auc(labels, scores),
        "eer": binary_eer(labels, scores),
        "threshold": float(threshold),
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
    }


def summarize_false_accept_rate(
    results: list[dict[str, Any]],
    *,
    threshold: float,
    total_hours: float,
) -> dict[str, float]:
    """Aggregate FA/hour metrics from negative-only evaluation results.

    All rows are expected to have ``label=0`` (no keyword present). The false
    accept rate is ``fp / total_hours``. Also reports the standard labeled
    metrics so the output can be consumed by the same tooling as
    :func:`summarize_labeled_results`.
    """
    summary = summarize_labeled_results(results, threshold=threshold)
    if not summary:
        return {}

    fp = float(summary.get("fp", 0.0))
    return {
        **summary,
        "total_hours": float(total_hours),
        "fa_per_hour": _safe_div(fp, total_hours),
        "fa_per_1000_hours": _safe_div(fp * 1000.0, total_hours),
    }
