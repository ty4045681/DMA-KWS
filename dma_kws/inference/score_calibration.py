"""Regularized affine calibration and fusion for Stage II raw logits.

The fitted model always operates on logits rather than probabilities::

    calibrated_logit = bias + features @ weights

For one feature this is ordinary Platt scaling.  With the utterance and
completion logits together it is a two-feature logistic fusion model.  The
optimizer standardizes features internally for numerical stability, applies
L2 only to slopes, then stores the equivalent coefficients in the original
raw-logit coordinate system.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


_CALIBRATOR_VERSION = 1


def sigmoid(values: np.ndarray | Sequence[float]) -> np.ndarray:
    """Return a finite, overflow-safe sigmoid as ``float64``."""

    values = np.asarray(values, dtype=np.float64)
    output = np.empty_like(values)
    positive = values >= 0.0
    output[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    negative_exp = np.exp(values[~positive])
    output[~positive] = negative_exp / (1.0 + negative_exp)
    return output


def _as_feature_matrix(features: np.ndarray | Sequence[Sequence[float]]) -> np.ndarray:
    matrix = np.asarray(features, dtype=np.float64)
    if matrix.ndim == 1:
        matrix = matrix.reshape(-1, 1)
    if matrix.ndim != 2 or matrix.shape[1] == 0:
        raise ValueError("features must have shape [num_samples, num_features]")
    if not np.isfinite(matrix).all():
        raise ValueError("features must contain only finite values")
    return matrix


def _as_binary_labels(labels: np.ndarray | Sequence[int], *, size: int) -> np.ndarray:
    values = np.asarray(labels)
    if values.ndim != 1 or values.size != size:
        raise ValueError(
            f"labels must have shape [{size}], got {tuple(values.shape)}"
        )
    if not np.isin(values, (0, 1)).all():
        raise ValueError("labels must contain only 0 and 1")
    values = values.astype(np.float64, copy=False)
    if np.unique(values).size != 2:
        raise ValueError("calibration requires both positive and negative samples")
    return values


@dataclass(frozen=True)
class AffineLogitCalibrator:
    """Serializable affine transform trained with binary cross entropy."""

    feature_names: tuple[str, ...]
    weights: tuple[float, ...]
    bias: float
    l2: float
    num_samples: int
    num_positive: int
    fit_iterations: int
    converged: bool

    def __post_init__(self) -> None:
        if not self.feature_names:
            raise ValueError("feature_names must not be empty")
        if len(self.feature_names) != len(self.weights):
            raise ValueError("feature_names and weights must have the same length")
        if len(set(self.feature_names)) != len(self.feature_names):
            raise ValueError("feature_names must be unique")
        values = np.asarray((*self.weights, self.bias, self.l2), dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError("calibrator coefficients must be finite")
        if self.l2 < 0.0:
            raise ValueError("l2 must be non-negative")
        if self.num_samples <= 0:
            raise ValueError("num_samples must be positive")
        if not 0 < self.num_positive < self.num_samples:
            raise ValueError("num_positive must leave both classes represented")
        if self.fit_iterations < 0:
            raise ValueError("fit_iterations must be non-negative")

    def transform_logits(
        self,
        features: np.ndarray | Sequence[Sequence[float]],
    ) -> np.ndarray:
        """Apply the affine transform to an ordered feature matrix."""

        matrix = _as_feature_matrix(features)
        if matrix.shape[1] != len(self.weights):
            raise ValueError(
                f"expected {len(self.weights)} features {self.feature_names}, "
                f"got {matrix.shape[1]}"
            )
        return matrix @ np.asarray(self.weights, dtype=np.float64) + self.bias

    def predict_proba(
        self,
        features: np.ndarray | Sequence[Sequence[float]],
    ) -> np.ndarray:
        return sigmoid(self.transform_logits(features))

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": _CALIBRATOR_VERSION,
            "feature_names": list(self.feature_names),
            "weights": list(self.weights),
            "bias": self.bias,
            "l2": self.l2,
            "fit": {
                "num_samples": self.num_samples,
                "num_positive": self.num_positive,
                "iterations": self.fit_iterations,
                "converged": self.converged,
            },
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "AffineLogitCalibrator":
        if int(value.get("version", -1)) != _CALIBRATOR_VERSION:
            raise ValueError(
                f"unsupported calibrator version {value.get('version')!r}; "
                f"expected {_CALIBRATOR_VERSION}"
            )
        fit = value.get("fit")
        if not isinstance(fit, dict):
            raise ValueError("calibrator fit metadata must be a mapping")
        return cls(
            feature_names=tuple(str(name) for name in value["feature_names"]),
            weights=tuple(float(weight) for weight in value["weights"]),
            bias=float(value["bias"]),
            l2=float(value["l2"]),
            num_samples=int(fit["num_samples"]),
            num_positive=int(fit["num_positive"]),
            fit_iterations=int(fit["iterations"]),
            converged=bool(fit["converged"]),
        )


def _objective(
    design: np.ndarray,
    labels: np.ndarray,
    parameters: np.ndarray,
    *,
    l2: float,
) -> float:
    logits = design @ parameters
    data_loss = np.mean(np.logaddexp(0.0, logits) - labels * logits)
    return float(data_loss + 0.5 * l2 * np.dot(parameters[:-1], parameters[:-1]))


def fit_affine_logit_calibrator(
    features: np.ndarray | Sequence[Sequence[float]],
    labels: np.ndarray | Sequence[int],
    *,
    feature_names: Sequence[str],
    l2: float = 1.0e-2,
    max_iterations: int = 100,
    tolerance: float = 1.0e-9,
) -> AffineLogitCalibrator:
    """Fit a regularized affine logit model with damped Newton updates.

    ``l2`` is defined in standardized-feature coordinates, so its meaning is
    not dominated by the raw scale of either score head.  The intercept is not
    regularized.  Returned weights are converted back to the caller's original
    feature coordinates.
    """

    matrix = _as_feature_matrix(features)
    targets = _as_binary_labels(labels, size=matrix.shape[0])
    names = tuple(str(name) for name in feature_names)
    if len(names) != matrix.shape[1]:
        raise ValueError(
            f"feature_names has {len(names)} entries for {matrix.shape[1]} features"
        )
    if not names or len(set(names)) != len(names):
        raise ValueError("feature_names must be non-empty and unique")
    if not np.isfinite(l2) or l2 < 0.0:
        raise ValueError("l2 must be a finite non-negative value")
    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive")
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be a finite positive value")

    means = matrix.mean(axis=0)
    scales = matrix.std(axis=0)
    scales = np.where(scales > np.finfo(np.float64).eps, scales, 1.0)
    standardized = (matrix - means) / scales
    design = np.column_stack(
        (standardized, np.ones(standardized.shape[0], dtype=np.float64))
    )

    parameters = np.zeros(design.shape[1], dtype=np.float64)
    prevalence = float(targets.mean())
    parameters[-1] = np.log(prevalence / (1.0 - prevalence))
    converged = False
    completed_iterations = 0

    for iteration in range(1, max_iterations + 1):
        completed_iterations = iteration
        logits = design @ parameters
        probabilities = sigmoid(logits)
        residual = probabilities - targets
        gradient = design.T @ residual / targets.size
        gradient[:-1] += l2 * parameters[:-1]

        if float(np.max(np.abs(gradient))) <= tolerance:
            converged = True
            break

        curvature = probabilities * (1.0 - probabilities)
        hessian = (design.T * curvature) @ design / targets.size
        hessian[:-1, :-1] += l2 * np.eye(matrix.shape[1], dtype=np.float64)
        # A tiny diagonal damping keeps zero-L2 and nearly separable probes
        # numerically total without materially changing a regularized fit.
        hessian += np.eye(hessian.shape[0], dtype=np.float64) * 1.0e-12
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(hessian, gradient, rcond=None)[0]

        old_objective = _objective(design, targets, parameters, l2=l2)
        step_scale = 1.0
        accepted = False
        while step_scale >= 2.0**-20:
            candidate = parameters - step_scale * step
            candidate_objective = _objective(design, targets, candidate, l2=l2)
            if np.isfinite(candidate_objective) and candidate_objective <= old_objective:
                parameters = candidate
                accepted = True
                break
            step_scale *= 0.5
        if not accepted:
            break
        if float(np.max(np.abs(step_scale * step))) <= tolerance:
            converged = True
            break

    standardized_weights = parameters[:-1]
    raw_weights = standardized_weights / scales
    raw_bias = float(parameters[-1] - np.dot(means, raw_weights))
    if not np.isfinite(raw_weights).all() or not np.isfinite(raw_bias):
        raise RuntimeError("calibrator fit produced non-finite coefficients")

    return AffineLogitCalibrator(
        feature_names=names,
        weights=tuple(float(value) for value in raw_weights),
        bias=raw_bias,
        l2=float(l2),
        num_samples=int(targets.size),
        num_positive=int(targets.sum()),
        fit_iterations=completed_iterations,
        converged=converged,
    )


__all__ = [
    "AffineLogitCalibrator",
    "fit_affine_logit_calibrator",
    "sigmoid",
]
