"""Monotone affine calibration for a Stage-II raw score.

Calibration is deliberately kept outside the acoustic model.  It maps one
raw score to a probability without changing the score ordering::

    calibrated_logit = slope * raw_score + bias
    probability = sigmoid(calibrated_logit)

``slope`` is always strictly positive.  Consequently calibration may move the
deployment threshold and repair probability calibration, but it cannot hide a
ranking regression or introduce a second inference path around the readout.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


CALIBRATION_SCHEMA_VERSION = 1
CALIBRATOR_TYPE = "positive_affine_logit"
_MIN_STANDARDIZED_SLOPE = 1.0e-12


def sigmoid(values: float | np.ndarray | Sequence[float]) -> np.ndarray:
    """Return an overflow-safe sigmoid in ``float64``."""

    array = np.asarray(values, dtype=np.float64)
    flat = array.reshape(-1)
    output = np.empty_like(flat)
    positive = flat >= 0.0
    output[positive] = 1.0 / (1.0 + np.exp(-flat[positive]))
    negative_exp = np.exp(flat[~positive])
    output[~positive] = negative_exp / (1.0 + negative_exp)
    return output.reshape(array.shape)


def _as_raw_scores(
    raw_scores: float | np.ndarray | Sequence[float],
    *,
    allow_scalar: bool,
) -> np.ndarray:
    values = np.asarray(raw_scores, dtype=np.float64)
    expected_dimensions = (0, 1) if allow_scalar else (1,)
    if values.ndim not in expected_dimensions:
        raise ValueError("raw_scores must be a scalar or one-dimensional sequence")
    if not allow_scalar and values.size == 0:
        raise ValueError("raw_scores must not be empty")
    if not np.isfinite(values).all():
        raise ValueError("raw_scores must contain only finite values")
    return values


def _as_binary_labels(
    labels: np.ndarray | Sequence[int],
    *,
    size: int,
) -> np.ndarray:
    values = np.asarray(labels)
    if values.ndim != 1 or values.size != size:
        raise ValueError(f"labels must have shape [{size}], got {tuple(values.shape)}")
    if not np.isin(values, (0, 1)).all():
        raise ValueError("labels must contain only 0 and 1")
    values = values.astype(np.float64, copy=False)
    if np.unique(values).size != 2:
        raise ValueError("calibration requires both positive and negative samples")
    return values


def _finite_float(value: object, *, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not np.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def _strict_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    return int(value)


@dataclass(frozen=True)
class CalibrationFit:
    """Metadata describing a held-out Platt fit."""

    num_samples: int
    num_positive: int
    l2: float
    iterations: int
    converged: bool

    def __post_init__(self) -> None:
        if self.num_samples <= 1:
            raise ValueError("fit num_samples must be greater than one")
        if not 0 < self.num_positive < self.num_samples:
            raise ValueError("fit metadata must represent both classes")
        if not np.isfinite(self.l2) or self.l2 < 0.0:
            raise ValueError("fit l2 must be finite and non-negative")
        if self.iterations < 0:
            raise ValueError("fit iterations must be non-negative")
        if not isinstance(self.converged, bool):
            raise ValueError("fit converged must be a boolean")

    def to_dict(self) -> dict[str, Any]:
        return {
            "num_samples": self.num_samples,
            "num_positive": self.num_positive,
            "l2": self.l2,
            "iterations": self.iterations,
            "converged": self.converged,
        }

    @classmethod
    def from_dict(cls, value: object) -> "CalibrationFit":
        if not isinstance(value, Mapping):
            raise ValueError("calibrator fit metadata must be a mapping or null")
        converged = value.get("converged")
        if not isinstance(converged, bool):
            raise ValueError("fit converged must be a boolean")
        return cls(
            num_samples=_strict_int(value.get("num_samples"), name="fit num_samples"),
            num_positive=_strict_int(
                value.get("num_positive"), name="fit num_positive"
            ),
            l2=_finite_float(value.get("l2"), name="fit l2"),
            iterations=_strict_int(value.get("iterations"), name="fit iterations"),
            converged=converged,
        )


@dataclass(frozen=True)
class PositiveAffineCalibrator:
    """Versioned, serializable ``z = slope * raw_score + bias`` calibrator."""

    slope: float
    bias: float
    score_name: str = "qbyt_raw_logit"
    fit: CalibrationFit | None = None

    def __post_init__(self) -> None:
        if not np.isfinite(self.slope) or self.slope <= 0.0:
            raise ValueError("calibrator slope must be finite and strictly positive")
        if not np.isfinite(self.bias):
            raise ValueError("calibrator bias must be finite")
        if not isinstance(self.score_name, str) or not self.score_name.strip():
            raise ValueError("score_name must be a non-empty string")
        if self.fit is not None and not isinstance(self.fit, CalibrationFit):
            raise ValueError("fit must be CalibrationFit or None")

    @property
    def raw_operating_threshold(self) -> float:
        """Raw score mapped to calibrated probability ``0.5``."""

        return -self.bias / self.slope

    def transform_logits(
        self,
        raw_scores: float | np.ndarray | Sequence[float],
    ) -> np.ndarray:
        """Apply the positive affine transform while preserving input shape."""

        values = _as_raw_scores(raw_scores, allow_scalar=True)
        with np.errstate(over="ignore", invalid="ignore"):
            logits = self.slope * values + self.bias
        # Finite inputs and coefficients may still overflow at the very edge of
        # float64.  Saturating retains the correct sigmoid limit and ordering.
        limit = np.finfo(np.float64).max
        return np.nan_to_num(logits, nan=0.0, posinf=limit, neginf=-limit)

    def predict_proba(
        self,
        raw_scores: float | np.ndarray | Sequence[float],
    ) -> np.ndarray:
        """Return calibrated positive-class probabilities."""

        return sigmoid(self.transform_logits(raw_scores))

    def transform_one(self, raw_score: float) -> float:
        """Scalar convenience wrapper for inference call sites."""

        values = _as_raw_scores(raw_score, allow_scalar=True)
        if values.ndim != 0:
            raise ValueError("raw_score must be a scalar")
        return float(self.transform_logits(values))

    def predict_one(self, raw_score: float) -> float:
        """Calibrate one score and return a Python float."""

        return float(sigmoid(self.transform_one(raw_score)))

    def at_operating_threshold(
        self,
        raw_threshold: float,
    ) -> "PositiveAffineCalibrator":
        """Recenter this calibrator so ``raw_threshold`` maps to ``0.5``.

        The fitted positive slope is retained, while the intercept is replaced
        by the deployment operating point.  Fit metadata is cleared because
        the returned intercept is selected from an operating curve rather than
        by the held-out binary cross-entropy fit.
        """

        return self.from_operating_threshold(
            raw_threshold,
            slope=self.slope,
            score_name=self.score_name,
        )

    @classmethod
    def from_operating_threshold(
        cls,
        raw_threshold: float,
        *,
        slope: float = 1.0,
        score_name: str = "qbyt_raw_logit",
    ) -> "PositiveAffineCalibrator":
        """Map ``raw_threshold`` exactly to probability ``0.5``.

        This constructor is useful when an operating point is selected from a
        deployment ROC/FA curve but there is not yet a representative labeled
        set for probability fitting.  ``slope`` controls only the sharpness.
        """

        threshold_value = _finite_float(raw_threshold, name="raw_threshold")
        slope_value = _finite_float(slope, name="slope")
        if slope_value <= 0.0:
            raise ValueError("slope must be strictly positive")
        bias = -slope_value * threshold_value
        if not np.isfinite(bias):
            raise ValueError("raw_threshold and slope produce a non-finite bias")
        return cls(
            slope=slope_value,
            bias=float(bias),
            score_name=score_name,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the stable JSON object for this schema version."""

        return {
            "schema_version": CALIBRATION_SCHEMA_VERSION,
            "calibrator_type": CALIBRATOR_TYPE,
            "score_name": self.score_name,
            "parameters": {
                "slope": self.slope,
                "bias": self.bias,
            },
            "fit": None if self.fit is None else self.fit.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: object) -> "PositiveAffineCalibrator":
        """Validate and construct a calibrator from its JSON object."""

        if not isinstance(value, Mapping):
            raise ValueError("calibrator JSON root must be a mapping")
        version = value.get("schema_version")
        if (
            isinstance(version, bool)
            or not isinstance(version, (int, np.integer))
            or int(version) != CALIBRATION_SCHEMA_VERSION
        ):
            raise ValueError(
                f"unsupported calibration schema version {version!r}; "
                f"expected {CALIBRATION_SCHEMA_VERSION}"
            )
        if value.get("calibrator_type") != CALIBRATOR_TYPE:
            raise ValueError(
                f"unsupported calibrator type {value.get('calibrator_type')!r}; "
                f"expected {CALIBRATOR_TYPE!r}"
            )
        parameters = value.get("parameters")
        if not isinstance(parameters, Mapping):
            raise ValueError("calibrator parameters must be a mapping")
        fit_value = value.get("fit")
        fit = None if fit_value is None else CalibrationFit.from_dict(fit_value)
        score_name = value.get("score_name")
        if not isinstance(score_name, str):
            raise ValueError("score_name must be a non-empty string")
        return cls(
            slope=_finite_float(parameters.get("slope"), name="calibrator slope"),
            bias=_finite_float(parameters.get("bias"), name="calibrator bias"),
            score_name=score_name,
            fit=fit,
        )

    def save_json(self, path: str | Path) -> Path:
        """Write a deterministic, versioned JSON artifact and return its path."""

        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(
            self.to_dict(),
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        destination.write_text(payload + "\n", encoding="utf-8")
        return destination

    @classmethod
    def load_json(cls, path: str | Path) -> "PositiveAffineCalibrator":
        """Load and validate a versioned JSON artifact."""

        source = Path(path)
        try:
            value = json.loads(source.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid calibrator JSON in {source}: {exc.msg}") from exc
        return cls.from_dict(value)


def _objective(
    standardized_scores: np.ndarray,
    labels: np.ndarray,
    parameters: np.ndarray,
    *,
    l2: float,
) -> float:
    logits = parameters[0] * standardized_scores + parameters[1]
    data_loss = np.mean(np.logaddexp(0.0, logits) - labels * logits)
    return float(data_loss + 0.5 * l2 * parameters[0] ** 2)


def fit_positive_affine_calibrator(
    raw_scores: np.ndarray | Sequence[float],
    labels: np.ndarray | Sequence[int],
    *,
    score_name: str = "qbyt_raw_logit",
    l2: float = 1.0e-2,
    max_iterations: int = 100,
    tolerance: float = 1.0e-9,
) -> PositiveAffineCalibrator:
    """Fit monotone Platt scaling to labeled held-out raw scores.

    A damped projected-Newton optimizer minimizes binary cross entropy in a
    standardized score coordinate.  L2 applies only to the standardized slope;
    the intercept remains unregularized.  The result is converted back to the
    caller's original score coordinate and needs only NumPy at runtime.
    """

    scores = _as_raw_scores(raw_scores, allow_scalar=False)
    targets = _as_binary_labels(labels, size=scores.size)
    if not isinstance(score_name, str) or not score_name.strip():
        raise ValueError("score_name must be a non-empty string")
    if not np.isfinite(l2) or l2 < 0.0:
        raise ValueError("l2 must be finite and non-negative")
    if isinstance(max_iterations, bool) or max_iterations <= 0:
        raise ValueError("max_iterations must be positive")
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be finite and positive")

    mean = float(scores.mean())
    scale = float(scores.std())
    if scale <= np.finfo(np.float64).eps:
        scale = 1.0
    standardized = (scores - mean) / scale

    prevalence = float(targets.mean())
    parameters = np.asarray(
        [1.0, np.log(prevalence / (1.0 - prevalence))],
        dtype=np.float64,
    )
    converged = False
    completed_iterations = 0

    for iteration in range(1, int(max_iterations) + 1):
        completed_iterations = iteration
        logits = parameters[0] * standardized + parameters[1]
        probabilities = sigmoid(logits)
        residual = probabilities - targets
        gradient = np.asarray(
            [
                float(np.mean(residual * standardized)) + l2 * parameters[0],
                float(np.mean(residual)),
            ],
            dtype=np.float64,
        )

        # At the positive-slope boundary, a positive slope gradient satisfies
        # the KKT condition because decreasing the slope is infeasible.
        projected_gradient = gradient.copy()
        if parameters[0] <= _MIN_STANDARDIZED_SLOPE and gradient[0] > 0.0:
            projected_gradient[0] = 0.0
        if float(np.max(np.abs(projected_gradient))) <= tolerance:
            converged = True
            break

        curvature = probabilities * (1.0 - probabilities)
        hessian = np.asarray(
            [
                [
                    float(np.mean(curvature * standardized * standardized)) + l2,
                    float(np.mean(curvature * standardized)),
                ],
                [
                    float(np.mean(curvature * standardized)),
                    float(np.mean(curvature)),
                ],
            ],
            dtype=np.float64,
        )
        hessian += np.eye(2, dtype=np.float64) * 1.0e-12
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(hessian, gradient, rcond=None)[0]

        old_objective = _objective(standardized, targets, parameters, l2=l2)
        step_scale = 1.0
        accepted = False
        while step_scale >= 2.0**-20:
            candidate = parameters - step_scale * step
            candidate[0] = max(candidate[0], _MIN_STANDARDIZED_SLOPE)
            candidate_objective = _objective(
                standardized,
                targets,
                candidate,
                l2=l2,
            )
            if (
                np.isfinite(candidate_objective)
                and candidate_objective <= old_objective
            ):
                parameters = candidate
                accepted = True
                break
            step_scale *= 0.5
        if not accepted:
            break
        if float(np.max(np.abs(step_scale * step))) <= tolerance:
            converged = True
            break

    raw_slope = float(parameters[0] / scale)
    raw_bias = float(parameters[1] - raw_slope * mean)
    if not np.isfinite(raw_slope) or raw_slope <= 0.0 or not np.isfinite(raw_bias):
        raise RuntimeError("calibrator fit produced invalid coefficients")

    return PositiveAffineCalibrator(
        slope=raw_slope,
        bias=raw_bias,
        score_name=score_name,
        fit=CalibrationFit(
            num_samples=int(targets.size),
            num_positive=int(targets.sum()),
            l2=float(l2),
            iterations=completed_iterations,
            converged=converged,
        ),
    )


__all__ = [
    "CALIBRATION_SCHEMA_VERSION",
    "CALIBRATOR_TYPE",
    "CalibrationFit",
    "PositiveAffineCalibrator",
    "fit_positive_affine_calibrator",
    "sigmoid",
]
