from __future__ import annotations

import json

import numpy as np
import pytest

from dma_kws.inference.score_calibration import (
    AffineLogitCalibrator,
    fit_affine_logit_calibrator,
    sigmoid,
)


def _log_loss(probabilities: np.ndarray, labels: np.ndarray) -> float:
    probabilities = np.clip(probabilities, 1.0e-12, 1.0 - 1.0e-12)
    return float(
        -np.mean(
            labels * np.log(probabilities)
            + (1.0 - labels) * np.log(1.0 - probabilities)
        )
    )


def test_sigmoid_is_finite_for_extreme_logits() -> None:
    probabilities = sigmoid(np.asarray([-1000.0, 0.0, 1000.0]))

    assert np.isfinite(probabilities).all()
    assert probabilities.tolist() == pytest.approx([0.0, 0.5, 1.0])


def test_platt_scaling_repairs_shifted_logits_without_changing_order() -> None:
    latent = np.linspace(-3.0, 3.0, 80)
    labels = (latent + 0.35 * np.sin(np.arange(latent.size)) > 0.0).astype(int)
    shifted_logits = 3.5 * latent + 5.0

    model = fit_affine_logit_calibrator(
        shifted_logits[:, None],
        labels,
        feature_names=("qbyt_logit",),
        l2=0.05,
    )
    calibrated_logits = model.transform_logits(shifted_logits[:, None])

    assert model.converged
    assert model.weights[0] > 0.0
    assert np.array_equal(np.argsort(calibrated_logits), np.argsort(shifted_logits))
    assert _log_loss(model.predict_proba(shifted_logits[:, None]), labels) < _log_loss(
        sigmoid(shifted_logits), labels
    )


def test_two_feature_fusion_uses_completion_evidence_and_stays_finite() -> None:
    labels = np.asarray([0, 1] * 20, dtype=int)
    # The utterance logit is deliberately uninformative. Completion carries
    # all class evidence and is perfectly separable, which also exercises L2's
    # protection against infinite coefficients.
    utterance = np.zeros(labels.size, dtype=np.float64)
    completion = np.where(labels == 1, 2.0, -2.0)
    features = np.column_stack((utterance, completion))

    model = fit_affine_logit_calibrator(
        features,
        labels,
        feature_names=("qbyt_logit", "completion_logit"),
        l2=0.1,
    )
    probabilities = model.predict_proba(features)

    assert np.isfinite(np.asarray((*model.weights, model.bias))).all()
    assert abs(model.weights[1]) > abs(model.weights[0])
    assert probabilities[labels == 1].min() > probabilities[labels == 0].max()


def test_calibrator_json_round_trip_preserves_predictions() -> None:
    features = np.asarray([[-2.0], [-1.0], [0.5], [2.0]])
    labels = np.asarray([0, 0, 1, 1])
    fitted = fit_affine_logit_calibrator(
        features,
        labels,
        feature_names=("qbyt_logit",),
        l2=0.01,
    )

    serialized = json.loads(json.dumps(fitted.to_dict()))
    loaded = AffineLogitCalibrator.from_dict(serialized)

    assert loaded == fitted
    assert loaded.predict_proba(features) == pytest.approx(fitted.predict_proba(features))


@pytest.mark.parametrize(
    ("features", "labels", "message"),
    [
        ([[0.0], [float("nan")]], [0, 1], "finite"),
        ([[0.0], [1.0]], [0, 2], "only 0 and 1"),
        ([[0.0], [1.0]], [1, 1], "both positive and negative"),
    ],
)
def test_fit_rejects_invalid_training_data(features, labels, message) -> None:
    with pytest.raises(ValueError, match=message):
        fit_affine_logit_calibrator(
            features,
            labels,
            feature_names=("score",),
        )


def test_transform_rejects_wrong_feature_count() -> None:
    model = AffineLogitCalibrator(
        feature_names=("a", "b"),
        weights=(1.0, 2.0),
        bias=0.0,
        l2=0.1,
        num_samples=2,
        num_positive=1,
        fit_iterations=1,
        converged=True,
    )

    with pytest.raises(ValueError, match="expected 2 features"):
        model.predict_proba([[1.0]])
