from __future__ import annotations

import json

import numpy as np
import pytest

from dma_kws.inference.score_calibration import (
    CALIBRATION_SCHEMA_VERSION,
    PositiveAffineCalibrator,
    fit_positive_affine_calibrator,
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


def test_fit_repairs_shifted_logits_and_strictly_preserves_order() -> None:
    latent = np.linspace(-3.0, 3.0, 80)
    labels = (latent + 0.35 * np.sin(np.arange(latent.size)) > 0.0).astype(int)
    shifted_logits = 3.5 * latent + 5.0

    model = fit_positive_affine_calibrator(
        shifted_logits,
        labels,
        l2=0.05,
    )
    calibrated_logits = model.transform_logits(shifted_logits)

    assert model.fit is not None
    assert model.fit.converged
    assert model.slope > 0.0
    assert np.array_equal(np.argsort(calibrated_logits), np.argsort(shifted_logits))
    assert _log_loss(model.predict_proba(shifted_logits), labels) < _log_loss(
        sigmoid(shifted_logits), labels
    )


def test_fit_keeps_positive_slope_when_scores_are_inversely_ranked() -> None:
    scores = np.asarray([-2.0, -1.0, 1.0, 2.0])
    labels = np.asarray([1, 1, 0, 0])

    model = fit_positive_affine_calibrator(scores, labels, l2=0.1)

    assert model.slope > 0.0
    assert model.predict_one(2.0) > model.predict_one(-2.0)


def test_operating_threshold_maps_exactly_to_probability_half() -> None:
    model = PositiveAffineCalibrator.from_operating_threshold(0.927, slope=3.5)

    assert model.raw_operating_threshold == pytest.approx(0.927)
    assert model.transform_one(0.927) == pytest.approx(0.0)
    assert model.predict_one(0.927) == pytest.approx(0.5)
    assert model.predict_one(0.928) > 0.5
    assert model.predict_one(0.926) < 0.5


def test_fitted_slope_can_be_recentered_at_deployment_operating_point() -> None:
    fitted = fit_positive_affine_calibrator(
        [-2.0, -1.0, 1.0, 2.0],
        [0, 0, 1, 1],
    )

    deployed = fitted.at_operating_threshold(0.95)

    assert deployed.slope == fitted.slope
    assert deployed.fit is None
    assert deployed.raw_operating_threshold == pytest.approx(0.95)
    assert deployed.predict_one(0.95) == pytest.approx(0.5)


def test_json_file_round_trip_preserves_model_and_predictions(tmp_path) -> None:
    scores = np.asarray([-2.0, -1.0, 0.5, 2.0])
    labels = np.asarray([0, 0, 1, 1])
    fitted = fit_positive_affine_calibrator(scores, labels, l2=0.01)
    destination = tmp_path / "qbyt-calibration.json"

    assert fitted.save_json(destination) == destination
    loaded = PositiveAffineCalibrator.load_json(destination)

    payload = json.loads(destination.read_text(encoding="utf-8"))
    assert payload["schema_version"] == CALIBRATION_SCHEMA_VERSION
    assert payload["calibrator_type"] == "positive_affine_logit"
    assert loaded == fitted
    assert loaded.predict_proba(scores) == pytest.approx(fitted.predict_proba(scores))


@pytest.mark.parametrize("version", [0, 2, "1", True, None])
def test_json_rejects_unsupported_or_ambiguous_schema_version(version) -> None:
    payload = PositiveAffineCalibrator.from_operating_threshold(0.5).to_dict()
    payload["schema_version"] = version

    with pytest.raises(ValueError, match="unsupported calibration schema version"):
        PositiveAffineCalibrator.from_dict(payload)


@pytest.mark.parametrize(
    ("scores", "labels", "message"),
    [
        ([0.0, float("nan")], [0, 1], "finite"),
        ([0.0, 1.0], [0, 2], "only 0 and 1"),
        ([0.0, 1.0], [1, 1], "both positive and negative"),
        ([[0.0], [1.0]], [0, 1], "one-dimensional"),
    ],
)
def test_fit_rejects_invalid_training_data(scores, labels, message) -> None:
    with pytest.raises(ValueError, match=message):
        fit_positive_affine_calibrator(scores, labels)


@pytest.mark.parametrize("slope", [0.0, -1.0, float("nan"), float("inf")])
def test_calibrator_rejects_non_positive_or_non_finite_slope(slope) -> None:
    with pytest.raises(ValueError, match="slope"):
        PositiveAffineCalibrator(slope=slope, bias=0.0)


def test_scalar_helpers_reject_non_scalar_input() -> None:
    model = PositiveAffineCalibrator(slope=1.0, bias=0.0)

    with pytest.raises(ValueError, match="scalar"):
        model.predict_one([0.0, 1.0])
