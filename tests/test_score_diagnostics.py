from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from torchmetrics.utilities.data import dim_zero_cat

from dma_kws.training.score_diagnostics import (
    BinaryScoreDiagnostics,
    binary_score_diagnostics,
)
from dma_kws.inference.metrics import binary_auc, binary_eer


def _as_floats(values: dict[str, torch.Tensor]) -> dict[str, float]:
    return {name: float(value.detach().cpu()) for name, value in values.items()}


def test_perfect_separation_reports_operating_and_calibration_metrics() -> None:
    scores = torch.tensor([0.9, 0.8, 0.2, 0.1])
    targets = torch.tensor([1, 1, 0, 0])

    result = binary_score_diagnostics(
        scores,
        targets,
        deployment_threshold=0.5,
        ece_num_bins=10,
    )

    assert int(result["num_samples"]) == 4
    assert int(result["num_pos"]) == 2
    assert int(result["num_neg"]) == 2
    assert bool(result["has_both_classes"])
    assert result["auc"].item() == pytest.approx(1.0)
    assert result["eer"].item() == pytest.approx(0.0)
    assert result["eer_threshold"].item() == pytest.approx(0.8)
    assert result["tpr_at_fpr_1e_2"].item() == pytest.approx(1.0)
    assert result["tpr_at_fpr_1e_3"].item() == pytest.approx(1.0)
    assert result["pauc_fpr_1e_2"].item() == pytest.approx(1.0)
    assert result["score_pos_mean"].item() == pytest.approx(0.85)
    assert result["score_pos_p05"].item() == pytest.approx(0.805)
    assert result["score_pos_p50"].item() == pytest.approx(0.85)
    assert result["score_pos_p95"].item() == pytest.approx(0.895)
    assert result["score_neg_mean"].item() == pytest.approx(0.15)
    assert result["score_neg_p05"].item() == pytest.approx(0.105)
    assert result["score_neg_p50"].item() == pytest.approx(0.15)
    assert result["score_neg_p95"].item() == pytest.approx(0.195)
    assert result["brier"].item() == pytest.approx(0.025)
    assert result["log_loss"].item() == pytest.approx(0.164252, rel=1e-5)
    assert result["ece"].item() == pytest.approx(0.15)
    assert result["deploy_tpr"].item() == pytest.approx(1.0)
    assert result["deploy_fpr"].item() == pytest.approx(0.0)


def test_metric_accumulates_batches_and_reset_drops_previous_samples() -> None:
    metric = BinaryScoreDiagnostics(deployment_threshold=0.7, ece_num_bins=10)
    metric.update(torch.tensor([0.9, 0.2]), torch.tensor([1, 0]))
    metric.update(torch.tensor([0.8, 0.1]), torch.tensor([1, 0]))

    result = metric.compute()

    assert int(result["num_samples"]) == 4
    assert result["deploy_tpr"].item() == pytest.approx(1.0)
    assert result["deploy_fpr"].item() == pytest.approx(0.0)

    metric.reset()
    metric.update(torch.tensor([]), torch.tensor([]))
    empty = metric.compute()
    assert int(empty["num_samples"]) == 0
    assert math.isnan(float(empty["eer"]))
    assert math.isnan(float(empty["brier"]))


def test_single_class_keeps_defined_metrics_and_marks_roc_metrics_undefined() -> None:
    result = binary_score_diagnostics(
        torch.tensor([0.1, 0.9]),
        torch.tensor([1, 1]),
        deployment_threshold=0.5,
        ece_num_bins=10,
    )

    assert not bool(result["has_both_classes"])
    assert result["score_pos_mean"].item() == pytest.approx(0.5)
    assert math.isnan(float(result["score_neg_mean"]))
    assert result["brier"].item() == pytest.approx(0.41)
    assert result["ece"].item() == pytest.approx(0.5)
    assert result["deploy_tpr"].item() == pytest.approx(0.5)
    assert math.isnan(float(result["deploy_fpr"]))
    for name in (
        "auc",
        "eer",
        "eer_threshold",
        "tpr_at_fpr_1e_2",
        "tpr_at_fpr_1e_3",
        "pauc_fpr_1e_2",
    ):
        assert math.isnan(float(result[name]))


def test_empty_input_has_stable_schema() -> None:
    result = binary_score_diagnostics(torch.tensor([]), torch.tensor([]))

    expected_names = {
        "num_samples",
        "num_pos",
        "num_neg",
        "has_both_classes",
        "deploy_threshold",
        "score_pos_mean",
        "score_pos_p05",
        "score_pos_p50",
        "score_pos_p95",
        "score_neg_mean",
        "score_neg_p05",
        "score_neg_p50",
        "score_neg_p95",
        "log_loss",
        "brier",
        "ece",
        "deploy_tpr",
        "deploy_fpr",
        "auc",
        "eer",
        "eer_threshold",
        "tpr_at_fpr_1e_2",
        "tpr_at_fpr_1e_3",
        "pauc_fpr_1e_2",
    }
    assert set(result) == expected_names
    assert int(result["num_samples"]) == 0
    assert result["deploy_threshold"].item() == pytest.approx(0.5)
    for name in expected_names - {
        "num_samples",
        "num_pos",
        "num_neg",
        "has_both_classes",
        "deploy_threshold",
    }:
        assert math.isnan(float(result[name]))


@pytest.mark.parametrize(
    ("scores", "targets", "message"),
    [
        (torch.tensor([0.1]), torch.tensor([0, 1]), "same number"),
        (torch.tensor([float("nan")]), torch.tensor([0]), "finite"),
        (torch.tensor([1.1]), torch.tensor([1]), "probabilities"),
        (torch.tensor([0.1]), torch.tensor([2]), "binary labels"),
    ],
)
def test_invalid_samples_are_rejected(
    scores: torch.Tensor,
    targets: torch.Tensor,
    message: str,
) -> None:
    metric = BinaryScoreDiagnostics()
    with pytest.raises(ValueError, match=message):
        metric.update(scores, targets)


def test_constructor_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="deployment_threshold"):
        BinaryScoreDiagnostics(deployment_threshold=1.1)
    with pytest.raises(ValueError, match="ece_num_bins"):
        BinaryScoreDiagnostics(ece_num_bins=0)


def test_states_use_exact_concatenation_reduction() -> None:
    metric = BinaryScoreDiagnostics(sync_on_compute=True)

    assert metric.sync_on_compute is True
    assert metric._reductions["scores"] is dim_zero_cat
    assert metric._reductions["targets"] is dim_zero_cat
    assert metric._reductions["sample_ids"] is dim_zero_cat


def test_sample_ids_remove_distributed_sampler_padding_duplicates() -> None:
    metric = BinaryScoreDiagnostics()
    metric.update(
        torch.tensor([0.9, 0.2, 0.9]),
        torch.tensor([1, 0, 1]),
        torch.tensor([0, 1, 0]),
    )

    result = metric.compute()

    assert int(result["num_samples"]) == 2
    assert int(result["num_pos"]) == 1
    assert int(result["num_neg"]) == 1
    assert result["brier"].item() == pytest.approx(0.025)


def test_sync_on_compute_concatenates_remote_rank_samples_before_scoring() -> None:
    # TorchMetrics invokes the sync function once for each ``cat`` state.  This
    # deterministic stand-in exercises the same synchronization/reduction path
    # without opening sockets in the test process.
    remote_states = [
        torch.tensor([0.8, 0.1]),
        torch.tensor([1.0, 0.0]),
        torch.tensor([-1.0, -1.0]),
    ]
    sync_calls: list[torch.Tensor] = []

    def gather_rank_states(value: torch.Tensor, group: object = None) -> list[torch.Tensor]:
        del group
        remote = remote_states[len(sync_calls)]
        sync_calls.append(value.clone())
        return [value, remote]

    metric = BinaryScoreDiagnostics(
        deployment_threshold=0.5,
        ece_num_bins=10,
        dist_sync_fn=gather_rank_states,
        distributed_available_fn=lambda: True,
    )
    metric.update(torch.tensor([0.9, 0.2]), torch.tensor([1, 0]))

    actual = _as_floats(metric.compute())
    expected = _as_floats(
        binary_score_diagnostics(
            torch.tensor([0.9, 0.2, 0.8, 0.1]),
            torch.tensor([1, 0, 1, 0]),
            deployment_threshold=0.5,
            ece_num_bins=10,
        )
    )
    assert len(sync_calls) == 3
    assert actual.keys() == expected.keys()
    for name, expected_value in expected.items():
        if math.isnan(expected_value):
            assert math.isnan(actual[name])
        else:
            assert actual[name] == pytest.approx(expected_value)


@pytest.mark.parametrize("seed", [0, 3, 11])
def test_training_auc_and_eer_match_offline_metrics_with_ties(seed: int) -> None:
    rng = np.random.default_rng(seed)
    scores = np.round(rng.random(101), 1)
    targets = np.array([0] * 93 + [1] * 8, dtype=np.int64)
    rng.shuffle(targets)

    diagnostics = binary_score_diagnostics(
        torch.as_tensor(scores, dtype=torch.float64),
        torch.as_tensor(targets),
    )
    offline_eer, offline_threshold = binary_eer(targets, scores)

    assert float(diagnostics["auc"]) == pytest.approx(binary_auc(targets, scores))
    assert float(diagnostics["eer"]) == pytest.approx(offline_eer)
    assert float(diagnostics["eer_threshold"]) == pytest.approx(offline_threshold)
