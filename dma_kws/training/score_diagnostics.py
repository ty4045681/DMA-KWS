"""Exact binary-score diagnostics for validation and offline evaluation.

The metric stores probabilities and targets instead of reducing per-rank
statistics.  ``dist_reduce_fx="cat"`` therefore reconstructs the complete
validation set before non-decomposable metrics such as EER and partial AUC are
computed.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor
from torchmetrics import Metric
from torchmetrics.functional.classification import binary_auroc, binary_roc


_FPR_1E_2 = 1.0e-2
_FPR_1E_3 = 1.0e-3


def _nan(*, device: torch.device, dtype: torch.dtype) -> Tensor:
    return torch.tensor(float("nan"), device=device, dtype=dtype)


def _cat_state(state: list[Tensor] | Tensor, *, device: torch.device) -> Tensor:
    if isinstance(state, Tensor):
        return state.reshape(-1)
    if not state:
        return torch.empty(0, device=device)
    return torch.cat([value.reshape(-1) for value in state])


def _score_summary(scores: Tensor, *, nan: Tensor) -> dict[str, Tensor]:
    if scores.numel() == 0:
        return {
            "mean": nan.clone(),
            "p05": nan.clone(),
            "p50": nan.clone(),
            "p95": nan.clone(),
        }
    quantiles = torch.quantile(
        scores,
        torch.tensor([0.05, 0.50, 0.95], device=scores.device, dtype=scores.dtype),
    )
    return {
        "mean": scores.mean(),
        "p05": quantiles[0],
        "p50": quantiles[1],
        "p95": quantiles[2],
    }


def _expected_calibration_error(scores: Tensor, targets: Tensor, *, num_bins: int) -> Tensor:
    """Return fixed-width, L1 expected calibration error."""

    bin_indices = torch.clamp((scores * num_bins).long(), max=num_bins - 1)
    counts = torch.zeros(num_bins, device=scores.device, dtype=scores.dtype)
    confidence_sums = torch.zeros_like(counts)
    positive_sums = torch.zeros_like(counts)
    ones = torch.ones_like(scores)
    counts.scatter_add_(0, bin_indices, ones)
    confidence_sums.scatter_add_(0, bin_indices, scores)
    positive_sums.scatter_add_(0, bin_indices, targets.to(scores.dtype))

    occupied = counts > 0
    confidence = confidence_sums[occupied] / counts[occupied]
    accuracy = positive_sums[occupied] / counts[occupied]
    weights = counts[occupied] / scores.numel()
    return ((confidence - accuracy).abs() * weights).sum()


def _max_tpr_at_fpr(fpr: Tensor, tpr: Tensor, target_fpr: float) -> Tensor:
    """Return the best empirically attainable TPR without exceeding FPR."""

    eligible = fpr <= target_fpr
    # A ROC curve always contains (0, 0), but keep this total for robustness.
    if not bool(eligible.any()):
        return torch.zeros((), device=tpr.device, dtype=tpr.dtype)
    return tpr[eligible].max()


def binary_score_diagnostics(
    scores: Tensor,
    targets: Tensor,
    *,
    deployment_threshold: float = 0.5,
    ece_num_bins: int = 15,
) -> dict[str, Tensor]:
    """Compute exact diagnostics from binary probabilities and labels.

    ``scores`` must be probabilities in ``[0, 1]``.  Metrics that require both
    classes are returned as ``NaN`` when either class is absent.  Calibration
    metrics and statistics for an available class remain defined.
    """

    if ece_num_bins <= 0:
        raise ValueError("ece_num_bins must be positive")
    if not 0.0 <= deployment_threshold <= 1.0:
        raise ValueError("deployment_threshold must be in [0, 1]")

    scores = scores.detach().reshape(-1)
    targets = targets.detach().reshape(-1)
    if scores.numel() != targets.numel():
        raise ValueError(
            f"scores and targets must contain the same number of values, got "
            f"{scores.numel()} and {targets.numel()}"
        )
    if not scores.is_floating_point():
        scores = scores.float()
    if scores.numel() and not bool(torch.isfinite(scores).all()):
        raise ValueError("scores must contain only finite values")
    if scores.numel() and not bool(((scores >= 0.0) & (scores <= 1.0)).all()):
        raise ValueError("scores must be probabilities in [0, 1]")
    if targets.numel() and not bool(((targets == 0) | (targets == 1)).all()):
        raise ValueError("targets must contain only binary labels 0 and 1")

    targets = targets.long()
    dtype = scores.dtype
    device = scores.device
    nan = _nan(device=device, dtype=dtype)
    num_samples = scores.numel()
    positive_mask = targets == 1
    negative_mask = targets == 0
    num_positive = int(positive_mask.sum())
    num_negative = int(negative_mask.sum())
    has_both_classes = num_positive > 0 and num_negative > 0

    result: dict[str, Tensor] = {
        "num_samples": torch.tensor(num_samples, device=device, dtype=torch.long),
        "num_pos": torch.tensor(num_positive, device=device, dtype=torch.long),
        "num_neg": torch.tensor(num_negative, device=device, dtype=torch.long),
        "has_both_classes": torch.tensor(has_both_classes, device=device, dtype=torch.bool),
        "deploy_threshold": torch.tensor(deployment_threshold, device=device, dtype=dtype),
    }

    positive_summary = _score_summary(scores[positive_mask], nan=nan)
    negative_summary = _score_summary(scores[negative_mask], nan=nan)
    result.update({f"score_pos_{name}": value for name, value in positive_summary.items()})
    result.update({f"score_neg_{name}": value for name, value in negative_summary.items()})

    if num_samples:
        float_targets = targets.to(dtype)
        result["log_loss"] = torch.nn.functional.binary_cross_entropy(
            scores,
            float_targets,
        )
        result["brier"] = ((scores - float_targets) ** 2).mean()
        result["ece"] = _expected_calibration_error(
            scores,
            targets,
            num_bins=ece_num_bins,
        )
    else:
        result["log_loss"] = nan.clone()
        result["brier"] = nan.clone()
        result["ece"] = nan.clone()

    accepted = scores >= deployment_threshold
    result["deploy_tpr"] = (
        accepted[positive_mask].to(dtype).mean() if num_positive else nan.clone()
    )
    result["deploy_fpr"] = (
        accepted[negative_mask].to(dtype).mean() if num_negative else nan.clone()
    )

    roc_metric_names = (
        "auc",
        "eer",
        "eer_threshold",
        "tpr_at_fpr_1e_2",
        "tpr_at_fpr_1e_3",
        "pauc_fpr_1e_2",
    )
    result.update({name: nan.clone() for name in roc_metric_names})
    if has_both_classes:
        # ``binary_roc`` preserves tied-score operating points and gives the
        # threshold corresponding to the same point used for the EER.
        fpr, tpr, thresholds = binary_roc(
            scores,
            targets,
            thresholds=None,
            validate_args=False,
        )
        fnr = 1.0 - tpr
        eer_index = torch.argmin((fpr - fnr).abs())
        result["auc"] = binary_auroc(
            scores,
            targets,
            thresholds=None,
            validate_args=False,
        )
        result["eer"] = (fpr[eer_index] + fnr[eer_index]) / 2.0
        result["eer_threshold"] = thresholds[eer_index]
        result["tpr_at_fpr_1e_2"] = _max_tpr_at_fpr(fpr, tpr, _FPR_1E_2)
        result["tpr_at_fpr_1e_3"] = _max_tpr_at_fpr(fpr, tpr, _FPR_1E_3)
        # TorchMetrics follows the standardized McClish partial-AUC definition;
        # chance performance is 0.5 and perfect performance is 1.0.
        result["pauc_fpr_1e_2"] = binary_auroc(
            scores,
            targets,
            max_fpr=_FPR_1E_2,
            thresholds=None,
            validate_args=False,
        )
    return result


class BinaryScoreDiagnostics(Metric):
    """Accumulate binary scores and compute exact, DDP-global diagnostics.

    Args:
        deployment_threshold: Threshold used by the deployed classifier.
        ece_num_bins: Number of equal-width bins used for L1 ECE.
        sync_on_compute: Gather state across ranks before ``compute``.  Keep this
            enabled for validation metrics under DDP.
        **kwargs: Additional :class:`torchmetrics.Metric` options.
    """

    is_differentiable = False
    higher_is_better = None
    full_state_update = False

    def __init__(
        self,
        *,
        deployment_threshold: float = 0.5,
        ece_num_bins: int = 15,
        sync_on_compute: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(sync_on_compute=sync_on_compute, **kwargs)
        if not 0.0 <= deployment_threshold <= 1.0:
            raise ValueError("deployment_threshold must be in [0, 1]")
        if ece_num_bins <= 0:
            raise ValueError("ece_num_bins must be positive")
        self.deployment_threshold = float(deployment_threshold)
        self.ece_num_bins = int(ece_num_bins)

        # Targets deliberately use the same floating dtype as scores.  This
        # keeps an empty rank compatible with TorchMetrics' empty ``cat`` state
        # during variable-size distributed validation.
        self.add_state("scores", default=[], dist_reduce_fx="cat")
        self.add_state("targets", default=[], dist_reduce_fx="cat")
        # Lightning's distributed evaluation sampler may pad with repeated
        # indices so every rank has equal work. IDs let compute() remove those
        # synthetic duplicates after gathering the exact global state.
        self.add_state("sample_ids", default=[], dist_reduce_fx="cat")

    def update(
        self,
        scores: Tensor,
        targets: Tensor,
        sample_ids: Tensor | None = None,
    ) -> None:
        scores = scores.detach().reshape(-1)
        targets = targets.detach().reshape(-1)
        if scores.numel() != targets.numel():
            raise ValueError(
                f"scores and targets must contain the same number of values, got "
                f"{scores.numel()} and {targets.numel()}"
            )
        if not scores.numel():
            return
        if not scores.is_floating_point():
            scores = scores.float()
        if not bool(torch.isfinite(scores).all()):
            raise ValueError("scores must contain only finite values")
        if not bool(((scores >= 0.0) & (scores <= 1.0)).all()):
            raise ValueError("scores must be probabilities in [0, 1]")
        if not bool(((targets == 0) | (targets == 1)).all()):
            raise ValueError("targets must contain only binary labels 0 and 1")
        if sample_ids is not None:
            sample_ids = sample_ids.detach().reshape(-1)
            if sample_ids.numel() != scores.numel():
                raise ValueError(
                    "sample_ids must contain the same number of values as scores"
                )
            if not bool(torch.isfinite(sample_ids).all()):
                raise ValueError("sample_ids must contain only finite values")
            if not bool(sample_ids.ge(0).all()):
                raise ValueError("sample_ids must be non-negative")

        # Use a stable state dtype across autocast/batches and across empty DDP
        # ranks.  Metric state follows the input device automatically.
        self.scores.append(scores.to(dtype=torch.float32))
        self.targets.append(targets.to(device=scores.device, dtype=torch.float32))
        self.sample_ids.append(
            (
                sample_ids.to(device=scores.device, dtype=torch.float32)
                if sample_ids is not None
                else torch.full_like(scores, -1.0, dtype=torch.float32)
            )
        )

    def compute(self) -> dict[str, Tensor]:
        scores = _cat_state(self.scores, device=self.device).to(dtype=torch.float32)
        targets = _cat_state(self.targets, device=self.device).to(dtype=torch.float32)
        sample_ids = _cat_state(self.sample_ids, device=self.device).to(dtype=torch.float32)
        if sample_ids.numel() and bool(sample_ids.ge(0).all()):
            order = torch.argsort(sample_ids, stable=True)
            ordered_ids = sample_ids[order]
            first = torch.ones_like(ordered_ids, dtype=torch.bool)
            first[1:] = ordered_ids[1:] != ordered_ids[:-1]
            keep = order[first]
            scores = scores[keep]
            targets = targets[keep]
        return binary_score_diagnostics(
            scores,
            targets,
            deployment_threshold=self.deployment_threshold,
            ece_num_bins=self.ece_num_bins,
        )


__all__ = ["BinaryScoreDiagnostics", "binary_score_diagnostics"]
