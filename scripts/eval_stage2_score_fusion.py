#!/usr/bin/env python3
"""Fit Stage II score calibration on one set and evaluate on another.

This script never runs the acoustic model.  It consumes the detailed JSONL
written by ``eval_stage2_clips.py``, fits regularized affine transforms using
only ``--calibration-results``, freezes them, and applies them to
``--evaluation-results``. Exact audio overlap and score-provenance mismatch are
always rejected. Composite group fields are required by default so augmented
TTS variants cannot silently cross the split.
"""

from __future__ import annotations

import argparse
import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from dma_kws.inference.score_calibration import (
    AffineLogitCalibrator,
    fit_affine_logit_calibrator,
    sigmoid,
)
from dma_kws.training.score_diagnostics import binary_score_diagnostics


_FEATURE_NAMES = ("qbyt_logit", "completion_logit")


@dataclass(frozen=True)
class ScoreDataset:
    path: Path
    summary_path: Path
    provenance: dict[str, Any]
    records: list[dict[str, Any]]
    eligible_indices: tuple[int, ...]
    features: np.ndarray
    labels: np.ndarray
    sample_paths: frozenset[str]
    groups: frozenset[tuple[str, ...]]
    num_skipped: int


def _resolve_results_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_dir():
        path = path / "results.jsonl"
    if not path.is_file():
        raise ValueError(f"results.jsonl not found: {path}")
    return path.resolve()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSON at {path}:{line_number}: {exc}") from exc
                if not isinstance(record, dict):
                    raise ValueError(f"expected a JSON object at {path}:{line_number}")
                records.append(record)
    except OSError as exc:
        raise ValueError(f"failed to read {path}: {exc}") from exc
    if not records:
        raise ValueError(f"no result rows found in {path}")
    return records


def _read_score_provenance(results_path: Path) -> tuple[Path, dict[str, Any]]:
    summary_path = results_path.parent / "summary.json"
    if not summary_path.is_file():
        raise ValueError(
            f"score provenance summary not found beside {results_path}: {summary_path}. "
            "Re-run eval_stage2_clips.py with the current exporter."
        )
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"failed to read score summary {summary_path}: {exc}") from exc
    if not isinstance(summary, dict) or not isinstance(summary.get("provenance"), dict):
        raise ValueError(
            f"{summary_path} has no score provenance; re-run eval_stage2_clips.py"
        )
    provenance = summary["provenance"]
    required = {
        "schema_version",
        "checkpoint",
        "qbyt_readout_mode",
        "stream",
        "audio_padding_ms",
        "fbank",
        "tokenizer",
        "sequence_objective",
    }
    missing = sorted(required - set(provenance))
    if missing:
        raise ValueError(f"{summary_path} provenance is missing fields: {missing}")
    if int(provenance["schema_version"]) != 1:
        raise ValueError(
            f"unsupported score provenance version {provenance['schema_version']!r} "
            f"in {summary_path}"
        )
    for section, section_fields in {
        "checkpoint": ("path", "size_bytes", "sha256"),
        "tokenizer": ("path", "size_bytes", "sha256", "split_with_space"),
    }.items():
        value = provenance.get(section)
        if not isinstance(value, dict):
            raise ValueError(f"{summary_path} provenance {section!r} must be a mapping")
        missing_section = sorted(set(section_fields) - set(value))
        if missing_section:
            raise ValueError(
                f"{summary_path} provenance {section!r} is missing: {missing_section}"
            )
    return summary_path.resolve(), provenance


def _semantic_provenance(provenance: dict[str, Any]) -> dict[str, Any]:
    """Drop location-only paths while retaining content identities."""

    value = json.loads(json.dumps(provenance, allow_nan=False))
    checkpoint = value.get("checkpoint")
    if isinstance(checkpoint, dict):
        checkpoint.pop("path", None)
    tokenizer = value.get("tokenizer")
    if isinstance(tokenizer, dict):
        tokenizer.pop("path", None)
    return value


def _assert_same_provenance(
    calibration: ScoreDataset,
    evaluation: ScoreDataset,
) -> None:
    calibration_value = _semantic_provenance(calibration.provenance)
    evaluation_value = _semantic_provenance(evaluation.provenance)
    if calibration_value != evaluation_value:
        raise ValueError(
            "calibration/evaluation score provenance mismatch; logits must come "
            "from the same checkpoint content, readout, stream, padding, fbank, "
            "tokenizer and sequence objective. "
            f"calibration={calibration.summary_path}, evaluation={evaluation.summary_path}"
        )


def _nested_value(record: dict[str, Any], dotted_field: str, *, context: str) -> Any:
    value: Any = record
    for component in dotted_field.split("."):
        if not component or not isinstance(value, dict) or component not in value:
            raise ValueError(f"{context} is missing group field {dotted_field!r}")
        value = value[component]
    if value is None or isinstance(value, (dict, list)) or str(value).strip() == "":
        raise ValueError(
            f"{context} group field {dotted_field!r} must be a non-empty scalar"
        )
    return value


def _sample_path(record: dict[str, Any], *, base_dir: Path, context: str) -> str:
    audio_path = record.get("audio_path")
    keyword = record.get("keyword")
    if not isinstance(audio_path, str) or not audio_path.strip():
        raise ValueError(f"{context} is missing a non-empty audio_path")
    if not isinstance(keyword, str) or not keyword.strip():
        raise ValueError(f"{context} is missing a non-empty keyword")
    path = Path(audio_path).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return str(path.resolve())


def _finite_float(record: dict[str, Any], field: str, *, context: str) -> float:
    if field not in record or record[field] is None:
        raise ValueError(f"{context} is missing raw logit field {field!r}")
    try:
        value = float(record[field])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context} has invalid {field}={record[field]!r}") from exc
    if not math.isfinite(value):
        raise ValueError(f"{context} has non-finite {field}={value!r}")
    return value


def _load_score_dataset(path: Path, *, group_fields: Sequence[str]) -> ScoreDataset:
    records = _read_jsonl(path)
    summary_path, provenance = _read_score_provenance(path)
    all_sample_paths: set[str] = set()
    groups: set[tuple[str, ...]] = set()
    eligible_indices: list[int] = []
    features: list[tuple[float, float]] = []
    labels: list[int] = []
    num_skipped = 0

    for index, record in enumerate(records):
        context = f"{path}:{index + 1}"
        sample_path = _sample_path(record, base_dir=path.parent, context=context)
        if sample_path in all_sample_paths:
            raise ValueError(
                f"duplicate resolved audio_path sample inside {path}: {sample_path}"
            )
        all_sample_paths.add(sample_path)
        if group_fields:
            groups.add(
                tuple(
                    str(_nested_value(record, field, context=context))
                    for field in group_fields
                )
            )

        if bool(record.get("skipped", False)):
            num_skipped += 1
            continue

        if "label" not in record:
            raise ValueError(f"{context} is missing label")
        try:
            label = int(record["label"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{context} has invalid label={record['label']!r}") from exc
        if label not in (0, 1):
            raise ValueError(f"{context} label must be 0 or 1, got {label}")

        qbyt_logit = _finite_float(record, "qbyt_logit", context=context)
        completion_logit = _finite_float(record, "completion_logit", context=context)
        qbyt_score = _finite_float(record, "qbyt_score", context=context)
        completion_score = _finite_float(record, "completion_score", context=context)
        expected_qbyt_score = float(sigmoid(np.asarray([qbyt_logit]))[0])
        if not math.isclose(
            qbyt_score,
            expected_qbyt_score,
            rel_tol=1.0e-5,
            abs_tol=1.0e-6,
        ):
            raise ValueError(
                f"{context} qbyt_score is inconsistent with sigmoid(qbyt_logit)"
            )
        expected_completion_score = float(
            sigmoid(np.asarray([completion_logit]))[0]
        )
        if not math.isclose(
            completion_score,
            expected_completion_score,
            rel_tol=1.0e-5,
            abs_tol=1.0e-6,
        ):
            raise ValueError(
                f"{context} completion_score is inconsistent with "
                "sigmoid(completion_logit)"
            )

        eligible_indices.append(index)
        features.append((qbyt_logit, completion_logit))
        labels.append(label)

    if not eligible_indices:
        raise ValueError(f"{path} contains no non-skipped labeled score rows")
    label_array = np.asarray(labels, dtype=np.int64)
    if np.unique(label_array).size != 2:
        raise ValueError(f"{path} must contain both positive and negative labels")
    return ScoreDataset(
        path=path,
        summary_path=summary_path,
        provenance=provenance,
        records=records,
        eligible_indices=tuple(eligible_indices),
        features=np.asarray(features, dtype=np.float64),
        labels=label_array,
        sample_paths=frozenset(all_sample_paths),
        groups=frozenset(groups),
        num_skipped=num_skipped,
    )


def _assert_disjoint(
    calibration: ScoreDataset,
    evaluation: ScoreDataset,
    *,
    group_fields: Sequence[str],
) -> None:
    sample_overlap = calibration.sample_paths & evaluation.sample_paths
    if sample_overlap:
        preview = sorted(sample_overlap)[:3]
        raise ValueError(
            "calibration/evaluation sample leakage: overlapping resolved "
            f"audio_path values, e.g. {preview}"
        )
    if group_fields:
        group_overlap = calibration.groups & evaluation.groups
        if group_overlap:
            preview = sorted(group_overlap)[:5]
            raise ValueError(
                "calibration/evaluation group leakage for composite fields "
                f"{list(group_fields)}: {preview}"
            )


def _completion_objective_is_safe(provenance: dict[str, Any]) -> bool:
    objective = provenance.get("sequence_objective")
    if not isinstance(objective, dict):
        return False
    try:
        completion_weight = float(objective.get("completion_weight", 0.0))
    except (TypeError, ValueError):
        return False
    return (
        str(objective.get("target_mode", "")) == "ordered_contiguous_prefix"
        and completion_weight > 0.0
    )


def _json_scalar(value: torch.Tensor) -> int | float | bool | None:
    item = value.detach().cpu().item()
    if isinstance(item, bool):
        return item
    if isinstance(item, int):
        return item
    number = float(item)
    return number if math.isfinite(number) else None


def _diagnostics(
    probabilities: np.ndarray,
    labels: np.ndarray,
    *,
    threshold: float,
    ece_num_bins: int,
) -> dict[str, int | float | bool | None]:
    result = binary_score_diagnostics(
        torch.as_tensor(probabilities, dtype=torch.float64),
        torch.as_tensor(labels, dtype=torch.long),
        deployment_threshold=threshold,
        ece_num_bins=ece_num_bins,
    )
    return {name: _json_scalar(value) for name, value in result.items()}


def _fit_models(dataset: ScoreDataset, *, l2: float) -> dict[str, AffineLogitCalibrator]:
    return {
        "utterance_platt": fit_affine_logit_calibrator(
            dataset.features[:, [0]],
            dataset.labels,
            feature_names=("qbyt_logit",),
            l2=l2,
        ),
        "completion_platt": fit_affine_logit_calibrator(
            dataset.features[:, [1]],
            dataset.labels,
            feature_names=("completion_logit",),
            l2=l2,
        ),
        "utterance_completion_fusion": fit_affine_logit_calibrator(
            dataset.features,
            dataset.labels,
            feature_names=_FEATURE_NAMES,
            l2=l2,
        ),
    }


def _head_probabilities(
    features: np.ndarray,
    models: dict[str, AffineLogitCalibrator],
) -> dict[str, np.ndarray]:
    return {
        "utterance_raw": sigmoid(features[:, 0]),
        "completion_raw": sigmoid(features[:, 1]),
        "utterance_platt": models["utterance_platt"].predict_proba(features[:, [0]]),
        "completion_platt": models["completion_platt"].predict_proba(features[:, [1]]),
        "utterance_completion_fusion": models[
            "utterance_completion_fusion"
        ].predict_proba(features),
    }


def _head_metrics(
    dataset: ScoreDataset,
    models: dict[str, AffineLogitCalibrator],
    *,
    threshold: float,
    ece_num_bins: int,
) -> dict[str, dict[str, int | float | bool | None]]:
    return {
        name: _diagnostics(
            probabilities,
            dataset.labels,
            threshold=threshold,
            ece_num_bins=ece_num_bins,
        )
        for name, probabilities in _head_probabilities(dataset.features, models).items()
    }


def _enriched_evaluation_records(
    dataset: ScoreDataset,
    models: dict[str, AffineLogitCalibrator],
) -> list[dict[str, Any]]:
    records = [dict(record) for record in dataset.records]
    probabilities = _head_probabilities(dataset.features, models)
    utterance_platt_logits = models["utterance_platt"].transform_logits(
        dataset.features[:, [0]]
    )
    completion_platt_logits = models["completion_platt"].transform_logits(
        dataset.features[:, [1]]
    )
    fusion_logits = models["utterance_completion_fusion"].transform_logits(
        dataset.features
    )

    detail_fields = (
        "completion_score",
        "calibrated_qbyt_logit",
        "calibrated_qbyt_score",
        "calibrated_completion_logit",
        "calibrated_completion_score",
        "fused_logit",
        "fused_score",
    )
    for record in records:
        for field in detail_fields:
            record.setdefault(field, None)
    for position, record_index in enumerate(dataset.eligible_indices):
        record = records[record_index]
        record["completion_score"] = float(probabilities["completion_raw"][position])
        record["calibrated_qbyt_logit"] = float(utterance_platt_logits[position])
        record["calibrated_qbyt_score"] = float(
            probabilities["utterance_platt"][position]
        )
        record["calibrated_completion_logit"] = float(
            completion_platt_logits[position]
        )
        record["calibrated_completion_score"] = float(
            probabilities["completion_platt"][position]
        )
        record["fused_logit"] = float(fusion_logits[position])
        record["fused_score"] = float(
            probabilities["utterance_completion_fusion"][position]
        )
    return records


def _write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def _write_jsonl(path: Path, records: Sequence[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")


def run_analysis(
    *,
    calibration_results: str | Path,
    evaluation_results: str | Path,
    output_dir: str | Path,
    l2: float = 1.0e-2,
    threshold: float = 0.5,
    ece_num_bins: int = 15,
    group_fields: Sequence[str] = (),
    allow_ungrouped: bool = False,
    allow_unsafe_completion: bool = False,
) -> dict[str, Any]:
    """Fit on calibration rows and evaluate frozen models on disjoint rows."""

    calibration_path = _resolve_results_path(calibration_results)
    evaluation_path = _resolve_results_path(evaluation_results)
    if calibration_path == evaluation_path:
        raise ValueError("calibration_results and evaluation_results must be different files")
    if not math.isfinite(l2) or l2 < 0.0:
        raise ValueError("l2 must be a finite non-negative value")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")
    if ece_num_bins <= 0:
        raise ValueError("ece_num_bins must be positive")
    normalized_group_fields = tuple(str(field).strip() for field in group_fields)
    if any(not field for field in normalized_group_fields):
        raise ValueError("group_fields must not contain empty names")
    if len(set(normalized_group_fields)) != len(normalized_group_fields):
        raise ValueError("group_fields must be unique")
    if not normalized_group_fields and not allow_ungrouped:
        raise ValueError(
            "at least one --group-field is required to prevent augmented/speaker "
            "leakage; pass --allow-ungrouped only for a deliberately pre-grouped "
            "independent dataset"
        )

    calibration = _load_score_dataset(
        calibration_path,
        group_fields=normalized_group_fields,
    )
    evaluation = _load_score_dataset(
        evaluation_path,
        group_fields=normalized_group_fields,
    )
    _assert_same_provenance(calibration, evaluation)
    _assert_disjoint(
        calibration,
        evaluation,
        group_fields=normalized_group_fields,
    )
    fit_warnings: list[str] = []
    if not normalized_group_fields:
        message = (
            "Group leakage checks were explicitly disabled. Exact resolved audio "
            "paths are still disjoint, but copied/augmented variants are not detectable."
        )
        warnings.warn(message, RuntimeWarning, stacklevel=2)
        fit_warnings.append(message)
    if not _completion_objective_is_safe(calibration.provenance):
        objective = calibration.provenance.get("sequence_objective")
        message = (
            "The checkpoint sequence objective did not train a full-keyword "
            f"completion endpoint: {objective!r}. Completion fusion is unsafe."
        )
        if not allow_unsafe_completion:
            raise ValueError(
                message + " Pass --allow-unsafe-completion only for a named legacy ablation."
            )
        warnings.warn(message, RuntimeWarning, stacklevel=2)
        fit_warnings.append(message)

    models = _fit_models(calibration, l2=l2)
    for name, model in models.items():
        if not model.converged:
            message = (
                f"Calibrator {name!r} did not converge after "
                f"{model.fit_iterations} iterations; coefficients are diagnostic only."
            )
            warnings.warn(message, RuntimeWarning, stacklevel=2)
            fit_warnings.append(message)
    calibration_metrics = _head_metrics(
        calibration,
        models,
        threshold=threshold,
        ece_num_bins=ece_num_bins,
    )
    evaluation_metrics = _head_metrics(
        evaluation,
        models,
        threshold=threshold,
        ece_num_bins=ece_num_bins,
    )

    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    calibrators_path = output_path / "score_fusion_calibrators.json"
    results_path = output_path / "score_fusion_results.jsonl"
    summary_path = output_path / "score_fusion_summary.json"

    calibrator_payload = {
        "schema_version": 1,
        "fit_source": str(calibration.path),
        "l2": float(l2),
        "fit_positive_prevalence": float(calibration.labels.mean()),
        "score_provenance": calibration.provenance,
        "warnings": fit_warnings,
        "models": {name: model.to_dict() for name, model in models.items()},
    }
    _write_json(calibrators_path, calibrator_payload)
    _write_jsonl(results_path, _enriched_evaluation_records(evaluation, models))

    summary: dict[str, Any] = {
        "schema_version": 1,
        "calibration_results": str(calibration.path),
        "evaluation_results": str(evaluation.path),
        "group_fields": list(normalized_group_fields),
        "allow_ungrouped": bool(allow_ungrouped),
        "allow_unsafe_completion": bool(allow_unsafe_completion),
        "score_provenance": calibration.provenance,
        "leakage_checks": {
            "paths_distinct": True,
            "provenance_match": True,
            "sample_overlap": 0,
            "group_overlap": 0 if normalized_group_fields else None,
        },
        "calibration_population": {
            "num_samples": int(calibration.labels.size),
            "num_positive": int(calibration.labels.sum()),
            "num_negative": int(calibration.labels.size - calibration.labels.sum()),
            "positive_prevalence": float(calibration.labels.mean()),
            "num_skipped_excluded": calibration.num_skipped,
        },
        "evaluation_population": {
            "num_samples": int(evaluation.labels.size),
            "num_positive": int(evaluation.labels.sum()),
            "num_negative": int(evaluation.labels.size - evaluation.labels.sum()),
            "num_skipped_excluded": evaluation.num_skipped,
        },
        "calibration_metrics": calibration_metrics,
        "evaluation_metrics": evaluation_metrics,
        "models": {name: model.to_dict() for name, model in models.items()},
        "fit_warnings": fit_warnings,
        "protocol_warning": (
            "Calibration metrics are in-sample. Evaluation labels are used only for "
            "reporting, but choosing a head or L2 value from evaluation_metrics makes "
            "this a selection/dev set, not a final test. The fitted intercept reflects "
            "the calibration set's positive prevalence and is not a deployment KWS "
            "posterior unless that prevalence is representative."
        ),
        "outputs": {
            "calibrators": str(calibrators_path),
            "evaluation_results": str(results_path),
        },
    }
    _write_json(summary_path, summary)
    summary["outputs"]["summary"] = str(summary_path)
    # Rewrite once so the on-disk summary is identical to the returned/printed
    # value, including its own discoverable path.
    _write_json(summary_path, summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--calibration-results",
        type=Path,
        required=True,
        help="Detailed calibration results.jsonl; labels are used for fitting.",
    )
    parser.add_argument(
        "--allow-ungrouped",
        action="store_true",
        help=(
            "Explicitly waive composite group checks. Exact resolved audio paths "
            "remain disjoint, but copied/augmented variants cannot be detected."
        ),
    )
    parser.add_argument(
        "--allow-unsafe-completion",
        action="store_true",
        help=(
            "Allow completion fusion for a checkpoint whose saved objective is "
            "legacy membership or has completion_weight<=0."
        ),
    )
    parser.add_argument(
        "--evaluation-results",
        type=Path,
        required=True,
        help="Independent detailed results.jsonl; labels are used only for reporting.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--l2",
        type=float,
        default=1.0e-2,
        help="L2 strength on standardized slopes (default: 0.01).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Probability operating point used only for diagnostics.",
    )
    parser.add_argument("--ece-num-bins", type=int, default=15)
    parser.add_argument(
        "--group-field",
        dest="group_fields",
        action="append",
        default=[],
        help=(
            "Dotted field used in a composite leakage group; repeat for provider+voice, "
            "e.g. --group-field manifest_meta.tts_provider --group-field "
            "manifest_meta.voice_id."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        summary = run_analysis(
            calibration_results=args.calibration_results,
            evaluation_results=args.evaluation_results,
            output_dir=args.output_dir,
            l2=args.l2,
            threshold=args.threshold,
            ece_num_bins=args.ece_num_bins,
            group_fields=args.group_fields,
            allow_ungrouped=args.allow_ungrouped,
            allow_unsafe_completion=args.allow_unsafe_completion,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
