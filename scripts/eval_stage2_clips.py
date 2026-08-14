#!/usr/bin/env python3
"""Evaluate Stage II-only DMA-KWS inference on a manifest of keyword clips.

Manifest rows may provide ``keyword_phonemes`` as a space-separated ARPAbet
string (or a string array in JSONL) to override keyword G2P for that row. Rows
without the field retain automatic G2P. ``text_variant_phonemes`` supports the
same override for the query reference; otherwise ``text_variant`` is converted
automatically. When a query reference is available, the JSONL also records
position-level sequence targets and raw per-sample diagnostic losses using the
sequence objective saved in the checkpoint.

Each clip receives 160 ms of zero-valued waveform context on both sides by
default. Override with ``+prep.left_padding_ms=...`` and
``+prep.right_padding_ms=...``; use zero to disable either side.
Padding is part of the scored model input and counts toward its minimum length.
When valid positive and negative labels are present, the output directory also
receives ``roc_curve.png`` and ``det_curve.png`` for the utterance QbyT score.
Optional ``prep.musan_mix`` settings add deterministic MUSAN noise, music and/or
overlapping speech in memory before the existing zero-valued padding.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from statistics import NormalDist
from typing import Any

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

from dma_kws.config import require_sections
from dma_kws.hydra_app import CONFIG_DIR, resolved_config
from dma_kws.inference.manifest import load_manifest
from dma_kws.inference.metrics import summarize_labeled_results
from dma_kws.inference.musan_mix import MusanWaveformMixer
from dma_kws.inference.stage2_clip import Stage2ClipRunner
from dma_kws.inference.stage2_reporting import (
    build_result_record as _result_record,
    build_score_provenance as _score_provenance,
)
from dma_kws.training.score_diagnostics import binary_score_diagnostics
from dma_kws.training.device import resolve_accelerator


DEFAULT_PADDING_MS = 160
DEFAULT_PLOT_DPI = 160


def _binary_roc_points(
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


def _write_detection_plots(
    records: list[dict],
    *,
    output_dir: Path,
    threshold: float,
    metrics: dict[str, float],
    dpi: int = DEFAULT_PLOT_DPI,
) -> dict[str, Any]:
    """Write ROC and normal-deviate DET plots for the utterance score."""

    score_field = "qbyt_score"
    curve = _binary_roc_points(records, score_field=score_field)
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


def _metrics_record(record: dict) -> dict:
    metrics_row = dict(record)
    metrics_row["best_qbyt_score"] = float(record.get("qbyt_score", 0.0))
    return metrics_row


def _score_head_diagnostics(
    records: list[dict],
    *,
    score_field: str,
    threshold: float,
    ece_num_bins: int,
) -> dict:
    import torch

    usable = [
        record
        for record in records
        if "label" in record
        and not bool(record.get("skipped", False))
        and record.get(score_field) is not None
    ]
    if not usable:
        return {}
    scores = torch.tensor(
        [float(record[score_field]) for record in usable],
        dtype=torch.float64,
    )
    labels = torch.tensor(
        [int(record["label"]) for record in usable],
        dtype=torch.long,
    )
    diagnostics = binary_score_diagnostics(
        scores,
        labels,
        deployment_threshold=threshold,
        ece_num_bins=ece_num_bins,
    )
    result = {}
    for name, value in diagnostics.items():
        item = value.item() if value.numel() == 1 else value.detach().cpu().tolist()
        if isinstance(item, float) and not math.isfinite(item):
            item = None
        result[name] = item
    return result


def _resolve_audio_padding_ms(prep: dict) -> tuple[int, int]:
    left_padding_ms = int(prep.get("left_padding_ms", DEFAULT_PADDING_MS))
    right_padding_ms = int(prep.get("right_padding_ms", DEFAULT_PADDING_MS))
    if left_padding_ms < 0 or right_padding_ms < 0:
        raise SystemExit("prep.left_padding_ms and prep.right_padding_ms must be >= 0")
    return left_padding_ms, right_padding_ms


def run_eval(cfg: DictConfig) -> dict:
    try:
        import torch
    except ImportError as exc:
        raise SystemExit(
            "Missing torch/torchaudio. Install CUDA PyTorch on the remote training machine first."
        ) from exc

    torch.multiprocessing.set_sharing_strategy("file_system")

    config = resolved_config(cfg)
    require_sections(config, ["paths", "stage1", "stage2", "demo", "tokenizer"])
    prep = OmegaConf.to_container(cfg.prep, resolve=True)
    if not isinstance(prep, dict):
        prep = {}
    run_cfg = cfg.run

    manifest_path = str(prep.get("manifest", ""))
    if not manifest_path:
        raise SystemExit("prep.manifest is required")
    stage2_ckpt = str(prep.get("stage2_ckpt", ""))
    if not stage2_ckpt:
        raise SystemExit("prep.stage2_ckpt is required")
    left_padding_ms, right_padding_ms = _resolve_audio_padding_ms(prep)
    output_dir_override = str(prep.get("output_dir", ""))
    if output_dir_override == "outputs/eval_two_stage_kws":
        output_dir_override = ""
    output_dir = Path(str(output_dir_override or prep.get("stage2_clip_output_dir", "outputs/eval_stage2_clips")))
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_manifest(manifest_path)
    try:
        musan_mixer = MusanWaveformMixer.from_prep(
            prep,
            audio_paths=[str(row["audio_path"]) for row in rows],
        )
    except (OSError, TypeError, ValueError) as exc:
        raise SystemExit(f"Invalid prep.musan_mix configuration: {exc}") from exc
    accelerator, _ = resolve_accelerator(str(run_cfg.device))
    device = torch.device(accelerator if accelerator == "cpu" else "cuda")
    runner = Stage2ClipRunner.from_config(config, prep, device)

    batch_size = int(prep.get("batch_size", 0) or 0)
    if batch_size <= 0:
        batch_size = 64
    num_workers = int(prep.get("num_workers", 0) or 0)
    if num_workers <= 0:
        num_workers = min(8, os.cpu_count() or 1)

    stream_description = runner.stream_policy.describe()
    provenance = _score_provenance(
        config,
        checkpoint_path=stage2_ckpt,
        stream=stream_description,
        left_padding_ms=left_padding_ms,
        right_padding_ms=right_padding_ms,
    )
    sequence_objective = provenance["sequence_objective"]

    runner_results = runner.run_batch(
        rows,
        batch_size=batch_size,
        num_workers=num_workers,
        left_padding_ms=left_padding_ms,
        right_padding_ms=right_padding_ms,
        waveform_transform=musan_mixer if musan_mixer.enabled else None,
        include_score_details=True,
        include_eps_positions=True,
        include_seq_positions=True,
    )
    results = []
    for index, (row, runner_result) in enumerate(zip(rows, runner_results)):
        record = _result_record(
            row,
            runner_result,
            sequence_objective=sequence_objective,
            qbyt_readout=provenance["qbyt_readout"],
        )
        if musan_mixer.enabled:
            record["musan_mix"] = musan_mixer.recipe_metadata(index)
        results.append(record)

    results_path = output_dir / "results.jsonl"
    with results_path.open("w", encoding="utf-8") as handle:
        for record in results:
            handle.write(
                json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n"
            )

    summary = {
        "manifest": str(Path(manifest_path).resolve()),
        "num_samples": len(results),
        "output_dir": str(output_dir.resolve()),
        "audio_padding_ms": {
            "left": left_padding_ms,
            "right": right_padding_ms,
        },
        "musan_mix": musan_mixer.summary(),
        "stream": stream_description,
        "num_skipped": sum(bool(record.get("skipped", False)) for record in results),
        "provenance": provenance,
    }
    scored_results = [record for record in results if not record.get("skipped", False)]
    deployment_threshold = float(runner._demo_cfg.get("qbyt_threshold", 0.5))
    validation_cfg = (config.get("stage2", {}) or {}).get("validation", {}) or {}
    completion_threshold = float(
        validation_cfg.get("seq_diagnostic_threshold", 0.5)
    )
    ece_num_bins = int(validation_cfg.get("ece_num_bins", 15))
    summary["score_diagnostic_config"] = {
        "utterance_threshold": deployment_threshold,
        "completion_threshold": completion_threshold,
        "ece_num_bins": ece_num_bins,
    }
    labeled_summary = summarize_labeled_results(
        [_metrics_record(record) for record in scored_results],
        threshold=deployment_threshold,
    )
    if labeled_summary:
        summary["metrics"] = labeled_summary
        summary["score_heads"] = {
            "utterance": _score_head_diagnostics(
                results,
                score_field="qbyt_score",
                threshold=deployment_threshold,
                ece_num_bins=ece_num_bins,
            ),
            "completion": _score_head_diagnostics(
                results,
                score_field="completion_score",
                threshold=completion_threshold,
                ece_num_bins=ece_num_bins,
            ),
        }
        if bool(prep.get("plot_curves", True)):
            summary["plots"] = _write_detection_plots(
                results,
                output_dir=output_dir,
                threshold=deployment_threshold,
                metrics=labeled_summary,
                dpi=int(prep.get("plot_dpi", DEFAULT_PLOT_DPI)),
            )

    summary_path = output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(
            summary,
            handle,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )

    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))
    return summary


@hydra.main(version_base=None, config_path=str(CONFIG_DIR), config_name="config")
def main(cfg: DictConfig) -> None:
    run_eval(cfg)


if __name__ == "__main__":
    main()
