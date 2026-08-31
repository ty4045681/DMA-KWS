#!/usr/bin/env python3
"""Evaluate Stage II-only DMA-KWS inference on a manifest of keyword clips.

Manifest rows may provide ``keyword_phonemes`` as a space-separated ARPAbet
string (or a string array in JSONL) to override keyword G2P for that row. Rows
without the field retain automatic G2P. Every JSONL row records the single
deployed bounded-alignment QbyT score.

Each clip receives 160 ms of zero-valued waveform context on both sides by
default. Override with ``+prep.left_padding_ms=...`` and
``+prep.right_padding_ms=...``; use zero to disable either side.
Padding is part of the scored model input and counts toward its minimum length.
When valid positive and negative labels are present, the output directory also
receives ``roc_curve.png``, ``det_curve.png``, and ``roc_curve.csv`` for the
utterance QbyT score. The CSV lists every empirical ROC point as
``threshold,tpr,fpr``.
Optional ``prep.audio_aug`` and ``prep.musan_mix`` settings apply deterministic
waveform augmentation in memory before the existing zero-valued padding. MUSAN
mixing also supports stationary synthetic noise, MUSAN noise bursts and
time-varying volume without modifying source files. Stationary noise and volume
variation need no ``prep.musan_root``; burst noise uses its ``noise/**`` pool.
Optional ``prep.audio_export`` writes selected post-augmentation, pre-padding
waveforms as unclipped IEEE float WAV files beside the evaluation report.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from dma_kws.config import require_sections
from dma_kws.hydra_app import CONFIG_DIR, resolved_config
from dma_kws.inference.audio_aug import AudioAugWaveformTransform
from dma_kws.inference.audio_export import SelectedWaveformExporter
from dma_kws.inference.detection_plots import (
    DEFAULT_PLOT_DPI,
    binary_roc_points as _binary_roc_points,
    write_detection_plots as _write_detection_plots,
)
from dma_kws.inference.manifest import load_manifest
from dma_kws.inference.metrics import summarize_labeled_results
from dma_kws.inference.musan_mix import MusanWaveformMixer
from dma_kws.inference.stage2_clip import Stage2ClipRunner
from dma_kws.inference.stage2_reporting import (
    build_result_record as _result_record,
    build_score_provenance as _score_provenance,
)
from dma_kws.inference.waveform_augmentation import WaveformAugmentationPipeline
from dma_kws.training.device import resolve_accelerator
from dma_kws.training.score_diagnostics import binary_score_diagnostics


DEFAULT_PADDING_MS = 160


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
    audio_paths = [str(row["audio_path"]) for row in rows]
    try:
        audio_aug = AudioAugWaveformTransform.from_prep(
            prep,
            audio_paths=audio_paths,
        )
    except (OSError, TypeError, ValueError) as exc:
        raise SystemExit(f"Invalid prep.audio_aug configuration: {exc}") from exc
    try:
        musan_mixer = MusanWaveformMixer.from_prep(
            prep,
            audio_paths=audio_paths,
        )
    except (OSError, TypeError, ValueError) as exc:
        raise SystemExit(f"Invalid prep.musan_mix configuration: {exc}") from exc
    waveform_augmentation = WaveformAugmentationPipeline(
        audio_aug=audio_aug,
        musan_mixer=musan_mixer,
    )
    try:
        audio_exporter = SelectedWaveformExporter.from_prep(
            prep,
            output_dir=output_dir / "exported_audio",
            audio_paths=audio_paths,
        )
        audio_exporter.prepare()
    except (OSError, TypeError, ValueError) as exc:
        raise SystemExit(f"Invalid prep.audio_export configuration: {exc}") from exc
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
    runner_results = runner.run_batch(
        rows,
        batch_size=batch_size,
        num_workers=num_workers,
        left_padding_ms=left_padding_ms,
        right_padding_ms=right_padding_ms,
        waveform_transform=(
            waveform_augmentation if waveform_augmentation.enabled else None
        ),
        waveform_observer=audio_exporter if audio_exporter.enabled else None,
    )
    try:
        audio_export_summary = audio_exporter.finalize()
    except (OSError, TypeError, ValueError) as exc:
        raise SystemExit(f"Failed to finalize transformed WAV exports: {exc}") from exc
    results = []
    for index, (row, runner_result) in enumerate(zip(rows, runner_results)):
        record = _result_record(row, runner_result)
        if waveform_augmentation.enabled:
            record.update(waveform_augmentation.recipe_metadata(index))
        exported_audio_path = audio_exporter.result_path(index)
        if exported_audio_path is not None:
            record["exported_audio_path"] = exported_audio_path
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
        "stream": stream_description,
        "num_skipped": sum(bool(record.get("skipped", False)) for record in results),
        "provenance": provenance,
        "audio_exports": audio_export_summary,
    }
    summary.update(waveform_augmentation.summary())
    scored_results = [record for record in results if not record.get("skipped", False)]
    deployment_threshold = float(runner._demo_cfg.get("qbyt_threshold", 0.5))
    validation_cfg = (config.get("stage2", {}) or {}).get("validation", {}) or {}
    ece_num_bins = int(validation_cfg.get("ece_num_bins", 15))
    summary["score_diagnostic_config"] = {
        "threshold": deployment_threshold,
        "ece_num_bins": ece_num_bins,
    }
    labeled_summary = summarize_labeled_results(
        [_metrics_record(record) for record in scored_results],
        threshold=deployment_threshold,
    )
    if labeled_summary:
        summary["metrics"] = labeled_summary
        summary["score_diagnostics"] = _score_head_diagnostics(
            results,
            score_field="qbyt_score",
            threshold=deployment_threshold,
            ece_num_bins=ece_num_bins,
        )
        if bool(prep.get("plot_curves", True)):
            summary["plots"] = _write_detection_plots(
                results,
                output_dir=output_dir,
                threshold=deployment_threshold,
                metrics=labeled_summary,
                dpi=int(prep.get("plot_dpi", DEFAULT_PLOT_DPI)),
                min_recall=prep.get("plot_min_recall"),
                max_fpr=prep.get("plot_max_fpr"),
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
