#!/usr/bin/env python3
"""Evaluate Stage II-only DMA-KWS inference on a manifest of keyword clips."""

from __future__ import annotations

import json
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from dma_kws.config import require_sections
from dma_kws.hydra_app import CONFIG_DIR, resolved_config
from dma_kws.inference.manifest import load_manifest
from dma_kws.inference.metrics import summarize_labeled_results
from dma_kws.inference.stage2_clip import Stage2ClipRunner
from dma_kws.training.device import resolve_accelerator


def _result_record(manifest_row: dict, runner_result: dict) -> dict:
    record = {
        "audio_path": manifest_row["audio_path"],
        "keyword": manifest_row["keyword"],
        "qbyt_score": float(runner_result.get("qbyt_score", 0.0)),
        "detected": bool(runner_result["detected"]),
        "threshold": float(runner_result["threshold"]),
        "skipped": bool(runner_result.get("skipped", False)),
    }
    if "label" in manifest_row:
        record["label"] = int(manifest_row["label"])

    manifest_meta = {
        key: value
        for key, value in manifest_row.items()
        if key not in {"audio_path", "keyword", "label"}
    }
    if manifest_meta:
        record["manifest_meta"] = manifest_meta
    return record


def _metrics_record(record: dict) -> dict:
    metrics_row = dict(record)
    metrics_row["best_qbyt_score"] = float(record.get("qbyt_score", 0.0))
    return metrics_row


def run_eval(cfg: DictConfig) -> dict:
    try:
        import torch
    except ImportError as exc:
        raise SystemExit(
            "Missing torch/torchaudio. Install CUDA PyTorch on the remote training machine first."
        ) from exc

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
    output_dir_override = str(prep.get("output_dir", ""))
    if output_dir_override == "outputs/eval_two_stage_kws":
        output_dir_override = ""
    output_dir = Path(str(output_dir_override or prep.get("stage2_clip_output_dir", "outputs/eval_stage2_clips")))
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_manifest(manifest_path)
    accelerator, _ = resolve_accelerator(str(run_cfg.device))
    device = torch.device(accelerator if accelerator == "cpu" else "cuda")
    runner = Stage2ClipRunner.from_config(config, prep, device)

    results: list[dict] = []
    for row in rows:
        runner_result = runner.run(row["audio_path"], row["keyword"])
        results.append(_result_record(row, runner_result))

    results_path = output_dir / "results.jsonl"
    with results_path.open("w", encoding="utf-8") as handle:
        for record in results:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary = {
        "manifest": str(Path(manifest_path).resolve()),
        "num_samples": len(results),
        "output_dir": str(output_dir.resolve()),
    }
    labeled_summary = summarize_labeled_results(
        [_metrics_record(record) for record in results],
        threshold=float(runner._demo_cfg.get("qbyt_threshold", 0.5)),
    )
    if labeled_summary:
        summary["metrics"] = labeled_summary

    summary_path = output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


@hydra.main(version_base=None, config_path=str(CONFIG_DIR), config_name="config")
def main(cfg: DictConfig) -> None:
    run_eval(cfg)


if __name__ == "__main__":
    main()
