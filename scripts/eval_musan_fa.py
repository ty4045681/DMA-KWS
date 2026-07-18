#!/usr/bin/env python3
"""Evaluate Stage-II-only DMA-KWS false accepts on the MUSAN corpus.

Slides a fixed-length window over each MUSAN audio file and scores every window
with the Stage-II QbyT verifier for a single keyword. Reports overall and
per-subset (music/noise/speech) FA/hour.

Output schema matches ``scripts/eval_stage2_clips.py``:
  - ``results.jsonl`` has one JSON object per scored window.
  - ``summary.json`` contains aggregate metrics.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import hydra
from omegaconf import DictConfig, OmegaConf

from dma_kws.config import require_sections
from dma_kws.hydra_app import CONFIG_DIR, resolved_config
from dma_kws.inference.manifest import iter_audio_files
from dma_kws.inference.metrics import summarize_false_accept_rate
from dma_kws.inference.musan_fa import (
    audio_duration_sec,
    detect_subset,
    metrics_record,
    musan_result_record,
    subset_summary,
)
from dma_kws.inference.stage2_clip import Stage2ClipRunner
from dma_kws.training.device import resolve_accelerator


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

    keyword = str(prep.get("keyword", "")).strip()
    if not keyword:
        raise SystemExit("prep.keyword is required")
    musan_root = str(prep.get("musan_root", ""))
    if not musan_root:
        raise SystemExit("prep.musan_root is required")
    musan_root_path = Path(musan_root)
    if not musan_root_path.is_dir():
        raise SystemExit(f"MUSAN root not found: {musan_root}")
    stage2_ckpt = str(prep.get("stage2_ckpt", ""))
    if not stage2_ckpt:
        raise SystemExit("prep.stage2_ckpt is required")

    window_sec = float(prep.get("window_sec", 0.0) or 3.0)
    hop_sec = float(prep.get("hop_sec", 0.0) or 1.0)
    if window_sec <= 0 or hop_sec <= 0:
        raise SystemExit("window_sec and hop_sec must be positive")

    output_dir_override = str(prep.get("output_dir", ""))
    output_dir = Path(output_dir_override or "outputs/eval_musan_fa")
    output_dir.mkdir(parents=True, exist_ok=True)

    accelerator, _ = resolve_accelerator(str(run_cfg.device))
    device = torch.device(accelerator if accelerator == "cpu" else "cuda")
    runner = Stage2ClipRunner.from_config(config, prep, device)
    threshold = float(runner._demo_cfg.get("qbyt_threshold", 0.5))

    audio_files = iter_audio_files(musan_root_path)
    if not audio_files:
        raise SystemExit(f"No audio files found under {musan_root}")

    # Keep a reproducible source manifest.
    manifest_path = output_dir / "musan_manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as manifest_handle:
        for audio_path in audio_files:
            row = {
                "audio_path": str(audio_path.resolve()),
                "keyword": keyword,
                "label": 0,
                "subset": detect_subset(audio_path, musan_root_path),
            }
            manifest_handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    # Re-load manifest so ordering and metadata are explicit.
    with manifest_path.open("r", encoding="utf-8") as manifest_handle:
        source_rows = [json.loads(line) for line in manifest_handle if line.strip()]

    results_path = output_dir / "results.jsonl"
    results_handle = results_path.open("w", encoding="utf-8")

    all_results: list[dict[str, Any]] = []
    subset_results: dict[str, list[dict[str, Any]]] = defaultdict(list)
    subset_hours: dict[str, float] = defaultdict(float)
    total_hours = 0.0

    sample_rate = int(runner._sample_rate)
    for source_row in source_rows:
        audio_path = source_row["audio_path"]
        subset = source_row["subset"]
        duration = audio_duration_sec(audio_path, sample_rate=sample_rate)
        total_hours += duration / 3600.0
        subset_hours[subset] += duration / 3600.0

        window_results = runner.run_file_windows(
            audio_path,
            keyword,
            window_sec=window_sec,
            hop_sec=hop_sec,
        )
        for window_result in window_results:
            record = musan_result_record(audio_path, keyword, subset, window_result)
            all_results.append(record)
            subset_results[subset].append(record)
            results_handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    results_handle.close()

    summary: dict[str, Any] = {
        "manifest": str(manifest_path.resolve()),
        "num_samples": len(all_results),
        "output_dir": str(output_dir.resolve()),
        "musan_root": str(musan_root_path.resolve()),
        "keyword": keyword,
        "stage2_ckpt": stage2_ckpt,
        "window_sec": window_sec,
        "hop_sec": hop_sec,
        "total_files": len(source_rows),
        "total_hours": total_hours,
    }

    overall_metrics = summarize_false_accept_rate(
        [metrics_record(record) for record in all_results],
        threshold=threshold,
        total_hours=total_hours,
    )
    if overall_metrics:
        summary["metrics"] = overall_metrics

    subsets: dict[str, dict[str, Any]] = {}
    for subset in sorted(subset_results):
        subsets[subset] = subset_summary(
            subset_results[subset],
            threshold=threshold,
            total_hours=subset_hours[subset],
        )
    if subsets:
        summary["subsets"] = subsets

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
