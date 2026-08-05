#!/usr/bin/env python3
"""Evaluate Stage-II-only DMA-KWS false accepts on the MUSAN corpus.

Slides a fixed-length window over each MUSAN audio file and scores every window
with the Stage-II QbyT verifier for a single keyword. Reports overall and
per-subset (music/noise/speech) FA/hour.

Output schema matches ``scripts/eval_stage2_clips.py``:
  - ``results.jsonl`` has one JSON object per scored window.
  - ``summary.json`` contains aggregate metrics.

Set ``prep.keyword_phonemes`` to a space-separated ARPAbet sequence (or a
Hydra list) to override keyword G2P. A missing or blank value retains automatic
G2P. These are the only two files written to ``prep.output_dir``.
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
from dma_kws.inference.stage2_reporting import build_score_provenance
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

    raw_keyword_phonemes = prep.get("keyword_phonemes")
    use_keyword_phoneme_override = raw_keyword_phonemes is not None and not (
        isinstance(raw_keyword_phonemes, str)
        and not raw_keyword_phonemes.strip()
    )
    try:
        keyword_phonemes = runner.resolve_keyword_phonemes(
            keyword,
            raw_keyword_phonemes if use_keyword_phoneme_override else None,
            field_name="prep.keyword_phonemes",
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    keyword_phonemes_source = (
        "prep.keyword_phonemes" if use_keyword_phoneme_override else "g2p"
    )

    stream_description = runner.stream_policy.describe()
    provenance = build_score_provenance(
        config,
        checkpoint_path=stage2_ckpt,
        stream=stream_description,
        # Sliding windows are scored as-is; unlike clip evaluation, this path
        # does not add zero-valued waveform context around each window.
        left_padding_ms=0,
        right_padding_ms=0,
    )
    sequence_objective = provenance["sequence_objective"]

    audio_files = iter_audio_files(musan_root_path)
    if not audio_files:
        raise SystemExit(f"No audio files found under {musan_root}")

    source_rows = [
        {
            "audio_path": str(audio_path.resolve()),
            "subset": detect_subset(audio_path, musan_root_path),
        }
        for audio_path in audio_files
    ]

    results_path = output_dir / "results.jsonl"
    all_results: list[dict[str, Any]] = []
    subset_results: dict[str, list[dict[str, Any]]] = defaultdict(list)
    subset_hours: dict[str, float] = defaultdict(float)
    total_hours = 0.0

    with results_path.open("w", encoding="utf-8") as results_handle:
        for source_row in source_rows:
            audio_path = source_row["audio_path"]
            subset = source_row["subset"]
            duration = audio_duration_sec(audio_path)
            total_hours += duration / 3600.0
            subset_hours[subset] += duration / 3600.0

            window_results = runner.run_file_windows(
                audio_path,
                keyword,
                window_sec=window_sec,
                hop_sec=hop_sec,
                keyword_phonemes=(
                    keyword_phonemes if use_keyword_phoneme_override else None
                ),
                include_score_details=True,
                include_eps_positions=True,
                include_seq_positions=True,
            )
            for window_result in window_results:
                record = musan_result_record(
                    audio_path,
                    keyword,
                    subset,
                    window_result,
                    window_index=int(window_result["window_index"]),
                    sequence_objective=sequence_objective,
                    qbyt_readout=provenance["qbyt_readout"],
                )
                all_results.append(record)
                subset_results[subset].append(record)
                results_handle.write(
                    json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n"
                )

    summary: dict[str, Any] = {
        "num_samples": len(all_results),
        "num_skipped": sum(
            bool(record.get("skipped", False)) for record in all_results
        ),
        "output_dir": str(output_dir.resolve()),
        "musan_root": str(musan_root_path.resolve()),
        "keyword": keyword,
        "keyword_phonemes": keyword_phonemes,
        "keyword_phonemes_source": keyword_phonemes_source,
        "stage2_ckpt": stage2_ckpt,
        "window_sec": window_sec,
        "hop_sec": hop_sec,
        "total_files": len(source_rows),
        "total_hours": total_hours,
        "stream": stream_description,
        "provenance": provenance,
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
        json.dump(summary, handle, ensure_ascii=False, indent=2, allow_nan=False)

    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))
    return summary


@hydra.main(version_base=None, config_path=str(CONFIG_DIR), config_name="config")
def main(cfg: DictConfig) -> None:
    run_eval(cfg)


if __name__ == "__main__":
    main()
