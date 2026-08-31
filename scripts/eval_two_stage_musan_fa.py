#!/usr/bin/env python3
"""Evaluate two-stage DMA-KWS false accepts on the MUSAN corpus.

Runs Stage I on each whole MUSAN file, then scores every remaining span with
Stage II QbyT. FA/hour counts two-stage wake-ups (spans at or above the QbyT
threshold) over total audio hours. Multiple accepts in one file all count.
Files with no scored Stage I spans contribute hours only.

This is not comparable to the official 3s Stage-II-only grid in
``scripts/eval_musan_fa.py``.
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
from dma_kws.inference.detection_plots import (
    DEFAULT_PLOT_DPI,
    write_false_accept_rate_plot as _write_false_accept_rate_plot,
)
from dma_kws.inference.manifest import iter_audio_files
from dma_kws.inference.musan_fa import (
    TWO_STAGE_WAKEUP_PROTOCOL,
    audio_duration_sec,
    detect_subset,
    false_accept_metrics,
    select_shard,
    subset_summary,
    two_stage_wakeup_result_record,
)
from dma_kws.inference.pipeline import TwoStageKWSPipeline
from dma_kws.inference.stage2_reporting import build_score_provenance
from dma_kws.inference.stage2_verifier import resolve_inference_amp
from dma_kws.training.device import resolve_accelerator


def _locator_type(config: Mapping[str, Any]) -> str:
    locator_cfg = config.get("locator")
    if isinstance(locator_cfg, Mapping):
        return str(locator_cfg.get("type", "phoneme_ctc"))
    return "phoneme_ctc"


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
        raise SystemExit("prep.stage2_ckpt is required for Stage II verification")

    num_shards = int(prep.get("num_shards", 1) or 1)
    shard_index = int(prep.get("shard_index", 0) or 0)
    if num_shards < 1:
        raise SystemExit("prep.num_shards must be >= 1")
    if shard_index < 0 or shard_index >= num_shards:
        raise SystemExit(
            f"prep.shard_index must be in [0, {num_shards}), got {shard_index}"
        )

    try:
        amp = resolve_inference_amp(prep.get("amp"))
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    output_dir_override = str(prep.get("output_dir", ""))
    output_dir = Path(output_dir_override or "outputs/eval_two_stage_musan_fa")
    output_dir.mkdir(parents=True, exist_ok=True)

    accelerator, _ = resolve_accelerator(str(run_cfg.device))
    device = torch.device(accelerator if accelerator == "cpu" else "cuda")
    pipeline = TwoStageKWSPipeline.from_config(config, prep, device)
    threshold = float(pipeline._demo_cfg.get("qbyt_threshold", 0.5))

    raw_keyword_phonemes = prep.get("keyword_phonemes")
    use_keyword_phoneme_override = raw_keyword_phonemes is not None and not (
        isinstance(raw_keyword_phonemes, str)
        and not raw_keyword_phonemes.strip()
    )
    try:
        keyword_phonemes = pipeline.resolve_keyword_phonemes(
            keyword,
            raw_keyword_phonemes if use_keyword_phoneme_override else None,
            field_name="prep.keyword_phonemes",
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    keyword_phonemes_source = (
        "prep.keyword_phonemes" if use_keyword_phoneme_override else "g2p"
    )
    override_phonemes = (
        keyword_phonemes if use_keyword_phoneme_override else None
    )

    stream_description = pipeline.stream_policy.describe()
    provenance = build_score_provenance(
        config,
        checkpoint_path=stage2_ckpt,
        stream=stream_description,
        left_padding_ms=0,
        right_padding_ms=0,
    )
    locator_type = _locator_type(config)

    audio_files = iter_audio_files(musan_root_path)
    if not audio_files:
        raise SystemExit(f"No audio files found under {musan_root}")

    catalog = [
        {
            "audio_path": str(audio_path.resolve()),
            "subset": detect_subset(audio_path, musan_root_path),
            "duration_sec": audio_duration_sec(audio_path),
        }
        for audio_path in audio_files
    ]
    try:
        source_rows = select_shard(
            catalog,
            num_shards=num_shards,
            shard_index=shard_index,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    results_path = output_dir / "results.jsonl"
    all_results: list[dict[str, Any]] = []
    subset_results: dict[str, list[dict[str, Any]]] = defaultdict(list)
    subset_hours: dict[str, float] = defaultdict(float)
    total_hours = 0.0
    num_stage1_candidates = 0
    num_stage2_scored = 0

    with results_path.open("w", encoding="utf-8") as results_handle:
        for source_row in source_rows:
            audio_path = source_row["audio_path"]
            subset = source_row["subset"]
            duration = float(source_row["duration_sec"])
            total_hours += duration / 3600.0
            subset_hours[subset] += duration / 3600.0

            pipeline_result = pipeline.run(
                audio_path,
                keyword,
                keyword_phonemes=override_phonemes,
            )
            stage1_candidates = pipeline_result.get("stage1_candidates") or []
            stage2_scores = pipeline_result.get("stage2_scores") or []
            num_stage1_candidates += len(stage1_candidates)
            num_stage2_scored += len(stage2_scores)

            for candidate_index, scored in enumerate(stage2_scores):
                record = two_stage_wakeup_result_record(
                    audio_path,
                    keyword,
                    subset,
                    duration,
                    keyword_phonemes,
                    scored,
                    candidate_index=candidate_index,
                    threshold=threshold,
                )
                all_results.append(record)
                subset_results[subset].append(record)
                results_handle.write(
                    json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n"
                )

    num_wakeups = sum(bool(record.get("detected")) for record in all_results)
    summary: dict[str, Any] = {
        "eval_protocol": TWO_STAGE_WAKEUP_PROTOCOL,
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
        "locator": locator_type,
        "amp": amp or "off",
        "num_shards": num_shards,
        "shard_index": shard_index,
        "total_files": len(source_rows),
        "total_hours": total_hours,
        "num_stage1_candidates": num_stage1_candidates,
        "num_stage2_scored": num_stage2_scored,
        "num_wakeups": num_wakeups,
        "stream": stream_description,
        "provenance": provenance,
        "metrics": false_accept_metrics(
            all_results,
            threshold=threshold,
            total_hours=total_hours,
        ),
    }

    subsets: dict[str, dict[str, Any]] = {}
    for subset in sorted(subset_hours):
        subsets[subset] = subset_summary(
            subset_results[subset],
            threshold=threshold,
            total_hours=subset_hours[subset],
        )
    if subsets:
        summary["subsets"] = subsets

    if bool(prep.get("plot_curves", True)):
        summary["plots"] = _write_false_accept_rate_plot(
            all_results,
            output_dir=output_dir,
            threshold=threshold,
            total_hours=total_hours,
            dpi=int(prep.get("plot_dpi", DEFAULT_PLOT_DPI)),
        )

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
