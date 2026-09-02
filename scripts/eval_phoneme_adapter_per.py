#!/usr/bin/env python3
"""Evaluate phoneme-adapter PER on an ``eval_stage2_clips.py`` manifest.

By default, the true transcript is read from the manifest's ``text_variant``
column. A per-row ``text_variant_phonemes`` value overrides its automatic G2P.
Evaluation can also be restricted to positive rows before using ``keyword`` as
the reference for legacy manifests, with ``keyword_phonemes`` as its optional
per-row override. CSV overrides are space-separated ARPAbet strings; JSONL also
accepts arrays of strings.

Clips are decoded unpadded by default, matching the training features, MUSAN
evaluation and the deployed two-stage path. ``+prep.left_padding_ms=...`` and
``+prep.right_padding_ms=...`` add zero-valued waveform context on either side;
that padding is part of the decoded model input and counts toward its minimum
length.

Example for a manifest containing ``audio_path,keyword,label,text_variant``::

    python scripts/eval_phoneme_adapter_per.py \
      +experiment=icefall_zipformer_stage2_adapter \
      prep.manifest=/path/to/test.csv \
      prep.stage2_ckpt=/path/to/stage2.pt

Legacy manifest without ``text_variant``::

    python scripts/eval_phoneme_adapter_per.py \
      +experiment=icefall_zipformer_stage2_adapter \
      prep.manifest=/path/to/test.csv \
      prep.stage2_ckpt=/path/to/stage2.pt \
      prep.per_reference_column=keyword \
      prep.per_label_filter=1
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from dma_kws.config import require_sections
from dma_kws.hydra_app import CONFIG_DIR, resolved_config
from dma_kws.inference.manifest import load_manifest
from dma_kws.inference.phoneme_per import (
    PhonemePerRunner,
    select_per_rows,
    summarize_per_results,
)
from dma_kws.training.device import resolve_accelerator


DEFAULT_PADDING_MS = 0


def _optional_int(value) -> int | None:
    if value is None or str(value).strip() == "":
        return None
    return int(value)


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
    require_sections(config, ["paths", "stage1", "stage2", "tokenizer"])
    prep = OmegaConf.to_container(cfg.prep, resolve=True)
    if not isinstance(prep, dict):
        prep = {}

    manifest_path = str(prep.get("manifest", "")).strip()
    if not manifest_path:
        raise SystemExit("prep.manifest is required")
    stage2_ckpt = str(prep.get("stage2_ckpt", "")).strip()
    if not stage2_ckpt:
        raise SystemExit("prep.stage2_ckpt is required")
    left_padding_ms, right_padding_ms = _resolve_audio_padding_ms(prep)

    reference_column = str(prep.get("per_reference_column", "text_variant")).strip()
    label_filter = _optional_int(prep.get("per_label_filter"))
    limit = int(prep.get("limit", 0) or 0)
    all_rows = load_manifest(manifest_path)
    try:
        first_row_number = 2 if Path(manifest_path).suffix.lower() == ".csv" else 1
        rows = select_per_rows(
            all_rows,
            reference_column=reference_column,
            label_filter=label_filter,
            limit=limit,
            first_row_number=first_row_number,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    run_cfg = cfg.run
    accelerator, _ = resolve_accelerator(str(run_cfg.device))
    device = torch.device(accelerator if accelerator == "cpu" else "cuda")
    runner = PhonemePerRunner.from_config(config, prep, device)

    batch_size = int(prep.get("batch_size", 0) or 0)
    if batch_size <= 0:
        batch_size = 64
    num_workers = int(prep.get("num_workers", 0) or 0)
    if num_workers <= 0:
        num_workers = min(8, os.cpu_count() or 1)

    try:
        results = runner.run_batch(
            rows,
            reference_column=reference_column,
            batch_size=batch_size,
            num_workers=num_workers,
            left_padding_ms=left_padding_ms,
            right_padding_ms=right_padding_ms,
        )
    except (RuntimeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    output_dir = Path(
        str(prep.get("per_output_dir", "outputs/eval_phoneme_adapter_per"))
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "results.jsonl"
    with results_path.open("w", encoding="utf-8") as handle:
        for record in results:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary = {
        "manifest": str(Path(manifest_path).resolve()),
        "stage2_checkpoint": str(Path(stage2_ckpt).resolve()),
        "reference_column": reference_column,
        "label_filter": label_filter,
        "num_manifest_rows": len(all_rows),
        "num_selected_rows": len(rows),
        "audio_padding_ms": {
            "left": left_padding_ms,
            "right": right_padding_ms,
        },
        "stream": runner.stream_policy.describe(),
        "metrics": summarize_per_results(results),
        "results_path": str(results_path.resolve()),
    }
    if reference_column == "keyword":
        summary["reference_assumption"] = (
            "Each selected label=1 clip contains exactly the keyword and no extra speech"
        )
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
