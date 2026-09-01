#!/usr/bin/env python3
"""Diagnostic: why is a QbyT v6 clip score low -- no evidence, or no legal path?

``eval_stage2_clips.py`` records only the scalar at the end of the scoring
chain. A low score has two incompatible causes and the scalar cannot tell them
apart:

  A. the emission model has little evidence for the keyword's phones, or
  B. the evidence is there but the bounded graph cannot collect it, because the
     keyword's own span or an inter-phone gap exceeds its configured bound.

This probe reuses the exact features, padding, streaming point, precision and
checkpoint of ``eval_stage2_clips.py``, then re-scores the *same* frame
posteriors under alternative readouts:

  deployed                 the shipped score (reproduces ``qbyt_raw_logit``)
  one_vs_rest_filler       the competing phone stays in the denominator
  relaxed_bounds           gap 1->3 frames, keyword span 30->50 frames
  one_vs_rest_and_relaxed  both

Reading the report:

  ``phone_top_frames_llr_mean`` near or below 0 for positives  -> cause A: the
      frame classifier is the bottleneck; the filler and bound ablations cannot
      help much and frame-level supervision is the prerequisite.
  ``relaxed_bounds`` logit far above ``deployed`` for positives -> cause B: the
      bounds are clipping true keywords.
  ``masked_query_mass_mean`` large, and ``one_vs_rest_filler`` well below
      ``deployed`` on near-miss negatives -> the query-relative filler is
      discounting substitutions onto phones the keyword already contains.
  ``best_frame_span`` above ``max_keyword_span_frames`` -> that clip's keyword
      cannot be covered by any legal path.

Usage (same arguments as eval_stage2_clips.py):
  python3 scripts/probe_qbyt_emissions.py \
    +experiment=icefall_zipformer_stage2_alignment \
    prep.manifest=.../merged_checked_with_phonemes_new.csv \
    prep.stage2_ckpt=.../step_step_006500.pt \
    prep.batch_size=128 prep.num_workers=8 run.device=cuda \
    prep.output_dir=.../hey_eva_and_its_variants \
    +prep.group_field=variant +prep.limit=0
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
from dma_kws.inference.stage2_clip import (
    ClipFeatureDataset,
    Stage2ClipRunner,
    collate_clip_feature_batch,
)
from dma_kws.training.device import resolve_accelerator

DEFAULT_PADDING_MS = 160


def _resolve_group_field(rows: list[dict], group_field: str) -> str:
    """Validate ``prep.group_field`` against the manifest columns.

    A mistyped column name would otherwise silently fall back to label grouping
    and the whole per-variant breakdown -- the reason to run this at all -- would
    quietly collapse into two rows.
    """
    if not group_field:
        return ""
    known = {key for row in rows for key in row}
    if group_field not in known:
        raise SystemExit(
            f"prep.group_field={group_field!r} is not a manifest column. "
            f"Available columns: {', '.join(sorted(known))}"
        )
    return group_field


def _group_of(row: dict, group_field: str) -> str:
    if group_field:
        return str(row[group_field])
    if "label" in row:
        return f"label={int(row['label'])}"
    return "all"


def run_probe(cfg: DictConfig) -> dict:
    try:
        import torch
        from torch.utils.data import DataLoader
    except ImportError as exc:
        raise SystemExit("Missing torch on this machine.") from exc

    # Imported here so the friendly ImportError above fires first on a CPU box.
    from dma_kws.inference.qbyt_diagnostics import (
        DEFAULT_ABLATIONS,
        summarize_emission_diagnostics,
    )

    torch.multiprocessing.set_sharing_strategy("file_system")

    config = resolved_config(cfg)
    require_sections(config, ["paths", "stage1", "stage2", "demo", "tokenizer"])
    prep = OmegaConf.to_container(cfg.prep, resolve=True)
    if not isinstance(prep, dict):
        prep = {}

    manifest_path = str(prep.get("manifest", ""))
    if not manifest_path:
        raise SystemExit("prep.manifest is required")
    if not str(prep.get("stage2_ckpt", "")):
        raise SystemExit("prep.stage2_ckpt is required")

    left_padding_ms = int(prep.get("left_padding_ms", DEFAULT_PADDING_MS))
    right_padding_ms = int(prep.get("right_padding_ms", DEFAULT_PADDING_MS))
    if left_padding_ms < 0 or right_padding_ms < 0:
        raise SystemExit("prep.left_padding_ms and prep.right_padding_ms must be >= 0")

    output_dir = Path(str(prep.get("output_dir", "") or "outputs/probe_qbyt_emissions"))
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_manifest(manifest_path)
    limit = int(prep.get("limit", 0) or 0)
    if limit > 0:
        rows = rows[:limit]
    group_field = _resolve_group_field(rows, str(prep.get("group_field", "") or ""))

    accelerator, _ = resolve_accelerator(str(cfg.run.device))
    device = torch.device(accelerator if accelerator == "cpu" else "cuda")
    runner = Stage2ClipRunner.from_config(config, prep, device)
    verifier = runner.verifier

    # Enrollment goes through the runner so the probed anchors are byte-identical
    # to the ones eval_stage2_clips.py scored.
    anchors: list[list[int]] = []
    for row_index, row in enumerate(rows, start=1):
        phonemes = runner.resolve_keyword_phonemes(
            str(row["keyword"]),
            row.get("keyword_phonemes"),
            field_name=f"Manifest row {row_index} keyword_phonemes",
        )
        anchors.append(runner.enroll_phonemes(phonemes))

    batch_size = int(prep.get("batch_size", 0) or 0) or 64
    num_workers = int(prep.get("num_workers", 0) or 0)
    if num_workers <= 0:
        num_workers = min(8, os.cpu_count() or 1)

    dataset = ClipFeatureDataset(
        audio_paths=[row["audio_path"] for row in rows],
        sample_rate=runner.sample_rate,
        fbank_extractor=verifier.fbank_extractor,
        fbank_kwargs=verifier.fbank_kwargs,
        min_fbank_frames=verifier.min_fbank_frames,
        left_padding_ms=left_padding_ms,
        right_padding_ms=right_padding_ms,
    )
    loader = DataLoader(
        dataset,
        batch_size=max(1, batch_size),
        shuffle=False,
        num_workers=max(0, num_workers),
        collate_fn=collate_clip_feature_batch,
    )

    records: list[dict] = []
    num_skipped = 0
    for batch in loader:
        feats, ids_batch, indices = [], [], []
        for index, feat, _end_sec in batch:
            if feat is None:
                num_skipped += 1
                continue
            feats.append(feat)
            ids_batch.append(anchors[index])
            indices.append(index)
        if not feats:
            continue
        for index, diagnostics in zip(
            indices, verifier.emission_diagnostics(feats, ids_batch)
        ):
            row = rows[index]
            record = {
                "audio_path": row["audio_path"],
                "keyword": row["keyword"],
                "group": _group_of(row, group_field),
                **diagnostics,
            }
            if "label" in row:
                record["label"] = int(row["label"])
            records.append(record)

    records_path = output_dir / "emission_diagnostics.jsonl"
    with records_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")

    summary = {
        "manifest": str(Path(manifest_path).resolve()),
        "checkpoint": str(Path(str(prep["stage2_ckpt"])).resolve()),
        "num_clips": len(records),
        "num_skipped": num_skipped,
        "group_field": group_field or None,
        "qbyt_alignment": verifier.qbyt_alignment.as_dict(),
        "stream": verifier.stream_policy.describe(),
        "audio_padding_ms": {"left": left_padding_ms, "right": right_padding_ms},
        "ablations": [
            {
                "name": spec.name,
                "filler": spec.filler,
                "max_inter_phone_gap_frames": spec.max_inter_phone_gap_frames,
                "max_keyword_span_frames": spec.max_keyword_span_frames,
            }
            for spec in DEFAULT_ABLATIONS
        ],
        "by_group": summarize_emission_diagnostics(records),
        "records": str(records_path.resolve()),
    }
    summary_path = output_dir / "emission_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, allow_nan=False)

    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))
    return summary


@hydra.main(version_base=None, config_path=str(CONFIG_DIR), config_name="config")
def main(cfg: DictConfig) -> None:
    run_probe(cfg)


if __name__ == "__main__":
    main()
