#!/usr/bin/env python3
"""Generate a two-stage KWS manifest from an audio directory."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from dma_kws.hydra_app import CONFIG_DIR
from dma_kws.inference.manifest import (
    build_manifest_rows,
    build_manifest_rows_by_filename,
    iter_audio_files,
    write_manifest,
)


def _resolve_label(prep: dict) -> int | None:
    raw = prep.get("label", "")
    if raw is None or raw == "":
        return None
    return int(raw)


def _resolve_keywords(prep: dict) -> list[str]:
    raw = prep.get("keywords", [])
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise SystemExit("prep.keywords must be a list, e.g. ['hey eva', 'ok lamp']")
    keywords = [str(item).strip() for item in raw if str(item).strip()]
    return keywords


def _resolve_keyword_labels(prep: dict) -> dict[str, int]:
    raw = prep.get("keyword_labels", {})
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise SystemExit("prep.keyword_labels must be a mapping, e.g. {'hey eva': 1, 'ok lamp': 0}")
    labels: dict[str, int] = {}
    for key, value in raw.items():
        labels[str(key)] = int(value)
    return labels


@hydra.main(version_base=None, config_path=str(CONFIG_DIR), config_name="config")
def main(cfg: DictConfig) -> None:
    prep = OmegaConf.to_container(cfg.prep, resolve=True)
    if not isinstance(prep, dict):
        prep = {}

    input_dir = str(prep.get("input_dir", ""))
    if not input_dir:
        raise SystemExit("prep.input_dir is required")

    output = str(prep.get("output", ""))
    if not output:
        raise SystemExit("prep.output is required")

    mode = str(prep.get("keyword_mode", "single")).strip() or "single"
    recursive = bool(prep.get("recursive", True))
    skip_unmatched = bool(prep.get("skip_unmatched", True))
    match_casefold = bool(prep.get("match_casefold", True))
    manifest_format = str(prep.get("manifest_format", "auto"))
    limit = int(prep.get("limit", 0))
    label = _resolve_label(prep)
    keywords = _resolve_keywords(prep)
    keyword_labels = _resolve_keyword_labels(prep)

    audio_paths = iter_audio_files(input_dir, recursive=recursive)
    if not audio_paths:
        raise SystemExit(f"No audio files found under {input_dir}")
    if limit:
        audio_paths = audio_paths[:limit]

    output_path = Path(output)
    summary: dict[str, object] = {
        "input_dir": str(Path(input_dir).resolve()),
    }

    if mode == "single":
        keyword = str(prep.get("keyword", ""))
        if not keyword:
            raise SystemExit("prep.keyword is required when prep.keyword_mode=single")
        rows = build_manifest_rows(
            audio_paths,
            keyword,
            label=label,
            manifest_dir=output_path.parent,
        )
        summary["keyword_mode"] = "single"
        summary["keyword"] = keyword
    elif mode == "auto_assign":
        if not keywords:
            raise SystemExit("prep.keywords is required when prep.keyword_mode=auto_assign")
        if not keyword_labels:
            raise SystemExit("prep.keyword_labels is required when prep.keyword_mode=auto_assign")

        rows, unmatched = build_manifest_rows_by_filename(
            audio_paths,
            keywords,
            keyword_labels=keyword_labels,
            manifest_dir=output_path.parent,
            skip_unmatched=skip_unmatched,
            casefold=match_casefold,
        )
        distribution = Counter(row["keyword"] for row in rows)
        summary["keyword_mode"] = "auto_assign"
        summary["matched_samples"] = len(rows)
        summary["skipped_unmatched"] = len(unmatched)
        summary["unmatched_examples"] = unmatched[:10]
        summary["keyword_distribution"] = dict(sorted(distribution.items()))
    else:
        raise SystemExit("prep.keyword_mode must be 'single' or 'auto_assign'")

    written_path = write_manifest(output_path, rows, manifest_format=manifest_format)

    summary["num_samples"] = len(rows)
    summary["output"] = str(written_path.resolve())
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
