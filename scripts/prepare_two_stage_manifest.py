#!/usr/bin/env python3
"""Generate a two-stage KWS manifest from an audio directory and a single keyword."""

from __future__ import annotations

import json
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from dma_kws.hydra_app import CONFIG_DIR
from dma_kws.inference.manifest import build_manifest_rows, iter_audio_files, write_manifest


def _resolve_label(prep: dict) -> int | None:
    raw = prep.get("label", "")
    if raw is None or raw == "":
        return None
    return int(raw)


@hydra.main(version_base=None, config_path=str(CONFIG_DIR), config_name="config")
def main(cfg: DictConfig) -> None:
    prep = OmegaConf.to_container(cfg.prep, resolve=True)
    if not isinstance(prep, dict):
        prep = {}

    input_dir = str(prep.get("input_dir", ""))
    if not input_dir:
        raise SystemExit("prep.input_dir is required")

    keyword = str(prep.get("keyword", ""))
    if not keyword:
        raise SystemExit("prep.keyword is required")

    output = str(prep.get("output", ""))
    if not output:
        raise SystemExit("prep.output is required")

    recursive = bool(prep.get("recursive", True))
    manifest_format = str(prep.get("manifest_format", "auto"))
    limit = int(prep.get("limit", 0))
    label = _resolve_label(prep)

    audio_paths = iter_audio_files(input_dir, recursive=recursive)
    if not audio_paths:
        raise SystemExit(f"No audio files found under {input_dir}")
    if limit:
        audio_paths = audio_paths[:limit]

    output_path = Path(output)
    rows = build_manifest_rows(
        audio_paths,
        keyword,
        label=label,
        manifest_dir=output_path.parent,
    )
    written_path = write_manifest(output_path, rows, manifest_format=manifest_format)

    summary = {
        "num_samples": len(rows),
        "output": str(written_path.resolve()),
        "keyword": keyword,
        "input_dir": str(Path(input_dir).resolve()),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
