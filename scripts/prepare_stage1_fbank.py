#!/usr/bin/env python3
"""Precompute Stage I fbank features for train/dev JSONL manifests."""

from __future__ import annotations

from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from dma_kws.config import get_fbank_config, require_sections
from dma_kws.hydra_app import CONFIG_DIR, resolved_config
from dma_kws.stage1.prepare_fbank import prepare_manifest_fbank


def _resolve_manifest(override: str, default: Path) -> Path:
    return Path(override) if override else default


@hydra.main(version_base=None, config_path=str(CONFIG_DIR), config_name="config")
def main(cfg: DictConfig) -> None:
    config = resolved_config(cfg)
    require_sections(config, ["paths", "stage1"])
    prep = OmegaConf.to_container(cfg.prep, resolve=True)
    if not isinstance(prep, dict):
        prep = {}

    paths = config["paths"]
    stage1 = config["stage1"]
    processed_dir = Path(paths["processed_root"]) / "stage1_phoneme_ctc"

    train_manifest = _resolve_manifest(str(prep.get("train_manifest", "")), processed_dir / "train.jsonl")
    dev_manifest = _resolve_manifest(str(prep.get("dev_manifest", "")), processed_dir / "dev.jsonl")

    fbank_root = Path(
        prep.get("fbank_root", "")
        or stage1.get("fbank_root", "")
        or Path(paths.get("feature_root", "")) / "stage1_fbank"
    )
    audio_root = Path(
        prep.get("audio_root", "") or stage1.get("audio_root", "") or paths.get("librispeech_root", "")
    )

    sample_rate = int(stage1.get("sample_rate", 16000))
    fbank_cfg = get_fbank_config(config)
    limit = int(prep.get("limit", 0))
    skip_existing = not bool(prep.get("no_skip_existing", False))

    for label, manifest in (("train", train_manifest), ("dev", dev_manifest)):
        if not manifest.exists():
            print(f"Skipping {label}: manifest not found at {manifest}")
            continue

        out_path, written, skipped = prepare_manifest_fbank(
            manifest,
            fbank_root=fbank_root,
            output_manifest_path=manifest,
            audio_root=audio_root if audio_root.exists() else None,
            sample_rate=sample_rate,
            num_mel_bins=fbank_cfg.num_mel_bins,
            dither=fbank_cfg.dither,
            limit=limit,
            skip_existing=skip_existing,
        )
        print(
            f"{label}: wrote {written} fbank files, skipped {skipped} existing -> "
            f"manifest {out_path} under {fbank_root}"
        )


if __name__ == "__main__":
    main()
