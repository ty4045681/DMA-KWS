#!/usr/bin/env python3
"""Precompute Stage I fbank features for train/dev JSONL manifests."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dma_kws.config import get_fbank_config, load_config, require_sections
from dma_kws.stage1.prepare_fbank import prepare_manifest_fbank


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument(
        "--train-manifest",
        default="",
        help="Input train JSONL (default: processed_root/stage1_phoneme_ctc/train.jsonl)",
    )
    parser.add_argument(
        "--dev-manifest",
        default="",
        help="Input dev JSONL (default: processed_root/stage1_phoneme_ctc/dev.jsonl)",
    )
    parser.add_argument(
        "--fbank-root",
        default="",
        help="Output fbank root (default: feature_root/stage1_fbank from config)",
    )
    parser.add_argument(
        "--audio-root",
        default="",
        help="LibriSpeech root for relative fbank paths (default: paths.librispeech_root)",
    )
    parser.add_argument("--limit", type=int, default=0, help="Optional max utterances per manifest")
    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Recompute fbank even when the output .npy already exists",
    )
    return parser.parse_args()


def _resolve_manifest(override: str, default: Path) -> Path:
    return Path(override) if override else default


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    require_sections(config, ["paths", "stage1"])

    paths = config["paths"]
    stage1 = config["stage1"]
    processed_dir = Path(paths["processed_root"]) / "stage1_phoneme_ctc"

    train_manifest = _resolve_manifest(args.train_manifest, processed_dir / "train.jsonl")
    dev_manifest = _resolve_manifest(args.dev_manifest, processed_dir / "dev.jsonl")

    fbank_root = Path(
        args.fbank_root
        or stage1.get("fbank_root", "")
        or Path(paths.get("feature_root", "")) / "stage1_fbank"
    )
    audio_root = Path(
        args.audio_root or stage1.get("audio_root", "") or paths.get("librispeech_root", "")
    )

    sample_rate = int(stage1.get("sample_rate", 16000))
    fbank_cfg = get_fbank_config(config)

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
            limit=args.limit,
            skip_existing=not args.no_skip_existing,
        )
        print(
            f"{label}: wrote {written} fbank files, skipped {skipped} existing -> "
            f"manifest {out_path} under {fbank_root}"
        )


if __name__ == "__main__":
    main()
