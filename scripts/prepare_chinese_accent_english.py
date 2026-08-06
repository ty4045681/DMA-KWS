#!/usr/bin/env python3
"""Build Chinese-accent English adapter/evaluation manifests for remote Linux."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dma_kws.data_prep.chinese_accent_english import (
    DEFAULT_REMOTE_DATASET_ROOT,
    DEFAULT_SPLIT_SEED,
    build_chinese_accent_manifests,
    extract_l2_arctic_mandarin,
    materialize_adapter_mixtures,
)
from dma_kws.g2p import make_g2p


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        required=True,
        help="Local chinese_accent_english_datasets root; audio is read here",
    )
    parser.add_argument(
        "--manifest-root",
        default=str(DEFAULT_REMOTE_DATASET_ROOT),
        help="Absolute POSIX dataset root written into every wav_path",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Local manifest output (default: SOURCE_ROOT/manifests)",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        help="Local audit-report output (default: SOURCE_ROOT/reports)",
    )
    parser.add_argument("--speechocean-root", type=Path)
    parser.add_argument("--l2-arctic-root", type=Path)
    parser.add_argument("--edacc-root", type=Path)
    parser.add_argument(
        "--real-recordings-root",
        type=Path,
        help="Optional 281-WAV PC recording export containing recording_info.csv",
    )
    parser.add_argument(
        "--real-destination-root",
        type=Path,
        help="Anonymized copy destination (default: SOURCE_ROOT/raw/hey_eva_real/pc)",
    )
    parser.add_argument(
        "--l2-archive",
        type=Path,
        help="Optionally extract only BWC/LXC/NCC/TXHC before building manifests",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SPLIT_SEED)
    parser.add_argument("--speechocean-dev-speakers", type=int, default=25)
    parser.add_argument(
        "--keep-l2-source-rate",
        action="store_true",
        help="Do not build derived 16 kHz mono PCM16 L2 audio (not recommended)",
    )
    parser.add_argument("--l2-derived-root", type=Path)
    parser.add_argument("--ffmpeg-bin", default="ffmpeg")
    parser.add_argument("--audio-workers", type=int, default=4)
    parser.add_argument("--librispeech-train-manifest", type=Path)
    parser.add_argument("--librispeech-dev-manifest", type=Path)
    parser.add_argument("--train-mix-size", type=int, default=50000)
    parser.add_argument("--dev-select-size", type=int, default=5000)
    parser.add_argument(
        "--mixture-only",
        action="store_true",
        help="On Linux, combine existing accent manifests with supplied LibriSpeech manifests",
    )
    parser.add_argument(
        "--skip-audio-existence-check",
        action="store_true",
        help="For fixture/debug use only; normal builds should validate every audio path",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> dict:
    args = build_parser().parse_args(argv)
    source_root = args.source_root.expanduser().resolve()
    if args.mixture_only:
        if args.librispeech_train_manifest is None or args.librispeech_dev_manifest is None:
            raise SystemExit(
                "--mixture-only requires --librispeech-train-manifest and "
                "--librispeech-dev-manifest"
            )
        counts = materialize_adapter_mixtures(
            manifest_dir=(args.output_dir or source_root / "manifests"),
            librispeech_train_manifest=args.librispeech_train_manifest,
            librispeech_dev_manifest=args.librispeech_dev_manifest,
            train_size=args.train_mix_size,
            dev_size=args.dev_select_size,
            seed=args.seed,
        )
        summary = {
            "manifest_dir": str((args.output_dir or source_root / "manifests").resolve()),
            "manifest_root": args.manifest_root,
            "counts": counts,
            "mode": "mixture_only",
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
        return summary

    l2_root = (args.l2_arctic_root or source_root / "raw" / "l2_arctic_v5").resolve()
    if args.l2_archive:
        extract_l2_arctic_mandarin(args.l2_archive, l2_root)

    outputs = build_chinese_accent_manifests(
        source_root=source_root,
        manifest_root=args.manifest_root,
        g2p=make_g2p(),
        output_dir=args.output_dir,
        reports_dir=args.reports_dir,
        speechocean_root=args.speechocean_root,
        l2_arctic_root=l2_root,
        edacc_root=args.edacc_root,
        real_recordings_root=args.real_recordings_root,
        real_destination_root=args.real_destination_root,
        l2_derived_root=args.l2_derived_root,
        normalize_l2_audio=not args.keep_l2_source_rate,
        ffmpeg_bin=args.ffmpeg_bin,
        audio_workers=args.audio_workers,
        librispeech_train_manifest=args.librispeech_train_manifest,
        librispeech_dev_manifest=args.librispeech_dev_manifest,
        train_mix_size=args.train_mix_size,
        dev_select_size=args.dev_select_size,
        seed=args.seed,
        speechocean_dev_speakers=args.speechocean_dev_speakers,
        verify_audio=not args.skip_audio_existence_check,
    )
    summary = {
        "manifest_dir": str(outputs.manifest_dir),
        "reports_dir": str(outputs.reports_dir),
        "manifest_root": args.manifest_root,
        "counts": dict(outputs.counts),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return summary


if __name__ == "__main__":
    main()
