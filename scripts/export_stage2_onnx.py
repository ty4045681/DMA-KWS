#!/usr/bin/env python3
"""Export a converted Stage II QbyT v4.1 full-model ``.pt`` to ONNX."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dma_kws.inference.stage2_onnx import (  # noqa: E402
    Stage2OnnxExportError,
    export_stage2_onnx,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Export a complete Stage II QbyT v4.1 PyTorch .pt artifact to split and/or "
            "monolithic ONNX graphs. Split export runs the encoder once and is "
            "recommended for multiple keywords/pronunciations."
        )
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help=(
            "Full-model Stage II .pt. Convert Lightning .ckpt files first with "
            "scripts/convert_stage2_checkpoints.py; raw LoRA adapters are rejected."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for ONNX files and stage2_onnx_manifest.json.",
    )
    parser.add_argument(
        "--layout",
        choices=("split", "full", "both"),
        default="split",
        help=(
            "split: stage2_encoder.onnx + qbyt.onnx (default); "
            "full: stage2.onnx; both: write all three."
        ),
    )
    parser.add_argument(
        "--exporter",
        choices=("dynamo", "legacy"),
        default="legacy",
        help=(
            "PyTorch ONNX exporter backend. legacy is the validated default; "
            "dynamo requires onnxscript."
        ),
    )
    parser.add_argument(
        "--opset-version",
        type=int,
        default=17,
        help="ONNX opset version, at least 17 (default: 17).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Example/static batch size (default: 1).",
    )
    parser.add_argument(
        "--feature-frames",
        type=int,
        default=300,
        help="Example/static padded fbank-frame width (default: 300).",
    )
    parser.add_argument(
        "--anchor-tokens",
        type=int,
        default=16,
        help="Example/static padded phoneme-token width (default: 16).",
    )
    parser.add_argument(
        "--dynamic-batch",
        action="store_true",
        help=(
            "Make only the batch axis dynamic. Feature-frame and anchor-token "
            "widths stay fixed; pass their true lengths in the length inputs."
        ),
    )
    parser.add_argument(
        "--include-sequence-logits",
        action="store_true",
        help="Export the training/diagnostic sequence_logits output as well as raw_logit.",
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        help=(
            "Optional Stage II calibration JSON to hash and record in the manifest. "
            "Calibration remains outside ONNX; graph output is always raw_logit."
        ),
    )
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help=(
            "Skip ONNX Runtime CPU parity checks. ONNX structural validation still runs."
        ),
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1.0e-4,
        help="Absolute tolerance for ONNX Runtime parity (default: 1e-4).",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=1.0e-4,
        help="Relative tolerance for ONNX Runtime parity (default: 1e-4).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing files for this export layout after validation succeeds.",
    )
    return parser


def run(args: argparse.Namespace):
    result = export_stage2_onnx(
        args.checkpoint,
        args.output_dir,
        layout=args.layout,
        exporter=args.exporter,
        opset_version=args.opset_version,
        batch_size=args.batch_size,
        feature_frames=args.feature_frames,
        anchor_tokens=args.anchor_tokens,
        dynamic_batch=args.dynamic_batch,
        include_sequence_logits=args.include_sequence_logits,
        calibration_path=args.calibration,
        verify=not args.skip_verify,
        atol=args.atol,
        rtol=args.rtol,
        overwrite=args.overwrite,
    )
    for path in result.artifacts:
        print(f"ONNX: {path}")
    print(f"Manifest: {result.manifest}")
    if result.verification_max_abs_error:
        maximum = max(result.verification_max_abs_error.values(), default=0.0)
        print(f"ONNX Runtime parity passed; max_abs_error={maximum:.6g}")
    else:
        print("ONNX Runtime parity skipped")
    return result


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        run(args)
    except (Stage2OnnxExportError, FileNotFoundError, FileExistsError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
