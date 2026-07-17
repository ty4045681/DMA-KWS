#!/usr/bin/env python3
"""Generate a MindSpore Lite FULL_QUANT config from an encoder ONNX graph."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


def _safe_dir_name(index: int, input_name: str) -> str:
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", input_name).strip("._")
    return f"{index:03d}_{safe_name or 'input'}"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate an r2.9.0 MindSpore Lite FULL_QUANT config and one "
            "calibration directory for every real ONNX graph input."
        )
    )
    parser.add_argument("--model", type=Path, required=True, help="Encoder ONNX path")
    parser.add_argument(
        "--calibration-root",
        type=Path,
        required=True,
        help="Root directory that will contain per-input BIN directories",
    )
    parser.add_argument("--output", type=Path, required=True, help="Output .cfg path")
    parser.add_argument("--calibrate-size", type=int, default=100)
    parser.add_argument(
        "--debug-dir",
        type=Path,
        default=Path("quant_debug/zipformer_encoder"),
    )
    parser.add_argument(
        "--activation-method",
        choices=("MAX_MIN", "KL", "REMOVAL_OUTLIER"),
        default="MAX_MIN",
    )
    parser.add_argument(
        "--per-layer",
        action="store_true",
        help="Use per-layer weight quantization instead of the default per-channel mode",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.calibrate_size < 1:
        raise SystemExit("--calibrate-size must be at least 1")
    if not args.model.is_file():
        raise SystemExit(f"ONNX model not found: {args.model}")

    try:
        import onnx
    except ImportError as exc:
        raise SystemExit(
            "The 'onnx' Python package is required. Install it in the model-export "
            "environment, then rerun this command."
        ) from exc

    model = onnx.load(str(args.model), load_external_data=False)
    initializer_names = {initializer.name for initializer in model.graph.initializer}
    graph_inputs = [
        value_info
        for value_info in model.graph.input
        if value_info.name not in initializer_names
    ]
    if not graph_inputs:
        raise SystemExit("No runtime inputs were found in the ONNX graph")

    calibration_root = args.calibration_root.expanduser().resolve()
    calibration_root.mkdir(parents=True, exist_ok=True)
    mappings: list[str] = []
    inventory: list[str] = []

    for index, value_info in enumerate(graph_inputs):
        input_dir = calibration_root / _safe_dir_name(index, value_info.name)
        input_dir.mkdir(parents=True, exist_ok=True)
        mappings.append(f"{value_info.name}:{input_dir}")

        tensor_type = value_info.type.tensor_type
        dims = []
        for dim in tensor_type.shape.dim:
            if dim.HasField("dim_value"):
                dims.append(str(dim.dim_value))
            elif dim.HasField("dim_param"):
                dims.append(dim.dim_param)
            else:
                dims.append("?")
        dtype = onnx.TensorProto.DataType.Name(tensor_type.elem_type)
        inventory.append(
            f"{index:03d}  {value_info.name}  dtype={dtype}  shape=[{', '.join(dims)}]  dir={input_dir}"
        )

    debug_dir = args.debug_dir.expanduser().resolve()
    config = "\n".join(
        (
            "[common_quant_param]",
            "quant_type=FULL_QUANT",
            "bit_num=8",
            f"debug_info_save_path={debug_dir}",
            "",
            "[data_preprocess_param]",
            f"calibrate_path={','.join(mappings)}",
            f"calibrate_size={args.calibrate_size}",
            "input_type=BIN",
            "",
            "[full_quant_param]",
            f"activation_quant_method={args.activation_method}",
            "bias_correction=true",
            f"per_channel={'false' if args.per_layer else 'true'}",
            "",
        )
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(config, encoding="utf-8")

    print(f"Wrote config: {args.output.resolve()}")
    print(f"Calibration samples required per input: {args.calibrate_size}")
    print("ONNX runtime input inventory:")
    for line in inventory:
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
