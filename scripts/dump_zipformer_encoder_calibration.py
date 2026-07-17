#!/usr/bin/env python3
"""Dump paired MindSpore Lite BIN calibration samples from ONNX input feeds.

The intended use is to call ``writer.write(feed_dict)`` immediately before
``onnxruntime.InferenceSession.run`` in a real streaming inference loop.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


ORT_TO_NUMPY = {
    "tensor(bool)": np.dtype(np.bool_),
    "tensor(double)": np.dtype(np.float64),
    "tensor(float)": np.dtype(np.float32),
    "tensor(float16)": np.dtype(np.float16),
    "tensor(int8)": np.dtype(np.int8),
    "tensor(int16)": np.dtype(np.int16),
    "tensor(int32)": np.dtype(np.int32),
    "tensor(int64)": np.dtype(np.int64),
    "tensor(uint8)": np.dtype(np.uint8),
    "tensor(uint16)": np.dtype(np.uint16),
    "tensor(uint32)": np.dtype(np.uint32),
    "tensor(uint64)": np.dtype(np.uint64),
}


def _safe_dir_name(index: int, input_name: str) -> str:
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", input_name).strip("._")
    return f"{index:03d}_{safe_name or 'input'}"


def _parse_shape_overrides(value: str | None) -> dict[str, tuple[int, ...]]:
    """Parse converter-style shapes: ``x:1,39,80;state:1,2,3``."""
    if not value:
        return {}
    result: dict[str, tuple[int, ...]] = {}
    for item in value.split(";"):
        item = item.strip()
        if not item:
            continue
        try:
            name, raw_shape = item.split(":", 1)
            shape = tuple(int(dim.strip()) for dim in raw_shape.split(","))
        except (ValueError, TypeError) as exc:
            raise ValueError(
                "Invalid --shape-overrides. Expected "
                "'x:1,39,80;cached_state:1,2,3'."
            ) from exc
        if not name or not shape or any(dim < 1 for dim in shape):
            raise ValueError(f"Invalid positive static shape for input {name!r}: {shape}")
        result[name] = shape
    return result


@dataclass(frozen=True)
class InputSpec:
    index: int
    name: str
    ort_type: str
    shape: tuple[int | str | None, ...]
    directory: str

    @property
    def numpy_dtype(self) -> np.dtype[Any]:
        try:
            return ORT_TO_NUMPY[self.ort_type]
        except KeyError as exc:
            raise ValueError(
                f"Unsupported ONNX Runtime input type for {self.name!r}: {self.ort_type}"
            ) from exc


class CalibrationBinWriter:
    """Write one synchronized BIN file per model input and streaming step."""

    def __init__(
        self,
        input_specs: Sequence[InputSpec],
        output_root: Path | str,
        limit: int = 100,
    ) -> None:
        if limit < 1:
            raise ValueError("limit must be at least 1")
        if not input_specs:
            raise ValueError("input_specs must not be empty")

        self.input_specs = tuple(input_specs)
        self.output_root = Path(output_root).expanduser().resolve()
        self.limit = limit
        self.count = 0
        self.output_root.mkdir(parents=True, exist_ok=True)
        for spec in self.input_specs:
            (self.output_root / spec.directory).mkdir(parents=True, exist_ok=True)
        self._write_manifest()

    @classmethod
    def from_session(
        cls,
        session: Any,
        output_root: Path | str,
        limit: int = 100,
    ) -> "CalibrationBinWriter":
        specs = []
        for index, node_arg in enumerate(session.get_inputs()):
            specs.append(
                InputSpec(
                    index=index,
                    name=node_arg.name,
                    ort_type=node_arg.type,
                    shape=tuple(node_arg.shape),
                    directory=_safe_dir_name(index, node_arg.name),
                )
            )
        return cls(specs, output_root=output_root, limit=limit)

    def _write_manifest(self) -> None:
        manifest = {
            "limit": self.limit,
            "naming": "the same six-digit filename is one synchronized encoder call",
            "inputs": [asdict(spec) for spec in self.input_specs],
        }
        (self.output_root / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    @property
    def full(self) -> bool:
        return self.count >= self.limit

    @property
    def calibrate_path(self) -> str:
        return ",".join(
            f"{spec.name}:{self.output_root / spec.directory}"
            for spec in self.input_specs
        )

    def _validate_shape(self, spec: InputSpec, array: np.ndarray[Any, Any]) -> None:
        if len(spec.shape) != array.ndim:
            raise ValueError(
                f"Input {spec.name!r} rank mismatch: ONNX={spec.shape}, actual={array.shape}"
            )
        for axis, (expected, actual) in enumerate(zip(spec.shape, array.shape)):
            if isinstance(expected, int) and expected >= 0 and expected != actual:
                raise ValueError(
                    f"Input {spec.name!r} shape mismatch at axis {axis}: "
                    f"expected {expected}, got {actual}"
                )

    def write(self, feed_dict: Mapping[str, Any]) -> bool:
        """Write one paired sample; return False once the configured limit is full."""
        if self.full:
            return False

        expected_names = {spec.name for spec in self.input_specs}
        actual_names = set(feed_dict)
        missing = sorted(expected_names - actual_names)
        extra = sorted(actual_names - expected_names)
        if missing or extra:
            raise ValueError(f"Input-name mismatch: missing={missing}, extra={extra}")

        arrays: dict[str, np.ndarray[Any, Any]] = {}
        for spec in self.input_specs:
            array = np.asarray(feed_dict[spec.name])
            if array.dtype != spec.numpy_dtype:
                raise TypeError(
                    f"Input {spec.name!r} dtype mismatch: ONNX={spec.numpy_dtype}, "
                    f"actual={array.dtype}; do not silently cast calibration states"
                )
            self._validate_shape(spec, array)
            arrays[spec.name] = np.ascontiguousarray(array)

        filename = f"{self.count:06d}.bin"
        written: list[Path] = []
        try:
            for spec in self.input_specs:
                output_path = self.output_root / spec.directory / filename
                with output_path.open("wb") as output_file:
                    arrays[spec.name].tofile(output_file)
                written.append(output_path)
        except Exception:
            for output_path in written:
                output_path.unlink(missing_ok=True)
            raise

        self.count += 1
        return True


def _resolve_random_shape(
    spec: InputSpec,
    shape_overrides: Mapping[str, tuple[int, ...]],
) -> tuple[int, ...]:
    if spec.name in shape_overrides:
        shape = shape_overrides[spec.name]
        if len(shape) != len(spec.shape):
            raise ValueError(
                f"Shape override rank mismatch for {spec.name!r}: "
                f"ONNX={spec.shape}, override={shape}"
            )
        for axis, (expected, actual) in enumerate(zip(spec.shape, shape)):
            if isinstance(expected, int) and expected >= 0 and expected != actual:
                raise ValueError(
                    f"Shape override mismatch for {spec.name!r} at axis {axis}: "
                    f"ONNX requires {expected}, override gives {actual}"
                )
        return shape

    dynamic_dims = [dim for dim in spec.shape if not isinstance(dim, int) or dim < 0]
    if dynamic_dims:
        raise ValueError(
            f"Input {spec.name!r} has a dynamic shape {spec.shape}. Add an explicit "
            f"override such as --shape-overrides '{spec.name}:1,...'."
        )
    return tuple(int(dim) for dim in spec.shape)


def _make_random_array(
    spec: InputSpec,
    shape: tuple[int, ...],
    rng: np.random.Generator,
    float_low: float,
    float_high: float,
    integer_low: int,
    integer_high: int,
) -> np.ndarray[Any, Any]:
    dtype = spec.numpy_dtype
    if np.issubdtype(dtype, np.floating):
        return rng.uniform(float_low, float_high, size=shape).astype(dtype)
    if np.issubdtype(dtype, np.bool_):
        return rng.integers(0, 2, size=shape, dtype=np.int8).astype(dtype)
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        low = max(integer_low, int(info.min))
        high = min(integer_high, int(info.max) + 1)
        if low >= high:
            raise ValueError(
                f"Empty integer range [{integer_low}, {integer_high}) for {spec.name!r} "
                f"with dtype {dtype}"
            )
        return rng.integers(low, high, size=shape, dtype=dtype)
    raise ValueError(f"Random generation is unsupported for {spec.name!r}: {dtype}")


def write_random_samples(
    writer: CalibrationBinWriter,
    seed: int,
    shape_overrides: Mapping[str, tuple[int, ...]],
    float_low: float = -1.0,
    float_high: float = 1.0,
    integer_low: int = 0,
    integer_high: int = 32,
) -> None:
    if not float_low < float_high:
        raise ValueError("--float-low must be less than --float-high")
    if not integer_low < integer_high:
        raise ValueError("--integer-low must be less than --integer-high")

    shapes = {
        spec.name: _resolve_random_shape(spec, shape_overrides)
        for spec in writer.input_specs
    }
    unknown_overrides = sorted(set(shape_overrides) - set(shapes))
    if unknown_overrides:
        raise ValueError(f"Shape overrides contain unknown ONNX inputs: {unknown_overrides}")

    rng = np.random.default_rng(seed)
    while not writer.full:
        feed_dict = {
            spec.name: _make_random_array(
                spec,
                shapes[spec.name],
                rng,
                float_low=float_low,
                float_high=float_high,
                integer_low=integer_low,
                integer_high=integer_high,
            )
            for spec in writer.input_specs
        }
        writer.write(feed_dict)


def _load_session(model_path: Path) -> Any:
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise SystemExit(
            "The 'onnxruntime' Python package is required. Run this in the existing "
            "Zipformer ONNX inference environment."
        ) from exc
    return ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect an encoder ONNX or convert paired NPZ feed snapshots to BIN."
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    source_group = parser.add_mutually_exclusive_group()
    source_group.add_argument("--npz-dir", type=Path)
    source_group.add_argument(
        "--random",
        action="store_true",
        help="Generate reproducible random tensors from the ONNX input metadata",
    )
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument(
        "--shape-overrides",
        help="Static shapes for dynamic inputs, e.g. 'x:1,39,80;state:1,2,3'",
    )
    parser.add_argument("--float-low", type=float, default=-1.0)
    parser.add_argument("--float-high", type=float, default=1.0)
    parser.add_argument("--integer-low", type=int, default=0)
    parser.add_argument("--integer-high", type=int, default=32)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if not args.model.is_file():
        raise SystemExit(f"ONNX model not found: {args.model}")

    session = _load_session(args.model)
    if args.output_root is None:
        print("ONNX runtime inputs:")
        for index, node_arg in enumerate(session.get_inputs()):
            print(f"{index:03d}  {node_arg.name}  {node_arg.type}  {node_arg.shape}")
        return 0

    writer = CalibrationBinWriter.from_session(
        session,
        output_root=args.output_root,
        limit=args.limit,
    )

    if args.random:
        try:
            shape_overrides = _parse_shape_overrides(args.shape_overrides)
            write_random_samples(
                writer,
                seed=args.seed,
                shape_overrides=shape_overrides,
                float_low=args.float_low,
                float_high=args.float_high,
                integer_low=args.integer_low,
                integer_high=args.integer_high,
            )
        except (TypeError, ValueError) as exc:
            raise SystemExit(str(exc)) from exc
    elif args.npz_dir is not None:
        npz_files = sorted(args.npz_dir.glob("*.npz"))
        if not npz_files:
            raise SystemExit(f"No .npz feed snapshots found in: {args.npz_dir}")

        for npz_path in npz_files:
            if writer.full:
                break
            with np.load(npz_path, allow_pickle=False) as snapshot:
                writer.write({name: snapshot[name] for name in snapshot.files})
    else:
        raise SystemExit("Choose --random or --npz-dir when --output-root is set")

    print(f"Wrote {writer.count} paired calibration samples to {writer.output_root}")
    print("Use this config entry:")
    print(f"calibrate_path={writer.calibrate_path}")
    print(f"calibrate_size={writer.count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
