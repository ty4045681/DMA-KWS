#!/usr/bin/env python3
"""Convert Step A phoneme-adapter Lightning checkpoints to ``.pt``.

The Step A trainer already exports ``adapter_best_step*.pt`` at the end of a
completed run; this script covers interrupted runs and intermediate ``.ckpt``
files that were never exported.

Examples:

  # Convert one checkpoint with the exact training config.
  python scripts/convert_phoneme_adapter_checkpoints.py \
      --checkpoint exp/phoneme_adapter/checkpoints/adapter_0010000_0.1234.ckpt \
      --experiment paper_ls460

  # Convert last.ckpt of an interrupted run with a fully resolved config.
  python scripts/convert_phoneme_adapter_checkpoints.py \
      --checkpoint exp/phoneme_adapter/checkpoints/last.ckpt \
      --config outputs/train_ctc_adapter/config_resolved.yaml

  # Recursively convert a directory while preserving its relative layout.
  python scripts/convert_phoneme_adapter_checkpoints.py \
      --input-dir exp/phoneme_adapter/checkpoints \
      --recursive \
      --output-dir exp/converted \
      --experiment paper_ls460

A config source is mandatory: Step A ``.ckpt`` files do not embed the training
config, and the exported ``.pt`` must carry it so ``assert_stream_policy_matches``
can verify the streaming operating point at load time. The adapter is rebuilt
from the config and strict-loaded, so a config from the wrong experiment fails
here instead of inside Stage II.
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from typing import Any, Sequence

from omegaconf import DictConfig, OmegaConf

# Make direct ``python scripts/...`` and executable invocation work even when
# the project has not been installed into the active environment.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dma_kws.config import compose_config, config_to_dict
from dma_kws.phoneme_adapter.convert import (
    PhonemeAdapterConversionError,
    convert_phoneme_adapter_checkpoint,
)
from scripts.convert_stage2_checkpoints import discover_checkpoints


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Convert Step A phoneme-adapter Lightning .ckpt files to the "
            "adapter .pt export format."
        )
    )
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument(
        "--checkpoint",
        action="append",
        type=Path,
        help="Specific .ckpt to convert; repeat for multiple files.",
    )
    inputs.add_argument(
        "--input-dir",
        type=Path,
        help="Directory whose matching .ckpt files should be converted.",
    )
    parser.add_argument(
        "--pattern",
        default="*.ckpt",
        help="Glob used with --input-dir (default: *.ckpt).",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Use a recursive glob and preserve relative subdirectories.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Exact output path; valid only for one --checkpoint.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output root. Without it, each .pt is written beside its .ckpt.",
    )

    config_source = parser.add_mutually_exclusive_group(required=True)
    config_source.add_argument(
        "--config",
        type=Path,
        help="Fully resolved YAML config from the Step A training run.",
    )
    config_source.add_argument(
        "--experiment",
        help="Hydra experiment used by the training run, e.g. ctc_adapter_icefall.",
    )
    config_source.add_argument(
        "--default-config",
        action="store_true",
        help="Use the default config; only valid when training ran with defaults.",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Hydra override for the config fallback; repeatable and incompatible "
        "with --config.",
    )

    parser.add_argument(
        "--step",
        type=int,
        help=(
            "Explicit training step recorded in the export; valid only for one "
            "--checkpoint. Default: the checkpoint's global_step, then the step "
            "encoded in an 'adapter_<step>_*' filename."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing output files.",
    )
    return parser


def _resolve_config(args: argparse.Namespace) -> dict[str, Any]:
    try:
        if args.config is not None:
            if args.override:
                raise PhonemeAdapterConversionError(
                    "--override cannot be combined with --config; provide an already "
                    "resolved YAML config"
                )
            loaded = OmegaConf.load(args.config)
            if not isinstance(loaded, DictConfig):
                raise PhonemeAdapterConversionError(
                    f"Config must be a mapping: {args.config}"
                )
            return config_to_dict(loaded)
        return config_to_dict(
            compose_config(experiment=args.experiment, overrides=args.override)
        )
    except PhonemeAdapterConversionError:
        raise
    except Exception as exc:
        raise PhonemeAdapterConversionError(
            f"Failed to resolve training config: {exc}"
        ) from exc


def _output_for(
    source: Path,
    *,
    single_output: Path | None,
    output_dir: Path | None,
    input_dir: Path | None,
) -> Path:
    if single_output is not None:
        return single_output
    if output_dir is None:
        return source.with_suffix(".pt")
    if input_dir is not None:
        # Preserve the path as discovered under input_dir. Resolving symlinks
        # here can make a perfectly valid entry appear to escape the tree.
        relative = source.relative_to(input_dir)
        return output_dir / relative.with_suffix(".pt")
    return output_dir / source.with_suffix(".pt").name


def _validate_cli(args: argparse.Namespace, sources: Sequence[Path]) -> None:
    if args.output is not None:
        if len(sources) != 1:
            raise PhonemeAdapterConversionError(
                "--output is valid only when exactly one --checkpoint is supplied"
            )
        if args.output_dir is not None:
            raise PhonemeAdapterConversionError(
                "--output and --output-dir are mutually exclusive"
            )
    if args.step is not None and len(sources) != 1:
        raise PhonemeAdapterConversionError(
            "--step is valid only when exactly one --checkpoint is supplied"
        )


def run(args: argparse.Namespace) -> list[Path]:
    sources = discover_checkpoints(
        checkpoints=args.checkpoint,
        input_dir=args.input_dir,
        pattern=args.pattern,
        recursive=args.recursive,
    )
    _validate_cli(args, sources)
    config = _resolve_config(args)
    output_paths = [
        _output_for(
            source,
            single_output=args.output,
            output_dir=args.output_dir,
            input_dir=args.input_dir,
        )
        for source in sources
    ]
    if len({path.resolve() for path in output_paths}) != len(output_paths):
        raise PhonemeAdapterConversionError(
            "Multiple checkpoints map to the same output path; use distinct input "
            "names or a directory layout that can be preserved"
        )

    results: list[Path] = []
    failures: list[tuple[Path, Exception]] = []
    for source, output_path in zip(sources, output_paths):
        try:
            result = convert_phoneme_adapter_checkpoint(
                source,
                output_path,
                config=copy.deepcopy(config),
                step=args.step,
                overwrite=args.overwrite,
            )
        except Exception as exc:
            failures.append((source, exc))
            print(f"FAILED {source}: {exc}")
            continue
        results.append(result)
        print(f"{source} -> {result}")

    print(
        f"Converted {len(results)}/{len(sources)} checkpoint(s); "
        f"failed={len(failures)}"
    )
    if failures:
        failed_paths = ", ".join(str(path) for path, _ in failures)
        raise PhonemeAdapterConversionError(f"Conversion failed for: {failed_paths}")
    return results


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        run(args)
    except Exception as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
