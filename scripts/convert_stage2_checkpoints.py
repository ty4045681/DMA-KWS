#!/usr/bin/env python3
"""Convert one or many Stage II/LoRA Lightning checkpoints to ``.pt``.

Examples:

  # A new self-describing checkpoint needs no config arguments.
  python scripts/convert_stage2_checkpoints.py \
      --checkpoint exp/stage2/checkpoints/step_001000.ckpt

  # An early v5 checkpoint without embedded config needs its exact config.
  python scripts/convert_stage2_checkpoints.py \
      --checkpoint exp/stage2/checkpoints/step_001000.ckpt \
      --experiment paper_ls460 \
      --override stage1.stream.chunk_size=16

  # Recursively convert a directory while preserving its relative layout.
  python scripts/convert_stage2_checkpoints.py \
      --input-dir exp/stage2_adapt \
      --recursive \
      --output-dir exp/converted \
      --experiment adapt_hey_eva

LoRA checkpoints produce both a merged full-model ``.pt`` and an
``.adapter.pt`` by default.  Use ``--lora-output`` to select one.
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
from dma_kws.training.adapt_params import (
    load_adapt_params_file,
    merge_adapt_params,
)
from dma_kws.training.checkpoint_convert import (
    CheckpointConversionError,
    ConversionResult,
    convert_checkpoint,
    inspect_checkpoint_kind,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Convert Stage II and LoRA Lightning .ckpt files to the project's "
            "inference .pt format."
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

    config_source = parser.add_mutually_exclusive_group()
    config_source.add_argument(
        "--config",
        type=Path,
        help=(
            "Fully resolved YAML config for a v5 checkpoint without embedded config. Ignored when "
            "the checkpoint already embeds config."
        ),
    )
    config_source.add_argument(
        "--experiment",
        help=(
            "Hydra experiment used by a v5 checkpoint without embedded config. "
            "Ignored when the checkpoint already embeds config."
        ),
    )
    config_source.add_argument(
        "--default-config",
        action="store_true",
        help="Use the default config for a v5 checkpoint without embedded config.",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "Hydra override for a v5 config fallback; repeatable and ignored "
            "when the checkpoint embeds config."
        ),
    )
    parser.add_argument(
        "--adapt-params",
        type=Path,
        help=(
            "Adaptation params for a v5 config fallback; ignored when the "
            "checkpoint embeds config."
        ),
    )

    parser.add_argument(
        "--lora-output",
        choices=("merged", "adapter", "both"),
        default="both",
        help=(
            "LoRA artifact(s) to write (default: both). Adapter output still "
            "requires a complete frozen Stage II base in the checkpoint so its "
            "identity can be verified."
        ),
    )
    parser.add_argument(
        "--adapter-output",
        type=Path,
        help="Exact adapter path in single-checkpoint --lora-output=both mode.",
    )
    parser.add_argument(
        "--lora-alpha",
        type=float,
        help="Explicit LoRA alpha, checked against any checkpoint/config metadata.",
    )
    parser.add_argument(
        "--weights",
        choices=("state_dict", "current_model_state"),
        default="state_dict",
        help=(
            "Weight set to export. state_dict is the averaged/EMA state when "
            "Lightning WeightAveraging is enabled (default: state_dict)."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing output files.",
    )
    parser.add_argument(
        "--skip-tokenizer-validation",
        action="store_true",
        help=(
            "Do not require tokenizer.dict_path to exist locally. The vocab size "
            "is still checked against the QbyT embedding."
        ),
    )
    parser.add_argument(
        "--skip-encoder-validation",
        action="store_true",
        help=(
            "Skip rebuilding and strict-loading the encoder. Use only when the "
            "training backend dependencies are unavailable locally."
        ),
    )
    return parser


def discover_checkpoints(
    *,
    checkpoints: Sequence[Path] | None,
    input_dir: Path | None,
    pattern: str,
    recursive: bool,
) -> list[Path]:
    """Resolve deterministic, de-duplicated checkpoint inputs."""
    if checkpoints:
        candidates = list(checkpoints)
    else:
        if input_dir is None or not input_dir.is_dir():
            raise CheckpointConversionError(f"Input directory not found: {input_dir}")
        iterator = input_dir.rglob(pattern) if recursive else input_dir.glob(pattern)
        candidates = sorted(path for path in iterator if path.is_file())

    unique: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        if not candidate.is_file():
            raise CheckpointConversionError(f"Checkpoint not found: {candidate}")
        if candidate.suffix != ".ckpt":
            raise CheckpointConversionError(f"Input must end in .ckpt: {candidate}")
        seen.add(resolved)
        unique.append(candidate)
    if not unique:
        location = input_dir if input_dir is not None else "the supplied paths"
        raise CheckpointConversionError(
            f"No checkpoints matched {pattern!r} in {location}"
        )
    return unique


def _fallback_config(args: argparse.Namespace) -> dict[str, Any] | None:
    try:
        if args.config is not None:
            if args.override:
                raise CheckpointConversionError(
                    "--override cannot be combined with --config; provide an already "
                    "resolved YAML config"
                )
            loaded = OmegaConf.load(args.config)
            if not isinstance(loaded, DictConfig):
                raise CheckpointConversionError(
                    f"Config must be a mapping: {args.config}"
                )
            config = config_to_dict(loaded)
        elif args.experiment is not None or args.default_config or args.override:
            composed = compose_config(
                experiment=args.experiment,
                overrides=args.override,
            )
            config = config_to_dict(composed)
        else:
            config = None

        if args.adapt_params is not None:
            if config is None:
                raise CheckpointConversionError(
                    "--adapt-params needs --config, --experiment, or --default-config"
                )
            adapt = config.get("adapt")
            if not isinstance(adapt, dict):
                raise CheckpointConversionError("Resolved config has no adapt mapping")
            merge_adapt_params(
                adapt,
                load_adapt_params_file(args.adapt_params),
            )
        return config
    except CheckpointConversionError:
        raise
    except Exception as exc:
        raise CheckpointConversionError(f"Failed to resolve training config: {exc}") from exc


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
    if args.output is not None and len(sources) != 1:
        raise CheckpointConversionError(
            "--output is valid only when exactly one --checkpoint is supplied"
        )
    if args.output is not None and args.output_dir is not None:
        raise CheckpointConversionError("--output and --output-dir are mutually exclusive")
    if args.adapter_output is not None:
        if len(sources) != 1 or args.lora_output != "both":
            raise CheckpointConversionError(
                "--adapter-output requires one checkpoint and --lora-output=both"
            )
    if args.output is not None and args.output.suffix != ".pt":
        raise CheckpointConversionError(f"Output must end in .pt: {args.output}")


def run(args: argparse.Namespace) -> list[ConversionResult]:
    sources = discover_checkpoints(
        checkpoints=args.checkpoint,
        input_dir=args.input_dir,
        pattern=args.pattern,
        recursive=args.recursive,
    )
    _validate_cli(args, sources)
    fallback_config = _fallback_config(args)
    results: list[ConversionResult] = []
    failures: list[tuple[Path, Exception]] = []
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
        raise CheckpointConversionError(
            "Multiple checkpoints map to the same output path; use distinct input "
            "names or a directory layout that can be preserved"
        )
    if args.lora_output == "both":
        primary_owners = {
            path.resolve(): source
            for source, path in zip(sources, output_paths)
        }
        for source, primary in zip(sources, output_paths):
            adapter = primary.with_name(f"{primary.stem}.adapter.pt")
            conflicting_source = primary_owners.get(adapter.resolve())
            if (
                conflicting_source is not None
                and conflicting_source.resolve() != source.resolve()
                and inspect_checkpoint_kind(source, weights_key=args.weights) == "lora"
            ):
                raise CheckpointConversionError(
                    f"LoRA adapter output {adapter} for {source} collides with the "
                    f"full-model output for {conflicting_source}"
                )

    for source, output_path in zip(sources, output_paths):
        try:
            result = convert_checkpoint(
                source,
                output_path,
                fallback_config=copy.deepcopy(fallback_config),
                lora_output=args.lora_output,
                adapter_output_path=args.adapter_output,
                lora_alpha=args.lora_alpha,
                weights_key=args.weights,
                overwrite=args.overwrite,
                validate_tokenizer=not args.skip_tokenizer_validation,
                validate_encoder=not args.skip_encoder_validation,
            )
        except Exception as exc:
            failures.append((source, exc))
            print(f"FAILED {source}: {exc}")
            continue

        results.append(result)
        destinations = ", ".join(str(path) for path in result.outputs)
        print(f"[{result.kind}] {source} -> {destinations}")

    print(
        f"Converted {len(results)}/{len(sources)} checkpoint(s); "
        f"failed={len(failures)}"
    )
    if failures:
        failed_paths = ", ".join(str(path) for path, _ in failures)
        raise CheckpointConversionError(f"Conversion failed for: {failed_paths}")
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
