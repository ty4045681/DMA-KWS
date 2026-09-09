#!/usr/bin/env python3
"""Evaluate a model-by-noise-condition matrix with resumable summaries."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dma_kws.config import compose_config, config_to_dict, get_tokenizer_config
from dma_kws.inference.keyword_set import (
    KeywordEvalConfigError,
    KeywordSetManifestError,
    keyword_eval_mode,
    resolve_keyword_set,
)
from dma_kws.inference.manifest import (
    iter_audio_files,
    load_keyword_set_manifest,
    load_manifest,
)
from dma_kws.pathing import resolve_dict_path
from dma_kws.tokenizer import load_char_tokenizer


EVAL_SCRIPT = PROJECT_ROOT / "scripts" / "eval_stage2_clips.py"
CONFIG_ROOT = PROJECT_ROOT / "configs"
STATUS_FILENAME = "status.json"
SUMMARY_FILENAME = "summary.json"
RESULTS_FILENAME = "results.jsonl"
RUN_LOG_FILENAME = "run.log"
MATRIX_JSON_FILENAME = "matrix_summary.json"
MATRIX_CSV_FILENAME = "matrix_summary.csv"
_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_FINGERPRINT_SCHEMA_VERSION = 2
_RESERVED_OVERRIDE_KEYS = {
    "+experiment",
    "experiment",
    "+eval_condition",
    "eval_condition",
    "prep.manifest",
    "prep.stage2_ckpt",
    "prep.output_dir",
    "prep.audio_export.mode",
    "prep.audio_export.count",
    "prep.audio_export.seed",
    "prep.musan_mix.seed",
    "prep.audio_aug.seed",
    "run.device",
}
_RESERVED_OVERRIDE_PREFIXES = (
    "prep.audio_export.",
    "prep.musan_mix.",
    "prep.audio_aug.",
)


@dataclass(frozen=True)
class ConditionSpec:
    name: str
    family: str
    snr_db: float | None = None
    music_snr_db: float | None = None
    speech_relative_db: float | None = None
    requires_noise: bool = False
    requires_music: bool = False
    requires_speech: bool = False


@dataclass(frozen=True)
class AudioExportSpec:
    mode: str
    count: int
    seed: int

    @property
    def requested_count(self) -> int | str:
        if self.mode == "all":
            return "all"
        return self.count


CONDITIONS: tuple[ConditionSpec, ...] = (
    ConditionSpec("clean", "clean"),
    ConditionSpec("volume_variation", "volume_variation"),
    ConditionSpec("stationary_snr10", "stationary_noise", snr_db=10.0),
    ConditionSpec("stationary_snr20", "stationary_noise", snr_db=20.0),
    ConditionSpec(
        "burst_snr10", "burst_noise", snr_db=10.0, requires_noise=True
    ),
    ConditionSpec(
        "burst_snr20", "burst_noise", snr_db=20.0, requires_noise=True
    ),
    ConditionSpec(
        "musan_noise_snr10", "musan_noise", snr_db=10.0, requires_noise=True
    ),
    ConditionSpec(
        "musan_noise_snr20", "musan_noise", snr_db=20.0, requires_noise=True
    ),
    ConditionSpec(
        "musan_noise_snr10_music_snr10",
        "musan_noise_music",
        snr_db=10.0,
        music_snr_db=10.0,
        requires_noise=True,
        requires_music=True,
    ),
    ConditionSpec(
        "musan_noise_snr20_music_snr20",
        "musan_noise_music",
        snr_db=20.0,
        music_snr_db=20.0,
        requires_noise=True,
        requires_music=True,
    ),
    ConditionSpec(
        "musan_noise_snr10_speech_equal",
        "musan_noise_speech",
        snr_db=10.0,
        speech_relative_db=0.0,
        requires_noise=True,
        requires_speech=True,
    ),
    ConditionSpec(
        "musan_noise_snr20_speech_equal",
        "musan_noise_speech",
        snr_db=20.0,
        speech_relative_db=0.0,
        requires_noise=True,
        requires_speech=True,
    ),
    ConditionSpec(
        "musan_speech_quieter",
        "musan_speech",
        speech_relative_db=-6.0,
        requires_speech=True,
    ),
    ConditionSpec(
        "musan_speech_equal",
        "musan_speech",
        speech_relative_db=0.0,
        requires_speech=True,
    ),
    ConditionSpec(
        "musan_speech_louder",
        "musan_speech",
        speech_relative_db=6.0,
        requires_speech=True,
    ),
)
CONDITION_BY_NAME = {condition.name: condition for condition in CONDITIONS}


@dataclass(frozen=True)
class ModelSpec:
    name: str
    checkpoint: Path
    experiment: str
    overrides: tuple[str, ...] = ()


@dataclass(frozen=True)
class JobSpec:
    model: ModelSpec
    condition: ConditionSpec
    output_dir: Path
    fingerprint: str
    command: tuple[str, ...]
    audio_export: AudioExportSpec = AudioExportSpec(
        mode="disabled",
        count=0,
        seed=2025,
    )


class BatchConfigError(ValueError):
    """Raised when a batch request is unsafe or internally inconsistent."""


def _parse_audio_export(value: str) -> int | str:
    normalized = str(value).strip().lower()
    if normalized in {"all", "none"}:
        return normalized
    try:
        count = int(normalized)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "--export-audio expects a positive integer, all, or none"
        ) from exc
    if count < 0:
        raise argparse.ArgumentTypeError(
            "--export-audio expects a positive integer, all, or none"
        )
    return "none" if count == 0 else count


def resolve_audio_export(
    value: int | str,
    *,
    seed: int,
    export_seed: int | None,
) -> AudioExportSpec:
    resolved_seed = seed if export_seed is None else export_seed
    if resolved_seed < 0:
        raise BatchConfigError("--export-audio-seed must be >= 0")
    if value == "all":
        return AudioExportSpec(mode="all", count=0, seed=resolved_seed)
    if value == "none" or value == 0:
        return AudioExportSpec(mode="disabled", count=0, seed=resolved_seed)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise BatchConfigError(
            "--export-audio expects a positive integer, all, or none"
        )
    return AudioExportSpec(mode="random", count=value, seed=resolved_seed)


METRIC_FIELDS = (
    "threshold",
    "accuracy",
    "precision",
    "recall",
    "fpr",
    "fnr",
    "f1",
    "auc",
    "eer",
    "eer_threshold",
    "tp",
    "tn",
    "fp",
    "fn",
)

CSV_FIELDS = (
    "model",
    "condition",
    "family",
    "snr_db",
    "music_snr_db",
    "speech_relative_db",
    "status",
    "attempt",
    "checkpoint",
    "checkpoint_sha256",
    "fingerprint",
    "experiment",
    "output_dir",
    "summary_path",
    "returncode",
    "elapsed_seconds",
    "error",
    "num_samples",
    "num_skipped",
    "audio_export_mode",
    "audio_export_requested_count",
    "num_audio_exported",
    "audio_export_dir",
    "audio_export_manifest",
    *METRIC_FIELDS,
)


def _validate_safe_name(value: str, *, field: str) -> str:
    normalized = value.strip()
    if not _SAFE_NAME.fullmatch(normalized) or normalized in {".", ".."}:
        raise BatchConfigError(
            f"{field} must match {_SAFE_NAME.pattern!r}, got {value!r}"
        )
    return normalized


def _split_assignment(raw: str, *, option: str) -> tuple[str, str]:
    name, separator, value = raw.partition("=")
    if not separator or not name.strip() or not value.strip():
        raise BatchConfigError(f"{option} expects NAME=VALUE, got {raw!r}")
    return _validate_safe_name(name, field=f"{option} name"), value.strip()


def parse_model_argument(raw: str, *, default_experiment: str) -> ModelSpec:
    """Parse ``NAME=CHECKPOINT`` or ``NAME=CHECKPOINT::EXPERIMENT``."""

    name, value = _split_assignment(raw, option="--model")
    checkpoint_value = value
    experiment = default_experiment
    if "::" in value:
        checkpoint_value, experiment = value.rsplit("::", 1)
        checkpoint_value = checkpoint_value.strip()
        experiment = experiment.strip()
        if not checkpoint_value or not experiment:
            raise BatchConfigError(
                "--model expects NAME=CHECKPOINT or "
                f"NAME=CHECKPOINT::EXPERIMENT, got {raw!r}"
            )
    experiment = _validate_safe_name(experiment, field="experiment")
    return ModelSpec(
        name=name,
        checkpoint=Path(checkpoint_value).expanduser(),
        experiment=experiment,
    )


def _read_model_file(path: Path) -> list[str]:
    if not path.is_file():
        raise BatchConfigError(f"--models-file not found: {path}")
    entries: list[str] = []
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            name, value = _split_assignment(line, option=f"{path}:{line_number}")
        except BatchConfigError as exc:
            raise BatchConfigError(
                f"{path}:{line_number} expects NAME=CHECKPOINT[::EXPERIMENT]"
            ) from exc
        checkpoint_value = value
        experiment_suffix = ""
        if "::" in value:
            checkpoint_value, experiment = value.rsplit("::", 1)
            checkpoint_value = checkpoint_value.strip()
            experiment_suffix = f"::{experiment.strip()}"
        checkpoint = Path(checkpoint_value).expanduser()
        if not checkpoint.is_absolute():
            checkpoint = (path.parent / checkpoint).resolve()
        entries.append(f"{name}={checkpoint}{experiment_suffix}")
    return entries


def _named_values(
    values: Iterable[str], *, option: str, require_nested_assignment: bool = False
) -> dict[str, list[str]]:
    parsed: dict[str, list[str]] = {}
    for raw in values:
        name, value = _split_assignment(raw, option=option)
        if require_nested_assignment and "=" not in value:
            raise BatchConfigError(f"{option} expects NAME=KEY=VALUE, got {raw!r}")
        if require_nested_assignment:
            try:
                _validate_forwarded_override(value, option=option)
            except argparse.ArgumentTypeError as exc:
                raise BatchConfigError(str(exc)) from exc
        parsed.setdefault(name, []).append(value)
    return parsed


def resolve_models(args: argparse.Namespace) -> list[ModelSpec]:
    raw_models = list(args.model or [])
    if args.models_file is not None:
        raw_models.extend(_read_model_file(args.models_file.expanduser()))
    if not raw_models:
        raise BatchConfigError("at least one --model or --models-file entry is required")

    experiment_values = _named_values(
        args.model_experiment or [], option="--model-experiment"
    )
    override_values = _named_values(
        args.model_override or [],
        option="--model-override",
        require_nested_assignment=True,
    )
    models: list[ModelSpec] = []
    seen: set[str] = set()
    for raw in raw_models:
        model = parse_model_argument(raw, default_experiment=args.experiment)
        if model.name in seen:
            raise BatchConfigError(f"duplicate model name: {model.name}")
        seen.add(model.name)
        configured_experiments = experiment_values.pop(model.name, [])
        if len(configured_experiments) > 1:
            raise BatchConfigError(
                f"multiple --model-experiment values for {model.name}"
            )
        experiment = (
            _validate_safe_name(
                configured_experiments[0], field=f"experiment for {model.name}"
            )
            if configured_experiments
            else model.experiment
        )
        models.append(
            ModelSpec(
                name=model.name,
                checkpoint=model.checkpoint.expanduser().resolve(),
                experiment=experiment,
                overrides=tuple(override_values.pop(model.name, [])),
            )
        )

    unknown = sorted({*experiment_values, *override_values})
    if unknown:
        raise BatchConfigError(
            "model-specific options reference unknown models: " + ", ".join(unknown)
        )
    return models


def resolve_conditions(names: Sequence[str] | None) -> list[ConditionSpec]:
    requested = list(names) if names else [condition.name for condition in CONDITIONS]
    if len(set(requested)) != len(requested):
        raise BatchConfigError("--condition values must not contain duplicates")
    unknown = [name for name in requested if name not in CONDITION_BY_NAME]
    if unknown:
        raise BatchConfigError("unknown conditions: " + ", ".join(unknown))
    return [CONDITION_BY_NAME[name] for name in requested]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_identity(path: Path) -> dict[str, str | int]:
    resolved = path.expanduser().resolve()
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": _sha256_file(resolved),
    }


def _manifest_audio_identity_from_rows(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    entries = []
    for row in rows:
        path = Path(str(row["audio_path"])).resolve()
        stat = path.stat()
        entries.append(
            {
                "path": str(path),
                "size_bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
        )
    identity: dict[str, Any] = {"files": entries}
    identity["sha256"] = _canonical_digest(identity)
    return identity


def _manifest_audio_identity(manifest: Path) -> dict[str, Any]:
    return _manifest_audio_identity_from_rows(load_manifest(manifest))


def _resolved_from_cfg(cfg: Any) -> dict[str, Any]:
    from omegaconf import OmegaConf

    resolved = config_to_dict(cfg)
    prep = OmegaConf.to_container(cfg.prep, resolve=True)
    if isinstance(prep, dict):
        resolved["prep"] = prep
    return resolved


def _compose_model_config(
    model: ModelSpec,
    common_overrides: Sequence[str],
) -> dict[str, Any]:
    # Later Hydra overrides win. Match build_eval_command: common, then model.
    try:
        cfg = compose_config(
            model.experiment,
            overrides=[*list(common_overrides), *list(model.overrides)],
        )
        return _resolved_from_cfg(cfg)
    except Exception as exc:
        raise BatchConfigError(
            f"failed to compose config for model {model.name!r}: {exc}"
        ) from exc


def _load_matrix_manifest(
    manifest: Path,
    resolved_config: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], Any]:
    prep = resolved_config.get("prep")
    if not isinstance(prep, Mapping):
        prep = {}
    try:
        mode = keyword_eval_mode(prep)
    except KeywordEvalConfigError as exc:
        raise BatchConfigError(str(exc)) from exc
    if mode != "any":
        try:
            rows = load_manifest(manifest)
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise BatchConfigError(f"invalid manifest {manifest}: {exc}") from exc
        return rows, None
    try:
        dict_path = resolve_dict_path(resolved_config)
        tokenizer_cfg = get_tokenizer_config(dict(resolved_config))
        tokenizer = load_char_tokenizer(
            dict_path,
            split_with_space=tokenizer_cfg.get("split_with_space", " "),
        )
        keyword_set = resolve_keyword_set(
            prep,
            tokenizer,
            tokenizer_dict_path=dict_path,
        )
        if keyword_set is None:
            raise BatchConfigError("prep.keyword_eval.mode=any produced no keyword set")
        rows = load_keyword_set_manifest(manifest, keyword_set.texts)
    except (
        KeywordEvalConfigError,
        KeywordSetManifestError,
        OSError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:
        raise BatchConfigError(f"invalid any-mode manifest {manifest}: {exc}") from exc
    return rows, keyword_set


def _validate_experiment(experiment: str) -> None:
    path = CONFIG_ROOT / "experiment" / f"{experiment}.yaml"
    if not path.is_file():
        raise BatchConfigError(f"Hydra experiment not found: {experiment} ({path})")


def _validate_full_stage2_checkpoint(path: Path, *, model_name: str) -> None:
    try:
        import torch

        checkpoint = torch.load(path, map_location="cpu")
    except Exception as exc:
        raise BatchConfigError(
            f"failed to read checkpoint for model {model_name!r}: {path}: {exc}"
        ) from exc
    if not isinstance(checkpoint, Mapping):
        raise BatchConfigError(
            f"checkpoint for model {model_name!r} must contain a mapping: {path}"
        )
    checkpoint_kind = str(checkpoint.get("checkpoint_kind", ""))
    model_state = checkpoint.get("model_state_dict")
    adapter_only = checkpoint_kind == "stage2_lora_adapter" or (
        isinstance(checkpoint.get("lora_state_dict"), Mapping)
        and not isinstance(model_state, Mapping)
    )
    if adapter_only:
        raise BatchConfigError(
            f"model {model_name!r} is an adapter-only LoRA checkpoint; use the "
            f"merged Stage II .pt checkpoint instead: {path}"
        )
    state = model_state if isinstance(model_state, Mapping) else checkpoint
    keys = {str(key) for key in state}
    if not any(key.startswith("encoder.") for key in keys) or not any(
        key.startswith("qbyt.") for key in keys
    ):
        raise BatchConfigError(
            f"model {model_name!r} is not a full Stage II checkpoint with "
            f"encoder.* and qbyt.* weights: {path}"
        )


def _require_binary_labels(rows: Sequence[Mapping[str, Any]], *, manifest: Path) -> None:
    if not rows:
        raise BatchConfigError(f"manifest is empty: {manifest}")
    missing_labels = [index + 1 for index, row in enumerate(rows) if "label" not in row]
    if missing_labels:
        preview = ", ".join(str(index) for index in missing_labels[:5])
        raise BatchConfigError(
            "every manifest row must have a binary label; missing at data rows "
            f"{preview}"
        )
    labels = {int(row["label"]) for row in rows}
    if not labels <= {0, 1}:
        raise BatchConfigError(
            f"manifest labels must be 0 or 1, got {sorted(labels)}"
        )
    if labels != {0, 1}:
        raise BatchConfigError(
            "manifest must contain both label=0 and label=1 for ROC/DET evaluation"
        )
    missing_audio = [
        Path(str(row["audio_path"]))
        for row in rows
        if not Path(str(row["audio_path"])).is_file()
    ]
    if missing_audio:
        preview = ", ".join(str(path) for path in missing_audio[:3])
        suffix = " ..." if len(missing_audio) > 3 else ""
        raise BatchConfigError(f"manifest audio files not found: {preview}{suffix}")


def validate_inputs(
    *,
    manifest: Path,
    models: Sequence[ModelSpec],
    conditions: Sequence[ConditionSpec],
    musan_root: Path | None,
    common_overrides: Sequence[str] = (),
) -> list[dict[str, Any]]:
    if not manifest.is_file():
        raise BatchConfigError(f"manifest not found: {manifest}")
    rows: list[dict[str, Any]] | None = None
    for model in models:
        resolved = _compose_model_config(model, common_overrides)
        model_rows, _keyword_set = _load_matrix_manifest(manifest, resolved)
        _require_binary_labels(model_rows, manifest=manifest)
        if rows is None:
            rows = model_rows
    assert rows is not None

    for model in models:
        if not model.checkpoint.is_file():
            raise BatchConfigError(
                f"checkpoint for model {model.name!r} not found: {model.checkpoint}"
            )
        if model.checkpoint.suffix.lower() != ".pt":
            raise BatchConfigError(
                f"model {model.name!r} must use an exported/merged .pt checkpoint: "
                f"{model.checkpoint}"
            )
        if model.checkpoint.name.endswith(".adapter.pt"):
            raise BatchConfigError(
                f"model {model.name!r} points to adapter-only weights; use the "
                f"merged Stage II .pt checkpoint instead: {model.checkpoint}"
            )
        _validate_full_stage2_checkpoint(
            model.checkpoint,
            model_name=model.name,
        )
        _validate_experiment(model.experiment)

    needs_noise = any(condition.requires_noise for condition in conditions)
    needs_music = any(condition.requires_music for condition in conditions)
    needs_speech = any(condition.requires_speech for condition in conditions)
    if needs_noise or needs_music or needs_speech:
        if musan_root is None:
            raise BatchConfigError(
                "--musan-root is required by the selected burst/MUSAN conditions"
            )
        if not musan_root.is_dir():
            raise BatchConfigError(f"MUSAN root not found: {musan_root}")
        for subset, needed in (
            ("noise", needs_noise),
            ("music", needs_music),
            ("speech", needs_speech),
        ):
            subset_dir = musan_root / subset
            if needed and not subset_dir.is_dir():
                raise BatchConfigError(f"MUSAN subset not found: {subset_dir}")
            if needed and not iter_audio_files(subset_dir):
                raise BatchConfigError(
                    f"no supported audio files found under MUSAN subset: {subset_dir}"
                )

    for condition in conditions:
        path = CONFIG_ROOT / "eval_condition" / f"{condition.name}.yaml"
        if not path.is_file():
            raise BatchConfigError(
                f"Hydra eval condition not found: {condition.name} ({path})"
            )
    return rows


def _hydra_string(value: str | Path) -> str:
    return json.dumps(str(value), ensure_ascii=False)


def build_eval_command(
    *,
    model: ModelSpec,
    condition: ConditionSpec,
    manifest: Path,
    musan_root: Path | None,
    output_dir: Path,
    device: str,
    batch_size: int,
    num_workers: int,
    seed: int,
    common_overrides: Sequence[str],
    audio_export: AudioExportSpec | None = None,
) -> tuple[str, ...]:
    if audio_export is None:
        audio_export = AudioExportSpec(mode="random", count=5, seed=seed)
    command = [
        sys.executable,
        str(EVAL_SCRIPT),
        f"+experiment={model.experiment}",
        f"+eval_condition={condition.name}",
        *common_overrides,
        *model.overrides,
        f"prep.manifest={_hydra_string(manifest)}",
        f"prep.stage2_ckpt={_hydra_string(model.checkpoint)}",
        f"prep.output_dir={_hydra_string(output_dir)}",
        f"prep.musan_mix.seed={seed}",
        f"prep.audio_aug.seed={seed}",
        f"prep.audio_export.mode={audio_export.mode}",
        f"prep.audio_export.count={audio_export.count}",
        f"prep.audio_export.seed={audio_export.seed}",
        f"run.device={_hydra_string(device)}",
    ]
    if musan_root is not None:
        command.append(f"prep.musan_root={_hydra_string(musan_root)}")
    if batch_size > 0:
        command.append(f"prep.batch_size={batch_size}")
    if num_workers > 0:
        command.append(f"prep.num_workers={num_workers}")
    return tuple(command)


def _canonical_digest(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _evaluation_code_identity() -> dict[str, Any]:
    files = {EVAL_SCRIPT}
    for source_root in (PROJECT_ROOT / "dma_kws", PROJECT_ROOT / "qbyt"):
        files.update(source_root.rglob("*.py"))
    entries = [
        {
            "path": str(path.relative_to(PROJECT_ROOT)),
            "sha256": _sha256_file(path),
        }
        for path in sorted(files)
        if path.is_file()
    ]
    identity: dict[str, Any] = {"num_files": len(entries), "files": entries}
    identity["sha256"] = _canonical_digest(identity)
    return identity


def _resolved_job_config(
    *, model: ModelSpec, command: Sequence[str]
) -> dict[str, Any]:
    # command[2] is +experiment=...; compose_config adds that group itself.
    try:
        cfg = compose_config(model.experiment, overrides=command[3:])
        return _resolved_from_cfg(cfg)
    except Exception as exc:
        raise BatchConfigError(
            f"failed to compose config for model {model.name!r}: {exc}"
        ) from exc


def _musan_catalog_identity(
    musan_root: Path | None, conditions: Sequence[ConditionSpec]
) -> dict[str, Any] | None:
    if musan_root is None:
        return None
    needed_subsets = [
        subset
        for subset, needed in (
            ("noise", any(condition.requires_noise for condition in conditions)),
            ("music", any(condition.requires_music for condition in conditions)),
            ("speech", any(condition.requires_speech for condition in conditions)),
        )
        if needed
    ]
    catalog: dict[str, Any] = {"root": str(musan_root), "subsets": {}}
    for subset in needed_subsets:
        files = iter_audio_files(musan_root / subset)
        entries = []
        for path in files:
            stat = path.stat()
            entries.append(
                {
                    "path": str(path.relative_to(musan_root)),
                    "size_bytes": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                }
            )
        catalog["subsets"][subset] = entries
    catalog["sha256"] = _canonical_digest(catalog)
    return catalog


def _condition_musan_identity(
    catalog: Mapping[str, Any] | None, condition: ConditionSpec
) -> dict[str, Any] | None:
    if catalog is None or not (
        condition.requires_noise
        or condition.requires_music
        or condition.requires_speech
    ):
        return None
    all_subsets = catalog.get("subsets")
    if not isinstance(all_subsets, Mapping):
        raise BatchConfigError("invalid internal MUSAN catalog identity")
    subset_names = [
        subset
        for subset, needed in (
            ("noise", condition.requires_noise),
            ("music", condition.requires_music),
            ("speech", condition.requires_speech),
        )
        if needed
    ]
    selected = {
        "root": catalog.get("root"),
        "subsets": {subset: all_subsets[subset] for subset in subset_names},
    }
    selected["sha256"] = _canonical_digest(selected)
    return selected


def build_jobs(
    *,
    models: Sequence[ModelSpec],
    conditions: Sequence[ConditionSpec],
    manifest: Path,
    musan_root: Path | None,
    output_root: Path,
    device: str,
    batch_size: int,
    num_workers: int,
    seed: int,
    common_overrides: Sequence[str],
    audio_export: AudioExportSpec | None = None,
) -> tuple[list[JobSpec], dict[str, Any]]:
    if audio_export is None:
        audio_export = AudioExportSpec(mode="random", count=5, seed=seed)
    manifest_identity = _file_identity(manifest)
    audio_identity = None
    model_identities = {
        model.name: _file_identity(model.checkpoint) for model in models
    }
    musan_identity = _musan_catalog_identity(musan_root, conditions)
    runtime_identity = _evaluation_code_identity()
    tokenizer_identities: dict[Path, dict[str, str | int]] = {}
    jobs: list[JobSpec] = []
    for model in models:
        for condition in conditions:
            output_dir = output_root / model.name / condition.name
            condition_musan_identity = _condition_musan_identity(
                musan_identity, condition
            )
            command = build_eval_command(
                model=model,
                condition=condition,
                manifest=manifest,
                musan_root=musan_root,
                output_dir=output_dir,
                device=device,
                batch_size=batch_size,
                num_workers=num_workers,
                seed=seed,
                common_overrides=common_overrides,
                audio_export=audio_export,
            )
            resolved_config = _resolved_job_config(model=model, command=command)
            rows, keyword_set = _load_matrix_manifest(manifest, resolved_config)
            row_audio_identity = _manifest_audio_identity_from_rows(rows)
            if audio_identity is None:
                audio_identity = row_audio_identity
            tokenizer_path = resolve_dict_path(resolved_config).resolve()
            if not tokenizer_path.is_file():
                raise BatchConfigError(
                    f"tokenizer dictionary for model {model.name!r} not found: "
                    f"{tokenizer_path}"
                )
            if tokenizer_path not in tokenizer_identities:
                tokenizer_identities[tokenizer_path] = _file_identity(tokenizer_path)
            tokenizer_identity = tokenizer_identities[tokenizer_path]
            fingerprint_payload = {
                "schema_version": _FINGERPRINT_SCHEMA_VERSION,
                "manifest": manifest_identity,
                "manifest_audio": audio_identity,
                "model": {
                    **model_identities[model.name],
                    "name": model.name,
                    "experiment": model.experiment,
                    "overrides": list(model.overrides),
                },
                "condition": asdict(condition),
                "musan": condition_musan_identity,
                "resolved_config_sha256": _canonical_digest(resolved_config),
                "tokenizer": tokenizer_identity,
                "evaluation_code_sha256": runtime_identity["sha256"],
                "device": device,
                "batch_size": batch_size,
                "num_workers": num_workers,
                "seed": seed,
                "common_overrides": list(common_overrides),
                "audio_export": asdict(audio_export),
                "keyword_set_id": (
                    None if keyword_set is None else keyword_set.keyword_set_id
                ),
                "keyword_eval_mode": (
                    "per_row" if keyword_set is None else "any"
                ),
            }
            fingerprint = _canonical_digest(fingerprint_payload)
            jobs.append(
                JobSpec(
                    model=model,
                    condition=condition,
                    output_dir=output_dir,
                    fingerprint=fingerprint,
                    command=command,
                    audio_export=audio_export,
                )
            )
    if audio_identity is None:
        audio_identity = _manifest_audio_identity(manifest)
    identities = {
        "manifest": manifest_identity,
        "manifest_audio": audio_identity,
        "models": model_identities,
        "musan": musan_identity,
        "tokenizers": {
            str(path): identity for path, identity in tokenizer_identities.items()
        },
        "evaluation_code": runtime_identity,
        "audio_export": asdict(audio_export),
    }
    return jobs, identities


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _summary_checkpoint_sha256(summary: Mapping[str, Any]) -> str | None:
    provenance = summary.get("provenance")
    if not isinstance(provenance, Mapping):
        return None
    checkpoint = provenance.get("checkpoint")
    if not isinstance(checkpoint, Mapping):
        return None
    sha256 = checkpoint.get("sha256")
    return str(sha256) if sha256 is not None else None


def _result_line_count(path: Path) -> int | None:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return sum(bool(line.strip()) for line in handle)
    except OSError:
        return None


def _cached_audio_exports_valid(
    job: JobSpec,
    summary: Mapping[str, Any],
) -> bool:
    spec = job.audio_export
    exports = summary.get("audio_exports")
    if spec.mode == "disabled":
        return exports is None or (
            isinstance(exports, Mapping)
            and exports.get("status") == "disabled"
        )
    if not isinstance(exports, Mapping):
        return False
    if (
        exports.get("status") != "generated"
        or exports.get("mode") != spec.mode
        or exports.get("seed") != spec.seed
        or exports.get("requested_count") != spec.requested_count
    ):
        return False
    num_samples = summary.get("num_samples")
    if not isinstance(num_samples, int) or isinstance(num_samples, bool):
        return False
    expected = num_samples if spec.mode == "all" else min(spec.count, num_samples)
    if exports.get("num_exported") != expected:
        return False
    manifest_value = exports.get("manifest")
    if not isinstance(manifest_value, str) or not manifest_value:
        return False
    manifest_path = Path(manifest_value)
    if not manifest_path.is_absolute():
        manifest_path = job.output_dir / manifest_path
    try:
        lines = [
            json.loads(line)
            for line in manifest_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, json.JSONDecodeError):
        return False
    if len(lines) != expected or not all(isinstance(line, Mapping) for line in lines):
        return False
    for line in lines:
        exported_value = line.get("exported_audio_path")
        if not isinstance(exported_value, str) or not exported_value:
            return False
        exported_path = Path(exported_value)
        if not exported_path.is_absolute():
            exported_path = job.output_dir / exported_path
        if not exported_path.is_file():
            return False
    return True


def cached_summary(
    job: JobSpec, *, checkpoint_sha256: str
) -> dict[str, Any] | None:
    status = _read_json(job.output_dir / STATUS_FILENAME)
    summary = _read_json(job.output_dir / SUMMARY_FILENAME)
    if (
        status is None
        or status.get("status") != "succeeded"
        or status.get("fingerprint") != job.fingerprint
        or summary is None
    ):
        return None
    expected_samples = summary.get("num_samples")
    if not isinstance(expected_samples, int) or isinstance(expected_samples, bool):
        return None
    if _result_line_count(job.output_dir / RESULTS_FILENAME) != expected_samples:
        return None
    if _summary_checkpoint_sha256(summary) != checkpoint_sha256:
        return None
    if Path(str(summary.get("output_dir", ""))).resolve() != job.output_dir.resolve():
        return None
    if not _cached_audio_exports_valid(job, summary):
        return None
    return summary


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(
                payload,
                handle,
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _base_row(job: JobSpec, *, checkpoint_sha256: str) -> dict[str, Any]:
    return {
        "model": job.model.name,
        "condition": job.condition.name,
        "family": job.condition.family,
        "snr_db": job.condition.snr_db,
        "music_snr_db": job.condition.music_snr_db,
        "speech_relative_db": job.condition.speech_relative_db,
        "status": "pending",
        "attempt": 0,
        "checkpoint": str(job.model.checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "fingerprint": job.fingerprint,
        "experiment": job.model.experiment,
        "output_dir": str(job.output_dir),
        "summary_path": str(job.output_dir / SUMMARY_FILENAME),
        "returncode": None,
        "elapsed_seconds": None,
        "error": None,
        "num_samples": None,
        "num_skipped": None,
        "audio_export_mode": job.audio_export.mode,
        "audio_export_requested_count": job.audio_export.requested_count,
        "num_audio_exported": None,
        "audio_export_dir": None,
        "audio_export_manifest": None,
        **{field: None for field in METRIC_FIELDS},
    }


def _summary_row(
    job: JobSpec,
    summary: Mapping[str, Any],
    *,
    status: str,
    attempt: int,
    checkpoint_sha256: str,
    elapsed_seconds: float | None,
) -> dict[str, Any]:
    row = _base_row(job, checkpoint_sha256=checkpoint_sha256)
    row.update(
        {
            "status": status,
            "attempt": attempt,
            "returncode": 0,
            "elapsed_seconds": elapsed_seconds,
            "num_samples": summary.get("num_samples"),
            "num_skipped": summary.get("num_skipped"),
        }
    )
    metrics = summary.get("metrics")
    if isinstance(metrics, Mapping):
        for field in METRIC_FIELDS:
            row[field] = metrics.get(field)
    audio_exports = summary.get("audio_exports")
    if isinstance(audio_exports, Mapping):
        row["audio_export_mode"] = audio_exports.get(
            "mode", row["audio_export_mode"]
        )
        row["audio_export_requested_count"] = audio_exports.get(
            "requested_count", row["audio_export_requested_count"]
        )
        row["num_audio_exported"] = audio_exports.get("num_exported")
        row["audio_export_dir"] = audio_exports.get("directory")
        row["audio_export_manifest"] = audio_exports.get("manifest")
    return row


def _failure_row(
    job: JobSpec,
    *,
    status: str,
    attempt: int,
    checkpoint_sha256: str,
    returncode: int | None,
    elapsed_seconds: float | None,
    error: str,
) -> dict[str, Any]:
    row = _base_row(job, checkpoint_sha256=checkpoint_sha256)
    row.update(
        {
            "status": status,
            "attempt": attempt,
            "returncode": returncode,
            "elapsed_seconds": elapsed_seconds,
            "error": error,
        }
    )
    return row


def _previous_attempt(output_dir: Path) -> int:
    status = _read_json(output_dir / STATUS_FILENAME)
    if status is None:
        return 0
    attempt = status.get("attempt", 0)
    return attempt if isinstance(attempt, int) and attempt >= 0 else 0


def run_job(
    job: JobSpec,
    *,
    checkpoint_sha256: str,
    force: bool,
) -> dict[str, Any]:
    if not force:
        summary = cached_summary(job, checkpoint_sha256=checkpoint_sha256)
        if summary is not None:
            status = _read_json(job.output_dir / STATUS_FILENAME) or {}
            print(f"[cached] {job.model.name} / {job.condition.name}")
            return _summary_row(
                job,
                summary,
                status="cached",
                attempt=int(status.get("attempt", 0)),
                checkpoint_sha256=checkpoint_sha256,
                elapsed_seconds=status.get("elapsed_seconds"),
            )

    job.output_dir.mkdir(parents=True, exist_ok=True)
    attempt = _previous_attempt(job.output_dir) + 1
    started = time.time()
    status_path = job.output_dir / STATUS_FILENAME
    status_payload: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "attempt": attempt,
        "fingerprint": job.fingerprint,
        "model": job.model.name,
        "condition": job.condition.name,
        "command": list(job.command),
        "started_at_unix": started,
    }
    _atomic_json(status_path, status_payload)
    print(f"[run] {job.model.name} / {job.condition.name}")
    log_path = job.output_dir / RUN_LOG_FILENAME
    try:
        with log_path.open("w", encoding="utf-8") as log_handle:
            completed = subprocess.run(
                list(job.command),
                cwd=str(PROJECT_ROOT),
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
    except KeyboardInterrupt:
        elapsed = time.time() - started
        status_payload.update(
            {
                "status": "interrupted",
                "elapsed_seconds": elapsed,
                "finished_at_unix": time.time(),
                "error": "interrupted by user",
            }
        )
        _atomic_json(status_path, status_payload)
        raise
    except OSError as exc:
        elapsed = time.time() - started
        error = f"failed to start evaluator: {exc}"
        status_payload.update(
            {
                "status": "failed",
                "returncode": None,
                "elapsed_seconds": elapsed,
                "finished_at_unix": time.time(),
                "error": error,
            }
        )
        _atomic_json(status_path, status_payload)
        return _failure_row(
            job,
            status="failed",
            attempt=attempt,
            checkpoint_sha256=checkpoint_sha256,
            returncode=None,
            elapsed_seconds=elapsed,
            error=error,
        )

    elapsed = time.time() - started
    if completed.returncode != 0:
        error = f"evaluator exited with {completed.returncode}; see {log_path}"
        status_payload.update(
            {
                "status": "failed",
                "returncode": completed.returncode,
                "elapsed_seconds": elapsed,
                "finished_at_unix": time.time(),
                "error": error,
            }
        )
        _atomic_json(status_path, status_payload)
        print(f"[failed] {job.model.name} / {job.condition.name}: {error}")
        return _failure_row(
            job,
            status="failed",
            attempt=attempt,
            checkpoint_sha256=checkpoint_sha256,
            returncode=completed.returncode,
            elapsed_seconds=elapsed,
            error=error,
        )

    summary = _read_json(job.output_dir / SUMMARY_FILENAME)
    line_count = _result_line_count(job.output_dir / RESULTS_FILENAME)
    if summary is None or line_count != summary.get("num_samples"):
        error = (
            "evaluator returned success but summary.json/results.jsonl are "
            f"missing or inconsistent; see {log_path}"
        )
        status_payload.update(
            {
                "status": "failed",
                "returncode": completed.returncode,
                "elapsed_seconds": elapsed,
                "finished_at_unix": time.time(),
                "error": error,
            }
        )
        _atomic_json(status_path, status_payload)
        print(f"[failed] {job.model.name} / {job.condition.name}: {error}")
        return _failure_row(
            job,
            status="failed",
            attempt=attempt,
            checkpoint_sha256=checkpoint_sha256,
            returncode=completed.returncode,
            elapsed_seconds=elapsed,
            error=error,
        )

    if _summary_checkpoint_sha256(summary) != checkpoint_sha256:
        error = "evaluator summary checkpoint fingerprint does not match the requested model"
        status_payload.update(
            {
                "status": "failed",
                "returncode": completed.returncode,
                "elapsed_seconds": elapsed,
                "finished_at_unix": time.time(),
                "error": error,
            }
        )
        _atomic_json(status_path, status_payload)
        return _failure_row(
            job,
            status="failed",
            attempt=attempt,
            checkpoint_sha256=checkpoint_sha256,
            returncode=completed.returncode,
            elapsed_seconds=elapsed,
            error=error,
        )

    if not _cached_audio_exports_valid(job, summary):
        error = "evaluator audio export artifacts are missing or inconsistent"
        status_payload.update(
            {
                "status": "failed",
                "returncode": completed.returncode,
                "elapsed_seconds": elapsed,
                "finished_at_unix": time.time(),
                "error": error,
            }
        )
        _atomic_json(status_path, status_payload)
        return _failure_row(
            job,
            status="failed",
            attempt=attempt,
            checkpoint_sha256=checkpoint_sha256,
            returncode=completed.returncode,
            elapsed_seconds=elapsed,
            error=error,
        )

    status_payload.update(
        {
            "status": "succeeded",
            "returncode": 0,
            "elapsed_seconds": elapsed,
            "finished_at_unix": time.time(),
            "summary_path": str(job.output_dir / SUMMARY_FILENAME),
        }
    )
    _atomic_json(status_path, status_payload)
    print(f"[done] {job.model.name} / {job.condition.name} ({elapsed:.1f}s)")
    return _summary_row(
        job,
        summary,
        status="succeeded",
        attempt=attempt,
        checkpoint_sha256=checkpoint_sha256,
        elapsed_seconds=elapsed,
    )


def _atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def write_matrix_summary(
    *,
    output_root: Path,
    manifest: Path,
    musan_root: Path | None,
    seed: int,
    rows: Sequence[Mapping[str, Any]],
    input_identities: Mapping[str, Any] | None = None,
    audio_export: AudioExportSpec | None = None,
) -> dict[str, Any]:
    statuses = ("pending", "planned", "cached", "succeeded", "failed", "interrupted")
    counts = {status: sum(row.get("status") == status for row in rows) for status in statuses}
    counts["total"] = len(rows)
    payload = {
        "schema_version": 1,
        "manifest": str(manifest),
        "musan_root": str(musan_root) if musan_root is not None else None,
        "output_root": str(output_root),
        "seed": seed,
        "audio_export": asdict(audio_export) if audio_export is not None else None,
        "inputs": dict(input_identities) if input_identities is not None else None,
        "counts": counts,
        "runs": list(rows),
    }
    _atomic_json(output_root / MATRIX_JSON_FILENAME, payload)
    _atomic_csv(output_root / MATRIX_CSV_FILENAME, rows)
    return payload


def _validate_forwarded_override(value: str, *, option: str) -> str:
    key, separator, _ = value.partition("=")
    key = key.strip().lstrip("+~")
    if not separator or not key:
        raise argparse.ArgumentTypeError(f"{option} expects a Hydra KEY=VALUE override")
    if key in _RESERVED_OVERRIDE_KEYS or key.startswith(_RESERVED_OVERRIDE_PREFIXES):
        raise argparse.ArgumentTypeError(
            f"{option} cannot override runner-owned key {key!r}; "
            "create/select an eval_condition overlay for augmentation changes"
        )
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Batch-evaluate exported Stage II models across deterministic clean, "
            "source-volume variation, stationary, burst, MUSAN noise/music, "
            "and MUSAN speech conditions."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--manifest", type=Path, help="Labeled CSV/JSONL test manifest.")
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        metavar="NAME=CHECKPOINT[::EXPERIMENT]",
        help=(
            "Candidate exported/merged .pt model; repeat for every candidate. "
            "Append ::EXPERIMENT when this model needs a different architecture config."
        ),
    )
    parser.add_argument(
        "--models-file",
        type=Path,
        help="Text file with one NAME=CHECKPOINT[::EXPERIMENT] entry per line.",
    )
    parser.add_argument(
        "--experiment",
        default="icefall_zipformer_stage2",
        help="Default Hydra model experiment.",
    )
    parser.add_argument(
        "--model-experiment",
        action="append",
        default=[],
        metavar="NAME=EXPERIMENT",
        help="Override the Hydra experiment for one named model; repeatable.",
    )
    parser.add_argument(
        "--model-override",
        action="append",
        default=[],
        metavar="NAME=KEY=VALUE",
        help="Hydra override applied to one named model; repeatable.",
    )
    parser.add_argument(
        "--condition",
        action="append",
        choices=tuple(CONDITION_BY_NAME),
        help="Run only this condition; repeatable. Omit to run all conditions.",
    )
    parser.add_argument(
        "--list-conditions",
        action="store_true",
        help="Print the built-in condition catalog and exit.",
    )
    parser.add_argument(
        "--musan-root",
        type=Path,
        help="MUSAN root containing noise/, music/, and speech/ subsets.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/stage2_noise_matrix"),
        help="Root for model/condition outputs and matrix summaries.",
    )
    parser.add_argument("--device", default="cuda", help="Evaluator run.device value.")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=0,
        help="prep.batch_size; zero keeps evaluator auto-selection.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="prep.num_workers; zero keeps evaluator auto-selection.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2025,
        help="Shared augmentation seed for fair cross-model comparison.",
    )
    parser.add_argument(
        "--export-audio",
        type=_parse_audio_export,
        default=5,
        metavar="N|all|none",
        help=(
            "Export a deterministic random sample of N transformed WAVs per "
            "model/condition; use all for every row or none to disable."
        ),
    )
    parser.add_argument(
        "--export-audio-seed",
        type=int,
        help="Sampling seed for WAV export; defaults to --seed.",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        type=lambda value: _validate_forwarded_override(value, option="--override"),
        metavar="KEY=VALUE",
        help="Common Hydra override forwarded to every evaluator; repeatable.",
    )
    parser.add_argument(
        "--force", action="store_true", help="Rerun successful matching jobs."
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Validate and print commands only."
    )
    parser.add_argument(
        "--fail-fast", action="store_true", help="Stop after the first failed job."
    )
    return parser


def print_conditions() -> None:
    print(
        "condition\tfamily\tnoise/burst SNR(dB)\tmusic SNR(dB)\t"
        "speech relative(dB)\tMUSAN subsets"
    )
    for condition in CONDITIONS:
        subsets = ",".join(
            subset
            for subset, needed in (
                ("noise", condition.requires_noise),
                ("music", condition.requires_music),
                ("speech", condition.requires_speech),
            )
            if needed
        )
        snr = "" if condition.snr_db is None else f"{condition.snr_db:g}"
        music_snr = (
            "" if condition.music_snr_db is None else f"{condition.music_snr_db:g}"
        )
        speech = (
            ""
            if condition.speech_relative_db is None
            else f"{condition.speech_relative_db:g}"
        )
        print(
            f"{condition.name}\t{condition.family}\t{snr}\t{music_snr}\t"
            f"{speech}\t{subsets or '-'}"
        )


def run_batch(args: argparse.Namespace) -> dict[str, Any] | None:
    if args.list_conditions:
        print_conditions()
        return None
    if args.manifest is None:
        raise BatchConfigError("--manifest is required")
    if args.batch_size < 0:
        raise BatchConfigError("--batch-size must be >= 0")
    if args.num_workers < 0:
        raise BatchConfigError("--num-workers must be >= 0")
    if args.seed < 0:
        raise BatchConfigError("--seed must be >= 0")
    audio_export = resolve_audio_export(
        args.export_audio,
        seed=args.seed,
        export_seed=args.export_audio_seed,
    )
    if not str(args.device).strip():
        raise BatchConfigError("--device must not be empty")
    for override in args.override:
        try:
            _validate_forwarded_override(override, option="--override")
        except argparse.ArgumentTypeError as exc:
            raise BatchConfigError(str(exc)) from exc

    manifest = args.manifest.expanduser().resolve()
    musan_root = args.musan_root.expanduser().resolve() if args.musan_root else None
    output_root = args.output_root.expanduser().resolve()
    if output_root.exists() and not output_root.is_dir():
        raise BatchConfigError(f"--output-root is not a directory: {output_root}")
    models = resolve_models(args)
    conditions = resolve_conditions(args.condition)
    validate_inputs(
        manifest=manifest,
        models=models,
        conditions=conditions,
        musan_root=musan_root,
        common_overrides=args.override,
    )
    jobs, identities = build_jobs(
        models=models,
        conditions=conditions,
        manifest=manifest,
        musan_root=musan_root,
        output_root=output_root,
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        common_overrides=args.override,
        audio_export=audio_export,
    )

    if args.dry_run:
        print(f"Planned {len(jobs)} evaluation job(s):")
        for job in jobs:
            print(shlex.join(job.command))
        return {
            "counts": {"total": len(jobs), "planned": len(jobs)},
            "audio_export": asdict(audio_export),
            "runs": [],
        }

    rows = []
    for job in jobs:
        checkpoint_sha256 = str(identities["models"][job.model.name]["sha256"])
        summary = (
            None
            if args.force
            else cached_summary(job, checkpoint_sha256=checkpoint_sha256)
        )
        if summary is None:
            rows.append(_base_row(job, checkpoint_sha256=checkpoint_sha256))
            continue
        status = _read_json(job.output_dir / STATUS_FILENAME) or {}
        rows.append(
            _summary_row(
                job,
                summary,
                status="cached",
                attempt=_previous_attempt(job.output_dir),
                checkpoint_sha256=checkpoint_sha256,
                elapsed_seconds=status.get("elapsed_seconds"),
            )
        )
    payload = write_matrix_summary(
        output_root=output_root,
        manifest=manifest,
        musan_root=musan_root,
        seed=args.seed,
        rows=rows,
        input_identities=identities,
        audio_export=audio_export,
    )

    interrupted = False
    for index, job in enumerate(jobs):
        checkpoint_sha256 = str(identities["models"][job.model.name]["sha256"])
        if rows[index]["status"] == "cached":
            print(f"[cached] {job.model.name} / {job.condition.name}")
            continue
        try:
            rows[index] = run_job(
                job,
                checkpoint_sha256=checkpoint_sha256,
                force=args.force,
            )
        except KeyboardInterrupt:
            status = _read_json(job.output_dir / STATUS_FILENAME) or {}
            rows[index] = _failure_row(
                job,
                status="interrupted",
                attempt=int(status.get("attempt", 0)),
                checkpoint_sha256=checkpoint_sha256,
                returncode=None,
                elapsed_seconds=status.get("elapsed_seconds"),
                error="interrupted by user",
            )
            interrupted = True
        payload = write_matrix_summary(
            output_root=output_root,
            manifest=manifest,
            musan_root=musan_root,
            seed=args.seed,
            rows=rows,
            input_identities=identities,
            audio_export=audio_export,
        )
        if interrupted or (args.fail_fast and rows[index]["status"] == "failed"):
            break

    if interrupted:
        raise KeyboardInterrupt
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        payload = run_batch(args)
    except BatchConfigError as exc:
        parser.error(str(exc))
    except KeyboardInterrupt:
        print("Interrupted; completed results and status files were preserved.", file=sys.stderr)
        return 130
    if payload is None:
        return 0
    counts = payload["counts"]
    if args.dry_run:
        return 0
    print(
        "Matrix complete: "
        f"succeeded={counts['succeeded']}, cached={counts['cached']}, "
        f"failed={counts['failed']}, pending={counts['pending']}"
    )
    print(f"CSV : {args.output_root.expanduser().resolve() / MATRIX_CSV_FILENAME}")
    print(f"JSON: {args.output_root.expanduser().resolve() / MATRIX_JSON_FILENAME}")
    return 1 if counts["failed"] or counts["interrupted"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
