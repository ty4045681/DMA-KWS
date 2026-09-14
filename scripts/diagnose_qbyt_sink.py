#!/usr/bin/env python3
"""Score clips with pooling QbyT sink-attention capture and write CSV/NPZ/JSON.

Reuses the Stage II clip Hydra stack (``resolved_config``,
``Stage2ClipRunner.from_config``, ``ClipFeatureDataset``, clip padding) and the
Task 3 ``attention_diagnostics`` API. HTML/PNG figures come from
``dma_kws.inference.qbyt_attention_report``.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import hydra
from omegaconf import DictConfig, OmegaConf

from dma_kws.config import require_sections
from dma_kws.hydra_app import CONFIG_DIR, resolved_config
from dma_kws.inference.keyword_set import (
    KeywordEvalConfigError,
    file_sha256,
    reject_unsupported_any_mode,
    require_qbyt_threshold,
)
from dma_kws.inference.manifest import load_manifest
from dma_kws.inference.qbyt_attention_diagnostics import (
    AttentionCaptureSpec,
    AttentionParityError,
    SampleAttentionDiagnostics,
    SinkAblationResult,
    SinkAblationSpec,
    assert_pooling_sink_attention_compatible,
    estimate_attention_workspace_bytes,
)
from dma_kws.inference.qbyt_attention_manifest import (
    PAIR_INCONSISTENT,
    PAIR_MISSING_CLEAN,
    PAIR_MULTIPLE_CLEAN,
    PAIR_OK,
    AttentionManifestRow,
    validate_attention_manifest_rows,
)
from dma_kws.inference.qbyt_attention_report import (
    FbankTimeSpec,
    SYNTHETIC_FIXTURE_BANNER,
    SampleTimeAxis,
    build_sample_time_axis,
    extra_region_metric_rows,
    inspect_encoder_time_map,
    noise_region_masks,
    pair_time_grids_comparable,
    render_sink_attention_report,
    serialize_encoder_time_map,
)
from dma_kws.inference.stage2_clip import (
    ClipFeatureDataset,
    Stage2ClipRunner,
    collate_clip_feature_batch,
    parse_phoneme_sequence,
    resolve_clip_audio_padding_ms,
)
from dma_kws.nn import encoder_output_frames
from dma_kws.pathing import resolve_dict_path
from dma_kws.training.device import resolve_accelerator


RUN_SCHEMA_VERSION = 1
_TOOL_MARKERS = ("run.json", "summary.json")
_PARTIAL_DIRNAME = ".partial"
_MANIFEST_GROUP_FIELDS = frozenset(
    {
        "audio_path",
        "keyword",
        "label",
        "keyword_phonemes",
        "sample_id",
        "condition",
        "pair_id",
        "keyword_spans",
        "noise_spans",
        "pronunciation_id",
    }
)

SINK_DIAGNOSTICS_DEFAULTS: dict[str, Any] = {
    "mode": "clips",
    "capture_layers": "all",
    "capture_heads": "all",
    "save_traces": True,
    "save_full_attention": False,
    "ablations": ["block_sink_all", "block_sink_each_layer"],
    "max_combined_tokens": 1024,
    "max_attention_bytes": 268435456,
    "parity_atol": 0.00001,
    "parity_rtol": 0.0001,
    "max_report_samples": 40,
    "plot_dpi": 160,
    "length_bins": [0, 100, 200, 400, 800],
    "group_field": "condition",
    "synthetic_fixture": False,
}

RECORDS_FIELDS = [
    "run_id",
    "sample_id",
    "manifest_record_number",
    "audio_path",
    "keyword",
    "keyword_phonemes",
    "query_id",
    "condition",
    "pair_id",
    "label",
    "ablation",
    "blocked_layers",
    "qbyt_raw_logit",
    "qbyt_score",
    "threshold",
    "detected",
    "delta_raw_logit",
    "delta_qbyt_score",
    "text_length",
    "audio_length",
    "status",
    "skip_reason",
    "trace_path",
    "report_selected",
]
METRICS_FIELDS = [
    "run_id",
    "sample_id",
    "ablation",
    "layer",
    "head",
    "region",
    "mean",
    "min",
    "max",
    "query_count",
    "key_count",
    "row_sum_error",
]
POSITION_FIELDS = [
    "run_id",
    "sample_id",
    "ablation",
    "position",
    "phoneme",
    "position_logit",
    "delta_position_logit",
]
PAIR_FIELDS = [
    "baseline_sample_id",
    "variant_sample_id",
    "match_key",
    "pair_status",
    "normal_score_delta",
    "time_grid_comparable",
    "pair_reason",
]


class DiagnoseRunFailed(RuntimeError):
    """Scoring failed; write ``summary.status=failed`` and exit non-zero."""

    def __init__(self, message: str, *, max_parity_error: float | None = None) -> None:
        super().__init__(message)
        self.max_parity_error = max_parity_error


@dataclass
class PreparedSample:
    row: AttentionManifestRow
    raw_row: dict[str, Any]
    phonemes: list[str]
    token_ids: list[int]
    query_id: str
    source_duration_sec: float
    source_sample_rate: int = 0
    feat: Any | None = None
    audio_len: int | None = None
    skip_reason: str | None = None


@dataclass
class SampleOutcome:
    prepared: PreparedSample
    status: str
    skip_reason: str | None
    normal_qbyt_score: float | None = None
    normal_detected: bool | None = None
    trace_paths: dict[str, str] = field(default_factory=dict)
    record_rows: list[dict[str, str]] = field(default_factory=list)
    metric_rows: list[dict[str, str]] = field(default_factory=list)
    position_rows: list[dict[str, str]] = field(default_factory=list)
    time_axis: dict[str, Any] = field(default_factory=dict)


def resolve_diagnostic_batch_size(prep: Mapping[str, Any]) -> int:
    raw = prep.get("batch_size", 0)
    if raw is None or raw == "":
        return 1
    if isinstance(raw, bool):
        raise SystemExit("prep.batch_size must be an integer, not bool")
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"prep.batch_size must be an integer, got {raw!r}") from exc
    return 1 if value <= 0 else value


def resolve_diagnostic_num_workers(prep: Mapping[str, Any]) -> int:
    raw = prep.get("num_workers", 0)
    if raw is None or raw == "":
        return 0
    if isinstance(raw, bool):
        raise SystemExit("prep.num_workers must be an integer, not bool")
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"prep.num_workers must be an integer, got {raw!r}") from exc
    return 0 if value <= 0 else value


def _plain(value: Any) -> Any:
    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=True)
    return value


def _as_dict(value: Any, *, field: str) -> dict[str, Any]:
    plain = _plain(value)
    if plain is None:
        return {}
    if not isinstance(plain, dict):
        raise SystemExit(f"{field} must be a mapping, got {type(plain).__name__}")
    return plain


def _as_list(value: Any) -> list[Any]:
    plain = _plain(value)
    if plain is None:
        return []
    if isinstance(plain, list):
        return plain
    if isinstance(plain, tuple):
        return list(plain)
    return [plain]


def _require_bool(value: Any, *, field: str) -> bool:
    if isinstance(value, bool):
        return value
    raise SystemExit(f"{field} must be a bool, got {value!r}")


def _require_int(value: Any, *, field: str, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SystemExit(f"{field} must be an integer, got {value!r}")
    if minimum is not None and value < minimum:
        raise SystemExit(f"{field} must be >= {minimum}, got {value!r}")
    return value


def _require_float(value: Any, *, field: str, minimum: float | None = None) -> float:
    if isinstance(value, bool):
        raise SystemExit(f"{field} must be a finite number, not bool")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"{field} must be a finite number, got {value!r}") from exc
    if not math.isfinite(number):
        raise SystemExit(f"{field} must be a finite number, got {value!r}")
    if minimum is not None and number < minimum:
        raise SystemExit(f"{field} must be >= {minimum}, got {value!r}")
    return number


def resolve_sink_diagnostics(prep: Mapping[str, Any]) -> dict[str, Any]:
    raw = _as_dict(prep.get("sink_diagnostics", {}), field="prep.sink_diagnostics")
    unknown = sorted(set(raw) - set(SINK_DIAGNOSTICS_DEFAULTS))
    if unknown:
        raise SystemExit(
            "unknown prep.sink_diagnostics keys: "
            + ", ".join(unknown)
            + ". Known keys: "
            + ", ".join(sorted(SINK_DIAGNOSTICS_DEFAULTS))
        )
    merged = {**SINK_DIAGNOSTICS_DEFAULTS, **raw}
    merged["save_traces"] = _require_bool(
        merged["save_traces"], field="prep.sink_diagnostics.save_traces"
    )
    merged["save_full_attention"] = _require_bool(
        merged["save_full_attention"],
        field="prep.sink_diagnostics.save_full_attention",
    )
    merged["max_combined_tokens"] = _require_int(
        merged["max_combined_tokens"],
        field="prep.sink_diagnostics.max_combined_tokens",
        minimum=1,
    )
    merged["max_attention_bytes"] = _require_int(
        merged["max_attention_bytes"],
        field="prep.sink_diagnostics.max_attention_bytes",
        minimum=1,
    )
    merged["parity_atol"] = _require_float(
        merged["parity_atol"], field="prep.sink_diagnostics.parity_atol", minimum=0.0
    )
    merged["parity_rtol"] = _require_float(
        merged["parity_rtol"], field="prep.sink_diagnostics.parity_rtol", minimum=0.0
    )
    merged["max_report_samples"] = _require_int(
        merged["max_report_samples"],
        field="prep.sink_diagnostics.max_report_samples",
        minimum=0,
    )
    merged["plot_dpi"] = _require_int(
        merged["plot_dpi"], field="prep.sink_diagnostics.plot_dpi", minimum=1
    )
    bins = _as_list(merged["length_bins"])
    parsed_bins = [
        _require_float(item, field="prep.sink_diagnostics.length_bins") for item in bins
    ]
    if len(parsed_bins) < 2 or any(
        parsed_bins[index] >= parsed_bins[index + 1]
        for index in range(len(parsed_bins) - 1)
    ):
        raise SystemExit(
            "prep.sink_diagnostics.length_bins must be a strictly increasing list"
        )
    merged["length_bins"] = parsed_bins
    merged["group_field"] = str(merged["group_field"] or "condition").strip() or "condition"
    merged["ablations"] = [str(item) for item in _as_list(merged["ablations"])]
    merged["mode"] = str(merged["mode"] or "").strip()
    merged["synthetic_fixture"] = _require_bool(
        merged["synthetic_fixture"],
        field="prep.sink_diagnostics.synthetic_fixture",
    )
    return merged


def _parse_index_selection(value: Any, *, bound: int, name: str) -> tuple[int, ...]:
    if isinstance(value, str) and value.strip().lower() == "all":
        return tuple(range(bound))
    if value is None:
        return tuple(range(bound))
    if isinstance(value, bool):
        raise SystemExit(
            f"prep.sink_diagnostics.{name} must be 'all' or a list of 0-based integers"
        )
    sequence = _as_list(value)
    selected: list[int] = []
    seen: set[int] = set()
    for raw in sequence:
        if isinstance(raw, bool) or not isinstance(raw, int):
            raise SystemExit(
                f"prep.sink_diagnostics.{name} must be 0-based integers, got {raw!r}"
            )
        if raw < 0 or raw >= bound:
            raise SystemExit(
                f"prep.sink_diagnostics.{name} index {raw} is out of range [0, {bound})"
            )
        if raw in seen:
            raise SystemExit(
                f"prep.sink_diagnostics.{name} contains duplicate index {raw}"
            )
        seen.add(raw)
        selected.append(raw)
    return tuple(selected)


def expand_ablations(
    names: Sequence[str], *, n_layers: int
) -> list[SinkAblationSpec]:
    specs: list[SinkAblationSpec] = []
    seen: set[str] = set()
    for raw in names:
        name = str(raw).strip()
        if not name:
            continue
        if name == "normal":
            raise SystemExit(
                "prep.sink_diagnostics.ablations must not include 'normal'; "
                "the unintervened pass always runs"
            )
        if name == "block_sink_all":
            spec = SinkAblationSpec(
                name="block_sink_all",
                blocked_layers=tuple(range(n_layers)),
            )
            if spec.name in seen:
                raise SystemExit("prep.sink_diagnostics.ablations contains duplicates")
            seen.add(spec.name)
            specs.append(spec)
            continue
        if name == "block_sink_each_layer":
            for layer_id in range(n_layers):
                spec = SinkAblationSpec(
                    name=f"block_sink_layer_{layer_id}",
                    blocked_layers=(layer_id,),
                )
                if spec.name in seen:
                    raise SystemExit(
                        "prep.sink_diagnostics.ablations contains duplicates"
                    )
                seen.add(spec.name)
                specs.append(spec)
            continue
        raise SystemExit(
            "Unknown prep.sink_diagnostics.ablations entry "
            f"{name!r}; expected block_sink_all or block_sink_each_layer"
        )
    return specs


def _online_augmentation_enabled(prep: Mapping[str, Any]) -> bool:
    audio_aug = _as_dict(prep.get("audio_aug", {}), field="prep.audio_aug")
    transforms = _as_dict(
        audio_aug.get("transforms", {}), field="prep.audio_aug.transforms"
    )
    for section in transforms.values():
        if isinstance(section, Mapping) and section.get("enabled") is True:
            return True
    musan = _as_dict(prep.get("musan_mix", {}), field="prep.musan_mix")
    for key in (
        "noise",
        "music",
        "speech",
        "stationary_noise",
        "burst_noise",
        "volume_variation",
    ):
        section = musan.get(key, {})
        if isinstance(section, Mapping) and section.get("enabled") is True:
            return True
    return False


def _source_audio_info(path: str) -> tuple[float, int]:
    audio_path = Path(path)
    if not audio_path.is_file():
        raise SystemExit(f"audio file not found: {audio_path}")
    try:
        import soundfile as sf

        info = sf.info(str(audio_path))
    except Exception as exc:
        raise SystemExit(f"Failed to decode audio {audio_path}: {exc}") from exc
    rate = int(getattr(info, "samplerate", 0) or 0)
    duration = float(getattr(info, "duration", 0.0))
    if not math.isfinite(duration) or duration < 0.0:
        frames = int(getattr(info, "frames", 0) or 0)
        if frames < 0 or rate <= 0:
            raise SystemExit(f"Failed to decode audio {audio_path}: invalid header")
        duration = float(frames) / float(rate)
    if rate <= 0:
        raise SystemExit(f"Failed to decode audio {audio_path}: invalid sample rate")
    return duration, rate


def _git_identity(repo: Path) -> dict[str, Any]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"],
                cwd=repo,
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        )
        return {"git_commit": commit or None, "git_dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"git_commit": None, "git_dirty": None}


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, default=str)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _query_id(token_ids: Sequence[int]) -> str:
    return json.dumps([int(token) for token in token_ids], separators=(",", ":"))


def _csv_optional(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def _csv_float(value: Any) -> str:
    if value is None:
        return ""
    number = float(value)
    if not math.isfinite(number):
        return ""
    return repr(number)


def _csv_bool(value: bool | None) -> str:
    if value is None:
        return ""
    return json.dumps(bool(value), allow_nan=False)


def _csv_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _atomic_write_json(path: Path, payload: Any) -> None:
    _atomic_write_text(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
    )


def _append_jsonl(path: Path, rows: Sequence[Mapping[str, str]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    tmp.replace(path)


def _refuse_existing_output(output_dir: Path) -> None:
    existing = [name for name in _TOOL_MARKERS if (output_dir / name).exists()]
    if existing:
        raise SystemExit(
            f"output directory {output_dir} already contains sink diagnostics "
            f"results ({', '.join(existing)}); choose another prep.output_dir"
        )


def _spec_mapping(spec) -> dict[str, Any]:
    if spec is None:
        return {}
    as_dict = getattr(spec, "as_dict", None)
    if callable(as_dict):
        payload = dict(as_dict())
        payload.setdefault("version", getattr(spec, "version", None))
        payload.setdefault("family", getattr(spec, "family", None))
        return payload
    return {"repr": repr(spec)}


def _live_position_knobs(qbyt) -> dict[str, Any]:
    return {
        "text_position": getattr(qbyt, "text_position", None),
        "audio_position": getattr(qbyt, "audio_position", None),
        "readout_mode": getattr(qbyt, "readout_mode", None),
        "has_sink_token": getattr(qbyt, "sink_token", None) is not None,
    }


def _assert_live_v41_knobs(qbyt, qbyt_score) -> None:
    assert_pooling_sink_attention_compatible(qbyt, qbyt_score)
    value = getattr(qbyt_score, "value", None)
    if value is None:
        return
    live_text = getattr(qbyt, "text_position", None)
    live_audio = getattr(qbyt, "audio_position", None)
    spec_text = getattr(value, "text_position", None)
    spec_audio = getattr(value, "audio_position", None)
    mismatches = []
    if live_text is not None and spec_text is not None and live_text != spec_text:
        mismatches.append(f"text_position live={live_text!r} spec={spec_text!r}")
    if live_audio is not None and spec_audio is not None and live_audio != spec_audio:
        mismatches.append(f"audio_position live={live_audio!r} spec={spec_audio!r}")
    temperature = getattr(value, "temperature", None)
    live_temp = getattr(qbyt, "readout_temperature", None)
    if (
        temperature is not None
        and live_temp is not None
        and math.isfinite(float(live_temp))
        and abs(float(live_temp) - float(temperature)) > 0.0
    ):
        mismatches.append(
            f"temperature live={live_temp!r} spec={temperature!r}"
        )
    if mismatches:
        raise SystemExit(
            "pooling v4.1 readout knobs disagree with the live QbyT: "
            + "; ".join(mismatches)
        )


def _estimated_audio_len(encoder, num_fbank_frames: int) -> int:
    try:
        return int(encoder_output_frames(encoder, int(num_fbank_frames)))
    except (TypeError, ValueError):
        return int(num_fbank_frames)


def _attention_work_bytes(
    batch: int, n_layers: int, n_heads: int, packed_l: int
) -> int:
    return estimate_attention_workspace_bytes(batch, n_layers, n_heads, packed_l)


def _hooked_layer_count(
    capture_layers: Sequence[int], ablation_specs: Sequence[SinkAblationSpec]
) -> int:
    capture_set = set(capture_layers)
    count = len(capture_set)
    for spec in ablation_specs:
        count = max(count, len(capture_set | set(spec.blocked_layers)))
    return count


def _record_fieldnames(group_field: str) -> list[str]:
    fields = list(RECORDS_FIELDS)
    if group_field not in fields:
        fields.append(group_field)
    return fields


def _group_resource_ok(
    items: Sequence[PreparedSample],
    *,
    capture_spec: AttentionCaptureSpec,
    n_capture_layers: int,
    n_capture_heads: int,
) -> bool:
    if not items:
        return True
    packed = max(len(item.token_ids) for item in items) + max(
        int(item.audio_len or 0) for item in items
    ) + 1
    if packed > capture_spec.max_combined_tokens:
        return False
    return (
        _attention_work_bytes(
            len(items), n_capture_layers, n_capture_heads, packed
        )
        <= capture_spec.max_attention_bytes
    )


def _single_resource_reason(
    item: PreparedSample, *, capture_spec: AttentionCaptureSpec, n_layers: int, n_heads: int
) -> str | None:
    audio_len = int(item.audio_len or 0)
    if audio_len <= 0:
        return "zero_encoder_frames"
    packed = len(item.token_ids) + audio_len + 1
    if packed > capture_spec.max_combined_tokens:
        return "resource_limit"
    if _attention_work_bytes(1, n_layers, n_heads, packed) > capture_spec.max_attention_bytes:
        return "resource_limit"
    return None


def _partition_scoreable(
    items: Sequence[PreparedSample],
    *,
    capture_spec: AttentionCaptureSpec,
    n_capture_layers: int,
    n_capture_heads: int,
) -> tuple[list[list[PreparedSample]], list[tuple[PreparedSample, str]]]:
    groups: list[list[PreparedSample]] = []
    skipped: list[tuple[PreparedSample, str]] = []
    current: list[PreparedSample] = []
    for item in items:
        reason = _single_resource_reason(
            item,
            capture_spec=capture_spec,
            n_layers=n_capture_layers,
            n_heads=n_capture_heads,
        )
        if reason is not None:
            skipped.append((item, reason))
            continue
        trial = current + [item]
        if current and not _group_resource_ok(
            trial,
            capture_spec=capture_spec,
            n_capture_layers=n_capture_layers,
            n_capture_heads=n_capture_heads,
        ):
            groups.append(current)
            current = [item]
        else:
            current = trial
    if current:
        groups.append(current)
    return groups, skipped


def _merge_parity_error(current: float, measured: float | None) -> float:
    if measured is None:
        return current
    value = float(measured)
    if not math.isfinite(value):
        return value
    if not math.isfinite(current):
        return current
    return max(current, value)


def _json_parity_error(value: float, *, emit: bool) -> float | None:
    if not emit or not math.isfinite(value):
        return None
    return value


def _require_finite(value: Any, *, what: str, sample_id: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise DiagnoseRunFailed(
            f"non-finite {what} for sample {sample_id}: {value!r}"
        ) from exc
    if not math.isfinite(number):
        raise DiagnoseRunFailed(f"non-finite {what} for sample {sample_id}: {value!r}")
    return number


def _npz_payload(
    result: SinkAblationResult,
    *,
    phonemes: Sequence[str],
    prepared_waveform=None,
    prepared_sample_rate: int | None = None,
) -> dict[str, Any]:
    trace = result.trace
    payload: dict[str, Any] = {
        "layer_ids": np_int_array(trace.layer_ids),
        "head_ids": np_int_array(trace.head_ids),
        "audio_to_sink": np_float_array(trace.audio_to_sink),
        "text_to_sink": np_float_array(trace.text_to_sink),
        "text_to_audio": np_float_array(trace.text_to_audio),
        "position_logits": np_float_array(trace.position_logits),
        "delta_position_logits": np_float_array(result.delta_position_logits),
        "raw_logit": np_float_array(result.raw_logit),
        "qbyt_score": np_float_array(result.qbyt_score),
        "text_length": np_int_array(trace.text_length),
        "audio_length": np_int_array(trace.audio_length),
        "sink_index": np_int_array(trace.sink_index),
        "row_sum_max_error": np_float_array(trace.row_sum_max_error),
        "padding_mass_max": np_float_array(trace.padding_mass_max),
        "phonemes": np_str_array(phonemes),
        "blocked_layers": np_int_array(result.spec.blocked_layers),
        "ablation": np_str_array([result.spec.name]),
    }
    if trace.full_attention is not None:
        payload["full_attention"] = np_float_array(trace.full_attention)
    if prepared_waveform is not None:
        payload["prepared_waveform"] = np_float_array(prepared_waveform)
    if prepared_sample_rate is not None:
        payload["prepared_sample_rate"] = np_int_array(prepared_sample_rate)
    return payload


def np_float_array(value) -> Any:
    import numpy as np
    import torch

    if isinstance(value, torch.Tensor):
        array = value.detach().cpu().numpy()
    else:
        array = np.asarray(value)
    if array.dtype == object:
        raise TypeError("NPZ payload must not contain object arrays")
    if array.dtype.kind == "f":
        return array.astype(np.float32, copy=False)
    return array


def np_int_array(value) -> Any:
    import numpy as np

    return np.asarray(value, dtype=np.int64)


def np_str_array(value) -> Any:
    import numpy as np

    return np.asarray(value, dtype=str)


def _save_npz(path: Path, payload: Mapping[str, Any]) -> None:
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as handle:
        np.savez(handle, **payload)
    tmp.replace(path)


def _metrics_rows(
    *,
    run_id: str,
    sample_id: str,
    result: SinkAblationResult,
    time_axis=None,
    noise_spans=None,
) -> list[dict[str, str]]:
    trace = result.trace
    rows: list[dict[str, str]] = []
    for layer_index, layer_id in enumerate(trace.layer_ids):
        for head_index, head_id in enumerate(trace.head_ids):
            for region, tensor in (
                ("audio", trace.audio_to_sink[layer_index, head_index]),
                ("text", trace.text_to_sink[layer_index, head_index]),
            ):
                if tensor.numel() == 0:
                    continue
                values = tensor.detach().cpu().reshape(-1)
                rows.append(
                    {
                        "run_id": run_id,
                        "sample_id": sample_id,
                        "ablation": result.spec.name,
                        "layer": str(int(layer_id)),
                        "head": str(int(head_id)),
                        "region": region,
                        "mean": _csv_float(float(values.mean())),
                        "min": _csv_float(float(values.min())),
                        "max": _csv_float(float(values.max())),
                        "query_count": str(int(values.numel())),
                        "key_count": "1",
                        "row_sum_error": _csv_float(trace.row_sum_max_error),
                    }
                )
    if time_axis is not None:
        rows.extend(
            extra_region_metric_rows(
                run_id=run_id,
                sample_id=sample_id,
                ablation=result.spec.name,
                layer_ids=trace.layer_ids,
                head_ids=trace.head_ids,
                audio_to_sink=trace.audio_to_sink,
                time_axis=time_axis,
                noise_spans=noise_spans,
                row_sum_error=float(trace.row_sum_max_error),
            )
        )
    return rows


def _position_rows(
    *,
    run_id: str,
    sample_id: str,
    result: SinkAblationResult,
    phonemes: Sequence[str],
) -> list[dict[str, str]]:
    logits = result.trace.position_logits.detach().cpu().reshape(-1)
    deltas = result.delta_position_logits.detach().cpu().reshape(-1)
    count = min(len(phonemes), int(logits.numel()), int(deltas.numel()))
    rows = []
    for position in range(count):
        rows.append(
            {
                "run_id": run_id,
                "sample_id": sample_id,
                "ablation": result.spec.name,
                "position": str(position),
                "phoneme": phonemes[position],
                "position_logit": _csv_float(float(logits[position])),
                "delta_position_logit": _csv_float(float(deltas[position])),
            }
        )
    return rows


def _skipped_record(
    *,
    run_id: str,
    prepared: PreparedSample,
    ablation: str,
    blocked_layers: Sequence[int],
    skip_reason: str,
) -> dict[str, str]:
    row = prepared.row
    return {
        "run_id": run_id,
        "sample_id": row.sample_id,
        "manifest_record_number": str(row.record_number),
        "audio_path": row.audio_path,
        "keyword": row.keyword,
        "keyword_phonemes": " ".join(prepared.phonemes),
        "query_id": prepared.query_id,
        "condition": row.condition,
        "pair_id": _csv_optional(row.pair_id),
        "label": "" if row.label is None else str(int(row.label)),
        "ablation": ablation,
        "blocked_layers": _csv_json(list(blocked_layers)),
        "qbyt_raw_logit": "",
        "qbyt_score": "",
        "threshold": "",
        "detected": "",
        "delta_raw_logit": "",
        "delta_qbyt_score": "",
        "text_length": "",
        "audio_length": "",
        "status": "skipped",
        "skip_reason": skip_reason,
        "trace_path": "",
        "report_selected": "false",
    }


def _scored_record(
    *,
    run_id: str,
    prepared: PreparedSample,
    result: SinkAblationResult,
    trace_path: str,
) -> dict[str, str]:
    row = prepared.row
    return {
        "run_id": run_id,
        "sample_id": row.sample_id,
        "manifest_record_number": str(row.record_number),
        "audio_path": row.audio_path,
        "keyword": row.keyword,
        "keyword_phonemes": " ".join(prepared.phonemes),
        "query_id": prepared.query_id,
        "condition": row.condition,
        "pair_id": _csv_optional(row.pair_id),
        "label": "" if row.label is None else str(int(row.label)),
        "ablation": result.spec.name,
        "blocked_layers": _csv_json(list(result.spec.blocked_layers)),
        "qbyt_raw_logit": _csv_float(result.raw_logit),
        "qbyt_score": _csv_float(result.qbyt_score),
        "threshold": _csv_float(result.threshold),
        "detected": _csv_bool(result.detected),
        "delta_raw_logit": _csv_float(result.delta_raw_logit),
        "delta_qbyt_score": _csv_float(result.delta_qbyt_score),
        "text_length": str(int(result.trace.text_length)),
        "audio_length": str(int(result.trace.audio_length)),
        "status": "ok",
        "skip_reason": "",
        "trace_path": trace_path,
        "report_selected": "false",
    }


def _materialize_scored_sample(
    *,
    run_id: str,
    prepared: PreparedSample,
    sample: SampleAttentionDiagnostics,
    trace_paths: Mapping[str, str],
    time_axis=None,
    num_fbank_frames: int | None = None,
) -> SampleOutcome:
    """Copy scalars/CSV rows out of capture results and drop tensor handles."""

    conditions = [sample.normal, *sample.ablations]
    record_rows: list[dict[str, str]] = []
    metric_rows: list[dict[str, str]] = []
    position_rows: list[dict[str, str]] = []
    axis_payload: dict[str, Any] = {}
    if time_axis is not None:
        axis_payload = time_axis.as_dict()
        if num_fbank_frames is not None:
            axis_payload["num_fbank_frames"] = int(num_fbank_frames)
        if time_axis.status == "ok":
            _noise, _outside, intervals = noise_region_masks(
                time_axis, prepared.row.noise_spans
            )
            axis_payload["noise_intervals"] = intervals
    for result in conditions:
        record_rows.append(
            _scored_record(
                run_id=run_id,
                prepared=prepared,
                result=result,
                trace_path=trace_paths.get(result.spec.name, ""),
            )
        )
        metric_rows.extend(
            _metrics_rows(
                run_id=run_id,
                sample_id=prepared.row.sample_id,
                result=result,
                time_axis=time_axis,
                noise_spans=prepared.row.noise_spans,
            )
        )
        position_rows.extend(
            _position_rows(
                run_id=run_id,
                sample_id=prepared.row.sample_id,
                result=result,
                phonemes=prepared.phonemes,
            )
        )
    outcome = SampleOutcome(
        prepared=prepared,
        status="ok",
        skip_reason=None,
        normal_qbyt_score=float(sample.normal_qbyt_score),
        normal_detected=bool(sample.normal.detected),
        trace_paths=dict(trace_paths),
        record_rows=record_rows,
        metric_rows=metric_rows,
        position_rows=position_rows,
        time_axis=axis_payload,
    )
    del conditions
    return outcome


def _persist_sample_partial(partial_dir: Path, outcome: SampleOutcome) -> None:
    _append_jsonl(partial_dir / "records.jsonl", outcome.record_rows)
    _append_jsonl(partial_dir / "attention_metrics.jsonl", outcome.metric_rows)
    _append_jsonl(partial_dir / "position_scores.jsonl", outcome.position_rows)


def _select_report_samples(
    outcomes: Sequence[SampleOutcome], *, max_report_samples: int
) -> set[str]:
    if max_report_samples <= 0:
        return set()
    selected: list[str] = []
    selected_set: set[str] = set()

    def add(sample_id: str) -> bool:
        if sample_id in selected_set:
            return True
        if len(selected) >= max_report_samples:
            return False
        selected.append(sample_id)
        selected_set.add(sample_id)
        return True

    for outcome in outcomes:
        if outcome.status != "ok" or outcome.normal_detected is None:
            continue
        label = outcome.prepared.row.label
        if label is None:
            continue
        if bool(outcome.normal_detected) != bool(label):
            add(outcome.prepared.row.sample_id)

    groups: dict[str, list[SampleOutcome]] = {}
    for outcome in outcomes:
        pair_id = outcome.prepared.row.pair_id
        if pair_id and outcome.prepared.row.pair_status == PAIR_OK:
            groups.setdefault(pair_id, []).append(outcome)
    for members in groups.values():
        missing = [
            member.prepared.row.sample_id
            for member in members
            if member.prepared.row.sample_id not in selected_set
        ]
        if not missing:
            continue
        if len(selected) + len(missing) > max_report_samples:
            continue
        for sample_id in missing:
            add(sample_id)

    seen_conditions: list[str] = []
    for outcome in outcomes:
        condition = outcome.prepared.row.condition
        if condition not in seen_conditions:
            seen_conditions.append(condition)
    for condition in seen_conditions:
        for outcome in outcomes:
            if outcome.prepared.row.condition != condition:
                continue
            if not add(outcome.prepared.row.sample_id):
                break
    return selected_set


def _pair_rows(
    outcomes: Sequence[SampleOutcome], *, fbank_spec: FbankTimeSpec | None
) -> list[dict[str, str]]:
    by_pair: dict[str, list[SampleOutcome]] = {}
    for outcome in outcomes:
        pair_id = outcome.prepared.row.pair_id
        if pair_id:
            by_pair.setdefault(pair_id, []).append(outcome)
    rows: list[dict[str, str]] = []
    for members in by_pair.values():
        status = members[0].prepared.row.pair_status
        reason = members[0].prepared.row.pair_reason
        normal_scores: dict[str, float | None] = {}
        for member in members:
            if member.status == "ok" and member.normal_qbyt_score is not None:
                normal_scores[member.prepared.row.sample_id] = float(
                    member.normal_qbyt_score
                )
            else:
                normal_scores[member.prepared.row.sample_id] = None
        if status != PAIR_OK:
            for member in members:
                rows.append(
                    {
                        "baseline_sample_id": "",
                        "variant_sample_id": member.prepared.row.sample_id,
                        "match_key": _csv_json(
                            {
                                "keyword": member.prepared.row.keyword,
                                "token_ids": list(member.prepared.token_ids),
                            }
                        ),
                        "pair_status": status,
                        "normal_score_delta": "",
                        "time_grid_comparable": "false",
                        "pair_reason": reason or "",
                    }
                )
            continue
        cleans = [
            member
            for member in members
            if member.prepared.row.condition == "clean"
        ]
        baseline = cleans[0]
        for member in members:
            if member is baseline:
                continue
            base_score = normal_scores.get(baseline.prepared.row.sample_id)
            var_score = normal_scores.get(member.prepared.row.sample_id)
            delta = (
                ""
                if base_score is None or var_score is None
                else _csv_float(var_score - base_score)
            )
            comparable, grid_reason = _pair_time_grid(baseline, member, fbank_spec)
            rows.append(
                {
                    "baseline_sample_id": baseline.prepared.row.sample_id,
                    "variant_sample_id": member.prepared.row.sample_id,
                    "match_key": _csv_json(
                        {
                            "keyword": member.prepared.row.keyword,
                            "token_ids": list(member.prepared.token_ids),
                        }
                    ),
                    "pair_status": PAIR_OK,
                    "normal_score_delta": delta,
                    "time_grid_comparable": _csv_bool(comparable),
                    "pair_reason": "" if comparable else grid_reason,
                }
            )
    return rows


def _pair_time_grid(
    baseline: SampleOutcome,
    variant: SampleOutcome,
    fbank_spec: FbankTimeSpec | None,
) -> tuple[bool, str]:
    if baseline.status != "ok" or variant.status != "ok":
        return False, "member_not_scored"
    base_len = _outcome_audio_length(baseline)
    var_len = _outcome_audio_length(variant)
    axis_a = SampleTimeAxis.from_dict(baseline.time_axis, audio_length=base_len)
    axis_b = SampleTimeAxis.from_dict(variant.time_axis, audio_length=var_len)
    return pair_time_grids_comparable(
        axis_a,
        axis_b,
        token_ids_a=baseline.prepared.token_ids,
        token_ids_b=variant.prepared.token_ids,
        source_duration_a=baseline.prepared.source_duration_sec,
        source_duration_b=variant.prepared.source_duration_sec,
        fbank_a=fbank_spec,
        fbank_b=fbank_spec,
    )


def _outcome_audio_length(outcome: SampleOutcome) -> int:
    for row in outcome.record_rows:
        if row.get("audio_length"):
            try:
                return int(row["audio_length"])
            except (TypeError, ValueError):
                continue
    return 0


def _sample_run_metadata(
    outcome: SampleOutcome, *, model_sample_rate: int, time_axis_method: str
) -> dict[str, Any]:
    row = outcome.prepared.row
    payload = dict(outcome.time_axis)
    payload.update(
        {
            "internal_sample_id": row.internal_sample_id,
            "source_duration_sec": float(outcome.prepared.source_duration_sec),
            "source_sample_rate": int(outcome.prepared.source_sample_rate or 0),
            "model_sample_rate": int(model_sample_rate),
            "audio_length": _outcome_audio_length(outcome) or None,
            "text_length": len(outcome.prepared.token_ids),
            "keyword_spans": (
                [list(span) for span in row.keyword_spans]
                if row.keyword_spans is not None
                else None
            ),
            "noise_spans": (
                [list(span) for span in row.noise_spans]
                if row.noise_spans is not None
                else None
            ),
            "time_axis_status": payload.get("status") or "unavailable",
            "time_axis_method": payload.get("method") or time_axis_method,
            "query_id": outcome.prepared.query_id,
            "token_ids": list(outcome.prepared.token_ids),
            "phonemes": list(outcome.prepared.phonemes),
            "status": outcome.status,
            "skip_reason": outcome.skip_reason,
        }
    )
    return payload


def _pair_stats(outcomes: Sequence[SampleOutcome]) -> dict[str, int]:
    statuses = {}
    seen: set[str] = set()
    for outcome in outcomes:
        pair_id = outcome.prepared.row.pair_id
        if not pair_id or pair_id in seen:
            continue
        seen.add(pair_id)
        status = outcome.prepared.row.pair_status
        statuses[status] = statuses.get(status, 0) + 1
    return {
        "n_pair_ids": len(seen),
        "n_ok": statuses.get(PAIR_OK, 0),
        "n_missing_clean": statuses.get(PAIR_MISSING_CLEAN, 0),
        "n_multiple_clean": statuses.get(PAIR_MULTIPLE_CLEAN, 0),
        "n_inconsistent": statuses.get(PAIR_INCONSISTENT, 0),
    }


def _group_value(row: AttentionManifestRow, group_field: str) -> str:
    """Return the grouping key for a validated row.

    First-class diagnostic columns live on the row, not in ``extra_fields``.
    Missing/empty values follow the condition default of ``\"unknown\"``.
    """

    if group_field == "label":
        return "unlabeled" if row.label is None else str(int(row.label))
    if group_field in _MANIFEST_GROUP_FIELDS:
        value = getattr(row, group_field)
    else:
        value = row.extra_fields.get(group_field)
    if value is None or value == "":
        return "unknown"
    if isinstance(value, (list, tuple, dict)):
        return _csv_json(value)
    return str(value)


def _group_metrics(outcomes: Sequence[SampleOutcome], *, group_field: str) -> dict[str, Any]:
    buckets: dict[str, list[float]] = {}
    labeled: dict[str, int] = {}
    for outcome in outcomes:
        key = _group_value(outcome.prepared.row, group_field)
        if outcome.status != "ok" or outcome.normal_qbyt_score is None:
            buckets.setdefault(key, [])
            continue
        buckets.setdefault(key, []).append(float(outcome.normal_qbyt_score))
        if outcome.prepared.row.label is not None:
            labeled[key] = labeled.get(key, 0) + 1
    metrics = {}
    for key, scores in buckets.items():
        entry: dict[str, Any] = {
            "n": len(scores),
            "n_labeled": labeled.get(key, 0),
            "mean_qbyt_score": (
                None if not scores else float(sum(scores) / len(scores))
            ),
        }
        metrics[key] = entry
    return metrics


def _write_report_html(
    path: Path,
    *,
    run_id: str,
    summary: Mapping[str, Any],
) -> None:
    del run_id, summary
    render_sink_attention_report(path.parent)


def _is_resource_error(exc: BaseException) -> bool:
    message = str(exc)
    return "max_combined_tokens" in message or "max_attention_bytes" in message


def _is_zero_frame_error(exc: BaseException) -> bool:
    return "T == 0" in str(exc)


def _diagnose_group(
    verifier,
    group: list[PreparedSample],
    *,
    capture_spec: AttentionCaptureSpec,
    ablations: Sequence[SinkAblationSpec],
    parity_atol: float,
    parity_rtol: float,
) -> list[SampleAttentionDiagnostics]:
    feats = [item.feat for item in group]
    ids = [item.token_ids for item in group]
    try:
        diagnostics = verifier.attention_diagnostics(
            feats,
            ids,
            capture_spec=capture_spec,
            ablations=ablations,
            parity_atol=parity_atol,
            parity_rtol=parity_rtol,
        )
    except AttentionParityError as exc:
        raise DiagnoseRunFailed(
            str(exc), max_parity_error=float(exc.max_error)
        ) from exc
    except ValueError as exc:
        if len(group) > 1 and (_is_resource_error(exc) or _is_zero_frame_error(exc)):
            mid = max(1, len(group) // 2)
            return _diagnose_group(
                verifier,
                group[:mid],
                capture_spec=capture_spec,
                ablations=ablations,
                parity_atol=parity_atol,
                parity_rtol=parity_rtol,
            ) + _diagnose_group(
                verifier,
                group[mid:],
                capture_spec=capture_spec,
                ablations=ablations,
                parity_atol=parity_atol,
                parity_rtol=parity_rtol,
            )
        if len(group) == 1 and _is_resource_error(exc):
            raise _SkipSample("resource_limit") from exc
        if len(group) == 1 and _is_zero_frame_error(exc):
            raise _SkipSample("zero_encoder_frames") from exc
        raise SystemExit(str(exc)) from exc
    if len(diagnostics) != len(group):
        raise DiagnoseRunFailed(
            "attention_diagnostics returned "
            f"{len(diagnostics)} results for {len(group)} clips"
        )
    return diagnostics


class _SkipSample(Exception):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def _enroll_row(
    runner: Stage2ClipRunner, raw_row: Mapping[str, Any], *, record_number: int
) -> tuple[list[str], list[int]]:
    keyword = str(raw_row["keyword"])
    if "keyword_phonemes" in raw_row:
        phonemes = parse_phoneme_sequence(
            raw_row["keyword_phonemes"],
            field_name=f"Manifest row {record_number} keyword_phonemes",
        )
    else:
        phonemes = runner.resolve_keyword_phonemes(keyword)
    token_ids = runner.enroll_phonemes(phonemes)
    return phonemes, token_ids


def _group_field_available(
    loaded_rows: Sequence[Mapping[str, Any]], field: str
) -> bool:
    if field == "condition":
        return True
    if not loaded_rows:
        return True
    return any(field in row for row in loaded_rows)


def _calibrator_fingerprint(calibrator) -> dict[str, Any]:
    payload = {
        "type": type(calibrator).__name__,
        "slope": float(getattr(calibrator, "slope", 1.0)),
        "bias": float(getattr(calibrator, "bias", 0.0)),
        "score_name": getattr(calibrator, "score_name", "qbyt_raw_logit"),
    }
    payload["sha256"] = _sha256_text(_canonical_json(payload))
    return payload


def _write_outputs(
    *,
    output_dir: Path,
    run_id: str,
    run_payload: dict[str, Any],
    summary: dict[str, Any],
    record_rows: list[dict[str, str]],
    metric_rows: list[dict[str, str]],
    position_rows: list[dict[str, str]],
    pair_rows: list[dict[str, str]],
    group_field: str = "condition",
) -> None:
    (output_dir / "traces").mkdir(parents=True, exist_ok=True)
    (output_dir / "figures").mkdir(parents=True, exist_ok=True)
    _write_csv(
        output_dir / "records.csv", _record_fieldnames(group_field), record_rows
    )
    _write_csv(output_dir / "attention_metrics.csv", METRICS_FIELDS, metric_rows)
    _write_csv(output_dir / "position_scores.csv", POSITION_FIELDS, position_rows)
    _write_csv(output_dir / "pairs.csv", PAIR_FIELDS, pair_rows)
    _atomic_write_json(output_dir / "run.json", run_payload)
    _atomic_write_json(output_dir / "summary.json", summary)
    _write_report_html(output_dir / "report.html", run_id=run_id, summary=summary)
    shutil.rmtree(output_dir / _PARTIAL_DIRNAME, ignore_errors=True)


def run_diagnose(cfg: DictConfig) -> dict:
    try:
        import torch
    except ImportError as exc:
        raise SystemExit(
            "Missing torch/torchaudio. Install CUDA PyTorch on the remote training machine first."
        ) from exc

    torch.multiprocessing.set_sharing_strategy("file_system")

    config = resolved_config(cfg)
    require_sections(config, ["paths", "stage1", "stage2", "demo", "tokenizer"])
    prep = _as_dict(OmegaConf.to_container(cfg.prep, resolve=True), field="prep")
    run_cfg = cfg.run
    sink = resolve_sink_diagnostics(prep)

    if sink["mode"] == "windows":
        raise SystemExit(
            "prep.sink_diagnostics.mode=windows is stage D not implemented"
        )
    if sink["mode"] != "clips":
        raise SystemExit(
            "prep.sink_diagnostics.mode must be 'clips', got "
            f"{sink['mode']!r}"
        )

    from dma_kws.inference.stage2_verifier import resolve_inference_amp

    try:
        amp = resolve_inference_amp(prep.get("amp", "off"))
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if amp is not None:
        raise SystemExit(
            "prep.amp fp16/bf16 is unsupported for sink attention diagnostics; "
            "use prep.amp=off (FP32 eager)"
        )

    try:
        reject_unsupported_any_mode(prep, entry="scripts/diagnose_qbyt_sink.py")
    except KeywordEvalConfigError as exc:
        raise SystemExit(str(exc)) from exc

    try:
        threshold = require_qbyt_threshold(
            (config.get("demo") or {}).get("qbyt_threshold", 0.5),
            field="demo.qbyt_threshold",
        )
    except KeywordEvalConfigError as exc:
        raise SystemExit(str(exc)) from exc

    if _online_augmentation_enabled(prep):
        raise SystemExit(
            "sink attention diagnostics do not apply online audio_aug or musan_mix; "
            "use already-exported fixed audio and disable prep.audio_aug / prep.musan_mix"
        )

    manifest_path = str(prep.get("manifest", "")).strip()
    if not manifest_path:
        raise SystemExit("prep.manifest is required")
    if Path(manifest_path).suffix.lower() != ".csv":
        raise SystemExit("prep.manifest must be a .csv file")
    stage2_ckpt = str(prep.get("stage2_ckpt", "")).strip()
    if not stage2_ckpt:
        raise SystemExit("prep.stage2_ckpt is required")

    output_dir = Path(str(prep.get("output_dir", "")).strip() or "outputs/diagnose_qbyt_sink")
    _refuse_existing_output(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    batch_size = resolve_diagnostic_batch_size(prep)
    num_workers = resolve_diagnostic_num_workers(prep)
    stage2_cfg = config.get("stage2") if isinstance(config.get("stage2"), dict) else {}
    left_padding_ms, right_padding_ms = resolve_clip_audio_padding_ms(prep, stage2_cfg)

    try:
        loaded_rows = load_manifest(manifest_path)
    except (OSError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    group_field = sink["group_field"]
    if not _group_field_available(loaded_rows, group_field):
        raise SystemExit(
            f"prep.sink_diagnostics.group_field={group_field!r} is not a manifest column"
        )

    durations: list[float] = []
    source_rates: list[int] = []
    for row in loaded_rows:
        duration, source_rate = _source_audio_info(str(row["audio_path"]))
        durations.append(duration)
        source_rates.append(source_rate)
    try:
        validated = validate_attention_manifest_rows(
            loaded_rows,
            source_durations_sec=durations,
            first_record_number=2,
        )
    except (TypeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    accelerator, _ = resolve_accelerator(str(run_cfg.device))
    device = torch.device("cpu" if accelerator == "cpu" else "cuda")
    try:
        runner = Stage2ClipRunner.from_config(config, prep, device)
    except (KeywordEvalConfigError, ValueError, RuntimeError) as exc:
        raise SystemExit(str(exc)) from exc

    qbyt = runner.verifier._model.qbyt
    qbyt_score = getattr(runner.verifier, "qbyt_score", None)
    encoder = runner.verifier._model.encoder
    time_map = inspect_encoder_time_map(encoder)
    fbank_kwargs = runner.verifier.fbank_kwargs
    model_sample_rate = int(
        runner.verifier.fbank_extractor.output_sample_rate(runner.sample_rate)
    )
    fbank_spec = FbankTimeSpec(
        frame_length_ms=float(fbank_kwargs["frame_length"]),
        frame_shift_ms=float(fbank_kwargs["frame_shift"]),
        snip_edges=bool(fbank_kwargs.get("snip_edges", True)),
        model_sample_rate=model_sample_rate,
        backend=str(fbank_kwargs.get("backend", "")),
    )
    time_axis_method = (
        time_map.method if time_map is not None else "unavailable"
    )
    time_axis_status = "ok" if time_map is not None else "unavailable"
    try:
        _assert_live_v41_knobs(qbyt, qbyt_score)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    n_layers = len(qbyt.phone_matchor.layers)
    n_heads = int(getattr(qbyt, "nhead", 4))
    capture_layers = _parse_index_selection(
        sink["capture_layers"], bound=n_layers, name="capture_layers"
    )
    capture_heads = _parse_index_selection(
        sink["capture_heads"], bound=n_heads, name="capture_heads"
    )
    capture_spec = AttentionCaptureSpec(
        layers=capture_layers,
        heads=capture_heads,
        save_full_attention=bool(sink["save_full_attention"]),
        max_combined_tokens=int(sink["max_combined_tokens"]),
        max_attention_bytes=int(sink["max_attention_bytes"]),
    )
    ablation_specs = expand_ablations(sink["ablations"], n_layers=n_layers)
    n_hooked_layers = _hooked_layer_count(capture_layers, ablation_specs)
    ablation_catalog = [
        SinkAblationSpec(name="normal", blocked_layers=()),
        *ablation_specs,
    ]

    created_at = datetime.now(timezone.utc).isoformat()
    manifest_sha = file_sha256(manifest_path)
    checkpoint_sha = file_sha256(stage2_ckpt)
    run_id = (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + "_"
        + manifest_sha[:8]
    )
    repo = Path(__file__).resolve().parents[1]
    git_info = _git_identity(repo)
    dict_path = resolve_dict_path(config)
    tokenizer_fp = {
        "path": str(Path(dict_path).expanduser().resolve()),
        "sha256": file_sha256(dict_path),
    }
    calibrator_fp = _calibrator_fingerprint(runner.verifier.calibrator)
    config_fingerprint = _sha256_text(_canonical_json({"config": config, "prep": prep}))

    prepared_by_index: list[PreparedSample] = []
    query_catalog: list[dict[str, Any]] = []
    seen_queries: set[str] = set()
    try:
        for validated_row, raw_row, duration, source_rate in zip(
            validated, loaded_rows, durations, source_rates, strict=True
        ):
            phonemes, token_ids = _enroll_row(
                runner, raw_row, record_number=validated_row.record_number
            )
            query_id = _query_id(token_ids)
            if query_id not in seen_queries:
                seen_queries.add(query_id)
                query_catalog.append(
                    {
                        "query_id": query_id,
                        "keyword": validated_row.keyword,
                        "phonemes": list(phonemes),
                        "token_ids": list(token_ids),
                    }
                )
            prepared_by_index.append(
                PreparedSample(
                    row=validated_row,
                    raw_row=dict(raw_row),
                    phonemes=list(phonemes),
                    token_ids=list(token_ids),
                    query_id=query_id,
                    source_duration_sec=float(duration),
                    source_sample_rate=int(source_rate),
                )
            )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    prepared_waveforms: dict[int, tuple[Any, int]] = {}
    partial_dir = output_dir / _PARTIAL_DIRNAME

    def _observer(index: int, waveform, sample_rate: int) -> None:
        prepared_waveforms[index] = (
            waveform.detach().cpu().contiguous().clone(),
            int(sample_rate),
        )

    outcomes: list[SampleOutcome | None] = [None] * len(prepared_by_index)
    max_parity_error = 0.0
    failed_message: str | None = None
    status = "complete"

    def _skip_outcome(prepared: PreparedSample, reason: str) -> SampleOutcome:
        outcome = SampleOutcome(
            prepared=prepared,
            status="skipped",
            skip_reason=reason,
            record_rows=[
                _skipped_record(
                    run_id=run_id,
                    prepared=prepared,
                    ablation=spec.name,
                    blocked_layers=spec.blocked_layers,
                    skip_reason=reason,
                )
                for spec in ablation_catalog
            ],
        )
        _persist_sample_partial(partial_dir, outcome)
        prepared.feat = None
        prepared_waveforms.pop(prepared.row.record_number - 2, None)
        return outcome

    if prepared_by_index:
        from torch.utils.data import DataLoader

        dataset = ClipFeatureDataset(
            audio_paths=[item.row.audio_path for item in prepared_by_index],
            sample_rate=runner.sample_rate,
            fbank_extractor=runner.verifier.fbank_extractor,
            fbank_kwargs=runner.verifier.fbank_kwargs,
            min_fbank_frames=runner.verifier.min_fbank_frames,
            left_padding_ms=left_padding_ms,
            right_padding_ms=right_padding_ms,
            waveform_observer=_observer,
            include_augmented_duration=True,
        )
        # Observer mutates a parent-process dict; worker processes would drop
        # spectrograms and NPZ waveforms.
        if num_workers > 0:
            num_workers = 0
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_clip_feature_batch,
        )
        try:
            for batch in loader:
                pending: list[PreparedSample] = []
                for index, feat, _end_sec, _aug_dur in batch:
                    prepared = prepared_by_index[index]
                    if feat is None:
                        outcomes[index] = _skip_outcome(prepared, "zero_encoder_frames")
                        continue
                    prepared.feat = feat
                    prepared.audio_len = _estimated_audio_len(encoder, int(feat.size(0)))
                    pending.append(prepared)
                groups, skipped = _partition_scoreable(
                    pending,
                    capture_spec=capture_spec,
                    n_capture_layers=n_hooked_layers,
                    n_capture_heads=n_heads,
                )
                for prepared, reason in skipped:
                    outcomes[prepared.row.record_number - 2] = _skip_outcome(
                        prepared, reason
                    )
                for group in groups:
                    try:
                        diagnostics = _diagnose_group(
                            runner.verifier,
                            group,
                            capture_spec=capture_spec,
                            ablations=ablation_specs,
                            parity_atol=float(sink["parity_atol"]),
                            parity_rtol=float(sink["parity_rtol"]),
                        )
                    except _SkipSample as exc:
                        for prepared in group:
                            outcomes[prepared.row.record_number - 2] = _skip_outcome(
                                prepared, exc.reason
                            )
                        continue
                    except DiagnoseRunFailed as exc:
                        failed_message = str(exc)
                        status = "failed"
                        max_parity_error = _merge_parity_error(
                            max_parity_error, exc.max_parity_error
                        )
                        break
                    for prepared, sample in zip(group, diagnostics):
                        max_parity_error = _merge_parity_error(
                            max_parity_error, sample.max_parity_error
                        )
                        _require_finite(
                            sample.normal_raw_logit,
                            what="raw logit",
                            sample_id=prepared.row.sample_id,
                        )
                        _require_finite(
                            sample.normal_qbyt_score,
                            what="qbyt_score",
                            sample_id=prepared.row.sample_id,
                        )
                        conditions = [sample.normal, *sample.ablations]
                        for result in conditions:
                            _require_finite(
                                result.raw_logit,
                                what="raw logit",
                                sample_id=prepared.row.sample_id,
                            )
                            _require_finite(
                                result.qbyt_score,
                                what="qbyt_score",
                                sample_id=prepared.row.sample_id,
                            )
                        sample_index = prepared.row.record_number - 2
                        trace_paths: dict[str, str] = {}
                        waveform_pack = prepared_waveforms.get(sample_index)
                        observed_rate = (
                            None if waveform_pack is None else int(waveform_pack[1])
                        )
                        if observed_rate:
                            model_sample_rate = observed_rate
                        num_fbank = (
                            int(prepared.feat.size(0)) if prepared.feat is not None else None
                        )
                        time_axis = build_sample_time_axis(
                            audio_length=int(sample.normal.trace.audio_length),
                            encoder_map=time_map,
                            fbank=fbank_spec,
                            left_padding_ms=left_padding_ms,
                            right_padding_ms=right_padding_ms,
                            source_duration_sec=prepared.source_duration_sec,
                            num_fbank_frames=num_fbank,
                        )
                        for result in conditions:
                            name = (
                                f"{prepared.row.internal_sample_id}__{result.spec.name}.npz"
                            )
                            payload = _npz_payload(
                                result,
                                phonemes=prepared.phonemes,
                                prepared_waveform=(
                                    None if waveform_pack is None else waveform_pack[0]
                                ),
                                prepared_sample_rate=(
                                    None if waveform_pack is None else waveform_pack[1]
                                ),
                            )
                            _save_npz(partial_dir / "traces" / name, payload)
                            if sink["save_traces"]:
                                rel = Path("traces") / name
                                _save_npz(output_dir / rel, payload)
                                trace_paths[result.spec.name] = str(rel).replace("\\", "/")
                        outcome = _materialize_scored_sample(
                            run_id=run_id,
                            prepared=prepared,
                            sample=sample,
                            trace_paths=trace_paths,
                            time_axis=time_axis,
                            num_fbank_frames=num_fbank,
                        )
                        _persist_sample_partial(partial_dir, outcome)
                        outcomes[sample_index] = outcome
                        prepared.feat = None
                        prepared_waveforms.pop(sample_index, None)
                        del conditions
                        del sample
                    del diagnostics
                    if status == "failed":
                        break
                if status == "failed":
                    break
        except DiagnoseRunFailed as exc:
            failed_message = str(exc)
            status = "failed"
            max_parity_error = _merge_parity_error(
                max_parity_error, exc.max_parity_error
            )
        except Exception as exc:
            if isinstance(exc, SystemExit):
                raise
            raise SystemExit(f"Failed to load or score audio: {exc}") from exc

    finalized: list[SampleOutcome] = []
    for index, prepared in enumerate(prepared_by_index):
        outcome = outcomes[index]
        if outcome is None:
            finalized.append(_skip_outcome(prepared, "failed"))
        else:
            finalized.append(outcome)

    selected = _select_report_samples(
        finalized, max_report_samples=int(sink["max_report_samples"])
    )
    record_rows: list[dict[str, str]] = []
    metric_rows: list[dict[str, str]] = []
    position_rows: list[dict[str, str]] = []
    for outcome in finalized:
        report_flag = "true" if outcome.prepared.row.sample_id in selected else "false"
        for row in outcome.record_rows:
            patched = dict(row)
            patched["report_selected"] = report_flag
            if group_field not in RECORDS_FIELDS:
                patched[group_field] = _group_value(
                    outcome.prepared.row, group_field
                )
            record_rows.append(patched)
        metric_rows.extend(outcome.metric_rows)
        position_rows.extend(outcome.position_rows)

    pair_rows = _pair_rows(finalized, fbank_spec=fbank_spec)
    num_input = len(finalized)
    num_success = sum(outcome.status == "ok" for outcome in finalized)
    num_skipped = sum(outcome.status == "skipped" for outcome in finalized)
    num_fail = sum(outcome.status not in {"ok", "skipped"} for outcome in finalized)
    if status == "failed":
        num_fail = max(num_fail, 1)
    num_unlabeled = sum(
        outcome.prepared.row.label is None for outcome in finalized
    )
    summary = {
        "status": status,
        "run_id": run_id,
        "num_input": num_input,
        "num_success": num_success,
        "num_skipped": num_skipped,
        "num_fail": num_fail,
        "num_unlabeled": num_unlabeled,
        "max_parity_error": _json_parity_error(
            max_parity_error, emit=bool(num_success or status == "failed")
        ),
        "pairs": _pair_stats(finalized),
        "group_metrics": _group_metrics(finalized, group_field=group_field),
        "batch_size": batch_size,
        "num_workers": num_workers,
        "output_index": {
            "records_csv": "records.csv",
            "attention_metrics_csv": "attention_metrics.csv",
            "position_scores_csv": "position_scores.csv",
            "pairs_csv": "pairs.csv",
            "summary_json": "summary.json",
            "run_json": "run.json",
            "report_html": "report.html",
            "traces_dir": "traces",
            "figures_dir": "figures",
        },
    }
    if failed_message:
        summary["error"] = failed_message

    expanded_ablations = [
        {"name": spec.name, "blocked_layers": list(spec.blocked_layers)}
        for spec in ablation_catalog
    ]
    run_payload = {
        "schema_version": RUN_SCHEMA_VERSION,
        "run_id": run_id,
        "created_at": created_at,
        **git_info,
        "manifest": str(Path(manifest_path).expanduser().resolve()),
        "manifest_sha256": manifest_sha,
        "checkpoint": str(Path(stage2_ckpt).expanduser().resolve()),
        "checkpoint_sha256": checkpoint_sha,
        "tokenizer": tokenizer_fp,
        "calibrator": calibrator_fp,
        "queries": query_catalog,
        "config": config,
        "prep": prep,
        "sink_diagnostics": {
            **sink,
            "capture_layers": list(capture_layers),
            "capture_heads": list(capture_heads),
            "ablations": [spec.name for spec in ablation_specs],
        },
        "qbyt_readout": _spec_mapping(qbyt_score),
        "qbyt_live": _live_position_knobs(qbyt),
        "stream": runner.stream_policy.describe(),
        "audio_padding_ms": {"left": left_padding_ms, "right": right_padding_ms},
        "device": str(device),
        "dtype": "float32",
        "torch_version": torch.__version__,
        "expanded_capture_layers": list(capture_layers),
        "expanded_capture_heads": list(capture_heads),
        "expanded_ablations": expanded_ablations,
        "time_axis_method": time_axis_method,
        "time_axis_status": time_axis_status,
        "time_axis": {
            "status": time_axis_status,
            "method": time_axis_method,
            "fbank": fbank_spec.as_dict(),
            "model_sample_rate": model_sample_rate,
            "padding_ms": {"left": left_padding_ms, "right": right_padding_ms},
            "stream": runner.stream_policy.describe(),
            "encoder": serialize_encoder_time_map(time_map),
        },
        "samples": {
            outcome.prepared.row.sample_id: _sample_run_metadata(
                outcome,
                model_sample_rate=model_sample_rate,
                time_axis_method=time_axis_method,
            )
            for outcome in finalized
        },
        "parity_atol": float(sink["parity_atol"]),
        "parity_rtol": float(sink["parity_rtol"]),
        "seed": (config.get("training") or {}).get("seed"),
        "limits": {
            "max_combined_tokens": int(sink["max_combined_tokens"]),
            "max_attention_bytes": int(sink["max_attention_bytes"]),
            "max_report_samples": int(sink["max_report_samples"]),
        },
        "input_fingerprint": manifest_sha,
        "config_fingerprint": config_fingerprint,
        "threshold": threshold,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "report_selected": sorted(selected),
    }
    synthetic = bool(sink["synthetic_fixture"])
    run_payload["synthetic_fixture"] = synthetic
    summary["synthetic_fixture"] = synthetic
    if synthetic:
        run_payload["banner"] = SYNTHETIC_FIXTURE_BANNER
        summary["banner"] = SYNTHETIC_FIXTURE_BANNER

    _write_outputs(
        output_dir=output_dir,
        run_id=run_id,
        run_payload=run_payload,
        summary=summary,
        record_rows=record_rows,
        metric_rows=metric_rows,
        position_rows=position_rows,
        pair_rows=pair_rows,
        group_field=group_field,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))
    if status != "complete":
        raise SystemExit(failed_message or "sink attention diagnostics failed")
    return summary


@hydra.main(version_base=None, config_path=str(CONFIG_DIR), config_name="config")
def main(cfg: DictConfig) -> None:
    run_diagnose(cfg)


if __name__ == "__main__":
    main()
