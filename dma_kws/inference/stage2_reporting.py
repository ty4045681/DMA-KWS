"""Shared JSON result and provenance helpers for Stage-II evaluation."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from dma_kws.config import (
    fbank_kwargs,
    get_eval_fbank_config,
    get_tokenizer_config,
)
from dma_kws.pathing import resolve_dict_path
from dma_kws.inference.score_provenance import PROVENANCE_SCHEMA_VERSION
from dma_kws.stage2.objective import checkpoint_sequence_objective
from dma_kws.stage2.readout import resolve_qbyt_score_spec

def _file_identity(path: str | Path, *, kind: str) -> dict[str, str | int]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise SystemExit(f"{kind} file not found: {resolved}")
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def build_score_provenance(
    config: dict,
    *,
    checkpoint_path: str | Path,
    calibration_path: str | Path | None = None,
    stream: object,
    left_padding_ms: int,
    right_padding_ms: int,
    keyword_eval: Mapping[str, Any] | None = None,
) -> dict:
    """Return the semantic fingerprint needed to compare Stage-II scores."""

    stage2 = config.get("stage2", {}) or {}
    tokenizer = get_tokenizer_config(config)
    tokenizer_identity = _file_identity(
        resolve_dict_path(config),
        kind="Tokenizer dictionary",
    )
    tokenizer_identity["split_with_space"] = str(
        tokenizer.get("split_with_space", " ")
    )
    try:
        import torch

        checkpoint = torch.load(
            Path(checkpoint_path).expanduser().resolve(),
            map_location="cpu",
        )
    except Exception as exc:
        raise SystemExit(
            f"Failed to read Stage II checkpoint objective metadata: {exc}"
        ) from exc
    try:
        sequence_objective = checkpoint_sequence_objective(checkpoint)
    except ValueError as exc:
        raise SystemExit(
            f"Stage II checkpoint is missing v6 objective metadata: {exc}"
        ) from exc
    score = resolve_qbyt_score_spec(stage2)
    if score.family == "pooling":
        qbyt_alignment = {"topology": "pooling", **score.value.as_dict()}
    else:
        qbyt_alignment = score.value.as_dict()

    return {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "checkpoint": _file_identity(checkpoint_path, kind="Stage II checkpoint"),
        "calibration": (
            _file_identity(calibration_path, kind="Stage II calibration")
            if calibration_path
            else {"type": "identity_logit_sigmoid"}
        ),
        "qbyt_readout_version": score.version,
        "qbyt_alignment": qbyt_alignment,
        "stream": stream,
        "audio_padding_ms": {
            "left": int(left_padding_ms),
            "right": int(right_padding_ms),
        },
        "fbank": fbank_kwargs(get_eval_fbank_config(config)),
        "tokenizer": tokenizer_identity,
        # Derive this from the checkpoint, not the runtime config.
        "sequence_objective": sequence_objective.as_dict(),
        "keyword_eval": (
            dict(keyword_eval) if keyword_eval is not None else {"mode": "per_row"}
        ),
    }


def _finite_float(value: object, *, field: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a finite number, got {value!r}") from exc
    if not math.isfinite(number):
        raise ValueError(f"{field} must be finite, got {number!r}")
    return number


def _optional_phoneme_sequence(value: object, *, field: str) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple)) and all(
        isinstance(phone, str) for phone in value
    ):
        return list(value)
    raise ValueError(
        f"{field} must be a sequence of strings or None, got {value!r}"
    )


def build_result_record(
    manifest_row: dict,
    runner_result: dict,
) -> dict:
    """Build a rich and strictly validated Stage-II ``results.jsonl`` row."""

    keyword_phonemes = _optional_phoneme_sequence(
        runner_result.get("keyword_phonemes"),
        field="keyword_phonemes",
    )
    record = {
        "audio_path": manifest_row["audio_path"],
        "keyword": manifest_row["keyword"],
        "keyword_phonemes": keyword_phonemes,
        "qbyt_score": _finite_float(
            runner_result.get("qbyt_score", 0.0),
            field="qbyt_score",
        ),
        "detected": bool(runner_result["detected"]),
        "threshold": _finite_float(runner_result["threshold"], field="threshold"),
        "skipped": bool(runner_result.get("skipped", False)),
    }
    if "qbyt_raw_logit" in runner_result:
        record["qbyt_raw_logit"] = _finite_float(
            runner_result["qbyt_raw_logit"],
            field="qbyt_raw_logit",
        )
    if "label" in manifest_row:
        record["label"] = int(manifest_row["label"])
    if "clip_span_sec" in runner_result:
        span = runner_result["clip_span_sec"]
        if not isinstance(span, Mapping):
            raise TypeError("clip_span_sec must be a mapping")
        start_sec = _finite_float(
            span.get("start_sec"),
            field="clip_span_sec.start_sec",
        )
        end_sec = _finite_float(span.get("end_sec"), field="clip_span_sec.end_sec")
        if start_sec < 0.0:
            raise ValueError("clip_span_sec.start_sec must be >= 0")
        if end_sec < start_sec:
            raise ValueError("clip_span_sec.end_sec must be >= start_sec")
        record["clip_span_sec"] = {
            "start_sec": start_sec,
            "end_sec": end_sec,
        }
    if "augmented_duration_sec" in runner_result:
        augmented_duration_sec = _finite_float(
            runner_result["augmented_duration_sec"],
            field="augmented_duration_sec",
        )
        if augmented_duration_sec <= 0.0:
            raise ValueError("augmented_duration_sec must be > 0")
        record["augmented_duration_sec"] = augmented_duration_sec

    manifest_meta = {
        key: value
        for key, value in manifest_row.items()
        if key not in {"audio_path", "keyword", "keyword_phonemes", "label"}
    }
    if manifest_meta:
        record["manifest_meta"] = manifest_meta
    return record


def _validate_nested_keyword_results(keyword_results: object, *, skipped: bool) -> list[dict]:
    if skipped:
        if keyword_results:
            raise ValueError("skipped any-mode rows must use empty keyword_results")
        return []
    if not isinstance(keyword_results, list):
        raise ValueError("keyword_results must be a list")
    validated: list[dict] = []
    for keyword in keyword_results:
        if not isinstance(keyword, Mapping):
            raise ValueError("keyword_results entries must be mappings")
        item = dict(keyword)
        item["qbyt_score"] = _finite_float(item.get("qbyt_score"), field="keyword qbyt_score")
        if item.get("qbyt_raw_logit") is not None:
            item["qbyt_raw_logit"] = _finite_float(
                item["qbyt_raw_logit"],
                field="keyword qbyt_raw_logit",
            )
        pronunciations = item.get("pronunciation_results", [])
        if not isinstance(pronunciations, list):
            raise ValueError("pronunciation_results must be a list")
        nested = []
        for pronunciation in pronunciations:
            if not isinstance(pronunciation, Mapping):
                raise ValueError("pronunciation_results entries must be mappings")
            pron = dict(pronunciation)
            pron["qbyt_score"] = _finite_float(
                pron.get("qbyt_score"),
                field="pronunciation qbyt_score",
            )
            if pron.get("qbyt_raw_logit") is not None:
                pron["qbyt_raw_logit"] = _finite_float(
                    pron["qbyt_raw_logit"],
                    field="pronunciation qbyt_raw_logit",
                )
            nested.append(pron)
        item["pronunciation_results"] = nested
        validated.append(item)
    return validated


def build_keyword_set_result_record(
    manifest_row: dict,
    runner_result: dict,
    *,
    eval_protocol: str | None = None,
) -> dict:
    """Build a nested any-mode result. Never forges a top-level ``keyword``."""

    from dma_kws.inference.keyword_set import (
        ANY_RESULT_SCHEMA_VERSION,
        CLIP_EVAL_PROTOCOL,
    )

    if "keyword" in runner_result:
        raise ValueError("any-mode results must not forge a top-level keyword field")
    skipped = bool(runner_result.get("skipped", False))
    skip_reason = runner_result.get("skip_reason")
    qbyt_score = _finite_float(runner_result.get("qbyt_score", 0.0), field="qbyt_score")
    threshold = _finite_float(runner_result["threshold"], field="threshold")
    detected = bool(runner_result["detected"])
    if skipped:
        if detected:
            raise ValueError("skipped any-mode rows cannot be detected")
        if skip_reason != "too_short":
            raise ValueError("skipped any-mode rows must use skip_reason='too_short'")
        qbyt_score = 0.0
    record: dict[str, Any] = {
        "result_schema_version": int(
            runner_result.get("result_schema_version", ANY_RESULT_SCHEMA_VERSION)
        ),
        "eval_protocol": str(
            eval_protocol
            or runner_result.get("eval_protocol")
            or CLIP_EVAL_PROTOCOL
        ),
        "keyword_eval_mode": "any",
        "keyword_set_id": str(runner_result["keyword_set_id"]),
        "audio_path": manifest_row["audio_path"],
        "qbyt_score": qbyt_score,
        "threshold": threshold,
        "detected": detected,
        "skipped": skipped,
        "skip_reason": skip_reason,
        "best_keyword": runner_result.get("best_keyword"),
        "best_pronunciation_id": runner_result.get("best_pronunciation_id"),
        "matched_keywords": list(runner_result.get("matched_keywords") or []),
        "keyword_results": _validate_nested_keyword_results(
            runner_result.get("keyword_results") or [],
            skipped=skipped,
        ),
    }
    if skipped or runner_result.get("qbyt_raw_logit") is None:
        record["qbyt_raw_logit"] = None
    else:
        record["qbyt_raw_logit"] = _finite_float(
            runner_result["qbyt_raw_logit"],
            field="qbyt_raw_logit",
        )
    if "label" in manifest_row:
        record["label"] = int(manifest_row["label"])
        record["label_scope"] = str(manifest_row.get("label_scope", "target_set"))
    if "keyword_labels" in manifest_row:
        record["keyword_labels"] = dict(manifest_row["keyword_labels"])
    if "clip_span_sec" in runner_result:
        span = runner_result["clip_span_sec"]
        if not isinstance(span, Mapping):
            raise TypeError("clip_span_sec must be a mapping")
        start_sec = _finite_float(span.get("start_sec"), field="clip_span_sec.start_sec")
        end_sec = _finite_float(span.get("end_sec"), field="clip_span_sec.end_sec")
        if start_sec < 0.0:
            raise ValueError("clip_span_sec.start_sec must be >= 0")
        if end_sec < start_sec:
            raise ValueError("clip_span_sec.end_sec must be >= start_sec")
        record["clip_span_sec"] = {"start_sec": start_sec, "end_sec": end_sec}
    if "augmented_duration_sec" in runner_result:
        augmented_duration_sec = _finite_float(
            runner_result["augmented_duration_sec"],
            field="augmented_duration_sec",
        )
        if augmented_duration_sec <= 0.0:
            raise ValueError("augmented_duration_sec must be > 0")
        record["augmented_duration_sec"] = augmented_duration_sec

    reserved = {
        "audio_path",
        "keyword",
        "keyword_phonemes",
        "label",
        "label_scope",
        "keyword_labels",
        "target_texts",
        "unused_keyword_labels",
        "_manifest_row_number",
    }
    manifest_meta = {
        key: value for key, value in manifest_row.items() if key not in reserved
    }
    if "unused_keyword_labels" in manifest_row:
        manifest_meta["unused_keyword_labels"] = manifest_row["unused_keyword_labels"]
    if manifest_meta:
        record["manifest_meta"] = manifest_meta
    return record


__all__ = [
    "build_keyword_set_result_record",
    "build_result_record",
    "build_score_provenance",
]
