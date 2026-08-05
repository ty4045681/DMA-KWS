"""Shared JSON result and provenance helpers for Stage-II evaluation."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from pathlib import Path

from dma_kws.config import (
    fbank_kwargs,
    get_eval_fbank_config,
    get_tokenizer_config,
)
from dma_kws.pathing import resolve_dict_path
from dma_kws.stage2.objective import checkpoint_sequence_objective
from dma_kws.stage2.readout import resolve_qbyt_readout
from dma_kws.tokenizer import (
    SEQ_LABEL_ORDERED_CONTIGUOUS_PREFIX,
    build_seq_label,
    normalize_seq_label_mode,
)


PROVENANCE_SCHEMA_VERSION = 1


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
    stream: object,
    left_padding_ms: int,
    right_padding_ms: int,
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
    sequence_objective = checkpoint_sequence_objective(checkpoint)
    qbyt_readout = resolve_qbyt_readout(stage2)

    return {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "checkpoint": _file_identity(checkpoint_path, kind="Stage II checkpoint"),
        # Keep the scalar mode for older consumers and record the complete
        # score semantics so two soft-min temperatures cannot be compared as
        # though they produced interchangeable logits.
        "qbyt_readout_mode": qbyt_readout.mode,
        "qbyt_readout": qbyt_readout.as_dict(),
        "stream": stream,
        "audio_padding_ms": {
            "left": int(left_padding_ms),
            "right": int(right_padding_ms),
        },
        "fbank": fbank_kwargs(get_eval_fbank_config(config)),
        "tokenizer": tokenizer_identity,
        # Derive this from the checkpoint, not the runtime config. A checkpoint
        # without embedded objective metadata is intentionally classified as
        # the released membership/zero-completion legacy objective.
        "sequence_objective": sequence_objective.as_dict(),
    }


def _finite_float(value: object, *, field: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a finite number, got {value!r}") from exc
    if not math.isfinite(number):
        raise ValueError(f"{field} must be finite, got {number!r}")
    return number


def _eps_readout_logit(
    position_logits: list[float],
    *,
    mode: str,
    temperature: float,
) -> float:
    """Reproduce QbyT's length-normalized EPS aggregation in Python."""

    if not position_logits:
        return 0.0
    if mode == "eps_mean":
        return math.fsum(position_logits) / len(position_logits)
    if mode != "eps_softmin":
        raise ValueError(
            f"EPS position logits are incompatible with readout mode {mode!r}"
        )
    minimum = min(position_logits)
    mean_exp = math.fsum(
        math.exp(-(value - minimum) / temperature)
        for value in position_logits
    ) / len(position_logits)
    return minimum - temperature * math.log(mean_exp)


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


def _optional_float_sequence(value: object, *, field: str) -> list[float] | None:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)):
        raise ValueError(
            f"{field} must be a sequence of finite numbers or None, got {value!r}"
        )
    return [
        _finite_float(item, field=f"{field}[{index}]")
        for index, item in enumerate(value)
    ]


def _sigmoid(logit: float) -> float:
    if logit >= 0.0:
        return 1.0 / (1.0 + math.exp(-logit))
    exp_logit = math.exp(logit)
    return exp_logit / (1.0 + exp_logit)


def _bce_with_logits(logit: float, target: int) -> float:
    """Numerically stable scalar BCEWithLogits used for JSON diagnostics."""

    return max(logit, 0.0) - float(target) * logit + math.log1p(
        math.exp(-abs(logit))
    )


def _sequence_diagnostic_fields(
    manifest_row: Mapping[str, object],
    *,
    keyword_phonemes: list[str] | None,
    text_variant_phonemes: list[str] | None,
    seq_position_logits: list[float] | None,
    completion_logit: float | None,
    completion_score: float | None,
    sequence_objective: Mapping[str, object] | None,
) -> dict:
    """Derive sequence scores, pseudo-targets and raw per-sample losses."""

    target_mode = None
    if sequence_objective is not None:
        target_mode = normalize_seq_label_mode(
            str(sequence_objective.get("target_mode", ""))
        )

    fields = {
        "seq_target_mode": target_mode,
        "seq_target_source": None,
        "seq_position_logits": seq_position_logits,
        "seq_position_scores": None,
        "expected_prefix_length": None,
        "target_prefix_length": None,
        "seq_position_targets": None,
        "seq_progress_loss_sample": None,
        "seq_completion_loss_sample": None,
    }

    if seq_position_logits is not None:
        if keyword_phonemes is None:
            raise ValueError(
                "keyword_phonemes is required when seq_position_logits is present"
            )
        if len(seq_position_logits) != len(keyword_phonemes):
            raise ValueError(
                "seq_position_logits must contain one value per keyword phoneme: "
                f"logits={len(seq_position_logits)}, "
                f"phonemes={len(keyword_phonemes)}"
            )
        scores = [_sigmoid(logit) for logit in seq_position_logits]
        fields["seq_position_scores"] = scores
        if target_mode == SEQ_LABEL_ORDERED_CONTIGUOUS_PREFIX:
            # This is a soft count, not a structurally constrained prefix
            # posterior: the current sequence head predicts positions
            # independently and can be non-monotonic.
            fields["expected_prefix_length"] = math.fsum(scores)

        if seq_position_logits:
            if completion_logit is None:
                raise ValueError(
                    "completion_logit is required when seq_position_logits is non-empty"
                )
            if not math.isclose(
                seq_position_logits[-1],
                completion_logit,
                rel_tol=1e-5,
                abs_tol=1e-6,
            ):
                raise ValueError(
                    "seq_position_logits[-1] must equal completion_logit: "
                    f"final={seq_position_logits[-1]}, completion={completion_logit}"
                )
            if completion_score is not None and not math.isclose(
                scores[-1],
                completion_score,
                rel_tol=1e-5,
                abs_tol=1e-6,
            ):
                raise ValueError(
                    "sigmoid(seq_position_logits[-1]) must equal completion_score: "
                    f"final={scores[-1]}, completion={completion_score}"
                )
        elif completion_logit is not None or completion_score is not None:
            raise ValueError(
                "Empty seq_position_logits require null completion logit and score"
            )

    if text_variant_phonemes is None or keyword_phonemes is None or target_mode is None:
        return fields

    text_variant_override = manifest_row.get("text_variant_phonemes")
    has_text_variant_override = text_variant_override is not None and not (
        isinstance(text_variant_override, str) and not text_variant_override.strip()
    )
    if has_text_variant_override:
        fields["seq_target_source"] = "manifest_text_variant_phonemes"
    elif str(manifest_row.get("text_variant", "")).strip():
        fields["seq_target_source"] = "text_variant_g2p"
    else:
        fields["seq_target_source"] = "runner_text_variant_phonemes"

    # build_seq_label compares symbols for equality; using the exported
    # ARPAbet strings here is therefore equivalent to token ids as long as the
    # tokenizer has not collapsed unsupported phones to <unk>. Enrollment
    # overrides are validated before scoring, and automatic G2P emits the same
    # inventory used by training.
    targets = build_seq_label(
        keyword_phonemes,  # type: ignore[arg-type]
        text_variant_phonemes,  # type: ignore[arg-type]
        mode=target_mode,
    )
    fields["seq_position_targets"] = targets
    if target_mode == SEQ_LABEL_ORDERED_CONTIGUOUS_PREFIX:
        fields["target_prefix_length"] = int(sum(targets))

    if seq_position_logits is None or not seq_position_logits:
        return fields
    if len(targets) != len(seq_position_logits):
        raise ValueError(
            "seq_position_targets must match seq_position_logits: "
            f"targets={len(targets)}, logits={len(seq_position_logits)}"
        )

    per_position_losses = [
        _bce_with_logits(logit, target)
        for logit, target in zip(seq_position_logits, targets)
    ]
    fields["seq_progress_loss_sample"] = math.fsum(per_position_losses) / len(
        per_position_losses
    )
    if target_mode == SEQ_LABEL_ORDERED_CONTIGUOUS_PREFIX:
        fields["seq_completion_loss_sample"] = per_position_losses[-1]
    return fields


def build_result_record(
    manifest_row: dict,
    runner_result: dict,
    *,
    sequence_objective: Mapping[str, object] | None = None,
    qbyt_readout: Mapping[str, object] | None = None,
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
    if qbyt_readout is None:
        # Backward-compatible default for direct helper callers and historical
        # EPS JSONL fixtures. Real evaluation always supplies provenance.
        readout_mode = "eps_mean"
        readout_temperature = 1.0
    else:
        resolved_readout = resolve_qbyt_readout(
            {"qbyt_readout": dict(qbyt_readout)}
        )
        readout_mode = resolved_readout.mode
        readout_temperature = resolved_readout.temperature
        record["qbyt_readout_mode"] = readout_mode
        record["qbyt_readout_temperature"] = readout_temperature

    text_variant_phonemes = None
    if "text_variant_phonemes" in runner_result:
        text_variant_phonemes = _optional_phoneme_sequence(
            runner_result["text_variant_phonemes"],
            field="text_variant_phonemes",
        )
        record["text_variant_phonemes"] = text_variant_phonemes
    if "label" in manifest_row:
        record["label"] = int(manifest_row["label"])
    for name in ("qbyt_logit", "completion_logit", "completion_score"):
        if name in runner_result:
            value = runner_result[name]
            record[name] = None if value is None else _finite_float(value, field=name)

    position_logits_raw = runner_result.get("eps_position_logits")
    if position_logits_raw is None:
        record["eps_position_logits"] = None
    elif isinstance(position_logits_raw, (list, tuple)):
        position_logits = [
            _finite_float(value, field=f"eps_position_logits[{index}]")
            for index, value in enumerate(position_logits_raw)
        ]
        if keyword_phonemes is None:
            raise ValueError(
                "keyword_phonemes is required when eps_position_logits is present"
            )
        if len(position_logits) != len(keyword_phonemes):
            raise ValueError(
                "eps_position_logits must contain one value per keyword phoneme: "
                f"logits={len(position_logits)}, phonemes={len(keyword_phonemes)}"
            )
        qbyt_logit = record.get("qbyt_logit")
        if qbyt_logit is None:
            raise ValueError(
                "qbyt_logit is required when eps_position_logits is present"
            )
        expected_logit = _eps_readout_logit(
            position_logits,
            mode=readout_mode,
            temperature=readout_temperature,
        )
        if not math.isclose(
            expected_logit,
            qbyt_logit,
            rel_tol=1e-5,
            abs_tol=1e-6,
        ):
            raise ValueError(
                "EPS position-logit aggregation must equal qbyt_logit: "
                f"mode={readout_mode}, expected={expected_logit}, "
                f"qbyt_logit={qbyt_logit}"
            )
        record["eps_position_logits"] = position_logits
    else:
        raise ValueError(
            "eps_position_logits must be a sequence of finite numbers or None, "
            f"got {position_logits_raw!r}"
        )

    seq_position_logits = _optional_float_sequence(
        runner_result.get("seq_position_logits"),
        field="seq_position_logits",
    )
    record.update(
        _sequence_diagnostic_fields(
            manifest_row,
            keyword_phonemes=keyword_phonemes,
            text_variant_phonemes=text_variant_phonemes,
            seq_position_logits=seq_position_logits,
            completion_logit=record.get("completion_logit"),
            completion_score=record.get("completion_score"),
            sequence_objective=sequence_objective,
        )
    )

    manifest_meta = {
        key: value
        for key, value in manifest_row.items()
        if key not in {"audio_path", "keyword", "keyword_phonemes", "label"}
    }
    if manifest_meta:
        record["manifest_meta"] = manifest_meta
    return record


__all__ = ["build_result_record", "build_score_provenance"]
