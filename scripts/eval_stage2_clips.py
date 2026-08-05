#!/usr/bin/env python3
"""Evaluate Stage II-only DMA-KWS inference on a manifest of keyword clips.

Manifest rows may provide ``keyword_phonemes`` as a space-separated ARPAbet
string (or a string array in JSONL) to override keyword G2P for that row. Rows
without the field retain automatic G2P. ``text_variant_phonemes`` supports the
same override for the query reference; otherwise ``text_variant`` is converted
automatically. When a query reference is available, the JSONL also records
position-level sequence targets and raw per-sample diagnostic losses using the
sequence objective saved in the checkpoint.

Each clip receives 160 ms of zero-valued waveform context on both sides by
default. Override with ``+prep.left_padding_ms=...`` and
``+prep.right_padding_ms=...``; use zero to disable either side.
Padding is part of the scored model input and counts toward its minimum length.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Mapping
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from dma_kws.config import (
    fbank_kwargs,
    get_eval_fbank_config,
    get_tokenizer_config,
    require_sections,
)
from dma_kws.hydra_app import CONFIG_DIR, resolved_config
from dma_kws.inference.manifest import load_manifest
from dma_kws.inference.metrics import summarize_labeled_results
from dma_kws.inference.stage2_clip import Stage2ClipRunner
from dma_kws.pathing import resolve_dict_path
from dma_kws.stage2.objective import checkpoint_sequence_objective
from dma_kws.stage2.readout import resolve_qbyt_readout_mode
from dma_kws.tokenizer import (
    SEQ_LABEL_ORDERED_CONTIGUOUS_PREFIX,
    build_seq_label,
    normalize_seq_label_mode,
)
from dma_kws.training.score_diagnostics import binary_score_diagnostics
from dma_kws.training.device import resolve_accelerator


DEFAULT_PADDING_MS = 160
_PROVENANCE_SCHEMA_VERSION = 1


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


def _score_provenance(
    config: dict,
    *,
    checkpoint_path: str | Path,
    stream: object,
    left_padding_ms: int,
    right_padding_ms: int,
) -> dict:
    """Return the small, semantic fingerprint needed to compare score files."""

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

    return {
        "schema_version": _PROVENANCE_SCHEMA_VERSION,
        "checkpoint": _file_identity(checkpoint_path, kind="Stage II checkpoint"),
        "qbyt_readout_mode": resolve_qbyt_readout_mode(stage2),
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


def _result_record(
    manifest_row: dict,
    runner_result: dict,
    *,
    sequence_objective: Mapping[str, object] | None = None,
) -> dict:
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
            record[name] = (
                None if value is None else _finite_float(value, field=name)
            )

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
        expected_logit = (
            math.fsum(position_logits) / len(position_logits)
            if position_logits
            else 0.0
        )
        if not math.isclose(
            expected_logit,
            qbyt_logit,
            rel_tol=1e-5,
            abs_tol=1e-6,
        ):
            raise ValueError(
                "mean(eps_position_logits) must equal qbyt_logit: "
                f"mean={expected_logit}, qbyt_logit={qbyt_logit}"
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


def _metrics_record(record: dict) -> dict:
    metrics_row = dict(record)
    metrics_row["best_qbyt_score"] = float(record.get("qbyt_score", 0.0))
    return metrics_row


def _score_head_diagnostics(
    records: list[dict],
    *,
    score_field: str,
    threshold: float,
    ece_num_bins: int,
) -> dict:
    import torch

    usable = [
        record
        for record in records
        if "label" in record
        and not bool(record.get("skipped", False))
        and record.get(score_field) is not None
    ]
    if not usable:
        return {}
    scores = torch.tensor(
        [float(record[score_field]) for record in usable],
        dtype=torch.float64,
    )
    labels = torch.tensor(
        [int(record["label"]) for record in usable],
        dtype=torch.long,
    )
    diagnostics = binary_score_diagnostics(
        scores,
        labels,
        deployment_threshold=threshold,
        ece_num_bins=ece_num_bins,
    )
    result = {}
    for name, value in diagnostics.items():
        item = value.item() if value.numel() == 1 else value.detach().cpu().tolist()
        if isinstance(item, float) and not math.isfinite(item):
            item = None
        result[name] = item
    return result


def _resolve_audio_padding_ms(prep: dict) -> tuple[int, int]:
    left_padding_ms = int(prep.get("left_padding_ms", DEFAULT_PADDING_MS))
    right_padding_ms = int(prep.get("right_padding_ms", DEFAULT_PADDING_MS))
    if left_padding_ms < 0 or right_padding_ms < 0:
        raise SystemExit("prep.left_padding_ms and prep.right_padding_ms must be >= 0")
    return left_padding_ms, right_padding_ms


def run_eval(cfg: DictConfig) -> dict:
    try:
        import torch
    except ImportError as exc:
        raise SystemExit(
            "Missing torch/torchaudio. Install CUDA PyTorch on the remote training machine first."
        ) from exc

    torch.multiprocessing.set_sharing_strategy("file_system")

    config = resolved_config(cfg)
    require_sections(config, ["paths", "stage1", "stage2", "demo", "tokenizer"])
    prep = OmegaConf.to_container(cfg.prep, resolve=True)
    if not isinstance(prep, dict):
        prep = {}
    run_cfg = cfg.run

    manifest_path = str(prep.get("manifest", ""))
    if not manifest_path:
        raise SystemExit("prep.manifest is required")
    stage2_ckpt = str(prep.get("stage2_ckpt", ""))
    if not stage2_ckpt:
        raise SystemExit("prep.stage2_ckpt is required")
    left_padding_ms, right_padding_ms = _resolve_audio_padding_ms(prep)
    output_dir_override = str(prep.get("output_dir", ""))
    if output_dir_override == "outputs/eval_two_stage_kws":
        output_dir_override = ""
    output_dir = Path(str(output_dir_override or prep.get("stage2_clip_output_dir", "outputs/eval_stage2_clips")))
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_manifest(manifest_path)
    accelerator, _ = resolve_accelerator(str(run_cfg.device))
    device = torch.device(accelerator if accelerator == "cpu" else "cuda")
    runner = Stage2ClipRunner.from_config(config, prep, device)

    batch_size = int(prep.get("batch_size", 0) or 0)
    if batch_size <= 0:
        batch_size = 64
    num_workers = int(prep.get("num_workers", 0) or 0)
    if num_workers <= 0:
        num_workers = min(8, os.cpu_count() or 1)

    stream_description = runner.stream_policy.describe()
    provenance = _score_provenance(
        config,
        checkpoint_path=stage2_ckpt,
        stream=stream_description,
        left_padding_ms=left_padding_ms,
        right_padding_ms=right_padding_ms,
    )
    sequence_objective = provenance["sequence_objective"]

    runner_results = runner.run_batch(
        rows,
        batch_size=batch_size,
        num_workers=num_workers,
        left_padding_ms=left_padding_ms,
        right_padding_ms=right_padding_ms,
        include_score_details=True,
        include_eps_positions=True,
        include_seq_positions=True,
    )
    results = [
        _result_record(
            row,
            runner_result,
            sequence_objective=sequence_objective,
        )
        for row, runner_result in zip(rows, runner_results)
    ]

    results_path = output_dir / "results.jsonl"
    with results_path.open("w", encoding="utf-8") as handle:
        for record in results:
            handle.write(
                json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n"
            )

    summary = {
        "manifest": str(Path(manifest_path).resolve()),
        "num_samples": len(results),
        "output_dir": str(output_dir.resolve()),
        "audio_padding_ms": {
            "left": left_padding_ms,
            "right": right_padding_ms,
        },
        "stream": stream_description,
        "num_skipped": sum(bool(record.get("skipped", False)) for record in results),
        "provenance": provenance,
    }
    scored_results = [record for record in results if not record.get("skipped", False)]
    deployment_threshold = float(runner._demo_cfg.get("qbyt_threshold", 0.5))
    validation_cfg = (config.get("stage2", {}) or {}).get("validation", {}) or {}
    completion_threshold = float(
        validation_cfg.get("seq_diagnostic_threshold", 0.5)
    )
    ece_num_bins = int(validation_cfg.get("ece_num_bins", 15))
    summary["score_diagnostic_config"] = {
        "utterance_threshold": deployment_threshold,
        "completion_threshold": completion_threshold,
        "ece_num_bins": ece_num_bins,
    }
    labeled_summary = summarize_labeled_results(
        [_metrics_record(record) for record in scored_results],
        threshold=deployment_threshold,
    )
    if labeled_summary:
        summary["metrics"] = labeled_summary
        summary["score_heads"] = {
            "utterance": _score_head_diagnostics(
                results,
                score_field="qbyt_score",
                threshold=deployment_threshold,
                ece_num_bins=ece_num_bins,
            ),
            "completion": _score_head_diagnostics(
                results,
                score_field="completion_score",
                threshold=completion_threshold,
                ece_num_bins=ece_num_bins,
            ),
        }

    summary_path = output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(
            summary,
            handle,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )

    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))
    return summary


@hydra.main(version_base=None, config_path=str(CONFIG_DIR), config_name="config")
def main(cfg: DictConfig) -> None:
    run_eval(cfg)


if __name__ == "__main__":
    main()
