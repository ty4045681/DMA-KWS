"""Helpers for MUSAN false-accept evaluation with sliding-window Stage-II inference."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from dma_kws.inference.metrics import summarize_false_accept_rate
from dma_kws.inference.stage2_reporting import build_result_record


def audio_duration_sec(path: str | Path) -> float:
    """Return audio duration in seconds using ``torchaudio.info``."""
    try:
        import torchaudio
    except ImportError as exc:
        raise SystemExit(
            "Missing torch/torchaudio. Install CUDA PyTorch on the remote training machine first."
        ) from exc

    info = torchaudio.info(str(path))
    return info.num_frames / info.sample_rate


def detect_subset(audio_path: str | Path, musan_root: Path) -> str:
    """Return the top-level MUSAN subset name (music/noise/speech/...)."""
    try:
        rel = Path(audio_path).resolve().relative_to(musan_root.resolve())
    except ValueError:
        return "other"
    if not rel.parts:
        return "other"
    return rel.parts[0]


def musan_result_record(
    source_path: str,
    keyword: str,
    subset: str,
    runner_result: dict,
    *,
    window_index: int | None = None,
    sequence_objective: Mapping[str, object] | None = None,
    qbyt_readout: Mapping[str, object] | None = None,
) -> dict:
    """Build a rich Stage-II result row for one MUSAN sliding window."""

    span = runner_result["clip_span_sec"]
    manifest_row = {
        "audio_path": source_path,
        "keyword": keyword,
        "label": 0,
        "subset": subset,
        "start_sec": float(span["start_sec"]),
        "end_sec": float(span["end_sec"]),
    }
    if window_index is not None:
        manifest_row["window_index"] = int(window_index)
    return build_result_record(
        manifest_row,
        runner_result,
        sequence_objective=sequence_objective,
        qbyt_readout=qbyt_readout,
    )


def metrics_record(record: dict) -> dict:
    """Convert a result row into the metric-summarizer input format."""
    metrics_row = dict(record)
    metrics_row["best_qbyt_score"] = float(record.get("qbyt_score", 0.0))
    return metrics_row


def subset_summary(
    results: list[dict[str, Any]],
    *,
    threshold: float,
    total_hours: float,
) -> dict[str, Any]:
    """Build a per-subset metric dict consistent with the overall summary."""
    summary = summarize_false_accept_rate(
        [metrics_record(record) for record in results],
        threshold=threshold,
        total_hours=total_hours,
    )
    return {
        "total_hours": float(total_hours),
        "num_samples": len(results),
        "metrics": summary,
    }


def group_results_by_subset(
    results: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Group MUSAN window results by their ``manifest_meta.subset`` value."""
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in results:
        subset = record.get("manifest_meta", {}).get("subset", "other")
        groups[subset].append(record)
    return dict(groups)
