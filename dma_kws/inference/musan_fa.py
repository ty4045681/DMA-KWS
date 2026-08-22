"""Helpers for MUSAN false-accept evaluation with sliding-window Stage-II inference."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
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


_MERGE_IDENTITY_KEYS = (
    "keyword",
    "keyword_phonemes",
    "keyword_phonemes_source",
    "stage2_ckpt",
    "window_sec",
    "hop_sec",
    "musan_root",
    "stream",
    "provenance",
    "amp",
    "fbank_windows",
    "batch_size",
)


def assign_files_to_shards(
    files: Sequence[Mapping[str, Any]],
    *,
    num_shards: int,
) -> list[list[dict[str, Any]]]:
    """Partition files across shards, balancing total duration.

    ``files`` must already be in the desired global order (normally sorted
    audio paths). Assignment is greedy: each next file goes to the currently
    lightest shard. That is deterministic and keeps long speech files from
    stacking on one GPU.
    """
    shard_count = int(num_shards)
    if shard_count < 1:
        raise ValueError("num_shards must be >= 1")
    buckets: list[list[dict[str, Any]]] = [[] for _ in range(shard_count)]
    hours = [0.0] * shard_count
    for row in files:
        duration_hours = float(row["duration_sec"]) / 3600.0
        shard_index = min(range(shard_count), key=lambda index: hours[index])
        assigned = dict(row)
        assigned["shard_index"] = shard_index
        buckets[shard_index].append(assigned)
        hours[shard_index] += duration_hours
    return buckets


def select_shard(
    files: Sequence[Mapping[str, Any]],
    *,
    num_shards: int,
    shard_index: int,
) -> list[dict[str, Any]]:
    """Return the files assigned to ``shard_index``."""
    shard_count = int(num_shards)
    index = int(shard_index)
    if shard_count < 1:
        raise ValueError("num_shards must be >= 1")
    if index < 0 or index >= shard_count:
        raise ValueError(
            f"shard_index must be in [0, {shard_count}), got {shard_index}"
        )
    if shard_count == 1:
        return [dict(row) for row in files]
    return assign_files_to_shards(files, num_shards=shard_count)[index]


def result_sort_key(record: Mapping[str, Any]) -> tuple[str, int]:
    """Stable merge order: source path, then hop-grid window index."""
    meta = record.get("manifest_meta") or {}
    window_index = record.get("window_index", meta.get("window_index", 0))
    return (str(record.get("audio_path", "")), int(window_index or 0))


def load_jsonl_records(path: str | Path) -> list[dict[str, Any]]:
    """Load one JSON object per line from ``path``."""
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                record = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path}:{line_number} is not valid JSON"
                ) from exc
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number} must be a JSON object")
            records.append(record)
    return records


def discover_shard_dirs(root: str | Path) -> list[Path]:
    """Return sibling ``shard_*`` directories that contain a summary."""
    directory = Path(root)
    shards = [
        path
        for path in sorted(directory.iterdir())
        if path.is_dir()
        and path.name.startswith("shard_")
        and (path / "summary.json").is_file()
    ]
    if shards:
        return shards
    if (directory / "summary.json").is_file():
        return [directory]
    raise FileNotFoundError(f"No shard_*/summary.json files found under {directory}")


def merge_musan_summaries(
    shard_summaries: Sequence[Mapping[str, Any]],
    shard_results: Sequence[Sequence[Mapping[str, Any]]],
    *,
    output_dir: str | Path,
    threshold: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Pool shard outputs into one single-process-style summary + result list."""

    if not shard_summaries:
        raise ValueError("At least one shard summary is required")
    if len(shard_summaries) != len(shard_results):
        raise ValueError("Each shard summary needs a matching results list")

    reference = dict(shard_summaries[0])
    for summary in shard_summaries[1:]:
        for key in _MERGE_IDENTITY_KEYS:
            if summary.get(key) != reference.get(key):
                raise ValueError(
                    f"Cannot merge shards with disagreeing {key}: "
                    f"{reference.get(key)!r} vs {summary.get(key)!r}"
                )

    all_results = [
        dict(record) for results in shard_results for record in results
    ]
    all_results.sort(key=result_sort_key)
    grouped = group_results_by_subset(all_results)

    total_hours = sum(float(summary.get("total_hours", 0.0)) for summary in shard_summaries)
    total_files = sum(int(summary.get("total_files", 0)) for summary in shard_summaries)
    subset_hours: dict[str, float] = defaultdict(float)
    for summary in shard_summaries:
        for subset, subset_data in (summary.get("subsets") or {}).items():
            subset_hours[subset] += float(subset_data.get("total_hours", 0.0))

    merged: dict[str, Any] = {
        "num_samples": len(all_results),
        "num_skipped": sum(
            bool(record.get("skipped", False)) for record in all_results
        ),
        "output_dir": str(Path(output_dir).resolve()),
        "musan_root": reference.get("musan_root"),
        "keyword": reference.get("keyword"),
        "keyword_phonemes": reference.get("keyword_phonemes"),
        "keyword_phonemes_source": reference.get("keyword_phonemes_source"),
        "stage2_ckpt": reference.get("stage2_ckpt"),
        "window_sec": reference.get("window_sec"),
        "hop_sec": reference.get("hop_sec"),
        "batch_size": reference.get("batch_size"),
        "amp": reference.get("amp", "off"),
        "fbank_windows": reference.get("fbank_windows", "independent"),
        "num_shards": int(reference.get("num_shards") or len(shard_summaries)),
        "total_files": total_files,
        "total_hours": total_hours,
        "stream": reference.get("stream"),
        "provenance": reference.get("provenance"),
    }
    overall_metrics = summarize_false_accept_rate(
        [metrics_record(record) for record in all_results],
        threshold=threshold,
        total_hours=total_hours,
    )
    if overall_metrics:
        merged["metrics"] = overall_metrics

    subsets: dict[str, dict[str, Any]] = {}
    for subset in sorted(grouped):
        subsets[subset] = subset_summary(
            grouped[subset],
            threshold=threshold,
            total_hours=subset_hours.get(subset, 0.0),
        )
    if subsets:
        merged["subsets"] = subsets
    return merged, all_results
