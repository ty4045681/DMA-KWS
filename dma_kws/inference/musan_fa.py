"""Helpers for MUSAN false-accept evaluation.

The official path scores Stage-II sliding windows. The two-stage path scores
Stage I spans that QbyT verifies; FA/hour then counts wake-ups per audio hour.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from dma_kws.inference.metrics import summarize_false_accept_rate
from dma_kws.inference.stage2_reporting import build_result_record

TWO_STAGE_WAKEUP_PROTOCOL = "two_stage_wakeup"


def musan_catalog_sha256(
    audio_files: Sequence[str | Path],
    musan_root: str | Path,
) -> str:
    """Hash the sorted root-relative canonical paths in a MUSAN catalog.

    This identity is independent of the host's absolute MUSAN location and the
    order in an allowlist. Paths outside the canonical root are rejected. The
    digest payload is the UTF-8 encoding of sorted relative POSIX paths, one
    per line including the final newline. This matches the split manifest's
    catalog identity contract.
    """

    root = Path(musan_root).resolve()
    relative_paths: list[str] = []
    seen: set[Path] = set()
    for audio_file in audio_files:
        resolved = Path(audio_file).resolve()
        try:
            relative = resolved.relative_to(root)
        except ValueError as exc:
            raise ValueError(
                f"MUSAN audio path is outside musan_root: {resolved} (root: {root})"
            ) from exc
        if resolved in seen:
            raise ValueError(f"Duplicate canonical MUSAN audio path: {resolved}")
        seen.add(resolved)
        relative_paths.append(relative.as_posix())

    payload = "".join(f"{path}\n" for path in sorted(relative_paths)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


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
    return build_result_record(manifest_row, runner_result)


def two_stage_wakeup_result_record(
    source_path: str,
    keyword: str,
    subset: str,
    duration_sec: float,
    keyword_phonemes: Sequence[str],
    scored: Mapping[str, Any],
    *,
    candidate_index: int,
    threshold: float,
) -> dict:
    """Build one MUSAN result row for a Stage II-scored Stage I span."""

    start_sec = float(scored["start_sec"])
    end_sec = float(scored["end_sec"])
    qbyt_score = float(scored["qbyt_score"])
    runner_result = {
        "keyword_phonemes": list(keyword_phonemes),
        "clip_span_sec": {"start_sec": start_sec, "end_sec": end_sec},
        "qbyt_score": qbyt_score,
        "detected": qbyt_score >= float(threshold),
        "threshold": float(threshold),
        "skipped": False,
    }
    if "qbyt_raw_logit" in scored:
        runner_result["qbyt_raw_logit"] = float(scored["qbyt_raw_logit"])
    manifest_row = {
        "audio_path": source_path,
        "keyword": keyword,
        "label": 0,
        "subset": subset,
        "duration_sec": float(duration_sec),
        "candidate_index": int(candidate_index),
        "start_sec": start_sec,
        "end_sec": end_sec,
    }
    record = build_result_record(manifest_row, runner_result)
    if "stage1_score" in scored and scored["stage1_score"] is not None:
        record["stage1_score"] = float(scored["stage1_score"])
    return record


def empty_false_accept_metrics(
    *,
    threshold: float,
    total_hours: float,
) -> dict[str, float]:
    """FA/hour metrics when a shard or subset has hours but no scored rows."""

    return {
        "num_samples": 0.0,
        "accuracy": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "fpr": 0.0,
        "fnr": 0.0,
        "auc": 0.0,
        "eer": 0.0,
        "eer_threshold": 0.0,
        "threshold": float(threshold),
        "tp": 0.0,
        "tn": 0.0,
        "fp": 0.0,
        "fn": 0.0,
        "total_hours": float(total_hours),
        "fa_per_hour": 0.0,
        "fa_per_1000_hours": 0.0,
    }


def false_accept_metrics(
    results: list[dict[str, Any]],
    *,
    threshold: float,
    total_hours: float,
) -> dict[str, float]:
    """Like :func:`summarize_false_accept_rate`, but zero-fill empty result lists."""

    summary = summarize_false_accept_rate(
        [metrics_record(record) for record in results],
        threshold=threshold,
        total_hours=total_hours,
    )
    if summary:
        return summary
    return empty_false_accept_metrics(threshold=threshold, total_hours=total_hours)


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
    return {
        "total_hours": float(total_hours),
        "num_samples": len(results),
        "metrics": false_accept_metrics(
            results,
            threshold=threshold,
            total_hours=total_hours,
        ),
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
    "eval_protocol",
    "keyword",
    "keyword_phonemes",
    "keyword_phonemes_source",
    "stage2_ckpt",
    "stage2_calibration",
    "window_sec",
    "hop_sec",
    "musan_root",
    "musan_catalog_sha256",
    "stream",
    "provenance",
    "amp",
    "fbank_windows",
    "batch_size",
    "locator",
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
    """Stable merge order: source path, then window or candidate index."""
    meta = record.get("manifest_meta") or {}
    index = record.get(
        "window_index",
        meta.get(
            "window_index",
            record.get("candidate_index", meta.get("candidate_index", 0)),
        ),
    )
    return (str(record.get("audio_path", "")), int(index or 0))


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


def parse_shard_dir_index(path: Path) -> int | None:
    """Return the integer in ``shard_N`` or ``None`` if the name is not that form."""

    name = path.name
    if not name.startswith("shard_"):
        return None
    suffix = name.removeprefix("shard_")
    if not suffix.isdigit() or str(int(suffix)) != suffix:
        return None
    return int(suffix)


def discover_shard_dirs(
    root: str | Path,
    *,
    num_shards: int | None = None,
) -> list[Path]:
    """Return sibling ``shard_*`` directories that contain a summary.

    When ``num_shards`` is set, only ``shard_0`` … ``shard_{N-1}`` are used.
    Leftover higher-index directories from an earlier wider run are ignored, and
    a missing expected shard fails instead of silently pooling an incomplete set.
    """
    directory = Path(root)
    found: dict[int, Path] = {}
    extras: list[str] = []
    for path in sorted(directory.iterdir()):
        if not path.is_dir() or not (path / "summary.json").is_file():
            continue
        index = parse_shard_dir_index(path)
        if index is None:
            extras.append(path.name)
            continue
        found[index] = path

    if num_shards is None:
        if found:
            return [found[index] for index in sorted(found)]
        if (directory / "summary.json").is_file():
            return [directory]
        raise FileNotFoundError(
            f"No shard_*/summary.json files found under {directory}"
        )

    expected = int(num_shards)
    if expected < 1:
        raise ValueError("num_shards must be >= 1")
    missing = [index for index in range(expected) if index not in found]
    if missing:
        missing_names = ", ".join(f"shard_{index}" for index in missing)
        raise FileNotFoundError(
            f"Expected {expected} shards under {directory}; missing {missing_names}"
        )
    leftover = sorted(index for index in found if index >= expected)
    ignored = leftover + extras
    if ignored:
        names = ", ".join(
            f"shard_{index}" if isinstance(index, int) else index for index in ignored
        )
        import warnings

        warnings.warn(
            f"Ignoring leftover shard directories under {directory}: {names}",
            UserWarning,
            stacklevel=2,
        )
    return [found[index] for index in range(expected)]


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
        "musan_audio_list_path": reference.get("musan_audio_list_path"),
        "musan_catalog_sha256": reference.get("musan_catalog_sha256"),
        "keyword": reference.get("keyword"),
        "keyword_phonemes": reference.get("keyword_phonemes"),
        "keyword_phonemes_source": reference.get("keyword_phonemes_source"),
        "stage2_ckpt": reference.get("stage2_ckpt"),
        "stage2_calibration": reference.get("stage2_calibration"),
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
    if "eval_protocol" in reference:
        merged["eval_protocol"] = reference.get("eval_protocol")
    if "locator" in reference:
        merged["locator"] = reference.get("locator")
    for key in (
        "num_stage1_candidates",
        "num_stage2_scored",
        "num_wakeups",
    ):
        if any(key in summary for summary in shard_summaries):
            merged[key] = sum(int(summary.get(key, 0)) for summary in shard_summaries)
    overall_metrics = false_accept_metrics(
        all_results,
        threshold=threshold,
        total_hours=total_hours,
    )
    merged["metrics"] = overall_metrics

    subsets: dict[str, dict[str, Any]] = {}
    for subset in sorted(set(grouped) | set(subset_hours)):
        subsets[subset] = subset_summary(
            grouped.get(subset, []),
            threshold=threshold,
            total_hours=subset_hours.get(subset, 0.0),
        )
    if subsets:
        merged["subsets"] = subsets
    return merged, all_results
