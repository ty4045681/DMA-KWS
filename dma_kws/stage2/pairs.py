"""Decoded-parquet audio helpers for Stage II paper-format preparation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping

STAGE2_DATASETS: dict[str, dict[str, str]] = {
    "LP-100": {
        "clip_prefix": "LP-100/",
        "decoded_glob": "LP-100-decoded-*.parquet",
        "config_root_key": "libriphrase100_root",
    },
    "LP-460": {
        "clip_prefix": "LP-460/",
        "decoded_glob": "LP-460-decoded-*.parquet",
        "config_root_key": "libriphrase460_root",
    },
    "GP-1000": {
        "clip_prefix": "GP-1000/",
        "decoded_glob": "LP-100-decoded-*.parquet",
        "config_root_key": "gigaphrase1000_root",
    },
}


def infer_dataset_id(clips: Iterable[str]) -> str | None:
    """Infer dataset ID (e.g. ``GP-1000``) from clip ``audio_path`` samples."""
    counts: dict[str, int] = {}
    for clip_path in clips:
        for dataset_id, entry in STAGE2_DATASETS.items():
            if clip_path.startswith(entry["clip_prefix"]):
                counts[dataset_id] = counts.get(dataset_id, 0) + 1
                break
    if not counts:
        return None
    return max(counts, key=counts.get)


def decoded_glob_for_dataset(dataset_id: str | None) -> str:
    """Return the decoded-parquet shard glob for a registered dataset."""
    if dataset_id is None:
        return "LP-100-decoded-*.parquet"
    entry = STAGE2_DATASETS.get(dataset_id)
    if entry is None:
        raise ValueError(f"Unknown dataset_id: {dataset_id}")
    return entry["decoded_glob"]


def resolve_data_root(paths: Mapping[str, Any], dataset_id: str | None) -> Path | None:
    """Resolve the decoded-parquet root from config ``paths`` for a dataset."""
    if dataset_id is None:
        return None
    entry = STAGE2_DATASETS.get(dataset_id)
    if entry is None:
        return None
    root = paths.get(entry["config_root_key"])
    if root is None or root == "":
        return None
    return Path(str(root))


def clip_to_audio_rel(clip_path: str, *, dataset_id: str | None = None) -> str:
    """Map a ``clips`` path to a decoded-parquet ``audio_rel`` key.

    Aggregated parquet stores clip paths like ``LP-100/<ngram>/<id>.wav`` or
    ``GP-1000/<ngram>/<id>.wav`` while decoded shards key audio by
    ``audio_rel`` = ``<ngram>/<id>.wav``.
    """
    if dataset_id is not None:
        entry = STAGE2_DATASETS.get(dataset_id)
        if entry is None:
            raise ValueError(f"Unknown dataset_id: {dataset_id}")
        prefix = entry["clip_prefix"]
        if clip_path.startswith(prefix):
            return clip_path[len(prefix):]
        return clip_path

    for entry in STAGE2_DATASETS.values():
        prefix = entry["clip_prefix"]
        if clip_path.startswith(prefix):
            return clip_path[len(prefix):]
    return clip_path


def iter_decoded_audio_rows(
    parquet_paths: Iterable[Path],
    needed_keys: set[str],
    *,
    read_parquet: Callable[[Path], Any],
) -> Iterator[tuple[str, Any, int]]:
    """Yield (audio_rel, audio, sampling_rate) for rows whose audio_rel is needed.

    Streams one parquet shard at a time via the injected ``read_parquet``
    callable so the full ~5 GB of decoded audio never materializes at once.
    """
    remaining = set(needed_keys)
    for parquet_path in parquet_paths:
        if not remaining:
            break
        frame = read_parquet(parquet_path)
        for _, row in frame.iterrows():
            audio_rel = str(row["audio_rel"])
            if audio_rel not in remaining:
                continue
            remaining.discard(audio_rel)
            yield audio_rel, row["audio"], int(row["sampling_rate"])
            if not remaining:
                break
