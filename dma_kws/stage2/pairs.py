"""Decoded-parquet audio helpers for Stage II paper-format preparation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Iterable, Iterator


def clip_to_audio_rel(clip_path: str) -> str:
    """Map a LibriPhrase `clips` path to a decoded-parquet `audio_rel` key.

    The aggregated parquet stores clip paths like `LP-100/<ngram>/<id>.wav`
    while the decoded shards key audio by `audio_rel` = `<ngram>/<id>.wav`.
    """
    prefix = "LP-100/"
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
