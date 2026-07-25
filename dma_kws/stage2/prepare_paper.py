"""Paper-format Stage II data preparation (parquet + clips/distances npy + fbank)."""

from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

from dma_kws.g2p import clean_phoneme_tokens, has_stress_markers, make_g2p, text_to_phonemes
from dma_kws.metrics import edit_distance
from dma_kws.stage2.pairs import clip_to_audio_rel
from dma_kws.tokenizer import unsupported_phones

PARQUET_COLUMNS = ["ngram", "ngram_g2p", "clips_file", "distances_file"]
OUTPUT_PARQUET_NAME = "aggregated_segments_with_g2p_distance.parquet"
DEFAULT_HARD_NEGATIVE_TOP_K = 5


def slug_from_ngram(ngram: str) -> str:
    """Filesystem-safe slug for npy shard filenames."""
    slug = re.sub(r"[^\w]+", "_", ngram.strip().lower()).strip("_")
    return slug or "anchor"


def resolve_fbank_rel_path(audio_path: str) -> str:
    """Map clip ``audio_path`` to precomputed fbank relative path under ``features/fbank/``."""
    path = audio_path
    for prefix in ("LP-460", "GP-1000", "LP-100"):
        path = path.replace(prefix, f"{prefix}-fbank")
    return path.replace(".wav", ".npy")


def parse_clips(raw_clips: Any) -> list[dict[str, str]]:
    """Normalize aggregated-parquet ``clips`` column to clip dict records."""
    if isinstance(raw_clips, str):
        try:
            raw_clips = json.loads(raw_clips)
        except json.JSONDecodeError:
            return [{"audio_path": raw_clips}]
    if isinstance(raw_clips, np.ndarray):
        raw_clips = raw_clips.tolist()

    clips: list[dict[str, str]] = []
    if isinstance(raw_clips, list):
        for item in raw_clips:
            if isinstance(item, dict) and "audio_path" in item:
                clips.append({"audio_path": str(item["audio_path"])})
            elif isinstance(item, str):
                clips.append({"audio_path": item})
    return clips


def parse_distances(raw_distances: Any) -> list[dict[str, Any]]:
    """Normalize optional aggregated-parquet ``distances`` column."""
    if raw_distances is None:
        return []
    if isinstance(raw_distances, float) and np.isnan(raw_distances):
        return []
    if isinstance(raw_distances, str):
        try:
            raw_distances = json.loads(raw_distances)
        except json.JSONDecodeError:
            return []
    if isinstance(raw_distances, np.ndarray):
        raw_distances = raw_distances.tolist()
    if not isinstance(raw_distances, list):
        return []

    entries: list[dict[str, Any]] = []
    for item in raw_distances:
        if isinstance(item, dict) and "ngram" in item:
            entries.append(dict(item))
        elif isinstance(item, str):
            entries.append({"ngram": item})
    return entries


def build_clips_npy(clips: list[dict], output_path: Path) -> str:
    """Save clip records as a pickle-backed npy array; return the output path string."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    array = np.array(clips, dtype=object)
    np.save(output_path, array, allow_pickle=True)
    return str(output_path)


def build_distances_npy(hard_negatives: list[dict], output_path: Path) -> str:
    """Save hard-negative metadata npy; count 0 in filename when ``hard_negatives`` is empty."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    array = np.array(hard_negatives, dtype=object)
    np.save(output_path, array, allow_pickle=True)
    return str(output_path)


def compute_fbank_for_clip(
    waveform_path: Path | str,
    fbank_out_path: Path,
    *,
    waveform: np.ndarray | None = None,
    sample_rate: int = 16000,
    num_mel_bins: int = 80,
    frame_length: int = 25,
    frame_shift: int = 10,
    dither: float = 0.1,
    window_type: str = "povey",
    backend: str = "torchaudio_kaldi",
    target_sample_rate: int | None = None,
    snip_edges: bool = True,
    low_freq: float = 20.0,
    high_freq: float = 0.0,
    extractor: Any | None = None,
) -> str:
    """Compute fbank for one clip and save it as ``.npy``."""
    import torch

    from dma_kws.stage2.features import compute_fbank

    fbank_out_path = Path(fbank_out_path)
    fbank_out_path.parent.mkdir(parents=True, exist_ok=True)

    if waveform is None:
        import soundfile as sf

        array, sr = sf.read(
            str(waveform_path),
            dtype="float32",
            always_2d=True,
        )
        array = np.asarray(array, dtype=np.float32).mean(axis=1)
        sample_rate = int(sr)
        waveform_tensor = torch.from_numpy(array).unsqueeze(0)
    else:
        array = np.asarray(waveform, dtype=np.float32)
        if array.ndim != 1:
            raise ValueError(f"Expected mono 1-D waveform, got shape {array.shape}")
        waveform_tensor = torch.from_numpy(array).unsqueeze(0)

    sample = compute_fbank(
        {
            "key": str(waveform_path),
            "wav": waveform_tensor,
            "sample_rate": sample_rate,
        },
        num_mel_bins=num_mel_bins,
        frame_length=frame_length,
        frame_shift=frame_shift,
        dither=dither,
        window_type=window_type,
        backend=backend,
        target_sample_rate=target_sample_rate,
        snip_edges=snip_edges,
        low_freq=low_freq,
        high_freq=high_freq,
        extractor=extractor,
    )
    feat = sample["feat"].detach().cpu().numpy().astype(np.float32)
    np.save(fbank_out_path, feat)
    return str(fbank_out_path)


def compute_hard_negatives_from_phonemes(
    anchor_ngram: str,
    anchor_g2p: str,
    candidates: Iterable[tuple[str, str]],
    *,
    top_k: int = DEFAULT_HARD_NEGATIVE_TOP_K,
) -> list[dict[str, str]]:
    """Pick top-K confusable anchors by phoneme edit distance within the anchor set."""
    anchor_tokens = anchor_g2p.split()
    scored: list[tuple[int, str]] = []
    for ngram, g2p in candidates:
        if ngram == anchor_ngram:
            continue
        distance = edit_distance(anchor_tokens, g2p.split())
        if distance <= 0:
            continue
        scored.append((distance, ngram))

    scored.sort(key=lambda item: (item[0], item[1]))
    return [{"ngram": ngram} for _, ngram in scored[:top_k]]


def needs_g2p_recompute(df, *, force: bool = False, sample_size: int = 200) -> bool:
    """Decide whether ``ngram_g2p`` has to be regenerated from the anchor text.

    Legacy parquet files carry stress-stripped phonemes (``AH`` instead of
    ``AH1``), which no longer exist in the vocabulary. Such a column is dropped
    and recomputed with g2p_en so training anchors match what evaluation and
    inference produce for the same text.
    """
    if force or "ngram_g2p" not in getattr(df, "columns", []):
        return True
    values = [str(value) for value in df["ngram_g2p"].head(sample_size).tolist() if value]
    return not any(has_stress_markers(value) for value in values)


def _resolve_g2p(
    row: dict[str, Any], g2p: Any | None, *, recompute: bool = False
) -> tuple[str, list[str]]:
    text = str(row["ngram"])
    if not recompute and row.get("ngram_g2p"):
        phonemes = clean_phoneme_tokens(str(row["ngram_g2p"]).split())
    else:
        if g2p is None:
            g2p = make_g2p()
        phonemes = text_to_phonemes(g2p, text)
    unsupported = unsupported_phones(phonemes)
    if unsupported:
        raise ValueError(
            f"Anchor {text!r} produced phonemes outside the vocabulary: "
            f"{', '.join(unsupported)} (full sequence: {' '.join(phonemes)})"
        )
    return " ".join(phonemes), phonemes


@dataclass(frozen=True)
class _FbankJob:
    audio_path: str
    fbank_path: Path
    waveform: np.ndarray
    sample_rate: int


def _run_fbank_job(job: _FbankJob, compute_fbank: Callable[..., str]) -> str:
    import torch

    torch.set_num_threads(1)
    return compute_fbank(
        job.audio_path,
        job.fbank_path,
        waveform=job.waveform,
        sample_rate=job.sample_rate,
    )


def _compute_fbank_jobs(
    jobs: list[_FbankJob],
    *,
    compute_fbank: Callable[..., str],
    num_workers: int = 1,
    on_progress: Callable[[str, int], None] | None = None,
) -> int:
    if not jobs:
        return 0

    if num_workers <= 1:
        for job in jobs:
            _run_fbank_job(job, compute_fbank)
            if on_progress is not None:
                on_progress("fbank", 1)
        return len(jobs)

    written = 0
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(_run_fbank_job, job, compute_fbank) for job in jobs]
        for future in as_completed(futures):
            future.result()
            written += 1
            if on_progress is not None:
                on_progress("fbank", 1)
    return written


@dataclass(frozen=True)
class FbankTarget:
    """A fbank file that still needs to be computed for a given decoded clip."""

    audio_path: str
    fbank_path: Path


def build_anchor_metadata(
    df,
    *,
    clips_dir: Path,
    distances_dir: Path,
    fbank_dir: Path,
    limit_anchors: int = 0,
    hard_negative_top_k: int = DEFAULT_HARD_NEGATIVE_TOP_K,
    g2p: Any | None = None,
    force_g2p_recompute: bool = False,
    on_progress: Callable[[str, int], None] | None = None,
) -> tuple[Any, dict[str, FbankTarget], dict[str, int]]:
    """Write paper-format anchor metadata without touching any audio.

    Returns ``(paper_df, fbank_targets, stats)`` where ``fbank_targets`` maps each
    still-missing decoded ``audio_rel`` to the clip/fbank paths needed to produce
    it. Audio is never loaded here, so memory stays bounded regardless of dataset
    size; fbank computation is handled separately (see ``stream_fbank_from_decoded``).
    """
    import pandas as pd

    clips_dir = Path(clips_dir)
    distances_dir = Path(distances_dir)
    fbank_dir = Path(fbank_dir)

    rows: list[dict[str, str]] = []
    pending: list[dict[str, Any]] = []
    recompute_g2p = needs_g2p_recompute(df, force=force_g2p_recompute)

    for _, row in df.iterrows():
        text = str(row["ngram"])
        clips = parse_clips(row.get("clips"))
        if not clips:
            continue

        g2p_text, _phonemes = _resolve_g2p(row, g2p, recompute=recompute_g2p)
        pending.append({"ngram": text, "ngram_g2p": g2p_text, "clips": clips, "distances": parse_distances(row.get("distances"))})
        if limit_anchors and len(pending) >= limit_anchors:
            break

    stats = {
        "anchors": 0,
        "missing_audio": 0,
        "fbank_written": 0,
        "fbank_skipped": 0,
        "clips_total": 0,
        "g2p_recomputed": int(recompute_g2p),
    }

    candidate_pairs = [(item["ngram"], item["ngram_g2p"]) for item in pending]
    fbank_targets: dict[str, FbankTarget] = {}

    if on_progress is not None:
        on_progress("anchor_total", len(pending))

    for item in pending:
        ngram = item["ngram"]
        slug = slug_from_ngram(ngram)
        clips = item["clips"]
        clips_path = clips_dir / f"clips-{len(clips)}-{slug}.npy"
        clips_file = build_clips_npy(clips, clips_path)

        hard_negatives = item["distances"]
        if not hard_negatives:
            hard_negatives = compute_hard_negatives_from_phonemes(
                ngram,
                item["ngram_g2p"],
                candidate_pairs,
                top_k=hard_negative_top_k,
            )
        distances_path = distances_dir / f"dist-{len(hard_negatives)}-{slug}.npy"
        distances_file = build_distances_npy(hard_negatives, distances_path)

        for clip in clips:
            stats["clips_total"] += 1
            audio_path = clip["audio_path"]
            audio_rel = clip_to_audio_rel(audio_path)
            fbank_path = fbank_dir / resolve_fbank_rel_path(audio_path)

            if fbank_path.exists():
                stats["fbank_skipped"] += 1
                continue

            fbank_targets[audio_rel] = FbankTarget(audio_path=audio_path, fbank_path=fbank_path)

        rows.append(
            {
                "ngram": ngram,
                "ngram_g2p": item["ngram_g2p"],
                "clips_file": clips_file,
                "distances_file": distances_file,
            }
        )
        stats["anchors"] += 1
        if on_progress is not None:
            on_progress("anchor", 1)

    paper_df = pd.DataFrame(rows, columns=PARQUET_COLUMNS)
    return paper_df, fbank_targets, stats


def stream_fbank_from_decoded(
    decoded_parquet_paths: Iterable[Path],
    fbank_targets: dict[str, FbankTarget],
    *,
    read_parquet: Callable[[Path], Any],
    compute_fbank: Callable[..., str] = compute_fbank_for_clip,
    num_workers: int = 1,
    on_progress: Callable[[str, int], None] | None = None,
    on_shard_done: Callable[[Path, int], None] | None = None,
) -> tuple[int, int]:
    """Compute fbank features by streaming decoded shards one at a time.

    Each shard is read, filtered to rows whose ``audio_rel`` is still needed,
    turned into fbank ``.npy`` files, then released before the next shard is read.
    Peak memory is therefore bounded by a single decoded shard rather than the
    full dataset. Returns ``(fbank_written, missing_audio)``.
    """
    if on_progress is not None:
        on_progress("fbank_total", len(fbank_targets))

    remaining: set[str] = set(fbank_targets)
    written = 0

    for parquet_path in decoded_parquet_paths:
        if not remaining:
            if on_shard_done is not None:
                on_shard_done(parquet_path, 0)
            continue

        frame = read_parquet(parquet_path)
        matched = frame[frame["audio_rel"].isin(remaining)]

        jobs: list[_FbankJob] = []
        for audio_rel, audio, sample_rate in zip(
            matched["audio_rel"], matched["audio"], matched["sampling_rate"]
        ):
            audio_rel = str(audio_rel)
            if audio_rel not in remaining:
                continue
            remaining.discard(audio_rel)
            target = fbank_targets[audio_rel]
            array = np.asarray(audio, dtype=np.float32)
            if array.ndim != 1:
                raise SystemExit(
                    f"Expected mono 1-D audio for {audio_rel}, got shape {array.shape}"
                )
            jobs.append(
                _FbankJob(
                    audio_path=target.audio_path,
                    fbank_path=target.fbank_path,
                    waveform=array,
                    sample_rate=int(sample_rate),
                )
            )

        shard_written = _compute_fbank_jobs(
            jobs,
            compute_fbank=compute_fbank,
            num_workers=num_workers,
            on_progress=on_progress,
        )
        written += shard_written
        if on_shard_done is not None:
            on_shard_done(parquet_path, shard_written)
        del frame, matched, jobs

    return written, len(remaining)


def convert_aggregated_to_paper_parquet(
    df,
    *,
    clips_dir: Path,
    distances_dir: Path,
    fbank_dir: Path,
    audio_by_rel: dict[str, tuple[np.ndarray, int]] | None = None,
    limit_anchors: int = 0,
    hard_negative_top_k: int = DEFAULT_HARD_NEGATIVE_TOP_K,
    g2p: Any | None = None,
    force_g2p_recompute: bool = False,
    compute_fbank: Callable[..., str] = compute_fbank_for_clip,
    num_workers: int = 1,
    on_progress: Callable[[str, int], None] | None = None,
) -> tuple[Any, dict[str, int]]:
    """Convert aggregated parquet rows to paper-format metadata using in-memory audio.

    Returns ``(paper_df, stats)`` where ``stats`` counts missing clips/audio/fbank.
    Suitable for small datasets/tests where all referenced audio fits in
    ``audio_by_rel``; large datasets should use ``build_anchor_metadata`` plus
    ``stream_fbank_from_decoded`` to avoid materializing every waveform at once.
    """
    audio_by_rel = audio_by_rel or {}

    paper_df, fbank_targets, stats = build_anchor_metadata(
        df,
        clips_dir=clips_dir,
        distances_dir=distances_dir,
        fbank_dir=fbank_dir,
        limit_anchors=limit_anchors,
        hard_negative_top_k=hard_negative_top_k,
        g2p=g2p,
        force_g2p_recompute=force_g2p_recompute,
        on_progress=on_progress,
    )

    fbank_jobs: list[_FbankJob] = []
    for audio_rel, target in fbank_targets.items():
        audio_entry = audio_by_rel.get(audio_rel)
        if audio_entry is None:
            stats["missing_audio"] += 1
            continue
        waveform, sample_rate = audio_entry
        fbank_jobs.append(
            _FbankJob(
                audio_path=target.audio_path,
                fbank_path=target.fbank_path,
                waveform=waveform,
                sample_rate=sample_rate,
            )
        )

    if on_progress is not None:
        on_progress("fbank_total", len(fbank_jobs))

    stats["fbank_written"] = _compute_fbank_jobs(
        fbank_jobs,
        compute_fbank=compute_fbank,
        num_workers=num_workers,
        on_progress=on_progress,
    )

    return paper_df, stats
