"""Paper-format Stage II data preparation (parquet + clips/distances npy + fbank)."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

from dma_kws.g2p import clean_phoneme_tokens, make_g2p, text_to_phonemes
from dma_kws.metrics import edit_distance
from dma_kws.stage2.pairs import clip_to_audio_rel

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
) -> str:
    """Compute Kaldi fbank for one clip and save as ``.npy``; return output path string."""
    import torch

    from dma_kws.stage2.features import compute_fbank

    fbank_out_path = Path(fbank_out_path)
    fbank_out_path.parent.mkdir(parents=True, exist_ok=True)

    if waveform is None:
        import torchaudio

        loaded, sr = torchaudio.load(str(waveform_path))
        if loaded.shape[0] > 1:
            loaded = loaded.mean(dim=0, keepdim=True)
        sample_rate = int(sr)
        waveform_tensor = loaded
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


def _resolve_g2p(row: dict[str, Any], g2p: Any | None) -> tuple[str, list[str]]:
    text = str(row["ngram"])
    if "ngram_g2p" in row and row.get("ngram_g2p"):
        phonemes = clean_phoneme_tokens(str(row["ngram_g2p"]).split())
    else:
        if g2p is None:
            g2p = make_g2p()
        phonemes = text_to_phonemes(g2p, text)
    return " ".join(phonemes), phonemes


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
    compute_fbank: Callable[..., str] = compute_fbank_for_clip,
) -> tuple[Any, dict[str, int]]:
    """Convert aggregated LibriPhrase parquet rows to paper-format metadata.

    Returns ``(paper_df, stats)`` where ``stats`` counts missing clips/audio/fbank.
    """
    import pandas as pd

    clips_dir = Path(clips_dir)
    distances_dir = Path(distances_dir)
    fbank_dir = Path(fbank_dir)
    audio_by_rel = audio_by_rel or {}

    rows: list[dict[str, str]] = []
    anchor_g2p_by_ngram: dict[str, str] = {}
    pending: list[dict[str, Any]] = []

    for _, row in df.iterrows():
        text = str(row["ngram"])
        clips = parse_clips(row.get("clips"))
        if not clips:
            continue

        g2p_text, _phonemes = _resolve_g2p(row, g2p)
        anchor_g2p_by_ngram[text] = g2p_text
        pending.append({"ngram": text, "ngram_g2p": g2p_text, "clips": clips, "distances": parse_distances(row.get("distances"))})
        if limit_anchors and len(pending) >= limit_anchors:
            break

    stats = {"anchors": 0, "missing_audio": 0, "fbank_written": 0}

    candidate_pairs = [(item["ngram"], item["ngram_g2p"]) for item in pending]

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
            audio_path = clip["audio_path"]
            audio_rel = clip_to_audio_rel(audio_path)
            fbank_rel = resolve_fbank_rel_path(audio_path)
            fbank_path = fbank_dir / fbank_rel

            if fbank_path.exists():
                continue

            audio_entry = audio_by_rel.get(audio_rel)
            if audio_entry is None:
                stats["missing_audio"] += 1
                continue

            waveform, sample_rate = audio_entry
            compute_fbank(
                audio_path,
                fbank_path,
                waveform=waveform,
                sample_rate=sample_rate,
            )
            stats["fbank_written"] += 1

        rows.append(
            {
                "ngram": ngram,
                "ngram_g2p": item["ngram_g2p"],
                "clips_file": clips_file,
                "distances_file": distances_file,
            }
        )
        stats["anchors"] += 1

    paper_df = pd.DataFrame(rows, columns=PARQUET_COLUMNS)
    return paper_df, stats
