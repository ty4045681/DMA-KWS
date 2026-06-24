"""Precompute Stage I fbank features for JSONL manifests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import numpy as np

from dma_kws.audio import extract_fbank, load_audio
from dma_kws.jsonl import read_jsonl


def resolve_stage1_fbank_path(
    fbank_root: Path | str,
    wav_path: Path | str,
    *,
    audio_root: Path | str | None = None,
) -> Path:
    """Map a wav path to its precomputed fbank ``.npy`` location under ``fbank_root``."""
    wav_path = Path(wav_path)
    fbank_root = Path(fbank_root)

    if audio_root is not None:
        audio_root = Path(audio_root)
        try:
            rel = wav_path.relative_to(audio_root)
            return fbank_root / rel.with_suffix(".npy")
        except ValueError:
            pass

    return fbank_root / wav_path.with_suffix(".npy").name


def _feat_to_numpy(feat) -> np.ndarray:
    if hasattr(feat, "detach"):
        return feat.detach().cpu().numpy()
    return np.asarray(feat)


def compute_fbank_for_wav(
    wav_path: Path | str,
    fbank_out_path: Path | str,
    *,
    sample_rate: int = 16000,
    num_mel_bins: int = 80,
    dither: float = 0.1,
) -> str:
    """Compute Kaldi fbank for one utterance and save as ``.npy``; return output path."""
    wav_path = Path(wav_path)
    fbank_out_path = Path(fbank_out_path)
    fbank_out_path.parent.mkdir(parents=True, exist_ok=True)

    waveform, sr = load_audio(wav_path, sample_rate=sample_rate)
    feat = extract_fbank(
        waveform,
        num_mel_bins=num_mel_bins,
        sample_rate=sr,
        dither=dither,
    )
    np.save(fbank_out_path, _feat_to_numpy(feat))
    return str(fbank_out_path)


def resolve_record_fbank_path(
    record: dict[str, Any],
    *,
    fbank_root: Path | str | None = None,
    audio_root: Path | str | None = None,
) -> Path | None:
    """Resolve precomputed fbank path for a manifest record, if known."""
    if "fbank_path" in record and record["fbank_path"]:
        return Path(record["fbank_path"])
    if fbank_root is None:
        return None
    return resolve_stage1_fbank_path(fbank_root, record["wav_path"], audio_root=audio_root)


def prepare_manifest_fbank(
    manifest_path: Path | str,
    *,
    fbank_root: Path | str,
    output_manifest_path: Path | str | None = None,
    audio_root: Path | str | None = None,
    sample_rate: int = 16000,
    num_mel_bins: int = 80,
    dither: float = 0.1,
    limit: int = 0,
    skip_existing: bool = True,
    write_fbank_path: bool = True,
    compute_fn: Callable[..., str] | None = None,
) -> tuple[Path, int, int]:
    """Precompute fbank for manifest records and optionally rewrite manifest with ``fbank_path``."""
    manifest_path = Path(manifest_path)
    fbank_root = Path(fbank_root)
    output_manifest_path = Path(output_manifest_path or manifest_path)
    records = read_jsonl(manifest_path)

    compute = compute_fn or compute_fbank_for_wav
    written = 0
    skipped = 0
    output_manifest_path.parent.mkdir(parents=True, exist_ok=True)

    with output_manifest_path.open("w", encoding="utf-8") as writer:
        for index, record in enumerate(records):
            if limit and index >= limit:
                break

            wav_path = Path(record["wav_path"])
            fbank_path = resolve_stage1_fbank_path(fbank_root, wav_path, audio_root=audio_root)

            if skip_existing and fbank_path.exists():
                skipped += 1
            else:
                compute(
                    wav_path,
                    fbank_path,
                    sample_rate=sample_rate,
                    num_mel_bins=num_mel_bins,
                    dither=dither,
                )
                written += 1

            out_record = dict(record)
            if write_fbank_path:
                out_record["fbank_path"] = str(fbank_path)
            writer.write(json.dumps(out_record, ensure_ascii=False) + "\n")

    return output_manifest_path, written, skipped
