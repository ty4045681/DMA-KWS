"""Prepare keyword adaptation data: fbank features and train/eval manifests."""

from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from dma_kws.g2p import make_g2p, text_to_phonemes
from dma_kws.stage2.adapt_paths import neg_slug_to_text, slugify, wav_to_fbank_mirror
from dma_kws.stage2.features import waveform_to_fbank


@dataclass
class AdaptSample:
    audio_path: str
    text: str
    label: int
    phase: str


def _import_torchaudio():
    try:
        import torchaudio
    except ImportError as exc:
        raise ImportError(
            "Missing torchaudio. Install CUDA PyTorch/torchaudio on the training machine first."
        ) from exc
    return torchaudio


def validate_g2p(text: str, g2p: Any) -> None:
    phonemes = text_to_phonemes(g2p, text)
    if not phonemes:
        raise ValueError(f"G2P produced empty phoneme sequence for text: {text!r}")


def scan_raw_tree(data_root: Path, keyword: str) -> list[AdaptSample]:
    """Scan ``raw/{tts,real}/{positive,negative/*}`` for adaptation samples."""
    keyword = keyword.strip()
    raw_root = data_root / "raw"
    if not raw_root.is_dir():
        raise FileNotFoundError(f"Raw adaptation directory not found: {raw_root}")

    samples: list[AdaptSample] = []
    for phase_dir in sorted(raw_root.iterdir()):
        if not phase_dir.is_dir():
            continue
        phase = phase_dir.name
        positive_dir = phase_dir / "positive"
        if positive_dir.is_dir():
            for wav_path in sorted(positive_dir.rglob("*.wav")):
                rel = wav_path.relative_to(data_root)
                samples.append(
                    AdaptSample(
                        audio_path=str(rel),
                        text=keyword,
                        label=1,
                        phase=phase,
                    )
                )

        negative_root = phase_dir / "negative"
        if negative_root.is_dir():
            for neg_dir in sorted(path for path in negative_root.iterdir() if path.is_dir()):
                neg_text = neg_slug_to_text(neg_dir.name)
                for wav_path in sorted(neg_dir.rglob("*.wav")):
                    rel = wav_path.relative_to(data_root)
                    samples.append(
                        AdaptSample(
                            audio_path=str(rel),
                            text=neg_text,
                            label=0,
                            phase=phase,
                        )
                    )
    if not samples:
        raise ValueError(f"No wav files found under {raw_root}")
    return samples


def load_manifest_csv(path: Path) -> list[AdaptSample]:
    rows: list[AdaptSample] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"audio_path", "text", "label"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Manifest {path} must contain columns: {sorted(required)}")
        phase = "custom"
        if "phase" in (reader.fieldnames or []):
            pass
        for row in reader:
            rows.append(
                AdaptSample(
                    audio_path=str(row["audio_path"]),
                    text=str(row["text"]),
                    label=int(row["label"]),
                    phase=str(row.get("phase", phase)),
                )
            )
    return rows


def split_train_eval(
    samples: list[AdaptSample],
    *,
    eval_fraction: float,
    seed: int,
) -> tuple[list[AdaptSample], list[AdaptSample]]:
    rng = random.Random(seed)
    indices = list(range(len(samples)))
    rng.shuffle(indices)
    eval_count = max(1, int(round(len(samples) * eval_fraction))) if samples else 0
    eval_indices = set(indices[:eval_count])
    train = [samples[i] for i in range(len(samples)) if i not in eval_indices]
    eval_set = [samples[i] for i in range(len(samples)) if i in eval_indices]
    return train, eval_set


def compute_and_save_fbank(
    wav_path: Path,
    fbank_path: Path,
    *,
    fbank_params: dict[str, Any],
    skip_existing: bool = True,
) -> bool:
    if skip_existing and fbank_path.is_file():
        return False
    fbank_path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio = _import_torchaudio()
    waveform, sample_rate = torchaudio.load(str(wav_path))
    feat = waveform_to_fbank(waveform, sample_rate=sample_rate, **fbank_params)
    np.save(fbank_path, feat.cpu().numpy())
    return True


def write_train_manifest(path: Path, rows: Iterable[AdaptSample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["audio_path", "text", "label"])
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "audio_path": row.audio_path,
                    "text": row.text,
                    "label": row.label,
                }
            )


def write_eval_manifest(path: Path, rows: Iterable[AdaptSample], keyword: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["audio_path", "text", "keyword", "label"])
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "audio_path": row.audio_path,
                    "text": row.text,
                    "keyword": keyword,
                    "label": row.label,
                }
            )


def prepare_keyword_adaptation(
    *,
    keyword: str,
    data_root: Path,
    fbank_params: dict[str, Any],
    eval_fraction: float = 0.2,
    eval_seed: int = 2025,
    manifest_csv: Path | None = None,
    skip_existing: bool = True,
) -> dict[str, Any]:
    """Prepare fbank features and manifests for one keyword slug tree."""
    g2p = make_g2p()
    if manifest_csv is not None and manifest_csv.is_file():
        all_samples = load_manifest_csv(manifest_csv)
    else:
        all_samples = scan_raw_tree(data_root, keyword)

    for sample in all_samples:
        validate_g2p(sample.text, g2p)

    written = 0
    skipped = 0
    for sample in all_samples:
        wav_path = data_root / sample.audio_path
        if not wav_path.is_file():
            raise FileNotFoundError(f"Missing wav file: {wav_path}")
        fbank_path = wav_to_fbank_mirror(data_root / "fbank", wav_path)
        if compute_and_save_fbank(
            wav_path,
            fbank_path,
            fbank_params=fbank_params,
            skip_existing=skip_existing,
        ):
            written += 1
        else:
            skipped += 1

    manifest_dir = data_root / "manifests"
    stats: dict[str, Any] = {"keyword": keyword, "slug": slugify(keyword), "phases": {}}

    by_phase: dict[str, list[AdaptSample]] = {}
    for sample in all_samples:
        by_phase.setdefault(sample.phase, []).append(sample)

    for phase, phase_samples in sorted(by_phase.items()):
        train_rows, eval_rows = split_train_eval(
            phase_samples,
            eval_fraction=eval_fraction,
            seed=eval_seed,
        )
        train_path = manifest_dir / f"{phase}_train.csv"
        eval_path = manifest_dir / f"{phase}_eval.csv"
        write_train_manifest(train_path, train_rows)
        write_eval_manifest(eval_path, eval_rows, keyword)
        stats["phases"][phase] = {
            "total": len(phase_samples),
            "train": len(train_rows),
            "eval": len(eval_rows),
            "train_manifest": str(train_path),
            "eval_manifest": str(eval_path),
        }

    stats["fbank_written"] = written
    stats["fbank_skipped"] = skipped
    return stats
