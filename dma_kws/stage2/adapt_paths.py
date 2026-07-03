"""Path helpers for keyword continual adaptation."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any


def slugify(keyword: str) -> str:
    """Convert keyword text to a filesystem slug, e.g. 'hey eva' -> 'hey_eva'."""
    text = keyword.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_") or "keyword"


def neg_slug_to_text(slug: str) -> str:
    """Convert negative folder slug back to phrase text."""
    return slug.replace("_", " ").strip()


def adapt_data_root(config: dict[str, Any], keyword: str) -> Path:
    adapt = config.get("adapt", {}) or {}
    slug = slugify(keyword)
    if adapt.get("data_root"):
        return Path(str(adapt["data_root"]))
    paths = config.get("paths", {})
    processed_root = Path(paths.get("processed_root", "data/dma-kws/processed"))
    return processed_root / "adapt" / slug


def adapt_exp_root(config: dict[str, Any], keyword: str) -> Path:
    adapt = config.get("adapt", {}) or {}
    slug = slugify(keyword)
    if adapt.get("exp_root"):
        return Path(str(adapt["exp_root"]))
    paths = config.get("paths", {})
    exp_root = Path(paths.get("exp_root", "data/dma-kws/exp"))
    return exp_root / "stage2_adapt" / slug


def manifest_paths(data_root: Path) -> dict[str, Path]:
    manifest_dir = data_root / "manifests"
    return {
        "tts_train": manifest_dir / "tts_train.csv",
        "tts_eval": manifest_dir / "tts_eval.csv",
        "real_train": manifest_dir / "real_train.csv",
        "real_eval": manifest_dir / "real_eval.csv",
    }


def phase_manifest(data_root: Path, phase: str, *, split: str) -> Path:
    return data_root / "manifests" / f"{phase}_{split}.csv"


def clips_eval_manifest_from_adapt(eval_manifest: Path, keyword: str, output_path: Path) -> Path:
    from dma_kws.stage2.adapt_dataset import clips_eval_manifest_from_adapt as _fn

    return _fn(eval_manifest, keyword, output_path)


def fbank_path_for_wav(fbank_root: Path, wav_path: Path) -> Path:
    rel = wav_path.name.replace(".wav", ".npy")
    return fbank_root / rel


def wav_to_fbank_mirror(fbank_root: Path, wav_path: Path) -> Path:
    """Mirror raw tree under fbank_root, swapping extension to .npy."""
    parts = list(wav_path.parts)
    if "raw" in parts:
        idx = parts.index("raw")
        rel = Path(*parts[idx + 1 :]).with_suffix(".npy")
        return fbank_root / rel
    return fbank_root / wav_path.with_suffix(".npy").name
