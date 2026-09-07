"""Path helpers for keyword continual adaptation."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any


DEFAULT_ADAPT_TRAIN_PHASES = ("tts", "real")


def resolve_adapt_train_phases(adapt: dict[str, Any]) -> tuple[str, ...]:
    """Return the validated ordered phases for orchestration and sweeps."""

    raw = adapt.get("train_phases", DEFAULT_ADAPT_TRAIN_PHASES)
    if isinstance(raw, str):
        phases = [part.strip().casefold() for part in raw.split(",") if part.strip()]
    elif isinstance(raw, (list, tuple)):
        phases = [str(part).strip().casefold() for part in raw if str(part).strip()]
    else:
        raise ValueError("adapt.train_phases must be a list or comma-separated string")
    if not phases:
        raise ValueError("adapt.train_phases must contain at least one phase")
    invalid = sorted(set(phases) - {"tts", "real", "joint"})
    if invalid:
        raise ValueError(
            f"adapt.train_phases contains unsupported phases {invalid}; expected tts/real or joint"
        )
    if len(phases) != len(set(phases)):
        raise ValueError(f"adapt.train_phases contains duplicates: {phases}")
    if "joint" in phases and len(phases) != 1:
        raise ValueError("adapt.train_phases=[joint] cannot be combined with tts/real")
    if str(adapt.get("phase", "")).strip().casefold() == "joint" and phases != ["joint"]:
        raise ValueError("adapt.phase=joint requires adapt.train_phases=[joint]")
    return tuple(phases)


def slugify(keyword: str) -> str:
    """Convert keyword text to a filesystem slug, e.g. 'hey eva' -> 'hey_eva'."""
    text = keyword.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_") or "keyword"


def directory_name_to_text(name: str) -> str:
    """Normalize a directory name into a phrase for G2P."""
    text = re.sub(r"[^a-z0-9]+", " ", name.casefold()).strip()
    if not text:
        raise ValueError(f"Directory name does not contain a usable phrase: {name!r}")
    return text


def neg_slug_to_text(slug: str) -> str:
    """Convert negative folder slug back to phrase text."""
    return directory_name_to_text(slug)


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
    slug = str(adapt.get("slug", "")) or slugify(keyword)
    if adapt.get("exp_root"):
        return Path(str(adapt["exp_root"]))
    paths = config.get("paths", {})
    exp_root = Path(paths.get("exp_root", "data/dma-kws/exp"))
    joint = resolve_adapt_train_phases(adapt) == ("joint",)
    return exp_root / ("stage2_adapt_joint" if joint else "stage2_adapt") / slug


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


def target_eval_manifest(data_root: Path, phase: str) -> Path:
    """Use held-out real recordings for the joint run's primary target metric."""
    return phase_manifest(data_root, "real" if phase == "joint" else phase, split="eval")


def clips_eval_manifest_from_adapt(
    eval_manifest: Path,
    keyword: str,
    output_path: Path,
    *,
    manifest_root: Path | None = None,
) -> Path:
    from dma_kws.stage2.adapt_dataset import clips_eval_manifest_from_adapt as _fn

    return _fn(eval_manifest, keyword, output_path, manifest_root=manifest_root)


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
    if wav_path.is_absolute():
        rel = Path(*wav_path.parts[1:]).with_suffix(".npy")
        return fbank_root / "external" / rel
    return fbank_root / wav_path.with_suffix(".npy").name
