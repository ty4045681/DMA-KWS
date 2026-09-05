"""Shared Stage II train dataset / DataLoader construction.

Official training and the input benchmark must call these builders so A/B
comparisons use the same rank seed, collate, shuffle, and worker init.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from dma_kws.config import fbank_kwargs, get_fbank_config, require_sections
from dma_kws.stage2.collate import train_collate_fn
from dma_kws.stage2.dataset import LibriPhraseTrainDataset, stage2_worker_init_fn
from dma_kws.stage2.objective import resolve_sequence_objective
from dma_kws.training.ddp import process_rank
from dma_kws.training.loaders import build_loader_kwargs


def resolve_stage2_data_path(stage2: dict[str, Any], key: str, default: Path) -> Path:
    """Resolve an optional Stage II path, falling back to ``default``."""
    raw = stage2.get(key, "")
    if raw:
        return Path(raw)
    return default


def build_stage2_train_dataset(config: dict[str, Any], tokenizer: Any):
    """Construct the official Stage II ``LibriPhraseTrainDataset``."""
    require_sections(config, ["paths", "stage1", "stage2", "training"])
    paths = config["paths"]
    stage2 = config["stage2"]
    training = config["training"]

    processed_root = Path(paths["processed_root"])
    feature_root = Path(paths.get("feature_root", processed_root))
    parquet_file = resolve_stage2_data_path(
        stage2,
        "parquet_file",
        processed_root / "stage2_qbyt" / "aggregated_segments_with_g2p_distance.parquet",
    )
    wav_dir = resolve_stage2_data_path(stage2, "wav_dir", feature_root / "fbank")
    if not parquet_file.exists():
        raise SystemExit(f"Stage II parquet not found: {parquet_file}")

    seed = int(training.get("seed", 2025))
    dataset_seed = seed + 1_000_003 * process_rank()
    sequence_objective = resolve_sequence_objective(stage2)
    noise_augmentation = stage2.get("noise_augmentation", {}) or {}
    if not isinstance(noise_augmentation, dict):
        raise ValueError("stage2.noise_augmentation must be a mapping")
    background_negative = stage2.get("background_negative", {}) or {}
    if not isinstance(background_negative, dict):
        raise ValueError("stage2.background_negative must be a mapping")

    return LibriPhraseTrainDataset(
        parquet_file=parquet_file,
        wav_dir=wav_dir,
        tokenizer=tokenizer,
        negative_ratio=int(stage2.get("negative_ratio", 1)),
        hard_negative_ratio=int(stage2.get("hard_negative_ratio", 1)),
        sample_lens=int(stage2.get("sample_lens", 5000)),
        seed=dataset_seed,
        noise_augmentation=noise_augmentation,
        background_negative=background_negative,
        fbank_kwargs=fbank_kwargs(get_fbank_config(config)),
        seq_label_mode=sequence_objective.target_mode,
        metadata_cache=stage2.get("metadata_cache", {}) or {},
    )


def build_stage2_train_dataloader(config: dict[str, Any], dataset: Any):
    """Construct the official Stage II training DataLoader."""
    from torch.utils.data import DataLoader

    stage1 = config["stage1"]
    stage2 = config["stage2"]
    batch_size = int(stage2.get("batch_size_per_gpu", 64))
    num_workers = int(stage2.get("num_workers", stage1.get("num_workers", 2)))
    loader_kwargs = build_loader_kwargs(num_workers, stage2.get("dataloader", {}) or {})
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=train_collate_fn,
        drop_last=True,
        # Without this the dataset's RNG is forked in an identical state into every
        # worker, so all of them draw the same negatives and clips.
        worker_init_fn=stage2_worker_init_fn,
        **loader_kwargs,
    )
