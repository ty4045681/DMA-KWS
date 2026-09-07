"""Fixed held-out background crops for LoRA validation (clip FPR, not FA/h)."""

from __future__ import annotations

import random

import torch
from torch.utils.data import Dataset

from dma_kws.stage2.features import TrainingBackgroundSampler


class BackgroundValidationDataset(Dataset):
    def __init__(
        self, *, audio_list_path, anchor_seq, num_samples=512, seed=2026,
        duration_seconds_min=1.0, duration_seconds_max=3.0, fbank_kwargs=None,
    ):
        if isinstance(num_samples, bool) or int(num_samples) != num_samples or num_samples < 1:
            raise ValueError("adapt.joint.background_val_samples must be a positive integer")
        self.num_samples = int(num_samples)
        self.seed = int(seed)
        self.anchor_seq = torch.as_tensor(anchor_seq, dtype=torch.long).clone()
        if not self.anchor_seq.numel():
            raise ValueError("Background validation requires a nonempty keyword")
        self.background_sampler = TrainingBackgroundSampler(
            audio_list_path=audio_list_path,
            duration_seconds_min=duration_seconds_min,
            duration_seconds_max=duration_seconds_max,
            fbank_kwargs=fbank_kwargs,
        )

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        seed = self.seed + int(index)
        # Audio crops and feature dither must both stay fixed across validations
        # and worker counts, without perturbing the caller's RNG stream.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            feat = self.background_sampler.extract(rng=random.Random(seed))
        return {
            "sample_id": torch.tensor(index, dtype=torch.long),
            "anchor_seq": self.anchor_seq.clone(),
            "feat": feat,
            "label": torch.tensor(0, dtype=torch.long),
        }
