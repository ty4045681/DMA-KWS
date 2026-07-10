"""Stage II online fbank extraction with optional speed/noise augmentation."""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from dma_kws.stage2.fbank import FbankExtractor

DEFAULT_NUM_MEL_BINS = 80
DEFAULT_DITHER = 0.1
DEFAULT_FRAME_LENGTH = 25
DEFAULT_FRAME_SHIFT = 10


def _import_torchaudio():
    try:
        import torchaudio
    except ImportError as exc:
        raise ImportError(
            "Missing torchaudio. Install CUDA PyTorch/torchaudio on the training machine first."
        ) from exc
    return torchaudio


def compute_fbank(
    sample: dict[str, Any],
    *,
    num_mel_bins: int = DEFAULT_NUM_MEL_BINS,
    frame_length: int = DEFAULT_FRAME_LENGTH,
    frame_shift: int = DEFAULT_FRAME_SHIFT,
    dither: float = DEFAULT_DITHER,
    window_type: str = "povey",
    backend: str = "torchaudio_kaldi",
    target_sample_rate: int | None = None,
    snip_edges: bool = True,
    low_freq: float = 20.0,
    high_freq: float = 0.0,
    extractor: FbankExtractor | None = None,
) -> dict[str, Any]:
    """Compute fbank features with the configured Stage II backend."""
    sample_rate = sample["sample_rate"]
    waveform = sample["wav"]
    if extractor is None:
        extractor = FbankExtractor(
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
        )
    sample["feat"] = extractor.extract(waveform, sample_rate)
    return sample


def waveform_to_fbank(
    waveform: torch.Tensor,
    *,
    sample_rate: int,
    num_mel_bins: int = DEFAULT_NUM_MEL_BINS,
    frame_length: int = DEFAULT_FRAME_LENGTH,
    frame_shift: int = DEFAULT_FRAME_SHIFT,
    dither: float = DEFAULT_DITHER,
    window_type: str = "povey",
    backend: str = "torchaudio_kaldi",
    target_sample_rate: int | None = None,
    snip_edges: bool = True,
    low_freq: float = 20.0,
    high_freq: float = 0.0,
    extractor: FbankExtractor | None = None,
) -> torch.Tensor:
    """Compute configured fbank features from a mono waveform tensor."""
    sample = compute_fbank(
        {"wav": waveform, "sample_rate": sample_rate, "key": ""},
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
    return sample["feat"]


class FeatureExtractor:
    """Load raw wav, optionally augment, and compute 80-dim fbank features."""

    def __init__(
        self,
        *,
        augment: bool = True,
        wav_dir: str | Path,
        noise_list_path: str | Path | None = None,
    ) -> None:
        self.data_aug = {
            "speed_perturb": augment,
            "add_noise": augment,
        }
        self.wav_dir = str(wav_dir)
        if augment:
            if noise_list_path is None:
                raise ValueError("noise_list_path is required when augment=True")
            self.noise_lists = self._load_noise_files(noise_list_path)
        else:
            self.noise_lists = []

    def _load_noise_files(self, noise_file_path: str | Path) -> list[str]:
        path = Path(noise_file_path)
        if not path.is_file():
            raise FileNotFoundError(f"Noise list file not found: {path}")
        with path.open("r", encoding="utf-8") as handle:
            return handle.readlines()

    def _apply_speed_perturbation(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
        speeds: list[float] | tuple[float, ...] = (0.9, 1.0, 1.1),
    ) -> torch.Tensor:
        speed = random.choice(speeds)
        if speed != 1.0:
            torchaudio = _import_torchaudio()
            waveform, _ = torchaudio.sox_effects.apply_effects_tensor(
                waveform,
                sample_rate,
                [["speed", str(speed)], ["rate", str(sample_rate)]],
            )
        return waveform

    def _add_noise_with_snr(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
        target_snr_db: float,
    ) -> torch.Tensor:
        torchaudio = _import_torchaudio()
        noise_path = random.choice(self.noise_lists).strip()
        noise_wav, noise_sr = torchaudio.load(noise_path)

        if noise_sr != sample_rate:
            noise_wav = torchaudio.transforms.Resample(noise_sr, sample_rate)(noise_wav)

        if noise_wav.shape[1] < waveform.shape[1]:
            repeat_times = (waveform.shape[1] // noise_wav.shape[1]) + 1
            noise_wav = noise_wav.repeat(1, repeat_times)[:, : waveform.shape[1]]
        else:
            noise_wav = noise_wav[:, : waveform.shape[1]]

        signal_power = torch.mean(waveform**2)
        noise_power = torch.mean(noise_wav**2)
        snr_linear = 10 ** (target_snr_db / 10)
        scaling_factor = torch.sqrt(signal_power / (snr_linear * noise_power))
        noise_wav *= scaling_factor
        return waveform + noise_wav

    def load_npy_fbank(self, query_wav: str) -> torch.Tensor:
        """Load precomputed fbank from the main LibriPhrase npy layout."""
        from dma_kws.stage2.dataset import _resolve_fbank_path

        fbank_path = _resolve_fbank_path(self.wav_dir, query_wav)
        return torch.from_numpy(np.load(fbank_path))

    def process(self, wav_path: str) -> dict[str, Any]:
        """Load raw wav, optionally augment, and return a sample dict with ``feat``."""
        torchaudio = _import_torchaudio()
        full_path = os.path.join(self.wav_dir, wav_path)
        waveform, sr = torchaudio.load(full_path)

        if self.data_aug["speed_perturb"]:
            waveform = self._apply_speed_perturbation(waveform, sr)

        if self.data_aug["add_noise"] and random.random() < 0.5:
            snr = random.choice(range(5, 20))
            waveform = self._add_noise_with_snr(waveform, sr, snr)

        return compute_fbank(
            {
                "key": full_path,
                "wav": waveform,
                "sample_rate": sr,
            }
        )
