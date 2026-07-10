"""Shared Stage II fbank extraction backends."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


def _import_torchaudio():
    try:
        import torchaudio
    except ImportError as exc:
        raise ImportError(
            "Missing torchaudio. Install CUDA PyTorch/torchaudio on the training machine first."
        ) from exc
    return torchaudio


def _import_lhotse():
    try:
        from lhotse import Fbank, FbankConfig
    except ImportError as exc:
        raise ImportError(
            'Missing Lhotse. Install the Zipformer feature dependencies with '
            '`pip install -e ".[icefall]"`.'
        ) from exc
    return Fbank, FbankConfig


class FbankExtractor:
    """Extract fbank features using one configured backend and sample-rate policy."""

    def __init__(
        self,
        *,
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
    ) -> None:
        if backend not in {"torchaudio_kaldi", "lhotse_fbank"}:
            raise ValueError(f"Unsupported fbank backend: {backend}")
        if target_sample_rate is not None and target_sample_rate <= 0:
            raise ValueError("target_sample_rate must be positive or None")

        self.num_mel_bins = num_mel_bins
        self.frame_length = frame_length
        self.frame_shift = frame_shift
        self.dither = dither
        self.window_type = window_type
        self.backend = backend
        self.target_sample_rate = target_sample_rate
        self.snip_edges = snip_edges
        self.low_freq = low_freq
        self.high_freq = high_freq
        self._lhotse_types = _import_lhotse() if backend == "lhotse_fbank" else None
        self._lhotse_extractors: dict[tuple[int, str], Any] = {}

    def output_sample_rate(self, source_sample_rate: int) -> int:
        if source_sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        return self.target_sample_rate or source_sample_rate

    def prepare_waveform(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
    ) -> tuple[torch.Tensor, int]:
        """Return a mono waveform at the configured target sample rate."""
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        if waveform.dim() != 2 or waveform.size(0) != 1:
            raise ValueError(
                f"Expected mono waveform with shape (samples,) or (1, samples), got {tuple(waveform.shape)}"
            )

        output_sample_rate = self.output_sample_rate(sample_rate)
        if sample_rate != output_sample_rate:
            torchaudio = _import_torchaudio()
            waveform = torchaudio.functional.resample(
                waveform,
                sample_rate,
                output_sample_rate,
            )
        return waveform.contiguous(), output_sample_rate

    def extract(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        waveform, sample_rate = self.prepare_waveform(waveform, sample_rate)
        if self.backend == "torchaudio_kaldi":
            return self._extract_torchaudio(waveform, sample_rate)
        return self._extract_lhotse(waveform, sample_rate)

    def _extract_torchaudio(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
    ) -> torch.Tensor:
        torchaudio = _import_torchaudio()
        mat = torchaudio.compliance.kaldi.fbank(
            waveform * (1 << 15),
            num_mel_bins=self.num_mel_bins,
            frame_length=self.frame_length,
            frame_shift=self.frame_shift,
            dither=self.dither,
            energy_floor=0.0,
            sample_frequency=sample_rate,
            window_type=self.window_type,
            snip_edges=self.snip_edges,
            low_freq=self.low_freq,
            high_freq=self.high_freq,
        )
        return mat.to(dtype=torch.float32)

    def _extract_lhotse(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
    ) -> torch.Tensor:
        device = str(waveform.device)
        cache_key = (sample_rate, device)
        extractor = self._lhotse_extractors.get(cache_key)
        if extractor is None:
            if self._lhotse_types is None:
                raise RuntimeError("Lhotse backend was not initialized")
            Fbank, LhotseFbankConfig = self._lhotse_types
            extractor = Fbank(
                LhotseFbankConfig(
                    sampling_rate=sample_rate,
                    frame_length=self.frame_length / 1000.0,
                    frame_shift=self.frame_shift / 1000.0,
                    window_type=self.window_type,
                    dither=self.dither,
                    snip_edges=self.snip_edges,
                    low_freq=self.low_freq,
                    high_freq=self.high_freq,
                    num_mel_bins=self.num_mel_bins,
                    device=device,
                )
            )
            self._lhotse_extractors[cache_key] = extractor

        mat = extractor.extract(waveform.squeeze(0), sample_rate)
        if isinstance(mat, np.ndarray):
            mat = torch.from_numpy(mat)
        return mat.to(dtype=torch.float32)
