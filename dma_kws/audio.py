"""Audio loading and fbank feature extraction shared across DMA-KWS scripts.

Consolidates the duplicated torchaudio.load + downmix + resample logic and the
``torchaudio.compliance.kaldi.fbank`` calls in train_stage1_ctc.py and
run_two_stage_demo.py. torch/torchaudio are imported lazily inside the
functions, matching the scripts' "friendly SystemExit on ImportError" pattern.
"""

from __future__ import annotations


def _load_audio_soundfile(path):
    """Load a file with soundfile when torchaudio needs an optional decoder."""

    try:
        import numpy as np
        import soundfile as sf
        import torch
    except ImportError as exc:
        raise SystemExit(
            "Missing soundfile. Install it with: pip install soundfile"
        ) from exc

    data, sr = sf.read(str(path), always_2d=True, dtype="float32")
    waveform = torch.from_numpy(np.ascontiguousarray(data.T))
    return waveform, int(sr)


def load_audio(path, *, sample_rate: int):
    """Load ``path`` and return ``(waveform, sample_rate)`` as mono at ``sample_rate``.

    Multi-channel audio is downmixed to mono and resampled to ``sample_rate``
    when the source rate differs.
    """
    try:
        import torchaudio
    except ImportError as exc:
        raise SystemExit(
            "Missing torch/torchaudio. Install CUDA PyTorch on the remote training machine first."
        ) from exc

    try:
        waveform, sr = torchaudio.load(path)
    except ImportError:
        # torchaudio 2.10+ may route load() through torchcodec, which is not
        # part of the core inference extras. Soundfile can still read WAV.
        waveform, sr = _load_audio_soundfile(path)
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != sample_rate:
        waveform = torchaudio.transforms.Resample(sr, sample_rate)(waveform)
    return waveform, sample_rate


def extract_fbank(waveform, *, num_mel_bins: int, sample_rate: int, dither: float):
    """Compute Kaldi-compatible fbank features for ``waveform``.

    Uses ``frame_length=25`` and ``frame_shift=10`` to match Stage I/II.
    """
    try:
        import torchaudio.compliance.kaldi as kaldi
    except ImportError as exc:
        raise SystemExit(
            "Missing torch/torchaudio. Install CUDA PyTorch on the remote training machine first."
        ) from exc

    return kaldi.fbank(
        waveform,
        num_mel_bins=num_mel_bins,
        frame_length=25,
        frame_shift=10,
        dither=dither,
        sample_frequency=sample_rate,
    )
