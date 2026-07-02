from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from dma_kws.stage2.features import FeatureExtractor, compute_fbank, waveform_to_fbank


def _write_noise_list(path: Path) -> None:
    path.write_text("/tmp/noise-a.wav\n/tmp/noise-b.wav\n", encoding="utf-8")


def test_feature_extractor_init_with_tmp_noise_list(tmp_path):
    noise_list = tmp_path / "noise.list"
    _write_noise_list(noise_list)

    extractor = FeatureExtractor(
        augment=True,
        wav_dir=tmp_path / "wav",
        noise_list_path=noise_list,
    )

    assert extractor.wav_dir == str(tmp_path / "wav")
    assert len(extractor.noise_lists) == 2
    assert extractor.data_aug["speed_perturb"] is True
    assert extractor.data_aug["add_noise"] is True


def test_feature_extractor_init_without_augment_skips_noise_list(tmp_path):
    extractor = FeatureExtractor(
        augment=False,
        wav_dir=tmp_path / "wav",
    )

    assert extractor.noise_lists == []
    assert extractor.data_aug["speed_perturb"] is False
    assert extractor.data_aug["add_noise"] is False


def test_speed_perturb_preserves_waveform_shape(monkeypatch, tmp_path):
    noise_list = tmp_path / "noise.list"
    _write_noise_list(noise_list)
    extractor = FeatureExtractor(
        augment=True,
        wav_dir=tmp_path / "wav",
        noise_list_path=noise_list,
    )

    waveform = torch.randn(1, 16000)

    class _FakeSoxEffects:
        @staticmethod
        def apply_effects_tensor(wav, sample_rate, effects):
            return wav, sample_rate

    class _FakeTorchaudio:
        sox_effects = _FakeSoxEffects

    monkeypatch.setattr(
        "dma_kws.stage2.features._import_torchaudio",
        lambda: _FakeTorchaudio(),
    )
    monkeypatch.setattr("dma_kws.stage2.features.random.choice", lambda seq: seq[0])

    perturbed = extractor._apply_speed_perturbation(waveform, 16000, speeds=[0.9, 1.0, 1.1])

    assert perturbed.shape == waveform.shape

    monkeypatch.setattr("dma_kws.stage2.features.random.choice", lambda seq: 1.0)
    unchanged = extractor._apply_speed_perturbation(waveform, 16000, speeds=[0.9, 1.0, 1.1])

    assert unchanged.shape == waveform.shape
    assert torch.equal(unchanged, waveform)


def test_waveform_to_fbank_matches_compute_fbank():
    waveform = torch.randn(1, 16000)
    kwargs = {
        "sample_rate": 16000,
        "num_mel_bins": 80,
        "frame_length": 25,
        "frame_shift": 10,
        "dither": 0.0,
        "window_type": "povey",
    }

    direct = compute_fbank(
        {"wav": waveform, "sample_rate": 16000, "key": "test"},
        **{key: value for key, value in kwargs.items() if key != "sample_rate"},
    )["feat"]
    via_helper = waveform_to_fbank(waveform, **kwargs)

    assert torch.equal(direct, via_helper)


def test_waveform_to_fbank_accepts_1d_waveform():
    waveform = torch.randn(16000)
    feat = waveform_to_fbank(waveform, sample_rate=16000, dither=0.0)

    assert feat.ndim == 2
    assert feat.size(1) == 80


def test_waveform_to_fbank_differs_from_extract_fbank():
    from dma_kws.audio import extract_fbank

    waveform = torch.randn(1, 16000)
    legacy = extract_fbank(waveform, num_mel_bins=80, sample_rate=16000, dither=0.0)
    aligned = waveform_to_fbank(waveform, sample_rate=16000, dither=0.0)

    assert not torch.allclose(legacy, aligned)
    assert (legacy - aligned).abs().mean() > 0.1
