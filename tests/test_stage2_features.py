from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from dma_kws.stage2.features import FeatureExtractor, compute_fbank, waveform_to_fbank
from dma_kws.stage2.fbank import FbankExtractor


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


def test_torchaudio_backend_preserves_wenet_fbank_formula():
    import torchaudio.compliance.kaldi as kaldi

    waveform = torch.linspace(-0.5, 0.5, 16000).unsqueeze(0)
    expected = kaldi.fbank(
        waveform * (1 << 15),
        num_mel_bins=80,
        frame_length=25,
        frame_shift=10,
        dither=0.0,
        energy_floor=0.0,
        sample_frequency=16000,
        window_type="povey",
        snip_edges=True,
        low_freq=20.0,
        high_freq=0.0,
    )

    actual = waveform_to_fbank(waveform, sample_rate=16000, dither=0.0)

    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_lhotse_fbank_backend_matches_direct_extractor():
    lhotse = pytest.importorskip("lhotse")
    waveform = torch.linspace(-0.5, 0.5, 16000).unsqueeze(0)
    config = lhotse.FbankConfig(
        sampling_rate=16000,
        frame_length=0.025,
        frame_shift=0.01,
        window_type="povey",
        dither=0.0,
        snip_edges=False,
        low_freq=20.0,
        high_freq=-400.0,
        num_mel_bins=80,
        device="cpu",
    )

    expected = lhotse.Fbank(config).extract(waveform.squeeze(0), 16000)
    actual = waveform_to_fbank(
        waveform,
        sample_rate=16000,
        backend="lhotse_fbank",
        target_sample_rate=16000,
        dither=0.0,
        snip_edges=False,
        low_freq=20.0,
        high_freq=-400.0,
    )

    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    assert actual.shape == (100, 80)
    assert actual.dtype == torch.float32


@pytest.mark.parametrize("sample_rate", [8000, 22050])
def test_lhotse_fbank_backend_resamples_to_target_rate(sample_rate):
    pytest.importorskip("lhotse")
    waveform = torch.linspace(-0.5, 0.5, sample_rate).unsqueeze(0)

    feat = waveform_to_fbank(
        waveform,
        sample_rate=sample_rate,
        backend="lhotse_fbank",
        target_sample_rate=16000,
        dither=0.0,
        snip_edges=False,
        high_freq=-400.0,
    )

    assert feat.shape == (100, 80)


def test_fbank_extractor_rejects_unknown_backend():
    with pytest.raises(ValueError, match="Unsupported fbank backend"):
        FbankExtractor(backend="unknown")


def test_compute_fbank_for_clip_loads_and_resamples_wav(tmp_path):
    lhotse = pytest.importorskip("lhotse")
    soundfile = pytest.importorskip("soundfile")
    del lhotse
    from dma_kws.stage2.prepare_paper import compute_fbank_for_clip

    sample_rate = 22050
    waveform = torch.linspace(-0.5, 0.5, sample_rate).numpy()
    wav_path = tmp_path / "sample.wav"
    output_path = tmp_path / "sample.npy"
    soundfile.write(wav_path, waveform, sample_rate, subtype="FLOAT")

    compute_fbank_for_clip(
        wav_path,
        output_path,
        backend="lhotse_fbank",
        target_sample_rate=16000,
        dither=0.0,
        snip_edges=False,
        high_freq=-400.0,
    )

    feat = torch.from_numpy(np.load(output_path))
    assert feat.shape == (100, 80)
    assert feat.dtype == torch.float32


def test_waveform_to_fbank_differs_from_extract_fbank():
    from dma_kws.audio import extract_fbank

    waveform = torch.randn(1, 16000)
    legacy = extract_fbank(waveform, num_mel_bins=80, sample_rate=16000, dither=0.0)
    aligned = waveform_to_fbank(waveform, sample_rate=16000, dither=0.0)

    assert not torch.allclose(legacy, aligned)
    assert (legacy - aligned).abs().mean() > 0.1


def test_file_fbank_slice_matches_independent_window_when_snip_edges():
    from dma_kws.inference.audio_utils import window_fbank_frame_span

    sample_rate = 16000
    waveform = torch.linspace(-0.4, 0.4, sample_rate * 6).unsqueeze(0)
    extractor = FbankExtractor(dither=0.0, snip_edges=True)
    file_feat = waveform_to_fbank(
        waveform,
        sample_rate=sample_rate,
        extractor=extractor,
        dither=0.0,
        snip_edges=True,
    )
    start_sample = sample_rate
    end_sample = start_sample + 3 * sample_rate
    start_frame, end_frame = window_fbank_frame_span(
        start_sample,
        end_sample,
        sample_rate=sample_rate,
        snip_edges=True,
    )
    sliced = file_feat[start_frame:end_frame]
    independent = waveform_to_fbank(
        waveform[:, start_sample:end_sample],
        sample_rate=sample_rate,
        extractor=extractor,
        dither=0.0,
        snip_edges=True,
    )
    assert sliced.shape == independent.shape
    assert torch.allclose(sliced, independent, atol=1e-5, rtol=1e-5)


def test_file_fbank_slice_keeps_icefall_window_width():
    from dma_kws.inference.audio_utils import window_fbank_frame_span

    sample_rate = 16000
    waveform = torch.linspace(-0.4, 0.4, sample_rate * 6).unsqueeze(0)
    extractor = FbankExtractor(dither=0.0, snip_edges=False, high_freq=-400.0)
    file_feat = waveform_to_fbank(
        waveform,
        sample_rate=sample_rate,
        extractor=extractor,
        dither=0.0,
        snip_edges=False,
        high_freq=-400.0,
    )
    start_sample = sample_rate
    end_sample = start_sample + 3 * sample_rate
    start_frame, end_frame = window_fbank_frame_span(
        start_sample,
        end_sample,
        sample_rate=sample_rate,
        snip_edges=False,
    )
    independent = waveform_to_fbank(
        waveform[:, start_sample:end_sample],
        sample_rate=sample_rate,
        extractor=extractor,
        dither=0.0,
        snip_edges=False,
        high_freq=-400.0,
    )
    assert file_feat[start_frame:end_frame].shape == independent.shape
