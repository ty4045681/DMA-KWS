from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from dma_kws.stage2.features import FeatureExtractor


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
