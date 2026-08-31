import ast
import random
from pathlib import Path

import pytest
import torch

from dma_kws.stage2.features import TrainingBackgroundSampler


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _audio_list(tmp_path: Path) -> tuple[Path, Path]:
    audio = tmp_path / "music.wav"
    audio.write_bytes(b"placeholder")
    audio_list = tmp_path / "background.list"
    audio_list.write_text("# comment\nmusic.wav\n", encoding="utf-8")
    return audio, audio_list


def test_background_sampler_decodes_only_requested_duration_and_extracts_fbank(
    tmp_path,
    monkeypatch,
):
    audio, audio_list = _audio_list(tmp_path)
    calls = {}

    def fake_load_audio(path, **kwargs):
        calls["path"] = Path(path)
        calls["load_kwargs"] = kwargs
        return torch.ones(2, 24_000), 16_000

    class _FakeFbankExtractor:
        def __init__(self, **kwargs):
            calls["fbank_kwargs"] = kwargs

        def extract(self, waveform, sample_rate):
            calls["waveform"] = waveform
            calls["sample_rate"] = sample_rate
            return torch.full((149, 40), 3.0)

    monkeypatch.setattr("dma_kws.stage2.features._load_audio", fake_load_audio)
    monkeypatch.setattr(
        "dma_kws.stage2.features.FbankExtractor",
        _FakeFbankExtractor,
    )
    sampler = TrainingBackgroundSampler(
        audio_list_path=audio_list,
        duration_seconds_min=1.5,
        duration_seconds_max=1.5,
        fbank_kwargs={"num_mel_bins": 40, "dither": 0.0},
    )
    rng = random.Random(4)

    feats = sampler.extract(rng=rng)

    assert feats.shape == (149, 40)
    assert calls["path"] == audio
    assert calls["load_kwargs"] == {
        "rng": rng,
        "target_samples": 1_500_000,
        "target_sample_rate": 1_000_000,
    }
    # Stereo input is converted to mono and kept at exactly 1.5 seconds.
    assert calls["waveform"].shape == (1, 24_000)
    assert calls["sample_rate"] == 16_000
    assert calls["fbank_kwargs"] == {"num_mel_bins": 40, "dither": 0.0}


def test_background_sampler_repeats_a_short_recording_to_target_duration(
    tmp_path,
    monkeypatch,
):
    _audio, audio_list = _audio_list(tmp_path)

    monkeypatch.setattr(
        "dma_kws.stage2.features._load_audio",
        lambda *_args, **_kwargs: (torch.tensor([[1.0, 2.0]]), 4),
    )

    class _FakeFbankExtractor:
        def __init__(self, **_kwargs):
            pass

        def extract(self, waveform, sample_rate):
            assert waveform.shape == (1, 6)
            assert sample_rate == 4
            return waveform.transpose(0, 1)

    monkeypatch.setattr(
        "dma_kws.stage2.features.FbankExtractor",
        _FakeFbankExtractor,
    )
    sampler = TrainingBackgroundSampler(
        audio_list_path=audio_list,
        duration_seconds_min=1.5,
        duration_seconds_max=1.5,
    )

    feats = sampler.extract(rng=random.Random(2))

    assert feats.shape == (6, 1)


@pytest.mark.parametrize(
    ("minimum", "maximum", "message"),
    [
        (0.0, 1.0, "must be positive"),
        (2.0, 1.0, "must be <="),
        (float("nan"), 1.0, "must be finite"),
        (1.0, float("inf"), "must be finite"),
    ],
)
def test_background_sampler_rejects_invalid_duration_before_source_io(
    minimum,
    maximum,
    message,
):
    with pytest.raises(ValueError, match=message):
        TrainingBackgroundSampler(
            audio_list_path="/missing/background.list",
            duration_seconds_min=minimum,
            duration_seconds_max=maximum,
        )


def test_lora_replay_forwards_background_noise_and_fbank_configuration():
    """LoRA replay must keep the base-training background contract."""

    tree = ast.parse(
        (PROJECT_ROOT / "dma_kws/stage2/adapt.py").read_text(encoding="utf-8")
    )
    adaptation = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "run_stage2_adaptation"
    )
    constructor = next(
        node
        for node in ast.walk(adaptation)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "LibriPhraseTrainDataset"
    )
    keywords = {item.arg: ast.unparse(item.value) for item in constructor.keywords}

    assert keywords["noise_augmentation"] == "noise_augmentation"
    assert keywords["background_negative"] == "background_negative"
    assert keywords["fbank_kwargs"] == "fbank_kwargs(get_fbank_config(config))"
