import ast
import math
import random
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

from dma_kws.stage2.features import TrainingBackgroundSampler


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _write_wav(path: Path, array: np.ndarray, sample_rate: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(
        str(path),
        np.asarray(array, dtype=np.float32),
        sample_rate,
        subtype="FLOAT",
    )
    return path


def _write_audio_list(tmp_path: Path, filenames: list[str]) -> Path:
    audio_list = tmp_path / "background.list"
    audio_list.write_text("\n".join(filenames) + "\n", encoding="utf-8")
    return audio_list


def _oracle_load_audio(
    path: str | Path,
    *,
    rng: random.Random,
    target_samples: int,
    target_sample_rate: int,
) -> tuple[torch.Tensor, int]:
    """Independent copy of pre-refactor features._load_audio (partial-read mode)."""
    with sf.SoundFile(str(path)) as handle:
        sample_rate = int(handle.samplerate)
        frames = -1
        if target_samples <= 0 or target_sample_rate <= 0:
            raise ValueError("Target noise duration and sample rate must be positive")
        frames = max(
            1,
            math.ceil(target_samples * sample_rate / target_sample_rate),
        )
        if len(handle) > frames:
            handle.seek(rng.randint(0, len(handle) - frames))
        else:
            frames = -1
        array = handle.read(
            frames=frames,
            dtype="float32",
            always_2d=True,
        )
    waveform = torch.from_numpy(np.asarray(array, dtype=np.float32).T.copy())
    return waveform, int(sample_rate)


def _oracle_as_mono(waveform: torch.Tensor, *, source: Path) -> torch.Tensor:
    """Independent copy of pre-refactor TrainingNoiseAugmenter._as_mono."""
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    if waveform.dim() != 2 or waveform.size(0) == 0:
        raise ValueError(
            f"Expected audio shaped (channels, samples) for {source}, "
            f"got {tuple(waveform.shape)}"
        )
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if waveform.size(1) == 0:
        raise ValueError(f"Audio contains no samples: {source}")
    return waveform.to(dtype=torch.float32).contiguous()


def _oracle_match_noise_length(
    noise: torch.Tensor,
    target_samples: int,
    *,
    rng: random.Random,
) -> torch.Tensor:
    """Independent copy of pre-refactor TrainingNoiseAugmenter._match_noise_length."""
    noise_samples = int(noise.size(1))
    if noise_samples >= target_samples:
        max_offset = noise_samples - target_samples
        offset = rng.randint(0, max_offset) if max_offset else 0
        return noise[:, offset : offset + target_samples]

    offset = rng.randrange(noise_samples) if noise_samples > 1 else 0
    repeats = math.ceil((target_samples + offset) / noise_samples)
    tiled = noise.repeat(1, repeats)
    return tiled[:, offset : offset + target_samples]


def _oracle_online_crop(
    audio_paths: list[Path] | tuple[Path, ...],
    duration_seconds_min: float,
    duration_seconds_max: float,
    rng: random.Random,
) -> tuple[torch.Tensor, int]:
    """Independent copy of pre-refactor TrainingBackgroundSampler.extract crop path."""
    duration_seconds = rng.uniform(duration_seconds_min, duration_seconds_max)
    source_path = rng.choice(audio_paths)
    duration_units = max(1, round(duration_seconds * 1_000_000))
    waveform, sample_rate = _oracle_load_audio(
        source_path,
        rng=rng,
        target_samples=duration_units,
        target_sample_rate=1_000_000,
    )
    waveform = _oracle_as_mono(waveform, source=source_path)
    target_samples = max(1, round(duration_seconds * sample_rate))
    waveform = _oracle_match_noise_length(waveform, target_samples, rng=rng)
    return waveform, sample_rate


class _CapturingFbankExtractor:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.waveform: torch.Tensor | None = None
        self.sample_rate: int | None = None

    def extract(self, waveform, sample_rate):
        self.waveform = waveform.detach().clone()
        self.sample_rate = sample_rate
        return waveform.transpose(0, 1)


def _extract_captured(monkeypatch, sampler, rng):
    captured = {}

    class _Fake(_CapturingFbankExtractor):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            captured["extractor"] = self

        def extract(self, waveform, sample_rate):
            result = super().extract(waveform, sample_rate)
            captured["waveform"] = self.waveform
            captured["sample_rate"] = self.sample_rate
            captured["fbank_kwargs"] = self.kwargs
            return result

    monkeypatch.setattr("dma_kws.stage2.features.FbankExtractor", _Fake)
    feats = sampler.extract(rng=rng)
    return feats, captured


def test_background_sampler_decodes_only_requested_duration_and_extracts_fbank(
    tmp_path,
    monkeypatch,
):
    from dma_kws.stage2.background_sampling import draw_crop_spec, probe_source_info

    stereo = np.stack(
        [
            np.full(48_000, 0.25, dtype=np.float32),
            np.full(48_000, 0.75, dtype=np.float32),
        ],
        axis=1,
    )
    audio = _write_wav(tmp_path / "music.wav", stereo, 16_000)
    audio_list = _write_audio_list(tmp_path, ["music.wav"])
    calls = {}

    class _FakeFbankExtractor:
        def __init__(self, **kwargs):
            calls["fbank_kwargs"] = kwargs

        def extract(self, waveform, sample_rate):
            calls["waveform"] = waveform
            calls["sample_rate"] = sample_rate
            return torch.full((149, 40), 3.0)

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

    spec_rng = random.Random(4)
    spec_rng.uniform(1.5, 1.5)
    chosen = spec_rng.choice(sampler.audio_paths)
    spec = draw_crop_spec(probe_source_info(chosen), 1.5, rng=spec_rng)
    duration_units = max(1, round(1.5 * 1_000_000))

    assert feats.shape == (149, 40)
    assert chosen == audio
    assert duration_units == 1_500_000
    assert spec.read_num_frames == max(
        1, math.ceil(duration_units * 16_000 / 1_000_000)
    )
    assert spec.target_num_samples == 24_000
    # Stereo input is converted to mono and kept at exactly 1.5 seconds.
    assert calls["waveform"].shape == (1, 24_000)
    assert calls["sample_rate"] == 16_000
    assert torch.equal(calls["waveform"], torch.full((1, 24_000), 0.5))
    assert calls["fbank_kwargs"] == {"num_mel_bins": 40, "dither": 0.0}


def test_background_sampler_repeats_a_short_recording_to_target_duration(
    tmp_path,
    monkeypatch,
):
    _write_wav(tmp_path / "music.wav", np.array([1.0, 2.0], dtype=np.float32), 4)
    audio_list = _write_audio_list(tmp_path, ["music.wav"])

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
    assert keywords["metadata_cache"] == "stage2.get('metadata_cache', {}) or {}"


def test_online_extract_matches_prerefactor_oracle_long_stereo(tmp_path, monkeypatch):
    sample_rate = 16_000
    num_frames = 48_000
    left = np.linspace(-0.8, 0.9, num_frames, dtype=np.float32)
    right = np.linspace(0.4, -0.6, num_frames, dtype=np.float32)
    stereo = np.stack([left, right], axis=1)
    _write_wav(tmp_path / "music.wav", stereo, sample_rate)
    audio_list = _write_audio_list(tmp_path, ["music.wav"])
    sampler = TrainingBackgroundSampler(
        audio_list_path=audio_list,
        duration_seconds_min=1.5,
        duration_seconds_max=1.5,
        fbank_kwargs={"num_mel_bins": 40, "dither": 0.0},
    )
    rng = random.Random(4)
    feats, captured = _extract_captured(monkeypatch, sampler, rng)

    oracle_waveform, oracle_sample_rate = _oracle_online_crop(
        sampler.audio_paths,
        1.5,
        1.5,
        random.Random(4),
    )
    assert captured["waveform"].shape == (1, 24_000)
    assert captured["sample_rate"] == sample_rate
    assert oracle_sample_rate == sample_rate
    assert torch.equal(captured["waveform"], oracle_waveform)
    assert feats.shape == (24_000, 1)
    assert captured["fbank_kwargs"] == {"num_mel_bins": 40, "dither": 0.0}


def test_online_extract_matches_prerefactor_oracle_short_repeat(tmp_path, monkeypatch):
    _write_wav(tmp_path / "music.wav", np.array([1.0, 2.0], dtype=np.float32), 4)
    audio_list = _write_audio_list(tmp_path, ["music.wav"])
    sampler = TrainingBackgroundSampler(
        audio_list_path=audio_list,
        duration_seconds_min=1.5,
        duration_seconds_max=1.5,
    )
    rng = random.Random(2)
    _feats, captured = _extract_captured(monkeypatch, sampler, rng)

    oracle_waveform, oracle_sample_rate = _oracle_online_crop(
        sampler.audio_paths,
        1.5,
        1.5,
        random.Random(2),
    )
    assert captured["waveform"].shape == (1, 6)
    assert captured["sample_rate"] == 4
    assert oracle_sample_rate == 4
    assert torch.equal(captured["waveform"], oracle_waveform)


def test_online_extract_preserves_rng_stream(tmp_path, monkeypatch):
    long_left = np.linspace(-1.0, 1.0, 32_000, dtype=np.float32)
    long_right = np.linspace(0.5, -0.25, 32_000, dtype=np.float32)
    _write_wav(
        tmp_path / "long.wav",
        np.stack([long_left, long_right], axis=1),
        16_000,
    )
    _write_wav(
        tmp_path / "short.wav",
        np.array([0.25, -0.5, 0.75], dtype=np.float32),
        8_000,
    )
    audio_list = _write_audio_list(tmp_path, ["long.wav", "short.wav"])
    sampler = TrainingBackgroundSampler(
        audio_list_path=audio_list,
        duration_seconds_min=1.0,
        duration_seconds_max=2.0,
        fbank_kwargs={"dither": 0.0},
    )
    extract_rng = random.Random(11)
    _feats, captured = _extract_captured(monkeypatch, sampler, extract_rng)
    extract_next = extract_rng.random()

    oracle_rng = random.Random(11)
    oracle_waveform, oracle_sample_rate = _oracle_online_crop(
        sampler.audio_paths,
        1.0,
        2.0,
        oracle_rng,
    )
    oracle_next = oracle_rng.random()

    assert torch.equal(captured["waveform"], oracle_waveform)
    assert captured["sample_rate"] == oracle_sample_rate
    assert extract_next == oracle_next


def test_online_extract_matches_prerefactor_oracle_8khz_noninteger_ms(
    tmp_path, monkeypatch
):
    duration_seconds = 1.234567
    sample_rate = 8_000
    num_frames = 24_000
    samples = np.linspace(-0.7, 0.65, num_frames, dtype=np.float32)
    _write_wav(tmp_path / "noise.wav", samples, sample_rate)
    audio_list = _write_audio_list(tmp_path, ["noise.wav"])
    sampler = TrainingBackgroundSampler(
        audio_list_path=audio_list,
        duration_seconds_min=duration_seconds,
        duration_seconds_max=duration_seconds,
    )
    rng = random.Random(21)
    _feats, captured = _extract_captured(monkeypatch, sampler, rng)
    oracle_waveform, oracle_sample_rate = _oracle_online_crop(
        sampler.audio_paths,
        duration_seconds,
        duration_seconds,
        random.Random(21),
    )
    expected_target = max(1, round(duration_seconds * sample_rate))
    assert captured["waveform"].shape == (1, expected_target)
    assert captured["sample_rate"] == 8_000
    assert oracle_sample_rate == 8_000
    assert torch.equal(captured["waveform"], oracle_waveform)
    extract_rng = random.Random(21)
    sampler.extract(rng=extract_rng)
    oracle_rng = random.Random(21)
    _oracle_online_crop(
        sampler.audio_paths, duration_seconds, duration_seconds, oracle_rng
    )
    assert extract_rng.random() == oracle_rng.random()


def test_draw_crop_spec_single_source_does_not_consume_choice_rng(tmp_path):
    from dma_kws.stage2.background_sampling import (
        draw_crop_spec,
        materialize_crop,
        probe_source_info,
    )

    primary = _write_wav(
        tmp_path / "primary.wav",
        np.linspace(-0.4, 0.4, 48_000, dtype=np.float32),
        16_000,
    )
    other = _write_wav(
        tmp_path / "other.wav",
        np.linspace(0.9, -0.9, 8_000, dtype=np.float32),
        16_000,
    )
    source = probe_source_info(primary)
    duration_seconds = 1.234567

    class _ChoiceMustNotRun(random.Random):
        def choice(self, seq):
            raise AssertionError("draw_crop_spec must not choose among recordings")

        def choices(self, *args, **kwargs):
            raise AssertionError("draw_crop_spec must not choose among recordings")

    rng = _ChoiceMustNotRun(5)
    spec = draw_crop_spec(source, duration_seconds, rng=rng)
    assert spec.source_id == source.source_id
    assert spec.duration_seconds == duration_seconds
    assert spec.target_num_samples == max(1, round(duration_seconds * 16_000))

    oracle_rng = random.Random(5)
    waveform, sample_rate = _oracle_load_audio(
        primary,
        rng=oracle_rng,
        target_samples=max(1, round(duration_seconds * 1_000_000)),
        target_sample_rate=1_000_000,
    )
    waveform = _oracle_as_mono(waveform, source=primary)
    waveform = _oracle_match_noise_length(
        waveform,
        spec.target_num_samples,
        rng=oracle_rng,
    )
    got, got_sr = materialize_crop(source, spec)
    assert torch.equal(got, waveform)
    assert got_sr == sample_rate
    assert rng.random() == oracle_rng.random()
    assert sample_rate == 16_000
    assert other.exists()


def test_materialize_crop_does_not_decode_whole_long_file(tmp_path, monkeypatch):
    import dma_kws.stage2.background_sampling as sampling

    sample_rate = 16_000
    num_frames = 80_000
    path = _write_wav(
        tmp_path / "long.wav",
        np.linspace(-1.0, 1.0, num_frames, dtype=np.float32),
        sample_rate,
    )
    source = sampling.probe_source_info(path)
    duration_seconds = 1.25
    spec = sampling.draw_crop_spec(
        source, duration_seconds, rng=random.Random(3)
    )
    assert spec.read_num_frames < num_frames
    assert spec.read_start_frame + spec.read_num_frames <= num_frames

    reads: list[tuple[int, int, int]] = []
    original_read = sampling.sf.SoundFile.read

    def tracking_read(self, frames=-1, dtype="float64", always_2d=False, **kwargs):
        reads.append((int(frames), int(self.tell()), int(len(self))))
        return original_read(
            self, frames=frames, dtype=dtype, always_2d=always_2d, **kwargs
        )

    monkeypatch.setattr(sampling.sf.SoundFile, "read", tracking_read)
    waveform, out_sample_rate = sampling.materialize_crop(source, spec)

    assert out_sample_rate == sample_rate
    assert waveform.shape == (1, spec.target_num_samples)
    assert reads, "materialize_crop must read audio"
    assert all(frames != -1 for frames, _tell, _nframes in reads)
    assert all(frames == spec.read_num_frames for frames, _tell, _nframes in reads)
    assert all(frames < num_frames for frames, _tell, _nframes in reads)
    assert all(tell == spec.read_start_frame for _frames, tell, _nframes in reads)

    oracle_rng = random.Random(3)
    oracle_waveform, oracle_sample_rate = _oracle_load_audio(
        path,
        rng=oracle_rng,
        target_samples=max(1, round(duration_seconds * 1_000_000)),
        target_sample_rate=1_000_000,
    )
    oracle_waveform = _oracle_as_mono(oracle_waveform, source=path)
    oracle_waveform = _oracle_match_noise_length(
        oracle_waveform, spec.target_num_samples, rng=oracle_rng
    )
    assert oracle_sample_rate == sample_rate
    assert torch.equal(waveform, oracle_waveform)


def test_header_cache_two_crops_seek_read_without_full_decode(tmp_path, monkeypatch):
    import dma_kws.stage2.background_sampling as sampling
    import dma_kws.stage2.features as features

    sample_rate = 16_000
    num_frames = 64_000
    _write_wav(
        tmp_path / "music.wav",
        np.linspace(-0.2, 0.8, num_frames, dtype=np.float32),
        sample_rate,
    )
    audio_list = _write_audio_list(tmp_path, ["music.wav"])
    sampler = TrainingBackgroundSampler(
        audio_list_path=audio_list,
        duration_seconds_min=1.25,
        duration_seconds_max=1.25,
        fbank_kwargs={"dither": 0.0},
    )

    probe_calls = {"n": 0}
    original_probe = features.probe_source_info

    def counting_probe(path, *, source_id=None):
        probe_calls["n"] += 1
        return original_probe(path, source_id=source_id)

    reads: list[int] = []
    original_read = sampling.sf.SoundFile.read

    def tracking_read(self, frames=-1, dtype="float64", always_2d=False, **kwargs):
        reads.append(int(frames))
        return original_read(
            self, frames=frames, dtype=dtype, always_2d=always_2d, **kwargs
        )

    monkeypatch.setattr(features, "probe_source_info", counting_probe)
    monkeypatch.setattr(sampling.sf.SoundFile, "read", tracking_read)
    monkeypatch.setattr(
        "dma_kws.stage2.features.FbankExtractor", _CapturingFbankExtractor
    )

    sampler.extract(rng=random.Random(8))
    assert probe_calls["n"] == 1
    reads_after_first = list(reads)
    sampler.extract(rng=random.Random(9))

    assert probe_calls["n"] == 1
    assert reads_after_first
    assert all(frames != -1 and 0 < frames < num_frames for frames in reads)
    assert sum(reads) < 2 * num_frames
