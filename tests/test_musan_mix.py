from __future__ import annotations

from array import array
import math
import pickle
from pathlib import Path
import random
import sys
import wave

import numpy as np
import pytest

import dma_kws.inference.musan_mix as musan_mix
from dma_kws.inference.musan_mix import MusanWaveformMixer


def _write_audio_placeholders(root: Path, subset: str, *names: str) -> None:
    directory = root / subset
    directory.mkdir(parents=True, exist_ok=True)
    for name in names:
        (directory / name).write_bytes(b"placeholder")


def _prep(
    root: Path,
    *,
    noise: bool = False,
    noise_snr_db: float = 20.0,
    music: bool = False,
    music_snr_db: float = 20.0,
    speech: bool = False,
    speech_relative_db: float = 0.0,
    seed: int = 2025,
    stationary_noise: dict | None = None,
    burst_noise: dict | None = None,
    volume_variation: dict | None = None,
) -> dict:
    mix = {
        "seed": seed,
        "noise": {"enabled": noise, "snr_db": noise_snr_db},
        "music": {"enabled": music, "snr_db": music_snr_db},
        "speech": {
            "enabled": speech,
            "relative_db": speech_relative_db,
        },
    }
    if stationary_noise is not None:
        mix["stationary_noise"] = stationary_noise
    if burst_noise is not None:
        mix["burst_noise"] = burst_noise
    if volume_variation is not None:
        mix["volume_variation"] = volume_variation
    return {
        "musan_root": str(root),
        "musan_mix": mix,
    }


def _rms(waveform) -> float:
    return float(waveform.double().square().mean().sqrt())


def _write_pcm_wav(path: Path, samples, sample_rate: int) -> None:
    pcm = array(
        "h",
        (
            round(max(-1.0, min(1.0, float(sample))) * 32767.0)
            for sample in samples
        ),
    )
    if sys.byteorder != "little":
        pcm.byteswap()
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())


class _MixerDataset:
    def __init__(self, mixer, clean, sample_rate: int) -> None:
        self._mixer = mixer
        self._clean = clean
        self._sample_rate = sample_rate

    def __len__(self) -> int:
        return 2

    def __getitem__(self, index: int):
        return self._mixer(index, self._clean.clone(), self._sample_rate)


def test_disabled_mixer_is_noop_and_does_not_require_musan_root():
    mixer = MusanWaveformMixer.from_prep({}, audio_paths=["clean.wav"])
    sentinel = object()

    assert mixer.enabled is False
    assert mixer(0, sentinel, 16000) is sentinel
    assert mixer.summary() == {
        "enabled": False,
        "seed": 2025,
        "musan_root": None,
        "noise": {"enabled": False, "snr_db": 20.0, "num_files": 0},
        "music": {"enabled": False, "snr_db": 20.0, "num_files": 0},
        "speech": {"enabled": False, "relative_db": 0.0, "num_files": 0},
        "stationary_noise": {
            "enabled": False,
            "kind": "white_gaussian",
            "snr_db": 20.0,
        },
        "burst_noise": {
            "enabled": False,
            "snr_db": 10.0,
            "snr_scope": "active_event",
            "event_count_min": 1,
            "event_count_max": 1,
            "duration_ms_min": 100.0,
            "duration_ms_max": 400.0,
            "fade_ms": 10.0,
            "allow_overlap": False,
            "min_gap_ms": 50.0,
            "num_files": 0,
        },
        "volume_variation": {
            "enabled": False,
            "low_gain_db": -12.0,
            "high_gain_db": 6.0,
            "segment_ms_min": 250.0,
            "segment_ms_max": 750.0,
            "transition_ms": 50.0,
        },
        "length_policy": "random_crop_or_repeat",
        "mix_policy": "scale_each_against_clean_rms_then_sum",
        "processing_order": (
            "volume_variation_then_scale_all_additive_components_against_"
            "the_same_varied_clean_then_sum_without_clipping"
        ),
    }


def test_legacy_explicit_config_without_music_stays_compatible(tmp_path):
    _write_audio_placeholders(tmp_path, "noise", "noise.wav")
    prep = {
        "musan_root": str(tmp_path),
        "musan_mix": {
            "seed": 7,
            "noise": {"enabled": True, "snr_db": 10.0},
            "speech": {"enabled": False, "relative_db": 0.0},
        },
    }

    mixer = MusanWaveformMixer.from_prep(prep, audio_paths=["clean.wav"])

    assert mixer.summary()["music"] == {
        "enabled": False,
        "snr_db": 20.0,
        "num_files": 0,
    }
    assert "music" not in mixer.recipe_metadata(0)


def test_component_helpers_reject_unknown_kind():
    with pytest.raises(ValueError, match="Unsupported MUSAN component kind"):
        musan_mix._target_amplitude_ratio("invalid", 0.0, field="invalid")

    component = musan_mix._ComponentRecipe(
        kind="invalid",
        source_path="invalid.wav",
        source="invalid.wav",
        level_db=0.0,
        offset_fraction=0.0,
        recipe_seed=1,
    )
    with pytest.raises(ValueError, match="Unsupported MUSAN component kind"):
        component.metadata()


def test_enabled_mixer_requires_only_the_enabled_subset(tmp_path):
    _write_audio_placeholders(tmp_path, "noise", "noise.wav")

    mixer = MusanWaveformMixer.from_prep(
        _prep(tmp_path, noise=True),
        audio_paths=["clean.wav"],
    )

    assert mixer.enabled is True
    assert mixer.summary()["noise"]["num_files"] == 1
    assert mixer.summary()["music"]["num_files"] == 0
    assert mixer.summary()["speech"]["num_files"] == 0


def test_music_only_discovers_recursive_pool_and_records_recipe(tmp_path):
    music_dir = tmp_path / "music" / "genre"
    music_dir.mkdir(parents=True)
    (music_dir / "track.wav").write_bytes(b"placeholder")

    mixer = MusanWaveformMixer.from_prep(
        _prep(tmp_path, music=True, music_snr_db=12.0, seed=9),
        audio_paths=["clean.wav"],
    )

    assert mixer.enabled is True
    assert mixer.summary()["music"] == {
        "enabled": True,
        "snr_db": 12.0,
        "num_files": 1,
    }
    metadata = mixer.recipe_metadata(0)
    assert metadata["music"]["source"] == "music/genre/track.wav"
    assert metadata["music"]["snr_db"] == 12.0
    assert "noise" not in metadata
    assert "speech" not in metadata


def test_recipe_metadata_keeps_symlinked_source_relative_to_musan_root(tmp_path):
    noise_dir = tmp_path / "musan" / "noise"
    noise_dir.mkdir(parents=True)
    outside = tmp_path / "outside.wav"
    outside.write_bytes(b"placeholder")
    try:
        (noise_dir / "linked.wav").symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"symlinks are unavailable: {exc}")

    mixer = MusanWaveformMixer.from_prep(
        _prep(tmp_path / "musan", noise=True),
        audio_paths=["clean.wav"],
    )

    assert mixer.recipe_metadata(0)["noise"]["source"] == "noise/linked.wav"


@pytest.mark.parametrize("subset", ["noise", "music"])
def test_enabled_mixer_rejects_missing_or_empty_subset(tmp_path, subset):
    with pytest.raises(NotADirectoryError, match=f"{subset} directory not found"):
        MusanWaveformMixer.from_prep(
            _prep(tmp_path, **{subset: True}),
            audio_paths=["clean.wav"],
        )

    (tmp_path / subset).mkdir()
    with pytest.raises(ValueError, match="No supported audio files"):
        MusanWaveformMixer.from_prep(
            _prep(tmp_path, **{subset: True}),
            audio_paths=["clean.wav"],
        )


@pytest.mark.parametrize(
    ("mix_config", "message"),
    [
        ({"seed": True}, "seed must be a non-negative integer"),
        ({"seed": -1}, "seed must be a non-negative integer"),
        ({"noise": {"enabled": 1}}, "noise.enabled must be true or false"),
        ({"music": {"enabled": 1}}, "music.enabled must be true or false"),
        (
            {"music": {"enabled": False, "snr_db": float("nan")}},
            "music.snr_db must be a finite number",
        ),
        (
            {"speech": {"enabled": False, "relative_db": float("nan")}},
            "speech.relative_db must be a finite number",
        ),
        ({"unknown": {}}, "unsupported keys"),
        (
            {"stationary_noise": {"kind": "pink_gaussian"}},
            "kind must be white_gaussian",
        ),
        (
            {"burst_noise": {"snr_scope": "full_clip"}},
            "snr_scope must be active_event or whole_clip",
        ),
        (
            {"burst_noise": {"event_count_min": 2, "event_count_max": 1}},
            "event_count_max must be >= event_count_min",
        ),
        (
            {"burst_noise": {"duration_ms_min": 0}},
            "duration_ms_min must be > 0",
        ),
        (
            {"burst_noise": {"fade_ms": -1}},
            "fade_ms must be >= 0",
        ),
        (
            {"volume_variation": {"low_gain_db": 1, "high_gain_db": -1}},
            "high_gain_db must be > low_gain_db",
        ),
        (
            {"volume_variation": {"low_gain_db": 1, "high_gain_db": 1}},
            "high_gain_db must be > low_gain_db",
        ),
        (
            {"volume_variation": {"segment_ms_min": 2, "segment_ms_max": 1}},
            "segment_ms_max must be >= segment_ms_min",
        ),
        (
            {"volume_variation": {"transition_ms": -1}},
            "transition_ms must be >= 0",
        ),
        (
            {
                "volume_variation": {
                    "segment_ms_min": 10,
                    "segment_ms_max": 20,
                    "transition_ms": 11,
                }
            },
            "transition_ms must be <= segment_ms_min",
        ),
    ],
)
def test_mixer_validates_config_even_when_components_are_disabled(
    mix_config,
    message,
):
    with pytest.raises((TypeError, ValueError), match=message):
        MusanWaveformMixer.from_prep(
            {"musan_mix": mix_config},
            audio_paths=["clean.wav"],
        )


def test_recipes_are_deterministic_and_component_independent(tmp_path):
    _write_audio_placeholders(tmp_path, "noise", "n1.wav", "n2.wav")
    _write_audio_placeholders(tmp_path, "speech", "s1.wav", "s2.wav")
    audio_paths = ["clean-a.wav", "clean-b.wav"]

    noise_only = MusanWaveformMixer.from_prep(
        _prep(tmp_path, noise=True, seed=7),
        audio_paths=audio_paths,
    )
    both_a = MusanWaveformMixer.from_prep(
        _prep(tmp_path, noise=True, speech=True, seed=7),
        audio_paths=audio_paths,
    )
    both_b = MusanWaveformMixer.from_prep(
        _prep(tmp_path, noise=True, speech=True, seed=7),
        audio_paths=audio_paths,
    )
    different_seed = MusanWaveformMixer.from_prep(
        _prep(tmp_path, noise=True, speech=True, seed=8),
        audio_paths=audio_paths,
    )

    assert both_a.recipe_metadata(0) == both_b.recipe_metadata(0)
    assert both_a.recipe_metadata(1) == both_b.recipe_metadata(1)
    restored = pickle.loads(pickle.dumps(both_a))
    assert restored.recipe_metadata(0) == both_a.recipe_metadata(0)
    assert both_a.recipe_metadata(0) != different_seed.recipe_metadata(0)
    assert noise_only.recipe_metadata(0)["noise"] == both_a.recipe_metadata(0)[
        "noise"
    ]
    assert both_a.recipe_metadata(0)["noise"]["source"].startswith("noise/")
    assert both_a.recipe_metadata(0)["speech"]["source"].startswith("speech/")
    with pytest.raises(IndexError, match="row index out of range"):
        both_a.recipe_metadata(-1)


@pytest.mark.parametrize("snr_db", [10.0, 20.0])
def test_noise_mixing_reaches_requested_snr(tmp_path, monkeypatch, snr_db):
    torch = pytest.importorskip("torch")
    _write_audio_placeholders(tmp_path, "noise", "noise.wav")
    clean = torch.tensor([[0.5, -0.5, 0.25, -0.25]])
    noise = torch.tensor([[1.0, -1.0, -1.0, 1.0]])
    monkeypatch.setattr(
        musan_mix,
        "load_audio",
        lambda _path, *, sample_rate: (noise, sample_rate),
    )
    mixer = MusanWaveformMixer.from_prep(
        _prep(tmp_path, noise=True, noise_snr_db=snr_db),
        audio_paths=["clean.wav"],
    )

    mixed = mixer(0, clean, 16000)
    added = mixed - clean
    actual_snr_db = 20.0 * math.log10(_rms(clean) / _rms(added))

    assert mixed.shape == clean.shape
    assert actual_snr_db == pytest.approx(snr_db, abs=1.0e-5)


@pytest.mark.parametrize("snr_db", [10.0, 20.0])
def test_music_mixing_reaches_requested_snr(tmp_path, monkeypatch, snr_db):
    torch = pytest.importorskip("torch")
    _write_audio_placeholders(tmp_path, "music", "music.wav")
    clean = torch.tensor([[0.5, -0.5, 0.25, -0.25]])
    music = torch.tensor([[1.0, 1.0, -1.0, -1.0]])
    monkeypatch.setattr(
        musan_mix,
        "load_audio",
        lambda _path, *, sample_rate: (music, sample_rate),
    )
    mixer = MusanWaveformMixer.from_prep(
        _prep(tmp_path, music=True, music_snr_db=snr_db),
        audio_paths=["clean.wav"],
    )

    mixed = mixer(0, clean, 16000)
    added = mixed - clean
    actual_snr_db = 20.0 * math.log10(_rms(clean) / _rms(added))

    assert mixed.shape == clean.shape
    assert actual_snr_db == pytest.approx(snr_db, abs=1.0e-5)


@pytest.mark.parametrize("relative_db", [-6.0, 0.0, 6.0])
def test_speech_mixing_reaches_requested_relative_level(
    tmp_path,
    monkeypatch,
    relative_db,
):
    torch = pytest.importorskip("torch")
    _write_audio_placeholders(tmp_path, "speech", "speech.wav")
    clean = torch.tensor([[0.5, -0.5, 0.25, -0.25]])
    speech = torch.tensor([[1.0, 1.0, -1.0, -1.0]])
    monkeypatch.setattr(
        musan_mix,
        "load_audio",
        lambda _path, *, sample_rate: (speech, sample_rate),
    )
    mixer = MusanWaveformMixer.from_prep(
        _prep(tmp_path, speech=True, speech_relative_db=relative_db),
        audio_paths=["clean.wav"],
    )

    mixed = mixer(0, clean, 16000)
    added = mixed - clean
    actual_relative_db = 20.0 * math.log10(_rms(added) / _rms(clean))

    assert actual_relative_db == pytest.approx(relative_db, abs=1.0e-5)


def test_noise_and_speech_are_both_scaled_against_clean(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    _write_audio_placeholders(tmp_path, "noise", "noise.wav")
    _write_audio_placeholders(tmp_path, "speech", "speech.wav")
    clean = torch.tensor([[0.5, -0.5, 0.25, -0.25]])
    sources = {
        "noise": torch.tensor([[1.0, -1.0, -1.0, 1.0]]),
        "speech": torch.tensor([[1.0, 1.0, -1.0, -1.0]]),
    }

    def loader(path, *, sample_rate):
        kind = Path(path).parent.name
        return sources[kind], sample_rate

    monkeypatch.setattr(musan_mix, "load_audio", loader)
    common = {
        "noise_snr_db": 10.0,
        "speech_relative_db": 6.0,
        "seed": 11,
    }
    noise_only = MusanWaveformMixer.from_prep(
        _prep(tmp_path, noise=True, **common),
        audio_paths=["clean.wav"],
    )
    speech_only = MusanWaveformMixer.from_prep(
        _prep(tmp_path, speech=True, **common),
        audio_paths=["clean.wav"],
    )
    both = MusanWaveformMixer.from_prep(
        _prep(tmp_path, noise=True, speech=True, **common),
        audio_paths=["clean.wav"],
    )

    noise_component = noise_only(0, clean, 16000) - clean
    speech_component = speech_only(0, clean, 16000) - clean
    expected = clean + noise_component + speech_component

    assert torch.allclose(both(0, clean, 16000), expected, atol=1.0e-7)
    assert 20.0 * math.log10(_rms(clean) / _rms(noise_component)) == pytest.approx(
        10.0,
        abs=1.0e-5,
    )
    assert 20.0 * math.log10(_rms(speech_component) / _rms(clean)) == pytest.approx(
        6.0,
        abs=1.0e-5,
    )


def test_noise_and_music_are_independent_and_scaled_against_clean(
    tmp_path,
    monkeypatch,
):
    torch = pytest.importorskip("torch")
    _write_audio_placeholders(tmp_path, "noise", "noise.wav")
    _write_audio_placeholders(tmp_path, "music", "music.wav")
    clean = torch.tensor([[0.5, -0.5, 0.25, -0.25]])
    sources = {
        "noise": torch.tensor([[1.0, -1.0, -1.0, 1.0]]),
        "music": torch.tensor([[1.0, 1.0, -1.0, -1.0]]),
    }

    def loader(path, *, sample_rate):
        kind = Path(path).parent.name
        return sources[kind], sample_rate

    monkeypatch.setattr(musan_mix, "load_audio", loader)
    common = {"noise_snr_db": 10.0, "music_snr_db": 6.0, "seed": 11}
    noise_only = MusanWaveformMixer.from_prep(
        _prep(tmp_path, noise=True, **common),
        audio_paths=["clean.wav"],
    )
    music_only = MusanWaveformMixer.from_prep(
        _prep(tmp_path, music=True, **common),
        audio_paths=["clean.wav"],
    )
    both = MusanWaveformMixer.from_prep(
        _prep(tmp_path, noise=True, music=True, **common),
        audio_paths=["clean.wav"],
    )

    assert noise_only.recipe_metadata(0)["noise"] == both.recipe_metadata(0)[
        "noise"
    ]
    assert music_only.recipe_metadata(0)["music"] == both.recipe_metadata(0)[
        "music"
    ]
    noise_component = noise_only(0, clean, 16000) - clean
    music_component = music_only(0, clean, 16000) - clean
    expected = clean + noise_component + music_component

    assert torch.allclose(both(0, clean, 16000), expected, atol=1.0e-7)
    assert 20.0 * math.log10(_rms(clean) / _rms(noise_component)) == pytest.approx(
        10.0,
        abs=1.0e-5,
    )
    assert 20.0 * math.log10(_rms(clean) / _rms(music_component)) == pytest.approx(
        6.0,
        abs=1.0e-5,
    )


def test_match_length_random_crops_or_repeats_exactly():
    torch = pytest.importorskip("torch")
    long_source = torch.arange(1.0, 11.0).unsqueeze(0)
    short_source = torch.tensor([[1.0, 2.0, 3.0]])

    cropped = musan_mix._match_length(
        long_source,
        4,
        0.5,
        field="long source",
    )
    repeated = musan_mix._match_length(
        short_source,
        8,
        0.5,
        field="short source",
    )

    assert torch.equal(cropped, torch.tensor([[4.0, 5.0, 6.0, 7.0]]))
    assert torch.equal(
        repeated,
        torch.tensor([[2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0]]),
    )


def test_long_source_uses_bounded_partial_read(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchaudio")
    noise_dir = tmp_path / "noise"
    noise_dir.mkdir()
    noise_path = noise_dir / "long.wav"
    _write_pcm_wav(
        noise_path,
        torch.linspace(-0.75, 0.75, 32000),
        16000,
    )
    monkeypatch.setattr(
        musan_mix,
        "load_audio",
        lambda *_args, **_kwargs: pytest.fail("full source load was used"),
    )
    mixer = MusanWaveformMixer.from_prep(
        _prep(tmp_path, noise=True),
        audio_paths=["clean.wav"],
    )
    clean = torch.linspace(-0.5, 0.5, 8000).unsqueeze(0)

    mixed = mixer(0, clean, 16000)

    assert mixed.shape == clean.shape
    assert torch.isfinite(mixed).all()


def test_pcm_partial_read_preserves_last_valid_crop_start(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    source_path = tmp_path / "source.wav"
    _write_pcm_wav(source_path, [-0.8, -0.4, 0.0, 0.4, 0.8], 16000)
    monkeypatch.setattr(
        musan_mix,
        "load_audio",
        lambda *_args, **_kwargs: pytest.fail("full source load was used"),
    )
    component = musan_mix._ComponentRecipe(
        kind="noise",
        source_path=str(source_path),
        source="noise/source.wav",
        level_db=20.0,
        offset_fraction=0.999999,
        recipe_seed=1,
    )

    waveform, sample_rate = musan_mix._load_source_audio(
        component,
        target_samples=4,
        sample_rate=16000,
    )

    assert sample_rate == 16000
    assert waveform.squeeze(0).tolist() == pytest.approx(
        [-0.4, 0.0, 0.4, 0.8],
        abs=5.0e-5,
    )


def test_short_pcm_source_is_read_once_then_repeated(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    source_path = tmp_path / "source.wav"
    _write_pcm_wav(source_path, [0.25, 0.5, 0.75], 16000)
    monkeypatch.setattr(
        musan_mix,
        "load_audio",
        lambda *_args, **_kwargs: pytest.fail("generic full source load was used"),
    )
    component = musan_mix._ComponentRecipe(
        kind="speech",
        source_path=str(source_path),
        source="speech/source.wav",
        level_db=0.0,
        offset_fraction=0.5,
        recipe_seed=1,
    )

    waveform, sample_rate = musan_mix._load_source_audio(
        component,
        target_samples=8,
        sample_rate=16000,
    )
    repeated = musan_mix._match_length(
        waveform,
        8,
        component.offset_fraction,
        field="short source",
    )

    assert sample_rate == 16000
    assert waveform.shape == (1, 3)
    assert repeated.squeeze(0).tolist() == pytest.approx(
        [0.5, 0.75, 0.25, 0.5, 0.75, 0.25, 0.5, 0.75],
        abs=5.0e-5,
    )


def test_spawned_dataloader_workers_match_single_process_output(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchaudio")
    noise_dir = tmp_path / "noise"
    noise_dir.mkdir()
    _write_pcm_wav(
        noise_dir / "noise.wav",
        torch.sin(torch.linspace(0.0, 20.0, 16000)),
        16000,
    )
    music_dir = tmp_path / "music"
    music_dir.mkdir()
    _write_pcm_wav(
        music_dir / "music.wav",
        torch.cos(torch.linspace(0.0, 15.0, 16000)),
        16000,
    )
    mixer = MusanWaveformMixer.from_prep(
        _prep(
            tmp_path,
            noise=True,
            noise_snr_db=10.0,
            music=True,
            music_snr_db=12.0,
            seed=3,
        ),
        audio_paths=["clean-a.wav", "clean-b.wav"],
    )
    clean = torch.linspace(-0.5, 0.5, 4000).unsqueeze(0)
    dataset = _MixerDataset(mixer, clean, 16000)
    expected = [dataset[index] for index in range(len(dataset))]

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=None,
        num_workers=2,
        multiprocessing_context="spawn",
    )
    actual = list(loader)

    assert len(actual) == len(expected)
    assert all(
        torch.allclose(worker_value, expected_value, atol=1.0e-7)
        for worker_value, expected_value in zip(actual, expected)
    )


@pytest.mark.parametrize(
    ("clean_values", "source_values", "message"),
    [
        ([0.0, 0.0], [1.0, -1.0], "clean waveform.*RMS"),
        ([1.0, -1.0], [0.0, 0.0], "MUSAN noise segment.*RMS"),
        ([1.0, -1.0], [1.0, float("nan")], "non-finite samples"),
    ],
)
def test_mixer_rejects_undefined_level_inputs(
    tmp_path,
    monkeypatch,
    clean_values,
    source_values,
    message,
):
    torch = pytest.importorskip("torch")
    _write_audio_placeholders(tmp_path, "noise", "noise.wav")
    source = torch.tensor([source_values])
    monkeypatch.setattr(
        musan_mix,
        "load_audio",
        lambda _path, *, sample_rate: (source, sample_rate),
    )
    mixer = MusanWaveformMixer.from_prep(
        _prep(tmp_path, noise=True),
        audio_paths=["clean.wav"],
    )

    with pytest.raises(ValueError, match=message):
        mixer(0, torch.tensor([clean_values]), 16000)


def test_synthetic_and_volume_only_do_not_require_musan_root():
    torch = pytest.importorskip("torch")
    stationary = MusanWaveformMixer.from_prep(
        {
            "musan_mix": {
                "stationary_noise": {
                    "enabled": True,
                    "kind": "white_gaussian",
                    "snr_db": 20.0,
                }
            }
        },
        audio_paths=["clean.wav"],
    )
    volume = MusanWaveformMixer.from_prep(
        {
            "musan_mix": {
                "volume_variation": {
                    "enabled": True,
                    "low_gain_db": -6.0,
                    "high_gain_db": -5.0,
                    "segment_ms_min": 100.0,
                    "segment_ms_max": 100.0,
                    "transition_ms": 10.0,
                }
            }
        },
        audio_paths=["clean.wav"],
    )

    assert stationary.summary()["musan_root"] is None
    assert volume.summary()["musan_root"] is None
    assert stationary.enabled and stationary.additive_enabled
    assert volume.enabled and volume.pre_mix_enabled
    silence = torch.zeros(1, 100)
    assert torch.equal(volume(0, silence, 1000), silence)


def test_burst_noise_requires_musan_noise_pool(tmp_path):
    config = {
        "musan_mix": {
            "burst_noise": {
                "enabled": True,
                "snr_db": 10.0,
                "snr_scope": "active_event",
            }
        }
    }
    with pytest.raises(ValueError, match="musan_root is required"):
        MusanWaveformMixer.from_prep(config, audio_paths=["clean.wav"])

    config["musan_root"] = str(tmp_path)
    with pytest.raises(NotADirectoryError, match="noise directory not found"):
        MusanWaveformMixer.from_prep(config, audio_paths=["clean.wav"])


@pytest.mark.parametrize("snr_db", [0.0, 10.0, 20.0])
def test_stationary_white_noise_reaches_exact_snr_and_is_deterministic(snr_db):
    torch = pytest.importorskip("torch")
    prep = {
        "musan_mix": {
            "seed": 17,
            "stationary_noise": {
                "enabled": True,
                "kind": "white_gaussian",
                "snr_db": snr_db,
            },
        }
    }
    mixer = MusanWaveformMixer.from_prep(prep, audio_paths=["clean.wav"])
    rebuilt = MusanWaveformMixer.from_prep(prep, audio_paths=["clean.wav"])
    clean = torch.sin(torch.linspace(0.0, 50.0, 4096)).unsqueeze(0) * 0.4

    delta = mixer.apply_additive_delta(0, clean, 16000)
    achieved = 20.0 * math.log10(_rms(clean) / _rms(delta))

    assert achieved == pytest.approx(snr_db, abs=1.0e-5)
    assert torch.equal(delta, rebuilt.apply_additive_delta(0, clean, 16000))
    assert mixer.recipe_metadata(0)["stationary_noise"]["kind"] == "white_gaussian"
    assert abs(float(delta.double().mean())) < 1.0e-8


def test_volume_variation_uses_seeded_extreme_gain_and_does_not_clip():
    torch = pytest.importorskip("torch")
    mixer = MusanWaveformMixer.from_prep(
        {
            "musan_mix": {
                "seed": 2,
                "volume_variation": {
                    "enabled": True,
                    "low_gain_db": -6.0,
                    "high_gain_db": 6.0,
                    "segment_ms_min": 20.0,
                    "segment_ms_max": 20.0,
                    "transition_ms": 10.0,
                }
            }
        },
        audio_paths=["clean.wav"],
    )
    clean = torch.tensor([[0.75, -0.75, 0.25, -0.25]])
    recipe_seed = mixer.recipe_metadata(0)["volume_variation"]["recipe_seed"]
    starts_high = random.Random(recipe_seed).randrange(2) == 1
    expected_db = 6.0 if starts_high else -6.0

    varied = mixer.apply_pre_mix(0, clean, 1000)

    assert torch.allclose(
        varied,
        clean * (10.0 ** (expected_db / 20.0)),
        atol=1.0e-7,
    )
    assert starts_high
    assert varied.abs().max().item() > 1.0
    assert torch.count_nonzero(mixer.apply_additive_delta(0, varied, 1000)) == 0
    assert torch.equal(mixer(0, clean, 1000), varied)


def test_volume_envelope_segments_and_transitions_are_seeded():
    config = musan_mix._VolumeVariationConfig(
        enabled=True,
        low_gain_db=-12.0,
        high_gain_db=6.0,
        segment_ms_min=100.0,
        segment_ms_max=100.0,
        transition_ms=10.0,
    )
    first = musan_mix._volume_envelope(
        target_samples=500,
        sample_rate=1000,
        config=config,
        recipe_seed=123,
    )
    rebuilt = musan_mix._volume_envelope(
        target_samples=500,
        sample_rate=1000,
        config=config,
        recipe_seed=123,
    )
    different = musan_mix._volume_envelope(
        target_samples=500,
        sample_rate=1000,
        config=config,
        recipe_seed=124,
    )

    assert first.shape == (500,)
    assert first.min() >= 10.0 ** (-12.0 / 20.0)
    assert first.max() <= 10.0 ** (6.0 / 20.0)
    assert np.array_equal(first, rebuilt)
    assert not np.array_equal(first, different)
    assert first[0] == first[99]
    assert first[110] == first[199]
    assert first[210] == first[299]
    assert first[0] == first[210]
    assert first[0] != first[110]
    assert len(np.unique(first[100:110])) == 10
    seeded = random.Random(123)
    first_db = -12.0 if seeded.randrange(2) == 0 else 6.0
    second_db = 6.0 if first_db == -12.0 else -12.0
    phase = np.arange(1, 11, dtype=np.float64) / 11.0
    raised_cosine = 0.5 - (0.5 * np.cos(np.pi * phase))
    expected_transition = np.power(
        10.0,
        (first_db + raised_cosine * (second_db - first_db)) / 20.0,
    )
    assert np.allclose(first[100:110], expected_transition)


@pytest.mark.parametrize("snr_scope", ["active_event", "whole_clip"])
def test_burst_noise_snr_scope_math(tmp_path, monkeypatch, snr_scope):
    torch = pytest.importorskip("torch")
    _write_audio_placeholders(tmp_path, "noise", "burst.wav")
    source = torch.ones(1, 1000)
    monkeypatch.setattr(
        musan_mix,
        "load_audio",
        lambda _path, *, sample_rate: (source, sample_rate),
    )
    mixer = MusanWaveformMixer.from_prep(
        _prep(
            tmp_path,
            burst_noise={
                "enabled": True,
                "snr_db": 10.0,
                "snr_scope": snr_scope,
                "event_count_min": 1,
                "event_count_max": 1,
                "duration_ms_min": 100.0,
                "duration_ms_max": 100.0,
                "fade_ms": 0.0,
                "allow_overlap": False,
                "min_gap_ms": 0.0,
            },
        ),
        audio_paths=["clean.wav"],
    )
    clean = torch.linspace(0.1, 0.9, 1000).unsqueeze(0)

    delta = mixer.apply_additive_delta(0, clean, 1000)
    active = delta[delta != 0]

    assert active.numel() == 100
    if snr_scope == "active_event":
        achieved = 20.0 * math.log10(_rms(clean) / _rms(active))
    else:
        achieved = 20.0 * math.log10(_rms(clean) / _rms(delta))
    assert achieved == pytest.approx(10.0, abs=1.0e-5)


def test_active_event_multi_burst_with_fade_uses_global_clean_rms(
    tmp_path,
    monkeypatch,
):
    torch = pytest.importorskip("torch")
    _write_audio_placeholders(tmp_path, "noise", "burst.wav")
    monkeypatch.setattr(
        musan_mix,
        "load_audio",
        lambda _path, *, sample_rate: (torch.ones(1, 1000), sample_rate),
    )
    mixer = MusanWaveformMixer.from_prep(
        _prep(
            tmp_path,
            seed=41,
            burst_noise={
                "enabled": True,
                "snr_db": 12.0,
                "snr_scope": "active_event",
                "event_count_min": 2,
                "event_count_max": 2,
                "duration_ms_min": 150.0,
                "duration_ms_max": 150.0,
                "fade_ms": 20.0,
                "allow_overlap": False,
                "min_gap_ms": 50.0,
            },
        ),
        audio_paths=["clean.wav"],
    )
    clean = torch.linspace(0.05, 0.95, 1000).unsqueeze(0)

    delta = mixer.apply_additive_delta(0, clean, 1000)
    burst_recipe = mixer._recipe_at(0).burst_noise
    assert burst_recipe is not None
    placed = []
    for event in burst_recipe.events:
        event_samples = round(event.duration_ms)
        start = musan_mix._burst_start(
            target_samples=clean.size(1),
            event_samples=event_samples,
            placement_fraction=event.placement_fraction,
            allow_overlap=False,
            min_gap_samples=50,
            placed=placed,
        )
        end = start + event_samples
        placed.append((start, end))
        event_delta = delta[:, start:end]
        achieved = 20.0 * math.log10(_rms(clean) / _rms(event_delta))
        assert achieved == pytest.approx(12.0, abs=1.0e-5)

    assert len(placed) == 2
    first, second = sorted(placed)
    assert second[0] - first[1] >= 50


def test_whole_clip_multi_burst_with_fade_and_overlap_reaches_exact_snr(
    tmp_path,
    monkeypatch,
):
    torch = pytest.importorskip("torch")
    _write_audio_placeholders(tmp_path, "noise", "burst.wav")
    monkeypatch.setattr(
        musan_mix,
        "load_audio",
        lambda _path, *, sample_rate: (torch.ones(1, 1000), sample_rate),
    )
    mixer = MusanWaveformMixer.from_prep(
        _prep(
            tmp_path,
            seed=43,
            burst_noise={
                "enabled": True,
                "snr_db": 7.0,
                "snr_scope": "whole_clip",
                "event_count_min": 2,
                "event_count_max": 2,
                "duration_ms_min": 800.0,
                "duration_ms_max": 800.0,
                "fade_ms": 25.0,
                "allow_overlap": True,
                "min_gap_ms": 50.0,
            },
        ),
        audio_paths=["clean.wav"],
    )
    clean = torch.linspace(0.05, 0.95, 1000).unsqueeze(0)

    delta = mixer.apply_additive_delta(0, clean, 1000)
    achieved = 20.0 * math.log10(_rms(clean) / _rms(delta))

    assert achieved == pytest.approx(7.0, abs=1.0e-5)
    recipe = mixer._recipe_at(0).burst_noise
    assert recipe is not None
    starts = [
        musan_mix._burst_start(
            target_samples=clean.size(1),
            event_samples=round(event.duration_ms),
            placement_fraction=event.placement_fraction,
            allow_overlap=True,
            min_gap_samples=50,
            placed=[],
        )
        for event in recipe.events
    ]
    assert abs(starts[0] - starts[1]) < 800


@pytest.mark.parametrize(
    "component",
    ["stationary_noise", "burst_active_event", "burst_whole_clip"],
)
def test_positive_additive_recipes_reject_silent_clean(
    tmp_path,
    monkeypatch,
    component,
):
    torch = pytest.importorskip("torch")
    _write_audio_placeholders(tmp_path, "noise", "unused.wav")
    monkeypatch.setattr(
        musan_mix,
        "load_audio",
        lambda *_args, **_kwargs: pytest.fail("silent clean loaded a source"),
    )
    if component == "stationary_noise":
        prep = _prep(
            tmp_path,
            stationary_noise={
                "enabled": True,
                "kind": "white_gaussian",
                "snr_db": 10.0,
            },
        )
    else:
        prep = _prep(
            tmp_path,
            burst_noise={
                "enabled": True,
                "snr_db": 10.0,
                "snr_scope": component.removeprefix("burst_"),
                "event_count_min": 1,
                "event_count_max": 1,
                "duration_ms_min": 100.0,
                "duration_ms_max": 100.0,
                "fade_ms": 10.0,
                "allow_overlap": False,
                "min_gap_ms": 0.0,
            },
        )
    mixer = MusanWaveformMixer.from_prep(prep, audio_paths=["clean.wav"])

    with pytest.raises(
        ValueError,
        match=r"post-volume clean waveform for row 0 RMS must be greater",
    ):
        mixer.apply_additive_delta(0, torch.zeros(1, 1000), 1000)


def test_burst_fade_never_zeroes_length_one_or_two_events():
    assert np.array_equal(musan_mix._fade_envelope(1, 100), np.ones(1))
    assert np.array_equal(musan_mix._fade_envelope(2, 100), np.ones(2))
    faded = musan_mix._fade_envelope(5, 100)
    assert faded.tolist() == pytest.approx([0.0, 0.5, 1.0, 0.5, 0.0])
    assert faded[0] == faded[-1] == 0.0
    assert np.array_equal(faded, faded[::-1])
    even = musan_mix._fade_envelope(6, 100)
    assert even.tolist() == pytest.approx([0.0, 0.5, 1.0, 1.0, 0.5, 0.0])


def test_burst_events_respect_non_overlap_and_minimum_gap(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    _write_audio_placeholders(tmp_path, "noise", "burst.wav")
    monkeypatch.setattr(
        musan_mix,
        "load_audio",
        lambda _path, *, sample_rate: (torch.ones(1, 1000), sample_rate),
    )
    mixer = MusanWaveformMixer.from_prep(
        _prep(
            tmp_path,
            seed=23,
            burst_noise={
                "enabled": True,
                "snr_db": 10.0,
                "snr_scope": "active_event",
                "event_count_min": 2,
                "event_count_max": 2,
                "duration_ms_min": 100.0,
                "duration_ms_max": 100.0,
                "fade_ms": 0.0,
                "allow_overlap": False,
                "min_gap_ms": 50.0,
            },
        ),
        audio_paths=["clean.wav"],
    )

    delta = mixer.apply_additive_delta(0, torch.full((1, 500), 0.5), 1000)
    indexes = torch.nonzero(delta[0], as_tuple=False).flatten().tolist()
    split = next(i for i in range(1, len(indexes)) if indexes[i] > indexes[i - 1] + 1)
    first, second = indexes[:split], indexes[split:]

    assert len(first) == len(second) == 100
    assert second[0] - first[-1] - 1 >= 50
    metadata = mixer.recipe_metadata(0)["burst_noise"]
    assert metadata["event_count"] == 2
    assert all(0.0 <= event["start_fraction"] < 1.0 for event in metadata["events"])
    assert all(
        0.0 <= event["source_offset_fraction"] < 1.0
        for event in metadata["events"]
    )


def test_impossible_non_overlapping_burst_schedule_fails_clearly(
    tmp_path,
    monkeypatch,
):
    torch = pytest.importorskip("torch")
    _write_audio_placeholders(tmp_path, "noise", "burst.wav")
    monkeypatch.setattr(
        musan_mix,
        "load_audio",
        lambda _path, *, sample_rate: (torch.ones(1, 100), sample_rate),
    )
    mixer = MusanWaveformMixer.from_prep(
        _prep(
            tmp_path,
            burst_noise={
                "enabled": True,
                "snr_db": 10.0,
                "snr_scope": "active_event",
                "event_count_min": 2,
                "event_count_max": 2,
                "duration_ms_min": 100.0,
                "duration_ms_max": 100.0,
                "fade_ms": 0.0,
                "allow_overlap": False,
                "min_gap_ms": 1.0,
            },
        ),
        audio_paths=["clean.wav"],
    )

    with pytest.raises(ValueError, match="cannot fit without overlap"):
        mixer(0, torch.full((1, 100), 0.5), 1000)


@pytest.mark.parametrize("snr_scope", ["active_event", "whole_clip"])
def test_zero_burst_events_return_zero_without_loading_or_rms(
    tmp_path,
    monkeypatch,
    snr_scope,
):
    torch = pytest.importorskip("torch")
    _write_audio_placeholders(tmp_path, "noise", "unused.wav")
    monkeypatch.setattr(
        musan_mix,
        "load_audio",
        lambda *_args, **_kwargs: pytest.fail("zero-event burst loaded a source"),
    )
    mixer = MusanWaveformMixer.from_prep(
        _prep(
            tmp_path,
            burst_noise={
                "enabled": True,
                "snr_db": 10.0,
                "snr_scope": snr_scope,
                "event_count_min": 0,
                "event_count_max": 0,
                "duration_ms_min": 100.0,
                "duration_ms_max": 100.0,
                "fade_ms": 10.0,
                "allow_overlap": False,
                "min_gap_ms": 50.0,
            },
        ),
        audio_paths=["clean.wav"],
    )
    silence = torch.zeros(1, 100)

    delta = mixer.apply_additive_delta(0, silence, 1000)

    assert torch.equal(delta, torch.zeros_like(silence))
    assert torch.equal(mixer(0, silence, 1000), silence)
    assert mixer.recipe_metadata(0)["burst_noise"]["event_count"] == 0


def test_new_component_recipes_are_pickle_safe_and_component_independent(tmp_path):
    _write_audio_placeholders(tmp_path, "noise", "a.wav", "b.wav")
    common = dict(
        seed=31,
        stationary_noise={"enabled": True, "kind": "white_gaussian", "snr_db": 15.0},
        burst_noise={
            "enabled": True,
            "snr_db": 8.0,
            "snr_scope": "active_event",
            "event_count_min": 2,
            "event_count_max": 2,
            "duration_ms_min": 50.0,
            "duration_ms_max": 80.0,
            "fade_ms": 5.0,
            "allow_overlap": True,
            "min_gap_ms": 0.0,
        },
    )
    both = MusanWaveformMixer.from_prep(
        _prep(tmp_path, volume_variation={"enabled": True}, **common),
        audio_paths=["a-clean.wav", "b-clean.wav"],
    )
    without_volume = MusanWaveformMixer.from_prep(
        _prep(tmp_path, **common),
        audio_paths=["a-clean.wav", "b-clean.wav"],
    )
    restored = pickle.loads(pickle.dumps(both))

    assert restored.recipe_metadata(0) == both.recipe_metadata(0)
    assert both.recipe_metadata(1) == restored.recipe_metadata(1)
    assert both.recipe_metadata(0)["stationary_noise"] == without_volume.recipe_metadata(0)[
        "stationary_noise"
    ]
    assert both.recipe_metadata(0)["burst_noise"] == without_volume.recipe_metadata(0)[
        "burst_noise"
    ]


def test_all_additive_components_use_same_volume_varied_clean(
    tmp_path,
    monkeypatch,
):
    torch = pytest.importorskip("torch")
    _write_audio_placeholders(tmp_path, "noise", "source.wav")
    source = torch.sin(torch.linspace(0.1, 30.0, 1000)).unsqueeze(0)
    monkeypatch.setattr(
        musan_mix,
        "load_audio",
        lambda _path, *, sample_rate: (source, sample_rate),
    )
    volume_config = {
        "enabled": True,
        "low_gain_db": -6.0,
        "high_gain_db": -5.0,
        "segment_ms_min": 2000.0,
        "segment_ms_max": 2000.0,
        "transition_ms": 10.0,
    }
    burst_config = {
        "enabled": True,
        "snr_db": 8.0,
        "snr_scope": "active_event",
        "event_count_min": 1,
        "event_count_max": 1,
        "duration_ms_min": 100.0,
        "duration_ms_max": 100.0,
        "fade_ms": 5.0,
        "allow_overlap": False,
        "min_gap_ms": 0.0,
    }
    base_kwargs = {"seed": 41, "volume_variation": volume_config}
    old_noise = MusanWaveformMixer.from_prep(
        _prep(tmp_path, noise=True, noise_snr_db=10.0, **base_kwargs),
        audio_paths=["clean.wav"],
    )
    stationary = MusanWaveformMixer.from_prep(
        _prep(
            tmp_path,
            stationary_noise={
                "enabled": True,
                "kind": "white_gaussian",
                "snr_db": 12.0,
            },
            **base_kwargs,
        ),
        audio_paths=["clean.wav"],
    )
    burst = MusanWaveformMixer.from_prep(
        _prep(tmp_path, burst_noise=burst_config, **base_kwargs),
        audio_paths=["clean.wav"],
    )
    combined = MusanWaveformMixer.from_prep(
        _prep(
            tmp_path,
            noise=True,
            noise_snr_db=10.0,
            stationary_noise={
                "enabled": True,
                "kind": "white_gaussian",
                "snr_db": 12.0,
            },
            burst_noise=burst_config,
            **base_kwargs,
        ),
        audio_paths=["clean.wav"],
    )
    clean = torch.cos(torch.linspace(0.0, 20.0, 1000)).unsqueeze(0) * 0.5
    varied = combined.apply_pre_mix(0, clean, 1000)

    expected_delta = (
        old_noise.apply_additive_delta(0, varied, 1000)
        + stationary.apply_additive_delta(0, varied, 1000)
        + burst.apply_additive_delta(0, varied, 1000)
    )
    actual_delta = combined.apply_additive_delta(0, varied, 1000)

    assert torch.allclose(actual_delta, expected_delta, atol=2.0e-7)
    assert torch.allclose(combined(0, clean, 1000), varied + actual_delta, atol=1.0e-7)


def test_legacy_noise_music_speech_callable_preserves_sequential_formula(
    tmp_path,
    monkeypatch,
):
    torch = pytest.importorskip("torch")
    _write_audio_placeholders(tmp_path, "noise", "noise.wav")
    _write_audio_placeholders(tmp_path, "music", "music.wav")
    _write_audio_placeholders(tmp_path, "speech", "speech.wav")
    sources = {
        "noise": torch.tensor([[1.0, -1.0, -1.0, 1.0]]),
        "music": torch.tensor([[1.0, 1.0, -1.0, -1.0]]),
        "speech": torch.tensor([[0.5, -0.5, 1.0, -1.0]]),
    }

    def loader(path, *, sample_rate):
        return sources[Path(path).parent.name], sample_rate

    monkeypatch.setattr(musan_mix, "load_audio", loader)
    mixer = MusanWaveformMixer.from_prep(
        _prep(
            tmp_path,
            noise=True,
            noise_snr_db=10.0,
            music=True,
            music_snr_db=12.0,
            speech=True,
            speech_relative_db=-3.0,
            seed=51,
        ),
        audio_paths=["clean.wav"],
    )
    clean = torch.tensor([[0.5, -0.5, 0.25, -0.25]])
    clean_rms = _rms(clean)
    noise = sources["noise"] * (
        clean_rms * (10.0 ** (-10.0 / 20.0)) / _rms(sources["noise"])
    )
    music = sources["music"] * (
        clean_rms * (10.0 ** (-12.0 / 20.0)) / _rms(sources["music"])
    )
    speech = sources["speech"] * (
        clean_rms * (10.0 ** (-3.0 / 20.0)) / _rms(sources["speech"])
    )
    expected = ((clean + noise) + music) + speech

    assert torch.equal(mixer(0, clean, 16000), expected)
