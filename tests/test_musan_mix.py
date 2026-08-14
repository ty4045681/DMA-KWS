from __future__ import annotations

from array import array
import math
import pickle
from pathlib import Path
import sys
import wave

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
) -> dict:
    return {
        "musan_root": str(root),
        "musan_mix": {
            "seed": seed,
            "noise": {"enabled": noise, "snr_db": noise_snr_db},
            "music": {"enabled": music, "snr_db": music_snr_db},
            "speech": {
                "enabled": speech,
                "relative_db": speech_relative_db,
            },
        },
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
        "length_policy": "random_crop_or_repeat",
        "mix_policy": "scale_each_against_clean_rms_then_sum",
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
