import io
import wave
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from scripts.generate_hey_eva_tts import (
    KokoroProvider,
    PiperProvider,
    ProviderError,
    build_jobs,
    build_provider,
    load_plan,
)


def _wav_bytes(seconds: float = 0.3) -> bytes:
    rate = 16000
    frames = int(rate * seconds)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(b"\x00\x01" * frames)
    return buffer.getvalue()


class FakeRunner:
    """Stands in for subprocess.run: records the call and writes the output file."""

    def __init__(self) -> None:
        self.calls: list[tuple[list[str], bytes]] = []

    def __call__(self, command, payload) -> None:
        self.calls.append(([str(part) for part in command], payload))
        if "-f" in command:
            target = Path(command[command.index("-f") + 1])
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(_wav_bytes())


class FakeKokoroPipeline:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def generate_from_tokens(self, *, tokens, voice, speed):
        self.calls.append({"tokens": tokens, "voice": voice, "speed": speed})
        return [SimpleNamespace(audio=np.zeros(2400, dtype="float32"))]


def _write_ipa_plan(tmp_path: Path) -> Path:
    path = tmp_path / "local.yaml"
    path.write_text(
        f"run_root: {tmp_path / 'run'}\n"
        "keyword: hey eva\n"
        "words:\n"
        '  hey: {phones: [HH, EY1], ipa: "ˈheɪ"}\n'
        '  eva: {phones: [IY1, V, AH0], ipa: "ˈiːvə"}\n'
        "phoneme_words: [eva]\n"
        "phrases:\n"
        '  - {text: "Hey Eva", label: 1}\n'
        "providers:\n"
        "  piper:\n"
        "    speeds: [0.9, 1.0, 1.1]\n"
        "    volumes: [0.8, 1.0, 1.2]\n"
        "    voices:\n"
        "      - {voice_id: en_US-lessac-medium, name: Lessac}\n"
        "  kokoro:\n"
        "    speeds: [0.9, 1.0, 1.1]\n"
        "    voices:\n"
        "      - {voice_id: af_bella, name: Bella}\n",
        encoding="utf-8",
    )
    return path


def test_build_provider_dispatches_by_name(tmp_path):
    plan = load_plan(_write_ipa_plan(tmp_path))
    assert isinstance(build_provider("piper", plan), PiperProvider)
    assert isinstance(build_provider("kokoro", plan), KokoroProvider)
    with pytest.raises(ValueError, match="unknown provider"):
        build_provider("azure", plan)


def test_piper_command_carries_length_scale_and_volume(tmp_path):
    plan = load_plan(_write_ipa_plan(tmp_path))
    jobs = build_jobs(plan, providers=["piper"])
    job = next(j for j in jobs if j.speed == 0.9 and j.volume == 0.8)

    runner = FakeRunner()
    provider = PiperProvider(
        python=("python3",),
        data_dir=tmp_path / "models",
        runner=runner,
    )
    provider.prepare(plan, [job], tmp_path, lambda _message: None)
    destination = tmp_path / "piper.wav"
    size = provider.synthesize(job, destination)

    download_command = runner.calls[0][0]
    assert "piper.download_voices" in download_command
    assert download_command[-1] == str(tmp_path / "models")

    command, payload = runner.calls[-1]
    assert command[:3] == ["python3", "-m", "piper"]
    assert command[command.index("--length-scale") + 1] == "1.111111"
    assert command[command.index("--volume") + 1] == "0.800"
    assert command[3:5] == ["-m", "en_US-lessac-medium"]
    assert command[command.index("-f") + 1] == str(destination)
    assert payload.decode("utf-8") == "[[ˈheɪ ˈiːvə]]"
    assert size > 0


def test_piper_skips_the_download_when_the_model_exists(tmp_path):
    plan = load_plan(_write_ipa_plan(tmp_path))
    jobs = build_jobs(plan, providers=["piper"])
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    (model_dir / "en_US-lessac-medium.onnx").write_bytes(b"stub")

    runner = FakeRunner()
    provider = PiperProvider(data_dir=model_dir, runner=runner)
    provider.prepare(plan, jobs, tmp_path, lambda _message: None)
    assert runner.calls == []


def test_kokoro_passes_misaki_tokens_and_speed(tmp_path):
    plan = load_plan(_write_ipa_plan(tmp_path))
    jobs = build_jobs(plan, providers=["kokoro"])
    job = next(j for j in jobs if j.speed == 0.9)

    pipeline = FakeKokoroPipeline()
    provider = KokoroProvider(pipeline_factory=lambda: pipeline)
    provider.prepare(plan, [job], tmp_path, lambda _message: None)
    destination = tmp_path / "kokoro.wav"
    size = provider.synthesize(job, destination)

    assert pipeline.calls == [
        {"tokens": "hˈA ˈivə", "voice": "af_bella", "speed": 0.9}
    ]
    assert size > 0
    assert sf.info(str(destination)).samplerate == 24000


def test_kokoro_reports_an_empty_result(tmp_path):
    class EmptyPipeline:
        def generate_from_tokens(self, **_kwargs):
            return []

    plan = load_plan(_write_ipa_plan(tmp_path))
    job = build_jobs(plan, providers=["kokoro"])[0]
    provider = KokoroProvider(pipeline_factory=EmptyPipeline)
    provider.prepare(plan, [job], tmp_path, lambda _message: None)
    with pytest.raises(ProviderError, match="no audio"):
        provider.synthesize(job, tmp_path / "empty.wav")
