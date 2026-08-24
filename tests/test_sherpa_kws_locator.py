from __future__ import annotations

from pathlib import Path

import pytest

from dma_kws.inference.locators.sherpa_kws import SherpaOnnxKwsLocator


class FakeStream:
    def __init__(self) -> None:
        self.waveforms: list[tuple[int, object]] = []
        self.finished = False

    def accept_waveform(self, sample_rate: int, samples) -> None:
        self.waveforms.append((sample_rate, samples))

    def input_finished(self) -> None:
        self.finished = True


class FakeKeywordSpotter:
    last_kwargs: dict | None = None
    last_stream_keywords: object = "unset"

    def __init__(self, **kwargs) -> None:
        FakeKeywordSpotter.last_kwargs = dict(kwargs)
        self.stream = FakeStream()
        self._ready_calls = 0

    def create_stream(self, keywords=None):
        FakeKeywordSpotter.last_stream_keywords = keywords
        return self.stream

    def is_ready(self, stream) -> bool:
        del stream
        self._ready_calls += 1
        return self._ready_calls == 1

    def decode_stream(self, stream) -> None:
        del stream

    def get_result(self, stream) -> str:
        del stream
        return ""

    def timestamps(self, stream):
        del stream
        return []

    def reset_stream(self, stream) -> None:
        del stream


def _locator_config(tmp_path: Path, *, keywords_file: str | None) -> dict:
    tokens = tmp_path / "tokens.txt"
    encoder = tmp_path / "encoder.onnx"
    decoder = tmp_path / "decoder.onnx"
    joiner = tmp_path / "joiner.onnx"
    for path in (tokens, encoder, decoder, joiner):
        path.write_text("", encoding="utf-8")
    locator = {
        "tokens": str(tokens),
        "encoder": str(encoder),
        "decoder": str(decoder),
        "joiner": str(joiner),
        "keywords_threshold": 0.25,
    }
    if keywords_file is not None:
        locator["keywords_file"] = keywords_file
    return {"locator": locator, "demo": {"stage1_candidate_margin_sec": 0.15}}


def test_sherpa_locator_requires_keywords_file(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "dma_kws.inference.locators.sherpa_kws._import_sherpa_onnx",
        lambda: type("Mod", (), {"KeywordSpotter": FakeKeywordSpotter}),
    )
    with pytest.raises(ValueError, match="locator.keywords_file"):
        SherpaOnnxKwsLocator(config=_locator_config(tmp_path, keywords_file=None))


def test_sherpa_locator_requires_existing_keywords_file(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "dma_kws.inference.locators.sherpa_kws._import_sherpa_onnx",
        lambda: type("Mod", (), {"KeywordSpotter": FakeKeywordSpotter}),
    )
    missing = tmp_path / "missing_keywords.txt"
    with pytest.raises(ValueError, match="not found"):
        SherpaOnnxKwsLocator(
            config=_locator_config(tmp_path, keywords_file=str(missing))
        )


def test_sherpa_locator_uses_keywords_file_stream_without_inline_keyword(
    tmp_path, monkeypatch
):
    keywords = tmp_path / "keywords.txt"
    keywords.write_text("h e y e v a @hey eva\n", encoding="utf-8")
    monkeypatch.setattr(
        "dma_kws.inference.locators.sherpa_kws._import_sherpa_onnx",
        lambda: type("Mod", (), {"KeywordSpotter": FakeKeywordSpotter}),
    )

    import numpy as np
    import torch

    captured = {}

    def fake_load_audio(path, *, sample_rate: int):
        captured["path"] = path
        captured["sample_rate"] = sample_rate
        return torch.zeros(1, sample_rate), sample_rate

    monkeypatch.setattr(
        "dma_kws.inference.locators.sherpa_kws.load_audio",
        fake_load_audio,
    )

    locator = SherpaOnnxKwsLocator(
        config=_locator_config(tmp_path, keywords_file=str(keywords))
    )
    assert FakeKeywordSpotter.last_kwargs is not None
    assert Path(FakeKeywordSpotter.last_kwargs["keywords_file"]) == keywords.resolve()

    locator.locate("/tmp/audio.wav", "hey eva", keyword_phonemes=["HH", "EY1"])
    assert captured["sample_rate"] == 16000
    assert FakeKeywordSpotter.last_stream_keywords is None
    assert locator._kws.stream.finished is True
    assert len(locator._kws.stream.waveforms) == 2
    assert locator._kws.stream.waveforms[0][0] == 16000
    assert isinstance(locator._kws.stream.waveforms[0][1], np.ndarray)
