from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from dma_kws.inference.locators.sherpa_kws import (
    SherpaOnnxKwsLocator,
    span_from_sherpa_hit,
)


class FakeStream:
    def __init__(self) -> None:
        self.waveforms: list[tuple[int, np.ndarray]] = []
        self.finished = False
        self.accepted_samples = 0
        self.current_frontier_samples = 0
        self.pending_decodes = 0

    def accept_waveform(self, sample_rate: int, samples) -> None:
        values = np.asarray(samples, dtype=np.float32).copy()
        self.waveforms.append((sample_rate, values))
        self.accepted_samples += int(values.size)
        self.pending_decodes += 1

    def input_finished(self) -> None:
        self.finished = True


class PlannedFakeKeywordSpotter:
    def __init__(self, *, hit_frontiers_sec: tuple[float, ...], **kwargs) -> None:
        self.kwargs = dict(kwargs)
        self.stream = FakeStream()
        self.hit_frontiers_sec = hit_frontiers_sec
        self.next_hit = 0
        self.decode_calls = 0
        self.result_calls = 0
        self.timestamps_calls = 0
        self.reset_count = 0
        self.hit_finished_states: list[bool] = []
        self.stream_keywords: object = "unset"

    def create_stream(self, keywords=None):
        self.stream_keywords = keywords
        return self.stream

    def is_ready(self, stream) -> bool:
        return stream.pending_decodes > 0

    def decode_stream(self, stream) -> None:
        assert stream.pending_decodes > 0
        stream.pending_decodes -= 1
        stream.current_frontier_samples = stream.accepted_samples
        self.decode_calls += 1

    def get_result(self, stream) -> str:
        self.result_calls += 1
        if self.next_hit >= len(self.hit_frontiers_sec):
            return ""
        sample_rate = stream.waveforms[-1][0]
        hit_sample = round(self.hit_frontiers_sec[self.next_hit] * sample_rate)
        if stream.current_frontier_samples < hit_sample:
            return ""
        self.next_hit += 1
        self.hit_finished_states.append(stream.finished)
        return "hey eva"

    def timestamps(self, stream):
        del stream
        self.timestamps_calls += 1
        raise AssertionError("a sherpa result must not be read a second time")

    def reset_stream(self, stream) -> None:
        del stream
        self.reset_count += 1


def _fake_sherpa_module(*, hit_frontiers_sec: tuple[float, ...] = ()):
    def build_spotter(**kwargs):
        return PlannedFakeKeywordSpotter(
            hit_frontiers_sec=hit_frontiers_sec,
            **kwargs,
        )

    return SimpleNamespace(KeywordSpotter=build_spotter)


def _locator_config(
    tmp_path: Path,
    *,
    keywords_file: str | None,
    **locator_overrides,
) -> dict:
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
        **locator_overrides,
    }
    if keywords_file is not None:
        locator["keywords_file"] = keywords_file
    return {
        "locator": locator,
        "stage1": {"sample_rate": 16000},
        "demo": {"stage1_candidate_margin_sec": 0.15},
    }


def _keywords_file(tmp_path: Path) -> Path:
    path = tmp_path / "keywords.txt"
    path.write_text("h e y e v a @hey eva\n", encoding="utf-8")
    return path


def _patch_audio(monkeypatch, waveform: torch.Tensor, *, sample_rate: int = 16000):
    captured = {}

    def fake_load_audio(path, *, sample_rate: int):
        captured["path"] = path
        captured["sample_rate"] = sample_rate
        return waveform, sample_rate

    monkeypatch.setattr(
        "dma_kws.inference.locators.sherpa_kws.load_audio",
        fake_load_audio,
    )
    return captured


def test_span_from_sherpa_hit_uses_absolute_decode_frontier():
    assert span_from_sherpa_hit(
        audio_duration_sec=90.0,
        wakeup_window_sec=1.5,
        decoded_sec=12.0,
    ) == (10.5, 12.0)
    assert span_from_sherpa_hit(
        audio_duration_sec=90.0,
        wakeup_window_sec=1.5,
        decoded_sec=100.0,
    ) == (88.5, 90.0)


def test_sherpa_locator_requires_keywords_file(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "dma_kws.inference.locators.sherpa_kws._import_sherpa_onnx",
        _fake_sherpa_module,
    )
    with pytest.raises(ValueError, match="locator.keywords_file"):
        SherpaOnnxKwsLocator(config=_locator_config(tmp_path, keywords_file=None))


def test_sherpa_locator_requires_existing_keywords_file(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "dma_kws.inference.locators.sherpa_kws._import_sherpa_onnx",
        _fake_sherpa_module,
    )
    missing = tmp_path / "missing_keywords.txt"
    with pytest.raises(ValueError, match="not found"):
        SherpaOnnxKwsLocator(
            config=_locator_config(tmp_path, keywords_file=str(missing))
        )


@pytest.mark.parametrize("stream_chunk_sec", [0.0, -0.1, float("inf")])
def test_sherpa_locator_rejects_invalid_stream_chunk(
    tmp_path, monkeypatch, stream_chunk_sec
):
    monkeypatch.setattr(
        "dma_kws.inference.locators.sherpa_kws._import_sherpa_onnx",
        _fake_sherpa_module,
    )
    with pytest.raises(ValueError, match="locator.stream_chunk_sec"):
        SherpaOnnxKwsLocator(
            config=_locator_config(
                tmp_path,
                keywords_file=str(_keywords_file(tmp_path)),
                stream_chunk_sec=stream_chunk_sec,
            )
        )


@pytest.mark.parametrize(
    ("stream_chunk_sec", "full_chunk_samples"),
    [(0.1, 1600), (0.25, 4000)],
)
def test_locator_streams_chunks_and_localizes_middle_hit_without_timestamps(
    tmp_path,
    monkeypatch,
    stream_chunk_sec,
    full_chunk_samples,
):
    sample_rate = 16000
    num_samples = round(4.05 * sample_rate)
    waveform = torch.arange(num_samples, dtype=torch.float32).unsqueeze(0)
    captured = _patch_audio(monkeypatch, waveform, sample_rate=sample_rate)
    monkeypatch.setattr(
        "dma_kws.inference.locators.sherpa_kws._import_sherpa_onnx",
        lambda: _fake_sherpa_module(hit_frontiers_sec=(2.0,)),
    )

    keywords = _keywords_file(tmp_path)
    locator = SherpaOnnxKwsLocator(
        config=_locator_config(
            tmp_path,
            keywords_file=str(keywords),
            stream_chunk_sec=stream_chunk_sec,
        )
    )
    candidates = locator.locate(
        "/tmp/audio.wav", "hey eva", keyword_phonemes=["HH", "EY1"]
    )

    assert captured == {"path": "/tmp/audio.wav", "sample_rate": sample_rate}
    assert Path(locator._kws.kwargs["keywords_file"]) == keywords.resolve()
    assert locator._kws.stream_keywords is None

    real_chunks = locator._kws.stream.waveforms[:-1]
    assert [values.size for _, values in real_chunks] == [
        full_chunk_samples
    ] * (num_samples // full_chunk_samples) + [num_samples % full_chunk_samples]
    np.testing.assert_array_equal(
        np.concatenate([values for _, values in real_chunks]),
        waveform.squeeze(0).numpy(),
    )
    tail_sample_rate, tail = locator._kws.stream.waveforms[-1]
    assert tail_sample_rate == sample_rate
    assert tail.size == round(0.66 * sample_rate)
    assert np.count_nonzero(tail) == 0
    assert locator._kws.stream.finished is True

    assert len(candidates) == 1
    assert candidates[0].start_sec == pytest.approx(0.35)
    assert candidates[0].end_sec == pytest.approx(2.15)
    assert locator._kws.hit_finished_states == [False]
    assert locator._kws.timestamps_calls == 0
    assert locator._kws.result_calls == locator._kws.decode_calls


def test_locator_preserves_multiple_middle_hits_without_timestamps(
    tmp_path, monkeypatch
):
    sample_rate = 16000
    waveform = torch.zeros(1, 5 * sample_rate)
    _patch_audio(monkeypatch, waveform, sample_rate=sample_rate)
    monkeypatch.setattr(
        "dma_kws.inference.locators.sherpa_kws._import_sherpa_onnx",
        lambda: _fake_sherpa_module(hit_frontiers_sec=(2.0, 4.0)),
    )
    locator = SherpaOnnxKwsLocator(
        config=_locator_config(
            tmp_path,
            keywords_file=str(_keywords_file(tmp_path)),
            stream_chunk_sec=0.1,
        )
    )

    candidates = locator.locate("/tmp/audio.wav", "hey eva")

    assert [(item.start_sec, item.end_sec) for item in candidates] == pytest.approx(
        [(0.35, 2.15), (2.35, 4.15)]
    )
    assert locator._kws.reset_count == 2
    assert locator._kws.hit_finished_states == [False, False]
    assert locator._kws.timestamps_calls == 0
    assert locator._kws.result_calls == locator._kws.decode_calls


def test_tail_padding_does_not_advance_candidate_clock(tmp_path, monkeypatch):
    sample_rate = 16000
    waveform = torch.zeros(1, sample_rate)
    _patch_audio(monkeypatch, waveform, sample_rate=sample_rate)
    monkeypatch.setattr(
        "dma_kws.inference.locators.sherpa_kws._import_sherpa_onnx",
        lambda: _fake_sherpa_module(hit_frontiers_sec=(1.1,)),
    )
    locator = SherpaOnnxKwsLocator(
        config=_locator_config(
            tmp_path,
            keywords_file=str(_keywords_file(tmp_path)),
        )
    )

    candidates = locator.locate("/tmp/audio.wav", "hey eva")

    assert len(candidates) == 1
    assert candidates[0].start_sec == 0.0
    assert candidates[0].end_sec == 1.0
    assert locator._kws.hit_finished_states == [True]
