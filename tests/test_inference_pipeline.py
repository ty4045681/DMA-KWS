from __future__ import annotations

import pytest
import torch

from dma_kws.inference.pipeline import TwoStageKWSPipeline
from dma_kws.stage1.candidates import KeywordCandidate
from dma_kws.tokenizer import load_char_tokenizer


def _fake_audio_loader(path: str, *, sample_rate: int):
    del path
    waveform = torch.zeros(1, sample_rate)
    return waveform, sample_rate


class FakeLocator:
    def __init__(self, candidates: list[KeywordCandidate] | None = None) -> None:
        if candidates is None:
            candidates = [
                KeywordCandidate(
                    start_sec=0.5,
                    end_sec=1.0,
                    stage1_score=-1.0,
                    phonemes=["HH", "AH", "L", "OW"],
                )
            ]
        self._candidates = candidates
        self.calls: list[tuple[str, str, tuple[str, ...] | None]] = []

    def locate(
        self,
        audio_path: str,
        keyword: str,
        keyword_phonemes=None,
    ):
        phonemes = None if keyword_phonemes is None else tuple(keyword_phonemes)
        self.calls.append((audio_path, keyword, phonemes))
        return list(self._candidates)


class FakeVerifier:
    def __init__(self, scores: list[float] | None = None) -> None:
        self._scores = [0.9] if scores is None else scores
        self.calls: list[tuple] = []

    def verify_candidates(self, waveform, sample_rate: int, keyword_ids: list[int], candidates):
        self.calls.append((waveform, sample_rate, keyword_ids, candidates))
        output = []
        for candidate, score in zip(candidates, self._scores):
            output.append(
                {
                    "start_sec": candidate.start_sec,
                    "end_sec": candidate.end_sec,
                    "stage1_score": candidate.stage1_score,
                    "qbyt_score": score,
                }
            )
        return output


def _fake_phonemes(text: str) -> list[str]:
    """Mimic g2p_en: upper-case ARPAbet with stress digits on vowels."""
    phones = {"hey eva": ["HH", "EY1", "IY1", "V", "AH0"]}
    return phones.get(text.lower(), text.upper().split())


def _fake_g2p():
    return _fake_phonemes


def _build_pipeline(
    *,
    threshold: float,
    locator: FakeLocator,
    verifier: FakeVerifier,
    monkeypatch=None,
) -> TwoStageKWSPipeline:
    if monkeypatch is not None:
        monkeypatch.setattr("dma_kws.inference.pipeline.make_g2p", _fake_g2p)
        monkeypatch.setattr(
            "dma_kws.inference.pipeline.text_to_phonemes",
            lambda _g2p, text: _fake_phonemes(text),
        )
    tokenizer = load_char_tokenizer("data/dict/lang_char.txt", split_with_space=" ")
    return TwoStageKWSPipeline(
        locator=locator,
        verifier=verifier,
        tokenizer=tokenizer,
        demo_cfg={"qbyt_threshold": threshold},
        sample_rate=16000,
    )


def test_pipeline_detects_when_verifier_score_above_threshold(monkeypatch):
    monkeypatch.setattr("dma_kws.inference.pipeline.load_audio", _fake_audio_loader)
    pipeline = _build_pipeline(
        threshold=0.6,
        locator=FakeLocator(),
        verifier=FakeVerifier(scores=[0.85]),
        monkeypatch=monkeypatch,
    )

    result = pipeline.run("/tmp/audio.wav", "hello")

    assert result["detected"] is True
    assert result["best_qbyt_score"] == 0.85
    assert result["stage2_scores"][0]["qbyt_score"] == 0.85
    assert pipeline._locator.calls == [
        ("/tmp/audio.wav", "hello", ("HELLO",))
    ]


def test_pipeline_not_detected_when_scores_below_threshold(monkeypatch):
    monkeypatch.setattr("dma_kws.inference.pipeline.load_audio", _fake_audio_loader)
    pipeline = _build_pipeline(
        threshold=0.8,
        locator=FakeLocator(),
        verifier=FakeVerifier(scores=[0.2]),
        monkeypatch=monkeypatch,
    )

    result = pipeline.run("/tmp/audio.wav", "hello")

    assert result["detected"] is False
    assert result["best_qbyt_score"] == 0.2


def test_pipeline_no_candidates_yields_zero_score_and_not_detected(monkeypatch):
    monkeypatch.setattr("dma_kws.inference.pipeline.load_audio", _fake_audio_loader)
    pipeline = _build_pipeline(
        threshold=0.5,
        locator=FakeLocator(candidates=[]),
        verifier=FakeVerifier(scores=[]),
        monkeypatch=monkeypatch,
    )

    result = pipeline.run("/tmp/audio.wav", "hello")

    assert result["detected"] is False
    assert result["best_qbyt_score"] == 0.0
    assert result["stage2_scores"] == []
    assert pipeline._verifier.calls == []


def test_pipeline_uses_g2p_unless_phonemes_overridden(monkeypatch):
    monkeypatch.setattr("dma_kws.inference.pipeline.load_audio", _fake_audio_loader)
    locator = FakeLocator()
    verifier = FakeVerifier(scores=[0.4])
    pipeline = _build_pipeline(
        threshold=0.5,
        locator=locator,
        verifier=verifier,
        monkeypatch=monkeypatch,
    )

    result = pipeline.run("/tmp/audio.wav", "hey eva")
    assert result["keyword_phonemes"] == ["HH", "EY1", "IY1", "V", "AH0"]
    default_ids = list(verifier.calls[0][2])
    assert locator.calls[0][2] == ("HH", "EY1", "IY1", "V", "AH0")

    override = pipeline.run(
        "/tmp/audio.wav",
        "hey eva",
        keyword_phonemes="HH EY1 EY1 V AH0",
    )
    assert override["keyword_phonemes"] == ["HH", "EY1", "EY1", "V", "AH0"]
    assert list(verifier.calls[1][2]) != default_ids
    assert locator.calls[1][2] == ("HH", "EY1", "EY1", "V", "AH0")


def test_phoneme_ctc_locator_uses_explicit_enrollment(monkeypatch):
    from dma_kws.inference.locators import phoneme_ctc as phoneme_ctc_mod

    g2p_calls: list[str] = []
    recorded: dict[str, object] = {}

    class Dummy:
        pass

    locator = Dummy()
    locator._torch = object()
    locator._tokenizer = load_char_tokenizer(
        "data/dict/lang_char.txt", split_with_space=" "
    )
    locator._g2p = object()
    locator._sample_rate = 16000
    locator._num_mel_bins = 80
    locator._device = "cpu"
    locator._demo_cfg = {}
    locator.last_keyword_phonemes = []
    locator.last_decoded_phonemes = []

    monkeypatch.setattr(
        phoneme_ctc_mod,
        "text_to_phonemes",
        lambda _g2p, text: g2p_calls.append(text) or ["SHOULD", "NOT", "USE"],
    )

    def fake_tokenize(tokenizer, text):
        recorded["tokenized"] = text
        return [1, 2, 3]

    monkeypatch.setattr(phoneme_ctc_mod, "tokenize_phoneme_string", fake_tokenize)
    monkeypatch.setattr(
        phoneme_ctc_mod,
        "load_audio",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("stop")),
    )

    with pytest.raises(RuntimeError, match="stop"):
        phoneme_ctc_mod.PhonemeCtcLocator.locate(
            locator,
            "/tmp/audio.wav",
            "hey eva",
            keyword_phonemes=["HH", "EY1", "EY1", "V", "AH0"],
        )
    assert locator.last_keyword_phonemes == ["HH", "EY1", "EY1", "V", "AH0"]
    assert recorded["tokenized"] == "HH EY1 EY1 V AH0"
    assert g2p_calls == []


def test_pipeline_rejects_unsupported_phoneme_override(monkeypatch):
    pipeline = _build_pipeline(
        threshold=0.5,
        locator=FakeLocator(),
        verifier=FakeVerifier(scores=[]),
        monkeypatch=monkeypatch,
    )

    with pytest.raises(ValueError, match="unsupported phonemes"):
        pipeline.resolve_keyword_phonemes("hey eva", "HH NOT_A_PHONE")
