from __future__ import annotations

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
        self.calls: list[tuple[str, str]] = []

    def locate(self, audio_path: str, keyword: str):
        self.calls.append((audio_path, keyword))
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


def _fake_g2p():
    return lambda text: text.upper().split()


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
            lambda _g2p, text: text.upper().split(),
        )
    tokenizer = load_char_tokenizer("data/dict/lang_char.txt", split_with_space=" ")
    return TwoStageKWSPipeline(
        locator=locator,
        verifier=verifier,
        tokenizer=tokenizer,
        demo_cfg={"qbyt_threshold": threshold, "min_stage2_fbank_frames": 7},
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
    assert pipeline._locator.calls == [("/tmp/audio.wav", "hello")]


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
