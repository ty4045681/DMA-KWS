from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from dma_kws.config import fbank_kwargs
from dma_kws.configs.schema import FbankConfig
from dma_kws.inference.stage2_verifier import Stage2Verifier
from dma_kws.stage1.candidates import KeywordCandidate
from dma_kws.stage2.fbank import FbankExtractor


class _FakeStage2Model:
    def __call__(self, feats, feat_lengths, anchors, anchor_lengths):
        del feat_lengths, anchors, anchor_lengths
        return torch.tensor([0.75], dtype=feats.dtype, device=feats.device)


def _build_verifier_for_fbank_test(monkeypatch, captured: dict) -> Stage2Verifier:
    fbank_cfg = FbankConfig(dither=0.0, window_type="povey")

    def _capture_waveform_to_fbank(waveform, *, sample_rate, **kwargs):
        captured["sample_rate"] = sample_rate
        captured["num_samples"] = waveform.size(1)
        captured["kwargs"] = kwargs
        return torch.zeros(40, 80)

    monkeypatch.setattr(
        "dma_kws.inference.stage2_verifier.waveform_to_fbank",
        _capture_waveform_to_fbank,
    )

    verifier = Stage2Verifier.__new__(Stage2Verifier)
    verifier._torch = torch
    verifier._demo_cfg = {"min_stage2_fbank_frames": 7}
    verifier._device = torch.device("cpu")
    verifier._fbank_kwargs = fbank_kwargs(fbank_cfg)
    verifier._fbank_extractor = FbankExtractor(**verifier._fbank_kwargs)
    verifier._model = _FakeStage2Model()
    return verifier


def test_stage2_verifier_uses_waveform_to_fbank(monkeypatch):
    captured: dict = {}
    verifier = _build_verifier_for_fbank_test(monkeypatch, captured)

    waveform = torch.randn(1, 16000)
    candidates = [
        KeywordCandidate(
            start_sec=0.1,
            end_sec=0.6,
            stage1_score=0.4,
            phonemes=[],
        )
    ]

    scores = verifier.verify_candidates(waveform, 16000, [17, 14, 16], candidates)

    assert captured["sample_rate"] == 16000
    assert captured["kwargs"]["window_type"] == "povey"
    assert captured["kwargs"]["dither"] == 0.0
    assert scores[0]["qbyt_score"] == pytest.approx(0.75)


def test_stage2_verifier_resamples_before_candidate_slicing(monkeypatch):
    captured: dict = {}
    fbank_cfg = FbankConfig(
        backend="lhotse_fbank",
        target_sample_rate=16000,
        dither=0.0,
        snip_edges=False,
        high_freq=-400.0,
    )

    def _capture_waveform_to_fbank(waveform, *, sample_rate, **kwargs):
        captured["sample_rate"] = sample_rate
        captured["num_samples"] = waveform.size(1)
        captured["kwargs"] = kwargs
        return torch.zeros(50, 80)

    monkeypatch.setattr(
        "dma_kws.inference.stage2_verifier.waveform_to_fbank",
        _capture_waveform_to_fbank,
    )
    verifier = Stage2Verifier.__new__(Stage2Verifier)
    verifier._torch = torch
    verifier._demo_cfg = {"min_stage2_fbank_frames": 7}
    verifier._device = torch.device("cpu")
    verifier._fbank_kwargs = fbank_kwargs(fbank_cfg)
    verifier._fbank_extractor = FbankExtractor(**verifier._fbank_kwargs)
    verifier._model = _FakeStage2Model()

    waveform = torch.randn(1, 8000)
    candidates = [
        KeywordCandidate(
            start_sec=0.25,
            end_sec=0.75,
            stage1_score=0.4,
            phonemes=[],
        )
    ]

    scores = verifier.verify_candidates(waveform, 8000, [17, 14, 16], candidates)

    assert captured["sample_rate"] == 16000
    assert captured["num_samples"] == 8000
    assert captured["kwargs"]["snip_edges"] is False
    assert scores[0]["qbyt_score"] == pytest.approx(0.75)
