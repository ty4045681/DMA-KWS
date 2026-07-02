from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from dma_kws.configs.schema import FbankConfig
from dma_kws.inference.stage2_verifier import Stage2Verifier
from dma_kws.stage1.candidates import KeywordCandidate


def _fbank_kwargs(cfg: FbankConfig) -> dict:
    return {
        "num_mel_bins": cfg.num_mel_bins,
        "frame_length": cfg.frame_length,
        "frame_shift": cfg.frame_shift,
        "dither": cfg.dither,
        "window_type": cfg.window_type,
    }


class _FakeStage2Model:
    def __call__(self, feats, feat_lengths, anchors, anchor_lengths):
        del feat_lengths, anchors, anchor_lengths
        return torch.tensor([0.75], dtype=feats.dtype, device=feats.device)


def _build_verifier_for_fbank_test(monkeypatch, captured: dict) -> Stage2Verifier:
    fbank_cfg = FbankConfig(dither=0.0, window_type="povey")

    def _capture_waveform_to_fbank(waveform, *, sample_rate, **kwargs):
        captured["sample_rate"] = sample_rate
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
    verifier._fbank_kwargs = _fbank_kwargs(fbank_cfg)
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
