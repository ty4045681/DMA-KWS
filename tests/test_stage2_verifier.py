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
    verifier._demo_cfg = {}
    verifier._device = torch.device("cpu")
    verifier._fbank_kwargs = fbank_kwargs(fbank_cfg)
    verifier._fbank_extractor = FbankExtractor(**verifier._fbank_kwargs)
    verifier._model = _FakeStage2Model()
    verifier._min_fbank_frames = 7
    return verifier


def test_stage2_verifier_scores_at_the_deployment_point(monkeypatch):
    """Inference must never inherit the randomized training chunk config."""
    import torch.nn as nn

    calls: list[dict] = []

    class _StubEncoder(nn.Module):
        def output_frames(self, num_input_frames):
            return num_input_frames

        def forward(self, feats, feat_lengths):
            mask = torch.ones(feats.size(0), 1, feats.size(1), dtype=torch.bool)
            return torch.zeros(feats.size(0), feats.size(1), 8), mask

    class _StubQbyT(nn.Module):
        def __init__(self, **kwargs):
            super().__init__()

        def forward(self, speech, text, speech_lengths=None, text_lengths=None):
            del speech_lengths, text_lengths
            return (
                torch.zeros(speech.size(0)),
                torch.zeros(text.size(0), text.size(1)),
            )

    def _spy_run_encoder(encoder, feat, feat_lengths, *, policy, mode="eval"):
        calls.append({"mode": mode, "chunk_size": policy.chunk_size})
        return encoder(feat, feat_lengths)

    monkeypatch.setattr(
        "dma_kws.inference.stage2_verifier.build_encoder", lambda *_a, **_k: _StubEncoder()
    )
    monkeypatch.setattr(
        "dma_kws.stage2.model_factory.load_qbyt_class", lambda: _StubQbyT
    )
    monkeypatch.setattr(
        "dma_kws.inference.stage2_verifier._load_model_state",
        lambda model, *_args, **_kwargs: model,
    )
    monkeypatch.setattr("dma_kws.inference.stage2_verifier.run_encoder", _spy_run_encoder)

    stage1_cfg = {
        "encoder_type": "icefall_zipformer",
        "causal": True,
        "downsampling_factor": "1,2,4,8,4,2",
        "cnn_module_kernel": "31,31,15,15,15,31",
        "stream": {
            "chunk_size": 16,
            "left_context_frames": 64,
            "train_policy": "multi",
            "train_chunk_size": "16,32,64,-1",
            "train_left_context_frames": "64,128,256,-1",
        },
    }
    verifier = Stage2Verifier(
        stage1_cfg=stage1_cfg,
        stage2_cfg={"encoder_output_dim": 8},
        demo_cfg={},
        fbank_cfg=FbankConfig(dither=0.0, window_type="povey"),
        stage2_ckpt="unused-stage2.pt",
        device=torch.device("cpu"),
        vocab_size=73,
    )

    verifier.score_clip_feats([torch.zeros(20, 80)], [[1, 2, 3]])

    assert calls == [{"mode": "eval", "chunk_size": 16}]
    assert "eval=16/64" in verifier.stream_policy.describe()
    assert verifier.amp is None


def test_resolve_inference_amp_accepts_aliases():
    from dma_kws.inference.stage2_verifier import resolve_inference_amp

    assert resolve_inference_amp("off") is None
    assert resolve_inference_amp("fp16") == "fp16"
    assert resolve_inference_amp("float16") == "fp16"
    assert resolve_inference_amp("bf16") == "bf16"
    with pytest.raises(ValueError, match="prep.amp"):
        resolve_inference_amp("int8")


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
    verifier._demo_cfg = {}
    verifier._device = torch.device("cpu")
    verifier._fbank_kwargs = fbank_kwargs(fbank_cfg)
    verifier._fbank_extractor = FbankExtractor(**verifier._fbank_kwargs)
    verifier._model = _FakeStage2Model()
    verifier._min_fbank_frames = 7

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


def test_stage2_verifier_decodes_batched_adapter_ctc(monkeypatch):
    calls: list[dict] = []

    class _FakeAdapter:
        blank_id = 0

        def __call__(self, encoder_out, encoder_mask, *, with_log_probs):
            del encoder_out, encoder_mask
            assert with_log_probs is True
            frame_ids = torch.tensor(
                [
                    [0, 2, 2, 4],
                    [3, 3, 0, 4],
                ]
            )
            log_probs = torch.nn.functional.one_hot(frame_ids, num_classes=5).float()
            return torch.zeros(2, 4, 8), log_probs

    class _FakeModel:
        encoder = object()
        adapter = _FakeAdapter()

    def _fake_run_encoder(encoder, feats, feat_lengths, *, policy, mode):
        calls.append({"encoder": encoder, "policy": policy, "mode": mode})
        positions = torch.arange(feats.size(1), device=feats.device)
        mask = positions.unsqueeze(0) < feat_lengths.unsqueeze(1)
        return torch.zeros(feats.size(0), feats.size(1), 8), mask.unsqueeze(1)

    monkeypatch.setattr(
        "dma_kws.inference.stage2_verifier.run_encoder",
        _fake_run_encoder,
    )
    verifier = Stage2Verifier.__new__(Stage2Verifier)
    verifier._torch = torch
    verifier._device = torch.device("cpu")
    verifier._model = _FakeModel()
    verifier._stream_policy = "deployment-policy"

    hypotheses = verifier.decode_phoneme_feats(
        [torch.zeros(3, 80), torch.zeros(4, 80)]
    )

    assert hypotheses == [[2], [3, 4]]
    assert calls == [
        {
            "encoder": verifier._model.encoder,
            "policy": "deployment-policy",
            "mode": "eval",
        }
    ]


def test_stage2_verifier_per_decode_requires_adapter():
    verifier = Stage2Verifier.__new__(Stage2Verifier)
    verifier._torch = torch
    verifier._model = _FakeStage2Model()

    with pytest.raises(RuntimeError, match="phoneme_adapter.enabled=true"):
        verifier.decode_phoneme_feats([torch.zeros(3, 80)])


def test_stage2_verifier_rejects_empty_keyword_ids():
    verifier = Stage2Verifier.__new__(Stage2Verifier)
    verifier._torch = torch
    verifier._device = torch.device("cpu")
    verifier._model = _FakeStage2Model()

    with pytest.raises(ValueError, match="non-empty"):
        verifier.score_clip_feats([torch.zeros(3, 80)], [[]])
