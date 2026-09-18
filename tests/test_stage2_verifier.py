from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from dma_kws.config import fbank_kwargs
from dma_kws.configs.schema import FbankConfig
from dma_kws.inference.score_calibration import PositiveAffineCalibrator
from dma_kws.inference.stage2_verifier import Stage2ScoreDetails, Stage2Verifier
from dma_kws.stage1.candidates import KeywordCandidate
from dma_kws.stage2.fbank import FbankExtractor
from dma_kws.stage2.readout import QbyTScoreSpec
from dma_kws.stage2.readout_pooling import QbyTReadoutConfig


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
    verifier._calibrator = PositiveAffineCalibrator(slope=2.0, bias=-1.0)
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


def test_stage2_verifier_passes_versioned_score_spec_to_checkpoint_guard(monkeypatch):
    import torch.nn as nn

    from dma_kws.stage2.readout import QbyTAlignmentSpec, QbyTScoreSpec

    captured: dict = {}

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
            return (
                torch.zeros(speech.size(0)),
                torch.zeros(text.size(0), text.size(1)),
            )

    def _capture_load(model, *_args, **kwargs):
        captured["expected"] = kwargs.get("expected_qbyt_alignment")
        return model

    monkeypatch.setattr(
        "dma_kws.inference.stage2_verifier.build_encoder",
        lambda *_a, **_k: _StubEncoder(),
    )
    monkeypatch.setattr(
        "dma_kws.stage2.model_factory.load_qbyt_class", lambda: _StubQbyT
    )
    monkeypatch.setattr(
        "dma_kws.inference.stage2_verifier._load_model_state", _capture_load
    )
    monkeypatch.setattr(
        "dma_kws.inference.stage2_verifier.run_encoder",
        lambda encoder, feat, feat_lengths, *, policy, mode="eval": encoder(
            feat, feat_lengths
        ),
    )

    stage1_cfg = {
        "encoder_type": "icefall_zipformer",
        "causal": True,
        "downsampling_factor": "1,2,4,8,4,2",
        "cnn_module_kernel": "31,31,15,15,31",
        "stream": {"chunk_size": 16, "left_context_frames": 64},
    }
    Stage2Verifier(
        stage1_cfg=stage1_cfg,
        stage2_cfg={
            "encoder_output_dim": 8,
            "qbyt_readout_version": 6,
            "qbyt_alignment": QbyTAlignmentSpec().as_dict(),
        },
        demo_cfg={},
        fbank_cfg=FbankConfig(dither=0.0, window_type="povey"),
        stage2_ckpt="unused-stage2.pt",
        device=torch.device("cpu"),
        vocab_size=73,
    )
    expected = captured["expected"]
    assert isinstance(expected, QbyTScoreSpec)
    assert expected.version == 6
    assert expected.emission == "query_relative"


def test_stage2_verifier_exposes_raw_logits_and_calibrates_once():
    verifier = Stage2Verifier.__new__(Stage2Verifier)
    verifier._torch = torch
    verifier._device = torch.device("cpu")
    verifier._model = _FakeStage2Model()
    verifier._calibrator = PositiveAffineCalibrator(slope=2.0, bias=-1.0)

    scored = verifier.score_clip_feats_with_logits(
        [torch.zeros(3, 80)],
        [[17, 14, 16]],
    )

    expected_probability = torch.sigmoid(torch.tensor(0.5)).item()
    assert scored == [(pytest.approx(0.75), pytest.approx(expected_probability))]
    assert verifier.score_clip_feats(
        [torch.zeros(3, 80)],
        [[17, 14, 16]],
    ) == [pytest.approx(expected_probability)]


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
    assert scores[0]["qbyt_raw_logit"] == pytest.approx(0.75)
    assert scores[0]["qbyt_score"] == pytest.approx(
        torch.sigmoid(torch.tensor(0.5)).item()
    )


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
    verifier._calibrator = PositiveAffineCalibrator(slope=2.0, bias=-1.0)

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
    assert scores[0]["qbyt_raw_logit"] == pytest.approx(0.75)
    assert scores[0]["qbyt_score"] == pytest.approx(
        torch.sigmoid(torch.tensor(0.5)).item()
    )


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


_ATTENTION_ENCODER_DIM = 48


class _CountingEncodeModel(torch.nn.Module):
    def __init__(self, qbyt):
        super().__init__()
        self.qbyt = qbyt
        self.encode_calls = 0

    def encode_for_qbyt(self, feats, feat_lengths):
        self.encode_calls += 1
        return feats, feat_lengths

    def forward(self, feats, feat_lengths, anchors, anchor_lengths):
        speech, encoder_lens = self.encode_for_qbyt(feats, feat_lengths)
        logits, _extra = self.qbyt(
            speech,
            anchors,
            speech_lengths=encoder_lens,
            text_lengths=anchor_lengths,
        )
        return logits


def _attention_qbyt(*, sink_token=True, readout_mode="eps_softmin"):
    from qbyt.pooling import QbyT

    torch.manual_seed(0)
    return QbyT(
        encoder_output_size=_ATTENTION_ENCODER_DIM,
        num_embeds=73,
        embed_dim=64,
        post_num_layers=2,
        readout_mode=readout_mode,
        sink_token=sink_token,
        text_position="learned",
        audio_position="relative_bias",
    ).eval()


def _attention_score_spec(**kwargs):
    payload = {
        "mode": "eps_softmin",
        "temperature": 1.0,
        "sink_token": True,
        "text_position": "learned",
        "audio_position": "relative_bias",
    }
    payload.update(kwargs)
    return QbyTScoreSpec(version=4, value=QbyTReadoutConfig(**payload))


def _build_attention_verifier(qbyt=None, *, amp=None, threshold=0.5, slope=2.0, bias=-1.0):
    qbyt = _attention_qbyt() if qbyt is None else qbyt
    verifier = Stage2Verifier.__new__(Stage2Verifier)
    verifier._torch = torch
    verifier._device = torch.device("cpu")
    verifier._amp = amp
    verifier._demo_cfg = {"qbyt_threshold": threshold}
    verifier._calibrator = PositiveAffineCalibrator(slope=slope, bias=bias)
    verifier._model = _CountingEncodeModel(qbyt)
    verifier.qbyt_score = _attention_score_spec()
    return verifier


def test_t08_attention_diagnostics_rejects_unsupported_amp_and_spec():
    from dma_kws.inference.qbyt_attention_diagnostics import AttentionCaptureSpec

    feats = [torch.randn(9, _ATTENTION_ENCODER_DIM)]
    keywords = [[3, 4, 5]]
    spec = AttentionCaptureSpec(layers=(0, 1), heads=(0, 1, 2, 3))
    ablations = []

    amp_verifier = _build_attention_verifier(amp="fp16")
    with pytest.raises(RuntimeError, match="FP32|amp"):
        amp_verifier.attention_diagnostics(
            feats, keywords, capture_spec=spec, ablations=ablations
        )

    no_sink = _build_attention_verifier(_attention_qbyt(sink_token=False))
    with pytest.raises(ValueError, match="sink"):
        no_sink.attention_diagnostics(
            feats, keywords, capture_spec=spec, ablations=ablations
        )

    mismatch = _build_attention_verifier(_attention_qbyt(readout_mode="eps_mean"))
    with pytest.raises(ValueError, match="mismatch|eps_softmin"):
        mismatch.attention_diagnostics(
            feats, keywords, capture_spec=spec, ablations=ablations
        )

    version = _build_attention_verifier()
    version.qbyt_score = QbyTScoreSpec(version=7, value=version.qbyt_score.value)
    with pytest.raises(ValueError, match="version"):
        version.attention_diagnostics(
            feats, keywords, capture_spec=spec, ablations=ablations
        )


def test_attention_diagnostics_refuses_train_mode_before_encode():
    from dma_kws.inference.qbyt_attention_diagnostics import AttentionCaptureSpec

    qbyt = _attention_qbyt().train()
    verifier = _build_attention_verifier(qbyt)
    feats = [torch.randn(9, _ATTENTION_ENCODER_DIM)]
    keywords = [[3, 4, 5]]
    spec = AttentionCaptureSpec(layers=(0, 1), heads=(0, 1, 2, 3))
    with pytest.raises(RuntimeError, match="eval"):
        verifier.attention_diagnostics(
            feats, keywords, capture_spec=spec, ablations=[]
        )
    assert verifier._model.encode_calls == 0
    assert qbyt.training is True


def test_t09_attention_diagnostics_reuses_encoder_and_preserves_score_api(monkeypatch):
    from dma_kws.inference.qbyt_attention_diagnostics import (
        AttentionCaptureSpec,
        AttentionParityError,
        SinkAblationSpec,
    )
    from dma_kws.inference.score_calibration import sigmoid

    verifier = _build_attention_verifier(threshold=0.4, slope=2.0, bias=-1.0)
    snapshot = {
        key: tensor.detach().clone()
        for key, tensor in verifier._model.qbyt.state_dict().items()
    }
    feats = [
        torch.randn(8, _ATTENTION_ENCODER_DIM),
        torch.randn(11, _ATTENTION_ENCODER_DIM),
    ]
    keywords = [[3, 5, 7], [4, 6, 8, 9]]
    capture_spec = AttentionCaptureSpec(layers=(0, 1), heads=(0, 1, 2, 3))
    ablations = [
        SinkAblationSpec(name="block_sink_all", blocked_layers=(0, 1)),
        SinkAblationSpec(name="block_sink_layer_0", blocked_layers=(0,)),
    ]

    assert verifier._model.encode_calls == 0
    diagnostics = verifier.attention_diagnostics(
        feats,
        keywords,
        capture_spec=capture_spec,
        ablations=ablations,
    )
    assert verifier._model.encode_calls == 1
    assert len(diagnostics) == 2
    assert diagnostics[0].max_parity_error >= 0.0
    assert math.isfinite(diagnostics[0].max_parity_error)

    threshold = 0.4
    for sample in diagnostics:
        assert sample.threshold == threshold
        assert sample.normal_raw_logit == pytest.approx(sample.normal.raw_logit)
        assert sample.normal_qbyt_score == pytest.approx(sample.normal.qbyt_score)
        expected_score = float(sigmoid(2.0 * sample.normal_raw_logit - 1.0))
        assert sample.normal_qbyt_score == pytest.approx(expected_score)
        assert sample.normal.delta_raw_logit == pytest.approx(0.0)
        assert sample.normal.delta_qbyt_score == pytest.approx(0.0)
        assert torch.equal(
            sample.normal.delta_position_logits,
            torch.zeros_like(sample.normal.delta_position_logits),
        )
        assert sample.normal.detected is (sample.normal_qbyt_score >= threshold)
        assert len(sample.ablations) == 2
        for ablated in sample.ablations:
            expected_ablated = float(sigmoid(2.0 * ablated.raw_logit - 1.0))
            assert ablated.qbyt_score == pytest.approx(expected_ablated)
            assert ablated.delta_raw_logit == pytest.approx(
                ablated.raw_logit - sample.normal_raw_logit
            )
            assert ablated.delta_qbyt_score == pytest.approx(
                ablated.qbyt_score - sample.normal_qbyt_score
            )
            if ablated.raw_logit != sample.normal_raw_logit:
                assert (ablated.delta_raw_logit > 0) is (
                    ablated.raw_logit > sample.normal_raw_logit
                )
            assert ablated.detected is (ablated.qbyt_score >= threshold)
            torch.testing.assert_close(
                ablated.delta_position_logits,
                ablated.trace.position_logits - sample.normal.trace.position_logits,
                atol=0.0,
                rtol=0.0,
            )
        assert sample.ablations[0].spec.name == "block_sink_all"
        assert not torch.allclose(
            torch.as_tensor(sample.ablations[0].raw_logit),
            torch.as_tensor(sample.normal_raw_logit),
        )

    current = verifier._model.qbyt.state_dict()
    assert current.keys() == snapshot.keys()
    for key, tensor in current.items():
        assert torch.equal(tensor, snapshot[key]), key

    scored = verifier.score_clip_feats_with_logits(feats, keywords)
    assert isinstance(scored, list)
    assert len(scored) == 2
    assert all(isinstance(item, tuple) and len(item) == 2 for item in scored)
    calibrated = verifier.score_clip_feats(feats, keywords)
    assert isinstance(calibrated, list)
    assert calibrated == [item[1] for item in scored]
    assert verifier._model.encode_calls == 3

    import dma_kws.inference.qbyt_attention_diagnostics as diagnostics_mod

    real_capture = diagnostics_mod.capture_pooling_attention

    def _shifted_capture(*args, **kwargs):
        traces = real_capture(*args, **kwargs)
        ablation_spec = kwargs.get("ablation_spec")
        if ablation_spec is not None and not ablation_spec.blocked_layers:
            first = traces[0]
            traces[0] = type(first)(
                **{
                    **first.__dict__,
                    "raw_logit": first.raw_logit + 1.0,
                }
            )
        return traces

    monkeypatch.setattr(diagnostics_mod, "capture_pooling_attention", _shifted_capture)
    with pytest.raises(AttentionParityError, match="sample 0"):
        verifier.attention_diagnostics(
            feats,
            keywords,
            capture_spec=capture_spec,
            ablations=ablations,
        )
    assert verifier._model.encode_calls == 4


def _eps_verifier(*, stage2_cfg, slope=2.0, bias=-1.0, seed=0):
    from dma_kws.stage2.readout import resolve_qbyt_score_spec
    from tests.test_stage2_multi_query import (
        _build_qbyt,
        _verifier_from_qbyt,
    )

    qbyt_score = resolve_qbyt_score_spec(stage2_cfg)
    qbyt = _build_qbyt(stage2_cfg, vocab_size=73, seed=seed)
    return _verifier_from_qbyt(
        qbyt,
        slope=slope,
        bias=bias,
        qbyt_score=qbyt_score,
    ), qbyt_score


def test_v41_detailed_api_returns_valid_position_logits_without_padding():
    from tests.test_stage2_multi_query import (
        _POOLING_V41,
        _feats,
        _reconstruct_softmin,
        _token_ids,
    )

    verifier, qbyt_score = _eps_verifier(stage2_cfg=_POOLING_V41, seed=6)
    assert verifier.supports_eps_position_logits is True
    queries = [
        _token_ids("HH", "EY1", "IY1", "V", "AH0"),
        _token_ids("OW1", "K"),
    ]
    feats = _feats(16, 9, seed=2)
    detailed = verifier.score_clip_feats_with_details(feats, queries)
    tuples = verifier.score_clip_feats_with_logits(feats, queries)
    assert len(detailed) == 2
    threshold = 0.5
    for details, (raw, score), query in zip(detailed, tuples, queries):
        assert isinstance(details, Stage2ScoreDetails)
        assert details.qbyt_eps_position_logits is not None
        assert len(details.qbyt_eps_position_logits) == len(query)
        assert all(math.isfinite(value) for value in details.qbyt_eps_position_logits)
        assert details.qbyt_raw_logit == pytest.approx(raw, abs=1e-6, rel=1e-6)
        assert details.qbyt_score == pytest.approx(score, abs=1e-6, rel=1e-6)
        reconstructed = _reconstruct_softmin(
            details.qbyt_eps_position_logits,
            qbyt_score.value.temperature,
        )
        assert reconstructed == pytest.approx(details.qbyt_raw_logit, abs=1e-5, rel=1e-5)
        assert (details.qbyt_score >= threshold) is (score >= threshold)


def test_non_eps_detailed_api_returns_none_positions_and_keeps_totals():
    from tests.test_stage2_multi_query import _POOLING_V4, _V7, _feats, _token_ids

    for stage2_cfg in (_POOLING_V4, _V7):
        verifier, _spec = _eps_verifier(stage2_cfg=stage2_cfg, seed=4)
        assert verifier.supports_eps_position_logits is False
        query = _token_ids("HH", "EY1", "IY1")
        feats = _feats(18, seed=9)
        detailed = verifier.score_clip_feats_with_details(feats, [query])
        tuples = verifier.score_clip_feats_with_logits(feats, [query])
        assert detailed[0].qbyt_eps_position_logits is None
        assert detailed[0].qbyt_raw_logit == pytest.approx(
            tuples[0][0], abs=1e-6, rel=1e-6
        )
        assert detailed[0].qbyt_score == pytest.approx(tuples[0][1], abs=1e-6, rel=1e-6)


def test_eps_model_missing_position_logits_fails_loudly():
    from qbyt.pooling import QbyTReadoutDetails
    from tests.test_stage2_multi_query import _POOLING_V41, _feats, _token_ids

    verifier, _spec = _eps_verifier(stage2_cfg=_POOLING_V41, seed=1)

    def _missing_positions(speech, text, speech_lengths=None, text_lengths=None):
        logits = torch.zeros(speech.size(0))
        text_mask = torch.ones(text.size(0), text.size(1), dtype=torch.bool)
        return logits, torch.zeros_like(text, dtype=torch.float32), QbyTReadoutDetails(
            position_logits=None,
            position_mask=text_mask,
        )

    verifier._model.qbyt.forward_with_readout_details = _missing_positions
    with pytest.raises(RuntimeError, match="position_logits"):
        verifier.score_clip_feats_with_details(
            _feats(8),
            [_token_ids("HH", "EY1")],
        )


def test_non_finite_position_logits_fail_loudly():
    from qbyt.pooling import QbyTReadoutDetails
    from tests.test_stage2_multi_query import _POOLING_V41, _feats, _token_ids

    verifier, _spec = _eps_verifier(stage2_cfg=_POOLING_V41, seed=1)
    query = _token_ids("HH", "EY1")

    def _nan_positions(speech, text, speech_lengths=None, text_lengths=None):
        del speech_lengths, text_lengths
        batch = speech.size(0)
        width = text.size(1)
        logits = torch.zeros(batch)
        positions = torch.zeros(batch, width)
        positions[0, 0] = float("nan")
        mask = torch.ones(batch, width, dtype=torch.bool)
        return logits, torch.zeros(batch, width), QbyTReadoutDetails(
            position_logits=positions,
            position_mask=mask,
        )

    verifier._model.qbyt.forward_with_readout_details = _nan_positions
    with pytest.raises((ValueError, RuntimeError), match="non-finite|finite"):
        verifier.score_clip_feats_with_details(_feats(8), [query])
