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


def test_stage2_verifier_scores_at_the_deployment_point(monkeypatch, tmp_path):
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
            del text, speech_lengths, text_lengths
            return torch.zeros(speech.size(0)), None

    def _spy_run_encoder(encoder, feat, feat_lengths, *, policy, mode="eval"):
        calls.append({"mode": mode, "chunk_size": policy.chunk_size})
        return encoder(feat, feat_lengths)

    monkeypatch.setattr(
        "dma_kws.inference.stage2_verifier.build_encoder", lambda *_a, **_k: _StubEncoder()
    )
    monkeypatch.setattr(
        "dma_kws.inference.stage2_verifier.load_qbyt_class", lambda: _StubQbyT
    )
    monkeypatch.setattr("dma_kws.inference.stage2_verifier.run_encoder", _spy_run_encoder)

    ckpt_path = tmp_path / "stage2.pt"
    torch.save({"model_state_dict": {}}, ckpt_path)

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
        stage2_ckpt=str(ckpt_path),
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


def test_stage2_verifier_detailed_scores_preserve_raw_logits():
    class _DetailedModel:
        def __call__(self, feats, feat_lengths, anchors, anchor_lengths):
            del feats, feat_lengths, anchors, anchor_lengths
            return torch.sigmoid(torch.tensor([8.0, -8.0]))

        def forward_logits(self, feats, feat_lengths, anchors, anchor_lengths):
            del feats, feat_lengths, anchors, anchor_lengths
            return (
                torch.tensor([8.0, -8.0]),
                torch.tensor([1.5, 0.0]),
                torch.tensor([True, False]),
            )

    verifier = Stage2Verifier.__new__(Stage2Verifier)
    verifier._torch = torch
    verifier._device = torch.device("cpu")
    verifier._model = _DetailedModel()

    details = verifier.score_clip_feats_detailed(
        [torch.zeros(3, 80), torch.zeros(5, 80)],
        [[1, 2], []],
    )

    assert details[0]["qbyt_logit"] == pytest.approx(8.0)
    assert details[0]["qbyt_score"] == pytest.approx(float(torch.sigmoid(torch.tensor(8.0))))
    assert details[0]["completion_logit"] == pytest.approx(1.5)
    assert details[0]["completion_score"] == pytest.approx(
        float(torch.sigmoid(torch.tensor(1.5)))
    )
    assert details[1]["qbyt_logit"] == pytest.approx(-8.0)
    assert details[1]["completion_logit"] is None
    assert details[1]["completion_score"] is None
    assert "eps_position_logits" not in details[0]

    legacy_scores = verifier.score_clip_feats(
        [torch.zeros(3, 80), torch.zeros(5, 80)],
        [[1, 2], []],
    )
    assert legacy_scores == pytest.approx(
        [float(torch.sigmoid(torch.tensor(8.0))), float(torch.sigmoid(torch.tensor(-8.0)))]
    )


def test_stage2_verifier_detailed_scores_export_trimmed_eps_positions():
    class _ReadoutDetails:
        position_logits = torch.tensor(
            [
                [1.0, 3.0, 0.0],
                [-1.0, 0.0, 0.0],
            ]
        )
        position_mask = torch.tensor(
            [
                [True, True, False],
                [True, False, False],
            ]
        )

    class _DetailedModel:
        def forward_logits_with_readout_details(
            self, feats, feat_lengths, anchors, anchor_lengths
        ):
            del feats, feat_lengths, anchors, anchor_lengths
            return (
                torch.tensor([2.0, -1.0]),
                torch.tensor([1.5, -0.5]),
                torch.tensor([True, True]),
                _ReadoutDetails(),
            )

    verifier = Stage2Verifier.__new__(Stage2Verifier)
    verifier._torch = torch
    verifier._device = torch.device("cpu")
    verifier._model = _DetailedModel()

    details = verifier.score_clip_feats_detailed(
        [torch.zeros(3, 80), torch.zeros(5, 80)],
        [[1, 2], [3]],
        include_eps_positions=True,
    )

    assert details[0]["eps_position_logits"] == pytest.approx([1.0, 3.0])
    assert details[1]["eps_position_logits"] == pytest.approx([-1.0])
    assert details[0]["qbyt_logit"] == pytest.approx(2.0)
    assert details[1]["qbyt_logit"] == pytest.approx(-1.0)


def test_stage2_verifier_validates_softmin_eps_positions():
    temperature = 0.5
    raw_position_logits = torch.tensor([[1.0, 3.0]])
    expected_logit = -temperature * (
        torch.logsumexp(-raw_position_logits[0] / temperature, dim=0)
        - torch.log(torch.tensor(2.0))
    )

    class _ReadoutDetails:
        position_logits = raw_position_logits
        position_mask = torch.tensor([[True, True]])

    class _DetailedModel:
        def forward_logits_with_readout_details(
            self, feats, feat_lengths, anchors, anchor_lengths
        ):
            del feats, feat_lengths, anchors, anchor_lengths
            return (
                expected_logit.unsqueeze(0),
                torch.tensor([0.0]),
                torch.tensor([True]),
                _ReadoutDetails(),
            )

    verifier = Stage2Verifier.__new__(Stage2Verifier)
    verifier._torch = torch
    verifier._device = torch.device("cpu")
    verifier._model = _DetailedModel()
    verifier.qbyt_readout_mode = "eps_softmin"
    verifier.qbyt_readout_temperature = temperature

    details = verifier.score_clip_feats_detailed(
        [torch.zeros(3, 80)],
        [[1, 2]],
        include_eps_positions=True,
    )

    assert details[0]["qbyt_logit"] == pytest.approx(float(expected_logit))
    assert details[0]["eps_position_logits"] == pytest.approx([1.0, 3.0])


def test_stage2_verifier_eps_position_export_is_none_for_gru_readout():
    class _ReadoutDetails:
        position_logits = None
        position_mask = torch.tensor([[True, True]])

    class _DetailedModel:
        def forward_logits_with_readout_details(
            self, feats, feat_lengths, anchors, anchor_lengths
        ):
            del feats, feat_lengths, anchors, anchor_lengths
            return (
                torch.tensor([0.5]),
                torch.tensor([0.25]),
                torch.tensor([True]),
                _ReadoutDetails(),
            )

    verifier = Stage2Verifier.__new__(Stage2Verifier)
    verifier._torch = torch
    verifier._device = torch.device("cpu")
    verifier._model = _DetailedModel()

    details = verifier.score_clip_feats_detailed(
        [torch.zeros(3, 80)],
        [[1, 2]],
        include_eps_positions=True,
    )

    assert details[0]["eps_position_logits"] is None


def test_stage2_verifier_exports_trimmed_sequence_positions():
    class _ReadoutDetails:
        position_logits = torch.tensor(
            [
                [0.25, 0.5, 0.75],
                [-0.25, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ]
        )
        position_mask = torch.tensor(
            [
                [True, True, True],
                [True, False, False],
                [False, False, False],
            ]
        )

    class _DetailedModel:
        def forward_logits_with_position_details(
            self, feats, feat_lengths, anchors, anchor_lengths
        ):
            del feats, feat_lengths, anchors, anchor_lengths
            return (
                torch.tensor([0.5, -0.25, 0.0]),
                torch.tensor([-1.0, -2.0, 0.0]),
                torch.tensor([True, True, False]),
                torch.tensor(
                    [
                        [2.0, 0.0, -1.0],
                        [-2.0, float("nan"), float("nan")],
                        [float("nan"), float("nan"), float("nan")],
                    ]
                ),
                _ReadoutDetails(),
            )

    verifier = Stage2Verifier.__new__(Stage2Verifier)
    verifier._torch = torch
    verifier._device = torch.device("cpu")
    verifier._model = _DetailedModel()

    details = verifier.score_clip_feats_detailed(
        [torch.zeros(3, 80), torch.zeros(4, 80), torch.zeros(5, 80)],
        [[1, 2, 3], [4], []],
        include_eps_positions=True,
        include_seq_positions=True,
    )

    assert details[0]["seq_position_logits"] == pytest.approx([2.0, 0.0, -1.0])
    assert details[1]["seq_position_logits"] == pytest.approx([-2.0])
    assert details[2]["seq_position_logits"] == []
    assert details[0]["completion_logit"] == pytest.approx(
        details[0]["seq_position_logits"][-1]
    )
    assert details[1]["completion_logit"] == pytest.approx(
        details[1]["seq_position_logits"][-1]
    )
    assert details[2]["completion_logit"] is None
    assert details[0]["eps_position_logits"] == pytest.approx([0.25, 0.5, 0.75])
    assert details[1]["eps_position_logits"] == pytest.approx([-0.25])
    assert details[2]["eps_position_logits"] == []
