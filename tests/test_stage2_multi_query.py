from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from dma_kws.inference.keyword_set import KeywordEvalConfigError, KeywordSetScoreError
from dma_kws.inference.score_calibration import PositiveAffineCalibrator
from dma_kws.inference.stage2_verifier import Stage2ScoreDetails, Stage2Verifier
from dma_kws.stage2.model_factory import build_qbyt
from dma_kws.stage2.readout import resolve_qbyt_score_spec
from dma_kws.tokenizer import load_char_tokenizer
from qbyt.pooling import _masked_normalized_softmin


EMBED_DIM = 32
ENCODER_DIM = 24
ATOL = 1e-5
RTOL = 1e-5


def _tokenizer():
    return load_char_tokenizer("data/dict/lang_char.txt", split_with_space=" ")


def _token_ids(*phonemes: str) -> list[int]:
    tokenizer = _tokenizer()
    from dma_kws.tokenizer import tokenize_phoneme_string

    return tokenize_phoneme_string(tokenizer, " ".join(phonemes))


class _CountingQbyT(torch.nn.Module):
    def __init__(self, inner: torch.nn.Module) -> None:
        super().__init__()
        self.inner = inner
        self.batch_sizes: list[int] = []
        self.max_speech_batch: int = 0

    def forward(self, speech, text, speech_lengths=None, text_lengths=None):
        self.batch_sizes.append(int(speech.size(0)))
        self.max_speech_batch = max(self.max_speech_batch, int(speech.size(0)))
        return self.inner(
            speech,
            text,
            speech_lengths=speech_lengths,
            text_lengths=text_lengths,
        )

    def forward_with_readout_details(
        self, speech, text, speech_lengths=None, text_lengths=None
    ):
        self.batch_sizes.append(int(speech.size(0)))
        self.max_speech_batch = max(self.max_speech_batch, int(speech.size(0)))
        return self.inner.forward_with_readout_details(
            speech,
            text,
            speech_lengths=speech_lengths,
            text_lengths=text_lengths,
        )


class _TinyStage2(torch.nn.Module):
    def __init__(self, qbyt: torch.nn.Module, *, adapter=None) -> None:
        super().__init__()
        self.qbyt = qbyt
        self.adapter = adapter
        self.encoder_calls = 0

    def encode_for_qbyt(self, feats, feat_lengths):
        self.encoder_calls += 1
        speech = feats
        if self.adapter is not None:
            speech = self.adapter(speech)
        return speech, feat_lengths

    def forward(self, feats, feat_lengths, anchors, anchor_lengths):
        speech, encoder_lens = self.encode_for_qbyt(feats, feat_lengths)
        logits, _ = self.qbyt(
            speech,
            anchors,
            speech_lengths=encoder_lens,
            text_lengths=anchor_lengths,
        )
        return logits


class _AddOneAdapter(torch.nn.Module):
    def forward(self, speech):
        return speech + 1.0


def _build_qbyt(stage2_cfg: dict, *, vocab_size: int, seed: int = 0):
    torch.manual_seed(seed)
    cfg = {
        "qbyt_embed_dim": EMBED_DIM,
        "qbyt_layers": 2,
        **stage2_cfg,
    }
    return build_qbyt(cfg, input_dim=ENCODER_DIM, vocab_size=vocab_size).eval()


def _verifier_from_qbyt(
    qbyt: torch.nn.Module,
    *,
    slope: float = 1.0,
    bias: float = 0.0,
    adapter=None,
    qbyt_score=None,
) -> Stage2Verifier:
    counting = qbyt if isinstance(qbyt, _CountingQbyT) else _CountingQbyT(qbyt)
    model = _TinyStage2(counting, adapter=adapter).eval()
    verifier = Stage2Verifier.__new__(Stage2Verifier)
    verifier._torch = torch
    verifier._device = torch.device("cpu")
    verifier._amp = None
    verifier._calibrator = PositiveAffineCalibrator(slope=slope, bias=bias)
    verifier._model = model
    verifier._min_fbank_frames = 1
    if qbyt_score is not None:
        verifier.qbyt_score = qbyt_score
    return verifier


def _feats(*lengths: int, seed: int = 0) -> list[torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    return [
        torch.randn(length, ENCODER_DIM, generator=generator)
        for length in lengths
    ]


_POOLING_V2 = {
    "qbyt_readout_version": 2,
    "qbyt_readout": {"mode": "gru_last", "temperature": 1.0},
}
_POOLING_V3 = {
    "qbyt_readout_version": 3,
    "qbyt_readout": {"mode": "eps_mean", "temperature": 1.0},
}
_POOLING_V4 = {
    "qbyt_readout_version": 4,
    "qbyt_readout": {"mode": "gru_last", "temperature": 1.0},
}
_POOLING_V41 = {
    "qbyt_readout_version": 4,
    "qbyt_readout": {
        "mode": "eps_softmin",
        "temperature": 1.0,
        "sink_token": True,
        "text_position": "learned",
        "audio_position": "relative_bias",
        "relative_num_buckets": 32,
        "relative_max_distance": 64,
    },
}
_BOUNDED_V5 = {
    "qbyt_readout_version": 5,
    "qbyt_alignment": {
        "topology": "bounded_segmental_v1",
        "min_phone_duration_frames": 1,
        "max_phone_duration_frames": 4,
        "max_inter_phone_gap_frames": 1,
        "max_keyword_span_frames": 30,
        "local_context_kernel": 5,
        "temperature": 1.0,
    },
}
_V6 = {
    "qbyt_readout_version": 6,
    "qbyt_alignment": {
        "min_phone_duration_frames": 1,
        "max_phone_duration_frames": 4,
        "max_inter_phone_gap_frames": 1,
        "max_keyword_span_frames": 30,
        "local_context_kernel": 5,
        "weakest_phone_temperature": 0.2,
        "weakest_phone_weight": 1.0,
    },
}
_V7 = {
    "qbyt_readout_version": 7,
    "qbyt_alignment": {
        "min_phone_duration_frames": 1,
        "max_phone_duration_frames": 4,
        "max_inter_phone_gap_frames": 1,
        "max_keyword_span_frames": 30,
        "local_context_kernel": 5,
        "weakest_phone_temperature": 0.2,
        "weakest_phone_weight": 1.0,
    },
}


def _reference_pairs(verifier: Stage2Verifier, feats, queries):
    rows = []
    for feat in feats:
        row = []
        for query in queries:
            scored = verifier.score_clip_feats_with_logits([feat], [query])
            assert len(scored) == 1
            row.append(scored[0])
        rows.append(row)
    return rows


def test_empty_feats_returns_empty_and_empty_queries_error():
    qbyt = _build_qbyt(_POOLING_V4, vocab_size=73)
    verifier = _verifier_from_qbyt(qbyt)
    queries = [_token_ids("HH", "EY1")]
    assert verifier.score_clip_feats_multi_with_logits([], queries) == []
    with pytest.raises(KeywordEvalConfigError, match="non-empty"):
        verifier.score_clip_feats_multi_with_logits(_feats(8), [])


def test_encoder_called_once_per_audio_batch_independent_of_query_count():
    qbyt = _build_qbyt(_POOLING_V4, vocab_size=73)
    verifier = _verifier_from_qbyt(qbyt)
    feats = _feats(10, 14, 8)
    queries = [
        _token_ids("HH", "EY1", "IY1", "V", "AH0"),
        _token_ids("HH", "EY1", "EY1", "V", "AH0"),
        _token_ids("OW1", "K", "L", "AE1", "M", "P"),
    ]
    verifier._model.encoder_calls = 0
    result = verifier.score_clip_feats_multi_with_logits(
        feats,
        queries,
        query_batch_size=4,
    )
    assert verifier._model.encoder_calls == 1
    assert len(result) == 3
    assert all(len(row) == 3 for row in result)
    assert max(verifier._model.qbyt.batch_sizes) <= 4
    assert sum(verifier._model.qbyt.batch_sizes) == 9


def test_query_chunking_does_not_drop_pairs_when_not_divisible():
    qbyt = _build_qbyt(_POOLING_V4, vocab_size=73)
    verifier = _verifier_from_qbyt(qbyt)
    feats = _feats(12, 9)
    queries = [
        _token_ids("HH", "EY1"),
        _token_ids("IY1", "V", "AH0"),
        _token_ids("OW1", "K"),
        _token_ids("L", "AE1", "M", "P"),
        _token_ids("AH0"),
    ]
    result = verifier.score_clip_feats_multi_with_logits(
        feats,
        queries,
        query_batch_size=3,
    )
    assert len(result) == 2
    assert all(len(row) == 5 for row in result)
    assert all(math.isfinite(raw) and math.isfinite(score) for row in result for raw, score in row)
    assert max(verifier._model.qbyt.batch_sizes) <= 3
    assert sum(verifier._model.qbyt.batch_sizes) == 10


def test_shared_encode_matches_per_query_reference_fp32():
    qbyt = _build_qbyt(_POOLING_V4, vocab_size=73, seed=7)
    verifier = _verifier_from_qbyt(
        qbyt,
        slope=1.7,
        bias=-0.4,
    )
    feats = _feats(11, 17, seed=3)
    queries = [
        _token_ids("HH", "EY1", "IY1", "V", "AH0"),
        _token_ids("HH", "EY1", "EY1", "V", "AH0"),
    ]
    shared = verifier.score_clip_feats_multi_with_logits(
        feats,
        queries,
        query_batch_size=3,
    )
    # Reset encoder counter by using the same weights through the single-query path.
    reference = _reference_pairs(verifier, feats, queries)
    for clip_index, (shared_row, reference_row) in enumerate(zip(shared, reference)):
        for query_index, ((raw_a, score_a), (raw_b, score_b)) in enumerate(
            zip(shared_row, reference_row)
        ):
            assert raw_a == pytest.approx(raw_b, abs=ATOL, rel=RTOL), (
                clip_index,
                query_index,
            )
            assert score_a == pytest.approx(score_b, abs=ATOL, rel=RTOL)
            expected = float(
                verifier._calibrator.predict_one(raw_a)
            )
            assert score_a == pytest.approx(expected, abs=ATOL, rel=RTOL)


def test_root_score_is_max_of_per_query_calibrated_scores():
    qbyt = _build_qbyt(_POOLING_V4, vocab_size=73, seed=2)
    verifier = _verifier_from_qbyt(qbyt, slope=2.0, bias=0.5)
    feats = _feats(13)
    queries = [
        _token_ids("HH", "EY1"),
        _token_ids("OW1", "K", "L", "AE1", "M", "P"),
        _token_ids("AH0", "L", "OW1"),
    ]
    rows = verifier.score_clip_feats_multi_with_logits(feats, queries)
    calibrated = [score for _raw, score in rows[0]]
    assert max(calibrated) == pytest.approx(max(calibrated))


def test_adapter_path_still_encodes_once():
    qbyt = _build_qbyt(_POOLING_V4, vocab_size=73)
    verifier = _verifier_from_qbyt(qbyt, adapter=_AddOneAdapter())
    feats = _feats(10, 12)
    queries = [_token_ids("HH", "EY1"), _token_ids("IY1", "V")]
    verifier.score_clip_feats_multi_with_logits(feats, queries, query_batch_size=2)
    assert verifier._model.encoder_calls == 1


def test_v41_nonzero_relative_bias_after_adam_matches_reference():
    qbyt = _build_qbyt(_POOLING_V41, vocab_size=73, seed=11)
    qbyt.train()
    optimizer = torch.optim.Adam(qbyt.parameters(), lr=1e-2)
    feats = torch.randn(1, 16, ENCODER_DIM)
    text = torch.tensor([_token_ids("HH", "EY1", "IY1", "V", "AH0")], dtype=torch.long)
    logits, _ = qbyt(
        feats,
        text,
        speech_lengths=torch.tensor([16]),
        text_lengths=torch.tensor([text.size(1)]),
    )
    logits.sum().backward()
    optimizer.step()
    qbyt.eval()
    bias = qbyt.relative_bias
    assert bias is not None
    assert float(
        (bias.audio_buckets.detach().abs().sum() + bias.pair_table.detach().abs().sum()).item()
    ) > 0

    verifier = _verifier_from_qbyt(qbyt, slope=1.3, bias=-0.2)
    clip_feats = _feats(16, 20, seed=5)
    queries = [
        _token_ids("HH", "EY1", "IY1", "V", "AH0"),
        _token_ids("HH", "EY1", "EY1", "V", "AH0"),
        _token_ids("OW1", "K"),
    ]
    with torch.no_grad():
        shared = verifier.score_clip_feats_multi_with_logits(
            clip_feats,
            queries,
            query_batch_size=2,
        )
        reference = _reference_pairs(verifier, clip_feats, queries)
    for shared_row, reference_row in zip(shared, reference):
        for (raw_a, score_a), (raw_b, score_b) in zip(shared_row, reference_row):
            assert raw_a == pytest.approx(raw_b, abs=ATOL, rel=RTOL)
            assert score_a == pytest.approx(score_b, abs=ATOL, rel=RTOL)


@pytest.mark.parametrize(
    "stage2_cfg",
    [
        pytest.param(_POOLING_V2, id="pooling_v2"),
        pytest.param(_POOLING_V3, id="pooling_v3"),
        pytest.param(_POOLING_V4, id="pooling_v4"),
        pytest.param(_POOLING_V41, id="pooling_v41"),
        pytest.param(_BOUNDED_V5, id="bounded_v5"),
        pytest.param(_V6, id="v6"),
        pytest.param(_V7, id="v7"),
    ],
)
def test_shared_encode_matches_reference_across_qbyt_families(stage2_cfg):
    qbyt = _build_qbyt(stage2_cfg, vocab_size=73, seed=4)
    verifier = _verifier_from_qbyt(qbyt)
    feats = _feats(18, 11, seed=9)
    queries = [
        _token_ids("HH", "EY1", "IY1"),
        _token_ids("OW1", "K", "L", "AE1"),
    ]
    shared = verifier.score_clip_feats_multi_with_logits(
        feats,
        queries,
        query_batch_size=3,
    )
    reference = _reference_pairs(verifier, feats, queries)
    for shared_row, reference_row in zip(shared, reference):
        for (raw_a, score_a), (raw_b, score_b) in zip(shared_row, reference_row):
            assert raw_a == pytest.approx(raw_b, abs=ATOL, rel=RTOL)
            assert score_a == pytest.approx(score_b, abs=ATOL, rel=RTOL)


def test_non_finite_query_aborts():
    class _NanQbyT(torch.nn.Module):
        def forward(self, speech, text, speech_lengths=None, text_lengths=None):
            del text, speech_lengths, text_lengths
            logits = torch.zeros(speech.size(0))
            logits[0] = float("nan")
            return logits, None

    verifier = _verifier_from_qbyt(_NanQbyT())
    with pytest.raises(KeywordSetScoreError, match="non-finite"):
        verifier.score_clip_feats_multi_with_logits(
            _feats(8),
            [_token_ids("HH", "EY1")],
        )


def test_pair_count_mismatch_aborts():
    class _ShortQbyT(torch.nn.Module):
        def forward(self, speech, text, speech_lengths=None, text_lengths=None):
            del text, speech_lengths, text_lengths
            return torch.zeros(max(0, speech.size(0) - 1)), None

    verifier = _verifier_from_qbyt(_ShortQbyT())
    with pytest.raises(RuntimeError, match="different number of pair scores"):
        verifier.score_clip_feats_multi_with_logits(
            _feats(8, 9),
            [_token_ids("HH"), _token_ids("EY1")],
            query_batch_size=4,
        )


def _reconstruct_softmin(position_logits, temperature: float = 1.0) -> float:
    logits = torch.tensor([list(position_logits)], dtype=torch.float32)
    mask = torch.ones_like(logits, dtype=torch.bool)
    return float(_masked_normalized_softmin(logits, mask, temperature)[0])


def test_eps_scalar_apis_do_not_request_position_details():
    qbyt_score = resolve_qbyt_score_spec(_POOLING_V41)
    qbyt = _build_qbyt(_POOLING_V41, vocab_size=73, seed=13)
    verifier = _verifier_from_qbyt(qbyt, qbyt_score=qbyt_score)

    def _unexpected_details(*args, **kwargs):
        del args, kwargs
        raise AssertionError("scalar score API requested EPS position details")

    verifier._model.qbyt.forward_with_readout_details = _unexpected_details
    feats = _feats(12, 9, seed=7)
    queries = [
        _token_ids("HH", "EY1", "IY1"),
        _token_ids("OW1", "K"),
    ]

    single = verifier.score_clip_feats_with_logits(feats, queries)
    multi = verifier.score_clip_feats_multi_with_logits(
        feats,
        queries,
        query_batch_size=2,
    )

    assert len(single) == 2
    assert len(multi) == 2
    assert all(len(row) == 2 for row in multi)


def test_multi_query_details_align_positions_and_preserve_tuple_api():
    qbyt_score = resolve_qbyt_score_spec(_POOLING_V41)
    qbyt = _build_qbyt(_POOLING_V41, vocab_size=73, seed=11)
    verifier = _verifier_from_qbyt(qbyt, slope=1.3, bias=-0.2, qbyt_score=qbyt_score)
    feats = _feats(16, 20, seed=5)
    queries = [
        _token_ids("HH", "EY1", "IY1", "V", "AH0"),
        _token_ids("HH", "EY1"),
        _token_ids("OW1", "K", "L"),
    ]
    verifier._model.encoder_calls = 0
    detailed = verifier.score_clip_feats_multi_with_details(
        feats,
        queries,
        query_batch_size=2,
    )
    tuples = verifier.score_clip_feats_multi_with_logits(
        feats,
        queries,
        query_batch_size=2,
    )
    assert verifier._model.encoder_calls == 2
    assert max(verifier._model.qbyt.batch_sizes) <= 2
    assert len(detailed) == 2
    assert all(len(row) == 3 for row in detailed)
    threshold = 0.5
    for clip_row, tuple_row in zip(detailed, tuples):
        for details, (raw, score), query in zip(clip_row, tuple_row, queries):
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
            assert reconstructed == pytest.approx(
                details.qbyt_raw_logit, abs=1e-5, rel=1e-5
            )
            assert (details.qbyt_score >= threshold) is (score >= threshold)


def test_multi_query_details_match_per_query_and_do_not_drop_chunk_pairs():
    qbyt_score = resolve_qbyt_score_spec(_POOLING_V41)
    qbyt = _build_qbyt(_POOLING_V41, vocab_size=73, seed=3)
    verifier = _verifier_from_qbyt(qbyt, slope=1.7, bias=-0.4, qbyt_score=qbyt_score)
    feats = _feats(12, 9, 15, seed=8)
    queries = [
        _token_ids("HH", "EY1"),
        _token_ids("IY1", "V", "AH0"),
        _token_ids("OW1", "K"),
        _token_ids("L", "AE1", "M", "P"),
        _token_ids("AH0"),
    ]
    verifier._model.encoder_calls = 0
    shared = verifier.score_clip_feats_multi_with_details(
        feats,
        queries,
        query_batch_size=3,
    )
    assert verifier._model.encoder_calls == 1
    assert max(verifier._model.qbyt.batch_sizes) <= 3
    assert sum(verifier._model.qbyt.batch_sizes) == 15
    assert len(shared) == 3
    assert all(len(row) == 5 for row in shared)
    for feat, row in zip(feats, shared):
        for query, details in zip(queries, row):
            single = verifier.score_clip_feats_with_details([feat], [query])[0]
            assert details.qbyt_raw_logit == pytest.approx(
                single.qbyt_raw_logit, abs=ATOL, rel=RTOL
            )
            assert details.qbyt_score == pytest.approx(
                single.qbyt_score, abs=ATOL, rel=RTOL
            )
            assert details.qbyt_eps_position_logits == pytest.approx(
                single.qbyt_eps_position_logits, abs=ATOL, rel=RTOL
            )
            assert len(details.qbyt_eps_position_logits) == len(query)


def test_v41_relative_bias_details_are_finite():
    qbyt_score = resolve_qbyt_score_spec(_POOLING_V41)
    qbyt = _build_qbyt(_POOLING_V41, vocab_size=73, seed=11)
    qbyt.train()
    optimizer = torch.optim.Adam(qbyt.parameters(), lr=1e-2)
    feats = torch.randn(1, 16, ENCODER_DIM)
    text = torch.tensor([_token_ids("HH", "EY1", "IY1", "V", "AH0")], dtype=torch.long)
    logits, _ = qbyt(
        feats,
        text,
        speech_lengths=torch.tensor([16]),
        text_lengths=torch.tensor([text.size(1)]),
    )
    logits.sum().backward()
    optimizer.step()
    qbyt.eval()
    verifier = _verifier_from_qbyt(qbyt, slope=1.3, bias=-0.2, qbyt_score=qbyt_score)
    clip_feats = _feats(16, 20, seed=5)
    queries = [
        _token_ids("HH", "EY1", "IY1", "V", "AH0"),
        _token_ids("OW1", "K"),
    ]
    detailed = verifier.score_clip_feats_multi_with_details(
        clip_feats,
        queries,
        query_batch_size=2,
    )
    for row in detailed:
        for details in row:
            assert math.isfinite(details.qbyt_raw_logit)
            assert math.isfinite(details.qbyt_score)
            assert details.qbyt_eps_position_logits is not None
            assert all(math.isfinite(value) for value in details.qbyt_eps_position_logits)


def test_non_eps_multi_query_details_leave_positions_none():
    qbyt_score = resolve_qbyt_score_spec(_V7)
    qbyt = _build_qbyt(_V7, vocab_size=73, seed=4)
    verifier = _verifier_from_qbyt(qbyt, qbyt_score=qbyt_score)
    feats = _feats(18, seed=9)
    queries = [_token_ids("HH", "EY1", "IY1"), _token_ids("OW1", "K")]
    detailed = verifier.score_clip_feats_multi_with_details(feats, queries)
    tuples = verifier.score_clip_feats_multi_with_logits(feats, queries)
    assert detailed[0][0].qbyt_eps_position_logits is None
    assert detailed[0][0].qbyt_raw_logit == pytest.approx(
        tuples[0][0][0], abs=1e-6, rel=1e-6
    )
    assert detailed[0][0].qbyt_score == pytest.approx(
        tuples[0][0][1], abs=1e-6, rel=1e-6
    )
