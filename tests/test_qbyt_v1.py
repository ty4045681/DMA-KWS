"""Paper-original v1 (SI) QbyT: registry, dictionary, and scoring semantics."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from dma_kws.stage2.model_factory import build_qbyt
from dma_kws.stage2.readout import (
    QBYT_V1_READOUT,
    QbyTV1Spec,
    resolve_qbyt_score_spec,
)
from dma_kws.tokenizer import (
    V1_DICT_PATH,
    VOCAB_PROFILE_CANONICAL,
    VOCAB_PROFILE_V1_73,
    load_char_tokenizer,
    resolve_vocab_profile,
    validate_lang_char_dict,
)
from dma_kws.training.checkpoint_io import (
    QBYT_READOUT_VERSION_KEY,
    assert_qbyt_readout_version,
    checkpoint_qbyt_readout_spec,
    stamp_qbyt_readout_version,
)

#: Raw author release; tests that need it skip when it is not present.
_AUTHOR_STAGE2 = Path(
    os.environ.get(
        "DMA_KWS_AUTHOR_STAGE2", "/tmp/dma-upstream/ckpts/stage2/155k-v2-ft.ckpt"
    )
)

_EXPECTED_V1_KEYS = frozenset(
    {
        "audio_projection.weight",
        "audio_projection.bias",
        "text_projection.weight",
        "pos_enc.pe",
        "modality_enc.text_emb",
        "modality_enc.audio_emb",
        "gru.weight_ih_l0",
        "gru.weight_hh_l0",
        "gru.bias_ih_l0",
        "gru.bias_hh_l0",
        "fc.weight",
        "fc.bias",
        "seq_fc.weight",
        "seq_fc.bias",
    }
    | {
        f"phone_matchor.layers.{layer}.{name}"
        for layer in (0, 1)
        for name in (
            "self_attn.in_proj_weight",
            "self_attn.in_proj_bias",
            "self_attn.out_proj.weight",
            "self_attn.out_proj.bias",
            "linear1.weight",
            "linear1.bias",
            "linear2.weight",
            "linear2.bias",
            "norm1.weight",
            "norm1.bias",
            "norm2.weight",
            "norm2.bias",
        )
    }
)

_V1_CONFIG = {"qbyt_readout_version": 1}


def _v1_model():
    model = build_qbyt(_V1_CONFIG, input_dim=144, vocab_size=73)
    model.eval()
    return model


def test_v1_spec_is_a_single_frozen_contract() -> None:
    spec = resolve_qbyt_score_spec(_V1_CONFIG)
    assert (spec.version, spec.family) == (1, "v1")
    assert spec.value == QbyTV1Spec()
    assert spec.value.as_dict() == {"readout": QBYT_V1_READOUT}
    assert spec.emission is None
    with pytest.raises(ValueError, match="Unsupported QbyT v1 readout"):
        QbyTV1Spec(readout="something_else")


def test_v1_rejects_readout_and_alignment_fields() -> None:
    with pytest.raises(ValueError, match="unsupported by QbyT v1"):
        resolve_qbyt_score_spec(
            {"qbyt_readout_version": 1, "qbyt_readout": {"mode": "gru_last"}}
        )
    with pytest.raises(ValueError, match="unsupported by QbyT v1"):
        resolve_qbyt_score_spec(
            {"qbyt_readout_version": 1, "qbyt_alignment": {"topology": "x"}}
        )


def test_v1_model_keys_match_the_released_architecture() -> None:
    assert frozenset(_v1_model().state_dict()) == _EXPECTED_V1_KEYS


def test_v1_stamp_roundtrip() -> None:
    payload = {
        "config": {"stage2": {"qbyt_readout_version": 1}},
        "model_state_dict": {"qbyt.fc.weight": torch.zeros(1)},
    }
    stamped = stamp_qbyt_readout_version(
        payload, alignment=resolve_qbyt_score_spec(_V1_CONFIG)
    )
    assert stamped[QBYT_READOUT_VERSION_KEY] == 1
    assert "qbyt_alignment_spec" not in stamped
    decoded = checkpoint_qbyt_readout_spec(stamped)
    assert (decoded.version, decoded.family) == (1, "v1")
    assert_qbyt_readout_version(stamped, source="v1.pt", expected_alignment=decoded)


def test_v1_qbyts_are_refused_by_a_v7_run() -> None:
    payload = {
        "config": {"stage2": {"qbyt_readout_version": 1}},
        QBYT_READOUT_VERSION_KEY: 1,
        "model_state_dict": {"qbyt.fc.weight": torch.zeros(1)},
    }
    with pytest.raises(SystemExit, match="Readout semantics differ"):
        assert_qbyt_readout_version(
            payload,
            source="v1.pt",
            expected_alignment=resolve_qbyt_score_spec({}),
        )


def test_v73_dictionary_profile() -> None:
    validate_lang_char_dict(V1_DICT_PATH)
    tokenizer = load_char_tokenizer(V1_DICT_PATH)
    assert resolve_vocab_profile(V1_DICT_PATH, tokenizer.symbol_table, None) == VOCAB_PROFILE_V1_73
    assert len(tokenizer.symbol_table) == 73
    assert tokenizer.symbol_table["AA0"] == 3
    _, ids = tokenizer.tokenize("HH EY1 S N IH1 P S")
    assert ids == [36, 32, 57, 47, 38, 55, 57]

    canonical = load_char_tokenizer(V1_DICT_PATH.parent / "lang_char.txt")
    assert resolve_vocab_profile(
        V1_DICT_PATH.parent / "lang_char.txt", canonical.symbol_table, None
    ) == VOCAB_PROFILE_CANONICAL
    assert len(canonical.symbol_table) == 71


def test_v1_readout_is_padding_dependent() -> None:
    """The paper readout scores the padded tail; this is intentional."""

    model = _v1_model()
    torch.manual_seed(0)
    text = torch.tensor([[36, 32, 57, 47, 38]])
    speech = torch.randn(1, 24, 144)
    padded = torch.cat([speech, torch.zeros(1, 16, 144)], dim=1)
    with torch.no_grad():
        alone, _ = model(speech, text)
        with_padding, _ = model(padded, text)
    assert alone.shape == with_padding.shape == (1,)
    assert abs(float(alone[0]) - float(with_padding[0])) > 1e-3

    with torch.no_grad():
        again, _ = model(speech, text)
    assert torch.equal(alone, again)


@pytest.mark.skipif(not _AUTHOR_STAGE2.is_file(), reason="author v1 checkpoint absent")
def test_unversioned_author_checkpoint_is_still_rejected() -> None:
    raw = torch.load(_AUTHOR_STAGE2, map_location="cpu", weights_only=False)
    with pytest.raises(SystemExit, match="Readout semantics differ"):
        assert_qbyt_readout_version(raw, source=_AUTHOR_STAGE2)


@pytest.mark.skipif(not _AUTHOR_STAGE2.is_file(), reason="author v1 checkpoint absent")
def test_author_v1_weights_load_strictly() -> None:
    from dma_kws.pathing import ensure_qbyt_on_path

    ensure_qbyt_on_path()
    from models.encoder import ConformerEncoder

    state = torch.load(_AUTHOR_STAGE2, map_location="cpu", weights_only=False)[
        "state_dict"
    ]
    container = torch.nn.Module()
    container.encoder = ConformerEncoder(
        input_size=80, output_size=144, attention_heads=4, linear_units=576,
        num_blocks=6, dropout_rate=0.1, positional_dropout_rate=0.1,
        attention_dropout_rate=0.0, use_cnn_module=True, input_layer="conv2d",
        pos_enc_layer_type="rel_pos", selfattention_layer_type="rel_selfattn",
        cnn_module_kernel=3,
    )
    container.qbyt = _v1_model()
    container.load_state_dict(state, strict=True)


def test_v1_verifier_scores_clips_one_at_a_time() -> None:
    from dma_kws.inference.stage2_verifier import Stage2Verifier

    class _Recording:
        _v1_readout = True

        def __init__(self):
            self.batch_sizes: list[int] = []

        def _score_clip_feats_logits_only(self, feats, keyword_ids_batch):
            self.batch_sizes.append(len(feats))
            return [(0.5, 0.5)] * len(feats)

    stub = _Recording()
    result = Stage2Verifier._score_clip_feats_logits_only(
        stub,
        [torch.zeros(1, 2)] * 3,
        [[1], [2], [3]],
    )
    assert result == [(0.5, 0.5)] * 3
    assert stub.batch_sizes == [1, 1, 1]


def test_v1_multi_query_scoring_delegates_to_single_clip_path() -> None:
    from dma_kws.inference.stage2_verifier import Stage2Verifier

    class _Stub:
        _v1_readout = True
        _torch = torch

        def __init__(self):
            self.delegated: tuple[int, int, bool] | None = None

        def _score_v1_readout_multi(self, feats, queries, *, return_details):
            self.delegated = (len(feats), len(queries), return_details)
            return [[(0.0, 0.0)] * len(queries) for _ in feats]

    stub = _Stub()
    matrix = Stage2Verifier._score_clip_feats_multi(
        stub,
        [torch.zeros(1, 2), torch.zeros(1, 2)],
        [[1], [2, 3]],
        query_batch_size=64,
        include_eps_positions=False,
        return_details=False,
    )
    assert len(matrix) == 2 and len(matrix[0]) == 2
    assert stub.delegated == (2, 2, False)
