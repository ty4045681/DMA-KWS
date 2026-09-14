"""T01–T08: real pooling QbyT attention capture and sink-key ablation."""

import inspect

import pytest
import torch

pytest.importorskip("torch")

from qbyt.pooling import QbyT

from dma_kws.inference.qbyt_attention_diagnostics import (
    AttentionCaptureSpec,
    SinkAblationSpec,
    assert_pooling_sink_attention_compatible,
    capture_pooling_attention,
    _clone_and_block_sink_key,
)
from dma_kws.stage2.readout import QbyTAlignmentSpec, QbyTScoreSpec
from dma_kws.stage2.readout_pooling import QbyTReadoutConfig


EMBED_DIM = 64
ENCODER_DIM = 48
PARITY_ATOL = 1e-5
PARITY_RTOL = 1e-4
ROW_SUM_ATOL = 1e-5
PADDING_MASS_ATOL = 1e-5


def _pooling_qbyt(
    *,
    seed: int = 0,
    text_position: str = "learned",
    audio_position: str = "relative_bias",
    distinctive_bias: bool = True,
):
    torch.manual_seed(seed)
    model = QbyT(
        encoder_output_size=ENCODER_DIM,
        num_embeds=73,
        embed_dim=EMBED_DIM,
        post_num_layers=2,
        readout_mode="eps_softmin",
        sink_token=True,
        text_position=text_position,
        audio_position=audio_position,
    ).eval()
    if distinctive_bias and model.relative_bias is not None:
        with torch.no_grad():
            nhead = model.nhead
            num_buckets = model.relative_bias.audio_buckets.size(0)
            model.relative_bias.audio_buckets.copy_(
                torch.arange(num_buckets, dtype=torch.float32)
                .unsqueeze(1)
                .expand(-1, nhead)
                .contiguous()
            )
            model.relative_bias.pair_table.copy_(
                torch.arange(3 * 3 * nhead, dtype=torch.float32).view(3, 3, nhead)
            )
    return model


def _sample(text_len: int, audio_len: int, seed: int):
    generator = torch.Generator().manual_seed(seed)
    text = torch.randint(3, 70, (text_len,), generator=generator)
    audio = torch.randn(audio_len, ENCODER_DIM, generator=generator)
    return text, audio


def _pack(texts, audios):
    text_width = max(text.size(0) for text in texts)
    audio_width = max(audio.size(0) for audio in audios)
    text_batch = torch.zeros(len(texts), text_width, dtype=torch.long)
    audio_batch = torch.zeros(len(audios), audio_width, ENCODER_DIM)
    for row, (text, audio) in enumerate(zip(texts, audios)):
        text_batch[row, : text.size(0)] = text
        audio_batch[row, : audio.size(0)] = audio
    text_lengths = torch.tensor([text.size(0) for text in texts], dtype=torch.long)
    speech_lengths = torch.tensor([audio.size(0) for audio in audios], dtype=torch.long)
    return audio_batch, text_batch, speech_lengths, text_lengths


def _spec(layers=(0, 1), heads=(0, 1, 2, 3), **kwargs):
    return AttentionCaptureSpec(layers=layers, heads=heads, **kwargs)


def _normal():
    return SinkAblationSpec(name="normal")


def _clone_state(model):
    return {key: tensor.detach().clone() for key, tensor in model.state_dict().items()}


def _assert_state_bitwise_equal(model, snapshot):
    current = model.state_dict()
    assert current.keys() == snapshot.keys()
    for key, tensor in current.items():
        assert torch.equal(tensor, snapshot[key]), key


def _capture(model, speech, anchors, speech_lengths, anchor_lengths, spec=None):
    return capture_pooling_attention(
        model,
        speech,
        anchors,
        speech_lengths,
        anchor_lengths,
        capture_spec=spec or _spec(),
        ablation_spec=_normal(),
    )


def test_t01_forward_details_and_capture_logits_agree():
    model = _pooling_qbyt()
    text, audio = _sample(5, 9, seed=1)
    speech, anchors, speech_lengths, anchor_lengths = _pack([text], [audio])
    snapshot = _clone_state(model)

    with torch.no_grad():
        forward_logits, _ = model(
            speech,
            anchors,
            speech_lengths=speech_lengths,
            text_lengths=anchor_lengths,
        )
        details_logits, _, details = model.forward_with_readout_details(
            speech,
            anchors,
            speech_lengths=speech_lengths,
            text_lengths=anchor_lengths,
        )
    traces = _capture(model, speech, anchors, speech_lengths, anchor_lengths)

    torch.testing.assert_close(
        forward_logits, details_logits, atol=PARITY_ATOL, rtol=PARITY_RTOL
    )
    assert len(traces) == 1
    text_len = int(anchor_lengths[0])
    torch.testing.assert_close(
        torch.as_tensor(traces[0].raw_logit, dtype=torch.float32),
        forward_logits[0].cpu(),
        atol=PARITY_ATOL,
        rtol=PARITY_RTOL,
    )
    valid_position = details.position_logits[0, :text_len]
    torch.testing.assert_close(
        traces[0].position_logits,
        valid_position.cpu(),
        atol=PARITY_ATOL,
        rtol=PARITY_RTOL,
    )
    assert traces[0].position_logits.shape == (text_len,)
    assert int(details.position_mask[0].sum()) == text_len
    _assert_state_bitwise_equal(model, snapshot)


def test_t02_mixed_batch_matches_unpadded_singles_and_uses_per_sample_sink_index():
    model = _pooling_qbyt()
    short_text, short_audio = _sample(3, 6, seed=1)
    long_text, long_audio = _sample(8, 12, seed=3)
    spec = _spec(save_full_attention=True)

    short_speech, short_anchors, short_sl, short_al = _pack([short_text], [short_audio])
    long_speech, long_anchors, long_sl, long_al = _pack([long_text], [long_audio])
    mixed_speech, mixed_anchors, mixed_sl, mixed_al = _pack(
        [short_text, long_text],
        [short_audio, long_audio],
    )

    short_trace = _capture(
        model, short_speech, short_anchors, short_sl, short_al, spec
    )[0]
    long_trace = _capture(model, long_speech, long_anchors, long_sl, long_al, spec)[0]
    mixed = _capture(model, mixed_speech, mixed_anchors, mixed_sl, mixed_al, spec)

    assert mixed[0].sink_index == int(short_al[0])
    assert mixed[1].sink_index == int(long_al[0])
    assert mixed[0].sink_index == mixed[0].text_length
    assert mixed[0].sink_index != mixed_anchors.size(1)
    assert mixed[0].text_length == 3
    assert mixed[0].audio_length == 6
    assert mixed[1].text_length == 8
    assert mixed[1].audio_length == 12

    for alone, batched in ((short_trace, mixed[0]), (long_trace, mixed[1])):
        torch.testing.assert_close(
            torch.as_tensor(alone.raw_logit),
            torch.as_tensor(batched.raw_logit),
            atol=PARITY_ATOL,
            rtol=PARITY_RTOL,
        )
        torch.testing.assert_close(
            alone.position_logits,
            batched.position_logits,
            atol=PARITY_ATOL,
            rtol=PARITY_RTOL,
        )
        torch.testing.assert_close(
            alone.audio_to_sink, batched.audio_to_sink, atol=PARITY_ATOL, rtol=PARITY_RTOL
        )
        torch.testing.assert_close(
            alone.text_to_sink, batched.text_to_sink, atol=PARITY_ATOL, rtol=PARITY_RTOL
        )
        torch.testing.assert_close(
            alone.text_to_audio, batched.text_to_audio, atol=PARITY_ATOL, rtol=PARITY_RTOL
        )

    valid_l = mixed[0].text_length + 1 + mixed[0].audio_length
    assert mixed[0].full_attention.shape[-2:] == (valid_l, valid_l)
    torch.testing.assert_close(
        mixed[0].text_to_sink,
        mixed[0].full_attention[:, :, : mixed[0].text_length, mixed[0].sink_index],
        atol=0.0,
        rtol=0.0,
    )
    audio_start = mixed[0].sink_index + 1
    audio_end = audio_start + mixed[0].audio_length
    torch.testing.assert_close(
        mixed[0].text_to_audio,
        mixed[0].full_attention[:, :, : mixed[0].text_length, audio_start:audio_end],
        atol=0.0,
        rtol=0.0,
    )
    text_to_text = mixed[0].full_attention[:, :, : mixed[0].text_length, : mixed[0].text_length]
    sliced_mass = mixed[0].text_to_sink + mixed[0].text_to_audio.sum(dim=-1)
    assert torch.all(sliced_mass < 1.0 - 1e-6)
    torch.testing.assert_close(
        sliced_mass + text_to_text.sum(dim=-1),
        torch.ones_like(sliced_mass),
        atol=ROW_SUM_ATOL,
        rtol=0.0,
    )


def test_t03_relative_bias_is_not_qk_only_and_padding_mass_is_near_zero():
    model = _pooling_qbyt()
    short_text, short_audio = _sample(4, 7, seed=1)
    long_text, long_audio = _sample(9, 14, seed=4)
    speech, anchors, speech_lengths, anchor_lengths = _pack(
        [short_text, long_text],
        [short_audio, long_audio],
    )
    biased = _capture(model, speech, anchors, speech_lengths, anchor_lengths)

    zeroed = _pooling_qbyt(distinctive_bias=False)
    zeroed.load_state_dict(model.state_dict())
    with torch.no_grad():
        zeroed.relative_bias.pair_table.zero_()
        zeroed.relative_bias.audio_buckets.zero_()
    unbiased = _capture(zeroed, speech, anchors, speech_lengths, anchor_lengths)

    assert not torch.allclose(
        biased[0].audio_to_sink,
        unbiased[0].audio_to_sink,
        atol=PARITY_ATOL,
        rtol=PARITY_RTOL,
    )
    assert not torch.allclose(
        biased[0].text_to_audio,
        unbiased[0].text_to_audio,
        atol=PARITY_ATOL,
        rtol=PARITY_RTOL,
    )
    for trace in biased:
        assert trace.row_sum_max_error < ROW_SUM_ATOL
        assert trace.padding_mass_max < PADDING_MASS_ATOL


def test_t03_sinusoidal_src_key_padding_mask_branch():
    model = _pooling_qbyt(text_position="sinusoidal", audio_position="sinusoidal")
    short_text, short_audio = _sample(4, 7, seed=1)
    long_text, long_audio = _sample(9, 14, seed=4)
    speech, anchors, speech_lengths, anchor_lengths = _pack(
        [short_text, long_text],
        [short_audio, long_audio],
    )
    with torch.no_grad():
        logits, _, details = model.forward_with_readout_details(
            speech,
            anchors,
            speech_lengths=speech_lengths,
            text_lengths=anchor_lengths,
        )
    traces = _capture(model, speech, anchors, speech_lengths, anchor_lengths)
    for index, trace in enumerate(traces):
        text_len = int(anchor_lengths[index])
        torch.testing.assert_close(
            torch.as_tensor(trace.raw_logit),
            logits[index].cpu(),
            atol=PARITY_ATOL,
            rtol=PARITY_RTOL,
        )
        torch.testing.assert_close(
            trace.position_logits,
            details.position_logits[index, :text_len].cpu(),
            atol=PARITY_ATOL,
            rtol=PARITY_RTOL,
        )
        assert trace.row_sum_max_error < ROW_SUM_ATOL
        assert trace.padding_mass_max < PADDING_MASS_ATOL
        assert trace.sink_index == text_len


def test_t04_layer_head_selection_keeps_original_ids_and_fires_once():
    model = _pooling_qbyt()
    text, audio = _sample(5, 8, seed=2)
    speech, anchors, speech_lengths, anchor_lengths = _pack([text], [audio])
    fires = {0: 0, 1: 0}

    def _counter(layer_id):
        def _hook(_module, _args, _output):
            fires[layer_id] += 1

        return _hook

    handles = [
        model.phone_matchor.layers[layer_id].self_attn.register_forward_hook(
            _counter(layer_id)
        )
        for layer_id in (0, 1)
    ]
    try:
        all_heads = _capture(
            model,
            speech,
            anchors,
            speech_lengths,
            anchor_lengths,
            _spec(layers=(0, 1), heads=(0, 1, 2, 3), save_full_attention=True),
        )[0]
        selected = _capture(
            model,
            speech,
            anchors,
            speech_lengths,
            anchor_lengths,
            _spec(layers=(1,), heads=(0, 2), save_full_attention=True),
        )[0]
    finally:
        for handle in handles:
            handle.remove()

    assert fires == {0: 2, 1: 2}
    assert all_heads.layer_ids == (0, 1)
    assert all_heads.head_ids == (0, 1, 2, 3)
    assert selected.layer_ids == (1,)
    assert selected.head_ids == (0, 2)
    assert selected.audio_to_sink.shape[:2] == (1, 2)
    assert 0 not in selected.layer_ids
    torch.testing.assert_close(
        selected.audio_to_sink[0],
        all_heads.audio_to_sink[1, (0, 2)],
        atol=PARITY_ATOL,
        rtol=PARITY_RTOL,
    )
    torch.testing.assert_close(
        selected.text_to_sink[0],
        all_heads.text_to_sink[1, (0, 2)],
        atol=PARITY_ATOL,
        rtol=PARITY_RTOL,
    )


def test_lifecycle_restores_fastpath_keeps_existing_hooks_and_drops_ours():
    model = _pooling_qbyt()
    text, audio = _sample(4, 6, seed=5)
    speech, anchors, speech_lengths, anchor_lengths = _pack([text], [audio])
    attn = model.phone_matchor.layers[0].self_attn
    existing_hits = []

    def _existing(_module, _args, _output):
        existing_hits.append(1)

    existing = attn.register_forward_hook(_existing)
    before_pre = set(attn._forward_pre_hooks)
    before_fwd = set(attn._forward_hooks)
    try:
        assert torch.backends.mha.get_fastpath_enabled() is True
        _capture(model, speech, anchors, speech_lengths, anchor_lengths, _spec(layers=(0,), heads=(1,)))
        assert torch.backends.mha.get_fastpath_enabled() is True
        assert set(attn._forward_pre_hooks) == before_pre
        assert set(attn._forward_hooks) == before_fwd
        assert existing_hits == [1]

        torch.backends.mha.set_fastpath_enabled(False)
        try:
            _capture(
                model,
                speech,
                anchors,
                speech_lengths,
                anchor_lengths,
                _spec(layers=(0,), heads=(1,)),
            )
            assert torch.backends.mha.get_fastpath_enabled() is False
        finally:
            torch.backends.mha.set_fastpath_enabled(True)
        assert set(attn._forward_hooks) == before_fwd
    finally:
        existing.remove()


def test_forward_exception_still_removes_capture_hooks_and_restores_fastpath():
    model = _pooling_qbyt()
    text, audio = _sample(4, 6, seed=6)
    speech, anchors, speech_lengths, anchor_lengths = _pack([text], [audio])
    attn = model.phone_matchor.layers[1].self_attn
    before_keys = [
        (set(layer.self_attn._forward_pre_hooks), set(layer.self_attn._forward_hooks))
        for layer in model.phone_matchor.layers
    ]

    def _boom(_module, _args, _output):
        raise RuntimeError("boom")

    boom = attn.register_forward_hook(_boom)
    try:
        with pytest.raises(RuntimeError, match="boom"):
            _capture(model, speech, anchors, speech_lengths, anchor_lengths)
        assert torch.backends.mha.get_fastpath_enabled() is True
        after_keys = [
            (set(layer.self_attn._forward_pre_hooks), set(layer.self_attn._forward_hooks))
            for layer in model.phone_matchor.layers
        ]
        assert after_keys[0] == before_keys[0]
        assert after_keys[1][0] == before_keys[1][0]
        assert after_keys[1][1] == {boom.id}
    finally:
        boom.remove()


def test_training_model_raises_without_switching_eval():
    model = _pooling_qbyt().train()
    text, audio = _sample(3, 5, seed=7)
    speech, anchors, speech_lengths, anchor_lengths = _pack([text], [audio])
    with pytest.raises(RuntimeError, match="eval"):
        _capture(model, speech, anchors, speech_lengths, anchor_lengths)
    assert model.training is True


def test_empty_keyword_and_zero_audio_are_rejected():
    model = _pooling_qbyt()
    text, audio = _sample(3, 5, seed=8)
    speech, anchors, speech_lengths, anchor_lengths = _pack([text], [audio])
    with pytest.raises(ValueError, match="empty keyword"):
        capture_pooling_attention(
            model,
            speech,
            anchors,
            speech_lengths,
            torch.zeros_like(anchor_lengths),
            capture_spec=_spec(),
            ablation_spec=_normal(),
        )
    empty_audio = torch.zeros(1, 0, ENCODER_DIM)
    empty_speech_lengths = torch.zeros(1, dtype=torch.long)
    with pytest.raises(ValueError, match="T == 0|unscorable"):
        capture_pooling_attention(
            model,
            empty_audio,
            anchors,
            empty_speech_lengths,
            anchor_lengths,
            capture_spec=_spec(),
            ablation_spec=SinkAblationSpec(name="block_sink_all", blocked_layers=(0, 1)),
        )
    query = torch.zeros(1, 1, EMBED_DIM)
    with pytest.raises(ValueError, match="every remaining key"):
        _clone_and_block_sink_key(
            query=query,
            attn_mask=None,
            key_padding_mask=query.new_zeros(1, 1),
            text_lens=[0],
            audio_lens=[0],
            num_heads=4,
            batch_first=True,
        )


def test_rejects_bool_duplicate_and_out_of_range_indices():
    model = _pooling_qbyt()
    text, audio = _sample(3, 5, seed=9)
    speech, anchors, speech_lengths, anchor_lengths = _pack([text], [audio])
    with pytest.raises((TypeError, ValueError)):
        _capture(
            model,
            speech,
            anchors,
            speech_lengths,
            anchor_lengths,
            AttentionCaptureSpec(layers=(True,), heads=(0,)),
        )
    with pytest.raises(ValueError, match="duplicate"):
        _capture(
            model,
            speech,
            anchors,
            speech_lengths,
            anchor_lengths,
            AttentionCaptureSpec(layers=(0, 0), heads=(0,)),
        )
    with pytest.raises(ValueError, match="range"):
        _capture(
            model,
            speech,
            anchors,
            speech_lengths,
            anchor_lengths,
            AttentionCaptureSpec(layers=(2,), heads=(0,)),
        )
    with pytest.raises(ValueError, match="range"):
        _capture(
            model,
            speech,
            anchors,
            speech_lengths,
            anchor_lengths,
            AttentionCaptureSpec(layers=(0,), heads=(-1,)),
        )


def test_empty_selection_still_runs_forward():
    model = _pooling_qbyt()
    text, audio = _sample(3, 5, seed=10)
    speech, anchors, speech_lengths, anchor_lengths = _pack([text], [audio])
    with torch.no_grad():
        logits, _, details = model.forward_with_readout_details(
            speech,
            anchors,
            speech_lengths=speech_lengths,
            text_lengths=anchor_lengths,
        )
    traces = _capture(
        model,
        speech,
        anchors,
        speech_lengths,
        anchor_lengths,
        AttentionCaptureSpec(layers=(), heads=()),
    )
    assert traces[0].layer_ids == ()
    assert traces[0].head_ids == ()
    assert traces[0].audio_to_sink.shape == (0, 0, 5)
    assert traces[0].text_to_sink.shape == (0, 0, 3)
    torch.testing.assert_close(
        torch.as_tensor(traces[0].raw_logit),
        logits[0].cpu(),
        atol=PARITY_ATOL,
        rtol=PARITY_RTOL,
    )
    torch.testing.assert_close(
        traces[0].position_logits,
        details.position_logits[0, :3].cpu(),
        atol=PARITY_ATOL,
        rtol=PARITY_RTOL,
    )


def test_packed_length_and_full_attention_bytes_precheck():
    model = _pooling_qbyt()
    text, audio = _sample(3, 5, seed=11)
    speech, anchors, speech_lengths, anchor_lengths = _pack([text], [audio])
    packed_l = anchors.size(1) + speech.size(1) + 1
    with pytest.raises(ValueError, match="max_combined_tokens"):
        _capture(
            model,
            speech,
            anchors,
            speech_lengths,
            anchor_lengths,
            AttentionCaptureSpec(
                layers=(0,),
                heads=(0,),
                max_combined_tokens=packed_l - 1,
            ),
        )
    with pytest.raises(ValueError, match="max_attention_bytes"):
        _capture(
            model,
            speech,
            anchors,
            speech_lengths,
            anchor_lengths,
            AttentionCaptureSpec(
                layers=(0, 1),
                heads=(0, 1, 2, 3),
                save_full_attention=True,
                max_attention_bytes=8,
            ),
        )


def test_nested_capture_on_same_model_raises():
    model = _pooling_qbyt()
    text, audio = _sample(3, 5, seed=12)
    speech, anchors, speech_lengths, anchor_lengths = _pack([text], [audio])

    def _reenter(_module, _args):
        capture_pooling_attention(
            model,
            speech,
            anchors,
            speech_lengths,
            anchor_lengths,
            capture_spec=_spec(layers=(0,), heads=(0,)),
            ablation_spec=_normal(),
        )

    handle = model.phone_matchor.register_forward_pre_hook(_reenter)
    try:
        with pytest.raises(RuntimeError, match="nested|concurrent"):
            _capture(model, speech, anchors, speech_lengths, anchor_lengths)
    finally:
        handle.remove()
    for layer in model.phone_matchor.layers:
        assert not layer.self_attn._forward_pre_hooks
        assert not layer.self_attn._forward_hooks


SINK_COL_ATOL = 1e-6


def _pooling_score_spec(**kwargs):
    payload = {
        "mode": "eps_softmin",
        "temperature": 1.0,
        "sink_token": True,
        "text_position": "learned",
        "audio_position": "relative_bias",
    }
    payload.update(kwargs)
    return QbyTScoreSpec(version=4, value=QbyTReadoutConfig(**payload))


def _block(name, *layers):
    return SinkAblationSpec(name=name, blocked_layers=tuple(layers))


def _sink_column(trace):
    assert trace.full_attention is not None
    return trace.full_attention[:, :, :, trace.sink_index]


@pytest.mark.parametrize(
    "text_position,audio_position",
    [
        ("learned", "relative_bias"),
        ("sinusoidal", "sinusoidal"),
    ],
)
def test_t05_block_sink_all_and_layer_i(text_position, audio_position):
    model = _pooling_qbyt(text_position=text_position, audio_position=audio_position)
    short_text, short_audio = _sample(4, 7, seed=21)
    long_text, long_audio = _sample(6, 11, seed=22)
    speech, anchors, speech_lengths, anchor_lengths = _pack(
        [short_text, long_text],
        [short_audio, long_audio],
    )
    spec = _spec(save_full_attention=True)
    normal = capture_pooling_attention(
        model,
        speech,
        anchors,
        speech_lengths,
        anchor_lengths,
        capture_spec=spec,
        ablation_spec=_normal(),
    )
    blocked_all = capture_pooling_attention(
        model,
        speech,
        anchors,
        speech_lengths,
        anchor_lengths,
        capture_spec=spec,
        ablation_spec=_block("block_sink_all", 0, 1),
    )
    blocked_layer0 = capture_pooling_attention(
        model,
        speech,
        anchors,
        speech_lengths,
        anchor_lengths,
        capture_spec=spec,
        ablation_spec=_block("block_sink_layer_0", 0),
    )
    blocked_layer1 = capture_pooling_attention(
        model,
        speech,
        anchors,
        speech_lengths,
        anchor_lengths,
        capture_spec=spec,
        ablation_spec=_block("block_sink_layer_1", 1),
    )

    for traces in (blocked_all, blocked_layer0, blocked_layer1):
        for trace in traces:
            assert trace.row_sum_max_error < ROW_SUM_ATOL
            assert trace.padding_mass_max < PADDING_MASS_ATOL

    for trace in blocked_all:
        sink_col = _sink_column(trace)
        assert torch.all(sink_col.abs() < SINK_COL_ATOL)
        torch.testing.assert_close(
            trace.text_to_sink,
            torch.zeros_like(trace.text_to_sink),
            atol=SINK_COL_ATOL,
            rtol=0.0,
        )
        torch.testing.assert_close(
            trace.audio_to_sink,
            torch.zeros_like(trace.audio_to_sink),
            atol=SINK_COL_ATOL,
            rtol=0.0,
        )

    for trace in blocked_layer0:
        assert torch.all(_sink_column(trace)[0].abs() < SINK_COL_ATOL)
        torch.testing.assert_close(
            trace.audio_to_sink[0],
            torch.zeros_like(trace.audio_to_sink[0]),
            atol=SINK_COL_ATOL,
            rtol=0.0,
        )
        # Downstream layer 1 may change because layer 0 representations changed.
        # Do not require it to match normal; only the blocked layer is forced.

    for index, trace in enumerate(blocked_layer1):
        assert torch.all(_sink_column(trace)[1].abs() < SINK_COL_ATOL)
        torch.testing.assert_close(
            trace.audio_to_sink[0],
            normal[index].audio_to_sink[0],
            atol=PARITY_ATOL,
            rtol=PARITY_RTOL,
        )
        torch.testing.assert_close(
            trace.text_to_sink[0],
            normal[index].text_to_sink[0],
            atol=PARITY_ATOL,
            rtol=PARITY_RTOL,
        )


def test_t06_mask_clone_restore_and_other_batch_rows_unpolluted():
    model = _pooling_qbyt()
    snapshot = _clone_state(model)
    short_text, short_audio = _sample(3, 6, seed=31)
    long_text, long_audio = _sample(8, 12, seed=32)
    speech, anchors, speech_lengths, anchor_lengths = _pack(
        [short_text, long_text],
        [short_audio, long_audio],
    )
    spec = _spec(save_full_attention=True)
    first_normal = capture_pooling_attention(
        model,
        speech,
        anchors,
        speech_lengths,
        anchor_lengths,
        capture_spec=spec,
        ablation_spec=_normal(),
    )

    layer1_masks = []

    def _watch_layer1(_module, args, kwargs):
        bound = inspect.signature(_module.forward).bind_partial(*args, **kwargs)
        attn_mask = bound.arguments.get("attn_mask")
        if attn_mask is not None:
            layer1_masks.append(attn_mask.detach().clone())
        return args, kwargs

    watch = model.phone_matchor.layers[1].self_attn.register_forward_pre_hook(
        _watch_layer1, with_kwargs=True
    )
    try:
        blocked_layer0 = capture_pooling_attention(
            model,
            speech,
            anchors,
            speech_lengths,
            anchor_lengths,
            capture_spec=spec,
            ablation_spec=_block("block_sink_layer_0", 0),
        )
        blocked_all = capture_pooling_attention(
            model,
            speech,
            anchors,
            speech_lengths,
            anchor_lengths,
            capture_spec=spec,
            ablation_spec=_block("block_sink_all", 0, 1),
        )
    finally:
        watch.remove()

    assert layer1_masks, "layer 1 should still receive the shared attn_mask"
    n_heads = 4
    short_u = int(anchor_lengths[0])
    long_u = int(anchor_lengths[1])
    shared = layer1_masks[0]
    # Shared mask passed to the unblocked layer must not have been filled with
    # -inf on the sink columns; that would mean the original attn_bias was
    # mutated in place.
    assert not torch.isneginf(shared[0:n_heads, :, short_u]).all()
    assert not torch.isneginf(shared[n_heads : 2 * n_heads, :, long_u]).all()

    second_normal = capture_pooling_attention(
        model,
        speech,
        anchors,
        speech_lengths,
        anchor_lengths,
        capture_spec=spec,
        ablation_spec=_normal(),
    )
    for first, second in zip(first_normal, second_normal):
        torch.testing.assert_close(
            first.audio_to_sink, second.audio_to_sink, atol=0.0, rtol=0.0
        )
        torch.testing.assert_close(
            first.text_to_sink, second.text_to_sink, atol=0.0, rtol=0.0
        )
        torch.testing.assert_close(
            first.text_to_audio, second.text_to_audio, atol=0.0, rtol=0.0
        )
        torch.testing.assert_close(
            first.full_attention, second.full_attention, atol=0.0, rtol=0.0
        )
        torch.testing.assert_close(
            first.position_logits, second.position_logits, atol=0.0, rtol=0.0
        )
        assert first.raw_logit == second.raw_logit

    short_trace, long_trace = blocked_all
    assert torch.all(_sink_column(short_trace).abs() < SINK_COL_ATOL)
    assert torch.all(_sink_column(long_trace).abs() < SINK_COL_ATOL)
    # The longer sample's key at the shorter sample's sink index is a text key,
    # not its sink. Blocking sample 0 must not zero that column on sample 1.
    long_at_short_sink = long_trace.full_attention[:, :, :, short_u]
    assert not torch.allclose(
        long_at_short_sink,
        torch.zeros_like(long_at_short_sink),
        atol=SINK_COL_ATOL,
        rtol=0.0,
    )
    # Unblocked layer 1 during block_sink_layer_0 must still have sink mass;
    # an in-place edit of the shared bias would force it to zero as well.
    layer1_sink = _sink_column(blocked_layer0[0])[1]
    assert not torch.allclose(
        layer1_sink, torch.zeros_like(layer1_sink), atol=SINK_COL_ATOL, rtol=0.0
    )
    _assert_state_bitwise_equal(model, snapshot)


def test_t07_lifecycle_ablation_hooks_fastpath_and_exceptions(monkeypatch):
    model = _pooling_qbyt()
    text, audio = _sample(4, 6, seed=41)
    speech, anchors, speech_lengths, anchor_lengths = _pack([text], [audio])
    spec = _spec(layers=(0, 1), heads=(0, 1))
    ablation = _block("block_sink_all", 0, 1)
    existing_hits = []

    def _existing(_module, _args, _output):
        existing_hits.append(1)

    attn0 = model.phone_matchor.layers[0].self_attn
    existing = attn0.register_forward_hook(_existing)
    before_pre = [set(layer.self_attn._forward_pre_hooks) for layer in model.phone_matchor.layers]
    before_fwd = [set(layer.self_attn._forward_hooks) for layer in model.phone_matchor.layers]
    try:
        assert torch.backends.mha.get_fastpath_enabled() is True
        capture_pooling_attention(
            model,
            speech,
            anchors,
            speech_lengths,
            anchor_lengths,
            capture_spec=spec,
            ablation_spec=ablation,
        )
        assert torch.backends.mha.get_fastpath_enabled() is True
        assert existing_hits == [1]
        for layer_id, layer in enumerate(model.phone_matchor.layers):
            assert set(layer.self_attn._forward_pre_hooks) == before_pre[layer_id]
            assert set(layer.self_attn._forward_hooks) == before_fwd[layer_id]

        torch.backends.mha.set_fastpath_enabled(False)
        try:
            capture_pooling_attention(
                model,
                speech,
                anchors,
                speech_lengths,
                anchor_lengths,
                capture_spec=spec,
                ablation_spec=ablation,
            )
            assert torch.backends.mha.get_fastpath_enabled() is False
        finally:
            torch.backends.mha.set_fastpath_enabled(True)

        def _boom(_module, _args, _output):
            raise RuntimeError("forward boom")

        boom = model.phone_matchor.layers[1].self_attn.register_forward_hook(_boom)
        try:
            with pytest.raises(RuntimeError, match="forward boom"):
                capture_pooling_attention(
                    model,
                    speech,
                    anchors,
                    speech_lengths,
                    anchor_lengths,
                    capture_spec=spec,
                    ablation_spec=ablation,
                )
        finally:
            boom.remove()
        assert torch.backends.mha.get_fastpath_enabled() is True
        for layer_id, layer in enumerate(model.phone_matchor.layers):
            assert set(layer.self_attn._forward_pre_hooks) == before_pre[layer_id]
            assert set(layer.self_attn._forward_hooks) == before_fwd[layer_id]

        import dma_kws.inference.qbyt_attention_diagnostics as diagnostics

        def _save_boom(**_kwargs):
            raise RuntimeError("save boom")

        monkeypatch.setattr(diagnostics, "_assemble_traces", _save_boom)
        with pytest.raises(RuntimeError, match="save boom"):
            capture_pooling_attention(
                model,
                speech,
                anchors,
                speech_lengths,
                anchor_lengths,
                capture_spec=spec,
                ablation_spec=ablation,
            )
        monkeypatch.undo()
        assert torch.backends.mha.get_fastpath_enabled() is True
        for layer_id, layer in enumerate(model.phone_matchor.layers):
            assert set(layer.self_attn._forward_pre_hooks) == before_pre[layer_id]
            assert set(layer.self_attn._forward_hooks) == before_fwd[layer_id]

        attn1 = model.phone_matchor.layers[1].self_attn
        original_register = attn1.register_forward_pre_hook

        def _install_boom(*args, **kwargs):
            raise RuntimeError("install boom")

        attn1.register_forward_pre_hook = _install_boom
        try:
            with pytest.raises(RuntimeError, match="install boom"):
                capture_pooling_attention(
                    model,
                    speech,
                    anchors,
                    speech_lengths,
                    anchor_lengths,
                    capture_spec=spec,
                    ablation_spec=ablation,
                )
        finally:
            attn1.register_forward_pre_hook = original_register
        assert torch.backends.mha.get_fastpath_enabled() is True
        for layer_id, layer in enumerate(model.phone_matchor.layers):
            assert set(layer.self_attn._forward_pre_hooks) == before_pre[layer_id]
            assert set(layer.self_attn._forward_hooks) == before_fwd[layer_id]
        assert existing.id in attn0._forward_hooks
    finally:
        existing.remove()


def test_t08_unsupported_spec_no_sink_mismatch_and_state_unchanged():
    model = _pooling_qbyt()
    snapshot = _clone_state(model)
    text, audio = _sample(4, 7, seed=51)
    speech, anchors, speech_lengths, anchor_lengths = _pack([text], [audio])
    capture_pooling_attention(
        model,
        speech,
        anchors,
        speech_lengths,
        anchor_lengths,
        capture_spec=_spec(save_full_attention=True),
        ablation_spec=_block("block_sink_all", 0, 1),
    )
    _assert_state_bitwise_equal(model, snapshot)

    compatible = _pooling_score_spec()
    assert_pooling_sink_attention_compatible(model, compatible)

    with pytest.raises(ValueError, match="version"):
        assert_pooling_sink_attention_compatible(
            model,
            QbyTScoreSpec(version=7, value=QbyTAlignmentSpec()),
        )
    with pytest.raises(ValueError, match="sink"):
        assert_pooling_sink_attention_compatible(
            model,
            _pooling_score_spec(sink_token=False),
        )

    no_sink = _pooling_qbyt()
    no_sink.sink_token = None
    with pytest.raises(ValueError, match="sink"):
        assert_pooling_sink_attention_compatible(no_sink, compatible)

    mismatch = _pooling_qbyt()
    mismatch.readout_mode = "eps_mean"
    with pytest.raises(ValueError, match="mismatch|eps_softmin"):
        assert_pooling_sink_attention_compatible(mismatch, compatible)

    with pytest.raises(ValueError, match="range"):
        capture_pooling_attention(
            model,
            speech,
            anchors,
            speech_lengths,
            anchor_lengths,
            capture_spec=_spec(),
            ablation_spec=_block("block_sink_layer_9", 9),
        )
