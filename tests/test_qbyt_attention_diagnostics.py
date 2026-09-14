"""T01–T04: real pooling QbyT attention capture, no mocked weights."""

import pytest
import torch

pytest.importorskip("torch")

from qbyt.pooling import QbyT

from dma_kws.inference.qbyt_attention_diagnostics import (
    AttentionCaptureSpec,
    SinkAblationSpec,
    capture_pooling_attention,
)


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


def test_nonempty_blocked_layers_raises_until_ablation_task():
    model = _pooling_qbyt()
    text, audio = _sample(3, 5, seed=8)
    speech, anchors, speech_lengths, anchor_lengths = _pack([text], [audio])
    with pytest.raises(ValueError, match="blocked_layers"):
        capture_pooling_attention(
            model,
            speech,
            anchors,
            speech_lengths,
            anchor_lengths,
            capture_spec=_spec(),
            ablation_spec=SinkAblationSpec(name="block_sink_all", blocked_layers=(0, 1)),
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
