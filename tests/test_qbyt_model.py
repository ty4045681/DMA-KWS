"""Structural and padding invariants of the QbyT v6 segmental-CRF scorer."""

import math

import pytest
import torch

pytest.importorskip("torch")

from dma_kws.pathing import load_qbyt_class

EMBED_DIM = 32
ENCODER_DIM = 24


def _model(
    seed: int = 0,
    *,
    layers: int = 2,
    kernel: int = 5,
    num_embeds: int = 73,
    emission: str = "one_vs_rest",
):
    torch.manual_seed(seed)
    QbyT = load_qbyt_class()
    return QbyT(
        encoder_output_size=ENCODER_DIM,
        num_embeds=num_embeds,
        embed_dim=EMBED_DIM,
        post_num_layers=layers,
        local_context_kernel=kernel,
        min_phone_duration_frames=1,
        max_phone_duration_frames=4,
        max_inter_phone_gap_frames=1,
        max_keyword_span_frames=64,
        weakest_phone_temperature=0.2,
        weakest_phone_weight=1.0,
        dropout=0.0,
        emission=emission,
    ).eval()


def _sample(phone_count: int, frame_count: int, seed: int):
    generator = torch.Generator().manual_seed(seed)
    phones = torch.randint(3, 70, (phone_count,), generator=generator)
    speech = torch.randn(frame_count, ENCODER_DIM, generator=generator)
    return phones, speech


def _batched_score(model, phones, speech):
    phone_width = max(value.size(0) for value in phones)
    frame_width = max(value.size(0) for value in speech)
    phone_batch = torch.zeros(len(phones), phone_width, dtype=torch.long)
    speech_batch = torch.zeros(len(speech), frame_width, ENCODER_DIM)
    for index, (query, frames) in enumerate(zip(phones, speech)):
        phone_batch[index, : query.size(0)] = query
        speech_batch[index, : frames.size(0)] = frames
    with torch.no_grad():
        return model(
            speech_batch,
            phone_batch,
            speech_lengths=torch.tensor([value.size(0) for value in speech]),
            text_lengths=torch.tensor([value.size(0) for value in phones]),
        )


def test_score_and_prefixes_are_invariant_to_batch_padding():
    model = _model()
    query, speech = _sample(5, 24, 1)
    companion_query, companion_speech = _sample(12, 40, 2)

    alone_logit, alone_prefix = _batched_score(model, [query], [speech])
    batch_logit, batch_prefix = _batched_score(
        model,
        [query, companion_query],
        [speech, companion_speech],
    )

    torch.testing.assert_close(alone_logit[0], batch_logit[0], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(
        alone_prefix[0], batch_prefix[0, : query.numel()], atol=1e-5, rtol=1e-5
    )


def test_audio_and_text_padding_values_cannot_change_score():
    model = _model()
    query, speech = _sample(5, 24, 1)
    padded_query = torch.cat([query, torch.tensor([68, 69, 70])])
    padded_speech = torch.cat([speech, torch.randn(11, ENCODER_DIM)], dim=0)

    with torch.no_grad():
        clean, _ = model(
            speech.unsqueeze(0),
            query.unsqueeze(0),
            speech_lengths=torch.tensor([24]),
            text_lengths=torch.tensor([5]),
        )
        padded, _ = model(
            padded_speech.unsqueeze(0),
            padded_query.unsqueeze(0),
            speech_lengths=torch.tensor([24]),
            text_lengths=torch.tensor([5]),
        )
    torch.testing.assert_close(clean, padded, atol=1e-5, rtol=1e-5)


def test_utterance_logit_is_exactly_the_final_valid_prefix():
    model = _model()
    queries = torch.tensor([[3, 4, 5, 0], [6, 7, 8, 9]])
    speech = torch.randn(2, 24, ENCODER_DIM)
    logits, prefixes = model(
        speech,
        queries,
        speech_lengths=torch.tensor([24, 20]),
        text_lengths=torch.tensor([3, 4]),
    )
    torch.testing.assert_close(logits, torch.stack([prefixes[0, 2], prefixes[1, 3]]))


def test_local_lattice_masks_padding():
    model = _model(layers=1, kernel=5)
    queries = torch.tensor([[3, 4, 0], [5, 6, 7]])
    speech = torch.randn(2, 18, ENCODER_DIM)
    with torch.no_grad():
        target_llr, filler, duration, _, _, phone_mask, frame_mask = model._encode_lattice(
            speech,
            queries,
            torch.tensor([12, 18]),
            torch.tensor([2, 3]),
        )
    assert target_llr.shape == (2, 3, 18)
    assert filler.shape == (2, 18)
    assert duration.shape == (2, 3, 4)
    assert phone_mask.tolist() == [[True, True, False], [True, True, True]]
    assert frame_mask[0].sum().item() == 12
    torch.testing.assert_close(
        target_llr[0, 2], torch.zeros_like(target_llr[0, 2])
    )
    torch.testing.assert_close(
        target_llr[0, :, 12:],
        torch.zeros_like(target_llr[0, :, 12:]),
    )


def test_audio_context_has_a_bounded_receptive_field():
    model = _model(layers=2, kernel=5)
    query = torch.tensor([[3, 4, 5]])
    speech = torch.randn(1, 32, ENCODER_DIM)
    changed = speech.clone()
    changed[:, 16] += 10.0
    lengths = torch.tensor([32])
    with torch.no_grad():
        first = model._encode_lattice(
            speech, query, lengths, torch.tensor([3])
        )[0]
        second = model._encode_lattice(
            changed, query, lengths, torch.tensor([3])
        )[0]
    # Two kernel-5 blocks have radius 4. Frames outside it are exactly unchanged.
    torch.testing.assert_close(first[:, :, :12], second[:, :, :12])
    torch.testing.assert_close(first[:, :, 21:], second[:, :, 21:])


def test_model_contains_no_global_text_audio_attention_path():
    model = _model()
    assert not any(isinstance(module, torch.nn.MultiheadAttention) for module in model.modules())
    assert not any("phone_matchor" in name for name, _ in model.named_modules())


def test_deployment_loss_reaches_competitive_emission_and_duration_parameters():
    model = _model().train()
    query = torch.tensor([[3, 4, 5]])
    speech = torch.randn(1, 24, ENCODER_DIM)
    logit, _ = model(speech, query, torch.tensor([24]), torch.tensor([3]))
    logit.sum().backward()
    for parameter in (
        model.audio_projection.weight,
        model.audio_key.weight,
        model.text_query.weight,
        model.text_projection.weight,
        model.phone_bias,
        model.blank_head.weight,
        model.noise_head.weight,
        model.duration_logits.weight,
        model.match_log_scale,
    ):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_encode_lattice_uses_one_vs_rest_log_odds():
    model = _model(layers=1)
    queries = torch.tensor([[3, 4, 5, 0], [6, 7, 0, 0]])
    speech = torch.randn(2, 18, ENCODER_DIM)
    speech_lengths = torch.tensor([18, 12])
    text_lengths = torch.tensor([3, 2])
    with torch.no_grad():
        class_log_probs = model.frame_class_log_probs(
            speech, model._length_mask(speech_lengths, 18)
        )
        target_llr, filler, *_ = model._encode_lattice(
            speech, queries, speech_lengths, text_lengths
        )

    torch.testing.assert_close(filler, torch.zeros_like(filler))
    for b in range(2):
        for u in range(4):
            for t in range(18):
                if u < int(text_lengths[b]) and t < int(speech_lengths[b]):
                    log_p = class_log_probs[b, t, int(queries[b, u]) - 1]
                    expected = log_p - torch.log1p(-log_p.exp().clamp(max=1.0 - 1e-6))
                    torch.testing.assert_close(
                        target_llr[b, u, t], expected, atol=1e-5, rtol=1e-5
                    )
                else:
                    assert target_llr[b, u, t].item() == 0.0


def test_in_query_substitution_is_charged_against_the_competing_phone(monkeypatch):
    # Five phone classes plus blank and noise. The frame is acoustically phone 2
    # and the query contains both phone 1 and phone 2, so the legacy
    # query-relative filler would drop phone 2's 0.60 from phone 1's denominator.
    model = _model(layers=0, num_embeds=6)
    probs = torch.tensor([[[0.15, 0.60, 0.10, 0.05, 0.03, 0.04, 0.03]]])
    monkeypatch.setattr(
        model, "frame_class_log_probs", lambda speech, frame_mask: probs.log()
    )
    query = torch.tensor([[1, 2]])
    with torch.no_grad():
        target_llr, *_ = model._encode_lattice(
            torch.zeros(1, 1, ENCODER_DIM), query, torch.tensor([1]), torch.tensor([2])
        )

    one_vs_rest = math.log(0.15 / 0.85)
    query_relative = math.log(0.15 / (1.0 - 0.15 - 0.60))
    assert target_llr[0, 0, 0].item() == pytest.approx(one_vs_rest, abs=1e-5)
    assert abs(target_llr[0, 0, 0].item() - query_relative) > 1.0
    assert target_llr[0, 1, 0].item() == pytest.approx(math.log(0.60 / 0.40), abs=1e-5)


def test_query_relative_emission_reproduces_legacy_filler(monkeypatch):
    model = _model(layers=0, num_embeds=6, emission="query_relative")
    probs = torch.tensor([[[0.15, 0.60, 0.10, 0.05, 0.03, 0.04, 0.03]]])
    monkeypatch.setattr(
        model, "frame_class_log_probs", lambda speech, frame_mask: probs.log()
    )
    query = torch.tensor([[1, 2]])
    with torch.no_grad():
        target_llr, filler, *_ = model._encode_lattice(
            torch.zeros(1, 1, ENCODER_DIM), query, torch.tensor([1]), torch.tensor([2])
        )

    expected_filler = math.log(1.0 - 0.15 - 0.60)
    assert filler[0, 0].item() == pytest.approx(expected_filler, abs=1e-5)
    assert target_llr[0, 0, 0].item() == pytest.approx(
        math.log(0.15) - expected_filler, abs=1e-5
    )


def test_query_relative_and_one_vs_rest_disagree_on_identical_weights():
    one_vs_rest = _model(seed=3, layers=1, emission="one_vs_rest")
    query_relative = _model(seed=3, layers=1, emission="query_relative")
    query_relative.load_state_dict(one_vs_rest.state_dict())
    query, speech = _sample(4, 20, 11)
    with torch.no_grad():
        rest_logit, _ = one_vs_rest(
            speech.unsqueeze(0),
            query.unsqueeze(0),
            speech_lengths=torch.tensor([speech.size(0)]),
            text_lengths=torch.tensor([query.size(0)]),
        )
        relative_logit, _ = query_relative(
            speech.unsqueeze(0),
            query.unsqueeze(0),
            speech_lengths=torch.tensor([speech.size(0)]),
            text_lengths=torch.tensor([query.size(0)]),
        )
    assert not torch.allclose(rest_logit, relative_logit, atol=1e-5, rtol=1e-5)


def test_unknown_emission_is_rejected():
    with pytest.raises(ValueError, match="Unsupported QbyT emission"):
        _model(emission="softmax_pool")


def test_zero_initialized_duration_potentials_are_centered_uniform():
    model = _model(layers=0)
    speech = torch.randn(1, 16, ENCODER_DIM)
    query = torch.tensor([[3, 4]])
    with torch.no_grad():
        _, _, duration, *_ = model._encode_lattice(
            speech, query, torch.tensor([16]), torch.tensor([2])
        )
    # Zero initialization is a centered uniform duration potential, so it adds
    # no arbitrary per-phone offset to the graph score.
    torch.testing.assert_close(duration, torch.zeros_like(duration))


def test_weakest_phone_term_is_veto_only():
    model = _model()
    query = torch.tensor([[3, 4, 5]])
    speech = torch.randn(1, 24, ENCODER_DIM)
    speech_lengths = torch.tensor([24])
    text_lengths = torch.tensor([3])
    with torch.no_grad():
        lattice = model._encode_lattice(
            speech, query, speech_lengths, text_lengths
        )
        alignment = model.aligner(
            lattice[0],
            lattice[1],
            lattice[3],
            lattice[4],
            duration_log_probs=lattice[2],
        )
        _, seq_logits = model(
            speech, query, speech_lengths, text_lengths
        )
    assert torch.all(seq_logits[:, :3] <= alignment.prefix_llr[:, :3])


def test_invalid_local_context_kernel_is_rejected():
    with pytest.raises(ValueError, match="positive odd"):
        _model(kernel=4)


def test_empty_anchor_has_a_finite_reject_logit():
    model = _model()
    logits, prefixes = model(
        torch.randn(1, 12, ENCODER_DIM),
        torch.zeros(1, 1, dtype=torch.long),
        torch.tensor([12]),
        torch.tensor([0]),
    )
    assert torch.isfinite(logits).all()
    assert logits.item() < -1000
    assert prefixes[0, 0].item() < -1000


def test_illegal_full_keyword_path_returns_exactly_one_invalid_score():
    model = _model()
    query = torch.tensor([[3, 4, 5, 6, 7]])

    # Five phones need at least five frames, so the full-keyword state has no
    # legal segmental path.  The completeness veto must not be added on top of
    # the aligner's reject sentinel (which would turn -1e4 into -2e4).
    logits, prefixes = model(
        torch.randn(1, 4, ENCODER_DIM),
        query,
        torch.tensor([4]),
        torch.tensor([5]),
    )

    invalid_score = model.aligner.invalid_score
    assert logits.item() == invalid_score
    assert prefixes[0, 4].item() == invalid_score
