"""Unit tests for stage2.sink_loss and the sink readout plumbing."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from dma_kws.stage2.losses import (
    normalize_sink_loss_form,
    sink_bce_loss,
    sink_rank_loss,
    validate_sink_loss,
)


def test_normalize_form_accepts_case_and_space():
    assert normalize_sink_loss_form(" BCE ") == "bce"
    assert normalize_sink_loss_form("Rank") == "rank"
    with pytest.raises(ValueError):
        normalize_sink_loss_form("triplet")


def test_validate_rejects_bad_values():
    with pytest.raises(ValueError):
        validate_sink_loss(weight=-0.1, form="bce", temperature=1.0)
    with pytest.raises(ValueError):
        validate_sink_loss(weight=0.2, form="bce", temperature=0.0)
    with pytest.raises(ValueError):
        validate_sink_loss(weight=0.2, form="nope", temperature=1.0)


def test_sink_bce_matches_torch_and_keeps_gradient():
    logits = torch.tensor([1.0, -2.0, 0.5], requires_grad=True)
    labels = torch.tensor([1, 0, 1], dtype=torch.long)
    expected = F.binary_cross_entropy_with_logits(logits, labels.float())
    actual = sink_bce_loss(logits, labels)
    assert torch.allclose(actual, expected)
    actual.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


def test_sink_rank_rewards_separation():
    labels = torch.tensor([1, 1, 0, 0], dtype=torch.long)
    good = torch.tensor([2.0, 1.5, -1.0, -2.0], requires_grad=True)
    bad = torch.tensor([-1.0, -2.0, 1.5, 2.0])
    assert float(sink_rank_loss(good, labels)) < float(sink_rank_loss(bad, labels))
    sink_rank_loss(good, labels).backward()
    assert good.grad is not None and torch.isfinite(good.grad).all()


def test_sink_rank_single_class_returns_zero_gradient():
    labels = torch.tensor([1, 1, 1], dtype=torch.long)
    logits = torch.tensor([0.5, -0.5, 0.1], requires_grad=True)
    loss = sink_rank_loss(logits, labels)
    loss.backward()
    assert float(loss) == 0.0
    assert logits.grad is not None
    assert float(logits.grad.abs().sum()) == 0.0


def _qbyt(**kwargs):
    from qbyt.pooling import QbyT

    return QbyT(
        encoder_output_size=32,
        num_embeds=11,
        embed_dim=16,
        post_num_layers=1,
        readout_mode="eps_softmin",
        sink_token=True,
        **kwargs,
    )


def test_details_expose_sink_logit_only_with_head():
    speech = torch.randn(2, 6, 32)
    text = torch.randint(3, 10, (2, 3))
    plain = _qbyt().eval()
    with torch.no_grad():
        _logits, _seq, details = plain.forward_with_readout_details(speech, text)
    assert details is not None
    assert details.sink_logit is None

    headed = _qbyt(sink_readout="additive", sink_zero_init=True).eval()
    with torch.no_grad():
        logits, _seq, details = headed.forward_with_readout_details(speech, text)
    assert details.sink_logit is not None
    assert details.sink_logit.shape == (2,)


def test_score_temperature_applies_only_at_eval():
    from qbyt.pooling import QbyT

    model = QbyT(
        encoder_output_size=32,
        num_embeds=11,
        embed_dim=16,
        post_num_layers=1,
        readout_mode="eps_softmin",
        sink_token=True,
        score_temperature=0.3,
    )
    model.train()
    assert model._effective_temperature() == 1.0
    model.eval()
    assert abs(model._effective_temperature() - 0.3) < 1e-9
    plain = QbyT(
        encoder_output_size=32,
        num_embeds=11,
        embed_dim=16,
        post_num_layers=1,
        readout_mode="eps_softmin",
        sink_token=True,
    ).eval()
    assert plain._effective_temperature() == 1.0


def test_freeze_all_but_sink_policy():
    from qbyt.pooling import QbyT

    model = QbyT(
        encoder_output_size=32,
        num_embeds=11,
        embed_dim=16,
        post_num_layers=1,
        readout_mode="eps_softmin",
        sink_token=True,
        sink_readout="additive",
        sink_zero_init=True,
    )
    for param in model.parameters():
        param.requires_grad_(False)
    for name, param in model.named_parameters():
        if name.startswith("sink_fc") or name == "sink_alpha":
            param.requires_grad_(True)
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert trainable == {"sink_fc.weight", "sink_fc.bias", "sink_alpha"}


def test_zero_init_branch_is_neutral_at_step_zero():
    speech = torch.randn(2, 6, 32)
    text = torch.randint(3, 10, (2, 3))
    headed = _qbyt(sink_readout="additive", sink_zero_init=True).eval()
    reference = _qbyt().eval()
    state = {
        key: value
        for key, value in headed.state_dict().items()
        if not key.startswith("sink_")
    }
    reference.load_state_dict(state, strict=False)
    with torch.no_grad():
        headed_logits, _seq, _details = headed.forward_with_readout_details(speech, text)
        reference_logits, _seq2 = reference(speech, text)
    assert torch.allclose(headed_logits, reference_logits, atol=1e-6)
