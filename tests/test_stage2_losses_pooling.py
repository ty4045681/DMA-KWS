import pytest
import torch

from dma_kws.stage2.losses_pooling import compute_stage2_losses


def test_utt_loss_near_zero_when_logits_match_labels():
    labels = torch.tensor([1.0, 0.0, 1.0])
    logits = torch.tensor([20.0, -20.0, 20.0])

    _, losses = compute_stage2_losses(
        logits=logits,
        seq_logits=torch.zeros(3, 2),
        labels=labels,
        seq_labels=torch.zeros(3, 2),
        seq_label_mask=torch.zeros(3, 2),
    )

    assert losses["utt_loss"].item() < 1e-6
    assert losses["seq_progress_loss"].item() == 0.0
    assert losses["seq_completion_loss"].item() == 0.0


def test_padding_does_not_change_seq_losses_or_receive_gradients():
    seq_logits = torch.tensor(
        [[2.0, 2.0, 2.0], [2.0, -2.0, float("inf")]],
        requires_grad=True,
    )
    seq_labels = torch.tensor([[1, 1, 1], [1, 0, -1]])
    mask = seq_labels.ne(-1).float()

    total, losses = compute_stage2_losses(
        logits=torch.zeros(2),
        seq_logits=seq_logits,
        labels=torch.zeros(2),
        seq_labels=seq_labels,
        seq_label_mask=mask,
    )
    changed_padding = seq_logits.detach().clone()
    changed_padding[1, 2] = float("-inf")
    _, changed_losses = compute_stage2_losses(
        logits=torch.zeros(2),
        seq_logits=changed_padding,
        labels=torch.zeros(2),
        seq_labels=seq_labels,
        seq_label_mask=mask,
    )

    assert torch.allclose(losses["seq_progress_loss"], changed_losses["seq_progress_loss"])
    assert torch.allclose(losses["seq_completion_loss"], changed_losses["seq_completion_loss"])
    expected_completion = torch.nn.functional.binary_cross_entropy_with_logits(
        torch.tensor([2.0, -2.0]),
        torch.tensor([1.0, 0.0]),
    )
    assert torch.allclose(losses["seq_completion_loss"], expected_completion)

    total.backward()
    assert seq_logits.grad[1, 2].item() == 0.0


def test_sample_normalization_gives_short_and_long_anchors_equal_weight():
    seq_logits = torch.tensor([[-4.0, 0.0, 0.0, 0.0], [4.0, 4.0, 4.0, 4.0]])
    seq_labels = torch.tensor([[1, -1, -1, -1], [1, 1, 1, 1]])
    mask = seq_labels.ne(-1).float()

    _, sample_losses = compute_stage2_losses(
        logits=torch.zeros(2),
        seq_logits=seq_logits,
        labels=torch.zeros(2),
        seq_labels=seq_labels,
        seq_label_mask=mask,
        seq_progress_weight=1.0,
        seq_completion_weight=0.0,
        seq_normalization="sample",
    )
    _, token_losses = compute_stage2_losses(
        logits=torch.zeros(2),
        seq_logits=seq_logits,
        labels=torch.zeros(2),
        seq_labels=seq_labels,
        seq_label_mask=mask,
        seq_progress_weight=1.0,
        seq_completion_weight=0.0,
        seq_normalization="token",
    )

    short = torch.nn.functional.softplus(torch.tensor(4.0))
    long = torch.nn.functional.softplus(torch.tensor(-4.0))
    assert torch.allclose(sample_losses["seq_progress_loss"], (short + long) / 2)
    assert torch.allclose(token_losses["seq_progress_loss"], (short + 4 * long) / 5)
    assert sample_losses["seq_progress_loss"] > token_losses["seq_progress_loss"]


def test_all_padding_sample_is_excluded_from_both_sequence_reductions():
    seq_logits = torch.tensor(
        [[2.0, -2.0], [float("nan"), float("inf")]],
        requires_grad=True,
    )
    seq_labels = torch.tensor([[1, 0], [-1, -1]])
    mask = seq_labels.ne(-1).float()

    total, losses = compute_stage2_losses(
        logits=torch.zeros(2),
        seq_logits=seq_logits,
        labels=torch.zeros(2),
        seq_labels=seq_labels,
        seq_label_mask=mask,
    )
    _, single = compute_stage2_losses(
        logits=torch.zeros(1),
        seq_logits=seq_logits.detach()[:1],
        labels=torch.zeros(1),
        seq_labels=seq_labels[:1],
        seq_label_mask=mask[:1],
    )

    assert torch.isfinite(losses["seq_loss"])
    assert torch.allclose(losses["seq_progress_loss"], single["seq_progress_loss"])
    assert torch.allclose(losses["seq_completion_loss"], single["seq_completion_loss"])
    total.backward()
    assert torch.equal(seq_logits.grad[1], torch.zeros(2))


def test_total_loss_uses_configured_progress_completion_and_ctc_weights():
    logits = torch.tensor([0.5, -0.3])
    seq_logits = torch.tensor([[0.2, -0.1], [0.4, 0.6]])
    labels = torch.tensor([1.0, 0.0])
    seq_labels = torch.tensor([[1.0, 0.0], [1.0, 1.0]])
    seq_label_mask = torch.tensor([[1.0, 1.0], [1.0, 0.0]])

    ctc_loss = torch.tensor(0.7)
    total_loss, losses = compute_stage2_losses(
        logits=logits,
        seq_logits=seq_logits,
        labels=labels,
        seq_labels=seq_labels,
        seq_label_mask=seq_label_mask,
        seq_progress_weight=0.25,
        seq_completion_weight=0.75,
        ctc_loss=ctc_loss,
        ctc_weight=0.2,
    )

    assert torch.allclose(
        losses["seq_loss"],
        0.25 * losses["seq_progress_loss"]
        + 0.75 * losses["seq_completion_loss"],
    )
    assert torch.allclose(
        total_loss,
        losses["utt_loss"] + losses["seq_loss"] + 0.2 * ctc_loss,
    )
    assert torch.allclose(
        losses["seq_loss"],
        losses["seq_progress_weighted_loss"]
        + losses["seq_completion_weighted_loss"],
    )
    assert torch.allclose(losses["ctc_weighted_loss"], 0.2 * ctc_loss)
    assert torch.allclose(losses["total_loss"], total_loss)
    assert torch.allclose(
        losses["total_loss"],
        losses["utt_loss"]
        + losses["seq_progress_weighted_loss"]
        + losses["seq_completion_weighted_loss"]
        + losses["ctc_weighted_loss"],
    )


def test_legacy_weights_and_token_normalization_reproduce_masked_seq_bce():
    seq_logits = torch.tensor([[0.2, -0.1], [0.4, 8.0]])
    seq_labels = torch.tensor([[1.0, 0.0], [1.0, -1.0]])
    mask = seq_labels.ne(-1).float()
    safe_labels = torch.where(mask.bool(), seq_labels, torch.zeros_like(seq_labels))
    expected = torch.nn.functional.binary_cross_entropy_with_logits(
        seq_logits,
        safe_labels,
        weight=mask,
        reduction="sum",
    ) / mask.sum()

    _, losses = compute_stage2_losses(
        logits=torch.zeros(2),
        seq_logits=seq_logits,
        labels=torch.zeros(2),
        seq_labels=seq_labels,
        seq_label_mask=mask,
        seq_progress_weight=1.0,
        seq_completion_weight=0.0,
        seq_normalization="token",
    )

    assert torch.allclose(losses["seq_loss"], expected)


def test_seq_loss_rejects_invalid_configuration():
    common = {
        "logits": torch.zeros(1),
        "seq_logits": torch.zeros(1, 1),
        "labels": torch.zeros(1),
        "seq_labels": torch.zeros(1, 1),
        "seq_label_mask": torch.ones(1, 1),
    }
    for value in (-1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="finite and non-negative"):
            compute_stage2_losses(**common, seq_progress_weight=value)

    with pytest.raises(ValueError, match="normalization"):
        compute_stage2_losses(**common, seq_normalization="batch")


def test_negative_tail_loss_is_added_with_configured_weight():
    logits = torch.tensor([1.5, -0.5, 0.25])
    labels = torch.tensor([0.0, 0.0, 1.0])
    seq_logits = torch.empty(3, 0)
    seq_labels = torch.empty(3, 0)
    seq_mask = torch.empty(3, 0)

    baseline, baseline_losses = compute_stage2_losses(
        logits,
        seq_logits,
        labels,
        seq_labels,
        seq_mask,
    )
    total, losses = compute_stage2_losses(
        logits,
        seq_logits,
        labels,
        seq_labels,
        seq_mask,
        negative_tail_weight=0.4,
        negative_tail_fraction=0.5,
    )

    expected_tail = torch.nn.functional.softplus(torch.tensor(1.5))
    assert torch.allclose(losses["negative_tail_loss"], expected_tail)
    assert torch.allclose(
        losses["negative_tail_weighted_loss"],
        0.4 * expected_tail,
    )
    assert torch.allclose(total, baseline + 0.4 * expected_tail)
    assert "negative_tail_loss" not in baseline_losses
    assert "negative_tail_weighted_loss" not in baseline_losses


def test_negative_tail_loss_zero_weight_matches_baseline_keys():
    logits = torch.tensor([1.5, -0.5, 0.25])
    labels = torch.tensor([0.0, 0.0, 1.0])
    seq_logits = torch.empty(3, 0)
    seq_labels = torch.empty(3, 0)
    seq_mask = torch.empty(3, 0)

    baseline, baseline_losses = compute_stage2_losses(
        logits,
        seq_logits,
        labels,
        seq_labels,
        seq_mask,
    )
    total, losses = compute_stage2_losses(
        logits,
        seq_logits,
        labels,
        seq_labels,
        seq_mask,
        negative_tail_weight=0.0,
        negative_tail_fraction=0.5,
    )

    assert torch.allclose(total, baseline)
    assert set(losses) == set(baseline_losses)
    assert "negative_tail_loss" not in losses


@pytest.mark.parametrize("weight", [-1.0, float("nan"), float("inf")])
def test_negative_tail_loss_rejects_invalid_weight(weight):
    with pytest.raises(ValueError, match="negative_tail_weight"):
        compute_stage2_losses(
            logits=torch.zeros(1),
            seq_logits=torch.empty(1, 0),
            labels=torch.zeros(1),
            seq_labels=torch.empty(1, 0),
            seq_label_mask=torch.empty(1, 0),
            negative_tail_weight=weight,
        )


@pytest.mark.parametrize("fraction", [0.0, -0.1, 1.1, float("nan"), float("inf")])
def test_negative_tail_loss_rejects_invalid_fraction(fraction):
    with pytest.raises(ValueError, match="negative_tail_fraction"):
        compute_stage2_losses(
            logits=torch.zeros(1),
            seq_logits=torch.empty(1, 0),
            labels=torch.zeros(1),
            seq_labels=torch.empty(1, 0),
            seq_label_mask=torch.empty(1, 0),
            negative_tail_fraction=fraction,
        )
