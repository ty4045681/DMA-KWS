import pytest
import torch

from dma_kws.stage2.losses import compute_stage2_losses, negative_tail_cvar_loss


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
    assert set(losses) == {
        "utt_loss",
        "seq_loss",
        "seq_progress_loss",
        "seq_progress_weighted_loss",
        "total_loss",
    }


def test_valid_path_mask_excludes_illegal_samples_from_all_qbyt_losses():
    logits = torch.tensor([0.4, 50.0], requires_grad=True)
    seq_logits = torch.tensor(
        [[0.2, -0.1, 0.8], [-20.0, 30.0, 40.0]],
        requires_grad=True,
    )
    labels = torch.tensor([1.0, 0.0])
    seq_labels = torch.tensor([[1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
    seq_mask = torch.ones_like(seq_labels)

    total, losses = compute_stage2_losses(
        logits=logits,
        seq_logits=seq_logits,
        labels=labels,
        seq_labels=seq_labels,
        seq_label_mask=seq_mask,
        valid_path_mask=torch.tensor([True, False]),
        seq_progress_weight=1.0,
        negative_tail_weight=1.0,
        negative_tail_fraction=1.0,
    )
    expected, expected_losses = compute_stage2_losses(
        logits=logits.detach()[:1],
        seq_logits=seq_logits.detach()[:1],
        labels=labels[:1],
        seq_labels=seq_labels[:1],
        seq_label_mask=seq_mask[:1],
        seq_progress_weight=1.0,
        negative_tail_weight=1.0,
        negative_tail_fraction=1.0,
    )

    assert torch.allclose(total, expected)
    for name in (
        "utt_loss",
        "seq_progress_loss",
        "seq_progress_weighted_loss",
        "seq_loss",
        "negative_tail_loss",
        "negative_tail_weighted_loss",
    ):
        assert torch.allclose(losses[name], expected_losses[name])

    total.backward()
    assert logits.grad[0].abs().item() > 0.0
    assert logits.grad[1].item() == 0.0
    assert seq_logits.grad[0, :2].abs().sum().item() > 0.0
    assert torch.equal(seq_logits.grad[1], torch.zeros(3))


def test_all_illegal_paths_return_a_differentiable_zero_loss():
    logits = torch.tensor([2.0, -3.0], requires_grad=True)
    seq_logits = torch.tensor(
        [[0.5, -0.5, 1.0], [2.0, -2.0, 3.0]],
        requires_grad=True,
    )

    total, losses = compute_stage2_losses(
        logits=logits,
        seq_logits=seq_logits,
        labels=torch.tensor([1.0, 0.0]),
        seq_labels=torch.tensor([[1.0, 0.0, 1.0], [1.0, 1.0, 0.0]]),
        seq_label_mask=torch.ones(2, 3),
        valid_path_mask=torch.zeros(2, dtype=torch.bool),
        seq_progress_weight=1.0,
        negative_tail_weight=1.0,
        negative_tail_fraction=1.0,
    )

    assert total.requires_grad
    assert total.item() == 0.0
    for name in (
        "utt_loss",
        "seq_progress_loss",
        "seq_progress_weighted_loss",
        "seq_loss",
        "negative_tail_loss",
        "negative_tail_weighted_loss",
        "total_loss",
    ):
        assert losses[name].item() == 0.0

    total.backward()
    assert torch.equal(logits.grad, torch.zeros_like(logits))
    assert torch.equal(seq_logits.grad, torch.zeros_like(seq_logits))


def test_padding_and_final_prefix_do_not_change_progress_loss_or_receive_gradients():
    seq_logits = torch.tensor(
        [[2.0, 2.0, -9.0], [2.0, -7.0, float("inf")]],
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
    changed_ignored = seq_logits.detach().clone()
    changed_ignored[0, 2] = 100.0
    changed_ignored[1, 1] = -100.0
    changed_ignored[1, 2] = float("-inf")
    _, changed_losses = compute_stage2_losses(
        logits=torch.zeros(2),
        seq_logits=changed_ignored,
        labels=torch.zeros(2),
        seq_labels=seq_labels,
        seq_label_mask=mask,
    )

    expected = torch.nn.functional.softplus(torch.tensor(-2.0))
    assert torch.allclose(losses["seq_progress_loss"], expected)
    assert torch.allclose(
        losses["seq_progress_loss"], changed_losses["seq_progress_loss"]
    )

    total.backward()
    assert seq_logits.grad[0, 2].item() == 0.0
    assert seq_logits.grad[1, 1].item() == 0.0
    assert seq_logits.grad[1, 2].item() == 0.0


def test_sample_normalization_gives_short_and_long_prefixes_equal_weight():
    seq_logits = torch.tensor([[-4.0, 0.0, 0.0, 0.0], [4.0, 4.0, 4.0, 0.0]])
    seq_labels = torch.tensor([[1, 0, -1, -1], [1, 1, 1, 1]])
    mask = seq_labels.ne(-1).float()

    _, sample_losses = compute_stage2_losses(
        logits=torch.zeros(2),
        seq_logits=seq_logits,
        labels=torch.zeros(2),
        seq_labels=seq_labels,
        seq_label_mask=mask,
        seq_progress_weight=1.0,
        seq_normalization="sample",
    )
    _, token_losses = compute_stage2_losses(
        logits=torch.zeros(2),
        seq_logits=seq_logits,
        labels=torch.zeros(2),
        seq_labels=seq_labels,
        seq_label_mask=mask,
        seq_progress_weight=1.0,
        seq_normalization="token",
    )

    short = torch.nn.functional.softplus(torch.tensor(4.0))
    long = torch.nn.functional.softplus(torch.tensor(-4.0))
    assert torch.allclose(sample_losses["seq_progress_loss"], (short + long) / 2)
    assert torch.allclose(token_losses["seq_progress_loss"], (short + 3 * long) / 4)
    assert sample_losses["seq_progress_loss"] > token_losses["seq_progress_loss"]


def test_all_padding_sample_is_excluded_from_progress_reduction():
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
    total.backward()
    assert seq_logits.grad[0, 1].item() == 0.0
    assert torch.equal(seq_logits.grad[1], torch.zeros(2))


def test_empty_progress_mask_returns_differentiable_zero():
    seq_logits = torch.tensor(
        [[float("nan"), float("inf")], [3.0, -4.0]],
        requires_grad=True,
    )
    seq_labels = torch.tensor([[1, -1], [0, -1]])
    mask = seq_labels.ne(-1).float()

    total, losses = compute_stage2_losses(
        logits=torch.zeros(2),
        seq_logits=seq_logits,
        labels=torch.zeros(2),
        seq_labels=seq_labels,
        seq_label_mask=mask,
        seq_progress_weight=1.0,
    )

    assert losses["seq_progress_loss"].item() == 0.0
    assert torch.isfinite(total)
    total.backward()
    assert torch.equal(seq_logits.grad, torch.zeros_like(seq_logits))


def test_zero_width_progress_tensor_returns_stable_zero():
    seq_logits = torch.empty(2, 0, requires_grad=True)

    total, losses = compute_stage2_losses(
        logits=torch.zeros(2),
        seq_logits=seq_logits,
        labels=torch.zeros(2),
        seq_labels=torch.empty(2, 0),
        seq_label_mask=torch.empty(2, 0),
    )

    assert losses["seq_progress_loss"].item() == 0.0
    total.backward()
    assert seq_logits.grad is not None
    assert seq_logits.grad.numel() == 0


def test_total_loss_uses_configured_progress_and_ctc_weights():
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
        ctc_loss=ctc_loss,
        ctc_weight=0.2,
    )

    assert torch.allclose(losses["seq_loss"], 0.25 * losses["seq_progress_loss"])
    assert torch.allclose(
        total_loss,
        losses["utt_loss"] + losses["seq_loss"] + 0.2 * ctc_loss,
    )
    assert torch.allclose(losses["seq_loss"], losses["seq_progress_weighted_loss"])
    assert torch.allclose(losses["ctc_weighted_loss"], 0.2 * ctc_loss)
    assert torch.allclose(losses["total_loss"], total_loss)


def test_token_normalization_applies_only_to_non_final_prefixes():
    seq_logits = torch.tensor([[0.2, -0.1], [0.4, 8.0]])
    seq_labels = torch.tensor([[1.0, 0.0], [1.0, -1.0]])
    mask = seq_labels.ne(-1).float()
    expected = torch.nn.functional.binary_cross_entropy_with_logits(
        seq_logits[:1, :1],
        seq_labels[:1, :1],
    )

    _, losses = compute_stage2_losses(
        logits=torch.zeros(2),
        seq_logits=seq_logits,
        labels=torch.zeros(2),
        seq_labels=seq_labels,
        seq_label_mask=mask,
        seq_progress_weight=1.0,
        seq_normalization="token",
    )

    assert torch.allclose(losses["seq_loss"], expected)


def test_final_prefix_target_does_not_affect_progress_loss():
    common = {
        "logits": torch.zeros(1),
        "seq_logits": torch.tensor([[0.2, -0.4, 0.8]]),
        "labels": torch.zeros(1),
        "seq_label_mask": torch.ones(1, 3),
        "seq_progress_weight": 1.0,
    }
    _, positive_final = compute_stage2_losses(
        **common,
        seq_labels=torch.tensor([[1.0, 0.0, 1.0]]),
    )
    _, negative_final = compute_stage2_losses(
        **common,
        seq_labels=torch.tensor([[1.0, 0.0, 0.0]]),
    )

    assert torch.allclose(
        positive_final["seq_progress_loss"], negative_final["seq_progress_loss"]
    )


def test_seq_loss_rejects_invalid_configuration():
    common = {
        "logits": torch.zeros(1),
        "seq_logits": torch.zeros(1, 2),
        "labels": torch.zeros(1),
        "seq_labels": torch.zeros(1, 2),
        "seq_label_mask": torch.ones(1, 2),
    }
    for value in (-1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="finite and non-negative"):
            compute_stage2_losses(**common, seq_progress_weight=value)

    with pytest.raises(ValueError, match="normalization"):
        compute_stage2_losses(**common, seq_normalization="batch")


def test_negative_tail_cvar_selects_highest_scoring_negatives_only():
    logits = torch.tensor([2.0, 0.0, -2.0, 100.0], requires_grad=True)
    labels = torch.tensor([0.0, 0.0, 0.0, 1.0])

    loss = negative_tail_cvar_loss(logits, labels, fraction=0.5)

    expected = torch.nn.functional.softplus(torch.tensor([2.0, 0.0])).mean()
    assert torch.allclose(loss, expected)
    loss.backward()
    assert logits.grad[0].item() > 0.0
    assert logits.grad[1].item() > 0.0
    assert logits.grad[2].item() == 0.0
    assert logits.grad[3].item() == 0.0


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


def test_negative_tail_loss_returns_differentiable_zero_without_negatives():
    logits = torch.tensor([0.5, -0.2], requires_grad=True)

    loss = negative_tail_cvar_loss(
        logits,
        torch.ones_like(logits),
        fraction=0.1,
    )

    assert loss.item() == 0.0
    loss.backward()
    assert torch.equal(logits.grad, torch.zeros_like(logits))


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
