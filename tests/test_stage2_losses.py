import torch

from dma_kws.stage2.losses import compute_stage2_losses


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


def test_seq_label_mask_zeros_out_padded_positions():
    seq_logits = torch.tensor([[2.0, -2.0], [2.0, -2.0]])
    seq_labels = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    full_mask = torch.tensor([[1.0, 1.0], [1.0, 1.0]])
    padded_mask = torch.tensor([[1.0, 0.0], [1.0, 0.0]])

    _, losses_full = compute_stage2_losses(
        logits=torch.zeros(2),
        seq_logits=seq_logits,
        labels=torch.zeros(2),
        seq_labels=seq_labels,
        seq_label_mask=full_mask,
    )
    _, losses_padded = compute_stage2_losses(
        logits=torch.zeros(2),
        seq_logits=seq_logits,
        labels=torch.zeros(2),
        seq_labels=seq_labels,
        seq_label_mask=padded_mask,
    )

    assert losses_padded["seq_loss"].item() < losses_full["seq_loss"].item()
    assert losses_padded["seq_loss"].item() > 0.0


def test_total_loss_equals_sum_of_components():
    logits = torch.tensor([0.5, -0.3])
    seq_logits = torch.tensor([[0.2, -0.1], [0.4, 0.6]])
    labels = torch.tensor([1.0, 0.0])
    seq_labels = torch.tensor([[1.0, 0.0], [1.0, 1.0]])
    seq_label_mask = torch.tensor([[1.0, 1.0], [1.0, 0.0]])

    total_loss, losses = compute_stage2_losses(
        logits=logits,
        seq_logits=seq_logits,
        labels=labels,
        seq_labels=seq_labels,
        seq_label_mask=seq_label_mask,
    )

    assert torch.allclose(total_loss, losses["utt_loss"] + losses["seq_loss"])
