import pytest

torch = pytest.importorskip("torch")

from dma_kws.metrics import collapse_ctc
from dma_kws.phoneme_adapter.module import build_phoneme_adapter

VOCAB_SIZE = 71


def _mask(lengths: list[int], frames: int) -> torch.Tensor:
    lens = torch.tensor(lengths, dtype=torch.long)
    return (torch.arange(frames).unsqueeze(0) < lens.unsqueeze(1)).unsqueeze(1)


def _adapter(**overrides):
    cfg = {"trunk": {"type": "conv", "output_dim": 24, "num_layers": 1, "kernel_size": 3}}
    cfg.update(overrides)
    return build_phoneme_adapter(cfg, input_dim=16, vocab_size=VOCAB_SIZE)


def test_forward_returns_trunk_features_and_log_probs():
    adapter = _adapter()
    features, log_probs = adapter(torch.randn(2, 9, 16), _mask([9, 5], 9))

    assert features.shape == (2, 9, 24)
    assert log_probs.shape == (2, 9, VOCAB_SIZE)
    assert adapter.output_dim == 24
    torch.testing.assert_close(
        log_probs.exp().sum(dim=-1), torch.ones(2, 9), rtol=1e-4, atol=1e-4
    )


def test_expose_posterior_concatenates_and_widens_output():
    adapter = _adapter(expose_posterior=True)
    features, log_probs = adapter(torch.randn(1, 4, 16), None)

    assert adapter.output_dim == 24 + VOCAB_SIZE
    assert features.shape == (1, 4, 24 + VOCAB_SIZE)
    torch.testing.assert_close(features[..., 24:], log_probs)


def test_ctc_loss_skips_samples_with_more_labels_than_frames():
    """CTC cannot align more labels than frames, and at 25 Hz a short phrase clip
    can genuinely have fewer frames than phonemes. Those samples are dropped and
    counted so a silently biased auxiliary loss is visible."""
    adapter = _adapter()
    encoder_mask = _mask([5, 2], 5)
    _features, log_probs = adapter(torch.randn(2, 5, 16), encoder_mask)
    targets = torch.tensor([[3, 4, 5, 0, 0], [3, 4, 5, 6, 7]], dtype=torch.long)
    target_lengths = torch.tensor([3, 5], dtype=torch.long)

    loss, num_skipped = adapter.ctc_loss(log_probs, encoder_mask, targets, target_lengths)

    assert num_skipped == 1
    assert torch.isfinite(loss)


def test_ctc_loss_returns_zero_when_every_sample_is_skipped():
    adapter = _adapter()
    mask = _mask([2], 2)
    _features, log_probs = adapter(torch.randn(1, 2, 16), mask)
    targets = torch.tensor([[3, 4, 5, 6]], dtype=torch.long)
    target_lengths = torch.tensor([4], dtype=torch.long)

    loss, num_skipped = adapter.ctc_loss(log_probs, mask, targets, target_lengths)

    assert num_skipped == 1
    assert float(loss.detach()) == 0.0


def test_ctc_loss_is_finite_for_ragged_batches():
    adapter = _adapter()
    encoder_mask = _mask([40, 25, 12], 40)
    _features, log_probs = adapter(torch.randn(3, 40, 16), encoder_mask)
    targets = torch.zeros(3, 8, dtype=torch.long)
    targets[:, :4] = torch.tensor([[3, 4, 5, 6], [7, 8, 9, 10], [11, 12, 13, 14]])
    target_lengths = torch.tensor([4, 4, 4], dtype=torch.long)

    loss, num_skipped = adapter.ctc_loss(log_probs, encoder_mask, targets, target_lengths)

    assert num_skipped == 0
    assert torch.isfinite(loss)


def test_ctc_loss_supervises_the_same_realization_qbyt_reads():
    """One adapter call feeds both consumers. Recomputing the trunk for the CTC
    loss would, with dropout on, supervise a different sample than the matcher
    saw and dissolve the coupling the trunk exists to create."""
    adapter = _adapter()
    adapter.train()
    mask = _mask([30], 30)

    features_a, log_probs_a = adapter(torch.randn(1, 30, 16), mask)
    torch.manual_seed(0)
    features_b, _ = adapter(torch.randn(1, 30, 16), mask)

    # Dropout makes two calls differ, which is why the caller must reuse one.
    assert not torch.allclose(features_a, features_b)
    torch.testing.assert_close(log_probs_a, adapter.ctc.log_softmax(features_a))


def test_adapter_overfits_a_tiny_batch():
    """End-to-end sanity check of the G2P/length/mask plumbing.

    If the adapter cannot drive CTC loss to ~0 and reproduce the labels by greedy
    decoding on a handful of fixed samples, the failure is in the wiring, not in
    the frozen representation, and there is no point looking at dev PER.
    """
    torch.manual_seed(0)
    adapter = build_phoneme_adapter(
        {"trunk": {"type": "conv", "output_dim": 64, "num_layers": 2, "kernel_size": 3}},
        input_dim=16,
        vocab_size=VOCAB_SIZE,
    )
    encoder_out = torch.randn(4, 30, 16)
    encoder_mask = _mask([30, 30, 30, 30], 30)
    targets = torch.tensor(
        [[3, 4, 5], [6, 7, 8], [9, 10, 11], [12, 13, 14]], dtype=torch.long
    )
    target_lengths = torch.tensor([3, 3, 3, 3], dtype=torch.long)

    optimizer = torch.optim.Adam(adapter.parameters(), lr=5e-3)
    for _ in range(300):
        optimizer.zero_grad()
        _features, log_probs = adapter(encoder_out, encoder_mask)
        loss, _ = adapter.ctc_loss(log_probs, encoder_mask, targets, target_lengths)
        loss.backward()
        optimizer.step()

    assert float(loss.detach()) < 0.05

    adapter.eval()
    with torch.no_grad():
        _, log_probs = adapter(encoder_out, encoder_mask)
    for i in range(4):
        hypothesis = collapse_ctc(log_probs[i].argmax(dim=-1).tolist(), blank_id=0)
        assert hypothesis == targets[i].tolist()
