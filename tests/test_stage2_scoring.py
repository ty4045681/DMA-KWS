from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from dma_kws.stage2.scoring import (
    gather_completion_logits,
    gather_last_valid_logits,
)


def test_gather_completion_logits_uses_each_anchor_length() -> None:
    seq_logits = torch.tensor(
        [
            [1.0, 2.0, 99.0, 100.0],
            [3.0, 4.0, 5.0, 101.0],
            [7.0, 8.0, 9.0, 10.0],
        ]
    )

    gathered, valid = gather_completion_logits(
        seq_logits,
        torch.tensor([2, 3, 0]),
    )

    torch.testing.assert_close(gathered, torch.tensor([2.0, 5.0, 0.0]))
    assert valid.tolist() == [True, True, False]


def test_gather_last_valid_logits_supports_non_prefix_masks() -> None:
    logits = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    mask = torch.tensor([[True, False, True], [False, False, False]])

    gathered, valid = gather_last_valid_logits(logits, mask)

    torch.testing.assert_close(gathered, torch.tensor([3.0, 0.0]))
    assert valid.tolist() == [True, False]


def test_gather_completion_logits_rejects_lengths_beyond_width() -> None:
    with pytest.raises(ValueError, match="cannot exceed"):
        gather_completion_logits(torch.zeros(1, 2), torch.tensor([3]))


def test_gather_last_valid_logits_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="same shape"):
        gather_last_valid_logits(torch.zeros(2, 3), torch.ones(2, 2, dtype=torch.bool))
