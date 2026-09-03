"""Bounded monotonic alignment must reject scattered phone evidence."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from qbyt.bounded_alignment import BoundedSegmentalAligner


def _bruteforce_prefix_evidence(
    emissions: torch.Tensor,
    *,
    min_duration: int,
    max_duration: int,
    max_gap: int,
    max_span: int,
    temperature: float,
) -> torch.Tensor:
    """Enumerate every legal path for one unpadded [U, T] lattice."""

    phone_count, frame_count = emissions.shape
    local = F.logsigmoid(emissions.float())
    paths: list[list[torch.Tensor]] = [[] for _ in range(phone_count)]

    def visit(
        phone_index: int,
        *,
        first_start: int,
        previous_end: int,
        energy: torch.Tensor,
    ) -> None:
        if phone_index == phone_count:
            return
        gaps = (0,) if phone_index == 0 else range(max_gap + 1)
        for gap in gaps:
            segment_start = first_start if phone_index == 0 else previous_end + gap
            for duration in range(min_duration, max_duration + 1):
                segment_end = segment_start + duration
                if segment_end > frame_count:
                    continue
                if segment_end - first_start > max_span:
                    continue
                segment = local[phone_index, segment_start:segment_end].mean()
                next_energy = energy + segment
                paths[phone_index].append(next_energy)
                visit(
                    phone_index + 1,
                    first_start=first_start,
                    previous_end=segment_end,
                    energy=next_energy,
                )

    zero = emissions.new_zeros((), dtype=torch.float32)
    for first_start in range(frame_count):
        visit(0, first_start=first_start, previous_end=first_start, energy=zero)

    scores = []
    for phone_index, energies in enumerate(paths):
        if not energies:
            scores.append(emissions.new_tensor(-1.0e4, dtype=torch.float32))
            continue
        values = torch.stack(energies)
        log_mean_exp = temperature * (
            torch.logsumexp(values / temperature, dim=0) - math.log(len(energies))
        )
        scores.append(log_mean_exp / float(phone_index + 1))
    return torch.stack(scores)


def _single_frame_aligner(*, max_gap: int, max_span: int) -> BoundedSegmentalAligner:
    return BoundedSegmentalAligner(
        min_phone_duration_frames=1,
        max_phone_duration_frames=1,
        max_inter_phone_gap_frames=max_gap,
        max_keyword_span_frames=max_span,
        temperature=0.02,
    )


def test_default_topology_matches_checkpoint_contract():
    aligner = BoundedSegmentalAligner()

    assert aligner.min_phone_duration_frames == 1
    assert aligner.max_phone_duration_frames == 8
    assert aligner.max_inter_phone_gap_frames == 2
    assert aligner.max_keyword_span_frames == 30
    assert aligner.temperature == 0.2


def test_compact_ordered_evidence_beats_the_same_peaks_in_reverse_order():
    compact = torch.full((1, 3, 7), -8.0)
    compact[0, 0, 2] = 8.0
    compact[0, 1, 3] = 8.0
    compact[0, 2, 4] = 8.0

    reversed_order = torch.full_like(compact, -8.0)
    reversed_order[0, 0, 4] = 8.0
    reversed_order[0, 1, 3] = 8.0
    reversed_order[0, 2, 2] = 8.0

    aligner = _single_frame_aligner(max_gap=0, max_span=3)
    lengths = torch.tensor([3])
    speech_lengths = torch.tensor([7])
    compact_score = aligner(compact, lengths, speech_lengths)[0, -1]
    reversed_score = aligner(reversed_order, lengths, speech_lengths)[0, -1]

    assert compact_score > reversed_score + 3.0


def test_gap_and_total_span_bounds_block_scattered_evidence():
    emissions = torch.full((1, 2, 6), -8.0)
    emissions[0, 0, 1] = 8.0
    emissions[0, 1, 4] = 8.0
    text_lengths = torch.tensor([2])
    speech_lengths = torch.tensor([6])

    permissive = _single_frame_aligner(max_gap=2, max_span=4)(
        emissions, text_lengths, speech_lengths
    )[0, -1]
    gap_blocked = _single_frame_aligner(max_gap=1, max_span=4)(
        emissions, text_lengths, speech_lengths
    )[0, -1]
    span_blocked = _single_frame_aligner(max_gap=2, max_span=3)(
        emissions, text_lengths, speech_lengths
    )[0, -1]

    assert permissive > gap_blocked + 2.0
    assert permissive > span_blocked + 2.0


def test_padding_and_batch_companions_do_not_change_prefix_evidence():
    generator = torch.Generator().manual_seed(4)
    sample = torch.randn(1, 2, 5, generator=generator)
    aligner = BoundedSegmentalAligner(
        min_phone_duration_frames=1,
        max_phone_duration_frames=2,
        max_inter_phone_gap_frames=1,
        max_keyword_span_frames=5,
        temperature=0.3,
    )

    alone = aligner(sample, torch.tensor([2]), torch.tensor([5]))[0]

    batched = torch.full((2, 4, 9), 50.0)
    batched[0, :2, :5] = sample[0]
    batched[1] = torch.randn(4, 9, generator=generator)
    with_companion = aligner(
        batched,
        torch.tensor([2, 4]),
        torch.tensor([5, 9]),
    )[0, :2]

    torch.testing.assert_close(alone, with_companion, atol=1e-6, rtol=1e-6)


def test_dynamic_program_matches_exhaustive_path_enumeration():
    emissions = torch.tensor(
        [
            [
                [1.2, -0.7, 0.3, -1.1],
                [-0.5, 1.5, -0.2, 0.8],
            ]
        ]
    )
    temperature = 0.7
    aligner = BoundedSegmentalAligner(
        min_phone_duration_frames=1,
        max_phone_duration_frames=2,
        max_inter_phone_gap_frames=1,
        max_keyword_span_frames=4,
        temperature=temperature,
    )

    actual = aligner(emissions, torch.tensor([2]), torch.tensor([4]))[0]
    expected = _bruteforce_prefix_evidence(
        emissions[0],
        min_duration=1,
        max_duration=2,
        max_gap=1,
        max_span=4,
        temperature=temperature,
    )

    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)


def test_no_legal_path_returns_a_finite_large_negative_score():
    aligner = BoundedSegmentalAligner(
        min_phone_duration_frames=2,
        max_phone_duration_frames=3,
        max_inter_phone_gap_frames=0,
        max_keyword_span_frames=8,
        temperature=0.2,
    )
    scores = aligner(
        torch.randn(1, 3, 4),
        torch.tensor([3]),
        torch.tensor([1]),
    )

    assert torch.isfinite(scores).all()
    assert torch.equal(scores, torch.full_like(scores, aligner.invalid_score))


def test_gradients_are_finite_and_padding_receives_none():
    emissions = torch.randn(
        2,
        3,
        7,
        generator=torch.Generator().manual_seed(8),
        requires_grad=True,
    )
    aligner = BoundedSegmentalAligner(
        min_phone_duration_frames=1,
        max_phone_duration_frames=3,
        max_inter_phone_gap_frames=1,
        max_keyword_span_frames=7,
        temperature=0.4,
    )
    scores = aligner(
        emissions,
        torch.tensor([3, 2]),
        torch.tensor([7, 5]),
    )
    loss = scores[0, :3].sum() + scores[1, :2].sum()
    loss.backward()

    assert emissions.grad is not None
    assert torch.isfinite(emissions.grad).all()
    assert emissions.grad[0].abs().sum() > 0
    assert emissions.grad[1, :2, :5].abs().sum() > 0
    assert torch.count_nonzero(emissions.grad[1, 2:, :]) == 0
    assert torch.count_nonzero(emissions.grad[1, :, 5:]) == 0


def test_half_precision_input_still_produces_float32_finite_evidence():
    aligner = _single_frame_aligner(max_gap=1, max_span=5)
    scores = aligner(
        torch.randn(1, 2, 5, dtype=torch.float16),
        torch.tensor([2]),
        torch.tensor([5]),
    )

    assert scores.dtype == torch.float32
    assert torch.isfinite(scores).all()
