"""Exact graph tests for the QbyT v6 segmental readout."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from qbyt.monotonic_alignment import (
    BoundedSegmentalAligner,
    SegmentalAlignmentOutput,
)


def _enumerate_paths(
    target_llr: torch.Tensor,
    phone_indices: tuple[int, ...],
    duration_log_probs: torch.Tensor,
    *,
    min_duration: int,
    max_duration: int,
    max_gap: int,
    max_span: int,
    deleted_phone_index: int | None = None,
) -> list[tuple[torch.Tensor, dict[int, torch.Tensor]]]:
    """Enumerate one unpadded query graph in relative-to-filler space.

    An internal one-phone deletion widens exactly the transition that crosses
    the deleted phone.  Its total filler-between length is canonicalized to one
    path per length in ``0..max_duration + max_gap``; it is not decomposed into
    separate edit-duration and ordinary-gap choices.  Leading and trailing
    deletions have no bracketing phone pair, so they add no unanchored filler
    span at all.
    """

    frame_count = target_llr.size(1)
    if not phone_indices:
        return [(target_llr.new_zeros(()), {})]

    paths: list[tuple[torch.Tensor, dict[int, torch.Tensor]]] = []

    def visit(
        position: int,
        *,
        first_start: int,
        previous_end: int,
        score: torch.Tensor,
        phone_scores: dict[int, torch.Tensor],
    ) -> None:
        if position == len(phone_indices):
            paths.append((score, phone_scores))
            return
        phone_index = phone_indices[position]
        if position == 0:
            gaps = (0,)
        else:
            previous_phone_index = phone_indices[position - 1]
            crosses_internal_deletion = (
                deleted_phone_index is not None
                and previous_phone_index < deleted_phone_index < phone_index
            )
            transition_limit = max_gap + (
                max_duration if crosses_internal_deletion else 0
            )
            gaps = range(transition_limit + 1)
        for gap in gaps:
            start = first_start if position == 0 else previous_end + gap
            for duration in range(min_duration, max_duration + 1):
                end = start + duration
                if end > frame_count or end - first_start > max_span:
                    continue
                local = target_llr[phone_index, start:end].sum()
                local = local + duration_log_probs[
                    phone_index, duration - min_duration
                ]
                visit(
                    position + 1,
                    first_start=first_start,
                    previous_end=end,
                    score=score + local,
                    phone_scores={**phone_scores, phone_index: local},
                )

    zero = target_llr.new_zeros(())
    for first_start in range(frame_count):
        visit(
            0,
            first_start=first_start,
            previous_end=first_start,
            score=zero,
            phone_scores={},
        )
    return paths


def _bruteforce(
    target_llr: torch.Tensor,
    filler_log_probs: torch.Tensor,
    duration_logits: torch.Tensor,
    *,
    min_duration: int,
    max_duration: int,
    max_gap: int,
    max_span: int,
    weakest_temperature: float,
) -> dict[str, torch.Tensor]:
    """Reference exact/filler/one-deletion partitions for one sample."""

    duration_log_probs = F.log_softmax(duration_logits.float(), dim=-1)
    duration_log_probs = duration_log_probs + math.log(duration_logits.size(-1))
    filler = filler_log_probs.float().sum()
    exact_values: list[torch.Tensor] = []
    alternative_values: list[torch.Tensor] = []
    deletion_values: list[torch.Tensor] = []
    phone_values: list[torch.Tensor] = []
    weakest_values: list[torch.Tensor] = []

    for prefix_end in range(target_llr.size(0)):
        phones = tuple(range(prefix_end + 1))
        exact_paths = _enumerate_paths(
            target_llr,
            phones,
            duration_log_probs,
            min_duration=min_duration,
            max_duration=max_duration,
            max_gap=max_gap,
            max_span=max_span,
        )
        exact_scores = torch.stack([score for score, _ in exact_paths])
        exact_relative = torch.logsumexp(exact_scores, dim=0)

        if prefix_end == 0:
            deletion_relative = target_llr.new_zeros(())
            alternative_relative = deletion_relative
        else:
            deleted_scores: list[torch.Tensor] = []
            for deleted in phones:
                retained = tuple(phone for phone in phones if phone != deleted)
                deleted_scores.extend(
                    score
                    for score, _ in _enumerate_paths(
                        target_llr,
                        retained,
                        duration_log_probs,
                        min_duration=min_duration,
                        max_duration=max_duration,
                        max_gap=max_gap,
                        max_span=max_span,
                        deleted_phone_index=deleted,
                    )
                )
            deletion_relative = torch.logsumexp(
                torch.stack(deleted_scores), dim=0
            )
            alternative_relative = torch.logaddexp(
                target_llr.new_zeros(()), deletion_relative
            )

        posterior = torch.softmax(exact_scores, dim=0)
        current_phone_scores = torch.stack(
            [phone_scores[prefix_end] for _, phone_scores in exact_paths]
        )
        phone_evidence = (posterior * current_phone_scores).sum()
        phone_values.append(phone_evidence)
        evidence = torch.stack(phone_values)
        weakest = -weakest_temperature * (
            torch.logsumexp(-evidence / weakest_temperature, dim=0)
            - math.log(prefix_end + 1)
        )

        exact_values.append(filler + exact_relative)
        alternative_values.append(filler + alternative_relative)
        deletion_values.append(filler + deletion_relative)
        weakest_values.append(weakest)

    keyword = torch.stack(exact_values)
    alternative = torch.stack(alternative_values)
    return {
        "keyword": keyword,
        "alternative": alternative,
        "one_edit": torch.stack(deletion_values),
        "prefix_llr": keyword - alternative,
        "phone_evidence": torch.stack(phone_values),
        "weakest": torch.stack(weakest_values),
    }


def _aligner(**overrides) -> BoundedSegmentalAligner:
    values = {
        "min_phone_duration_frames": 1,
        "max_phone_duration_frames": 2,
        "max_inter_phone_gap_frames": 1,
        "max_keyword_span_frames": 5,
        "temperature": 0.3,
    }
    values.update(overrides)
    return BoundedSegmentalAligner(**values)


def test_default_topology_matches_v6_contract():
    aligner = BoundedSegmentalAligner()

    assert aligner.min_phone_duration_frames == 1
    assert aligner.max_phone_duration_frames == 8
    assert aligner.max_inter_phone_gap_frames == 1
    assert aligner.max_keyword_span_frames == 30
    assert aligner.temperature == 0.2
    assert aligner.num_duration_bins == 8


def test_dynamic_program_matches_exhaustive_keyword_and_one_edit_graphs():
    target_llr = torch.tensor(
        [
            [
                [1.2, -0.7, 0.3, -1.1, 0.4],
                [-0.5, 1.5, -0.2, 0.8, -0.9],
                [0.1, -0.4, 1.1, -0.3, 0.7],
            ]
        ]
    )
    filler = torch.tensor([[-0.3, -0.7, -0.2, -1.0, -0.4]])
    duration_logits = torch.tensor(
        [[[0.2, -0.3], [-0.4, 0.7], [0.1, 0.4]]]
    )
    aligner = _aligner()

    actual = aligner(
        target_llr,
        filler,
        torch.tensor([3]),
        torch.tensor([5]),
        duration_log_probs=duration_logits,
    )
    expected = _bruteforce(
        target_llr[0],
        filler[0],
        duration_logits[0],
        min_duration=1,
        max_duration=2,
        max_gap=1,
        max_span=5,
        weakest_temperature=0.3,
    )

    assert isinstance(actual, SegmentalAlignmentOutput)
    torch.testing.assert_close(
        actual.keyword_log_partition[0], expected["keyword"], atol=1e-6, rtol=1e-6
    )
    torch.testing.assert_close(
        actual.alternative_log_partition[0],
        expected["alternative"],
        atol=1e-6,
        rtol=1e-6,
    )
    torch.testing.assert_close(
        actual.one_edit_log_partition[0],
        expected["one_edit"],
        atol=1e-6,
        rtol=1e-6,
    )
    torch.testing.assert_close(
        actual.prefix_llr[0], expected["prefix_llr"], atol=1e-6, rtol=1e-6
    )
    torch.testing.assert_close(
        actual.phone_evidence[0],
        expected["phone_evidence"],
        atol=1e-6,
        rtol=1e-6,
    )
    torch.testing.assert_close(
        actual.weakest_phone_evidence[0],
        expected["weakest"],
        atol=1e-6,
        rtol=1e-6,
    )
    torch.testing.assert_close(actual.raw_llr[0], expected["prefix_llr"][-1])
    assert actual.has_legal_path.all()


def test_partition_is_exact_logsumexp_not_path_mean_or_phone_average():
    aligner = _aligner(
        max_phone_duration_frames=1,
        max_inter_phone_gap_frames=0,
        max_keyword_span_frames=1,
    )
    value = 1.7
    output = aligner(
        torch.full((1, 1, 2), value),
        torch.zeros(1, 2),
        torch.tensor([1]),
        torch.tensor([2]),
    )

    # Two distinct legal starts contribute exp(value) each. The old readout
    # subtracted log(2); v6 intentionally retains the exact graph partition.
    expected = value + math.log(2.0)
    torch.testing.assert_close(output.prefix_llr[0, 0], torch.tensor(expected))
    torch.testing.assert_close(output.phone_evidence[0, 0], torch.tensor(value))


def test_all_valid_frames_are_scored_by_the_shared_filler_graph():
    generator = torch.Generator().manual_seed(13)
    target_llr = torch.randn(1, 2, 5, generator=generator)
    filler = torch.randn(1, 5, generator=generator)
    delta = torch.tensor([[0.4, -0.2, 0.7, -0.1, 0.3]])
    aligner = _aligner()

    before = aligner(
        target_llr, filler, torch.tensor([2]), torch.tensor([5])
    )
    after = aligner(
        target_llr, filler + delta, torch.tensor([2]), torch.tensor([5])
    )
    shift = delta.sum()

    torch.testing.assert_close(
        after.filler_log_partition, before.filler_log_partition + shift
    )
    torch.testing.assert_close(
        after.keyword_log_partition, before.keyword_log_partition + shift
    )
    torch.testing.assert_close(
        after.alternative_log_partition,
        before.alternative_log_partition + shift,
    )
    torch.testing.assert_close(
        after.one_edit_log_partition, before.one_edit_log_partition + shift
    )
    torch.testing.assert_close(after.prefix_llr, before.prefix_llr)
    torch.testing.assert_close(after.phone_evidence, before.phone_evidence)


def test_one_deletion_graph_vetoes_a_missing_middle_phone():
    aligner = _aligner(
        max_phone_duration_frames=1,
        max_inter_phone_gap_frames=1,
        max_keyword_span_frames=3,
        temperature=0.1,
    )
    complete = torch.full((1, 3, 3), -8.0)
    complete[0, 0, 0] = 6.0
    complete[0, 1, 1] = 6.0
    complete[0, 2, 2] = 6.0
    missing = complete.clone()
    missing[0, 1, 1] = -8.0
    lengths = torch.tensor([3])
    frames = torch.tensor([3])
    filler = torch.zeros(1, 3)

    complete_output = aligner(complete, filler, lengths, frames)
    missing_output = aligner(missing, filler, lengths, frames)

    assert complete_output.raw_llr > missing_output.raw_llr + 8.0
    assert missing_output.weakest_phone_evidence[0, -1] < -5.0
    # The missing phone is explained by filler while phones 0 and 2 remain in
    # order with one filler frame between them.
    assert missing_output.one_edit_log_partition[0, -1] > 10.0


def test_internal_deletion_bridges_one_bounded_canonical_filler_span():
    target = torch.full((1, 3, 6), -20.0)
    target[0, 0, 0] = 8.0
    target[0, 2, 4] = 8.0
    filler = torch.zeros(1, 6)
    lengths = torch.tensor([3])
    frames = torch.tensor([6])

    # With no ordinary gap, deleting phone 1 can still explain its three
    # filler frames and join phone 0 at frame 0 to phone 2 at frame 4.
    bounded = _aligner(
        max_phone_duration_frames=3,
        max_inter_phone_gap_frames=0,
        max_keyword_span_frames=6,
    )(target, filler, lengths, frames)
    assert bounded.one_edit_log_partition[0, -1] > 15.0

    # Move phone 2 one frame farther away: the four-frame separation is now
    # outside the deletion bound.  It becomes legal only when the configured
    # ordinary gap grows from zero to one; max_phone_duration alone must not
    # silently relax every normal inter-phone transition.
    too_far = target.clone()
    too_far[0, 2, 4] = -20.0
    too_far[0, 2, 5] = 8.0
    still_bounded = _aligner(
        max_phone_duration_frames=3,
        max_inter_phone_gap_frames=0,
        max_keyword_span_frames=6,
    )(too_far, filler, lengths, frames)
    gap_one = _aligner(
        max_phone_duration_frames=3,
        max_inter_phone_gap_frames=1,
        max_keyword_span_frames=6,
    )(too_far, filler, lengths, frames)

    assert still_bounded.one_edit_log_partition[0, -1] < 0.0
    assert gap_one.one_edit_log_partition[0, -1] > 15.0


def test_edge_deletions_do_not_duplicate_unanchored_filler_spans():
    target = torch.tensor(
        [
            [
                [-0.7, 1.1, -0.2, 0.4, -0.9, 0.3],
                [0.2, -0.4, 1.3, -0.8, 0.6, -0.1],
            ]
        ]
    )
    filler = torch.tensor([[-0.3, -0.2, -0.7, -0.1, -0.5, -0.4]])
    frames = torch.tensor([6])
    aligner = _aligner(
        max_phone_duration_frames=3,
        max_inter_phone_gap_frames=1,
        max_keyword_span_frames=6,
    )

    pair = aligner(target, filler, torch.tensor([2]), frames)
    leading_deleted = aligner(
        target[:, 1:2], filler, torch.tensor([1]), frames
    )
    trailing_deleted = aligner(
        target[:, 0:1], filler, torch.tensor([1]), frames
    )

    # A two-phone query has only these two one-deletion alternatives.  Any
    # enumeration of filler before the retained phone or after it would count
    # an acoustically identical edge deletion more than once and break this
    # exact union.
    expected = torch.logaddexp(
        leading_deleted.keyword_log_partition[0, 0],
        trailing_deleted.keyword_log_partition[0, 0],
    )
    torch.testing.assert_close(pair.one_edit_log_partition[0, 1], expected)


def test_padding_and_batch_companions_do_not_change_any_valid_output():
    generator = torch.Generator().manual_seed(4)
    target = torch.randn(1, 2, 5, generator=generator)
    filler = torch.randn(1, 5, generator=generator)
    duration = torch.randn(1, 2, 2, generator=generator)
    aligner = _aligner()

    alone = aligner(
        target,
        filler,
        torch.tensor([2]),
        torch.tensor([5]),
        duration_log_probs=duration,
    )

    batched_target = torch.full((2, 4, 9), 50.0)
    batched_target[0, :2, :5] = target[0]
    batched_target[1] = torch.randn(4, 9, generator=generator)
    batched_filler = torch.full((2, 9), 50.0)
    batched_filler[0, :5] = filler[0]
    batched_filler[1] = torch.randn(9, generator=generator)
    batched_duration = torch.full((2, 4, 2), 50.0)
    batched_duration[0, :2] = duration[0]
    batched_duration[1] = torch.randn(4, 2, generator=generator)
    together = aligner(
        batched_target,
        batched_filler,
        torch.tensor([2, 4]),
        torch.tensor([5, 9]),
        duration_log_probs=batched_duration,
    )

    for name in (
        "prefix_llr",
        "keyword_log_partition",
        "alternative_log_partition",
        "one_edit_log_partition",
        "phone_evidence",
        "weakest_phone_evidence",
    ):
        torch.testing.assert_close(
            getattr(alone, name)[0],
            getattr(together, name)[0, :2],
            atol=1e-6,
            rtol=1e-6,
        )
    torch.testing.assert_close(alone.raw_llr[0], together.raw_llr[0])
    torch.testing.assert_close(
        alone.filler_log_partition[0], together.filler_log_partition[0]
    )


def test_no_legal_path_and_empty_query_return_finite_invalid_scores():
    aligner = _aligner(
        min_phone_duration_frames=2,
        max_phone_duration_frames=3,
        max_inter_phone_gap_frames=0,
        max_keyword_span_frames=8,
    )
    output = aligner(
        torch.randn(2, 3, 4),
        torch.randn(2, 4),
        torch.tensor([3, 0]),
        torch.tensor([1, 4]),
    )

    for tensor in (
        output.raw_llr,
        output.prefix_llr,
        output.keyword_log_partition,
        output.alternative_log_partition,
        output.one_edit_log_partition,
        output.filler_log_partition,
        output.phone_evidence,
        output.weakest_phone_evidence,
    ):
        assert torch.isfinite(tensor).all()
    assert output.raw_llr.tolist() == [aligner.invalid_score, aligner.invalid_score]
    assert not output.has_legal_path.any()

    empty = aligner(
        torch.empty(2, 0, 4),
        torch.randn(2, 4),
        torch.tensor([0, 0]),
        torch.tensor([4, 3]),
    )
    assert empty.raw_llr.tolist() == [aligner.invalid_score, aligner.invalid_score]
    assert empty.prefix_llr.shape == (2, 0)
    assert empty.has_legal_path.shape == (2, 0)


def test_gradients_are_finite_and_padding_receives_exactly_zero():
    generator = torch.Generator().manual_seed(8)
    target = torch.randn(2, 3, 7, generator=generator, requires_grad=True)
    filler = torch.randn(2, 7, generator=generator, requires_grad=True)
    duration = torch.randn(2, 3, 3, generator=generator, requires_grad=True)
    aligner = _aligner(
        max_phone_duration_frames=3,
        max_keyword_span_frames=7,
        temperature=0.4,
    )
    output = aligner(
        target,
        filler,
        torch.tensor([3, 2]),
        torch.tensor([7, 5]),
        duration_log_probs=duration,
    )
    loss = (
        output.raw_llr.sum()
        + 0.1 * output.keyword_log_partition[0, :3].sum()
        + 0.1 * output.keyword_log_partition[1, :2].sum()
        + 0.1 * output.phone_evidence[0, :3].sum()
        + 0.1 * output.weakest_phone_evidence[1, :2].sum()
    )
    loss.backward()

    for gradient in (target.grad, filler.grad, duration.grad):
        assert gradient is not None
        assert torch.isfinite(gradient).all()
    assert target.grad[0].abs().sum() > 0
    assert target.grad[1, :2, :5].abs().sum() > 0
    assert filler.grad[:, :5].abs().sum() > 0
    assert duration.grad[0].abs().sum() > 0
    assert torch.count_nonzero(target.grad[1, 2:, :]) == 0
    assert torch.count_nonzero(target.grad[1, :, 5:]) == 0
    assert torch.count_nonzero(filler.grad[1, 5:]) == 0
    assert torch.count_nonzero(duration.grad[1, 2:, :]) == 0


def test_half_precision_inputs_still_produce_float32_outputs():
    aligner = _aligner()
    output = aligner(
        torch.randn(1, 2, 5, dtype=torch.float16),
        torch.randn(1, 5, dtype=torch.float16),
        torch.tensor([2]),
        torch.tensor([5]),
        duration_log_probs=torch.randn(1, 2, 2, dtype=torch.float16),
    )

    for tensor in (
        output.raw_llr,
        output.prefix_llr,
        output.keyword_log_partition,
        output.alternative_log_partition,
        output.one_edit_log_partition,
        output.filler_log_partition,
        output.phone_evidence,
        output.weakest_phone_evidence,
    ):
        assert tensor.dtype == torch.float32
        assert torch.isfinite(tensor).all()
    assert output.has_legal_path.dtype == torch.bool
