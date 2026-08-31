"""Exact keyword-vs-filler bounded segmental alignment for QbyT.

The readout operates on frame-level log likelihood ratios. A keyword path
assigns every query phone to one contiguous segment, keeps segments monotonic,
and obeys duration, inter-phone-gap, and total-span bounds. Every valid audio
frame starts in the shared filler graph; assigning a frame to a phone replaces
its filler score by adding that phone's ``target_llr``. Consequently gaps and
audio outside the keyword are explicitly filler-scored rather than skipped.

The competing graph is the union of the pure-filler path and paths obtained by
deleting exactly one query phone. With competitive target-vs-filler emissions,
this is an exact, useful first-order near-miss graph: a substituted or missing
phone is explained by filler while all remaining phones must still align. When
the deleted phone is internal, one canonical filler span may occupy up to one
phone duration in addition to the ordinary inter-phone gap. The one-phone case
deduplicates its deletion path from the identical filler path.

Unlike the previous readout, this module computes graph log partitions. It
does not subtract a path count, divide by query length, or average segment
frames. All DP arithmetic is float32, including under mixed precision.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class SegmentalAlignmentOutput:
    """Structured output of :class:`BoundedSegmentalAligner`.

    Shapes use ``B`` for batch size and ``U`` for padded query width.
    ``phone_evidence[:, i]`` is the posterior expected segment score of phone
    ``i`` under the exact graph for prefix ``0..i``. This makes
    ``weakest_phone_evidence[:, i]`` a legal-prefix cumulative soft minimum.
    Padded or impossible positions contain the aligner's finite invalid score.
    """

    raw_llr: torch.Tensor  # [B]
    prefix_llr: torch.Tensor  # [B, U]
    keyword_log_partition: torch.Tensor  # [B, U]
    alternative_log_partition: torch.Tensor  # [B, U]
    one_edit_log_partition: torch.Tensor  # [B, U]
    filler_log_partition: torch.Tensor  # [B]
    phone_evidence: torch.Tensor  # [B, U]
    weakest_phone_evidence: torch.Tensor  # [B, U]
    has_legal_path: torch.Tensor  # bool [B, U]


class BoundedSegmentalAligner(nn.Module):
    """Compute an exact bounded keyword-vs-filler segmental graph LLR.

    ``duration_log_probs``, when supplied to :meth:`forward`, has shape
    ``[B, U, D]`` where ``D = max_duration - min_duration + 1``. It is
    centered to zero log-mean-exp across ``D`` inside the aligner, preventing
    arbitrary per-phone offsets from leaking into the utterance score. Omitting
    it uses the neutral all-zero (uniform) duration potential.

    ``temperature`` controls only the weakest-phone soft minimum. Graph
    partitions themselves use exact unit-temperature log-sum-exp.
    """

    def __init__(
        self,
        *,
        min_phone_duration_frames: int = 1,
        max_phone_duration_frames: int = 8,
        max_inter_phone_gap_frames: int = 1,
        max_keyword_span_frames: int = 30,
        temperature: float = 0.2,
        invalid_score: float = -1.0e4,
    ) -> None:
        super().__init__()
        self.min_phone_duration_frames = self._positive_int(
            "min_phone_duration_frames", min_phone_duration_frames
        )
        self.max_phone_duration_frames = self._positive_int(
            "max_phone_duration_frames", max_phone_duration_frames
        )
        if self.max_phone_duration_frames < self.min_phone_duration_frames:
            raise ValueError(
                "max_phone_duration_frames must be greater than or equal to "
                "min_phone_duration_frames"
            )
        self.max_inter_phone_gap_frames = self._non_negative_int(
            "max_inter_phone_gap_frames", max_inter_phone_gap_frames
        )
        self.max_keyword_span_frames = self._positive_int(
            "max_keyword_span_frames", max_keyword_span_frames
        )

        if isinstance(temperature, bool):
            raise ValueError("temperature must be a finite number greater than 0")
        self.temperature = float(temperature)
        if not math.isfinite(self.temperature) or self.temperature <= 0.0:
            raise ValueError("temperature must be a finite number greater than 0")

        if isinstance(invalid_score, bool):
            raise ValueError("invalid_score must be a finite negative number")
        self.invalid_score = float(invalid_score)
        if not math.isfinite(self.invalid_score) or self.invalid_score >= 0.0:
            raise ValueError("invalid_score must be a finite negative number")

    @property
    def num_duration_bins(self) -> int:
        return (
            self.max_phone_duration_frames
            - self.min_phone_duration_frames
            + 1
        )

    @staticmethod
    def _positive_int(name: str, value: int) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
        return int(value)

    @staticmethod
    def _non_negative_int(name: str, value: int) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
        return int(value)

    @staticmethod
    def _lengths(
        value: torch.Tensor,
        *,
        name: str,
        batch_size: int,
        maximum: int,
        device: torch.device,
    ) -> torch.Tensor:
        if not isinstance(value, torch.Tensor):
            value = torch.as_tensor(value, device=device)
        else:
            value = value.to(device=device)
        if value.ndim != 1 or value.numel() != batch_size:
            raise ValueError(
                f"{name} must have shape [{batch_size}], got {tuple(value.shape)}"
            )
        if value.dtype == torch.bool or torch.is_floating_point(value):
            raise ValueError(f"{name} must contain integer lengths")
        value = value.to(dtype=torch.long)
        if torch.any(value < 0) or torch.any(value > maximum):
            raise ValueError(f"{name} entries must lie in [0, {maximum}]")
        return value

    @staticmethod
    def _safe_logsumexp(values: torch.Tensor, dim: int) -> torch.Tensor:
        """Log-sum-exp whose all-``-inf`` slices have finite backward."""

        has_value = torch.isfinite(values).any(dim=dim)
        safe_values = torch.where(
            has_value.unsqueeze(dim), values, torch.zeros_like(values)
        )
        reduced = torch.logsumexp(safe_values, dim=dim)
        return torch.where(
            has_value,
            reduced,
            torch.full_like(reduced, -torch.inf),
        )

    @classmethod
    def _merge_partitions(
        cls,
        left: torch.Tensor | None,
        right: torch.Tensor,
    ) -> torch.Tensor:
        if left is None:
            return right
        return cls._safe_logsumexp(torch.stack((left, right), dim=0), dim=0)

    @classmethod
    def _merge_with_expectation(
        cls,
        partition: torch.Tensor | None,
        expectation: torch.Tensor | None,
        candidate_partition: torch.Tensor,
        candidate_value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Merge path sets while retaining a scalar posterior expectation."""

        if partition is None:
            return candidate_partition, candidate_value
        assert expectation is not None
        values = torch.stack((partition, candidate_partition), dim=0)
        merged = cls._safe_logsumexp(values, dim=0)
        has_path = torch.isfinite(merged)
        denominator = torch.where(has_path, merged, torch.zeros_like(merged))
        weights = torch.where(
            torch.isfinite(values),
            torch.exp(values - denominator.unsqueeze(0)),
            torch.zeros_like(values),
        )
        moments = torch.stack((expectation, candidate_value), dim=0)
        return merged, (weights * moments).sum(dim=0)

    @classmethod
    def _reduce_partition(cls, partition: torch.Tensor) -> torch.Tensor:
        return cls._safe_logsumexp(partition.flatten(1), dim=1)

    @classmethod
    def _reduce_expectation(
        cls,
        partition: torch.Tensor,
        state_expectation: torch.Tensor,
        total_partition: torch.Tensor,
    ) -> torch.Tensor:
        has_path = torch.isfinite(total_partition)
        denominator = torch.where(
            has_path, total_partition, torch.zeros_like(total_partition)
        )
        weights = torch.where(
            torch.isfinite(partition),
            torch.exp(partition - denominator[:, None, None]),
            torch.zeros_like(partition),
        )
        return (weights * state_expectation).flatten(1).sum(dim=1)

    def _duration_potentials(
        self,
        duration_log_probs: torch.Tensor | None,
        *,
        batch_size: int,
        phone_width: int,
        device: torch.device,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        shape = (batch_size, phone_width, self.num_duration_bins)
        if duration_log_probs is None:
            return reference.new_zeros(shape)
        if not isinstance(duration_log_probs, torch.Tensor):
            raise ValueError(
                "duration_log_probs must be a floating-point tensor with shape "
                f"{list(shape)}"
            )
        if tuple(duration_log_probs.shape) != shape:
            raise ValueError(
                f"duration_log_probs must have shape {list(shape)}, got "
                f"{list(duration_log_probs.shape)}"
            )
        if not torch.is_floating_point(duration_log_probs):
            raise ValueError("duration_log_probs must be a floating-point tensor")
        values = duration_log_probs.to(device=device, dtype=torch.float32)
        if not torch.isfinite(values).all():
            raise ValueError("duration_log_probs must contain only finite values")
        return (
            values
            - torch.logsumexp(values, dim=-1, keepdim=True)
            + math.log(self.num_duration_bins)
        )

    def forward(
        self,
        target_llr: torch.Tensor,
        filler_log_probs: torch.Tensor,
        text_lengths: torch.Tensor,
        speech_lengths: torch.Tensor,
        *,
        duration_log_probs: torch.Tensor | None = None,
    ) -> SegmentalAlignmentOutput:
        """Score exact keyword and competing filler/one-deletion graphs."""

        if not isinstance(target_llr, torch.Tensor) or target_llr.ndim != 3:
            shape = getattr(target_llr, "shape", None)
            raise ValueError(f"target_llr must have shape [B, U, T], got {shape}")
        if not torch.is_floating_point(target_llr):
            raise ValueError("target_llr must be a floating-point tensor")
        if not isinstance(filler_log_probs, torch.Tensor) or filler_log_probs.ndim != 2:
            shape = getattr(filler_log_probs, "shape", None)
            raise ValueError(
                f"filler_log_probs must have shape [B, T], got {shape}"
            )

        batch_size, phone_width, frame_width = target_llr.shape
        if tuple(filler_log_probs.shape) != (batch_size, frame_width):
            raise ValueError(
                "filler_log_probs must match target_llr's batch/frame axes: "
                f"expected [{batch_size}, {frame_width}], got "
                f"{list(filler_log_probs.shape)}"
            )
        if not torch.is_floating_point(filler_log_probs):
            raise ValueError("filler_log_probs must be a floating-point tensor")

        text_lengths = self._lengths(
            text_lengths,
            name="text_lengths",
            batch_size=batch_size,
            maximum=phone_width,
            device=target_llr.device,
        )
        speech_lengths = self._lengths(
            speech_lengths,
            name="speech_lengths",
            batch_size=batch_size,
            maximum=frame_width,
            device=target_llr.device,
        )

        target_llr = target_llr.float()
        filler_log_probs = filler_log_probs.to(
            device=target_llr.device, dtype=torch.float32
        )
        if not torch.isfinite(target_llr).all():
            raise ValueError("target_llr must contain only finite values")
        if not torch.isfinite(filler_log_probs).all():
            raise ValueError("filler_log_probs must contain only finite values")

        duration_potentials = self._duration_potentials(
            duration_log_probs,
            batch_size=batch_size,
            phone_width=phone_width,
            device=target_llr.device,
            reference=target_llr,
        )
        frame_mask = (
            torch.arange(frame_width, device=target_llr.device).unsqueeze(0)
            < speech_lengths.unsqueeze(1)
        )
        filler_baseline = torch.where(
            frame_mask, filler_log_probs, torch.zeros_like(filler_log_probs)
        ).sum(dim=1)

        output_shape = (batch_size, phone_width)
        if phone_width == 0:
            empty = target_llr.new_empty(output_shape)
            return SegmentalAlignmentOutput(
                raw_llr=target_llr.new_full((batch_size,), self.invalid_score),
                prefix_llr=empty,
                keyword_log_partition=empty,
                alternative_log_partition=empty,
                one_edit_log_partition=empty,
                filler_log_partition=filler_baseline,
                phone_evidence=empty,
                weakest_phone_evidence=empty,
                has_legal_path=torch.empty(
                    output_shape, device=target_llr.device, dtype=torch.bool
                ),
            )

        cumulative = F.pad(target_llr.cumsum(dim=-1), (1, 0))
        state_shape = (
            batch_size,
            frame_width + 1,
            self.max_keyword_span_frames + 1,
        )
        negative_infinity = target_llr.new_tensor(-torch.inf)

        def first_phone_candidates(phone_index: int):
            active_phone = text_lengths.gt(phone_index)
            for duration in range(
                self.min_phone_duration_frames,
                self.max_phone_duration_frames + 1,
            ):
                if duration > frame_width or duration > self.max_keyword_span_frames:
                    continue
                segment_sum = (
                    cumulative[:, phone_index, duration:]
                    - cumulative[:, phone_index, :-duration]
                )
                local_score = segment_sum + duration_potentials[
                    :, phone_index, duration - self.min_phone_duration_frames
                ].unsqueeze(1)
                ends = torch.arange(
                    duration, frame_width + 1, device=target_llr.device
                )
                valid = (
                    ends.unsqueeze(0) <= speech_lengths.unsqueeze(1)
                ) & active_phone.unsqueeze(1)
                core = torch.where(valid, local_score, negative_infinity)
                partition = F.pad(
                    core.unsqueeze(-1),
                    (
                        duration,
                        self.max_keyword_span_frames - duration,
                        duration,
                        0,
                    ),
                    value=-torch.inf,
                )
                local = F.pad(
                    torch.where(valid, local_score, torch.zeros_like(local_score))
                    .unsqueeze(-1),
                    (
                        duration,
                        self.max_keyword_span_frames - duration,
                        duration,
                        0,
                    ),
                    value=0.0,
                )
                yield partition, local

        def advance_candidates(
            previous: torch.Tensor,
            phone_index: int,
            *,
            max_gap_frames: int | None = None,
        ):
            active_phone = text_lengths.gt(phone_index)
            gap_limit = (
                self.max_inter_phone_gap_frames
                if max_gap_frames is None
                else max_gap_frames
            )
            for duration in range(
                self.min_phone_duration_frames,
                self.max_phone_duration_frames + 1,
            ):
                if duration > frame_width or duration > self.max_keyword_span_frames:
                    continue
                segment_sum = (
                    cumulative[:, phone_index, duration:]
                    - cumulative[:, phone_index, :-duration]
                )
                duration_potential = duration_potentials[
                    :, phone_index, duration - self.min_phone_duration_frames
                ].view(batch_size, 1, 1)
                for gap in range(gap_limit + 1):
                    increment = duration + gap
                    if (
                        increment > frame_width
                        or increment > self.max_keyword_span_frames
                    ):
                        continue
                    predecessor = previous[
                        :, : frame_width + 1 - increment,
                        : self.max_keyword_span_frames + 1 - increment,
                    ]
                    current = segment_sum[:, gap:].unsqueeze(-1) + duration_potential
                    if current.size(1) != predecessor.size(1):
                        raise RuntimeError("internal segmental DP shape mismatch")
                    current = current.expand_as(predecessor)
                    core = predecessor + current
                    ends = torch.arange(
                        increment, frame_width + 1, device=target_llr.device
                    )
                    valid = (
                        ends.unsqueeze(0) <= speech_lengths.unsqueeze(1)
                    ) & active_phone.unsqueeze(1)
                    core = torch.where(valid.unsqueeze(2), core, negative_infinity)
                    partition = F.pad(
                        core,
                        (increment, 0, increment, 0),
                        value=-torch.inf,
                    )
                    local = F.pad(
                        torch.where(
                            valid.unsqueeze(2), current, torch.zeros_like(current)
                        ),
                        (increment, 0, increment, 0),
                        value=0.0,
                    )
                    yield partition, local

        exact_partition: torch.Tensor | None = None
        # A deleted phone becomes ``resolved`` once a later phone has aligned.
        # Keeping the trailing deletion separate lets exactly that next
        # transition consume one phone-sized filler span; subsequent gaps return
        # to the ordinary bound, so the graph still contains exactly one edit.
        resolved_deleted_partition: torch.Tensor | None = None
        delete_current_partition: torch.Tensor | None = None
        prefix_llrs: list[torch.Tensor] = []
        keyword_partitions: list[torch.Tensor] = []
        alternative_partitions: list[torch.Tensor] = []
        one_edit_partitions: list[torch.Tensor] = []
        phone_evidences: list[torch.Tensor] = []
        weakest_evidences: list[torch.Tensor] = []
        path_masks: list[torch.Tensor] = []

        for phone_index in range(phone_width):
            previous_exact = exact_partition
            previous_resolved_deleted = resolved_deleted_partition
            previous_delete_current = delete_current_partition

            exact_partition = None
            exact_local_expectation: torch.Tensor | None = None
            if phone_index == 0:
                exact_candidates = first_phone_candidates(phone_index)
            else:
                assert previous_exact is not None
                exact_candidates = advance_candidates(previous_exact, phone_index)
            for candidate, local in exact_candidates:
                exact_partition, exact_local_expectation = self._merge_with_expectation(
                    exact_partition,
                    exact_local_expectation,
                    candidate,
                    local,
                )
            if exact_partition is None:
                exact_partition = target_llr.new_full(state_shape, -torch.inf)
                exact_local_expectation = target_llr.new_zeros(state_shape)
            assert exact_local_expectation is not None

            # Deleting the current (trailing) phone preserves the exact previous
            # state and has one canonical path. If the deleted phone becomes
            # internal on the next iteration, the special transition below may
            # consume up to one phone duration of filler. Leading deletion is
            # canonicalized by starting the first kept phone from scratch.
            if phone_index == 0:
                delete_current_partition = None
                resolved_deleted_partition = None
                deleted_partition = None  # empty deletion path is pure filler
            else:
                active = text_lengths.gt(phone_index).view(batch_size, 1, 1)
                assert previous_exact is not None
                delete_current_partition = torch.where(
                    active, previous_exact, negative_infinity
                )
                resolved_deleted_partition = None
                if phone_index == 1:
                    earlier_deletion_candidates = first_phone_candidates(phone_index)
                    for candidate, _ in earlier_deletion_candidates:
                        resolved_deleted_partition = self._merge_partitions(
                            resolved_deleted_partition, candidate
                        )
                else:
                    assert previous_resolved_deleted is not None
                    assert previous_delete_current is not None
                    for candidate, _ in advance_candidates(
                        previous_resolved_deleted,
                        phone_index,
                    ):
                        resolved_deleted_partition = self._merge_partitions(
                            resolved_deleted_partition, candidate
                        )
                    for candidate, _ in advance_candidates(
                        previous_delete_current,
                        phone_index,
                        max_gap_frames=(
                            self.max_inter_phone_gap_frames
                            + self.max_phone_duration_frames
                        ),
                    ):
                        resolved_deleted_partition = self._merge_partitions(
                            resolved_deleted_partition, candidate
                        )
                if resolved_deleted_partition is None:
                    resolved_deleted_partition = target_llr.new_full(
                        state_shape, -torch.inf
                    )
                deleted_partition = self._merge_partitions(
                    delete_current_partition,
                    resolved_deleted_partition,
                )

            keyword_relative = self._reduce_partition(exact_partition)
            active_phone = text_lengths.gt(phone_index)
            has_path = active_phone & torch.isfinite(keyword_relative)

            if phone_index == 0:
                one_edit_relative = torch.zeros_like(filler_baseline)
            else:
                assert deleted_partition is not None
                one_edit_relative = self._reduce_partition(deleted_partition)
            zero = torch.zeros_like(one_edit_relative)
            # U=1 deletion and pure filler are identical: do not double-count.
            alternative_relative = (
                zero
                if phone_index == 0
                else torch.logaddexp(zero, one_edit_relative)
            )

            keyword_absolute = filler_baseline + keyword_relative
            alternative_absolute = filler_baseline + alternative_relative
            one_edit_absolute = filler_baseline + one_edit_relative
            keyword_absolute = torch.where(
                has_path,
                keyword_absolute,
                torch.full_like(keyword_absolute, self.invalid_score),
            )
            alternative_absolute = torch.where(
                active_phone,
                alternative_absolute,
                torch.full_like(alternative_absolute, self.invalid_score),
            )
            one_edit_absolute = torch.where(
                active_phone & torch.isfinite(one_edit_relative),
                one_edit_absolute,
                torch.full_like(one_edit_absolute, self.invalid_score),
            )
            prefix_llr = torch.where(
                has_path,
                keyword_absolute - alternative_absolute,
                torch.full_like(keyword_absolute, self.invalid_score),
            )

            phone_evidence = self._reduce_expectation(
                exact_partition,
                exact_local_expectation,
                keyword_relative,
            )
            phone_evidence = torch.where(
                has_path,
                phone_evidence,
                torch.full_like(phone_evidence, self.invalid_score),
            )
            phone_evidences.append(phone_evidence)
            evidence_so_far = torch.stack(phone_evidences, dim=1)
            safe_evidence = torch.where(
                has_path.unsqueeze(1),
                evidence_so_far,
                torch.zeros_like(evidence_so_far),
            )
            weakest = -self.temperature * (
                torch.logsumexp(-safe_evidence / self.temperature, dim=1)
                - math.log(phone_index + 1)
            )
            weakest = torch.where(
                has_path,
                weakest,
                torch.full_like(weakest, self.invalid_score),
            )

            prefix_llrs.append(prefix_llr)
            keyword_partitions.append(keyword_absolute)
            alternative_partitions.append(alternative_absolute)
            one_edit_partitions.append(one_edit_absolute)
            weakest_evidences.append(weakest)
            path_masks.append(has_path)

        prefix_llr_tensor = torch.stack(prefix_llrs, dim=1)
        keyword_partition_tensor = torch.stack(keyword_partitions, dim=1)
        alternative_partition_tensor = torch.stack(alternative_partitions, dim=1)
        one_edit_partition_tensor = torch.stack(one_edit_partitions, dim=1)
        phone_evidence_tensor = torch.stack(phone_evidences, dim=1)
        weakest_evidence_tensor = torch.stack(weakest_evidences, dim=1)
        path_mask_tensor = torch.stack(path_masks, dim=1)

        final_index = (text_lengths - 1).clamp_min(0)
        raw_llr = prefix_llr_tensor.gather(
            1, final_index.unsqueeze(1)
        ).squeeze(1)
        final_has_path = path_mask_tensor.gather(
            1, final_index.unsqueeze(1)
        ).squeeze(1)
        raw_llr = torch.where(
            text_lengths.gt(0) & final_has_path,
            raw_llr,
            torch.full_like(raw_llr, self.invalid_score),
        )

        return SegmentalAlignmentOutput(
            raw_llr=raw_llr,
            prefix_llr=prefix_llr_tensor,
            keyword_log_partition=keyword_partition_tensor,
            alternative_log_partition=alternative_partition_tensor,
            one_edit_log_partition=one_edit_partition_tensor,
            filler_log_partition=filler_baseline,
            phone_evidence=phone_evidence_tensor,
            weakest_phone_evidence=weakest_evidence_tensor,
            has_legal_path=path_mask_tensor,
        )

    def extra_repr(self) -> str:
        return (
            f"min_phone_duration_frames={self.min_phone_duration_frames}, "
            f"max_phone_duration_frames={self.max_phone_duration_frames}, "
            f"max_inter_phone_gap_frames={self.max_inter_phone_gap_frames}, "
            f"max_keyword_span_frames={self.max_keyword_span_frames}, "
            f"weakest_temperature={self.temperature}, "
            f"invalid_score={self.invalid_score}"
        )


__all__ = ["BoundedSegmentalAligner", "SegmentalAlignmentOutput"]
