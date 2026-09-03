"""Differentiable bounded segmental alignment for QbyT emissions.

The aligner consumes a query-conditioned phone/frame emission lattice.  A
legal path assigns every completed phone to one contiguous audio segment,
keeps those segments in order, permits only a bounded gap between neighbours,
and caps the complete keyword span.  Audio before the first segment and after
the last segment is free, so a keyword may occur anywhere in the valid clip.

The dynamic program deliberately operates on ``logsigmoid`` evidence.  Local
segment evidence is therefore at most zero: one exceptionally strong phone
cannot numerically compensate for a missing phone.  Each prefix is a
temperature-controlled log-mean-exp over *legal paths*, not a log-sum-exp; the
same topology is evaluated with zero energy to remove path-count bias.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


class BoundedSegmentalAligner(nn.Module):
    """Score every anchor prefix with a bounded monotonic segmental DP.

    Args:
        min_phone_duration_frames: Minimum number of audio frames assigned to
            each phone.
        max_phone_duration_frames: Maximum number of audio frames assigned to
            each phone.
        max_inter_phone_gap_frames: Maximum number of unscored frames between
            adjacent phone segments.  There is no unbounded CTC-style blank.
        max_keyword_span_frames: Maximum distance from the first segment's
            start to the last segment's end, including internal gaps.
        temperature: Positive soft-Viterbi temperature.  The returned value is
            a log-mean-exp over legal paths at this temperature.
        invalid_score: Finite value returned when a prefix has no legal path or
            is padding according to ``text_lengths``.

    ``forward`` accepts raw emission logits shaped ``[B, U, T]`` and returns
    normalized prefix evidence shaped ``[B, U]``.  Position ``i`` contains the
    path-count-normalized evidence for completing phones ``0..i``, divided by
    ``i + 1`` so anchors of different lengths remain on a comparable scale.

    The module is parameter-free.  All dynamic-programming arithmetic is
    forced to float32 even under autocast or when emissions are float16.
    """

    def __init__(
        self,
        *,
        min_phone_duration_frames: int = 1,
        max_phone_duration_frames: int = 8,
        max_inter_phone_gap_frames: int = 2,
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
        """Log-sum-exp whose all--inf slices also have a finite backward pass."""

        has_value = torch.isfinite(values).any(dim=dim)
        # torch.logsumexp([-inf, -inf]) returns -inf in forward but NaN gradients.
        # Replace an entirely empty reduction by constants before evaluating it,
        # then restore the structural -inf after the differentiable operation.
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
    def _combine(
        cls,
        candidates: list[torch.Tensor],
        shape: tuple[int, ...],
        ref: torch.Tensor,
    ) -> torch.Tensor:
        if not candidates:
            return ref.new_full(shape, -torch.inf)
        return cls._safe_logsumexp(torch.stack(candidates, dim=0), dim=0)

    def forward(
        self,
        emissions: torch.Tensor,
        text_lengths: torch.Tensor,
        speech_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Return normalized evidence for every completed anchor prefix.

        ``emissions[b, i, t]`` is the raw match logit between anchor phone ``i``
        and speech frame ``t``.  Values beyond either supplied length are never
        read by a legal path and receive exactly zero gradient.
        """

        if not isinstance(emissions, torch.Tensor) or emissions.ndim != 3:
            shape = getattr(emissions, "shape", None)
            raise ValueError(f"emissions must have shape [B, U, T], got {shape}")
        if not torch.is_floating_point(emissions):
            raise ValueError("emissions must be a floating-point tensor")

        batch_size, phone_width, frame_width = emissions.shape
        text_lengths = self._lengths(
            text_lengths,
            name="text_lengths",
            batch_size=batch_size,
            maximum=phone_width,
            device=emissions.device,
        )
        speech_lengths = self._lengths(
            speech_lengths,
            name="speech_lengths",
            batch_size=batch_size,
            maximum=frame_width,
            device=emissions.device,
        )

        # The DP stays in float32 under mixed precision.  logsigmoid produces
        # non-positive local evidence, which prevents a very strong phone from
        # compensating additively for an absent one.
        local_evidence = F.logsigmoid(emissions.float())
        cumulative = F.pad(local_evidence.cumsum(dim=-1), (1, 0))
        prefix_scores = local_evidence.new_full(
            (batch_size, phone_width), self.invalid_score
        )
        if phone_width == 0 or frame_width == 0:
            return prefix_scores

        end_positions = torch.arange(
            frame_width + 1, device=emissions.device
        ).unsqueeze(0)
        state_shape = (
            batch_size,
            frame_width + 1,
            self.max_keyword_span_frames + 1,
        )

        # ``partition`` stores energy / temperature.  ``path_count`` evaluates
        # exactly the same topology with zero energy and therefore stores log N.
        partition: torch.Tensor | None = None
        path_count: torch.Tensor | None = None

        for phone_index in range(phone_width):
            active_phone = text_lengths.gt(phone_index)
            energy_candidates: list[torch.Tensor] = []
            count_candidates: list[torch.Tensor] = []

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
                segment_mean = segment_sum / float(duration)

                if phone_index == 0:
                    # First phone: every segment start is legal.  Padding on the
                    # endpoint and span axes places a duration-d segment ending
                    # at e into state [e, d].
                    valid_end = duration + torch.arange(
                        frame_width + 1 - duration, device=emissions.device
                    )
                    valid = (
                        valid_end.unsqueeze(0) <= speech_lengths.unsqueeze(1)
                    ) & active_phone.unsqueeze(1)
                    candidate = torch.where(
                        valid,
                        segment_mean / self.temperature,
                        torch.full_like(segment_mean, -torch.inf),
                    ).unsqueeze(-1)
                    candidate = F.pad(
                        candidate,
                        (
                            duration,
                            self.max_keyword_span_frames - duration,
                            duration,
                            0,
                        ),
                        value=-torch.inf,
                    )
                    energy_candidates.append(candidate)

                    zero = torch.where(
                        valid,
                        torch.zeros_like(segment_mean),
                        torch.full_like(segment_mean, -torch.inf),
                    ).unsqueeze(-1)
                    count_candidates.append(
                        F.pad(
                            zero,
                            (
                                duration,
                                self.max_keyword_span_frames - duration,
                                duration,
                                0,
                            ),
                            value=-torch.inf,
                        )
                    )
                    continue

                assert partition is not None and path_count is not None
                for gap in range(self.max_inter_phone_gap_frames + 1):
                    span_increment = duration + gap
                    if (
                        span_increment > frame_width
                        or span_increment > self.max_keyword_span_frames
                    ):
                        continue

                    previous_partition = partition[
                        :, : frame_width + 1 - span_increment,
                        : self.max_keyword_span_frames + 1 - span_increment,
                    ]
                    previous_count = path_count[
                        :, : frame_width + 1 - span_increment,
                        : self.max_keyword_span_frames + 1 - span_increment,
                    ]
                    # A predecessor ending at p reaches e=p+gap+duration.  The
                    # current segment itself is [e-duration, e); gap frames are
                    # deliberately unscored but strictly bounded.
                    current_segment = segment_mean[:, gap:].unsqueeze(-1)
                    if current_segment.size(1) != previous_partition.size(1):
                        raise RuntimeError("internal segmental DP shape mismatch")
                    valid_end = end_positions[:, span_increment:]
                    valid = (
                        valid_end <= speech_lengths.unsqueeze(1)
                    ) & active_phone.unsqueeze(1)
                    candidate = previous_partition + current_segment / self.temperature
                    candidate = torch.where(
                        valid.unsqueeze(-1),
                        candidate,
                        torch.full_like(candidate, -torch.inf),
                    )
                    energy_candidates.append(
                        F.pad(
                            candidate,
                            (span_increment, 0, span_increment, 0),
                            value=-torch.inf,
                        )
                    )

                    count_candidate = torch.where(
                        valid.unsqueeze(-1),
                        previous_count,
                        torch.full_like(previous_count, -torch.inf),
                    )
                    count_candidates.append(
                        F.pad(
                            count_candidate,
                            (span_increment, 0, span_increment, 0),
                            value=-torch.inf,
                        )
                    )

            partition = self._combine(
                energy_candidates, state_shape, local_evidence
            )
            path_count = self._combine(
                count_candidates, state_shape, local_evidence
            )

            log_partition = self._safe_logsumexp(partition.flatten(1), dim=1)
            log_number_of_paths = self._safe_logsumexp(
                path_count.flatten(1), dim=1
            )
            has_path = (
                active_phone
                & torch.isfinite(log_partition)
                & torch.isfinite(log_number_of_paths)
            )
            safe_log_partition = torch.where(
                has_path, log_partition, torch.zeros_like(log_partition)
            )
            safe_log_number_of_paths = torch.where(
                has_path,
                log_number_of_paths,
                torch.zeros_like(log_number_of_paths),
            )
            normalized = (
                self.temperature
                * (safe_log_partition - safe_log_number_of_paths)
                / float(phone_index + 1)
            ).clamp_max(0.0)
            prefix_scores[:, phone_index] = torch.where(
                has_path,
                normalized,
                torch.full_like(normalized, self.invalid_score),
            )

        return prefix_scores

    def extra_repr(self) -> str:
        return (
            f"min_phone_duration_frames={self.min_phone_duration_frames}, "
            f"max_phone_duration_frames={self.max_phone_duration_frames}, "
            f"max_inter_phone_gap_frames={self.max_inter_phone_gap_frames}, "
            f"max_keyword_span_frames={self.max_keyword_span_frames}, "
            f"temperature={self.temperature}, invalid_score={self.invalid_score}"
        )


__all__ = ["BoundedSegmentalAligner"]
