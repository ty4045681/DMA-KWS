"""Resume-safe audio-duration clock for Icefall encoder fine-tuning."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping


@dataclass
class EncoderFinetuneSchedule:
    """Advance Icefall schedules by consumed unpadded audio across all ranks.

    Upstream estimates duration from its maximum batch duration. Adaptation uses
    fixed sample counts, so actual input frames define an unambiguous clock that
    is independent of gradient accumulation and does not accumulate float error.
    The next training forward uses ``batch_count``; validation never advances it.
    """

    start_batch_count: float
    reference_duration: float
    frame_shift_seconds: float
    consumed_frames: int = 0

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "EncoderFinetuneSchedule":
        raw = (config.get("adapt") or {}).get("encoder_schedule", {})
        if raw is None:
            raw = {}
        if not isinstance(raw, Mapping):
            raise ValueError("adapt.encoder_schedule must be a mapping")
        values = {
            "start_batch_count": float(raw.get("start_batch_count", 100000.0)),
            "reference_duration": float(raw.get("reference_duration", 600.0)),
            "frame_shift_seconds": float((config.get("fbank") or {}).get("frame_shift", 10)) / 1000.0,
        }
        for name, value in values.items():
            if not math.isfinite(value) or (value < 0 if name == "start_batch_count" else value <= 0):
                raise ValueError(f"Encoder schedule {name} must be finite and {'non-negative' if name == 'start_batch_count' else 'positive'}")
        return cls(**values)

    @property
    def config(self) -> dict[str, float]:
        return {
            "start_batch_count": self.start_batch_count,
            "reference_duration": self.reference_duration,
            "frame_shift_seconds": self.frame_shift_seconds,
        }

    @property
    def batch_count(self) -> float:
        return self.start_batch_count + self.consumed_frames * self.frame_shift_seconds / self.reference_duration

    def advance(self, global_frames: int) -> None:
        if type(global_frames) is not int or global_frames < 0:
            raise ValueError("Encoder schedule frame count must be a non-negative integer")
        self.consumed_frames += global_frames

    def state_dict(self) -> dict[str, Any]:
        return {"version": 1, "config": self.config, "consumed_frames": self.consumed_frames}

    def load_state_dict(self, state: Any) -> None:
        if not isinstance(state, Mapping) or state.get("version") != 1 or state.get("config") != self.config:
            raise ValueError("Encoder schedule configuration changed or checkpoint schedule is missing; use run.init_checkpoint for a new weights-only run")
        frames = state.get("consumed_frames")
        if type(frames) is not int or frames < 0:
            raise ValueError("Checkpoint encoder schedule consumed_frames must be a non-negative integer")
        self.consumed_frames = frames
