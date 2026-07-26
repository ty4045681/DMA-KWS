"""icefall PyTorch KWS locator via subprocess decode script."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

from dma_kws.config import resolve_stream_policy
from dma_kws.inference.audio_utils import apply_margin_to_span
from dma_kws.stage1.candidates import KeywordCandidate

DEFAULT_ICEFALL_ROOT = ""

#: Streaming flags owned by ``stage1.stream``; users must not set them by hand in
#: ``locator.decode_args`` or Stage I and Stage II could drift apart.
_MANAGED_DECODE_FLAGS = ("--causal", "--chunk-size", "--left-context-frames")


def _stream_decode_args(config: Mapping[str, Any]) -> list[str]:
    """Render ``stage1.stream`` as icefall decode flags.

    icefall's ``decode.py`` asserts single values for ``--chunk-size`` and
    ``--left-context-frames``, which the resolved policy already guarantees.
    """
    policy = resolve_stream_policy(config)
    if not policy.enabled:
        return ["--causal", "0"]
    return [
        "--causal",
        "1",
        "--chunk-size",
        str(policy.chunk_size),
        "--left-context-frames",
        str(policy.left_context_frames),
    ]


def _locator_section(config: Mapping[str, Any]) -> Mapping[str, Any]:
    locator_cfg = config.get("locator")
    if locator_cfg is None:
        return {}
    if not isinstance(locator_cfg, Mapping):
        raise ValueError("Config section 'locator' must be a mapping")
    return locator_cfg


def _optional_path(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _span_from_record(record: Mapping[str, Any]) -> tuple[float, float] | None:
    timestamps = record.get("timestamps")
    if isinstance(timestamps, list) and timestamps:
        return float(min(timestamps)), float(max(timestamps))

    start = record.get("start_time", record.get("start"))
    end = record.get("end_time", record.get("end"))
    if start is None or end is None:
        return None
    return float(start), float(end)


def _parse_decode_line(line: str) -> dict[str, Any] | None:
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    try:
        payload = json.loads(line)
    except json.JSONDecodeError:
        return None
    if isinstance(payload, dict):
        return payload
    return None


class IcefallPtKwsLocator:
    """Run an icefall KWS decode script and map JSON hits to candidates."""

    def __init__(
        self,
        *,
        config: Mapping[str, Any],
        prep: Mapping[str, Any] | None = None,
        device: Any = None,
    ) -> None:
        del prep, device
        locator_cfg = _locator_section(config)
        demo = config.get("demo", {})
        if not isinstance(demo, Mapping):
            demo = {}
        self._margin_sec = float(demo.get("stage1_candidate_margin_sec", 0.15))

        root = _optional_path(locator_cfg.get("root")) or os.environ.get(
            "ICEFALL_ROOT", DEFAULT_ICEFALL_ROOT
        )
        decode_script = _optional_path(locator_cfg.get("decode_script"))
        checkpoint = _optional_path(locator_cfg.get("checkpoint"))
        missing = []
        if not root:
            missing.append("locator.root or ICEFALL_ROOT")
        if not decode_script:
            missing.append("locator.decode_script")
        if not checkpoint:
            missing.append("locator.checkpoint")
        if missing:
            raise ValueError(f"Missing required icefall locator settings: {', '.join(missing)}")

        self._root = Path(root).expanduser()
        self._decode_script = Path(decode_script).expanduser()
        self._checkpoint = str(checkpoint)
        self._extra_args = [str(arg) for arg in locator_cfg.get("decode_args", [])]
        # Catch both "--chunk-size 16" and "--chunk-size=16".
        conflicting = sorted(
            {
                flag
                for arg in self._extra_args
                for flag in _MANAGED_DECODE_FLAGS
                if arg == flag or arg.startswith(f"{flag}=")
            }
        )
        if conflicting:
            raise ValueError(
                f"locator.decode_args must not set {', '.join(conflicting)}; these are derived "
                "from stage1.stream so Stage I and Stage II share one operating point. "
                "Set stage1.stream.chunk_size / stage1.stream.left_context_frames instead."
            )
        self._stream_args = _stream_decode_args(config)

    def _build_command(self, audio_path: str, keyword: str) -> list[str]:
        script = self._decode_script
        if not script.is_absolute():
            script = self._root / script
        return [
            sys.executable,
            str(script),
            "--checkpoint",
            self._checkpoint,
            "--wav",
            audio_path,
            "--keywords",
            keyword,
            *self._stream_args,
            *self._extra_args,
        ]

    def locate(self, audio_path: str, keyword: str) -> list[KeywordCandidate]:
        command = self._build_command(audio_path, keyword)
        proc = subprocess.run(
            command,
            cwd=str(self._root),
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            stderr = proc.stderr.strip()
            raise RuntimeError(
                f"icefall decode failed (exit {proc.returncode}): {stderr or command}"
            )

        candidates: list[KeywordCandidate] = []
        for line in proc.stdout.splitlines():
            record = _parse_decode_line(line)
            if record is None:
                continue
            span = _span_from_record(record)
            if span is None:
                continue
            start_sec, end_sec = apply_margin_to_span(
                span[0],
                span[1],
                margin_sec=self._margin_sec,
            )
            score = record.get("score", record.get("confidence", 1.0))
            candidates.append(
                KeywordCandidate(
                    start_sec=start_sec,
                    end_sec=end_sec,
                    stage1_score=float(score),
                    phonemes=[],
                )
            )
        return candidates
