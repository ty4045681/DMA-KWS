"""sherpa-onnx Zipformer keyword spotting locator."""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from dma_kws.audio import load_audio
from dma_kws.inference.audio_utils import apply_margin_to_span
from dma_kws.stage1.candidates import KeywordCandidate

DEFAULT_WAKEUP_WINDOW_SEC = 1.5
DEFAULT_STREAM_CHUNK_SEC = 0.1


def span_from_sherpa_hit(
    *,
    audio_duration_sec: float,
    wakeup_window_sec: float,
    decoded_sec: float,
) -> tuple[float, float]:
    """Return a short crop ending at the absolute decoded-audio frontier.

    sherpa timestamps are decoder-local after an internal or explicit reset,
    while its Python API does not expose the corresponding absolute origin.
    The caller-maintained source-audio frontier is therefore authoritative.
    """
    window = float(wakeup_window_sec)
    if not math.isfinite(window) or window <= 0.0:
        raise ValueError("locator.wakeup_window_sec must be positive")
    duration = float(audio_duration_sec)
    frontier = float(decoded_sec)
    if not math.isfinite(duration) or duration < 0.0:
        raise ValueError("audio_duration_sec must be finite and non-negative")
    if not math.isfinite(frontier):
        raise ValueError("decoded_sec must be finite")

    end_sec = min(duration, max(0.0, frontier))
    start_sec = max(0.0, end_sec - window)
    return start_sec, end_sec


def _import_sherpa_onnx():
    try:
        import sherpa_onnx
    except ImportError as exc:
        raise ImportError(
            "sherpa-onnx is required for SherpaOnnxKwsLocator. "
            "Install with: pip install 'sherpa-onnx>=1.10'"
        ) from exc
    return sherpa_onnx


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


class SherpaOnnxKwsLocator:
    """Zipformer KWS locator backed by ``sherpa_onnx.KeywordSpotter``."""

    def __init__(
        self,
        *,
        config: Mapping[str, Any],
        prep: Mapping[str, Any] | None = None,
        device: Any = None,
    ) -> None:
        del prep, device
        sherpa_onnx = _import_sherpa_onnx()
        locator_cfg = _locator_section(config)
        demo = config.get("demo", {})
        if not isinstance(demo, Mapping):
            demo = {}

        self._margin_sec = float(demo.get("stage1_candidate_margin_sec", 0.15))
        self._tail_padding_sec = float(locator_cfg.get("tail_padding_sec", 0.66))
        self._stream_chunk_sec = float(
            locator_cfg.get("stream_chunk_sec", DEFAULT_STREAM_CHUNK_SEC)
        )
        self._wakeup_window_sec = float(
            locator_cfg.get("wakeup_window_sec", DEFAULT_WAKEUP_WINDOW_SEC)
        )
        if not math.isfinite(self._stream_chunk_sec) or self._stream_chunk_sec <= 0.0:
            raise ValueError("locator.stream_chunk_sec must be positive")
        if not math.isfinite(self._tail_padding_sec) or self._tail_padding_sec < 0.0:
            raise ValueError("locator.tail_padding_sec must be non-negative")
        if not math.isfinite(self._wakeup_window_sec) or self._wakeup_window_sec <= 0.0:
            raise ValueError("locator.wakeup_window_sec must be positive")
        stage1_cfg = config.get("stage1")
        if not isinstance(stage1_cfg, Mapping):
            stage1_cfg = {}
        self._sample_rate = int(stage1_cfg.get("sample_rate", 16000))

        required = ("tokens", "encoder", "decoder", "joiner", "keywords_file")
        missing = [name for name in required if not _optional_path(locator_cfg.get(name))]
        if missing:
            joined = ", ".join(f"locator.{name}" for name in missing)
            raise ValueError(f"Missing required sherpa locator settings: {joined}")

        keywords_file = Path(str(locator_cfg["keywords_file"])).expanduser()
        if not keywords_file.is_file():
            raise ValueError(f"locator.keywords_file not found: {keywords_file}")

        spotter_kwargs: dict[str, Any] = {
            "tokens": str(locator_cfg["tokens"]),
            "encoder": str(locator_cfg["encoder"]),
            "decoder": str(locator_cfg["decoder"]),
            "joiner": str(locator_cfg["joiner"]),
            "keywords_file": str(keywords_file),
            "num_threads": int(locator_cfg.get("num_threads", 2)),
            "provider": str(locator_cfg.get("provider", "cpu")),
        }
        if "keywords_threshold" in locator_cfg:
            spotter_kwargs["keywords_threshold"] = float(locator_cfg["keywords_threshold"])
        if "keywords_score" in locator_cfg:
            spotter_kwargs["keywords_score"] = float(locator_cfg["keywords_score"])

        self._keywords_file = str(keywords_file)
        self._kws = sherpa_onnx.KeywordSpotter(**spotter_kwargs)

    def locate(
        self,
        audio_path: str,
        keyword: str,
        keyword_phonemes: Sequence[str] | None = None,
    ) -> list[KeywordCandidate]:
        del keyword, keyword_phonemes
        waveform, sample_rate = load_audio(
            audio_path, sample_rate=self._sample_rate
        )
        samples = waveform.squeeze(0).cpu().numpy().astype(np.float32, copy=False)
        audio_duration_sec = float(len(samples) / sample_rate)

        # Official sherpa-onnx path: keywords come only from keywords.txt.
        stream = self._kws.create_stream()
        candidates: list[KeywordCandidate] = []

        def drain_ready(*, source_samples_fed: int) -> None:
            # ``source_samples_fed`` excludes synthetic tail padding and remains
            # absolute across sherpa's internal and explicit decoder resets.
            decoded_sec = source_samples_fed / sample_rate
            while self._kws.is_ready(stream):
                self._kws.decode_stream(stream)
                # Read exactly once. Calling ``timestamps()`` afterwards would
                # consume the same native result again in sherpa-onnx 1.13.x.
                result = self._kws.get_result(stream)
                keyword_result = (
                    result.strip()
                    if isinstance(result, str)
                    else str(getattr(result, "keyword", "")).strip()
                )
                if not keyword_result:
                    continue

                start_sec, end_sec = span_from_sherpa_hit(
                    audio_duration_sec=audio_duration_sec,
                    wakeup_window_sec=self._wakeup_window_sec,
                    decoded_sec=decoded_sec,
                )
                start_sec, end_sec = apply_margin_to_span(
                    start_sec,
                    end_sec,
                    margin_sec=self._margin_sec,
                    audio_duration_sec=audio_duration_sec,
                )
                candidates.append(
                    KeywordCandidate(
                        start_sec=start_sec,
                        end_sec=end_sec,
                        stage1_score=1.0,
                        phonemes=[],
                    )
                )
                self._kws.reset_stream(stream)

        chunk_samples = max(1, round(self._stream_chunk_sec * sample_rate))
        for begin in range(0, len(samples), chunk_samples):
            end = min(len(samples), begin + chunk_samples)
            stream.accept_waveform(sample_rate, samples[begin:end])
            drain_ready(source_samples_fed=end)

        tail = np.zeros(round(self._tail_padding_sec * sample_rate), dtype=np.float32)
        if tail.size:
            stream.accept_waveform(sample_rate, tail)
        stream.input_finished()
        drain_ready(source_samples_fed=len(samples))

        return candidates
