"""sherpa-onnx Zipformer keyword spotting locator."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from dma_kws.audio import load_audio
from dma_kws.inference.audio_utils import apply_margin_to_span
from dma_kws.inference.locators.sherpa_keywords import format_sherpa_keyword
from dma_kws.stage1.candidates import KeywordCandidate


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
        self._modeling_unit = str(locator_cfg.get("modeling_unit", "cjkchar"))
        self._tokens_path = _optional_path(locator_cfg.get("tokens"))

        required = ("tokens", "encoder", "decoder", "joiner")
        missing = [name for name in required if not _optional_path(locator_cfg.get(name))]
        if missing:
            joined = ", ".join(f"locator.{name}" for name in missing)
            raise ValueError(f"Missing required sherpa locator settings: {joined}")

        spotter_kwargs: dict[str, Any] = {
            "tokens": str(locator_cfg["tokens"]),
            "encoder": str(locator_cfg["encoder"]),
            "decoder": str(locator_cfg["decoder"]),
            "joiner": str(locator_cfg["joiner"]),
            "num_threads": int(locator_cfg.get("num_threads", 2)),
            "provider": str(locator_cfg.get("provider", "cpu")),
        }
        keywords_file = _optional_path(locator_cfg.get("keywords_file"))
        if keywords_file is not None:
            spotter_kwargs["keywords_file"] = keywords_file
        if "keywords_threshold" in locator_cfg:
            spotter_kwargs["keywords_threshold"] = float(locator_cfg["keywords_threshold"])
        if "keywords_score" in locator_cfg:
            spotter_kwargs["keywords_score"] = float(locator_cfg["keywords_score"])

        self._kws = sherpa_onnx.KeywordSpotter(**spotter_kwargs)

    def locate(self, audio_path: str, keyword: str) -> list[KeywordCandidate]:
        waveform, sample_rate = load_audio(audio_path)
        samples = waveform.squeeze(0).cpu().numpy().astype(np.float32, copy=False)

        keyword_line = format_sherpa_keyword(
            keyword,
            modeling_unit=self._modeling_unit,
            tokens_path=self._tokens_path,
        )
        stream = self._kws.create_stream(keyword_line)
        stream.accept_waveform(sample_rate, samples)

        tail = np.zeros(int(self._tail_padding_sec * sample_rate), dtype=np.float32)
        stream.accept_waveform(sample_rate, tail)
        stream.input_finished()

        candidates: list[KeywordCandidate] = []
        while self._kws.is_ready(stream):
            self._kws.decode_stream(stream)
            result = self._kws.get_result(stream)
            if not result:
                continue

            timestamps = list(self._kws.timestamps(stream))
            if timestamps:
                start_sec = float(min(timestamps))
                end_sec = float(max(timestamps))
            else:
                start_sec = 0.0
                end_sec = float(len(samples) / sample_rate)

            start_sec, end_sec = apply_margin_to_span(
                start_sec,
                end_sec,
                margin_sec=self._margin_sec,
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

        return candidates
