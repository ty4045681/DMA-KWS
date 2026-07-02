"""WeKws + WeNet ASR streaming CTC keyword locator."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from typing import Any, Mapping

from dma_kws.inference.audio_utils import apply_margin_to_span
from dma_kws.stage1.candidates import KeywordCandidate

DEFAULT_WEKWS_ROOT = "/Users/e4/Documents/myfork/wekws"
SAMPLE_RATE = 16000


def _ensure_wekws_on_path() -> Path:
    root = Path(os.environ.get("WEKWS_ROOT", DEFAULT_WEKWS_ROOT)).expanduser()
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return root


def _import_we_net_keyword_spotter():
    _ensure_wekws_on_path()
    try:
        from wekws.bin.stream_kws_wenet import WeNetKeywordSpotter

        return WeNetKeywordSpotter
    except ImportError:
        root = _ensure_wekws_on_path()
        module_path = root / "wekws" / "bin" / "stream_kws_wenet.py"
        if not module_path.is_file():
            raise ImportError(
                f"Could not import WeNetKeywordSpotter; "
                f"set WEKWS_ROOT or install wekws (missing {module_path})"
            ) from None
        spec = importlib.util.spec_from_file_location("stream_kws_wenet", module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Failed to load wekws module from {module_path}") from None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module.WeNetKeywordSpotter


def _locator_section(config: Mapping[str, Any]) -> Mapping[str, Any]:
    locator_cfg = config.get("locator")
    if locator_cfg is None:
        return {}
    if not isinstance(locator_cfg, Mapping):
        raise ValueError("Config section 'locator' must be a mapping")
    return locator_cfg


def _wekws_settings(locator_cfg: Mapping[str, Any]) -> dict[str, Any]:
    nested = locator_cfg.get("wekws")
    if isinstance(nested, Mapping):
        settings = dict(nested)
    else:
        settings = {}
    for key in (
        "config",
        "checkpoint",
        "symbol_table",
        "cmvn",
        "bpe_model",
        "threshold",
        "min_frames",
        "max_frames",
        "interval_frames",
        "chunk_seconds",
        "decoding_chunk_size",
        "num_decoding_left_chunks",
        "score_beam_size",
        "path_beam_size",
        "gpu",
        "non_lang_syms",
    ):
        if key in locator_cfg and key not in settings:
            settings[key] = locator_cfg[key]
    return settings


def _optional_path(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _wav_to_pcm_bytes(wav_path: str, *, sample_rate: int = SAMPLE_RATE) -> bytes:
    try:
        import librosa
    except ImportError as exc:
        raise ImportError(
            "librosa is required for WeKwsWenetLocator. "
            "Install with: pip install 'dma-kws[wekws]'"
        ) from exc
    waveform, _ = librosa.load(wav_path, sr=sample_rate, mono=True)
    return (waveform * (1 << 15)).astype("int16").tobytes()


def _resolve_gpu(wekws_cfg: Mapping[str, Any], device: Any) -> int:
    if "gpu" in wekws_cfg:
        return int(wekws_cfg["gpu"])
    device_text = str(device)
    if device_text.startswith("cuda"):
        if ":" in device_text:
            return int(device_text.rsplit(":", 1)[-1])
        return 0
    return -1


class WeKwsWenetLocator:
    """Streaming WeNet ASR + wekws CTC keyword decoder locator."""

    def __init__(
        self,
        *,
        config: Mapping[str, Any],
        prep: Mapping[str, Any] | None = None,
        device: Any = None,
    ) -> None:
        del prep
        locator_cfg = _locator_section(config)
        wekws_cfg = _wekws_settings(locator_cfg)
        demo = config.get("demo", {})
        if not isinstance(demo, Mapping):
            demo = {}
        self._margin_sec = float(demo.get("stage1_candidate_margin_sec", 0.15))
        self._chunk_seconds = float(wekws_cfg.get("chunk_seconds", 0.3))
        self._spotter_cls = _import_we_net_keyword_spotter()
        self._wekws_cfg = wekws_cfg
        self._device = device

    def _build_spotter(self, keyword: str):
        cfg = self._wekws_cfg
        required = ("config", "checkpoint", "symbol_table")
        missing = [name for name in required if not _optional_path(cfg.get(name))]
        if missing:
            joined = ", ".join(f"locator.wekws.{name}" for name in missing)
            raise ValueError(f"Missing required wekws locator settings: {joined}")

        kwargs: dict[str, Any] = {
            "config_path": str(cfg["config"]),
            "checkpoint_path": str(cfg["checkpoint"]),
            "symbol_table": str(cfg["symbol_table"]),
            "keywords": keyword,
            "threshold": float(cfg.get("threshold", 0.0)),
            "min_frames": int(cfg.get("min_frames", 5)),
            "max_frames": int(cfg.get("max_frames", 250)),
            "interval_frames": int(cfg.get("interval_frames", 50)),
            "score_beam": int(cfg.get("score_beam_size", 3)),
            "path_beam": int(cfg.get("path_beam_size", 20)),
            "gpu": _resolve_gpu(cfg, self._device),
            "cmvn": _optional_path(cfg.get("cmvn")),
            "bpe_model": _optional_path(cfg.get("bpe_model")),
            "non_lang_syms": _optional_path(cfg.get("non_lang_syms")),
            "decoding_chunk_size": int(cfg.get("decoding_chunk_size", 16)),
            "num_decoding_left_chunks": int(cfg.get("num_decoding_left_chunks", -1)),
        }
        return self._spotter_cls(**kwargs)

    def locate(self, audio_path: str, keyword: str) -> list[KeywordCandidate]:
        spotter = self._build_spotter(keyword)
        pcm = _wav_to_pcm_bytes(audio_path)
        interval = int(self._chunk_seconds * SAMPLE_RATE) * 2
        candidates: list[KeywordCandidate] = []

        for offset in range(0, len(pcm), interval):
            chunk = pcm[offset : min(offset + interval, len(pcm))]
            result = spotter.forward(chunk)
            if result.get("state") != 1:
                continue
            start_sec = float(result["start"])
            end_sec = float(result["end"])
            start_sec, end_sec = apply_margin_to_span(
                start_sec,
                end_sec,
                margin_sec=self._margin_sec,
            )
            candidates.append(
                KeywordCandidate(
                    start_sec=start_sec,
                    end_sec=end_sec,
                    stage1_score=float(result.get("score", 0.0)),
                    phonemes=[],
                )
            )

        return candidates
