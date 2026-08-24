"""Keyword locator protocol and factory."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

from dma_kws.stage1.candidates import KeywordCandidate


class KeywordLocator(Protocol):
    """Propose keyword time spans in audio."""

    def locate(
        self,
        audio_path: str,
        keyword: str,
        keyword_phonemes: Sequence[str] | None = None,
    ) -> list[KeywordCandidate]:
        """Return Stage I candidate regions for ``keyword`` in ``audio_path``.

        ``keyword_phonemes`` is an optional enrollment sequence. Locators that
        search phonemes (the in-repo CTC locator) must use it when supplied.
        External locators whose modeling unit is not ARPAbet ignore it.
        """


def build_locator(config: dict[str, Any], prep: dict[str, Any], device: Any):
    """Build a locator from Hydra ``locator`` config and ``prep`` overrides."""
    locator_cfg = dict(config.get("locator") or {})
    locator_type = str(locator_cfg.get("type", "phoneme_ctc"))

    if locator_type == "phoneme_ctc":
        from dma_kws.inference.locators.phoneme_ctc import PhonemeCtcLocator

        return PhonemeCtcLocator(config=config, prep=prep, device=device)
    if locator_type in {"sherpa_kws", "sherpa_zipformer_kws"}:
        from dma_kws.inference.locators.sherpa_kws import SherpaOnnxKwsLocator

        return SherpaOnnxKwsLocator(config=config, prep=prep, device=device)
    if locator_type == "wekws_wenet":
        from dma_kws.inference.locators.wekws_wenet import WeKwsWenetLocator

        return WeKwsWenetLocator(config=config, prep=prep, device=device)
    if locator_type in {"icefall_pt", "icefall_pt_kws"}:
        from dma_kws.inference.locators.icefall_pt import IcefallPtKwsLocator

        return IcefallPtKwsLocator(config=config, prep=prep, device=device)

    raise ValueError(f"Unsupported locator type: {locator_type}")
