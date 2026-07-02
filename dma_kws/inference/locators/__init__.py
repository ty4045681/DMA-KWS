"""Keyword locator backends for DMA-KWS inference."""

__all__ = [
    "IcefallPtKwsLocator",
    "PhonemeCtcLocator",
    "SherpaOnnxKwsLocator",
    "WeKwsWenetLocator",
]


def __getattr__(name: str):
    if name == "PhonemeCtcLocator":
        from dma_kws.inference.locators.phoneme_ctc import PhonemeCtcLocator

        return PhonemeCtcLocator
    if name == "WeKwsWenetLocator":
        from dma_kws.inference.locators.wekws_wenet import WeKwsWenetLocator

        return WeKwsWenetLocator
    if name == "SherpaOnnxKwsLocator":
        from dma_kws.inference.locators.sherpa_kws import SherpaOnnxKwsLocator

        return SherpaOnnxKwsLocator
    if name == "IcefallPtKwsLocator":
        from dma_kws.inference.locators.icefall_pt import IcefallPtKwsLocator

        return IcefallPtKwsLocator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
