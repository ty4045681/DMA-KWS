"""DMA-KWS inference core: locators, Stage II verification, and pipeline."""

__all__ = [
    "KeywordLocator",
    "build_locator",
    "Stage2ClipRunner",
    "Stage2Verifier",
    "TwoStageKWSPipeline",
]


def __getattr__(name: str):
    if name in {"KeywordLocator", "build_locator"}:
        from dma_kws.inference import locator

        return getattr(locator, name)
    if name == "Stage2Verifier":
        from dma_kws.inference.stage2_verifier import Stage2Verifier

        return Stage2Verifier
    if name == "Stage2ClipRunner":
        from dma_kws.inference.stage2_clip import Stage2ClipRunner

        return Stage2ClipRunner
    if name == "TwoStageKWSPipeline":
        from dma_kws.inference.pipeline import TwoStageKWSPipeline

        return TwoStageKWSPipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
