"""Stage II QbyT utilities.

Attributes are resolved lazily so that torch-free submodules (path helpers,
console reporters, parameter resolution) can be imported without pulling the
training stack.
"""

_LAZY_ATTRS = {
    "LibriPhraseEvalDataset": "dma_kws.stage2.dataset",
    "LibriPhraseTrainDataset": "dma_kws.stage2.dataset",
    "Stage2LightningModule": "dma_kws.stage2.module",
    "compute_stage2_losses": "dma_kws.stage2.losses",
    "test_collate_fn": "dma_kws.stage2.collate",
    "train_collate_fn": "dma_kws.stage2.collate",
}

__all__ = sorted(_LAZY_ATTRS)


def __getattr__(name: str):
    module_path = _LAZY_ATTRS.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from importlib import import_module

    return getattr(import_module(module_path), name)


def __dir__() -> list[str]:
    return sorted({*globals(), *_LAZY_ATTRS})
