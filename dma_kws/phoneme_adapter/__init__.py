"""Phoneme CTC adapter shared by Stage I phoneme search and Stage II QbyT.

The adapter is a trainable trunk on top of a frozen encoder plus a CTC output
projection. Both the CTC loss and QbyT read the *same* trunk output, which is
what makes the phoneme supervision reach Stage II instead of only serving Stage
I candidate generation.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "PhonemeAdapter",
    "build_phoneme_adapter",
    "ctc_min_input_lengths",
    "build_trunk",
    "TRUNK_TYPES",
]

_EXPORTS = {
    "PhonemeAdapter": "dma_kws.phoneme_adapter.module",
    "build_phoneme_adapter": "dma_kws.phoneme_adapter.module",
    "ctc_min_input_lengths": "dma_kws.phoneme_adapter.module",
    "build_trunk": "dma_kws.phoneme_adapter.trunk",
    "TRUNK_TYPES": "dma_kws.phoneme_adapter.trunk",
}


def __getattr__(name: str) -> Any:
    # Lazy so importing the package does not pull in torch.
    module_path = _EXPORTS.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(module_path), name)
