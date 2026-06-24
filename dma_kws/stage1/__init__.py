"""Stage I phoneme CTC utilities."""

__all__ = [
    "Stage1LightningModule",
    "encode_manifest_target",
    "phonemes_to_g2p_string",
    "run_stage1_training",
]


def __getattr__(name: str):
    if name in {"Stage1LightningModule", "encode_manifest_target", "phonemes_to_g2p_string", "run_stage1_training"}:
        from dma_kws.stage1 import wenet_ctc

        return getattr(wenet_ctc, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
