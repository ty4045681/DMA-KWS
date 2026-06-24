"""Stage II QbyT utilities."""

from dma_kws.stage2.collate import test_collate_fn, train_collate_fn
from dma_kws.stage2.dataset import LibriPhraseEvalDataset, LibriPhraseTrainDataset
from dma_kws.stage2.losses import compute_stage2_losses

__all__ = [
    "LibriPhraseEvalDataset",
    "LibriPhraseTrainDataset",
    "Stage2LightningModule",
    "compute_stage2_losses",
    "test_collate_fn",
    "train_collate_fn",
]


def __getattr__(name: str):
    if name == "Stage2LightningModule":
        from dma_kws.stage2.module import Stage2LightningModule

        return Stage2LightningModule
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
