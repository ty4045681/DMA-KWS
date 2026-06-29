"""Stage I Wenet-aligned Conformer+CTC (re-exports for import stability)."""

from dma_kws.nn import build_encoder
from dma_kws.stage1.dataset import (
    Stage1Dataset,
    encode_manifest_target,
    phonemes_to_g2p_string,
    stage1_collate_fn,
)
from dma_kws.stage1.module import BLANK_ID, Stage1LightningModule, _load_ctc
from dma_kws.stage1.runner import Stage1TrainArgs, export_stage1_encoder_pt, run_stage1_training

__all__ = [
    "BLANK_ID",
    "Stage1Dataset",
    "Stage1LightningModule",
    "Stage1TrainArgs",
    "_load_ctc",
    "build_encoder",
    "encode_manifest_target",
    "export_stage1_encoder_pt",
    "phonemes_to_g2p_string",
    "run_stage1_training",
    "stage1_collate_fn",
]
