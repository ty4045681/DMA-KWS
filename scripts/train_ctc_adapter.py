#!/usr/bin/env python3
"""Step A: train the phoneme CTC adapter on a frozen encoder.

The trunk trained here is the tensor Stage II reads, so run this before
scripts/train_stage2_qbyt.py with stage2.phoneme_adapter.enabled=true.
"""

from __future__ import annotations

import hydra
from omegaconf import DictConfig

from dma_kws.hydra_app import CONFIG_DIR, resolved_config
from dma_kws.phoneme_adapter.runner import PhonemeAdapterTrainArgs, run_phoneme_adapter_training


@hydra.main(version_base=None, config_path=str(CONFIG_DIR), config_name="config")
def main(cfg: DictConfig) -> None:
    run = cfg.run
    run_phoneme_adapter_training(
        resolved_config(cfg),
        PhonemeAdapterTrainArgs(
            train_manifest=str(run.train_manifest),
            dev_manifest=str(run.dev_manifest),
            init_checkpoint=str(run.init_checkpoint),
            device=str(run.device),
            devices=int(run.devices),
            limit_steps=int(run.limit_steps),
            resume_from=str(run.resume_from),
        ),
    )


if __name__ == "__main__":
    main()
