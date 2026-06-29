#!/usr/bin/env python3
"""Train Stage I phoneme CTC with Wenet-aligned CharTokenizer targets."""

from __future__ import annotations

import hydra
from omegaconf import DictConfig

from dma_kws.hydra_app import CONFIG_DIR, resolved_config
from dma_kws.stage1.wenet_ctc import Stage1TrainArgs, run_stage1_training


@hydra.main(version_base=None, config_path=str(CONFIG_DIR), config_name="config")
def main(cfg: DictConfig) -> None:
    run = cfg.run
    run_stage1_training(
        resolved_config(cfg),
        Stage1TrainArgs(
            train_manifest=str(run.train_manifest),
            dev_manifest=str(run.dev_manifest),
            device=str(run.device),
            devices=int(run.devices),
            limit_steps=int(run.limit_steps),
            resume_from=str(run.resume_from),
        ),
    )


if __name__ == "__main__":
    main()
