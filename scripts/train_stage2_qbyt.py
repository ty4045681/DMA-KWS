#!/usr/bin/env python3
"""Train Stage II QbyT verifier with LibriPhrase parquet + dual loss."""

from __future__ import annotations

import hydra
from omegaconf import DictConfig

from dma_kws.hydra_app import CONFIG_DIR, resolved_config
from dma_kws.stage2.train import Stage2TrainArgs, run_stage2_training


@hydra.main(version_base=None, config_path=str(CONFIG_DIR), config_name="config")
def main(cfg: DictConfig) -> None:
    config = resolved_config(cfg)
    run = cfg.run
    stage2 = config.get("stage2", {})
    resume_checkpoint = str(run.resume_checkpoint) or stage2.get("resume_checkpoint", "")
    run_stage2_training(
        config,
        Stage2TrainArgs(
            init_checkpoint=str(run.init_checkpoint),
            resume_checkpoint=resume_checkpoint,
            resume_from=str(run.resume_from),
            device=str(run.device),
            devices=int(run.devices),
            limit_steps=int(run.limit_steps),
        ),
    )


if __name__ == "__main__":
    main()
