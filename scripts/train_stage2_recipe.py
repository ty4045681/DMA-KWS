#!/usr/bin/env python3
"""Multi-stage Stage II training recipes via Hydra experiment overlays."""

from __future__ import annotations

from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from dma_kws.hydra_app import CONFIG_DIR, resolved_config
from dma_kws.stage2.train import Stage2TrainArgs, run_stage2_training

_RECIPE_EXPERIMENTS = {
    "init-ls-460": "paper_ls460",
    "ft-ls-gs-1460": "paper_ls_gs1460",
    "frozen-wenet-encoder": "frozen_wenet_encoder",
}


def _default_stage1_init_checkpoint(config: dict) -> str:
    paths = config.get("paths", {})
    stage1 = config.get("stage1", {})
    exp_root = paths.get("exp_root", "/data/dma-kws/exp")
    checkpoint_dir = stage1.get("checkpoint_dir", f"{exp_root}/stage1_phoneme_ctc/checkpoints")
    avg_cfg = stage1.get("checkpoint_avg", {}) or {}
    output_name = str(avg_cfg.get("output_name", "avg_10.ckpt"))
    return str(Path(checkpoint_dir) / output_name)


@hydra.main(version_base=None, config_path=str(CONFIG_DIR), config_name="config")
def main(cfg: DictConfig) -> None:
    recipe = str(cfg.training.recipe)
    if recipe not in _RECIPE_EXPERIMENTS:
        known = ", ".join(sorted(_RECIPE_EXPERIMENTS))
        raise SystemExit(f"Unknown recipe {recipe!r}. Set training.recipe to one of: {known}")

    # Re-compose with the recipe experiment overlay if not already selected.
    if not OmegaConf.select(cfg, "experiment"):
        from dma_kws.config import compose_config, config_to_dict

        config = config_to_dict(compose_config(_RECIPE_EXPERIMENTS[recipe]))
    else:
        config = resolved_config(cfg)

    config["training"]["recipe"] = recipe
    stage2 = config.get("stage2", {})
    if recipe == "frozen-wenet-encoder" and not stage2.get("init_checkpoint"):
        stage2["init_checkpoint"] = _default_stage1_init_checkpoint(config)

    run = cfg.run
    resume_checkpoint = str(run.resume_checkpoint) or stage2.get("resume_checkpoint", "")
    if recipe.startswith("ft-") and not resume_checkpoint:
        resume_checkpoint = stage2.get("init_checkpoint", "")

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
