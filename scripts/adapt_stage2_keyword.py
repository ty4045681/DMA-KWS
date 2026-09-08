#!/usr/bin/env python3
"""Adapt a keyword with LoRA, full QbyT, or full encoder + QbyT training."""

from __future__ import annotations

import hydra
from omegaconf import DictConfig, OmegaConf

from dma_kws.hydra_app import CONFIG_DIR, resolved_config
from dma_kws.stage2.adapt import Stage2AdaptArgs, run_stage2_adaptation
from dma_kws.stage2.adapt_config import resolve_adapt_method


@hydra.main(version_base=None, config_path=str(CONFIG_DIR), config_name="config")
def main(cfg: DictConfig) -> None:
    config = resolved_config(cfg)
    adapt = OmegaConf.to_container(cfg.adapt, resolve=True)
    if not isinstance(adapt, dict):
        raise SystemExit("adapt config section must be a mapping")
    resolve_adapt_method(adapt)
    prep = OmegaConf.to_container(cfg.prep, resolve=True)
    if not isinstance(prep, dict):
        prep = {}
    run = cfg.run

    params_file = str(adapt.get("params_file", ""))
    if not params_file:
        from dma_kws.stage2.adapt_paths import adapt_exp_root

        best_params = adapt_exp_root(config, str(adapt.get("keyword", ""))) / "sweep" / "best_params.yaml"
        if best_params.is_file():
            params_file = str(best_params)

    run_stage2_adaptation(
        config,
        Stage2AdaptArgs(
            init_checkpoint=str(run.init_checkpoint or prep.get("stage2_ckpt", "")),
            resume_checkpoint=str(run.resume_checkpoint),
            resume_from=str(run.resume_from),
            device=str(run.device),
            devices=int(run.devices),
            limit_steps=int(run.limit_steps),
            params_file=params_file,
        ),
    )


if __name__ == "__main__":
    main()
