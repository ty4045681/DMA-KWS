#!/usr/bin/env python3
"""Run the two-stage DMA-KWS demo with trained Stage I and Stage II checkpoints."""

from __future__ import annotations

import json

import hydra
from omegaconf import DictConfig, OmegaConf

from dma_kws.config import require_sections
from dma_kws.hydra_app import CONFIG_DIR, resolved_config
from dma_kws.inference.pipeline import TwoStageKWSPipeline
from dma_kws.training.device import resolve_accelerator


def _locator_type(config: dict) -> str:
    locator_cfg = config.get("locator")
    if isinstance(locator_cfg, dict):
        return str(locator_cfg.get("type", "phoneme_ctc"))
    return "phoneme_ctc"


def run(cfg: DictConfig) -> None:
    try:
        import torch
    except ImportError as exc:
        raise SystemExit(
            "Missing torch/torchaudio. Install CUDA PyTorch on the remote training machine first."
        ) from exc

    config = resolved_config(cfg)
    require_sections(config, ["paths", "stage1", "stage2", "demo", "tokenizer"])
    prep = OmegaConf.to_container(cfg.prep, resolve=True)
    if not isinstance(prep, dict):
        prep = {}
    run_cfg = cfg.run

    stage1_ckpt = str(prep.get("stage1_ckpt", ""))
    stage2_ckpt = str(prep.get("stage2_ckpt", ""))
    audio_path = str(prep.get("audio", ""))
    keyword = str(prep.get("keyword", ""))
    if not all([stage2_ckpt, audio_path, keyword]):
        raise SystemExit("prep.stage2_ckpt, prep.audio, and prep.keyword are required")
    if _locator_type(config) == "phoneme_ctc" and not stage1_ckpt:
        raise SystemExit("prep.stage1_ckpt is required for the phoneme_ctc locator")

    accelerator, _ = resolve_accelerator(str(run_cfg.device))
    device = torch.device(accelerator if accelerator == "cpu" else "cuda")

    pipeline = TwoStageKWSPipeline.from_config(config, prep, device)
    result = pipeline.run(audio_path, keyword)
    print(json.dumps(result, ensure_ascii=False, indent=2))


@hydra.main(version_base=None, config_path=str(CONFIG_DIR), config_name="config")
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
