#!/usr/bin/env python3
"""Run Stage II-only DMA-KWS inference on a pre-cropped keyword clip."""

from __future__ import annotations

import json

import hydra
from omegaconf import DictConfig, OmegaConf

from dma_kws.config import require_sections
from dma_kws.hydra_app import CONFIG_DIR, resolved_config
from dma_kws.inference.stage2_clip import Stage2ClipRunner
from dma_kws.training.device import resolve_accelerator


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

    stage2_ckpt = str(prep.get("stage2_ckpt", ""))
    audio_path = str(prep.get("audio", ""))
    keyword = str(prep.get("keyword", ""))
    if not all([stage2_ckpt, audio_path, keyword]):
        raise SystemExit("prep.stage2_ckpt, prep.audio, and prep.keyword are required")

    accelerator, _ = resolve_accelerator(str(run_cfg.device))
    device = torch.device(accelerator if accelerator == "cpu" else "cuda")
    runner = Stage2ClipRunner.from_config(config, prep, device)
    result = runner.run(audio_path, keyword)
    print(json.dumps(result, ensure_ascii=False, indent=2))


@hydra.main(version_base=None, config_path=str(CONFIG_DIR), config_name="config")
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
