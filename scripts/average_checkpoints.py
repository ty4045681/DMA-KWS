#!/usr/bin/env python3
"""Average the last K checkpoints from a training run."""

from __future__ import annotations

from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from dma_kws.hydra_app import CONFIG_DIR
from dma_kws.training.checkpoint_avg import average_lightning_checkpoints


@hydra.main(version_base=None, config_path=str(CONFIG_DIR), config_name="config")
def main(cfg: DictConfig) -> None:
    prep = OmegaConf.to_container(cfg.prep, resolve=True)
    if not isinstance(prep, dict):
        prep = {}

    input_dir = Path(str(prep.get("input_dir", "")))
    if not input_dir:
        raise SystemExit("prep.input_dir is required")
    if not input_dir.is_dir():
        raise SystemExit(f"Input directory not found: {input_dir}")

    last_k = int(prep.get("last_k", 10))
    if last_k <= 0:
        raise SystemExit("prep.last_k must be positive")

    output = Path(str(prep.get("output", "")))
    if not output:
        raise SystemExit("prep.output is required")

    pattern = str(prep.get("pattern", "*.ckpt"))
    candidates = sorted(input_dir.glob(pattern))
    if not candidates:
        raise SystemExit(f"No checkpoints matched pattern {pattern!r} in {input_dir}")

    selected = candidates[-last_k:]
    output_path = average_lightning_checkpoints(selected, output)
    print(f"Averaged {len(selected)} checkpoints -> {output_path}")


if __name__ == "__main__":
    main()
