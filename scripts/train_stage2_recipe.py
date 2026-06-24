#!/usr/bin/env python3
"""Multi-stage Stage II training recipes aligned with main qbyt/train*.py scripts."""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dma_kws.config import load_config
from dma_kws.stage2.train import Stage2TrainArgs, run_stage2_training

RECIPE_CONFIG_PATHS: dict[str, Path] = {
    "init-ls-460": PROJECT_ROOT / "configs" / "paper_ls460.yaml",
    "ft-ls-gs-1460": PROJECT_ROOT / "configs" / "paper_ls_gs1460.yaml",
    "frozen-wenet-encoder": PROJECT_ROOT / "configs" / "paper_ls460.yaml",
}

RECIPE_EXTRA_OVERRIDES: dict[str, dict[str, Any]] = {
    "frozen-wenet-encoder": {
        "stage2": {
            "freeze_encoder": True,
        },
    },
}


def _default_stage1_init_checkpoint(config: dict[str, Any]) -> str:
    """Resolve self-trained Stage I avg checkpoint from config paths."""
    paths = config.get("paths", {})
    stage1 = config.get("stage1", {})
    exp_root = paths.get("exp_root", "/data/dma-kws/exp")
    checkpoint_dir = stage1.get(
        "checkpoint_dir",
        f"{exp_root}/stage1_phoneme_ctc/checkpoints",
    )
    avg_cfg = stage1.get("checkpoint_avg", {}) or {}
    output_name = str(avg_cfg.get("output_name", "avg_10.ckpt"))
    return str(Path(checkpoint_dir) / output_name)


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def apply_recipe(config: dict[str, Any], recipe: str) -> dict[str, Any]:
    """Overlay recipe defaults from paper configs onto a base config."""
    if recipe not in RECIPE_CONFIG_PATHS:
        known = ", ".join(sorted(RECIPE_CONFIG_PATHS))
        raise SystemExit(f"Unknown recipe {recipe!r}. Choose from: {known}")

    reference = load_config(RECIPE_CONFIG_PATHS[recipe])
    merged = _deep_merge(config, {"training": reference.get("training", {}), "stage2": reference.get("stage2", {})})
    merged["training"]["recipe"] = recipe

    extra = RECIPE_EXTRA_OVERRIDES.get(recipe)
    if extra:
        merged = _deep_merge(merged, extra)

    if recipe == "frozen-wenet-encoder":
        exp_root = merged.get("paths", {}).get("exp_root", "/data/dma-kws/exp")
        merged["stage2"]["run_name"] = recipe
        merged["stage2"]["checkpoint_dir"] = f"{exp_root}/stage2_qbyt/checkpoints/{recipe}"
        merged["stage2"]["log_dir"] = f"{exp_root}/stage2_qbyt/logs/{recipe}"
        if not merged["stage2"].get("init_checkpoint"):
            merged["stage2"]["init_checkpoint"] = _default_stage1_init_checkpoint(merged)

    return merged


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=(
            "Recipe chain (paper reproduction): "
            "1) init-ls-460 — train from scratch, checkpoints every 1000 steps; "
            "2) average last 10 Lightning checkpoints (scripts/average_checkpoints.py); "
            "3) ft-ls-gs-1460 — resume from averaged .ckpt via stage2.resume_checkpoint. "
            "frozen-wenet-encoder freezes your self-trained Stage I encoder (avg_10.ckpt) and trains QbyT only."
        ),
    )
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument(
        "--recipe",
        required=True,
        choices=sorted(RECIPE_CONFIG_PATHS),
        help="Training recipe (overrides training.recipe and stage2 defaults)",
    )
    parser.add_argument(
        "--init-checkpoint",
        default="",
        help="Override stage2.init_checkpoint for partial encoder/QbyT weight init (init recipes)",
    )
    parser.add_argument(
        "--resume-checkpoint",
        default="",
        help=(
            "Override stage2.resume_checkpoint for full Lightning resume (finetune recipes). "
            "Falls back to stage2.resume_checkpoint or stage2.init_checkpoint from config."
        ),
    )
    parser.add_argument("--device", default="cuda", help="cuda or cpu")
    parser.add_argument("--devices", type=int, default=1, help="Number of visible GPUs to use")
    parser.add_argument("--limit-steps", type=int, default=0, help="Optional training step cap for smoke runs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = apply_recipe(load_config(args.config), args.recipe)
    stage2 = config.get("stage2", {})

    resume_checkpoint = args.resume_checkpoint or stage2.get("resume_checkpoint", "")
    if args.recipe.startswith("ft-") and not resume_checkpoint:
        resume_checkpoint = stage2.get("init_checkpoint", "")

    run_stage2_training(
        config,
        Stage2TrainArgs(
            init_checkpoint=args.init_checkpoint,
            resume_checkpoint=resume_checkpoint,
            device=args.device,
            devices=args.devices,
            limit_steps=args.limit_steps,
        ),
    )


if __name__ == "__main__":
    main()
