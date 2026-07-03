"""Optuna hyperparameter search for Stage II LoRA adaptation."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Callable

import yaml

from dma_kws.stage2.adapt import Stage2AdaptArgs, run_stage2_adaptation
from dma_kws.stage2.adapt_paths import adapt_exp_root, slugify


def suggest_adapt_params(trial: Any, *, search_mix: bool = False) -> dict[str, Any]:
    rank = trial.suggest_categorical("rank", [4, 8, 16, 32])
    alpha_ratio = trial.suggest_categorical("alpha_ratio", [1.0, 2.0])
    params = {
        "rank": rank,
        "alpha": int(alpha_ratio * rank),
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 2e-3, log=True),
        "max_steps": trial.suggest_categorical("max_steps", [1000, 2000, 3000]),
    }
    if search_mix:
        params["mix_ratio"] = trial.suggest_float("mix_ratio", 0.3, 0.7)
    return params


def compute_sweep_score(
    *,
    target_auc: float,
    lph_auc_adapted: float,
    lph_auc_base: float,
    lambda_forget: float = 1.0,
) -> float:
    forget_penalty = max(0.0, lph_auc_base - lph_auc_adapted)
    return float(target_auc) - float(lambda_forget) * forget_penalty


def apply_trial_params(config: dict[str, Any], params: dict[str, Any]) -> None:
    adapt = config.setdefault("adapt", {})
    for key, value in params.items():
        if key.startswith("_"):
            continue
        adapt[key] = value


def run_adaptation_trial(
    config: dict[str, Any],
    *,
    params: dict[str, Any],
    base_args: Stage2AdaptArgs,
    single_phase: bool = False,
    eval_lph_fn: Callable[[str], float] | None = None,
    eval_target_fn: Callable[[str, str], float] | None = None,
) -> dict[str, Any]:
    """Run one TTS→real adaptation trial and return metrics for Optuna."""
    trial_config = copy.deepcopy(config)
    apply_trial_params(trial_config, params)
    adapt = trial_config["adapt"]
    keyword = str(adapt["keyword"])
    trial_number = params.get("_trial_number", "manual")
    trial_root = adapt_exp_root(trial_config, keyword) / "sweep" / f"trial_{trial_number}"
    trial_root.mkdir(parents=True, exist_ok=True)
    adapt["exp_root"] = str(trial_root)

    adapt["phase"] = "tts"
    tts_artifacts = run_stage2_adaptation(
        trial_config,
        Stage2AdaptArgs(
            init_checkpoint=base_args.init_checkpoint,
            device=base_args.device,
            devices=base_args.devices,
            limit_steps=base_args.limit_steps or int(adapt.get("max_steps", 3000)),
        ),
    )

    real_artifacts = None
    if not single_phase:
        adapt["phase"] = "real"
        real_artifacts = run_stage2_adaptation(
            trial_config,
            Stage2AdaptArgs(
                init_checkpoint=base_args.init_checkpoint,
                device=base_args.device,
                devices=base_args.devices,
                limit_steps=base_args.limit_steps or int(adapt.get("max_steps", 3000)),
            ),
        )

    merged_ckpt = str(trial_root / "stage2_adapted.pt")
    if eval_target_fn is not None:
        target_auc = eval_target_fn(trial_config, merged_ckpt, keyword)
    else:
        target_auc = 0.0

    lph_auc = eval_lph_fn(merged_ckpt) if eval_lph_fn is not None else 0.0

    return {
        "target_auc": target_auc,
        "lph_auc": lph_auc,
        "tts": tts_artifacts,
        "real": real_artifacts,
        "merged_checkpoint": merged_ckpt,
    }


def save_best_params(path: Path, params: dict[str, Any], *, score: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"score": score, **params}
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)
