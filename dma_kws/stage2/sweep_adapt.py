"""Optuna hyperparameter search for Stage II LoRA adaptation."""

from __future__ import annotations

import copy
import random
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import yaml

from dma_kws.stage2.adapt_paths import adapt_exp_root, slugify
from dma_kws.training.adapt_params import merge_adapt_params, normalize_adapt_params

if TYPE_CHECKING:
    from dma_kws.stage2.adapt import Stage2AdaptArgs


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


def select_eval_subset_indices(
    dataset_size: int, subset: int, *, seed: int = 2025
) -> list[int] | None:
    """Pick a reproducible random subset of eval indices, or ``None`` for the full set.

    Taking ``range(subset)`` instead would only ever cover the head of the
    LibriPhrase eval frame, which is ordered 1-word → 2-word → 3-word → 4-word,
    so the forgetting metric would be measured on 1-word pairs alone.
    """
    if subset <= 0 or subset >= dataset_size:
        return None
    return sorted(random.Random(seed).sample(range(dataset_size), subset))


def compute_sweep_score(
    *,
    target_auc: float,
    lph_auc_adapted: float,
    lph_auc_base: float,
    lambda_forget: float = 1.0,
) -> float:
    forget_penalty = max(0.0, lph_auc_base - lph_auc_adapted)
    return float(target_auc) - float(lambda_forget) * forget_penalty


def apply_trial_params(config: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
    """Write normalized trial params into ``config['adapt']`` and return them."""
    return merge_adapt_params(config.setdefault("adapt", {}), params)


def run_adaptation_trial(
    config: dict[str, Any],
    *,
    params: dict[str, Any],
    base_args: Stage2AdaptArgs,
    single_phase: bool = False,
    eval_lph_fn: Callable[[str], float] | None = None,
    eval_target_fn: Callable[[str, str], float] | None = None,
    on_event: Callable[[str, str], None] | None = None,
) -> dict[str, Any]:
    """Run one TTS→real adaptation trial and return metrics for Optuna.

    ``on_event(stage, detail)`` reports progress (``train``/``eval`` stages) so
    callers can render console output without this module knowing about rich.
    """
    # Imported lazily so scoring/param helpers stay importable without torch.
    from dma_kws.stage2.adapt import Stage2AdaptArgs, run_stage2_adaptation

    trial_config = copy.deepcopy(config)
    effective_params = apply_trial_params(trial_config, params)
    adapt = trial_config["adapt"]
    keyword = str(adapt["keyword"])
    trial_number = params.get("_trial_number", "manual")
    trial_root = adapt_exp_root(trial_config, keyword) / "sweep" / f"trial_{trial_number}"
    trial_root.mkdir(parents=True, exist_ok=True)
    adapt["exp_root"] = str(trial_root)

    def report(stage: str, detail: str) -> None:
        if on_event is not None:
            on_event(stage, detail)

    def train_phase(phase: str) -> dict[str, Path]:
        adapt["phase"] = phase
        report("train", phase)
        return run_stage2_adaptation(
            trial_config,
            Stage2AdaptArgs(
                init_checkpoint=base_args.init_checkpoint,
                device=base_args.device,
                devices=base_args.devices,
                limit_steps=base_args.limit_steps or int(adapt.get("max_steps", 3000)),
            ),
        )

    tts_artifacts = train_phase("tts")
    real_artifacts = None if single_phase else train_phase("real")

    merged_ckpt = str(trial_root / "stage2_adapted.pt")
    if eval_target_fn is not None:
        report("eval", "target")
        target_auc = eval_target_fn(trial_config, merged_ckpt, keyword)
    else:
        target_auc = 0.0

    if eval_lph_fn is not None:
        report("eval", "libriphrase")
        lph_auc = eval_lph_fn(merged_ckpt)
    else:
        lph_auc = 0.0

    return {
        "target_auc": target_auc,
        "lph_auc": lph_auc,
        "params": effective_params,
        "tts": tts_artifacts,
        "real": real_artifacts,
        "merged_checkpoint": merged_ckpt,
    }


def save_best_params(path: Path, params: dict[str, Any], *, score: float) -> None:
    """Persist sweep best params using ``adapt`` config keys (not Optuna search keys)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"score": score, **normalize_adapt_params(params)}
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)
