"""Optuna hyperparameter search for Stage II LoRA adaptation."""

from __future__ import annotations

import copy
import json
import os
import random
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import yaml

from dma_kws.pathing import PROJECT_ROOT
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


def disable_persistent_workers(config: dict[str, Any]) -> dict[str, Any]:
    """Turn off ``stage2.dataloader.persistent_workers`` for a sweep trial config.

    Trials run back-to-back inside a single process: persistent workers from the
    previous trial stay alive until its loaders are collected, so their file
    descriptors pile up across trials until the process hits ``ulimit -n``. Sweeps
    trade the (small) per-epoch worker startup cost for that stability.
    """
    stage2_cfg = config.setdefault("stage2", {})
    dataloader_cfg = dict(stage2_cfg.get("dataloader") or {})
    dataloader_cfg["persistent_workers"] = False
    stage2_cfg["dataloader"] = dataloader_cfg
    return config


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
    if base_args.devices > 1 and int(os.environ.get("WORLD_SIZE", "1")) == 1:
        training = launch_distributed_adaptation_trial(
            config,
            params=params,
            base_args=base_args,
            single_phase=single_phase,
            on_event=on_event,
        )
    else:
        training = run_adaptation_training_trial(
            config,
            params=params,
            base_args=base_args,
            single_phase=single_phase,
            on_event=on_event,
        )

    trial_config = training["config"]
    keyword = training["keyword"]
    merged_ckpt = str(training["merged_checkpoint"])

    def report(stage: str, detail: str) -> None:
        if on_event is not None:
            on_event(stage, detail)

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
        "params": training["params"],
        "tts": training["tts"],
        "real": training["real"],
        "merged_checkpoint": merged_ckpt,
    }


def _prepare_adaptation_trial(
    config: dict[str, Any],
    params: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], str, Path]:
    trial_config = disable_persistent_workers(copy.deepcopy(config))
    effective_params = apply_trial_params(trial_config, params)
    adapt = trial_config["adapt"]
    keyword = str(adapt["keyword"])
    trial_number = params.get("_trial_number", "manual")
    trial_root = (
        adapt_exp_root(trial_config, keyword) / "sweep" / f"trial_{trial_number}"
    ).resolve()
    trial_root.mkdir(parents=True, exist_ok=True)
    adapt["exp_root"] = str(trial_root)
    return trial_config, effective_params, keyword, trial_root


def _validate_sweep_runtime_args(base_args: Stage2AdaptArgs) -> None:
    unsupported = [
        name
        for name, value in (
            ("resume_from", base_args.resume_from),
            ("params_file", base_args.params_file),
        )
        if value
    ]
    if unsupported:
        raise ValueError(
            "LoRA sweep trials do not support "
            f"{', '.join(unsupported)}. Optuna resumes completed trials through its "
            "study storage, and trial parameters are already fixed in the request."
        )


def run_adaptation_training_trial(
    config: dict[str, Any],
    *,
    params: dict[str, Any],
    base_args: Stage2AdaptArgs,
    single_phase: bool = False,
    on_event: Callable[[str, str], None] | None = None,
) -> dict[str, Any]:
    """Train one fixed trial; unlike the controller, this is safe on every DDP rank."""
    # Imported lazily so scoring/param helpers stay importable without torch.
    from dma_kws.stage2.adapt import Stage2AdaptArgs, run_stage2_adaptation

    _validate_sweep_runtime_args(base_args)
    trial_config, effective_params, keyword, trial_root = _prepare_adaptation_trial(
        config,
        params,
    )
    adapt = trial_config["adapt"]

    def report(stage: str, detail: str) -> None:
        if on_event is not None:
            on_event(stage, detail)

    def train_phase(
        phase: str,
        *,
        adapter_checkpoint: str = "",
    ) -> dict[str, Path]:
        adapt["phase"] = phase
        report("train", phase)
        return run_stage2_adaptation(
            trial_config,
            Stage2AdaptArgs(
                init_checkpoint=base_args.init_checkpoint,
                resume_checkpoint=adapter_checkpoint,
                device=base_args.device,
                devices=base_args.devices,
                limit_steps=base_args.limit_steps or int(adapt.get("max_steps", 3000)),
            ),
        )

    tts_artifacts = train_phase(
        "tts",
        adapter_checkpoint=base_args.resume_checkpoint,
    )
    real_artifacts = (
        None
        if single_phase
        else train_phase("real", adapter_checkpoint=str(tts_artifacts["adapter"]))
    )

    return {
        "config": trial_config,
        "keyword": keyword,
        "params": effective_params,
        "tts": tts_artifacts,
        "real": real_artifacts,
        "merged_checkpoint": trial_root / "stage2_adapted.pt",
    }


def serialize_training_result(result: dict[str, Any]) -> dict[str, Any]:
    def stringify_paths(artifacts: dict[str, Path] | None) -> dict[str, str] | None:
        if artifacts is None:
            return None
        return {name: str(path) for name, path in artifacts.items()}

    return {
        "config": result["config"],
        "keyword": result["keyword"],
        "params": result["params"],
        "tts": stringify_paths(result["tts"]),
        "real": stringify_paths(result["real"]),
        "merged_checkpoint": str(result["merged_checkpoint"]),
    }


def _deserialize_training_result(payload: dict[str, Any]) -> dict[str, Any]:
    def restore_paths(artifacts: dict[str, str] | None) -> dict[str, Path] | None:
        if artifacts is None:
            return None
        return {name: Path(path) for name, path in artifacts.items()}

    return {
        "config": payload["config"],
        "keyword": str(payload["keyword"]),
        "params": dict(payload["params"]),
        "tts": restore_paths(payload["tts"]),
        "real": restore_paths(payload.get("real")),
        "merged_checkpoint": Path(payload["merged_checkpoint"]),
    }


def launch_distributed_adaptation_trial(
    config: dict[str, Any],
    *,
    params: dict[str, Any],
    base_args: Stage2AdaptArgs,
    single_phase: bool,
    on_event: Callable[[str, str], None] | None = None,
) -> dict[str, Any]:
    """Launch fixed-parameter training under torchrun, outside the Optuna controller."""
    _validate_sweep_runtime_args(base_args)
    request_config = copy.deepcopy(config)
    request_adapt = request_config.setdefault("adapt", {})
    request_keyword = str(request_adapt["keyword"])
    request_adapt["exp_root"] = str(
        adapt_exp_root(request_config, request_keyword).resolve()
    )
    _, _, _, trial_root = _prepare_adaptation_trial(request_config, params)
    request_path = trial_root / "distributed_trial_request.json"
    result_path = trial_root / "distributed_trial_result.json"
    request = {
        "config": request_config,
        "params": params,
        "base_args": {
            "init_checkpoint": (
                str(Path(base_args.init_checkpoint).resolve())
                if base_args.init_checkpoint
                else ""
            ),
            "resume_checkpoint": (
                str(Path(base_args.resume_checkpoint).resolve())
                if base_args.resume_checkpoint
                else ""
            ),
            "device": base_args.device,
            "devices": base_args.devices,
            "limit_steps": base_args.limit_steps,
        },
        "single_phase": single_phase,
        "result_path": str(result_path),
    }
    request_path.write_text(json.dumps(request, indent=2), encoding="utf-8")
    if result_path.exists():
        result_path.unlink()

    worker_script = PROJECT_ROOT / "scripts" / "run_adapt_lora_trial.py"
    if on_event is not None:
        on_event("train", f"distributed worker x{base_args.devices}")
    sys.stdout.flush()
    sys.stderr.flush()
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    existing_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        str(PROJECT_ROOT)
        if not existing_pythonpath
        else os.pathsep.join((str(PROJECT_ROOT), existing_pythonpath))
    )
    for launch_attempt in range(3):
        master_port = _find_free_local_port()
        command = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--master_addr=127.0.0.1",
            f"--master_port={master_port}",
            f"--nproc_per_node={base_args.devices}",
            str(worker_script),
            "--request",
            str(request_path),
        ]
        started = time.monotonic()
        try:
            subprocess.run(
                command,
                check=True,
                cwd=str(PROJECT_ROOT),
                env=environment,
            )
            break
        except subprocess.CalledProcessError:
            # The free-port probe and torchrun bind cannot be atomic. Retry only
            # the narrow signature of that race: an immediate failure while the
            # selected port is now held by another process. Never replay a trial
            # that reached actual training.
            port_was_stolen = (
                time.monotonic() - started < 10.0
                and not _local_port_is_available(master_port)
            )
            if launch_attempt == 2 or not port_was_stolen:
                raise
    if not result_path.is_file():
        raise RuntimeError(
            f"Distributed LoRA trial finished without a rank-zero result: {result_path}"
        )
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    return _deserialize_training_result(payload)


def _find_free_local_port() -> int:
    """Pick a loopback port for one single-node torchrun invocation."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _local_port_is_available(port: int) -> bool:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
            listener.bind(("127.0.0.1", port))
    except OSError:
        return False
    return True


def save_best_params(path: Path, params: dict[str, Any], *, score: float) -> None:
    """Persist sweep best params using ``adapt`` config keys (not Optuna search keys)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"score": score, **normalize_adapt_params(params)}
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)
