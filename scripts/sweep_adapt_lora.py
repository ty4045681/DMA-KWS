#!/usr/bin/env python3
"""Optuna hyperparameter sweep for Stage II LoRA keyword adaptation."""

from __future__ import annotations

import json
from pathlib import Path

import hydra
import yaml
from omegaconf import DictConfig, OmegaConf

from dma_kws.hydra_app import CONFIG_DIR, resolved_config
from dma_kws.stage2 import adapt_console
from dma_kws.stage2.adapt import Stage2AdaptArgs
from dma_kws.stage2.adapt_paths import adapt_exp_root, slugify
from dma_kws.stage2.sweep_adapt import (
    compute_sweep_score,
    run_adaptation_trial,
    save_best_params,
    suggest_adapt_params,
)


def _eval_lph_auc(config: dict, checkpoint: str, *, subset: int = 0) -> float:
    try:
        import pytorch_lightning as pl
        from torch.utils.data import DataLoader
    except ImportError as exc:
        raise SystemExit("Missing torch/pytorch-lightning for LPH eval.") from exc

    from dma_kws.config import get_tokenizer_config
    from dma_kws.pathing import resolve_dict_path
    from dma_kws.stage2.collate import test_collate_fn
    from dma_kws.stage2.dataset import LibriPhraseEvalDataset, resolve_stage2_eval_paths
    from dma_kws.stage2.module import Stage2LightningModule
    from dma_kws.tokenizer import load_char_tokenizer
    from dma_kws.training.checkpoint_io import extract_state_dict

    import torch

    tokenizer_cfg = get_tokenizer_config(config)
    dict_path = resolve_dict_path(config)
    tokenizer = load_char_tokenizer(dict_path, split_with_space=tokenizer_cfg.get("split_with_space", " "))
    vocab_size = len(tokenizer._symbol_table)

    eval_paths = resolve_stage2_eval_paths(config)
    split = str(config.get("stage2", {}).get("eval", {}).get("split", "hard"))
    dataset = LibriPhraseEvalDataset(
        test_dir=eval_paths["test_dir"],
        fbank_dir=eval_paths["fbank_dir"],
        split=split,
        csv_files=eval_paths["csv_files"],
        aggregate_csv=eval_paths["aggregate_csv"],
        tokenizer=tokenizer,
    )
    if subset > 0 and subset < len(dataset):
        dataset = torch.utils.data.Subset(dataset, list(range(subset)))

    dataloader = DataLoader(
        dataset,
        batch_size=eval_paths["batch_size"],
        shuffle=False,
        num_workers=0,
        collate_fn=test_collate_fn,
    )

    model = Stage2LightningModule(config, vocab_size=vocab_size)
    ckpt = torch.load(checkpoint, map_location="cpu")
    state = extract_state_dict(ckpt)
    model.load_state_dict(state, strict=False)
    trainer = pl.Trainer(accelerator="cpu", devices=1, logger=False, enable_checkpointing=False)
    results = trainer.test(model, dataloaders=dataloader, verbose=False)
    metrics = results[0] if results else {}
    return float(metrics.get("test/auc", 0.0))


def _eval_target_auc(config: dict, checkpoint: str, keyword: str) -> float:
    try:
        import pytorch_lightning as pl
        from torch.utils.data import DataLoader
    except ImportError as exc:
        raise SystemExit("Missing torch/pytorch-lightning for target eval.") from exc

    from dma_kws.config import get_tokenizer_config
    from dma_kws.pathing import resolve_dict_path
    from dma_kws.stage2.adapt_dataset import TargetKeywordValDataset
    from dma_kws.stage2.adapt_paths import adapt_data_root, phase_manifest
    from dma_kws.stage2.collate import test_collate_fn
    from dma_kws.stage2.module import Stage2LightningModule
    from dma_kws.tokenizer import load_char_tokenizer
    from dma_kws.training.checkpoint_io import extract_state_dict

    import torch

    tokenizer_cfg = get_tokenizer_config(config)
    dict_path = resolve_dict_path(config)
    tokenizer = load_char_tokenizer(dict_path, split_with_space=tokenizer_cfg.get("split_with_space", " "))
    vocab_size = len(tokenizer._symbol_table)
    adapt = config.get("adapt", {})
    phase = str(adapt.get("phase", "real"))

    data_root = adapt_data_root(config, keyword)
    eval_manifest = phase_manifest(data_root, phase, split="eval")
    dataset = TargetKeywordValDataset(
        manifest_path=eval_manifest,
        keyword=keyword,
        fbank_root=data_root / "fbank",
        tokenizer=tokenizer,
        manifest_root=data_root,
    )
    val_loader = DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=test_collate_fn)

    ckpt = torch.load(checkpoint, map_location="cpu")
    state = extract_state_dict(ckpt)

    class _TargetValModule(Stage2LightningModule):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.target_auc_metric = __import__("torchmetrics").AUROC(task="binary")

        def validation_step(self, batch, batch_idx):
            logits, _ = self(batch["feat"], batch["feat_lengths"], batch["anchor"])
            preds = torch.sigmoid(logits)
            labels = batch["label"].int()
            self.target_auc_metric.update(preds, labels)

        def on_validation_epoch_end(self):
            auc = self.target_auc_metric.compute()
            self.log("val/target_auc", auc, prog_bar=True)
            self.target_auc_metric.reset()

    model = _TargetValModule(config, vocab_size=vocab_size)
    model.load_state_dict(state, strict=False)

    trainer = pl.Trainer(accelerator="cpu", devices=1, logger=False, enable_checkpointing=False)
    results = trainer.validate(model, dataloaders=val_loader, verbose=False)
    metrics = results[0] if results else {}
    return float(metrics.get("val/target_auc", 0.0))


@hydra.main(version_base=None, config_path=str(CONFIG_DIR), config_name="config")
def main(cfg: DictConfig) -> None:
    try:
        import optuna
    except ImportError as exc:
        raise SystemExit("Install optuna: pip install 'dma-kws[adapt]'") from exc

    config = resolved_config(cfg)
    adapt = OmegaConf.to_container(cfg.adapt, resolve=True)
    if not isinstance(adapt, dict):
        raise SystemExit("adapt config section must be a mapping")
    sweep_cfg = adapt.get("sweep", {}) or {}
    run = cfg.run
    prep = OmegaConf.to_container(cfg.prep, resolve=True)
    if not isinstance(prep, dict):
        prep = {}

    keyword = str(adapt.get("keyword", "")).strip()
    if not keyword:
        raise SystemExit("adapt.keyword is required")
    slug = slugify(keyword)
    sweep_root = adapt_exp_root(config, keyword) / "sweep"
    sweep_root.mkdir(parents=True, exist_ok=True)

    storage = str(sweep_cfg.get("storage") or f"sqlite:///{sweep_root / 'optuna.db'}")
    study_name = str(sweep_cfg.get("study_name") or slug)
    n_trials = int(sweep_cfg.get("n_trials", 20))
    lambda_forget = float(sweep_cfg.get("lambda_forget", 1.0))
    lph_subset = int(sweep_cfg.get("lph_subset", 2000))
    search_mix = bool(sweep_cfg.get("search_mix", False))
    single_phase = bool(sweep_cfg.get("single_phase", False))

    base_ckpt = str(run.init_checkpoint or prep.get("stage2_ckpt", ""))
    if not base_ckpt:
        raise SystemExit("prep.stage2_ckpt or run.init_checkpoint is required for sweep baseline")

    reporter = adapt_console.adapt_reporter(config, prep)
    if reporter.use_rich:
        optuna.logging.set_verbosity(optuna.logging.WARNING)

    reporter.section(f"LoRA hyperparameter sweep · {keyword}")
    reporter.info("Measuring LibriPhrase baseline AUC for the un-adapted checkpoint...")
    lph_base = _eval_lph_auc(config, base_ckpt, subset=lph_subset)
    reporter.print_plan(
        adapt_console.sweep_baseline_rows(
            keyword=keyword,
            slug=slug,
            base_checkpoint=base_ckpt,
            lph_auc_base=lph_base,
            n_trials=n_trials,
            lambda_forget=lambda_forget,
            lph_subset=lph_subset,
            search_mix=search_mix,
            single_phase=single_phase,
            storage=storage,
            study_name=study_name,
        ),
        title="Sweep Plan",
    )
    print(json.dumps({"lph_auc_base": lph_base}))

    base_args = Stage2AdaptArgs(
        init_checkpoint=base_ckpt,
        device=str(run.device),
        devices=int(run.devices),
        limit_steps=int(run.limit_steps),
    )
    trial_log: list[dict] = []

    def objective(trial: optuna.Trial) -> float:
        params = suggest_adapt_params(trial, search_mix=search_mix)
        params["_trial_number"] = trial.number
        reporter.section(f"Trial {trial.number + 1}/{n_trials}")
        reporter.print_plan(adapt_console.trial_param_rows(params), title="Trial Parameters")
        metrics = run_adaptation_trial(
            config,
            params=params,
            base_args=base_args,
            single_phase=single_phase,
            eval_lph_fn=lambda ckpt: _eval_lph_auc(config, ckpt, subset=lph_subset),
            eval_target_fn=lambda cfg, ckpt, kw: _eval_target_auc(cfg, ckpt, kw),
            on_event=lambda stage, detail: reporter.info(f"trial {trial.number}: {stage} {detail}"),
        )
        score = compute_sweep_score(
            target_auc=metrics["target_auc"],
            lph_auc_adapted=metrics["lph_auc"],
            lph_auc_base=lph_base,
            lambda_forget=lambda_forget,
        )
        trial.set_user_attr("target_auc", metrics["target_auc"])
        trial.set_user_attr("lph_auc", metrics["lph_auc"])
        trial_log.append(
            {
                "number": trial.number,
                "score": score,
                "target_auc": metrics["target_auc"],
                "lph_auc": metrics["lph_auc"],
                "forget_penalty": max(0.0, lph_base - metrics["lph_auc"]),
                "params": metrics.get("params", {}),
            }
        )
        best_so_far = max(entry["score"] for entry in trial_log)
        reporter.info(
            f"trial {trial.number}: score={score:.4f} target_auc={metrics['target_auc']:.4f} "
            f"lph_auc={metrics['lph_auc']:.4f} (best={best_so_far:.4f})"
        )
        return score

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(),
        pruner=optuna.pruners.MedianPruner(),
    )
    study.optimize(objective, n_trials=n_trials)

    best = study.best_trial
    best_path = sweep_root / "best_params.yaml"
    save_best_params(best_path, best.params, score=float(best.value))
    persisted = yaml.safe_load(best_path.read_text(encoding="utf-8")) or {}
    best_params = {key: value for key, value in persisted.items() if key != "score"}

    reporter.section("Sweep results")
    if trial_log:
        reporter.print_table(*adapt_console.sweep_results_table(trial_log), title="Trials (best first)")
    reporter.print_plan(
        [("best_score", f"{float(best.value):.4f}")]
        + [(key, str(value)) for key, value in sorted(best_params.items())],
        title="Best Parameters",
    )
    reporter.done(f"Best params written to {best_path}")
    print(yaml.safe_dump({"best_score": best.value, "best_params": best_params, "best_path": str(best_path)}))


if __name__ == "__main__":
    main()
