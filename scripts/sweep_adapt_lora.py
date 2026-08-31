#!/usr/bin/env python3
"""Optuna hyperparameter sweep for Stage II LoRA keyword adaptation."""

from __future__ import annotations

import gc
import json

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
    select_eval_subset_indices,
    suggest_adapt_params,
)


def _eval_lph_auc(
    config: dict,
    checkpoint: str,
    *,
    subset: int = 0,
    accelerator: str = "cpu",
) -> float:
    try:
        import pytorch_lightning as pl
        from torch.utils.data import DataLoader
    except ImportError as exc:
        raise SystemExit("Missing torch/pytorch-lightning for LPH eval.") from exc

    from dma_kws.config import get_tokenizer_config
    from dma_kws.pathing import resolve_dict_path
    from dma_kws.stage2.collate import test_collate_fn
    from dma_kws.stage2.dataset import LibriPhraseEvalDataset, resolve_stage2_eval_paths
    from dma_kws.stage2.module import Stage2LightningModule, assert_adapter_weights_loaded
    from dma_kws.stage2.readout import assert_qbyt_alignment_state_loaded
    from dma_kws.tokenizer import load_char_tokenizer
    from dma_kws.training.checkpoint_io import (
        assert_qbyt_readout_version,
        extract_state_dict,
    )

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
    indices = select_eval_subset_indices(len(dataset), subset)
    if indices is not None:
        dataset = torch.utils.data.Subset(dataset, indices)

    dataloader = DataLoader(
        dataset,
        batch_size=eval_paths["batch_size"],
        shuffle=False,
        num_workers=eval_paths["num_workers"],
        collate_fn=test_collate_fn,
    )

    model = Stage2LightningModule(config, vocab_size=vocab_size)
    ckpt = torch.load(checkpoint, map_location="cpu")
    assert_qbyt_readout_version(
        ckpt,
        source=checkpoint,
        expected_alignment=model.qbyt_alignment,
    )
    state = extract_state_dict(ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    assert_adapter_weights_loaded(model, missing)
    assert_qbyt_alignment_state_loaded(
        missing,
        unexpected,
        source=checkpoint,
        expected_topology=model.qbyt_alignment.topology,
    )
    trainer = pl.Trainer(
        accelerator=accelerator, devices=1, logger=False, enable_checkpointing=False
    )
    results = trainer.test(model, dataloaders=dataloader, verbose=False)
    metrics = results[0] if results else {}
    return float(metrics.get("test/auc", 0.0))


def _eval_target_auc(
    config: dict, checkpoint: str, keyword: str, *, accelerator: str = "cpu"
) -> float:
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
    from dma_kws.stage2.module import Stage2LightningModule, assert_adapter_weights_loaded
    from dma_kws.stage2.readout import (
        assert_qbyt_alignment_state_loaded,
        resolve_qbyt_alignment,
    )
    from dma_kws.stage2.sweep_eval import (
        resolve_target_eval_output_dir,
        write_target_eval_report,
    )
    from dma_kws.tokenizer import load_char_tokenizer
    from dma_kws.training.checkpoint_io import (
        assert_qbyt_readout_version,
        extract_state_dict,
    )

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
    val_loader = DataLoader(
        dataset,
        batch_size=int(adapt.get("val_batch_size", 64)),
        shuffle=False,
        collate_fn=test_collate_fn,
    )

    ckpt = torch.load(checkpoint, map_location="cpu")
    expected_alignment = resolve_qbyt_alignment(config.get("stage2", {}))
    assert_qbyt_readout_version(
        ckpt,
        source=checkpoint,
        expected_alignment=expected_alignment,
    )
    state = extract_state_dict(ckpt)

    class _TargetValModule(Stage2LightningModule):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.target_auc_metric = __import__("torchmetrics").AUROC(task="binary")
            self.target_eval_records: list[dict] = []

        def validation_step(self, batch, batch_idx):
            del batch_idx
            logits, _ = self(batch["feat"], batch["feat_lengths"], batch["anchor"])
            probabilities = torch.sigmoid(logits)
            self.target_auc_metric.update(probabilities, batch["label"].int())
            scores = probabilities.detach().cpu().reshape(-1).tolist()
            labels = batch["label"].detach().cpu().reshape(-1).tolist()
            sample_ids = batch["sample_id"].detach().cpu().reshape(-1).tolist()
            self.target_eval_records.extend(
                {
                    "sample_id": int(sample_id),
                    "audio_path": str(dataset.df.iloc[int(sample_id)]["audio_path"]),
                    "keyword": keyword,
                    "label": int(label),
                    "qbyt_score": float(score),
                }
                for sample_id, label, score in zip(sample_ids, labels, scores)
            )

        def on_validation_epoch_end(self):
            auc = self.target_auc_metric.compute()
            self.log("val/target_auc", auc, prog_bar=True)
            self.target_auc_metric.reset()

    model = _TargetValModule(config, vocab_size=vocab_size)
    missing, unexpected = model.load_state_dict(state, strict=False)
    assert_adapter_weights_loaded(model, missing)
    assert_qbyt_alignment_state_loaded(
        missing,
        unexpected,
        source=checkpoint,
        expected_topology=model.qbyt_alignment.topology,
    )

    trainer = pl.Trainer(
        accelerator=accelerator, devices=1, logger=False, enable_checkpointing=False
    )
    validation_results = trainer.validate(model, dataloaders=val_loader, verbose=False)
    validation_metrics = validation_results[0] if validation_results else {}
    records = sorted(model.target_eval_records, key=lambda record: record["sample_id"])
    if len(records) != len(dataset):
        raise RuntimeError(
            "Target sweep evaluation did not score every manifest row: "
            f"expected {len(dataset)}, got {len(records)}"
        )

    sweep_cfg = adapt.get("sweep", {}) or {}
    deployment_threshold = float(
        (config.get("demo", {}) or {}).get("qbyt_threshold", 0.5)
    )
    output_dir = resolve_target_eval_output_dir(checkpoint, ckpt)
    report = write_target_eval_report(
        records,
        output_dir=output_dir,
        manifest_path=eval_manifest,
        checkpoint_path=checkpoint,
        threshold=deployment_threshold,
        plot_curves=bool(sweep_cfg.get("plot_curves", True)),
        plot_dpi=int(sweep_cfg.get("plot_dpi", 160)),
        plot_min_recall=sweep_cfg.get("plot_min_recall"),
        plot_max_fpr=sweep_cfg.get("plot_max_fpr"),
    )
    print(
        json.dumps(
            {
                "target_eval_summary": report["summary"],
                "target_eval_plots": report["plots"],
            },
            ensure_ascii=False,
        )
    )
    return float(
        validation_metrics.get(
            "val/target_auc",
            report["metrics"].get("auc", 0.0),
        )
    )


def _release_cuda_cache(torch_module) -> None:
    """Release controller-side eval memory before a torchrun trial starts."""
    gc.collect()
    if torch_module.cuda.is_available():
        torch_module.cuda.empty_cache()


@hydra.main(version_base=None, config_path=str(CONFIG_DIR), config_name="config")
def main(cfg: DictConfig) -> None:
    try:
        import optuna
    except ImportError as exc:
        raise SystemExit("Install optuna: pip install 'dma-kws[adapt]'") from exc
    try:
        import torch
    except ImportError as exc:
        raise SystemExit(
            "Missing torch. Install CUDA PyTorch on the remote training machine first."
        ) from exc

    # Every trial spins up a fresh set of dataloader workers inside this one process.
    # With the default file_descriptor sharing strategy their fds accumulate until the
    # process hits `ulimit -n`, which surfaces as "received 0 items of ancdata" and then
    # as sqlite "unable to open database file" when Optuna tries to record the trial.
    torch.multiprocessing.set_sharing_strategy("file_system")

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

    # Scores are only comparable within one streaming operating point, so record it
    # alongside the study instead of leaving it implicit in the config.
    from dma_kws.config import resolve_stream_policy

    stream_policy = resolve_stream_policy(config)
    reporter.info(f"Encoder stream policy: {stream_policy.describe()}")

    # Scoring runs the model over ~thousands of eval pairs twice per trial; keep it
    # on the training accelerator instead of falling back to CPU.
    from dma_kws.training.device import resolve_accelerator_and_devices

    eval_accelerator, _ = resolve_accelerator_and_devices(str(run.device), 1)

    reporter.section(f"LoRA hyperparameter sweep · {keyword}")
    reporter.info("Measuring LibriPhrase baseline AUC for the un-adapted checkpoint...")
    lph_base = _eval_lph_auc(config, base_ckpt, subset=lph_subset, accelerator=eval_accelerator)
    _release_cuda_cache(torch)
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
    print(json.dumps({"lph_auc_base": lph_base, "stream": stream_policy.describe()}))

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
        completed = sum(
            existing.state == optuna.trial.TrialState.COMPLETE
            for existing in trial.study.trials
        )
        reporter.section(
            f"Completed trial target {completed + 1}/{n_trials} "
            f"(Optuna trial #{trial.number})"
        )
        reporter.print_plan(adapt_console.trial_param_rows(params), title="Trial Parameters")
        try:
            metrics = run_adaptation_trial(
                config,
                params=params,
                base_args=base_args,
                single_phase=single_phase,
                eval_lph_fn=lambda ckpt: _eval_lph_auc(
                    config, ckpt, subset=lph_subset, accelerator=eval_accelerator
                ),
                eval_target_fn=lambda cfg, ckpt, kw: _eval_target_auc(
                    cfg, ckpt, kw, accelerator=eval_accelerator
                ),
                on_event=lambda stage, detail: reporter.info(
                    f"trial {trial.number}: {stage} {detail}"
                ),
            )
        finally:
            _release_cuda_cache(torch)
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
    # ``Optuna.optimize(n_trials=N)`` means N *additional* trials, not N total.
    # Count only successful trials so rerunning after this DDP infrastructure
    # failure fills the configured target instead of adding another full sweep.
    completed_before = sum(
        trial.state == optuna.trial.TrialState.COMPLETE
        for trial in study.trials
    )
    remaining_trials = max(0, n_trials - completed_before)
    if remaining_trials:
        # Completed trials already live in durable storage. Stop on infrastructure
        # or data errors instead of repeating the same broken setup for every trial.
        study.optimize(objective, n_trials=remaining_trials)

    if not any(trial.state == optuna.trial.TrialState.COMPLETE for trial in study.trials):
        raise SystemExit(
            f"No trial completed for study {study_name!r}; see the first traceback above "
            "for the failure and re-run with the same study_name to resume."
        )
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
    print(
        yaml.safe_dump(
            {
                "best_score": best.value,
                "best_params": best_params,
                "best_path": str(best_path),
                "stream": stream_policy.describe(),
            }
        )
    )


if __name__ == "__main__":
    main()
