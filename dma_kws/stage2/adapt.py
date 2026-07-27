"""Stage II LoRA continual adaptation training."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchmetrics

from dma_kws.pathing import resolve_dict_path
from dma_kws.stage2.adapt_dataset import (
    KeywordAdaptationDataset,
    MixedAdaptationDataset,
    TargetKeywordValDataset,
)
from dma_kws.stage2.adapt_paths import adapt_data_root, phase_manifest, slugify
from dma_kws.stage2.module import Stage2LightningModule
from dma_kws.stage2.train import _build_val_dataloader, _resolve_path
from dma_kws.training.adapt_params import (
    load_adapt_params_file,
    merge_adapt_params,
    resolve_adapt_lr,
)
from dma_kws.training.checkpoint_io import assert_stream_policy_matches
from dma_kws.training.lora import (
    count_lora_params,
    inject_qbyt_lora,
    load_lora_state_dict,
    lora_state_dict,
    merge_lora,
)


@dataclass
class Stage2AdaptArgs:
    """Runtime options for Stage II LoRA adaptation."""

    init_checkpoint: str = ""
    resume_checkpoint: str = ""
    resume_from: str = ""
    device: str = "cuda"
    devices: int = 1
    limit_steps: int = 0
    params_file: str = ""


def _adapt_section(config: dict[str, Any]) -> dict[str, Any]:
    adapt = config.get("adapt")
    if not isinstance(adapt, dict):
        raise ValueError("Config section 'adapt' must be a mapping")
    return adapt


def _resolve_adapt_paths(config: dict[str, Any]) -> dict[str, Path]:
    adapt = _adapt_section(config)
    paths = config["paths"]
    keyword = str(adapt.get("keyword", ""))
    if not keyword:
        raise ValueError("adapt.keyword is required")

    slug = str(adapt.get("slug", "")) or slugify(keyword)
    data_root = Path(adapt.get("data_root", "")) if adapt.get("data_root") else adapt_data_root(
        config, keyword
    )
    if adapt.get("exp_root"):
        exp_root = Path(str(adapt["exp_root"]))
    else:
        exp_root = Path(paths["exp_root"]) / "stage2_adapt" / slug
    phase = str(adapt.get("phase", "tts"))

    return {
        "keyword": Path(keyword),  # type: ignore[dict-item]
        "slug": Path(slug),  # type: ignore[dict-item]
        "keyword_str": keyword,
        "slug_str": slug,
        "data_root": data_root,
        "fbank_root": data_root / "fbank",
        "exp_root": exp_root,
        "phase": Path(phase),  # type: ignore[dict-item]
        "phase_str": phase,
        "phase_dir": exp_root / phase,
        "adapter_path": exp_root / f"adapter_{slug}.pt",
        "merged_path": exp_root / "stage2_adapted.pt",
        "train_manifest": phase_manifest(data_root, phase, split="train"),
        "eval_manifest": phase_manifest(data_root, phase, split="eval"),
    }


def _apply_adapt_overrides(config: dict[str, Any], args: Stage2AdaptArgs) -> dict[str, Any]:
    """Merge ``adapt.params_file`` overrides (e.g. sweep best params) into the config."""
    if not args.params_file:
        return {}
    return merge_adapt_params(
        _adapt_section(config),
        load_adapt_params_file(args.params_file),
    )


class Stage2LoraAdaptationModule(Stage2LightningModule):
    """Stage II module with frozen encoder/base QbyT and trainable LoRA adapters."""

    def __init__(
        self,
        config: dict[str, Any],
        *,
        vocab_size: int,
        init_checkpoint: str | Path | None = None,
        lora_rank: int = 16,
        lora_alpha: float = 32.0,
        lora_targets: tuple[str, ...] | None = None,
        adapter_checkpoint: str | Path | None = None,
    ) -> None:
        super().__init__(
            config,
            vocab_size=vocab_size,
            freeze_encoder=True,
            init_checkpoint=init_checkpoint,
        )
        for param in self.parameters():
            param.requires_grad = False
        # The phoneme adapter trunk is part of the shared Stage I/II forward
        # pass; letting LoRA move it would break the premise that both stages
        # read one encoder pass. Freeze it and turn off the auxiliary CTC loss.
        self.freeze_adapter = self.adapter is not None
        self.ctc_weight = 0.0

        self.lora_injected = inject_qbyt_lora(
            self.qbyt,
            rank=lora_rank,
            alpha=lora_alpha,
            targets=lora_targets,
        )
        if adapter_checkpoint:
            state = torch.load(adapter_checkpoint, map_location="cpu")
            # LoRA weights are tuned against a specific encoder operating point.
            assert_stream_policy_matches(state, self.stream_policy, source=adapter_checkpoint)
            adapter_state = state.get("lora_state_dict", state)
            load_lora_state_dict(self.qbyt, adapter_state, strict=False)

        self.lora_param_counts = count_lora_params(self)
        self._adapt_cfg = _adapt_section(config)
        self.target_auc_metric = torchmetrics.AUROC(task="binary")
        self.target_eer_metric = torchmetrics.classification.EER(task="binary")

    def configure_optimizers(self) -> dict:
        trainable = [param for param in self.parameters() if param.requires_grad]
        if not trainable:
            raise RuntimeError("No trainable LoRA parameters found")
        adapt = self._adapt_cfg
        lr, _ = resolve_adapt_lr(adapt)
        optim_cfg = {
            "optimizer": str(adapt.get("optimizer", "adam")).lower(),
            "lr": lr,
            "weight_decay": float(adapt.get("weight_decay", 0.0)),
            "warmup_steps": int(adapt.get("warmup_steps", 100)),
            "total_steps": int(adapt.get("max_steps", 3000)),
        }

        import torch as torch_mod
        from transformers import get_cosine_schedule_with_warmup

        optimizer_name = optim_cfg["optimizer"]
        if optimizer_name == "adamw":
            optimizer = torch_mod.optim.AdamW(
                trainable,
                lr=optim_cfg["lr"],
                weight_decay=optim_cfg["weight_decay"],
            )
        elif optimizer_name == "adam":
            optimizer = torch_mod.optim.Adam(trainable, lr=optim_cfg["lr"])
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name!r}")

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=optim_cfg["warmup_steps"],
            num_training_steps=optim_cfg["total_steps"],
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        total_loss, losses, logits = self._forward_train_losses(batch)
        self._log_train_losses(total_loss, losses)

        source = batch.get("source")
        if source is not None:
            keyword_mask = source.bool()
            per_sample = F.binary_cross_entropy_with_logits(
                logits, batch["label"].float(), reduction="none"
            )
            self.log("train/keyword_frac", keyword_mask.float().mean(), on_step=True)
            if keyword_mask.any():
                self.log("train/keyword_utt_loss", per_sample[keyword_mask].mean(), on_step=True)
            if (~keyword_mask).any():
                self.log("train/libri_utt_loss", per_sample[~keyword_mask].mean(), on_step=True)
        return total_loss

    def validation_step(
        self,
        batch: dict[str, torch.Tensor],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        logits, _ = self(batch["feat"], batch["feat_lengths"], batch["anchor"])
        preds = torch.sigmoid(logits)
        labels = batch["label"].int()

        if dataloader_idx == 0:
            utt_loss = F.binary_cross_entropy_with_logits(logits, labels.float())
            self.log("val/target_utt_loss", utt_loss, prog_bar=True, on_epoch=True, add_dataloader_idx=False)
            self.target_auc_metric.update(preds, labels)
            self.target_eer_metric.update(preds, labels)
        else:
            utt_loss = F.binary_cross_entropy_with_logits(logits, labels.float())
            self.log("val/utt_loss", utt_loss, prog_bar=True, on_epoch=True, add_dataloader_idx=False)
            self.auc_metric.update(preds, labels)
            self.eer_metric.update(preds, labels)

    def on_validation_epoch_end(self) -> None:
        target_auc = self.target_auc_metric.compute()
        target_eer = self.target_eer_metric.compute()
        self.log("val/target_auc", target_auc, prog_bar=True, sync_dist=True)
        self.log("val/target_eer", target_eer, prog_bar=True, sync_dist=True)

        lph_auc = self.auc_metric.compute()
        lph_eer = self.eer_metric.compute()
        self.log("val/auc", lph_auc, prog_bar=True, sync_dist=True)
        self.log("val/eer", lph_eer, prog_bar=True, sync_dist=True)
        # Checkpoint-filename alias of val/auc; kept out of CSV/TensorBoard.
        self.log("val_auc", lph_auc, sync_dist=True, logger=False)

        self.target_auc_metric.reset()
        self.target_eer_metric.reset()
        self.auc_metric.reset()
        self.eer_metric.reset()


def _resolve_init_checkpoint(config: dict[str, Any], adapt_paths: dict[str, Any], args: Stage2AdaptArgs) -> str:
    adapt = _adapt_section(config)
    prep = config.get("prep", {}) or {}

    init_checkpoint = (
        args.init_checkpoint
        or prep.get("stage2_ckpt", "")
        or adapt.get("init_checkpoint", "")
        or config.get("stage2", {}).get("init_checkpoint", "")
    )
    if not init_checkpoint:
        raise ValueError(
            "init_checkpoint is required for adaptation (set prep.stage2_ckpt or adapt.init_checkpoint)"
        )
    return str(init_checkpoint)


def _resolve_adapter_resume(adapt_paths: dict[str, Any], phase: str) -> str | None:
    if phase == "real":
        tts_adapter = adapt_paths["phase_dir"].parent / "tts" / f"adapter_{adapt_paths['slug_str']}.pt"
        if tts_adapter.exists():
            return str(tts_adapter)
    phase_adapter = adapt_paths["phase_dir"] / f"adapter_{adapt_paths['slug_str']}.pt"
    if phase_adapter.exists():
        return str(phase_adapter)
    return None


def run_stage2_adaptation(config: dict[str, Any], args: Stage2AdaptArgs) -> dict[str, Path]:
    """Run Stage II LoRA adaptation for the configured keyword phase."""
    try:
        import pytorch_lightning as pl_mod
        from torch.utils.data import DataLoader
    except ImportError as exc:
        raise SystemExit(
            "Missing torch/pytorch-lightning. Install CUDA PyTorch on the training machine first."
        ) from exc

    from dma_kws.config import get_tokenizer_config, require_sections
    from dma_kws.runlog import build_loggers
    from dma_kws.stage2 import adapt_console
    from dma_kws.stage2.collate import test_collate_fn, train_collate_fn
    from dma_kws.stage2.dataset import LibriPhraseTrainDataset
    from dma_kws.tokenizer import load_char_tokenizer
    from dma_kws.training import resolve_resume_path
    from dma_kws.training.callbacks import build_stage2_callbacks, print_run_summary
    from dma_kws.training.metrics_history import (
        append_wide_row,
        build_metrics_history_callback,
        build_run_record,
        collect_hparams,
        numeric_callback_metrics,
    )
    from dma_kws.training.ddp import apply_step_based_validation, build_trainer_kwargs
    from dma_kws.training.device import resolve_accelerator_and_devices

    _apply_adapt_overrides(config, args)
    require_sections(config, ["paths", "stage1", "stage2", "tokenizer", "training", "adapt"])

    adapt = _adapt_section(config)
    adapt_paths = _resolve_adapt_paths(config)
    paths = config["paths"]
    stage1 = config["stage1"]
    stage2 = config["stage2"]
    training = config["training"]
    tokenizer_cfg = get_tokenizer_config(config)
    reporter = adapt_console.adapt_reporter(config)

    train_manifest = adapt_paths["train_manifest"]
    eval_manifest = adapt_paths["eval_manifest"]
    if not train_manifest.exists():
        raise SystemExit(f"Adaptation train manifest not found: {train_manifest}")
    if not eval_manifest.exists():
        raise SystemExit(f"Adaptation eval manifest not found: {eval_manifest}")

    phase = adapt_paths["phase_str"]
    if args.limit_steps:
        # Keep the cosine schedule horizon in sync with the truncated run, otherwise
        # training stops while the LR is still on its way down.
        adapt["max_steps"] = int(args.limit_steps)
    init_checkpoint = _resolve_init_checkpoint(config, adapt_paths, args)
    adapter_resume = _resolve_adapter_resume(adapt_paths, phase)
    accelerator, devices = resolve_accelerator_and_devices(args.device, args.devices)

    adapt_paths["phase_dir"].mkdir(parents=True, exist_ok=True)
    checkpoint_dir = adapt_paths["phase_dir"] / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    resume_path = resolve_resume_path(args.resume_from, checkpoint_dir)

    reporter.section(f"LoRA adaptation · {adapt_paths['keyword_str']} · phase={phase}")
    reporter.print_plan(
        adapt_console.adapt_plan_rows(
            adapt_paths=adapt_paths,
            accelerator=accelerator,
            devices=devices,
            init_checkpoint=init_checkpoint,
            adapter_resume=adapter_resume,
            resume_path=resume_path,
            params_file=args.params_file,
        ),
        title="Adaptation Plan",
    )

    dict_path = resolve_dict_path(config)
    tokenizer = load_char_tokenizer(dict_path, split_with_space=tokenizer_cfg.get("split_with_space", " "))
    vocab_size = len(tokenizer._symbol_table)

    seed = int(training.get("seed", 2025))
    pl_mod.seed_everything(seed, workers=True)

    keyword_dataset = KeywordAdaptationDataset(
        manifest_path=train_manifest,
        keyword=adapt_paths["keyword_str"],
        fbank_root=adapt_paths["fbank_root"],
        tokenizer=tokenizer,
        manifest_root=adapt_paths["data_root"],
    )

    processed_root = Path(paths["processed_root"])
    feature_root = Path(paths.get("feature_root", processed_root))
    parquet_file = _resolve_path(
        stage2,
        "parquet_file",
        processed_root / "stage2_qbyt" / "aggregated_segments_with_g2p_distance.parquet",
    )
    wav_dir = _resolve_path(stage2, "wav_dir", feature_root / "fbank")

    libri_dataset = LibriPhraseTrainDataset(
        parquet_file=parquet_file,
        wav_dir=wav_dir,
        tokenizer=tokenizer,
        negative_ratio=int(stage2.get("negative_ratio", 1)),
        hard_negative_ratio=int(stage2.get("hard_negative_ratio", 1)),
        sample_lens=int(adapt.get("sample_lens", stage2.get("sample_lens", 5000))),
        seed=seed,
    )

    train_dataset = MixedAdaptationDataset(
        keyword_dataset=keyword_dataset,
        libri_dataset=libri_dataset,
        mix_ratio=float(adapt.get("mix_ratio", 0.5)),
        sample_lens=int(adapt.get("sample_lens", stage2.get("sample_lens", 5000))),
        seed=seed,
    )

    batch_size = int(adapt.get("batch_size_per_gpu", stage2.get("batch_size_per_gpu", 64)))
    num_workers = int(adapt.get("num_workers", stage2.get("num_workers", 2)))
    # The target-keyword val set is a few thousand clips against a 64k-sample virtual
    # train epoch; giving it the full train worker count only multiplies worker
    # processes and open file descriptors for no throughput gain.
    val_workers_cfg = adapt.get("val_num_workers")
    val_num_workers = min(num_workers, 4) if val_workers_cfg is None else int(val_workers_cfg)
    from dma_kws.training.loaders import build_loader_kwargs

    dataloader_cfg = stage2.get("dataloader", {}) or {}
    loader_kwargs = build_loader_kwargs(num_workers, dataloader_cfg)
    val_loader_kwargs = build_loader_kwargs(val_num_workers, dataloader_cfg)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=train_collate_fn,
        drop_last=True,
        **loader_kwargs,
    )

    target_val_dataset = TargetKeywordValDataset(
        manifest_path=eval_manifest,
        keyword=adapt_paths["keyword_str"],
        fbank_root=adapt_paths["fbank_root"],
        tokenizer=tokenizer,
        manifest_root=adapt_paths["data_root"],
    )
    target_val_loader = DataLoader(
        target_val_dataset,
        batch_size=int(adapt.get("val_batch_size", batch_size)),
        shuffle=False,
        num_workers=val_num_workers,
        collate_fn=test_collate_fn,
        drop_last=False,
        **val_loader_kwargs,
    )
    lph_val_loader = _build_val_dataloader(config, tokenizer)

    mix_ratio = float(adapt.get("mix_ratio", 0.5))
    reporter.print_table(
        *adapt_console.dataset_table(
            [
                (
                    "keyword train",
                    len(keyword_dataset),
                    adapt_console.label_breakdown(keyword_dataset),
                ),
                (
                    "libriphrase train pool",
                    len(libri_dataset),
                    f"negative_ratio={stage2.get('negative_ratio', 1)}",
                ),
                (
                    "mixed virtual epoch",
                    len(train_dataset),
                    f"mix_ratio={mix_ratio} (keyword:libriphrase)",
                ),
                (
                    "target keyword val",
                    len(target_val_dataset),
                    adapt_console.label_breakdown(target_val_dataset),
                ),
                (
                    "libriphrase val",
                    len(lph_val_loader.dataset),
                    f"split={(stage2.get('eval', {}) or {}).get('split', 'hard')}",
                ),
            ]
        ),
        title="Datasets",
    )

    lora_rank = int(adapt.get("rank", 16))
    lora_alpha = float(adapt.get("alpha", 32))
    lora_targets = tuple(adapt.get("lora_targets", ("in_proj_weight", "out_proj.weight")))

    if resume_path is not None:
        model = Stage2LoraAdaptationModule(
            config,
            vocab_size=vocab_size,
            init_checkpoint=init_checkpoint,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_targets=lora_targets,
        )
    else:
        model = Stage2LoraAdaptationModule(
            config,
            vocab_size=vocab_size,
            init_checkpoint=init_checkpoint,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_targets=lora_targets,
            adapter_checkpoint=adapter_resume,
        )

    reporter.print_plan(
        adapt_console.lora_rows(
            rank=lora_rank,
            alpha=lora_alpha,
            targets=lora_targets,
            injected=getattr(model, "lora_injected", None),
            param_counts=model.lora_param_counts,
        ),
        title="LoRA Adapters",
    )

    if accelerator == "gpu":
        torch.set_float32_matmul_precision("high")

    log_dir = adapt_paths["phase_dir"] / "logs"
    run_name = f"adapt_{adapt_paths['slug_str']}_{phase}"
    loggers = build_loggers(log_dir, run_name, config=config)

    hparams = collect_hparams(config, section="adapt", extra={"slug": adapt_paths["slug_str"]})
    for train_logger in loggers:
        train_logger.log_hyperparams(hparams)

    recipe = str(training.get("recipe", "adapt"))
    callbacks = build_stage2_callbacks(config, recipe, checkpoint_dir=checkpoint_dir)
    history_callback = build_metrics_history_callback(
        run_name=run_name,
        default_dir=log_dir / run_name,
    )
    callbacks.append(history_callback)

    limit_steps = args.limit_steps or int(adapt.get("max_steps", 3000)) or None
    trainer_kwargs = build_trainer_kwargs(
        config,
        devices,
        limit_steps=limit_steps,
        accelerator=accelerator,
    )
    trainer_kwargs["max_steps"] = limit_steps

    # Adaptation runs on short virtual epochs, so it needs its own validation
    # cadence instead of inheriting the Stage II pretraining schedule.
    adapt_validation = adapt.get("validation", {}) or {}
    if adapt_validation.get("val_check_interval") is not None:
        trainer_kwargs["val_check_interval"] = int(adapt_validation["val_check_interval"])
    if adapt_validation.get("limit_val_batches") is not None:
        trainer_kwargs["limit_val_batches"] = adapt_validation["limit_val_batches"]
    apply_step_based_validation(trainer_kwargs, len(train_dataloader))

    print_run_summary(
        config=config,
        devices=devices,
        accelerator=accelerator,
        section="adapt",
        train_samples=len(train_dataset),
        val_samples=len(target_val_dataset) + len(lph_val_loader.dataset),
        param_counts=model.lora_param_counts,
        extra_rows=[
            ("phase", phase),
            ("batches_per_epoch", str(len(train_dataloader))),
            (
                "val_check_interval",
                f"{trainer_kwargs['val_check_interval']} "
                f"({'steps' if trainer_kwargs.get('check_val_every_n_epoch', 1) is None else 'batches/epoch'})",
            ),
            ("mix_ratio", str(mix_ratio)),
            ("lora_rank", str(lora_rank)),
            ("lora_alpha", str(lora_alpha)),
        ],
        paths={
            "train_manifest": train_manifest,
            "eval_manifest": eval_manifest,
            "checkpoint_dir": checkpoint_dir,
            "log_dir": log_dir,
            "init_checkpoint": Path(init_checkpoint),
        },
    )

    reporter.section(f"Training · {run_name}")
    trainer = pl_mod.Trainer(
        accelerator=accelerator,
        callbacks=callbacks,
        logger=loggers,
        **trainer_kwargs,
    )
    trainer.fit(
        model,
        train_dataloaders=train_dataloader,
        val_dataloaders=[target_val_loader, lph_val_loader],
        ckpt_path=resume_path,
    )

    global_step = int(trainer.global_step)
    adapter_out = adapt_paths["phase_dir"] / f"adapter_{adapt_paths['slug_str']}.pt"
    torch.save(
        {
            "lora_state_dict": lora_state_dict(model.qbyt),
            "config": config,
            "step": global_step,
            "keyword": adapt_paths["keyword_str"],
            "slug": adapt_paths["slug_str"],
            "phase": phase,
            "rank": lora_rank,
            "alpha": lora_alpha,
        },
        adapter_out,
    )

    merge_lora(model.qbyt)
    merged_out = adapt_paths["merged_path"]
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config,
            "step": global_step,
            "keyword": adapt_paths["keyword_str"],
            "slug": adapt_paths["slug_str"],
            "phase": phase,
            "tokenizer_dict_path": str(dict_path),
            "vocab_size": vocab_size,
        },
        merged_out,
    )

    final_adapter = adapt_paths["adapter_path"]
    if adapter_out.resolve() != final_adapter.resolve():
        torch.save(torch.load(adapter_out, map_location="cpu"), final_adapter)

    artifacts = {
        "adapter": adapter_out,
        "merged": merged_out,
        "final_adapter": final_adapter,
    }

    final_metrics = numeric_callback_metrics(dict(trainer.callback_metrics))
    runs_csv = Path(paths["exp_root"]) / "stage2_adapt" / "runs.csv"
    append_wide_row(
        runs_csv,
        build_run_record(
            run_name=run_name,
            hparams=hparams,
            final_metrics=final_metrics,
            best_metrics=history_callback.best,
            global_step=global_step,
            duration_seconds=history_callback.duration_seconds,
        ),
    )

    reporter.section("Artifacts")
    reporter.print_table(*adapt_console.artifact_rows(artifacts), title="Saved Checkpoints")
    log_files = {"runs_csv": runs_csv}
    if history_callback.csv_path is not None:
        log_files["eval_history"] = history_callback.csv_path
    reporter.print_plan(
        [(name, str(path)) for name, path in sorted(log_files.items())],
        title="Metrics CSVs",
    )
    metrics = {
        key: round(value, 6)
        for key, value in final_metrics.items()
        if key.startswith("val/")
    }
    if metrics:
        reporter.print_plan(
            [(key, f"{value:.4f}") for key, value in sorted(metrics.items())],
            title="Final Validation Metrics",
        )
    reporter.done(f"LoRA adaptation complete for phase {phase!r} at step {global_step}.")
    print(
        json.dumps(
            {
                "keyword": adapt_paths["keyword_str"],
                "slug": adapt_paths["slug_str"],
                "phase": phase,
                "step": global_step,
                "metrics": metrics,
                "artifacts": {name: str(path) for name, path in artifacts.items()},
                "logs": {name: str(path) for name, path in log_files.items()},
            }
        )
    )

    return artifacts
