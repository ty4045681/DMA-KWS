"""Step A training runner: phoneme CTC adapter on a frozen encoder."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dma_kws.pathing import resolve_dict_path
from dma_kws.training.loaders import build_loader_kwargs

BLANK_ID = 0


@dataclass
class PhonemeAdapterTrainArgs:
    """Runtime options for Step A training."""

    train_manifest: str = ""
    dev_manifest: str = ""
    init_checkpoint: str = ""
    device: str = "cuda"
    devices: int = 1
    limit_steps: int = 0
    resume_from: str = ""


def _resolve_manifest(override: str, config_value: str, default: Path) -> Path:
    if override:
        return Path(override)
    if config_value:
        return Path(config_value)
    return default


def resolve_val_check_interval(adapter_cfg: dict[str, Any]) -> int:
    """Interval ``build_trainer_kwargs`` will actually use (``validation`` wins)."""
    validation_cfg = adapter_cfg.get("validation", {}) or {}
    return int(
        validation_cfg.get("val_check_interval", adapter_cfg.get("val_check_interval", 1000))
    )


def build_adapter_callbacks(checkpoint_dir: Path, adapter_cfg: dict[str, Any]) -> tuple[list, Any]:
    from pytorch_lightning.callbacks import ModelCheckpoint

    checkpoint_cfg = adapter_cfg.get("checkpoint", {}) or {}
    every_n_train_steps = int(checkpoint_cfg.get("every_n_train_steps", 0))
    val_check_interval = resolve_val_check_interval(adapter_cfg)

    # ``val/per`` only enters callback_metrics when validation has just run. If
    # the save cadence is not a multiple of the validation cadence, ModelCheckpoint
    # silently degrades to "monitor not available, skipping" and stops saving
    # best-by-PER without failing the run.
    if every_n_train_steps and every_n_train_steps % val_check_interval != 0:
        raise SystemExit(
            f"phoneme_adapter.checkpoint.every_n_train_steps={every_n_train_steps} must be a "
            f"multiple of the validation interval ({val_check_interval}); otherwise val/per is "
            "not available at save time and no best checkpoint is ever written."
        )

    checkpoint_callback = ModelCheckpoint(
        dirpath=str(checkpoint_dir),
        monitor="val/per",
        mode="min",
        save_top_k=int(checkpoint_cfg.get("save_top_k", 3)),
        every_n_train_steps=every_n_train_steps or None,
        filename="adapter_{step:07d}_{val_per:.4f}",
        save_last=True,
    )
    return [checkpoint_callback], checkpoint_callback


def run_phoneme_adapter_training(config: dict[str, Any], args: PhonemeAdapterTrainArgs) -> None:
    """Train the phoneme CTC adapter from a loaded config dict."""
    try:
        import pytorch_lightning as pl
        import torch
        from torch.utils.data import DataLoader
    except ImportError as exc:
        raise SystemExit(
            "Missing torch/pytorch-lightning. Install CUDA PyTorch on the training machine first."
        ) from exc

    from dma_kws.config import (
        fbank_kwargs,
        get_fbank_config,
        get_tokenizer_config,
        require_sections,
    )
    from dma_kws.phoneme_adapter.lightning import PhonemeAdapterCtcModule
    from dma_kws.runlog import build_loggers
    from dma_kws.stage1.dataset import Stage1Dataset, stage1_collate_fn
    from dma_kws.stage2.fbank import FbankExtractor
    from dma_kws.tokenizer import load_char_tokenizer
    from dma_kws.training.checkpoint_io import export_model_pt
    from dma_kws.training.ddp import apply_step_based_validation, build_trainer_kwargs
    from dma_kws.training.device import resolve_accelerator_and_devices
    from dma_kws.training.resume import resolve_resume_path

    require_sections(config, ["paths", "stage1", "phoneme_adapter", "tokenizer", "training"])
    paths = config["paths"]
    stage1 = config["stage1"]
    adapter_cfg = config["phoneme_adapter"]
    training = config["training"]

    processed_dir = Path(paths["processed_root"]) / "stage1_phoneme_ctc"
    train_manifest = _resolve_manifest(
        args.train_manifest,
        str(adapter_cfg.get("train_manifest", "")).strip(),
        processed_dir / "train.jsonl",
    )
    dev_manifest = _resolve_manifest(
        args.dev_manifest,
        str(adapter_cfg.get("dev_manifest", "")).strip(),
        processed_dir / "dev.jsonl",
    )
    if not train_manifest.exists():
        raise SystemExit(f"Step A train manifest not found: {train_manifest}")

    dict_path = resolve_dict_path(config)
    tokenizer_cfg = get_tokenizer_config(config)
    tokenizer = load_char_tokenizer(
        dict_path, split_with_space=tokenizer_cfg.get("split_with_space", " ")
    )
    vocab_size = len(tokenizer._symbol_table)
    blank_id = int(tokenizer.symbol_table.get("<blank>", BLANK_ID))

    # The trunk is trained to feed Stage II, so it must see Stage II's feature
    # space. dma_kws.audio.extract_fbank is not that space: it skips the 1<<15
    # scaling FbankExtractor applies, on top of differing dither/snip_edges.
    extractor = FbankExtractor(**fbank_kwargs(get_fbank_config(config)))
    sample_rate = int(stage1.get("sample_rate", 16000))
    num_mel_bins = int(stage1.get("input_dim", 80))

    pl.seed_everything(int(training.get("seed", 2025)), workers=True)

    def _dataset(manifest: Path) -> Stage1Dataset:
        return Stage1Dataset(
            manifest,
            tokenizer=tokenizer,
            sample_rate=sample_rate,
            num_mel_bins=num_mel_bins,
            feature_extractor=extractor,
        )

    train_dataset = _dataset(train_manifest)
    if len(train_dataset) == 0:
        raise SystemExit(f"No records found in {train_manifest}")

    num_workers = int(adapter_cfg.get("num_workers", 4))
    loader_kwargs = build_loader_kwargs(num_workers, adapter_cfg.get("dataloader", {}) or {})
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=int(adapter_cfg.get("batch_size_per_gpu", 32)),
        shuffle=True,
        num_workers=num_workers,
        collate_fn=stage1_collate_fn,
        drop_last=True,
        **loader_kwargs,
    )

    validation_cfg = adapter_cfg.get("validation", {}) or {}
    dev_dataloader = None
    if dev_manifest.exists():
        dev_dataset = _dataset(dev_manifest)
        if len(dev_dataset) > 0:
            dev_dataloader = DataLoader(
                dev_dataset,
                batch_size=int(
                    validation_cfg.get("batch_size", adapter_cfg.get("batch_size_per_gpu", 32))
                ),
                shuffle=False,
                num_workers=num_workers,
                collate_fn=stage1_collate_fn,
                **loader_kwargs,
            )
    if dev_dataloader is None:
        raise SystemExit(
            f"Step A dev manifest not found or empty: {dev_manifest}. Dev PER is the whole "
            "point of this stage (it decides whether the frozen representation can carry "
            "phonemes at all), so it is required rather than optional."
        )

    init_checkpoint = args.init_checkpoint or str(adapter_cfg.get("init_checkpoint", ""))
    model = PhonemeAdapterCtcModule(
        config,
        vocab_size=vocab_size,
        blank_id=blank_id,
        init_checkpoint=init_checkpoint or None,
    )

    accelerator, devices = resolve_accelerator_and_devices(args.device, args.devices)
    if accelerator == "gpu":
        torch.set_float32_matmul_precision("high")

    checkpoint_dir = Path(
        adapter_cfg.get("checkpoint_dir", Path(paths["exp_root"]) / "phoneme_adapter" / "checkpoints")
    )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    callbacks, checkpoint_callback = build_adapter_callbacks(checkpoint_dir, adapter_cfg)

    log_dir = adapter_cfg.get("log_dir", Path(paths["exp_root"]) / "phoneme_adapter" / "logs")
    run_name = str(adapter_cfg.get("run_name", "phoneme_adapter_ctc"))
    loggers = build_loggers(log_dir, run_name, config=config)

    trainer_kwargs = build_trainer_kwargs(
        config,
        devices,
        section="phoneme_adapter",
        limit_steps=args.limit_steps or None,
        accelerator=accelerator,
    )
    # Lightning reads an integer val_check_interval as a batch index inside one
    # epoch and raises when it exceeds the epoch length, which a small manifest
    # or a limit_steps smoke run hits immediately.
    apply_step_based_validation(trainer_kwargs, len(train_dataloader))
    trainer = pl.Trainer(
        accelerator=accelerator,
        logger=loggers,
        callbacks=callbacks,
        **trainer_kwargs,
    )

    resume_path = resolve_resume_path(args.resume_from, checkpoint_dir)
    trainer.fit(model, train_dataloader, dev_dataloader, ckpt_path=resume_path)

    if checkpoint_callback is not None and checkpoint_callback.best_model_path:
        best_state = torch.load(checkpoint_callback.best_model_path, map_location="cpu")
        model.load_state_dict(best_state["state_dict"])
        print(f"Loaded best adapter weights from {checkpoint_callback.best_model_path}")

    global_step = int(trainer.global_step)
    # Only the adapter is exported: Stage II rebuilds the encoder from its own
    # config, and the payload carries the config so assert_stream_policy_matches
    # can reject a Stage II run at a different operating point.
    output_path = export_model_pt(
        model.adapter,
        checkpoint_dir / f"adapter_step{global_step:06d}.pt",
        config=config,
        dict_path=dict_path,
        vocab_size=vocab_size,
        step=global_step,
        blank_id=blank_id,
    )
    print(f"Saved {output_path}")
    print(f"Use it as stage2.phoneme_adapter.init_checkpoint={output_path}")
