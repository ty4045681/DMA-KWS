#!/usr/bin/env python3
"""Evaluate Stage II QbyT on LibriPhrase hard/easy splits (AUC/EER)."""

from __future__ import annotations

import json
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from dma_kws.config import get_tokenizer_config, require_sections
from dma_kws.hydra_app import CONFIG_DIR, resolved_config
from dma_kws.pathing import resolve_dict_path
from dma_kws.stage2.collate import test_collate_fn
from dma_kws.stage2.dataset import LibriPhraseEvalDataset, resolve_stage2_eval_paths
from dma_kws.stage2.module import Stage2LightningModule, assert_adapter_weights_loaded
from dma_kws.tokenizer import load_char_tokenizer
from dma_kws.training.checkpoint_io import assert_qbyt_readout_version, extract_state_dict
from dma_kws.training.device import resolve_accelerator


def _load_model(config: dict, checkpoint_path: Path, vocab_size: int) -> Stage2LightningModule:
    import torch

    model = Stage2LightningModule(config, vocab_size=vocab_size)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    assert_qbyt_readout_version(
        checkpoint,
        source=checkpoint_path,
        expected_alignment=model.qbyt_alignment,
    )

    if checkpoint_path.suffix == ".pt":
        state = extract_state_dict(checkpoint)
    else:
        state = checkpoint.get("state_dict", checkpoint)

    missing, unexpected = model.load_state_dict(state, strict=False)
    assert_adapter_weights_loaded(model, missing)
    qbyt_mismatch = [
        key for key in (*missing, *unexpected) if key.startswith("qbyt.")
    ]
    if qbyt_mismatch:
        raise SystemExit(
            f"Checkpoint {checkpoint_path} does not carry the complete QbyT v6 "
            f"weights: {qbyt_mismatch}"
        )
    if missing:
        print(f"Warning: missing keys when loading checkpoint: {len(missing)}")
    if unexpected:
        print(f"Warning: unexpected keys when loading checkpoint: {len(unexpected)}")
    return model


@hydra.main(version_base=None, config_path=str(CONFIG_DIR), config_name="config")
def main(cfg: DictConfig) -> None:
    try:
        import pytorch_lightning as pl
        from torch.utils.data import DataLoader
    except ImportError as exc:
        raise SystemExit(
            "Missing torch/pytorch-lightning. Install CUDA PyTorch on the eval machine first."
        ) from exc

    config = resolved_config(cfg)
    require_sections(config, ["paths", "stage1", "stage2", "tokenizer"])
    prep = OmegaConf.to_container(cfg.prep, resolve=True)
    if not isinstance(prep, dict):
        prep = {}
    run = cfg.run

    tokenizer_cfg = get_tokenizer_config(config)
    dict_path = resolve_dict_path(config)
    tokenizer = load_char_tokenizer(dict_path, split_with_space=tokenizer_cfg.get("split_with_space", " "))
    vocab_size = len(tokenizer._symbol_table)

    eval_paths = resolve_stage2_eval_paths(config)
    batch_size = int(prep.get("batch_size", 0)) or eval_paths["batch_size"]
    # stage2.eval.split is the single source of truth — it also drives the
    # validation split inside training, so the two cannot report different
    # numbers under the same name. prep.split stays as an explicit per-run override.
    split = str(prep.get("split") or eval_paths["split"])

    dataset = LibriPhraseEvalDataset(
        test_dir=eval_paths["test_dir"],
        fbank_dir=eval_paths["fbank_dir"],
        split=split,
        csv_files=eval_paths["csv_files"],
        aggregate_csv=eval_paths["aggregate_csv"],
        tokenizer=tokenizer,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=eval_paths["num_workers"],
        collate_fn=test_collate_fn,
    )

    checkpoint_path = Path(str(prep.get("checkpoint", "")))
    if not checkpoint_path:
        raise SystemExit("prep.checkpoint is required")
    if not checkpoint_path.exists():
        raise SystemExit(f"Checkpoint not found: {checkpoint_path}")

    model = _load_model(config, checkpoint_path, vocab_size)

    accelerator, devices = resolve_accelerator(str(run.device))
    trainer = pl.Trainer(accelerator=accelerator, devices=devices, logger=False, enable_checkpointing=False)
    results = trainer.test(model, dataloaders=dataloader, verbose=False)

    metrics = results[0] if results else {}
    auc = float(metrics.get("test/auc", 0.0))
    eer = float(metrics.get("test/eer", 0.0))
    test_metrics = {
        str(name).removeprefix("test/"): float(value)
        for name, value in metrics.items()
        if str(name).startswith("test/")
    }

    output = {
        "split": split,
        "auc": auc,
        "eer": eer,
        "metrics": test_metrics,
        "stream": model.stream_policy.describe(),
    }
    print(json.dumps(output))


if __name__ == "__main__":
    main()
