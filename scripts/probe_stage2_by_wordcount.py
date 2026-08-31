#!/usr/bin/env python3
"""Diagnostic: Stage II AUC/EER broken down by anchor word count.

The LibriPhrase hard split is 58% single-word anchors while GigaPhrase training
anchors are only 12% single-word. A headline AUC therefore mostly reports the
bucket the model saw least. This script scores each bucket separately so a
train/eval length mismatch is visible instead of averaged away.

Reuses ``LibriPhraseEvalDataset`` and the eval checkpoint loader unchanged, so
the per-bucket numbers are comparable with ``eval_stage2_libriphrase.py``.

Usage:
  python scripts/probe_stage2_by_wordcount.py \
    +experiment=icefall_zipformer_stage2_adapter \
    prep.checkpoint=/path/to/step_011000.ckpt
"""

from __future__ import annotations

import json
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from dma_kws.config import get_tokenizer_config, require_sections
from dma_kws.hydra_app import CONFIG_DIR, resolved_config
from dma_kws.pathing import resolve_dict_path
from dma_kws.stage2.collate import test_collate_fn
from dma_kws.stage2.dataset import (
    LibriPhraseEvalDataset,
    _build_eval_dataframe,
    _filter_eval_split,
    resolve_stage2_eval_paths,
)
from dma_kws.tokenizer import load_char_tokenizer


@hydra.main(version_base=None, config_path=str(CONFIG_DIR), config_name="config")
def main(cfg: DictConfig) -> None:
    try:
        import torch
        import torchmetrics
        from torch.utils.data import DataLoader
    except ImportError as exc:
        raise SystemExit("Missing torch/torchmetrics on this machine.") from exc

    # Imported here so the friendly ImportError above fires first on a CPU box.
    from dma_kws.stage2.module import Stage2LightningModule, assert_adapter_weights_loaded
    from dma_kws.stage2.readout import assert_qbyt_alignment_state_loaded
    from dma_kws.training.checkpoint_io import assert_qbyt_readout_version, extract_state_dict

    config = resolved_config(cfg)
    require_sections(config, ["paths", "stage1", "stage2", "tokenizer"])
    prep = OmegaConf.to_container(cfg.prep, resolve=True)
    if not isinstance(prep, dict):
        prep = {}

    checkpoint_path = Path(str(prep.get("checkpoint", "")))
    if not str(checkpoint_path):
        raise SystemExit("prep.checkpoint is required")
    if not checkpoint_path.exists():
        raise SystemExit(f"Checkpoint not found: {checkpoint_path}")

    tokenizer_cfg = get_tokenizer_config(config)
    dict_path = resolve_dict_path(config)
    tokenizer = load_char_tokenizer(
        dict_path, split_with_space=tokenizer_cfg.get("split_with_space", " ")
    )
    vocab_size = len(tokenizer._symbol_table)

    eval_paths = resolve_stage2_eval_paths(config)
    split = str(prep.get("split") or eval_paths["split"])
    batch_size = int(prep.get("batch_size", 0)) or eval_paths["batch_size"]

    model = Stage2LightningModule(config, vocab_size=vocab_size)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    assert_qbyt_readout_version(
        checkpoint,
        source=checkpoint_path,
        expected_alignment=model.qbyt_alignment,
    )
    state = (
        extract_state_dict(checkpoint)
        if checkpoint_path.suffix == ".pt"
        else checkpoint.get("state_dict", checkpoint)
    )
    missing, unexpected = model.load_state_dict(state, strict=False)
    assert_adapter_weights_loaded(model, missing)
    assert_qbyt_alignment_state_loaded(
        missing,
        unexpected,
        source=checkpoint_path,
        expected_topology=model.qbyt_alignment.topology,
    )
    if missing:
        print(f"Warning: {len(missing)} missing keys when loading checkpoint")
    if unexpected:
        print(f"Warning: {len(unexpected)} unexpected keys when loading checkpoint")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval().to(device)

    frame = _filter_eval_split(
        _build_eval_dataframe(
            eval_paths["test_dir"], eval_paths["csv_files"], eval_paths["aggregate_csv"]
        ),
        split,
    )
    word_counts = frame["anchor_text"].astype(str).str.split().str.len()

    report: dict[str, dict[str, float]] = {}
    for num_words in sorted(word_counts.unique()):
        bucket = frame.loc[word_counts == num_words].reset_index(drop=True)
        dataset = LibriPhraseEvalDataset(
            test_dir=eval_paths["test_dir"],
            fbank_dir=eval_paths["fbank_dir"],
            split=split,
            tokenizer=tokenizer,
            df=bucket,
        )
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=eval_paths["num_workers"],
            collate_fn=test_collate_fn,
        )

        auc_metric = torchmetrics.AUROC(task="binary").to(device)
        eer_metric = torchmetrics.classification.EER(task="binary").to(device)
        with torch.no_grad():
            for batch in loader:
                logits, _ = model(
                    batch["feat"].to(device),
                    batch["feat_lengths"].to(device),
                    batch["anchor"].to(device),
                )
                preds = torch.sigmoid(logits)
                labels = batch["label"].to(device).int()
                auc_metric.update(preds, labels)
                eer_metric.update(preds, labels)

        report[f"{num_words}word"] = {
            "rows": int(len(bucket)),
            "share_pct": round(100.0 * len(bucket) / len(frame), 1),
            "auc": round(float(auc_metric.compute()), 4),
            "eer": round(float(eer_metric.compute()), 4),
        }
        print(f"{num_words}word: {json.dumps(report[f'{num_words}word'])}", flush=True)

    print("\n" + json.dumps({"split": split, "checkpoint": str(checkpoint_path), "buckets": report}))


if __name__ == "__main__":
    main()
