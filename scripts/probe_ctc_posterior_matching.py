#!/usr/bin/env python3
"""Diagnostic: score LibriPhrase pairs from the CTC posterior, bypassing QbyT.

Stage II reads the adapter trunk through QbyT and scores hard negatives at
~0.70 AUC. The same trunk feeds a phoneme CTC head whose loss says the correct
phoneme sequence is recovered well. Those two facts are only compatible if the
information is present in the trunk output and QbyT's readout is losing it --
or if it was never there and the CTC number is misleading.

This script settles it: it drops QbyT entirely and scores each (anchor text,
audio) pair with ``-log P(anchor phonemes | audio)`` from the CTC head, then
reports AUC/EER per word count. That is the same decision the verifier makes,
computed from the posterior directly.

  higher CTC AUC than QbyT  -> the trunk carries it, QbyT's readout drops it
  same AUC as QbyT (~0.70)  -> the trunk does not separate hard negatives

The score is length-normalized: a CTC loss is a sum over the sequence, so
longer anchors score worse for reasons unrelated to whether the phrase is
present. AUC is computed per word-count bucket anyway, but the normalization
keeps the buckets comparable with each other.

Usage:
  python scripts/probe_ctc_posterior_matching.py \
    +experiment=icefall_zipformer_stage2_adapter \
    stage2.phoneme_adapter.trunk.type=conformer \
    prep.checkpoint=/path/to/stage2.ckpt \
    prep.split=hard \
    prep.limit=20000
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
        import torch.nn.functional as F
        import torchmetrics
        from torch.utils.data import DataLoader
    except ImportError as exc:
        raise SystemExit("Missing torch/torchmetrics on this machine.") from exc

    from dma_kws.nn import run_encoder
    from dma_kws.phoneme_adapter.module import ctc_min_input_lengths
    from dma_kws.stage2.module import Stage2LightningModule, assert_adapter_weights_loaded
    from dma_kws.stage2.readout import assert_qbyt_readout_state_loaded
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
    # Every pair needs its own CTC forward, so cap the row count per bucket by
    # default; the full hard split is 270k rows.
    limit = int(prep.get("limit", 0)) or 20000

    model = Stage2LightningModule(config, vocab_size=vocab_size)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    assert_qbyt_readout_version(
        checkpoint,
        source=checkpoint_path,
        allow_legacy=False,
        expected_mode=model.qbyt_readout_mode,
    )
    state = (
        extract_state_dict(checkpoint)
        if checkpoint_path.suffix == ".pt"
        else checkpoint.get("state_dict", checkpoint)
    )
    missing, unexpected = model.load_state_dict(state, strict=False)
    assert_adapter_weights_loaded(model, missing)
    assert_qbyt_readout_state_loaded(
        missing,
        unexpected,
        source=checkpoint_path,
        expected_mode=model.qbyt_readout_mode,
    )
    if missing:
        print(f"Warning: {len(missing)} missing keys when loading checkpoint")
    if unexpected:
        print(f"Warning: {len(unexpected)} unexpected keys when loading checkpoint")
    if model.adapter is None:
        raise SystemExit(
            "This probe scores the phoneme CTC posterior, so it needs the adapter: "
            "run it with stage2.phoneme_adapter.enabled=true."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval().to(device)
    blank_id = model.adapter.blank_id

    frame = _filter_eval_split(
        _build_eval_dataframe(
            eval_paths["test_dir"], eval_paths["csv_files"], eval_paths["aggregate_csv"]
        ),
        split,
    )
    word_counts = frame["anchor_text"].astype(str).str.split().str.len()

    report: dict[str, dict[str, float]] = {}
    for num_words in sorted(word_counts.unique()):
        bucket = frame.loc[word_counts == num_words]
        total_rows = len(bucket)
        if total_rows > limit:
            # Stratify so the subsample keeps the 50/50 balance of the full bucket.
            bucket = (
                bucket.groupby("target", group_keys=False)
                .apply(lambda g: g.sample(n=limit // 2, random_state=2025))
            )
        bucket = bucket.reset_index(drop=True)

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

        ctc_auc = torchmetrics.AUROC(task="binary").to(device)
        ctc_eer = torchmetrics.classification.EER(task="binary").to(device)
        qbyt_auc = torchmetrics.AUROC(task="binary").to(device)
        qbyt_eer = torchmetrics.classification.EER(task="binary").to(device)
        skipped = 0

        with torch.no_grad():
            for batch in loader:
                feat = batch["feat"].to(device)
                feat_lengths = batch["feat_lengths"].to(device)
                anchor = batch["anchor"].to(device)
                labels = batch["label"].to(device).int()

                encoder_out, encoder_mask = run_encoder(
                    model.encoder, feat, feat_lengths, policy=model.stream_policy, mode="eval"
                )
                hidden, log_probs = model.adapter(encoder_out, encoder_mask, with_log_probs=True)
                input_lengths = encoder_mask.squeeze(1).sum(dim=1).to(dtype=torch.long)
                target_lengths = anchor.ne(0).sum(dim=1).to(dtype=torch.long)

                # QbyT score on the same forward pass, as the reference column.
                logits, _ = model.qbyt(
                    hidden,
                    anchor,
                    speech_lengths=input_lengths,
                    text_lengths=target_lengths,
                )
                qbyt_auc.update(torch.sigmoid(logits), labels)
                qbyt_eer.update(torch.sigmoid(logits), labels)

                # CTC score: -log P(anchor phonemes | audio), per sample.
                min_input_lengths = ctc_min_input_lengths(anchor, target_lengths)
                keep = (target_lengths > 0) & (min_input_lengths <= input_lengths)
                skipped += int((~keep).sum().item())
                if not bool(keep.any()):
                    continue
                idx = keep.nonzero(as_tuple=True)[0]
                per_sample = F.ctc_loss(
                    log_probs[idx].transpose(0, 1),
                    anchor[idx],
                    input_lengths[idx],
                    target_lengths[idx],
                    blank=blank_id,
                    reduction="none",
                    zero_infinity=False,
                )
                if not bool(torch.isfinite(per_sample).all()):
                    raise FloatingPointError(
                        "CTC probe produced non-finite losses after infeasible "
                        "targets were filtered"
                    )
                # Length-normalize, then negate: a low CTC loss means the phrase is
                # present, so the higher-is-better score AUC expects is its negative.
                score = -(per_sample / target_lengths[idx].clamp(min=1).float())
                ctc_auc.update(score, labels[idx])
                ctc_eer.update(score, labels[idx])

        entry = {
            "rows_scored": int(len(bucket)),
            "rows_in_split": int(total_rows),
            "ctc_skipped": skipped,
            "ctc_auc": round(float(ctc_auc.compute()), 4),
            "ctc_eer": round(float(ctc_eer.compute()), 4),
            "qbyt_auc": round(float(qbyt_auc.compute()), 4),
            "qbyt_eer": round(float(qbyt_eer.compute()), 4),
        }
        report[f"{num_words}word"] = entry
        print(f"{num_words}word: {json.dumps(entry)}", flush=True)

    print("\n" + json.dumps({"split": split, "checkpoint": str(checkpoint_path), "buckets": report}))


if __name__ == "__main__":
    main()
