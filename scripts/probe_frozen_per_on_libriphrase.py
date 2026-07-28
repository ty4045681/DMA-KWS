#!/usr/bin/env python3
"""Diagnostic: phoneme error rate of the frozen encoder + trunk on LibriPhrase clips.

Step A reports PER 5.5% on LibriSpeech *sentences*. That number has been used
here as evidence that the frozen representation carries phonetic detail, but the
two settings differ in every way that matters: LibriPhrase clips are 0.3-1.5 s
instead of several seconds, so there is no long context and no sentence-level
redundancy to fall back on.

This script decodes each LibriPhrase eval clip against *its own* transcript
(``comparison_text``), not the anchor, and reports PER per word count. It answers
one question with no scorer design in the way:

  can the frozen encoder + trunk read phonemes off these clips at all?

  PER near Step A's 5.5%  -> the phonetic detail is there; the frozen route's
                             ceiling is in how QbyT consumes it, and more trunk
                             or QbyT capacity is the lever.
  PER far worse (>25%)    -> the frozen stack cannot read these clips, so no
                             readout on top of it can separate hard negatives.

Greedy CTC decoding, so this is a lower bound on what the posterior encodes --
a beam search would do better. That is deliberate: the question is whether the
information is present at all, and greedy is the cheapest honest probe.

Usage:
  python scripts/probe_frozen_per_on_libriphrase.py \
    +experiment=icefall_zipformer_stage2_adapter \
    stage2.phoneme_adapter.init_checkpoint=/path/to/adapter.pt \
    prep.limit=4000
"""

from __future__ import annotations

import json
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

from dma_kws.config import get_tokenizer_config, require_sections
from dma_kws.g2p import make_g2p, text_to_phonemes
from dma_kws.hydra_app import CONFIG_DIR, resolved_config
from dma_kws.metrics import collapse_ctc, edit_distance
from dma_kws.pathing import resolve_dict_path
from dma_kws.stage2.dataset import (
    _build_eval_dataframe,
    _filter_eval_split,
    _resolve_eval_fbank_path,
    resolve_stage2_eval_paths,
)
from dma_kws.tokenizer import load_char_tokenizer, tokenize_phoneme_string


@hydra.main(version_base=None, config_path=str(CONFIG_DIR), config_name="config")
def main(cfg: DictConfig) -> None:
    try:
        import torch
    except ImportError as exc:
        raise SystemExit("Missing torch on this machine.") from exc

    from dma_kws.nn import run_encoder
    from dma_kws.stage2.module import Stage2LightningModule

    config = resolved_config(cfg)
    require_sections(config, ["paths", "stage1", "stage2", "tokenizer"])
    prep = OmegaConf.to_container(cfg.prep, resolve=True)
    if not isinstance(prep, dict):
        prep = {}

    tokenizer_cfg = get_tokenizer_config(config)
    dict_path = resolve_dict_path(config)
    tokenizer = load_char_tokenizer(
        dict_path, split_with_space=tokenizer_cfg.get("split_with_space", " ")
    )
    vocab_size = len(tokenizer._symbol_table)

    # No Stage II checkpoint on purpose: this measures the frozen encoder plus the
    # Step A trunk, which is exactly what every frozen Stage II run starts from.
    # QbyT stays random and is never called.
    model = Stage2LightningModule(config, vocab_size=vocab_size)
    if model.adapter is None:
        raise SystemExit(
            "This probe decodes the phoneme CTC head, so it needs the adapter: run with "
            "stage2.phoneme_adapter.enabled=true and an init_checkpoint."
        )
    stage2_ckpt = str(prep.get("checkpoint", "")).strip()
    if stage2_ckpt:
        from dma_kws.training.checkpoint_io import extract_state_dict

        raw = torch.load(stage2_ckpt, map_location="cpu")
        state = (
            extract_state_dict(raw)
            if Path(stage2_ckpt).suffix == ".pt"
            else raw.get("state_dict", raw)
        )
        missing, _ = model.load_state_dict(state, strict=False)
        print(f"Loaded Stage II weights from {stage2_ckpt} (missing={len(missing)})")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval().to(device)
    blank_id = model.adapter.blank_id

    eval_paths = resolve_stage2_eval_paths(config)
    split = str(prep.get("split") or eval_paths["split"])
    limit = int(prep.get("limit", 0)) or 4000

    frame = _filter_eval_split(
        _build_eval_dataframe(
            eval_paths["test_dir"], eval_paths["csv_files"], eval_paths["aggregate_csv"]
        ),
        split,
    )
    # One row per distinct clip: the same wav appears in many pairs, and decoding
    # it repeatedly would weight frequently-paired clips more heavily.
    clips = (
        frame[["comparison", "comparison_text"]]
        .dropna()
        .drop_duplicates(subset=["comparison"])
        .reset_index(drop=True)
    )
    clips["nw"] = clips["comparison_text"].astype(str).str.split().str.len()
    print(f"distinct clips in {split} split: {len(clips)}")

    g2p = make_g2p()
    report: dict[str, dict[str, float]] = {}

    for num_words in sorted(clips["nw"].unique()):
        bucket = clips.loc[clips["nw"] == num_words]
        total = len(bucket)
        if total > limit:
            bucket = bucket.sample(n=limit, random_state=2025)
        bucket = bucket.reset_index(drop=True)

        total_dist = 0
        total_ref = 0
        decoded = 0
        empty_hyps = 0
        for row in bucket.itertuples(index=False):
            fbank_path = _resolve_eval_fbank_path(
                eval_paths["test_dir"], row.comparison, fbank_dir=eval_paths["fbank_dir"]
            )
            if not Path(fbank_path).is_file():
                continue
            feats = torch.from_numpy(np.load(fbank_path)).unsqueeze(0).to(device)
            lengths = torch.tensor([feats.size(1)], dtype=torch.long, device=device)

            with torch.no_grad():
                encoder_out, encoder_mask = run_encoder(
                    model.encoder, feats, lengths, policy=model.stream_policy, mode="eval"
                )
                _hidden, log_probs = model.adapter(
                    encoder_out, encoder_mask, with_log_probs=True
                )
                frames = int(encoder_mask.squeeze(1).sum().item())
                hyp = collapse_ctc(
                    log_probs[0, :frames].argmax(dim=-1).tolist(), blank_id=blank_id
                )

            ref = tokenize_phoneme_string(
                tokenizer, " ".join(text_to_phonemes(g2p, str(row.comparison_text)))
            )
            if not ref:
                continue
            total_dist += edit_distance(ref, hyp)
            total_ref += len(ref)
            decoded += 1
            if not hyp:
                empty_hyps += 1

        entry = {
            "clips_decoded": decoded,
            "clips_in_split": int(total),
            "ref_phonemes": total_ref,
            "empty_hyps": empty_hyps,
            "per": round(total_dist / total_ref, 4) if total_ref else None,
        }
        report[f"{num_words}word"] = entry
        print(f"{num_words}word: {json.dumps(entry)}", flush=True)

    print(
        "\n"
        + json.dumps(
            {
                "split": split,
                "note": "Step A reference PER on LibriSpeech sentences: 0.0553",
                "buckets": report,
            }
        )
    )


if __name__ == "__main__":
    main()
