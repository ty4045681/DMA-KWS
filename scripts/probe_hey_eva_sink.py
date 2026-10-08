#!/usr/bin/env python3
"""In-domain pooled/sink probe for the "hey eva" adaptation.

``probe_qbyt_v41_offline.py`` dumps the LibriPhrase hard split. That is the right
dev set for a general model and the wrong one for a keyword-adapted model: the
post-hoc sink head fitted there reweights the sink branch toward general-domain
behaviour. Measured on the joint "hey eva" model, writing that head back cost
0.005 AUC on held-out real speakers (0.9849 -> 0.9796) while gaining 0.002 on
LibriPhrase. This probe dumps the same artefacts for an adaptation manifest, so
the logistic head can be fitted on the deployment domain instead.

The output is drop-in compatible with ``scripts/analyze_v41_offline.py``:
``label``, ``logits``, ``pos_logits``, ``pos_mask``, ``sink``, ``audio_len``,
``text_len``, ``sample_id``.

Usage:
  .venv/bin/python scripts/probe_hey_eva_sink.py \
    --checkpoint CKPT.pt \
    --manifest data/dma-kws/processed/adapt/hey_eva_v42/manifests/real_train.csv \
    --data-root data/dma-kws/processed/adapt/hey_eva_v42 \
    --output outputs/hey_eva_v42_sinkfit/probe_real/hard_probe.npz
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--keyword", default="hey eva")
    parser.add_argument("--experiment", default="adapt_hey_eva_v42")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--amp", default="fp16", choices=["fp16", "off"])
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    import torch
    from torch.utils.data import DataLoader, Subset

    from dma_kws.config import compose_config, config_to_dict, get_tokenizer_config
    from dma_kws.nn import run_encoder
    from dma_kws.pathing import resolve_dict_path
    from dma_kws.stage2.adapt_dataset import TargetKeywordValDataset
    from dma_kws.stage2.collate import test_collate_fn
    from dma_kws.stage2.module import Stage2LightningModule, assert_adapter_weights_loaded
    from dma_kws.tokenizer import load_char_tokenizer
    from dma_kws.training.checkpoint_io import assert_qbyt_readout_version, extract_state_dict

    config = config_to_dict(compose_config(overrides=[f"+experiment={args.experiment}"]))
    tokenizer_cfg = get_tokenizer_config(config)
    tokenizer = load_char_tokenizer(
        resolve_dict_path(config),
        split_with_space=tokenizer_cfg.get("split_with_space", " "),
    )
    vocab_size = len(tokenizer._symbol_table)

    model = Stage2LightningModule(config, vocab_size=vocab_size)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    assert_qbyt_readout_version(
        checkpoint, source=args.checkpoint, expected_alignment=model.qbyt_score
    )
    missing, unexpected = model.load_state_dict(extract_state_dict(checkpoint), strict=False)
    assert_adapter_weights_loaded(model, missing)
    if [k for k in (*missing, *unexpected) if k.startswith("qbyt.")]:
        raise SystemExit("checkpoint does not carry complete QbyT weights")

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    model.eval().to(device)
    print("readout spec:", getattr(model, "qbyt_score", None).value)

    dataset = TargetKeywordValDataset(
        manifest_path=args.manifest,
        keyword=args.keyword,
        fbank_root=args.data_root / "fbank",
        tokenizer=tokenizer,
        manifest_root=args.data_root,
    )
    if args.limit > 0:
        dataset = Subset(dataset, list(range(min(args.limit, len(dataset)))))
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=test_collate_fn,
    )
    print("samples:", len(dataset), "device:", device, "amp:", args.amp)

    captured: dict = {}
    handle = model.qbyt.phone_matchor.register_forward_hook(
        lambda _m, _i, output: captured.__setitem__("combined", output.detach())
    )

    logits_parts, sink_parts, audio_parts, text_parts = [], [], [], []
    label_parts, sample_parts, pos_parts, mask_parts = [], [], [], []
    started = time.time()

    with torch.no_grad():
        for step, batch in enumerate(loader):
            feat = batch["feat"].to(device, non_blocking=True)
            feat_lengths = batch["feat_lengths"].to(device)
            anchor = batch["anchor"].to(device)

            def _forward():
                encoder_out, encoder_mask = run_encoder(
                    model.encoder, feat, feat_lengths, policy=model.stream_policy, mode="eval"
                )
                encoder_lens = encoder_mask.squeeze(1).sum(dim=1).to(dtype=torch.long)
                text_lengths = anchor.ne(0).sum(dim=1).to(dtype=torch.long)
                logits, _seq, details = model.qbyt.forward_with_readout_details(
                    encoder_out,
                    anchor,
                    speech_lengths=encoder_lens,
                    text_lengths=text_lengths,
                )
                combined = captured["combined"]
                rows = torch.arange(combined.size(0), device=device)
                sink = combined[rows, text_lengths.clamp(max=combined.size(1) - 1)]
                return logits, details, sink, encoder_lens, text_lengths

            if use_cuda and args.amp != "off":
                with torch.autocast("cuda", dtype=torch.float16):
                    logits, details, sink, encoder_lens, text_lengths = _forward()
            else:
                logits, details, sink, encoder_lens, text_lengths = _forward()

            if details.position_logits is None or details.position_mask is None:
                raise SystemExit("this probe requires a pooling readout with position logits")
            logits_parts.append(logits.detach().float().cpu().numpy())
            sink_parts.append(sink.detach().float().cpu().numpy())
            audio_parts.append(encoder_lens.detach().cpu().numpy())
            text_parts.append(text_lengths.detach().cpu().numpy())
            label_parts.append(batch["label"].numpy())
            sample_parts.append(batch["sample_id"].numpy())
            pos_parts.append(details.position_logits.detach().float().cpu().numpy().astype(np.float32))
            mask_parts.append(details.position_mask.detach().cpu().numpy().astype(bool))

            if step % 20 == 0:
                done = sum(p.shape[0] for p in logits_parts)
                print("step", step, "samples", done,
                      "rate", round(done / max(time.time() - started, 1e-9), 1), flush=True)
    handle.remove()

    logits = np.concatenate(logits_parts).astype(np.float32)
    sink = np.concatenate(sink_parts).astype(np.float32)
    audio_len = np.concatenate(audio_parts).astype(np.int32)
    text_len = np.concatenate(text_parts).astype(np.int32)
    labels = np.concatenate(label_parts).astype(np.int8)
    sample_ids = np.concatenate(sample_parts).astype(np.int64)
    max_width = max(p.shape[1] for p in pos_parts)
    pos_logits = np.zeros((logits.shape[0], max_width), dtype=np.float32)
    pos_mask = np.zeros((logits.shape[0], max_width), dtype=bool)
    cursor = 0
    for part, part_mask in zip(pos_parts, mask_parts):
        n = part.shape[0]
        pos_logits[cursor:cursor + n, : part.shape[1]] = part
        pos_mask[cursor:cursor + n, : part.shape[1]] = part_mask
        cursor += n

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        logits=logits, sink=sink, pos_logits=pos_logits, pos_mask=pos_mask,
        audio_len=audio_len, text_len=text_len, label=labels, sample_id=sample_ids,
    )
    summary = {
        "checkpoint": str(args.checkpoint),
        "manifest": str(args.manifest),
        "num_samples": int(logits.shape[0]),
        "num_positive": int(labels.sum()),
        "max_text_width": int(max_width),
        "output": str(args.output),
    }
    (args.output.parent / "hard_probe_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
