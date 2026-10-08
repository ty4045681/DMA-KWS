#!/usr/bin/env python3
"""Step-0 offline probe for the QbyT v4.1 sink head.

Dumps LibriPhrase hard-split scoring artefacts for zero-training analysis:
pooled utterance logit, per-text-position logits + mask, the final sink hidden
state, audio/text lengths and labels.

Usage:
  source ~/.dma-kws-env.sh
  .venv/bin/python scripts/probe_qbyt_v41_offline.py +
    experiment=icefall_zipformer_stage2_eps_softmin_v41_musan_cached_zhen3m_50k
    prep.checkpoint=<v4.1 .pt>
    prep.output_dir=outputs/v41_offline_probe
    prep.limit=512            (optional smoke test)
    prep.amp=off              (optional fp32)
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

from dma_kws.config import get_tokenizer_config, require_sections
from dma_kws.hydra_app import CONFIG_DIR, resolved_config
from dma_kws.pathing import resolve_dict_path
from dma_kws.stage2.collate import test_collate_fn
from dma_kws.stage2.dataset import LibriPhraseEvalDataset, resolve_stage2_eval_paths
from dma_kws.stage2.module import Stage2LightningModule, assert_adapter_weights_loaded
from dma_kws.tokenizer import load_char_tokenizer
from dma_kws.training.checkpoint_io import assert_qbyt_readout_version, extract_state_dict


def _load_model(config, checkpoint_path, vocab_size, allow_readout_mismatch=False):
    import torch

    model = Stage2LightningModule(config, vocab_size=vocab_size)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if allow_readout_mismatch:
        try:
            assert_qbyt_readout_version(
                checkpoint,
                source=checkpoint_path,
                expected_alignment=model.qbyt_score,
            )
        except (Exception, SystemExit) as exc:
            print("WARNING: readout spec mismatch ignored for warm start:", exc)
    else:
        assert_qbyt_readout_version(
            checkpoint,
            source=checkpoint_path,
            expected_alignment=model.qbyt_score,
        )
    state = extract_state_dict(checkpoint)
    missing, unexpected = model.load_state_dict(state, strict=False)
    assert_adapter_weights_loaded(model, missing)
    if allow_readout_mismatch:
        print("warm start: missing", len(missing), "unexpected", len(unexpected))
        print("  missing qbyt keys:", [k for k in missing if k.startswith("qbyt.")])
        print("  unexpected qbyt keys:", [k for k in unexpected if k.startswith("qbyt.")])
        return model
    qbyt_mismatch = [k for k in (*missing, *unexpected) if k.startswith("qbyt.")]
    if qbyt_mismatch:
        raise SystemExit("Checkpoint does not carry the complete QbyT weights: " + str(qbyt_mismatch))
    if missing:
        print("Warning: missing keys:", len(missing))
    if unexpected:
        print("Warning: unexpected keys:", len(unexpected))
    return model


@hydra.main(version_base=None, config_path=str(CONFIG_DIR), config_name="config")
def main(cfg: DictConfig) -> None:
    import torch
    from torch.utils.data import DataLoader, Subset

    from dma_kws.nn import run_encoder

    config = resolved_config(cfg)
    require_sections(config, ["paths", "stage1", "stage2", "tokenizer"])
    prep = OmegaConf.to_container(cfg.prep, resolve=True)
    if not isinstance(prep, dict):
        prep = {}

    checkpoint_path = Path(str(prep.get("checkpoint", "")))
    if not checkpoint_path or not checkpoint_path.exists():
        raise SystemExit("prep.checkpoint is required and must exist")

    output_dir = Path(str(prep.get("output_dir", "outputs/v41_offline_probe")))
    output_dir.mkdir(parents=True, exist_ok=True)
    limit = int(prep.get("limit", 0) or 0)
    amp = str(prep.get("amp", "fp16")).lower()
    allow_readout_mismatch = bool(prep.get("allow_readout_mismatch", False))
    output_name = str(prep.get("output_name", "hard_probe.npz"))

    tokenizer_cfg = get_tokenizer_config(config)
    tokenizer = load_char_tokenizer(
        resolve_dict_path(config),
        split_with_space=tokenizer_cfg.get("split_with_space", " "),
    )
    vocab_size = len(tokenizer._symbol_table)
    eval_paths = resolve_stage2_eval_paths(config)
    split = str(prep.get("split") or eval_paths["split"])

    dataset = LibriPhraseEvalDataset(
        test_dir=eval_paths["test_dir"],
        fbank_dir=eval_paths["fbank_dir"],
        split=split,
        csv_files=eval_paths["csv_files"],
        aggregate_csv=eval_paths["aggregate_csv"],
        tokenizer=tokenizer,
    )
    if limit > 0:
        dataset = Subset(dataset, list(range(min(limit, len(dataset)))))

    batch_size = int(prep.get("batch_size", 0) or eval_paths["batch_size"])
    workers = int(prep.get("num_workers", eval_paths["num_workers"]))
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        collate_fn=test_collate_fn,
    )

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    model = _load_model(
        config,
        checkpoint_path,
        vocab_size,
        allow_readout_mismatch=allow_readout_mismatch,
    )
    model.eval()
    model.to(device)

    spec = getattr(model, "qbyt_score", None)
    print("readout spec:", spec.value if spec is not None else "unknown")
    print("total samples:", len(dataset), "batch:", batch_size, "workers:", workers, "device:", device, "amp:", amp)

    captured = {}

    def _hook(_module, _inputs, output):
        captured["combined"] = output.detach()

    handle = model.qbyt.phone_matchor.register_forward_hook(_hook)

    logits_parts = []
    pos_parts = []
    mask_parts = []
    sink_parts = []
    audio_parts = []
    text_parts = []
    label_parts = []
    sample_parts = []

    started = time.time()
    with torch.no_grad():
        for step, batch in enumerate(loader):
            feat = batch["feat"].to(device, non_blocking=True)
            feat_lengths = batch["feat_lengths"].to(device)
            anchor = batch["anchor"].to(device)
            labels = batch["label"].numpy()
            sample_ids = batch["sample_id"].numpy()

            def _forward():
                encoder_out, encoder_mask = run_encoder(
                    model.encoder,
                    feat,
                    feat_lengths,
                    policy=model.stream_policy,
                    mode="eval",
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
                sink_index = text_lengths.clamp(max=combined.size(1) - 1)
                rows = torch.arange(combined.size(0), device=device)
                sink = combined[rows, sink_index]
                return logits, details, sink, encoder_lens, text_lengths

            if use_cuda and amp != "off":
                with torch.autocast("cuda", dtype=torch.float16):
                    logits, details, sink, encoder_lens, text_lengths = _forward()
            else:
                logits, details, sink, encoder_lens, text_lengths = _forward()

            logits_parts.append(logits.detach().float().cpu().numpy())
            sink_parts.append(sink.detach().float().cpu().numpy())
            audio_parts.append(encoder_lens.detach().cpu().numpy())
            text_parts.append(text_lengths.detach().cpu().numpy())
            label_parts.append(labels)
            sample_parts.append(sample_ids)
            pos = details.position_logits
            mask = details.position_mask
            if pos is None or mask is None:
                raise SystemExit("This probe requires a pooling readout with position logits")
            pos_parts.append(pos.detach().float().cpu().numpy().astype(np.float32))
            mask_parts.append(mask.detach().cpu().numpy().astype(bool))

            if step % 100 == 0:
                done = sum(p.shape[0] for p in logits_parts)
                rate = done / max(time.time() - started, 1e-9)
                print("step", step, "samples", done, "rate", round(rate, 1), "per s", flush=True)

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
    for p, m in zip(pos_parts, mask_parts):
        n = p.shape[0]
        pos_logits[cursor:cursor + n, :p.shape[1]] = p
        pos_mask[cursor:cursor + n, :p.shape[1]] = m
        cursor += n

    out_path = output_dir / output_name
    np.savez(
        out_path,
        logits=logits,
        sink=sink,
        pos_logits=pos_logits,
        pos_mask=pos_mask,
        audio_len=audio_len,
        text_len=text_len,
        label=labels,
        sample_id=sample_ids,
    )
    summary = {
        "checkpoint": str(checkpoint_path),
        "split": split,
        "num_samples": int(logits.shape[0]),
        "num_positive": int(labels.sum()),
        "max_text_width": int(max_width),
        "audio_len_mean": float(audio_len.mean()),
        "output": str(out_path),
    }
    (output_dir / "hard_probe_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
