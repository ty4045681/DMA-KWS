#!/usr/bin/env python3
"""Numerically compare this repo's v1 QbyT against the author's original code.

Loads the author's decode/models/encoder.py and decode/stage2/model.py from an
upstream checkout and this repo's imported stage2_v1_si.pt, runs both on
identical fbank features, and requires agreement of the utterance logit. This is
the fidelity gate for the v1 port.

Usage:
  python scripts/parity_check_author_v1.py \
      --upstream-root /path/to/DMA-KWS \
      --stage2-ckpt ckpts/stage2/155k-v2-ft.ckpt \
      --repo-ckpt data/dma-kws/exp/author_v1/stage2_v1_si.pt \
      --audio path/to/clip.wav \
      --keyword-phonemes "HH EY1 S N IH1 P S"
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

TOLERANCE = 1e-5


def _load_upstream_module(upstream_root: Path, relative: str, name: str):
    path = upstream_root / relative
    if not path.is_file():
        raise SystemExit(f"Missing upstream file: {path}")
    decode_root = str(upstream_root / "decode")
    if decode_root not in sys.path:
        sys.path.insert(0, decode_root)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--stage2-ckpt", type=Path, required=True)
    parser.add_argument("--repo-ckpt", type=Path, required=True)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--keyword-phonemes", required=True)
    parser.add_argument(
        "--dict", type=Path, default=PROJECT_ROOT / "data" / "dict" / "lang_char_v1_73.txt"
    )
    args = parser.parse_args(argv)

    import torch

    from dma_kws.audio import load_audio
    from dma_kws.inference.stage2_verifier import build_stage2_inference_model
    from dma_kws.stage2.fbank import FbankExtractor
    from dma_kws.tokenizer import load_char_tokenizer

    torch.manual_seed(0)

    upstream_encoder_mod = _load_upstream_module(
        args.upstream_root, "decode/models/encoder.py", "upstream_encoder"
    )
    upstream_qbyt_mod = _load_upstream_module(
        args.upstream_root, "decode/stage2/model.py", "upstream_qbyt"
    )

    encoder = upstream_encoder_mod.ConformerEncoder(
        input_size=80, output_size=144, attention_heads=4, linear_units=576,
        num_blocks=6, dropout_rate=0.1, positional_dropout_rate=0.1,
        attention_dropout_rate=0.0, use_cnn_module=True, input_layer="conv2d",
        pos_enc_layer_type="rel_pos", selfattention_layer_type="rel_selfattn",
        cnn_module_kernel=3,
    )
    qbyt = upstream_qbyt_mod.QbyT(
        encoder_output_size=144, num_embeds=73, embed_dim=128, post_num_layers=2
    )
    raw = torch.load(args.stage2_ckpt, map_location="cpu", weights_only=False)
    state = raw.get("state_dict", raw)
    encoder.load_state_dict(
        {k[len("encoder."):]: v for k, v in state.items() if k.startswith("encoder.")},
        strict=True,
    )
    qbyt.load_state_dict(
        {k[len("qbyt."):]: v for k, v in state.items() if k.startswith("qbyt.")},
        strict=True,
    )
    encoder.eval()
    qbyt.eval()

    tokenizer = load_char_tokenizer(args.dict)
    _, keyword_ids = tokenizer.tokenize(args.keyword_phonemes)
    waveform, sample_rate = load_audio(str(args.audio), sample_rate=16000)
    fbank = FbankExtractor(num_mel_bins=80, dither=0.0)
    feats = fbank.extract(waveform, sample_rate).unsqueeze(0)

    with torch.no_grad():
        encoder_out, _ = encoder(feats, torch.tensor([feats.size(1)]))
        reference_logit, _ = qbyt(encoder_out, torch.tensor([keyword_ids]))

    config = {
        "stage1": {
            "input_dim": 80, "encoder_output_dim": 144, "attention_heads": 4,
            "linear_units": 576, "num_blocks": 6, "cnn_module_kernel": 3,
        },
        "stage2": {
            "encoder_output_dim": 144, "qbyt_embed_dim": 128, "qbyt_layers": 2,
            "qbyt_readout_version": 1,
        },
    }
    model, _, _ = build_stage2_inference_model(
        stage1_cfg=config["stage1"], stage2_cfg=config["stage2"], vocab_size=73
    )
    payload = torch.load(args.repo_ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()
    with torch.no_grad():
        repo_logit = model(
            feats,
            torch.tensor([feats.size(1)]),
            torch.tensor([keyword_ids]),
            torch.tensor([len(keyword_ids)]),
        )

    reference = float(reference_logit.reshape(-1)[0])
    repo = float(repo_logit.reshape(-1)[0])
    delta = abs(reference - repo)
    print(f"upstream logit = {reference:+.8f}")
    print(f"repo     logit = {repo:+.8f}")
    print(f"|delta|        = {delta:.3e} (tolerance {TOLERANCE:g})")
    if delta > TOLERANCE:
        raise SystemExit("v1 QbyT parity check FAILED")
    print("v1 QbyT parity check OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
