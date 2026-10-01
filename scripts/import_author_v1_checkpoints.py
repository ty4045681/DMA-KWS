#!/usr/bin/env python3
"""Import the paper-original (v1) SI checkpoints into this repository's format.

Why this exists
---------------
The author's releases are a raw Wenet ASR state dict (ls-gs-1460.pt) and an
unversioned PyTorch-Lightning Stage II checkpoint (155k-v2-ft.ckpt). Every
Stage II loader here rejects unversioned QbyT weights on purpose, because v1's
deployed score (GRU state at the last position of the padded [text][audio]
sequence) is not the same function as any later readout. This script is the
single sanctioned bridge:

* it proves the v1 architecture with a strict state-dict match,
* strips the ASR decoder (unused by the phoneme-CTC locator),
* stamps qbyt_readout_version=1 and embeds the resolved config,
* writes .pt payloads that load with strict=True.

The multimodal enrolment (SD) releases are deliberately refused.

Usage:
  python scripts/import_author_v1_checkpoints.py \
      --stage1 ckpts/stage1/ls-gs-1460.pt \
      --stage2 ckpts/stage2/155k-v2-ft.ckpt \
      --output-dir data/dma-kws/exp/author_v1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dma_kws.pathing import ensure_qbyt_on_path
from dma_kws.stage2.readout import QBYT_V1_READOUT, QbyTV1Spec

VOCAB_SIZE = 73
DEFAULT_DICT_PATH = PROJECT_ROOT / "data" / "dict" / "lang_char_v1_73.txt"

#: Encoder kwargs, verbatim from the author's train/two_stage/model.py.
STAGE1_ARCH: dict[str, Any] = {
    "input_size": 80,
    "output_size": 144,
    "attention_heads": 4,
    "linear_units": 576,
    "num_blocks": 6,
    "dropout_rate": 0.1,
    "positional_dropout_rate": 0.1,
    "attention_dropout_rate": 0.0,
    "use_cnn_module": True,
    "input_layer": "conv2d",
    "pos_enc_layer_type": "rel_pos",
    "selfattention_layer_type": "rel_selfattn",
    "cnn_module_kernel": 3,
}

STAGE2_ARCH: dict[str, Any] = {
    "encoder_output_dim": 144,
    "qbyt_embed_dim": 128,
    "qbyt_layers": 2,
    "qbyt_readout_version": 1,
}


def _torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - environment guard
        raise SystemExit(
            "Missing torch; install the training/inference extras first."
        ) from exc
    return torch


def _state_dict_of(payload: Any, source: Path) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise SystemExit(f"{source} did not load as a mapping")
    for key in ("state_dict", "model_state_dict"):
        value = payload.get(key)
        if isinstance(value, Mapping):
            return dict(value)
    return dict(payload)


def _split(state: Mapping[str, Any], prefix: str) -> dict[str, Any]:
    head = prefix + "."
    return {key[len(head):]: value for key, value in state.items() if key.startswith(head)}


def _assert_speaker_independent(state: Mapping[str, Any], source: Path) -> None:
    enrolment = sorted(
        key
        for key in state
        if key.endswith("modality_enc.enr_audio_emb") or "cross_attn." in key
    )
    if enrolment:
        shown = ", ".join(enrolment[:3]) + (" ..." if len(enrolment) > 3 else "")
        raise SystemExit(
            f"{source} carries the multimodal enrolment branch ({shown}). "
            "Only the speaker-independent v1 QbyT is supported; the SD "
            "(155k-mm-f1 / 155k-mm-f2) releases are out of scope."
        )


def _build_encoder():
    ensure_qbyt_on_path()
    from models.encoder import ConformerEncoder

    return ConformerEncoder(**STAGE1_ARCH)


def _build_qbyt():
    from dma_kws.stage2.model_factory import build_qbyt

    return build_qbyt(STAGE2_ARCH, input_dim=144, vocab_size=VOCAB_SIZE)


def _build_ctc():
    ensure_qbyt_on_path()
    from models.ctc import CTC

    return CTC(VOCAB_SIZE, 144, blank_id=0)


def load_stage1_state(path: Path) -> dict[str, Any]:
    """Return encoder.*/ctc.* tensors from the raw Wenet ASR export."""
    torch = _torch()
    raw = _state_dict_of(torch.load(path, map_location="cpu", weights_only=False), path)
    state = {
        **{"encoder." + key: value for key, value in _split(raw, "encoder").items()},
        **{"ctc." + key: value for key, value in _split(raw, "ctc").items()},
    }
    if not state:
        raise SystemExit(f"{path} has no encoder.*/ctc.* weights")
    ctc_lo = state.get("ctc.ctc_lo.weight")
    if ctc_lo is None or tuple(ctc_lo.shape) != (VOCAB_SIZE, 144):
        shape = None if ctc_lo is None else tuple(ctc_lo.shape)
        raise SystemExit(
            f"{path} CTC head is {shape}, expected ({VOCAB_SIZE}, 144); "
            "this is not the 73-symbol phoneme ASR release"
        )
    decoder_keys = [key for key in raw if key.startswith("decoder.")]
    container = torch.nn.Module()
    container.encoder = _build_encoder()
    container.ctc = _build_ctc()
    container.load_state_dict(state, strict=True)
    print(
        f"[stage1] {path}: strict match on encoder+ctc "
        f"({len(_split(state, 'encoder'))} + {len(_split(state, 'ctc'))} tensors); "
        f"dropped {len(decoder_keys)} decoder.* ASR tensors"
    )
    return state


def load_stage2_state(path: Path) -> dict[str, Any]:
    """Return the strictly validated encoder.*/qbyt.* v1 SI state."""
    torch = _torch()
    raw = torch.load(path, map_location="cpu", weights_only=False)
    state = _state_dict_of(raw, path)
    _assert_speaker_independent(state, path)
    qbyt_state = _split(state, "qbyt")
    encoder_state = _split(state, "encoder")
    if not qbyt_state or not encoder_state:
        raise SystemExit(
            f"{path} is not a complete Stage II v1 checkpoint "
            f"(encoder.*={len(encoder_state)} qbyt.*={len(qbyt_state)})"
        )
    text_projection = qbyt_state.get("text_projection.weight")
    if text_projection is None or int(text_projection.shape[0]) != VOCAB_SIZE:
        shape = None if text_projection is None else tuple(text_projection.shape)
        raise SystemExit(
            f"{path} text_projection is {shape}, expected ({VOCAB_SIZE}, 128); "
            "the v1 release uses the 73-symbol dictionary"
        )
    container = torch.nn.Module()
    container.encoder = _build_encoder()
    container.qbyt = _build_qbyt()
    container.load_state_dict(state, strict=True)
    print(
        f"[stage2] {path}: strict match on encoder+qbyt "
        f"({len(encoder_state)} + {len(qbyt_state)} tensors); "
        f"readout={QBYT_V1_READOUT}"
    )
    return state


def _stage1_payload(state: dict[str, Any], dict_path: Path) -> dict[str, Any]:
    return {
        "model_state_dict": state,
        "config": {
            "stage1": {**STAGE1_ARCH, "encoder_type": "conformer"},
            "tokenizer": {"dict_path": str(dict_path), "split_with_space": " "},
        },
        "step": 0,
        "tokenizer_dict_path": str(dict_path),
        "vocab_size": VOCAB_SIZE,
        "blank_id": 0,
    }


def _stage2_payload(state: dict[str, Any], dict_path: Path, *, step: int) -> dict[str, Any]:
    return {
        "model_state_dict": state,
        "config": {
            "stage1": {**STAGE1_ARCH, "encoder_type": "conformer"},
            "stage2": dict(STAGE2_ARCH),
            "tokenizer": {"dict_path": str(dict_path), "split_with_space": " "},
        },
        "step": int(step),
        "tokenizer_dict_path": str(dict_path),
        "vocab_size": VOCAB_SIZE,
        "blank_id": 0,
        "qbyt_readout_version": 1,
        "qbyt_v1_spec": QbyTV1Spec().as_dict(),
        "checkpoint_kind": "stage2",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage1", type=Path, required=True, help="raw ls-gs-1460.pt")
    parser.add_argument(
        "--stage2", type=Path, required=True, help="raw 155k-v2-ft.ckpt (SI only)"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dict", type=Path, default=DEFAULT_DICT_PATH)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    torch = _torch()
    if not args.dict.is_file():
        raise SystemExit(f"Dictionary not found: {args.dict}")

    stage1_state = load_stage1_state(args.stage1)
    raw_stage2 = torch.load(args.stage2, map_location="cpu", weights_only=False)
    stage2_state = load_stage2_state(args.stage2)
    step = int(raw_stage2.get("global_step", 0)) if isinstance(raw_stage2, Mapping) else 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    payloads = {
        "stage1_v1.pt": _stage1_payload(stage1_state, args.dict),
        "stage2_v1_si.pt": _stage2_payload(stage2_state, args.dict, step=step),
    }
    for name, payload in payloads.items():
        target = args.output_dir / name
        if target.exists() and not args.overwrite:
            raise SystemExit(f"{target} exists; pass --overwrite to replace it")
        torch.save(payload, target)
        print(f"[write] {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
