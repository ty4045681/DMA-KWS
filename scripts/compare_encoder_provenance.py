#!/usr/bin/env python3
"""Element-wise encoder provenance check for the Hey Eva base-selection tables.

``docs/hey-eva-operating-points.md`` and the strategy doc separate two things
that are easy to confuse:

  * 基础 encoder: where the encoder came from before adaptation.
  * 增训时 encoder: whether adaptation updated it.

This script supplies the provenance half. It compares the encoder tensors of a
stage-II checkpoint against an upstream stage-I release and reports the largest
absolute element-wise difference. A maximum of 0 means the checkpoint inherited
that encoder untouched; anything larger means the encoder moved at some point in
the lineage.

Checkpoint layouts differ: our stage-II files store ``model_state_dict`` with an
``encoder.`` prefix, while the author releases store ``{"model": state_dict}``
with bare names. The script strips that prefix and drops the decoder, joiner and
ctc tensors, which the released files carry and ours do not.

Usage:
  .venv/bin/python scripts/compare_encoder_provenance.py
  .venv/bin/python scripts/compare_encoder_provenance.py --out docs/encoder-provenance.md
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

ROOT = Path("/home/ubuntu/dma-kws")
UPSTREAM_ZH_EN = ROOT / "data/dma-kws/raw/kws-checkpoints/zh-en-3M-2025-12-20/pretrained-epoch-13-avg-2.pt"
UPSTREAM_GS_BASE = ROOT / "data/dma-kws/raw/kws-checkpoints/gigaspeech-20240219/exp/pretrained.pt"

SS_ZH_EN = ROOT / "data/dma-kws/exp/stage2_qbyt/sinkhead_stage2/SS-zh-en/checkpoints/SS-zh-en/version_0/stage2_step003000.pt"
SS_GS_BASE_STREAM = ROOT / "data/dma-kws/exp/stage2_qbyt/sinkhead_stage2/SS-gsbase-stream/checkpoints/SS-gsbase-stream/version_0/stage2_step003000.pt"
SS_R1 = ROOT / "data/dma-kws/exp/stage2_qbyt/sinkhead_stage2/SS-R1/checkpoints/SS-R1/version_0/stage2_step003000.pt"
R1_BARE = ROOT / "data/dma-kws/exp/stage2_qbyt/final/R1-unfreeze-hardneg100/checkpoints/R1-unfreeze-hardneg100/version_0/stage2_step012500.pt"

# (label, stage-II side, upstream or earlier stage-II side)
PAIRS = (
    ("SS-zh-en ← 上游 zh-en-3M", SS_ZH_EN, UPSTREAM_ZH_EN),
    ("SS-gsbase-stream ← 上游 GS-base", SS_GS_BASE_STREAM, UPSTREAM_GS_BASE),
    ("SS-R1 ← 上游 zh-en-3M", SS_R1, UPSTREAM_ZH_EN),
    ("SS-R1 ← R1-bare", SS_R1, R1_BARE),
)


def encoder_tensors(path: Path) -> dict[str, torch.Tensor]:
    """Return the encoder tensors of either checkpoint layout, unprefixed."""
    blob = torch.load(path, map_location="cpu", weights_only=False)
    if "model_state_dict" in blob:
        state = blob["model_state_dict"]
        return {key[len("encoder.") :]: value for key, value in state.items() if key.startswith("encoder.")}
    if "model" in blob:
        # Drop the released model's decoder side: RNN-T decoder, joiner, ctc head
        # and the simple-am/simple-lm projections. None of them are encoder.
        return {
            key: value
            for key, value in blob["model"].items()
            if not key.startswith(("decoder.", "joiner.", "ctc_output.", "simple_am_proj.", "simple_lm_proj."))
        }
    raise SystemExit(f"unrecognised checkpoint layout: {path}")


def compare(left: Path, right: Path) -> tuple[int, int, int, float, str]:
    a, b = encoder_tensors(left), encoder_tensors(right)
    common = sorted(set(a) & set(b))
    if not common:
        raise SystemExit(f"no shared encoder tensors between {left} and {right}")
    worst, where = 0.0, ""
    for key in common:
        delta = (a[key].float() - b[key].float()).abs().max().item()
        if delta > worst:
            worst, where = delta, key
    return len(common), len(set(a) - set(b)), len(set(b) - set(a)), worst, where


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    lines = [
        "| 比对 | 公共张量 | 仅左侧 | 仅右侧 | 逐元素最大差 | 最大差位置 |",
        "|---|---|---|---|---|---|",
    ]
    for label, left, right in PAIRS:
        common, only_left, only_right, worst, where = compare(left, right)
        lines.append(f"| {label} | {common} | {only_left} | {only_right} | {worst:.6g} | `{where}` |")
    table = "\n".join(lines)
    print(table)
    if args.out:
        args.out.write_text(
            "# Encoder 来历比对\n\n"
            "由 `scripts/compare_encoder_provenance.py` 生成。最大差 0 表示 encoder 逐元素等同于右侧，\n"
            "即在本仓库里从未被训练过。\n\n" + table + "\n",
            encoding="utf-8",
        )
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
