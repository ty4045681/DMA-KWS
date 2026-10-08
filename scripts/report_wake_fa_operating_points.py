#!/usr/bin/env python3
"""Operating points calibrated on generic background only.

Protocol this script implements (the deployment-facing one):

  * The threshold is chosen from the **generic background** false-alarm budget
    only: the held-out MUSAN music/noise grid plus the LibriSpeech speech
    subset, pooled by their measured hours.  A "target FA/h" therefore means
    "false accepts per hour of ordinary background", counting both corpora.
  * The wake rate is then read off the held-out real speakers (and, for the TTS
    arms, the held-out voices).
  * Near-miss phrases ("Hey Ava", "Hi Eva", ...) are **reported, never used for
    calibration**.  They are a product-specific negative class with no public
    corpus, so folding them into the threshold would silently trade wake rate
    for a number nobody can compare against.

This is deliberately different from calibrating on every negative available:
for a model whose near-miss scores overlap its positives (see the R1 arms),
that choice moves the reported wake rate by tens of points.

Usage:
  .venv/bin/python scripts/report_wake_fa_operating_points.py --out docs/hey-eva-operating-points.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path("/home/ubuntu/dma-kws")
EVAL = ROOT / "outputs/hey_eva_v42_eval"
TARGETS = (0.05, 0.1, 0.5, 1.0)

# LibriPhrase hard AUC is the forgetting metric. Adapted arms read it from their
# own run (``val/lph_auc``); the two un-adapted bases were measured by their
# producing runs, whose validation column is named ``val/auc``.
LPH_RUNS = {
    "增训前 SS-zh-en": ("outputs/hey_eva_v42_base/real/logs/*/version_*/metrics.csv", "val/lph_auc"),
    "增训前 SS-R1": ("data/dma-kws/exp/stage2_qbyt/sinkhead_stage2/SS-R1/logs/*/version_*/metrics.csv", "val/auc"),
    "C1 + LoRA（现役）": ("data/dma-kws/exp/stage2_adapt_v42/hey_eva_joint/joint/logs/*/version_*/metrics.csv", "val/lph_auc"),
    "R1 + LoRA": ("data/dma-kws/exp/stage2_adapt_v42/hey_eva_joint_r1/joint/logs/*/version_*/metrics.csv", "val/lph_auc"),
    "R1 + encoder lr5e-5": ("data/dma-kws/exp/stage2_adapt_v42/hey_eva_joint_r1_enc/joint/logs/*/version_*/metrics.csv", "val/lph_auc"),
    "R1 + encoder lr2e-5 +CVaR": ("data/dma-kws/exp/stage2_adapt_v42/r1enc_tail/joint/logs/*/version_*/metrics.csv", "val/lph_auc"),
    "R1 + encoder lr1e-5": ("data/dma-kws/exp/stage2_adapt_v42/r1enc_gentle/joint/logs/*/version_*/metrics.csv", "val/lph_auc"),
    "R1 + QbyT 全参": ("data/dma-kws/exp/stage2_adapt_v42/r1_qbytfull/joint/logs/*/version_*/metrics.csv", "val/lph_auc"),
    "C1 + QbyT 全参": ("data/dma-kws/exp/stage2_adapt_v42/c1_qbytfull/joint/logs/*/version_*/metrics.csv", "val/lph_auc"),
    "C1 + encoder 与 QbyT 全参": ("data/dma-kws/exp/stage2_adapt_v42/c1_encfull/joint/logs/*/version_*/metrics.csv", "val/lph_auc"),
    "GS-base 流式 + QbyT 全参": ("data/dma-kws/exp/stage2_adapt_v42/gsstream_qbytfull/joint/logs/*/version_*/metrics.csv", "val/lph_auc"),
    "GS-base 流式 + encoder 与 QbyT 全参": ("data/dma-kws/exp/stage2_adapt_v42/gsstream_encfull/joint/logs/*/version_*/metrics.csv", "val/lph_auc"),
}

# name -> (real clips dir, tts clips dir, MUSAN dir, LibriSpeech dir)
ARMS = {
    "增训前 SS-zh-en": (
        "real_base", "tts_base",
        ROOT / "data/dma-kws/exp/stage2_qbyt/fa/v42-zhen-musan",
        ROOT / "data/dma-kws/exp/stage2_qbyt/fa/v42-zhen-ls",
    ),
    "增训前 SS-R1": ("real_r1_base", "tts_r1_base", EVAL / "fa_r1_base-musan", EVAL / "fa_r1_base-ls"),
    "C1 + LoRA（现役）": ("real_joint", "tts_joint", EVAL / "fa_joint-musan", EVAL / "fa_joint-ls"),
    "R1 + LoRA": ("real_r1_adapted", "tts_r1_adapted", EVAL / "fa_r1_adapted-musan", EVAL / "fa_r1_adapted-ls"),
    "R1 + encoder lr5e-5": ("real_r1enc", "tts_r1enc", EVAL / "fa_r1enc-musan", EVAL / "fa_r1enc-ls"),
    "R1 + encoder lr2e-5 +CVaR": ("real_r1enctail", "tts_r1enctail", EVAL / "fa_r1enctail-musan", EVAL / "fa_r1enctail-ls"),
    "R1 + encoder lr1e-5": ("real_r1encgentle", "tts_r1encgentle", EVAL / "fa_r1encgentle-musan", EVAL / "fa_r1encgentle-ls"),
    "R1 + QbyT 全参": ("real_r1qbytfull", "tts_r1qbytfull", EVAL / "fa_r1qbytfull-musan", EVAL / "fa_r1qbytfull-ls"),
    "C1 + QbyT 全参": ("real_c1qbytfull", "tts_c1qbytfull", EVAL / "fa_c1qbytfull-musan", EVAL / "fa_c1qbytfull-ls"),
    "C1 + encoder 与 QbyT 全参": ("real_c1encfull", "tts_c1encfull", EVAL / "fa_c1encfull-musan", EVAL / "fa_c1encfull-ls"),
    "GS-base 流式 + QbyT 全参": ("real_gsstream_qbytfull", "tts_gsstream_qbytfull", EVAL / "fa_gsstream_qbytfull-musan", EVAL / "fa_gsstream_qbytfull-ls"),
    "GS-base 流式 + encoder 与 QbyT 全参": ("real_gsstream_encfull", "tts_gsstream_encfull", EVAL / "fa_gsstream_encfull-musan", EVAL / "fa_gsstream_encfull-ls"),
}


# (基础 encoder 来历, 增训时 encoder 是否可训)
# 来历用逐元素比对确认过：SS-zh-en 与上游 zh-en-3M 最大差 0，SS-gsbase-stream 与上游
# GS-base 最大差 0，R1-bare 与上游 zh-en-3M 最大差 1.0997，SS-R1 与 R1-bare 最大差 0。
ENCODER = {
    "增训前 SS-zh-en": ("zh-en-3M 原始", "未增训"),
    "增训前 SS-R1": ("zh-en-3M + R1 微调 13.3k 步", "未增训"),
    "C1 + LoRA（现役）": ("zh-en-3M 原始", "冻结"),
    "C1 + QbyT 全参": ("zh-en-3M 原始", "冻结"),
    "C1 + encoder 与 QbyT 全参": ("zh-en-3M 原始", "可训"),
    "R1 + LoRA": ("zh-en-3M + R1 微调", "冻结"),
    "R1 + QbyT 全参": ("zh-en-3M + R1 微调", "冻结"),
    "R1 + encoder lr1e-5": ("zh-en-3M + R1 微调", "可训"),
    "R1 + encoder lr2e-5 +CVaR": ("zh-en-3M + R1 微调", "可训"),
    "R1 + encoder lr5e-5": ("zh-en-3M + R1 微调", "可训"),
    "GS-base 流式 + QbyT 全参": ("GS-base 原始", "冻结"),
    "GS-base 流式 + encoder 与 QbyT 全参": ("GS-base 原始", "可训"),
}

ORDER = [
    "增训前 SS-zh-en", "增训前 SS-R1",
    "C1 + LoRA（现役）", "C1 + QbyT 全参", "C1 + encoder 与 QbyT 全参",
    "R1 + LoRA", "R1 + QbyT 全参",
    "R1 + encoder lr1e-5", "R1 + encoder lr2e-5 +CVaR", "R1 + encoder lr5e-5",
    "GS-base 流式 + QbyT 全参", "GS-base 流式 + encoder 与 QbyT 全参",
]


def retention(name: str) -> float | None:
    import glob
    import pandas as pd

    pattern, column = LPH_RUNS[name]
    paths = glob.glob(pattern)
    if not paths:
        return None
    frame = pd.read_csv(paths[0])
    if column not in frame.columns:
        return None
    values = frame[column].dropna()
    return float(values.iloc[-1]) if len(values) else None


def scores(path: Path) -> tuple[np.ndarray, np.ndarray]:
    values, labels = [], []
    with (path / "results.jsonl").open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("skipped"):
                continue
            values.append(float(row["qbyt_score"]))
            labels.append(int(row["label"]))
    return np.array(values), np.array(labels)


def hours(path: Path) -> float:
    return float(json.loads((path / "summary.json").read_text())["total_hours"])


def rate_at_threshold(scores_: np.ndarray, threshold: float) -> float:
    return float((scores_ > threshold).mean())


def accepts_per_hour(scores_: np.ndarray, threshold: float, corpus_hours: float) -> float:
    """False accepts per hour, derived from actual counts.

    ``rate * (window_count / hours)`` would be the same number; dividing the
    per-window rate by the corpus hours double-counts them and understates the
    rate by that factor.
    """
    return float((scores_ > threshold).sum()) / corpus_hours


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    rows: list[str] = []
    detail: dict = {}
    ordered = [name for name in ORDER if name in ARMS]
    pending: list[str] = []
    for name in ordered:
        real_dir, tts_dir, musan_dir, ls_dir = ARMS[name]
        needed = [EVAL / real_dir, EVAL / tts_dir, musan_dir, ls_dir]
        lph_value = retention(name)
        # A grid that is still running already has results.jsonl but no
        # summary.json yet; treat it as pending rather than crashing on the
        # missing hours.
        incomplete = any(
            not (path / "results.jsonl").is_file() or not (path / "summary.json").is_file()
            for path in needed
        )
        if incomplete:
            # 评测还没跑完的臂照样占一行，未测的格子标"测试中"，
            # 这样表格的行数和顺序在补齐过程中保持不变。
            pending.append(name)
            for target in TARGETS:
                rows.append(
                    "| {} | {} | {} | {} | 测试中 | 测试中 | 测试中 | 测试中 | 测试中 | 测试中 | {} |".format(
                        name, *ENCODER.get(name, ("?", "?")), target,
                        f"{lph_value:.4f}" if lph_value is not None else "测试中",
                    )
                )
            continue
        lph = lph_value
        real_s, real_l = scores(EVAL / real_dir)
        pos, neg = real_s[real_l == 1], real_s[real_l == 0]
        tts_s, tts_l = scores(EVAL / tts_dir)
        tts_pos = tts_s[tts_l == 1]
        musan_s, _ = scores(musan_dir)
        ls_s, _ = scores(ls_dir)
        musan_h, ls_h = hours(musan_dir), hours(ls_dir)

        # Pooled generic background: every window weighted by the hours it
        # represents, so MUSAN and LibriSpeech contribute proportionally.
        pool = np.concatenate([musan_s, ls_s])
        pool_hours = musan_h + ls_h
        ranked = np.sort(pool)[::-1]

        detail[name] = {"musan_hours": musan_h, "ls_hours": ls_h, "lph_auc": lph}
        for target in TARGETS:
            allowed = int(np.floor(target * pool_hours))
            threshold = float(ranked[allowed]) if allowed < ranked.size else 0.0
            fa_musan = accepts_per_hour(musan_s, threshold, musan_h)
            fa_ls = accepts_per_hour(ls_s, threshold, ls_h)
            rows.append(
                "| {} | {} | {} | {} | {:.4f} | **{:.4f}** | {:.4f} | {:.3f} | {:.3f} | {:.4f} | {} |".format(
                    name, *ENCODER.get(name, ("?", "?")), target, threshold,
                    rate_at_threshold(pos, threshold),
                    rate_at_threshold(tts_pos, threshold),
                    fa_musan, fa_ls,
                    rate_at_threshold(neg, threshold),
                    f"{lph:.4f}" if lph is not None else "n/a",
                )
            )
            detail[name][target] = {
                "threshold": threshold,
                "wake_real": rate_at_threshold(pos, threshold),
                "wake_tts": rate_at_threshold(tts_pos, threshold),
                "fa_musan_per_hour": fa_musan,
                "fa_ls_per_hour": fa_ls,
                "near_miss_trigger": rate_at_threshold(neg, threshold),
                "lph_auc": lph,
            }

    if pending:
        print("pending evaluation (marked 测试中): " + ", ".join(pending), file=sys.stderr)

    header = (
        "| 模型 | 基础 encoder | 增训时 encoder | 目标 FA/h | 阈值 | 真人唤醒率 | TTS 唤醒率 | MUSAN FA/h | LibriSpeech FA/h | 近音词误触发（仅报告） | 通用能力 LibriPhrase AUC |\n"
        "|---|---|---|---|---|---|---|---|---|---|---|"
    )
    text = header + "\n" + "\n".join(rows)
    print(text)
    if args.out:
        preamble = (
            "# Hey Eva 工作点对照表（标准口径）\n\n"
            "由 `scripts/report_wake_fa_operating_points.py` 生成，勿手改。\n\n"
            "读法：阈值只由通用背景定（留出的 MUSAN 3 s 音乐噪声网格 "
            f"{detail[list(ARMS)[0]]['musan_hours']:.1f} 小时 + LibriSpeech other-500 子集 "
            f"{detail[list(ARMS)[0]]['ls_hours']:.1f} 小时，按小时数合成）。"
            "唤醒率读真人留出说话人与 TTS 留出音色。近音词只报告，不参与定阈值。"
            "通用能力是 LibriPhrase hard AUC，越接近 0.9411（增训前）越说明没忘旧任务。\n\n"
            "完整策略见 `docs/hey-eva-adaptation-strategy.md` §6.12。\n\n"
        )
        args.out.write_text(preamble + text + "\n", encoding="utf-8")
        args.out.with_suffix(".json").write_text(json.dumps(detail, indent=1, ensure_ascii=False))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
