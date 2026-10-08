#!/usr/bin/env python3
"""Live progress plot for the A0-A5 QbyT v4.1 sink-head warm-start ablations.

Reads the Lightning metrics.csv of every ablation group, derives the wall-clock
step rate from the TensorBoard event file, and writes one dashboard PNG plus a
JSON summary. Read-only: it never touches the running jobs.

Usage:
    .venv/bin/python scripts/plot_ablation_progress.py
"""

from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
ABL = ROOT / "data/dma-kws/exp/stage2_qbyt/ablations"

GROUPS: list[tuple[str, str]] = [
    ("A0-control", "A0 control (v4.1 sink, as-is)"),
    ("A1-sink-additive", "A1 sink readout = additive"),
    ("A2-sink-mixture", "A2 sink readout = mixture"),
    ("A3-temp-fixed-0p1", "A3 temperature = 0.1 (fixed)"),
    ("A4-temp-learnable", "A4 temperature = 0.1 (learnable)"),
    ("A5-sink-identity", "A5 sink identity"),
]

MAX_STEPS = 2000
CKPT_EVERY = 1000

PANELS: list[tuple[str, str, bool]] = [
    ("train/microbatch/loss_total", "total loss", False),
    ("train/microbatch/loss_utt_raw", "utterance loss (raw)", False),
    ("train/optimizer/grad_norm_pre_clip", "grad norm (pre-clip)", True),
    ("train/background/musan/utt_bce", "MUSAN background utt BCE", False),
    ("train/microbatch/loss_seq_progress_raw", "sequence progress loss (raw)", False),
    ("train/microbatch/loss_negative_tail_raw", "negative-tail loss (raw)", False),
]


def metrics_path(group: str) -> Path | None:
    hits = sorted(glob.glob(str(ABL / group / "logs" / "*" / "version_*" / "metrics.csv")))
    return Path(hits[-1]) if hits else None


def progress(group: str, frame: pd.DataFrame) -> dict:
    step = int(frame["step"].dropna().max())
    events = sorted(glob.glob(str(ABL / group / "logs" / "*" / "version_*" / "events.out.tfevents*")))
    rate = float("nan")
    last_wall = float("nan")
    if events:
        try:
            from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

            acc = EventAccumulator(events[-1], size_guidance={"scalars": 0})
            acc.Reload()
            tag = "train/microbatch/loss_total"
            if tag not in acc.Tags()["scalars"]:
                tag = acc.Tags()["scalars"][0]
            scalars = acc.Scalars(tag)
            span = scalars[-1].wall_time - scalars[0].wall_time
            if span > 0 and scalars[-1].step > scalars[0].step:
                rate = (scalars[-1].step - scalars[0].step) / span
            last_wall = scalars[-1].wall_time
        except Exception:
            pass
    if rate != rate or last_wall != last_wall:
        mt = os.path.getmtime(metrics_path(group))
        last_wall = mt
        step_span = max(step - 9, 1)
        rate = step_span / 2400.0
    remaining = max(MAX_STEPS - step, 0)
    eta = last_wall + remaining / rate
    return {
        "step": step,
        "max_steps": MAX_STEPS,
        "percent": round(100.0 * step / MAX_STEPS, 1),
        "step_per_s": round(rate, 4),
        "eta": dt.datetime.fromtimestamp(eta).strftime("%H:%M") if eta == eta else "n/a",
        "eta_iso": dt.datetime.fromtimestamp(eta).isoformat(timespec="seconds") if eta == eta else None,
        "last_scalar_at": dt.datetime.fromtimestamp(last_wall).strftime("%H:%M:%S") if last_wall == last_wall else "n/a",
    }


def smooth(series: pd.Series, alpha: float = 0.12) -> pd.Series:
    return series.ewm(alpha=alpha, adjust=False).mean()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "outputs/v41_ablations")
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()

    frames: dict[str, pd.DataFrame] = {}
    stats: dict[str, dict] = {}
    for group, _ in GROUPS:
        path = metrics_path(group)
        if path is None:
            print("missing metrics:", group)
            continue
        frame = pd.read_csv(path)
        frames[group] = frame
        stats[group] = progress(group, frame)

    now = dt.datetime.now()
    fig = plt.figure(figsize=(17.5, 11.5))
    grid = fig.add_gridspec(3, 3, height_ratios=[1.15, 1.0, 1.0], hspace=0.45, wspace=0.24)
    colors = dict(zip([g for g, _ in GROUPS], plt.get_cmap("tab10").colors))

    # progress bars
    ax = fig.add_subplot(grid[0, :])
    ys = list(range(len(GROUPS)))[::-1]
    for y, (group, label) in zip(ys, GROUPS):
        info = stats.get(group)
        if not info:
            continue
        done = info["step"]
        ax.barh(y, done, height=0.55, color=colors[group], alpha=0.9, zorder=3)
        ax.barh(y, MAX_STEPS - done, left=done, height=0.55, color="0.86", zorder=3)
        ax.text(done / 2, y, str(done), ha="center", va="center", fontsize=9,
                color="white" if done > 400 else "black", fontweight="bold", zorder=4)
        ax.text(done + 50, y,
                "%.1f%%   ETA %s   (%.3f step/s)" % (info["percent"], info["eta"], info["step_per_s"]),
                va="center", fontsize=9.5, color="0.25", zorder=4)
    ax.axvline(CKPT_EVERY, color="0.45", ls="--", lw=1.2, zorder=2)
    ax.axvline(MAX_STEPS, color="0.2", ls="--", lw=1.4, zorder=2)
    ax.text(CKPT_EVERY, len(GROUPS) - 0.45, "ckpt @1000", ha="center", fontsize=8.5, color="0.35")
    ax.text(MAX_STEPS - 15, len(GROUPS) - 0.45, "val + export @2000", ha="right", fontsize=8.5, color="0.2")
    ax.set_yticks(ys)
    ax.set_yticklabels([label for _, label in GROUPS], fontsize=10.5)
    ax.set_xlim(0, MAX_STEPS * 1.02)
    ax.set_ylim(-0.7, len(GROUPS) - 0.05)
    ax.set_xlabel("optimizer step (all six runs share one V100)", fontsize=10)
    ax.set_title("QbyT v4.1 sink-head ablations A0-A5 - live progress   (as of %s +0800)" % now.strftime("%Y-%m-%d %H:%M"),
                 fontsize=13.5, fontweight="bold")
    ax.grid(axis="x", alpha=0.3, ls=":")
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    # metric panels
    for index, (key, title, logy) in enumerate(PANELS):
        ax = fig.add_subplot(grid[1 + index // 3, index % 3])
        for group, label in GROUPS:
            frame = frames.get(group)
            if frame is None or key not in frame.columns:
                continue
            sub = frame[["step", key]].dropna()
            if sub.empty:
                continue
            ax.plot(sub["step"], smooth(sub[key]), color=colors[group], lw=1.8,
                    label=label if index == 0 else None)
        if logy:
            ax.set_yscale("log")
        ax.set_title(title, fontsize=11.5)
        ax.set_xlabel("step", fontsize=9)
        ax.set_xlim(0, MAX_STEPS)
        ax.grid(alpha=0.3, ls=":")
        ax.tick_params(labelsize=9)
        ax.set_axisbelow(True)

    handles, legend_labels = fig.axes[1].get_legend_handles_labels()
    if handles:
        fig.legend(handles, legend_labels, loc="upper center", bbox_to_anchor=(0.5, 0.028),
                   ncol=6, fontsize=10, frameon=False)
    fig.text(0.5, 0.052,
             "Top: step progress on the shared V100 (val AUC/EER appear only at step 2000). "
             "Below: train-side metrics, EMA alpha=0.12.",
             ha="center", fontsize=9.5, color="0.35")

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / "training_progress.png"
    fig.savefig(png, dpi=args.dpi, facecolor="white", bbox_inches="tight")
    print("saved:", png, os.path.getsize(png), "bytes")

    payload = {"generated_at": now.isoformat(timespec="seconds"), "runs": stats}
    (out_dir / "training_progress.json").write_text(json.dumps(payload, indent=1) + chr(10))
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
