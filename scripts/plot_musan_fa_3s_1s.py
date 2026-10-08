#!/usr/bin/env python3
"""Compare MUSAN false alarms on the 3 s (official) and 1 s (stress) grids.

Reads the ``eval_musan_fa.py`` artefacts under
``data/dma-kws/exp/stage2_qbyt/fa/`` -- ``<tag>-musan/`` (3 s / 3 s) and
``<tag>-musan-1s0/`` (1 s / 1 s) -- and writes one figure plus one CSV:

    outputs/v41_final/figures/musan_fa_3s_vs_1s.png
    outputs/v41_final/figures/musan_fa_3s_vs_1s.csv

The script never reruns inference. It reads ``summary.json`` for the
threshold-0.5 counts and ``fa_per_hour_curve.csv`` for the highest score seen.

Example:
    .venv/bin/python scripts/plot_musan_fa_3s_1s.py
    .venv/bin/python scripts/plot_musan_fa_3s_1s.py --output-prefix /tmp/fa
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path("/home/ubuntu/dma-kws")
FA_ROOT = ROOT / "data/dma-kws/exp/stage2_qbyt/fa"

# label, family, tag (directory prefix); only models with a 3 s run are listed.
MODELS: list[tuple[str, str, str]] = [
    ("v3 (zh-en-3M)", "zh-en readout generations", "v3"),
    ("v4 (zh-en-3M)", "zh-en readout generations", "v4"),
    ("v4.1 (zh-en-3M)", "zh-en readout generations", "v41"),
    ("v4.2 zh-en (sinkfit)", "zh-en readout generations", "v42-zhen"),
    ("v4.2 GS-finetune", "v4.2 other encoders", "v42-gs"),
    ("v4.2 GS-base stream", "v4.2 other encoders", "v42-gsbase-stream"),
    ("v4.2 GS-base fullctx", "v4.2 other encoders", "v42-gsbase-full"),
    ("v4.2 paperstage1", "v4.2 other encoders", "v42-paper"),
    ("author v1 (paper release)", "author baseline (different pipeline)", "author-v1"),
]

GRID_3S = "3s"
GRID_1S = "1s"


def run_dirs(tag: str, grid: str) -> list[Path]:
    """Candidate directories for one (tag, grid) pair, most specific first."""

    suffix = "-musan" if grid == GRID_3S else "-musan-1s0"
    base = FA_ROOT / (tag + suffix)
    return [base / "merged", base]


def max_score(directory: Path) -> float | None:
    """Highest background score: last threshold in the sweep with an accept."""

    curve = directory / "fa_per_hour_curve.csv"
    if not curve.is_file():
        return None
    best = None
    with curve.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            try:
                accepts = int(float(row["false_accepts"]))
                threshold = float(row["threshold"])
            except (KeyError, TypeError, ValueError):
                continue
            if accepts > 0 and (best is None or threshold > best):
                best = threshold
    return 0.0 if best is None else best


def load_run(tag: str, grid: str) -> dict | None:
    """Return the metrics of one run, or None when it has not been scored."""

    for directory in run_dirs(tag, grid):
        summary_path = directory / "summary.json"
        if not summary_path.is_file():
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        metrics = summary.get("metrics") or {}
        hours = float(metrics.get("total_hours") or summary.get("total_hours") or 0.0)
        subsets = summary.get("subsets") or {}
        subset_accepts = {
            name: int((subsets.get(name) or {}).get("metrics", {}).get("fp") or 0)
            for name in ("music", "noise", "speech")
        }
        return {
            "tag": tag,
            "grid": grid,
            "window_sec": float(summary.get("window_sec") or (3.0 if grid == GRID_3S else 1.0)),
            "hours": hours,
            "windows": int(metrics.get("num_samples") or 0),
            "threshold": float(metrics.get("threshold") or 0.5),
            "accepts": int(metrics.get("fp") or 0),
            "fa_per_hour": float(metrics.get("fa_per_hour") or 0.0),
            "fa_per_24h": float(metrics.get("fa_per_hour") or 0.0) * 24.0,
            "max_score": max_score(directory),
            "subset_accepts": subset_accepts,
            "checkpoint": summary.get("stage2_ckpt") or "",
            "readout_version": (summary.get("provenance") or {}).get("qbyt_readout_version"),
        }
    return None


def collect() -> list[dict]:
    rows: list[dict] = []
    for label, family, tag in MODELS:
        for grid in (GRID_3S, GRID_1S):
            run = load_run(tag, grid)
            if run is None:
                rows.append({"label": label, "family": family, "grid": grid, "status": "pending"})
            else:
                run.update({"label": label, "family": family, "status": "done"})
                rows.append(run)
    return rows


def write_csv(rows: list[dict], path: Path) -> None:
    fields = [
        "label", "family", "grid", "status", "window_sec", "threshold", "windows", "hours",
        "accepts", "fa_per_hour", "fa_per_24h", "max_score",
        "music_accepts", "noise_accepts", "speech_accepts", "checkpoint", "readout_version",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            if row.get("status") != "done":
                blank = {key: "" for key in fields}
                blank.update({"label": row["label"], "family": row["family"],
                              "grid": row["grid"], "status": "pending"})
                writer.writerow(blank)
                continue
            subset = row["subset_accepts"]
            writer.writerow({
                "label": row["label"], "family": row["family"], "grid": row["grid"],
                "status": "done", "window_sec": row["window_sec"], "threshold": row["threshold"],
                "windows": row["windows"], "hours": round(row["hours"], 4),
                "accepts": row["accepts"], "fa_per_hour": round(row["fa_per_hour"], 4),
                "fa_per_24h": round(row["fa_per_24h"], 4),
                "max_score": row["max_score"],
                "music_accepts": subset["music"], "noise_accepts": subset["noise"],
                "speech_accepts": subset["speech"],
                "checkpoint": row["checkpoint"], "readout_version": row["readout_version"],
            })


ZERO_X = 0.12  # where "0 accepts" is drawn on the log axis
C3, C1 = "#1f77b4", "#ff7f0e"


def annotate(ax, x, y, text, color, dx=1.18):
    ax.annotate(text, (x, y), xytext=(x * dx if x > 0 else x, y), color=color,
                fontsize=7.5, va="center", ha="left" if dx >= 1 else "right")


def plot(rows: list[dict], path: Path) -> None:
    done = {}
    for row in rows:
        done[(row["label"], row["grid"])] = row if row.get("status") == "done" else None
    labels = [label for label, _family, _tag in MODELS]
    positions = list(range(len(labels)))

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(14.5, 6.4), sharey=True)
    fig.subplots_adjust(left=0.22, right=0.985, top=0.85, bottom=0.11, wspace=0.06)

    for label, ypos in zip(labels, positions):
        r3 = done.get((label, GRID_3S))
        r1 = done.get((label, GRID_1S))
        x3 = None if r3 is None else (r3["fa_per_24h"] if r3["fa_per_24h"] > 0 else ZERO_X)
        x1 = None if r1 is None else (r1["fa_per_24h"] if r1["fa_per_24h"] > 0 else ZERO_X)
        if r3 is not None and r1 is not None and r3["fa_per_24h"] > 0 and r1["fa_per_24h"] > 0:
            ax.plot([x3, x1], [ypos, ypos], color="0.75", linewidth=1.2, zorder=1)
        for run, value, color, marker, grid_name in (
            (r3, x3, C3, "o", "3 s / 3 s grid (official)"),
            (r1, x1, C1, "s", "1 s / 1 s grid (stress)"),
        ):
            if run is None:
                continue
            hit = run["fa_per_24h"] > 0
            ax.plot(value, ypos, marker=marker, color=color, markersize=7, zorder=3,
                    markerfacecolor=color if hit else "white",
                    label=grid_name if ypos == 0 else None)
            if hit:
                shown = f"{run['fa_per_24h']:.1f}" if run["fa_per_24h"] < 100 else f"{run['fa_per_24h']:.0f}"
            else:
                shown = "0"
            annotate(ax, value, ypos, shown, color)
        if r1 is None:
            ax.text(1.2, ypos, "1 s: pending", color="0.45", fontsize=7.5,
                    style="italic", va="center", ha="left")

    ax.set_xscale("log")
    ax.set_xlim(0.08, 60000)
    ax.set_yticks(positions)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_ylim(len(labels) - 0.5, -0.5)
    ax.set_xlabel("false accepts per 24 h @ threshold 0.5  (log scale)")
    ax.set_title('MUSAN held-out false alarms - 902 files / 43.72 h - keyword "hey eva"',
                 fontsize=10)
    ax.grid(axis="x", alpha=0.3)
    ax.legend(fontsize=8, loc="upper right", title="window grid", title_fontsize=8)

    for label, ypos in zip(labels, positions):
        r3 = done.get((label, GRID_3S))
        r1 = done.get((label, GRID_1S))
        scores = [r["max_score"] for r in (r3, r1) if r is not None and r["max_score"]]
        close = len(scores) == 2 and max(scores) / max(min(scores), 1e-9) < 2.0
        for grid_name, run in (("3 s", r3), ("1 s", r1)):
            if run is None or run["max_score"] is None:
                continue
            color = C3 if grid_name == "3 s" else C1
            marker = "o" if grid_name == "3 s" else "s"
            value = run["max_score"] if run["max_score"] > 0 else ZERO_X
            bx.plot(value, ypos, marker=marker, color=color, markersize=6.5,
                    markerfacecolor=color if run["max_score"] > 0 else "white", zorder=3)
            dx = 0.80 if (close and grid_name == "3 s") else 1.18
            annotate(bx, value, ypos, f"{run['max_score']:.3f}", "0.3", dx=dx)
    bx.axvline(0.5, color="tab:red", linestyle="--", linewidth=1)
    bx.text(0.53, len(labels) - 1.35, "0.5 threshold", color="tab:red", fontsize=8,
            va="center", ha="left")
    bx.set_xscale("log")
    bx.set_xlim(0.005, 3.0)
    bx.set_xlabel("highest background score seen (log scale)")
    bx.set_title("Safety margin: highest score over the whole corpus", fontsize=10)
    bx.grid(axis="x", alpha=0.3)

    seen_family = None
    for label, ypos in zip(labels, positions):
        family = next(fam for name, fam, _tag in MODELS if name == label)
        if seen_family is not None and family != seen_family:
            for axis in (ax, bx):
                axis.axhline(ypos - 0.5, color="0.85", linewidth=0.8, zorder=0)
        seen_family = family

    fig.suptitle("Which models have been tested on MUSAN 3 s / 1 s - false alarms and score margin",
                 fontsize=12.5)
    fig.text(0.5, 0.015,
             "0 = no accept in 43.72 h (corpus granularity 0.55 per 24 h).  "
             "FA/24 h is not comparable across hops; the 1 s grid is a finer-time stress test.",
             ha="center", fontsize=8, color="0.35")
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-prefix", type=Path,
                        default=ROOT / "outputs/v41_final/figures/musan_fa_3s_vs_1s",
                        help="output path without extension")
    args = parser.parse_args(argv)
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)

    rows = collect()
    write_csv(rows, args.output_prefix.with_suffix(".csv"))
    plot(rows, args.output_prefix.with_suffix(".png"))

    print(f"{'model':<28}{'grid':>5}{'accepts':>9}{'FA/24h':>11}{'max score':>11}  status")
    for row in rows:
        if row.get("status") != "done":
            print(f"{row['label']:<28}{row['grid']:>5}{'-':>9}{'-':>11}{'-':>11}  pending")
            continue
        max_score = "-" if row["max_score"] is None else f"{row['max_score']:.4f}"
        print(f"{row['label']:<28}{row['grid']:>5}{row['accepts']:>9}{row['fa_per_24h']:>11.2f}"
              f"{max_score:>11}  done")
    print("wrote", args.output_prefix.with_suffix(".png"), "+ .csv")


if __name__ == "__main__":
    main()
