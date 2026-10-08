#!/usr/bin/env python3
"""Zero-training analysis of the v4.1 sink head from hard_probe.npz.

Runs three step-0 checks:
  1. softmin temperature sweep (re-pooling stored position logits)
  2. sink-state linear probe: does the unused sink state carry signal?
  3. length-conditioned analysis of the pooled logit
and a fourth check on saved MUSAN score dumps:
  4. sliding-window smoothing / N-of-M persistence for false-accept de-spiking

Usage:
  .venv/bin/python scripts/analyze_v41_offline.py
    prep.probe=outputs/v41_offline_probe/hard_probe.npz
    prep.output=outputs/v41_offline_probe/analysis.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler

from qbyt.pooling import _masked_normalized_softmin


def metrics(y, s):
    y = np.asarray(y).astype(np.int32)
    s = np.asarray(s).astype(np.float64)
    auc = float(roc_auc_score(y, s))
    fpr, tpr, thr = roc_curve(y, s)
    tpr_at = {}
    for target in (0.01, 0.001):
        idx = np.searchsorted(fpr, target, side="right") - 1
        idx = max(idx, 0)
        tpr_at[target] = float(tpr[idx])
    # EER: interpolate where tpr = 1 - fpr
    fnr = 1.0 - tpr
    cross = np.where(fpr >= fnr)[0]
    if cross.size:
        i = int(cross[0])
        eer = float((fpr[i] + fnr[i]) / 2.0)
    else:
        eer = float("nan")
    pauc = float(roc_auc_score(y, s, max_fpr=0.01))
    return {
        "auc": auc,
        "eer": eer,
        "tpr_at_fpr_1e_2": tpr_at[0.01],
        "tpr_at_fpr_1e_3": tpr_at[0.001],
        "pauc_fpr_1e_2": pauc,
        "num_positive": int(y.sum()),
        "num_samples": int(y.size),
    }


def pooled_from_positions(pos_t, mask_t, temperature):
    logits = _masked_normalized_softmin(pos_t, mask_t, temperature)
    return logits.numpy().astype(np.float64)


def load_musan(path):
    import json as _json

    windows = {}
    total = 0
    with open(path) as handle:
        for line in handle:
            obj = _json.loads(line)
            if obj.get("skipped"):
                continue
            total += 1
            span = obj.get("clip_span_sec") or {}
            span = span if isinstance(span, dict) else {}
            start = float(span.get("start_sec", 0.0))
            end = float(span.get("end_sec", start + 1.0))
            item = (start, end, float(obj["qbyt_score"]))
            windows.setdefault(obj["audio_path"], []).append(item)
    for key in windows:
        windows[key].sort(key=lambda item: item[0])
    return windows, total


def despike(windows, thresholds, hop_sec):
    total_windows = sum(len(v) for v in windows.values())
    hours = total_windows * hop_sec / 3600.0
    out = {}
    for thr in thresholds:
        raw_windows = 0
        raw_events = 0
        smooth_windows = 0
        smooth_events = 0
        kof2_events = 0
        for _path, seq in windows.items():
            scores = [item[2] for item in seq]
            flags = [1 if value >= thr else 0 for value in scores]
            raw_windows += sum(flags)
            raw_events += sum(1 for i, flag in enumerate(flags) if flag and (i == 0 or not flags[i - 1]))
            smooth = []
            for i in range(len(scores)):
                lo = max(0, i - 1)
                hi = min(len(scores), i + 2)
                smooth.append(sum(scores[lo:hi]) / (hi - lo))
            sflags = [1 if value >= thr else 0 for value in smooth]
            smooth_windows += sum(sflags)
            smooth_events += sum(1 for i, flag in enumerate(sflags) if flag and (i == 0 or not sflags[i - 1]))
            kflags = [1 if (flags[i] and i + 1 < len(flags) and flags[i + 1]) else 0 for i in range(len(flags))]
            kof2_events += sum(1 for i, flag in enumerate(kflags) if flag and (i == 0 or not kflags[i - 1]))
        out[str(thr)] = {
            "raw_windows": raw_windows,
            "raw_events": raw_events,
            "smooth3_windows": smooth_windows,
            "smooth3_events": smooth_events,
            "kof2_events": kof2_events,
            "raw_per_24h": raw_events * 24.0 / hours,
            "smooth3_per_24h": smooth_events * 24.0 / hours,
            "kof2_per_24h": kof2_events * 24.0 / hours,
        }
    return {"total_windows": total_windows, "hours": hours, "thresholds": out}


def main() -> None:
    probe_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("outputs/v41_offline_probe/hard_probe.npz")
    out_path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("outputs/v41_offline_probe/analysis.json")

    data = np.load(probe_path)
    y = data["label"].astype(np.int32)
    pooled = data["logits"].astype(np.float64)
    pos = data["pos_logits"].astype(np.float32)
    mask = data["pos_mask"].astype(bool)
    sink = data["sink"].astype(np.float32)
    audio_len = data["audio_len"].astype(np.int32)
    text_len = data["text_len"].astype(np.int32)
    print("loaded", probe_path, "samples", pooled.size, "positives", int(y.sum()))

    report = {"probe": str(probe_path), "baseline": metrics(y, pooled)}

    pos_t = torch.from_numpy(pos)
    mask_t = torch.from_numpy(mask)
    recheck = pooled_from_positions(pos_t, mask_t, 1.0)
    report["baseline"]["repool_max_abs_diff"] = float(np.abs(recheck - pooled).max())

    temperatures = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0]
    sweep = {}
    for temperature in temperatures:
        score = pooled_from_positions(pos_t, mask_t, temperature)
        sweep[str(temperature)] = metrics(y, score)
    report["temperature_sweep"] = sweep
    report["temperature_best"] = max(sweep.items(), key=lambda item: item[1]["auc"])[0]

    rng = np.random.default_rng(2025)
    order = rng.permutation(pooled.size)
    half = pooled.size // 2
    fit_idx = order[:half]
    eval_idx = order[half:]
    scaler = StandardScaler().fit(sink[fit_idx])
    sink_fit = scaler.transform(sink[fit_idx])
    sink_eval = scaler.transform(sink[eval_idx])
    sink_model = LogisticRegression(max_iter=2000, C=1.0)
    sink_model.fit(sink_fit, y[fit_idx])
    sink_logit_fit = sink_model.decision_function(sink_fit)
    sink_logit_eval = sink_model.decision_function(sink_eval)
    sink_eval_acc = float((sink_model.predict(sink_eval) == y[eval_idx]).mean())
    report["sink_probe"] = {
        "sink_only_auc_eval_half": float(roc_auc_score(y[eval_idx], sink_logit_eval)),
        "sink_only_auc_fit_half": float(roc_auc_score(y[fit_idx], sink_logit_fit)),
        "sink_only_accuracy_eval_half": sink_eval_acc,
        "mean_sink_logit_positive": float(sink_logit_fit[y[fit_idx] == 1].mean()),
        "mean_sink_logit_negative": float(sink_logit_fit[y[fit_idx] == 0].mean()),
    }

    pooled_scaler = StandardScaler().fit(pooled[fit_idx].reshape(-1, 1))
    p_fit = pooled_scaler.transform(pooled[fit_idx].reshape(-1, 1)).ravel()
    p_eval = pooled_scaler.transform(pooled[eval_idx].reshape(-1, 1)).ravel()
    combo = LogisticRegression(max_iter=2000, C=1.0)
    combo.fit(np.column_stack([p_fit, sink_logit_fit]), y[fit_idx])
    combo_eval = combo.decision_function(np.column_stack([p_eval, sink_logit_eval]))
    report["sink_probe"]["pooled_only_auc_eval_half"] = float(roc_auc_score(y[eval_idx], p_eval))
    report["sink_probe"]["pooled_plus_sink_auc_eval_half"] = float(roc_auc_score(y[eval_idx], combo_eval))
    report["sink_probe"]["combo_metrics_eval_half"] = metrics(y[eval_idx], combo_eval)
    report["sink_probe"]["pooled_only_metrics_eval_half"] = metrics(y[eval_idx], p_eval)
    report["sink_probe"]["combo_coef"] = [float(v) for v in combo.coef_.ravel()]
    report["sink_probe"]["combo_intercept"] = float(combo.intercept_[0])
    report["sink_probe"]["sink_coef"] = [float(v) for v in sink_model.coef_.ravel()]
    report["sink_probe"]["sink_intercept"] = float(sink_model.intercept_[0])
    report["sink_probe"]["sink_mean"] = [float(v) for v in scaler.mean_]
    report["sink_probe"]["sink_scale"] = [float(v) for v in scaler.scale_]
    report["sink_probe"]["pooled_mean"] = float(pooled_scaler.mean_[0])
    report["sink_probe"]["pooled_scale"] = float(pooled_scaler.scale_[0])

    alpha_grid = [0.0, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
    alpha_results = {}
    for alpha in alpha_grid:
        score = pooled[eval_idx] + alpha * sink_logit_eval
        alpha_results[str(alpha)] = float(roc_auc_score(y[eval_idx], score))
    report["sink_probe"]["alpha_grid_auc_eval_half"] = alpha_results

    null_scores = {}
    for temperature in [0.25, 0.4, 0.5, 0.75, 1.0]:
        null_logit = torch.from_numpy(sink_logit_eval.astype(np.float32)).reshape(-1, 1)
        null_mask = torch.ones_like(null_logit, dtype=torch.bool)
        pos_eval = pos_t[eval_idx]
        mask_eval = mask_t[eval_idx]
        width = pos_eval.shape[1]
        combined_pos = torch.cat([pos_eval, null_logit], dim=1)
        combined_mask = torch.cat([mask_eval, null_mask], dim=1)
        score = pooled_from_positions(combined_pos, combined_mask, temperature)
        null_scores[str(temperature)] = metrics(y[eval_idx], score)
    report["sink_probe"]["null_mixture_eval_half"] = null_scores

    audio_bins = [0, 5, 10, 15, 20, 30, 10 ** 9]
    text_bins = [0, 3, 5, 7, 9, 10 ** 9]
    length_report = {}
    for name, values, bins in (("audio_len", audio_len, audio_bins), ("text_len", text_len, text_bins)):
        entries = []
        for low, high in zip(bins[:-1], bins[1:]):
            sel = (values >= low) & (values < high)
            if sel.sum() < 10:
                continue
            entry = {"range": [int(low), int(high)], "n": int(sel.sum()), "pos_rate": float(y[sel].mean())}
            if 0 < y[sel].sum() < sel.sum():
                entry["auc"] = float(roc_auc_score(y[sel], pooled[sel]))
            entry["mean_score_pos"] = float(pooled[sel & (y == 1)].mean()) if (sel & (y == 1)).any() else None
            entry["mean_score_neg"] = float(pooled[sel & (y == 0)].mean()) if (sel & (y == 0)).any() else None
            entries.append(entry)
        from scipy.stats import spearmanr

        pos_corr = spearmanr(values[y == 1], pooled[y == 1]).statistic
        neg_corr = spearmanr(values[y == 0], pooled[y == 0]).statistic
        length_report[name] = {
            "bins": entries,
            "spearman_score_vs_length_pos": float(pos_corr),
            "spearman_score_vs_length_neg": float(neg_corr),
        }

    # per-bin z-normalisation fitted on the fit half, applied to the eval half
    for name, values, bins in (("audio_len", audio_len, audio_bins), ("text_len", text_len, text_bins)):
        stats = {}
        for low, high in zip(bins[:-1], bins[1:]):
            sel = (values >= low) & (values < high) & (y == 0)
            sel_fit = sel.copy()
            sel_fit[eval_idx] = False
            if sel_fit.sum() > 20:
                stats[(low, high)] = (float(pooled[sel_fit].mean()), float(pooled[sel_fit].std() + 1e-6))
        normalized = pooled[eval_idx].copy()
        for i, sample in enumerate(eval_idx):
            value = values[sample]
            for (low, high), (mean, std) in stats.items():
                if low <= value < high:
                    normalized[i] = (pooled[sample] - mean) / std
                    break
        length_report[name]["auc_eval_half_raw"] = float(roc_auc_score(y[eval_idx], pooled[eval_idx]))
        length_report[name]["auc_eval_half_bin_z"] = float(roc_auc_score(y[eval_idx], normalized))
        length_report[name]["bin_stats"] = {str(key): value for key, value in stats.items()}
    report["length_analysis"] = length_report

    musan_1s = Path("data/dma-kws/exp/stage2_qbyt/fa/v41-musan-1s0/results.jsonl")
    musan_3s = Path("data/dma-kws/exp/stage2_qbyt/fa/v41-musan/results.jsonl")
    fa_report = {}
    if musan_1s.exists():
        windows, _total = load_musan(musan_1s)
        fa_report["musan_1s"] = despike(windows, [0.05, 0.1, 0.2, 0.3, 0.5], 1.0)
    if musan_3s.exists():
        windows, _total = load_musan(musan_3s)
        fa_report["musan_3s"] = despike(windows, [0.05, 0.1, 0.2, 0.3, 0.5], 3.0)
    report["false_alarm_despike"] = fa_report

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=1))
    print(json.dumps({k: report[k] for k in ("baseline", "temperature_best")}, indent=1))
    print("wrote", out_path)


if __name__ == "__main__":
    main()
