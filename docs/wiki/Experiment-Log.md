# Experiment Log

Archive of every Stage II QbyT run on this host: configuration, throughput
measurements, validation curves, artefacts and open evaluations.

* Environment and version pins: [Environment Setup](Environment-Setup)
* Data prep, icefall/k2/cuDNN setup, smoke test: [Stage II Training Pipeline](Stage-II-Training-Pipeline)

**Testbed:** 1× Tesla V100-SXM2-32GB (sm_70) · 10 CPUs · 38 GB RAM · torch 2.7.1+cu126 ·
k2 1.24.4 · icefall `3f848bb` · cuDNN 9.5.1 (forced through `LD_LIBRARY_PATH`).

## 0. Shared inputs for every run below

| item | value |
| --- | --- |
| training parquet | `data/dma-kws/processed/stage2_qbyt/ls-gs-1460/aggregated_segments_with_g2p_distance.parquet` — 155,619 anchors / 5,837,186 clips (GigaPhrase-1000 + LibriPhrase-460) |
| training fbank | `data/dma-kws/features/fbank_icefall_kws` — 5,837,186 `.npy`, 133 GB |
| validation | LibriPhrase **hard split** (270,684 rows = 2,115 batches), full split every time (`drop_last=False`) |
| eval fbank | `data/dma-kws/features/fbank_icefall_kws_eval` |
| background noise | MUSAN fbank cache `data/dma-kws/processed/background/musan/cache` — cache_id `f1ff880b3165c262ed7b6f2f9980ed698a00d74653bb9fe0a691a021466fc19b`, 993 train recordings, 320,739 crops, `mode: fbank_cache`, probability 0.25, 1–3 s crops |
| encoder | `raw/kws-checkpoints/zh-en-3M-2025-12-20/pretrained-epoch-13-avg-2.pt`, `cnn_module_kernel: 15,15,15,15,15,15`, causal, frozen (2,751,965 params, 0 missing / 0 unexpected) |
| optimiser / schedule | inherited from the icefall Stage II preset; `batch_size_per_gpu: 128`, `accumulate_grad_batches: 1`, `precision: 16-mixed`, `max_steps == total_scheduler_steps` |

## 1. Runs at a glance

| run | tmux session | experiment config | readout | background | steps | wall time | trainable | best val AUC | EER @ best | TPR@FPR1e-2 | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **v3** | `qbyt-v3-musan-zhen3m` | `icefall_zipformer_stage2_v3_musan_cached_zhen3m_50k` | version 3, `eps_mean` | MUSAN cache | 50,000 | **3.38 h** | 422 K | 0.931601 | 0.132239 | 0.1568 | ✅ complete |
| **v4** | `qbyt-v4-musan-zhen3m` | `icefall_zipformer_stage2_v4_musan_cached_zhen3m_50k` | version 4, `eps_softmin` | MUSAN cache | 50,000 | **3.59 h** | 422 K | 0.932327 | 0.133621 | 0.1703 | ✅ complete |
| **v4.1** | `qbyt-v41-musan-zhen3m` | `icefall_zipformer_stage2_eps_softmin_v41_musan_cached_zhen3m_50k` | version 4, `eps_softmin` + `sink_token` + learned text / relative-bias audio positions | MUSAN cache | 50,000 | **8.23 h**¹ | 439 K | **0.935007** | **0.131075** | **0.1832** | ✅ complete |
| v7 | `qbyt-zhen3m-50k` | `icefall_zipformer_stage2_zhen3m_50k` | version 7 keyword-filler (`one_vs_rest`) | none | 215 / 50,000 | — | 160 K | — | — | — | ⛔ aborted (switched to the v4.1 recipe on request) |

¹ includes the resume from step 5,000, the concurrent period with v3/v4 and 18 full-split
validations; pure training was 6 h 53 min.

Readout definitions (see AGENTS.md): versions 2/3/4 = pooling (`qbyt/pooling.py`),
5 = bounded, 6/7 = keyword-filler (`qbyt/model.py`, `query_relative` vs `one_vs_rest`).
Version 3 cannot carry `eps_softmin` or any v4.1 extension field, so the v3 run uses
`eps_mean` with the legacy defaults `sink_token=false, text_position=sinusoidal,
audio_position=sinusoidal`.

## 2. Readout ladder — validation results

### 2.1 Aligned milestones (nearest validation at or before the step)

| step | v3 AUC | v4 AUC | v4.1 AUC | v3 EER | v4 EER | v4.1 EER |
| --- | --- | --- | --- | --- | --- | --- |
| 5k | 0.8453 | 0.7956 | **0.8603** | 0.2289 | 0.2808 | 0.2178 |
| 10k | 0.8704 | 0.8839 | **0.8854** | 0.2039 | 0.1894 | 0.1858 |
| 15k | 0.8916 | 0.8999 | **0.9010** | 0.1787 | **0.1686** | 0.1728 |
| 20k | **0.9109** | 0.9036 | 0.9092 | **0.1577** | 0.1656 | 0.1627 |
| 25k | 0.9167 | 0.9111 | **0.9203** | 0.1514 | 0.1568 | **0.1505** |
| 30k | 0.9230 | 0.9171 | 0.9222 | 0.1417 | 0.1510 | 0.1440 |
| 35k | 0.9267 | 0.9258 | **0.9275** | **0.1392** | 0.1414 | 0.1394 |
| 40k | 0.9280 | 0.9309 | **0.9330** | 0.1360 | 0.1349 | **0.1341** |
| 45k | 0.9313 | 0.9314 | **0.9342** | 0.1325 | 0.1347 | **0.1318** |
| 50k | 0.9316 | 0.9323 | **0.9348** | 0.1322 | 0.1338 | **0.1313** |

### 2.2 Best-checkpoint metrics (what the exported `.pt` files contain)

| metric | v3 | v4 | v4.1 |
| --- | --- | --- | --- |
| best step | 48,000 | 48,000 | 47,500 |
| `val/auc` | 0.931601 | 0.932327 | **0.935007** |
| `val/eer` | 0.132239 | 0.133621 | **0.131075** |
| `val/tpr_at_fpr_1e_2` | 0.156781 | 0.170258 | **0.183210** |
| `val/tpr_at_fpr_1e_3` | 0.018331 | 0.017223 | **0.020533** |
| `val/pauc_fpr_1e_2` | 0.538021 | 0.539058 | **0.543203** |
| `val/brier` | **0.105891** | 0.106843 | 0.108810 |
| best `val/ece` seen | 0.0633 (@4k) | **0.0417** (@12k) | 0.0580 (@20k) |
| `val/utt_loss` | 0.392701 | 0.384556 | 0.396067 |

### 2.3 Reading

* All three converge smoothly with no overfitting signature: val loss 0.49 / 0.56 / 0.46 → 0.39 / 0.41 / 0.42, Brier → ~0.106–0.110, AUC monotone apart from ±0.002 noise.
* **Plateau after ~35–40 k steps**: the last 10 k steps add only +0.005 (v3), +0.006 (v4), +0.007 (v4.1) AUC; the last 5 k add ≤ 0.001. 40 k steps buys ≈ 99.8 % of the final quality.
* **v4.1 is consistently but modestly ahead** from ~25 k on (+0.002–0.003 AUC, −0.002–0.005 EER, and clearly more at 1 % FPR). It costs 2–2.5× per step, so use it for a final model, not for sweeps.
* **v3 ≈ v4** (final AUC within 0.0007, well below the ±0.002 validation noise). The differences are mainly early-phase (v4 starts much slower) and calibration (v4's best ECE 0.042 vs v3's 0.063).
* Metric noise: ~±0.002 AUC / ±0.005 EER between neighbouring points of the same run; `tpr_at_fpr_1e_3` is noisier still (0.005–0.008) — do not read single-point gaps below that.

### 2.4 Full validation curves (raw)

**v3 (eps_mean), 12 points**

| step | val_loss | AUC | EER | TPR@1e-2 | TPR@1e-3 | Brier | ECE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 4000 | 0.4888 | 0.8453 | 0.2289 | 0.0523 | 0.0051 | 0.1577 | 0.0633 |
| 8000 | 0.4598 | 0.8704 | 0.2039 | 0.0590 | 0.0068 | 0.1440 | 0.0759 |
| 12000 | 0.4274 | 0.8916 | 0.1787 | 0.0670 | 0.0082 | 0.1315 | 0.0752 |
| 16000 | 0.4393 | 0.9056 | 0.1655 | 0.1147 | 0.0127 | 0.1295 | 0.0789 |
| 20000 | 0.3974 | 0.9109 | 0.1577 | 0.1130 | 0.0138 | 0.1175 | 0.0634 |
| 24000 | 0.3965 | 0.9167 | 0.1514 | 0.1114 | 0.0156 | 0.1179 | 0.0782 |
| 28000 | 0.4699 | 0.9230 | 0.1417 | 0.1206 | 0.0170 | 0.1222 | 0.1048 |
| 32000 | 0.3927 | 0.9267 | 0.1392 | 0.1532 | 0.0183 | 0.1092 | 0.0771 |
| 36000 | 0.4160 | 0.9279 | 0.1355 | 0.1314 | 0.0127 | 0.1122 | 0.0905 |
| 40000 | 0.4189 | 0.9280 | 0.1360 | 0.1393 | 0.0115 | 0.1115 | 0.0889 |
| 44000 | 0.4085 | 0.9313 | 0.1325 | 0.1568 | 0.0123 | 0.1074 | 0.0831 |
| 48000 | 0.3934 | **0.9316** | **0.1322** | 0.1546 | 0.0136 | 0.1059 | 0.0787 |

**v4 (eps_softmin), 16 points**

| step | val_loss | AUC | EER | TPR@1e-2 | TPR@1e-3 | Brier | ECE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 3000 | 0.5609 | 0.7956 | 0.2808 | 0.0414 | 0.0037 | 0.1932 | 0.0940 |
| 6000 | 0.4705 | 0.8675 | 0.2049 | 0.0591 | 0.0061 | 0.1472 | 0.0768 |
| 9000 | 0.4652 | 0.8839 | 0.1894 | 0.0786 | 0.0066 | 0.1413 | 0.0927 |
| 12000 | 0.4091 | 0.8898 | 0.1830 | 0.0685 | 0.0092 | 0.1285 | 0.0417 |
| 15000 | 0.4360 | 0.8999 | 0.1686 | 0.0692 | 0.0068 | 0.1307 | 0.0912 |
| 18000 | 0.4237 | 0.9036 | 0.1656 | 0.0836 | 0.0093 | 0.1263 | 0.0828 |
| 21000 | 0.4499 | 0.9061 | 0.1600 | 0.0724 | 0.0078 | 0.1282 | 0.0992 |
| 24000 | 0.4024 | 0.9111 | 0.1568 | 0.0906 | 0.0106 | 0.1164 | 0.0665 |
| 27000 | 0.4684 | 0.9109 | 0.1576 | 0.0922 | 0.0102 | 0.1276 | 0.1023 |
| 30000 | 0.4169 | 0.9171 | 0.1510 | 0.1062 | 0.0135 | 0.1173 | 0.0842 |
| 33000 | 0.3987 | 0.9258 | 0.1414 | 0.1346 | 0.0152 | 0.1125 | 0.0836 |
| 36000 | 0.4054 | 0.9300 | 0.1370 | 0.1566 | 0.0131 | 0.1095 | 0.0815 |
| 39000 | 0.4215 | 0.9309 | 0.1349 | 0.1488 | 0.0152 | 0.1132 | 0.0931 |
| 42000 | 0.3846 | 0.9323 | 0.1336 | 0.1703 | 0.0172 | 0.1068 | 0.0785 |
| 45000 | 0.4165 | 0.9314 | 0.1347 | 0.1517 | 0.0171 | 0.1122 | 0.0918 |
| 48000 | 0.4065 | **0.9323** | 0.1338 | 0.1568 | 0.0159 | 0.1088 | 0.0858 |

**v4.1 (sink + learned text / relative-bias audio positions), 19 points**

| step | val_loss | AUC | EER | TPR@1e-2 | TPR@1e-3 | Brier | ECE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 5000¹ | 0.4647 | 0.8603 | 0.2178 | 0.0608 | 0.0069 | 0.1493 | 0.0581 |
| 7500 | 0.4444 | 0.8740 | 0.2038 | 0.0720 | 0.0072 | 0.1430 | 0.0604 |
| 10000 | 0.4374 | 0.8854 | 0.1858 | 0.0663 | 0.0059 | 0.1345 | 0.0741 |
| 12500 | 0.4459 | 0.8982 | 0.1734 | 0.0770 | 0.0057 | 0.1312 | 0.0814 |
| 15000 | 0.4399 | 0.9010 | 0.1728 | 0.0998 | 0.0099 | 0.1289 | 0.0799 |
| 17500 | 0.4458 | 0.9109 | 0.1612 | 0.1157 | 0.0127 | 0.1267 | 0.0936 |
| 20000 | 0.3974 | 0.9092 | 0.1627 | 0.1113 | 0.0126 | 0.1196 | 0.0580 |
| 22500 | 0.4201 | 0.9190 | 0.1524 | 0.1409 | 0.0175 | 0.1188 | 0.0855 |
| 25000 | 0.3961 | 0.9203 | 0.1505 | 0.1386 | 0.0135 | 0.1154 | 0.0763 |
| 27500 | 0.4302 | 0.9217 | 0.1470 | 0.1394 | 0.0163 | 0.1165 | 0.0875 |
| 30000 | 0.4331 | 0.9222 | 0.1440 | 0.1212 | 0.0136 | 0.1186 | 0.0924 |
| 32500 | 0.4040 | 0.9293 | 0.1377 | 0.1545 | 0.0167 | 0.1096 | 0.0804 |
| 35000 | 0.4428 | 0.9275 | 0.1394 | 0.1368 | 0.0177 | 0.1168 | 0.0966 |
| 37500 | 0.4272 | 0.9281 | 0.1385 | 0.1323 | 0.0124 | 0.1136 | 0.0898 |
| 40000 | 0.4119 | 0.9330 | 0.1341 | 0.1741 | 0.0180 | 0.1119 | 0.0897 |
| 42500 | 0.4316 | 0.9339 | 0.1330 | 0.1802 | 0.0179 | 0.1138 | 0.0962 |
| 45000 | 0.4212 | 0.9342 | 0.1318 | 0.1767 | 0.0191 | 0.1117 | 0.0922 |
| 47500 | 0.4109 | **0.9350** | **0.1311** | **0.1832** | 0.0199 | 0.1088 | 0.0873 |
| 50000 | 0.4173 | 0.9348 | 0.1313 | 0.1822 | **0.0205** | 0.1098 | 0.0895 |

¹ from the pre-resume segment of the same weight lineage (validation interval was 5000
before it was re-tuned to 2500).

## 3. Throughput and concurrency measurements

### 3.1 Batch-size sweep (v4.1 config, 30–40 steps per point, steady-state rates)

| batch | workers | steady state | samples/s |
| --- | --- | --- | --- |
| 128 | 4 | 2.30 steps/s | 294 |
| 128 | 8 | 2.32 steps/s | 297 |
| 256 | 8 | 1.16 steps/s | 297 |
| 512 | 8 / 10 | 0.57 steps/s | 279–292 |
| 1024 | 10 | 0.27 steps/s | 276 |

Throughput is flat (~276–300 samples/s): the V100 is saturated at batch 128. Larger
batches only lengthen each step (and batch 256 OOMs for the v7 readout: 31.7 GiB requested).

### 3.2 Per-step cost by readout

| readout | steady state | per step | GPU util | VRAM |
| --- | --- | --- | --- | --- |
| v3 `eps_mean` | ~5.5 steps/s | ~0.18 s | ~99 % (shared) | ~2 GB |
| v4 `eps_softmin` | ~6.2 steps/s | ~0.16 s | ~99 % (shared) | ~2 GB |
| v4.1 | 2.28 steps/s | 0.44 s | 88 % solo | 2.1 GB |
| v7 keyword-filler | 0.26–0.29 steps/s | 3.4–3.9 s | 20–25 % | 15–25 GB |

### 3.3 Running several readouts at once

| scenario | v4.1 | v4 | v3 | aggregate | GPU util | VRAM |
| --- | --- | --- | --- | --- | --- | --- |
| v4.1 alone | 2.28 | — | — | 2.28 steps/s | 87.9 % | 2.1 GB |
| v4.1 + v4 | 1.58 | 6.23 | — | 7.82 steps/s | 96.8 % | 4.0 GB |
| v4.1 + v4 + v3 | 1.27 | 4.89 | 4.93 | **11.09 steps/s** | **99.0 %** | 6.0 GB |
| after v4/v3 finished | 2.09 | done | done | 2.09 steps/s | ~90 % | 2.1 GB |

Light + heavy co-location fills the GPU (88 % → 99 %) and multiplies aggregate
throughput; two heavy runs would just split the same saturated GPU. The heavy run pays
for it (2.28 → 1.27 steps/s) but recovers completely once the light runs finish. CPU load
reached 7.4 of 10 cores with three runs; do not add a fourth.

### 3.4 Validation cost

| measurement | value |
| --- | --- |
| full hard split inside a live run (2,115 batches, 8 eval workers) | **2 min 10 s** = 16.25 it/s = 0.061 s/batch |
| 50k-step run with `val_check_interval=2500` (20 validations) | ~44 min of validation |
| 2,048-row sample (16 batches), same workers | 31 s → looks like 1.94 s/batch — **fixed start-up dominates a tiny sample; do not extrapolate** |

## 4. Aborted run — v7 keyword-filler

| item | value |
| --- | --- |
| config / session | `icefall_zipformer_stage2_zhen3m_50k` / `qbyt-zhen3m-50k` |
| readout | version 7, `one_vs_rest` emission, 160 K trainable |
| encoder / data | zh-en avg-2, LS-GS-1460, no background noise |
| progress when stopped | 215 / 50,000 steps (3.4 s/step, GPU 20–25 %) |
| why stopped | user switched the plan to v4.1 + MUSAN backgrounds |

The same model/config reached `loss_total 20.16, loss_utt_raw 17.26,
loss_seq_weighted 2.90, illegal_path_rate 0.0078` in the single-step smoke test, and
50 steps took 194.8 s (`runs.csv`).

## 5. Artefacts (exported best-val_AUC weights)

| run | exported checkpoint | stamp | size |
| --- | --- | --- | --- |
| v3 | `data/dma-kws/exp/stage2_qbyt/checkpoints/v3-musan-zhen3m-50k/v3-musan-zhen3m-50k/version_1/stage2_step048000.pt` | version 3, `mode=eps_mean`, no extensions | 15.5 MB |
| v4 | `data/dma-kws/exp/stage2_qbyt/checkpoints/v4-musan-zhen3m-50k/v4-musan-zhen3m-50k/version_1/stage2_step048000.pt` | version 4, `mode=eps_softmin`, no extensions | 15.5 MB |
| v4.1 | `data/dma-kws/exp/stage2_qbyt/checkpoints/v41-musan-zhen3m-50k/v41-musan-zhen3m-50k/version_9/stage2_step047500.pt` | version 4, `eps_softmin`, `sink_token=true`, learned text / relative-bias audio | 15.5 MB |

Each file carries `model_state_dict`, `config`, `qbyt_readout`, `qbyt_readout_version`,
`tokenizer_dict_path`, `vocab_size`, `qbyt_alignment_spec` and `dma_kws_run_context`.
Per-run logs/metrics/hparams live under
`data/dma-kws/exp/stage2_qbyt/logs/<run_name>/<run_name>/version_*/`
(`metrics.csv`, `eval_history.csv`, `hparams.yaml`, tfevents), and one summary row per run
is appended to `data/dma-kws/exp/stage2_qbyt/runs.csv`.
Console transcripts of the tmux panes: `data/dma-kws/exp/stage2_qbyt/<session>.pane.log`.

## 6. Evaluation status

| evaluation | status |
| --- | --- |
| LibriPhrase hard-split AUC/EER/TPR (above) | ✅ done for v3 / v4 / v4.1 |
| MUSAN held-out false alarms per 24 h (`processed/musan_split/eval_musan.list`, 902 recordings / 43.72 h, 51,994 windows) | ✅ **0 accepts at threshold 0.5 → 0 次/24h for v3, v4 and v4.1**; worst-case background score 0.019 / 0.015 / 0.409, so v4.1's margin is ~20× smaller ([False-Alarm Evaluation](False-Alarm-Evaluation)) |
| MUSAN 1 s window / 1 s hop grid (156,929 windows) | ✅ **v4.1 29.1 < v3 35.7 < v4 45.0 false accepts per 24 h at threshold 0.5**; the 3 s grid is still 0 for all three — the shorter window inverts the ordering and shows the 0.5 threshold is calibrated for 3 s inputs ([False-Alarm Evaluation](False-Alarm-Evaluation) §7.2) |
| LibriSpeech `train-other-500` false alarms per 24 h (stride-6 subset, 24,782 flac / 82.71 h, 87,243 windows) | ✅ v4.1 **0 accepts @ 0.5 → 0 次/24h** (0.58 per 24 h at 0.2, 6.4 at 0.1); v3 / v4 ⏳ running |
| Two-stage (QbyT + verifier) end-to-end event rate | ⏳ planned |

## 7. Reproducing a run

```bash
# start (or restart) a run in tmux; the session is named after the experiment unless given
SESSION_NAME=qbyt-v3-musan-zhen3m bash scripts/start_stage2_qbyt_tmux.sh \
  icefall_zipformer_stage2_v3_musan_cached_zhen3m_50k qbyt-v3-musan-zhen3m

# watch / attach / stop
tmux capture-pane -pt qbyt-v3-musan-zhen3m -S -40
tmux attach -t qbyt-v3-musan-zhen3m
tmux kill-session -t qbyt-v3-musan-zhen3m

# resume a finished or interrupted run from its Lightning checkpoint
EXTRA_OVERRIDES="run.resume_from=data/dma-kws/exp/stage2_qbyt/checkpoints/<run>/<run>/version_N/last.ckpt" \
  bash scripts/start_stage2_qbyt_tmux.sh <experiment> <session>
```

## 8. Template for the next record

```markdown
### <run name> — <date>

| item | value |
| --- | --- |
| config / session | ... |
| readout (version, mode, extensions) | ... |
| encoder checkpoint (params, missing/unexpected) | ... |
| data / background | ... |
| steps, batch, precision, workers, val interval | ... |
| wall time, GPU util, VRAM, final train loss | ... |
| best step, val_auc / eer / tpr@1e-2 / tpr@1e-3 / pauc / brier / ece | ... |
| exported checkpoint path + stamp | ... |
| evaluation results (MUSAN FA/h, LibriSpeech FA/h, ...) | ... |
| notes, anomalies, follow-ups | ... |
```
