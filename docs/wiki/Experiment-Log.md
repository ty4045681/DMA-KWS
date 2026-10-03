# Experiment Log

Archive of every Stage II QbyT run on this host: configuration, validation curves, best-checkpoint
metrics, throughput measurements, artefacts and open evaluations.

* Environment and version pins: [Environment Setup](Environment-Setup)
* Data prep, icefall/k2/cuDNN setup, smoke test: [Stage II Training Pipeline](Stage-II-Training-Pipeline)
* False alarms (误唤醒率): [False-Alarm Evaluation](False-Alarm-Evaluation)
* Paper Stage II recipe (negative mixing, finetune lineage): [Paper Stage II Recipe](Paper-Stage-II-Recipe)

**Last updated:** 2026-10-03 — **15 completed 50k-step runs**: 4 readouts (v2 / v3 / v4 / v4.1) ×
3 frozen encoders (zh-en-3M avg-2, GigaSpeech KWS finetune, GigaSpeech KWS base), plus the
GigaSpeech-base encoder at a second operating point (full context vs streaming).
Afterwards: MUSAN training backgrounds corrected to **music + noise only** (§6.6), the FA campaign
finished for all 12 remaining checkpoints (§9), and the paper Stage I conformer ladder (v2/v3/v4/v4.1)
started.

**Testbed:** 1× Tesla V100-SXM2-32GB (sm_70) · 10 CPUs · 38 GB RAM (+ 47 GB swap) · torch 2.7.1+cu126 ·
k2 1.24.4 · icefall `3f848bb` · cuDNN 9.5.1 (forced via `LD_LIBRARY_PATH`).

## 0. Shared inputs for every run below

| item | value |
| --- | --- |
| training parquet | `data/dma-kws/processed/stage2_qbyt/ls-gs-1460/aggregated_segments_with_g2p_distance.parquet` — 155,619 anchors / 5,837,186 clips (GigaPhrase-1000 + LibriPhrase-460) |
| training fbank | `data/dma-kws/features/fbank_icefall_kws` — 133 GB, lhotse profile (`snip_edges=False`, `high_freq=-400`, no `1<<15` scale) |
| validation | LibriPhrase **hard split** (270,684 rows = 2,115 batches), full split every time (`drop_last=False`) |
| eval fbank | `data/dma-kws/features/fbank_icefall_kws_eval` |
| background noise | MUSAN cache, **music + noise only** (speech excluded; see §6.6): icefall-profile `processed/background/musan/cache` — cache_id `2b83f85f30ff3af8…`, 774 train recordings, K=414, 320,436 crops; Wenet-profile `…/cache_wenet` — cache_id `3b73024a08bbdbf4…`, same counts; `mode: fbank_cache`, probability 0.25, 1–3 s crops |
| zh-en encoder | `raw/kws-checkpoints/zh-en-3M-2025-12-20/pretrained-epoch-13-avg-2.pt` — `cnn_module_kernel: 15,15,15,15,15,15`, causal, frozen 2.75 M params, 0 missing / 0 unexpected |
| GS-KWS finetune encoder | `raw/kws-checkpoints/gigaspeech-20240219/exp-finetune/pretrained.pt` — `cnn_module_kernel: 31,31,15,15,15,31`, causal, 0/0 |
| GS-KWS base encoder | `raw/kws-checkpoints/gigaspeech-20240219/exp/pretrained.pt` — same architecture, causal, 0/0 |
| optimiser / schedule | icefall Stage II preset; `batch_size_per_gpu: 128`, `accumulate_grad_batches: 1`, `precision: 16-mixed`, `max_steps == total_scheduler_steps == 50000` |
| readouts | v2 = pooling `gru_last` · v3 = pooling `eps_mean` · v4 = pooling `eps_softmin` · v4.1 = v4 + `sink_token` + learned text / relative-bias audio positions |

## 1. Runs at a glance — 15 completed 50k-step runs

| encoder | readout | operating point | best AUC | EER | TPR@1%FPR | TPR@0.1%FPR | pAUC(<=1%) | Brier | best step | wall time |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| zh-en-3M | **v2** | streaming 16/64 | 0.937647 | 0.128644 | 0.202073 | 0.023467 | 0.552360 | 0.135602 | 50000 | 3.18 h |
| zh-en-3M | **v3** | streaming 16/64 | 0.931601 | 0.132239 | 0.156781 | 0.018331 | 0.538021 | 0.105891 | 48000 | 3.38 h |
| zh-en-3M | **v4** | streaming 16/64 | 0.932327 | 0.133621 | 0.170258 | 0.017223 | 0.539058 | 0.106843 | 48000 | 3.59 h |
| zh-en-3M | **v4.1** | streaming 16/64 | 0.935007 | 0.131075 | 0.183210 | 0.020533 | 0.543203 | 0.108810 | 47500 | 8.23 h |
| GS-KWS finetune | **v2** | streaming 16/64 | 0.880924 | 0.195756 | 0.089869 | 0.009953 | 0.522315 | 0.188063 | 48000 | 3.03 h |
| GS-KWS finetune | **v3** | streaming 16/64 | 0.859759 | 0.216651 | 0.077079 | 0.012531 | 0.517957 | 0.156922 | 48000 | 3.57 h |
| GS-KWS finetune | **v4** | streaming 16/64 | 0.856506 | 0.219640 | 0.073658 | 0.010307 | 0.517027 | 0.158888 | 48000 | 3.85 h |
| GS-KWS finetune | **v4.1** | streaming 16/64 | 0.868674 | 0.207674 | 0.081423 | 0.010869 | 0.518470 | 0.153237 | 45000 | 8.61 h |
| GS-KWS base | **v3** | full context -1/-1 | 0.913611 | 0.157224 | 0.125918 | 0.017282 | 0.531374 | 0.128847 | 48000 | 3.88 h |
| GS-KWS base | **v4** | full context -1/-1 | 0.919524 | 0.147866 | 0.130580 | 0.018110 | 0.532495 | 0.121307 | 48000 | 4.15 h |
| GS-KWS base | **v4.1** | full context -1/-1 | 0.912886 | 0.156308 | 0.122231 | 0.013329 | 0.530486 | 0.126457 | 45000 | 9.76 h |
| GS-KWS base | **v2** | streaming 16/64 | 0.927132 | 0.141135 | 0.136883 | 0.012745 | 0.536057 | 0.147952 | 48000 | 2.82 h |
| GS-KWS base | **v3** | streaming 16/64 | 0.906355 | 0.163002 | 0.105111 | 0.013965 | 0.524005 | 0.131689 | 48000 | 3.51 h |
| GS-KWS base | **v4** | streaming 16/64 | 0.913148 | 0.156969 | 0.127388 | 0.014393 | 0.530762 | 0.125535 | 48000 | 3.78 h |
| GS-KWS base | **v4.1** | streaming 16/64 | 0.911919 | 0.159500 | 0.124721 | 0.015383 | 0.529277 | 0.128110 | 45000 | 8.29 h |

Each row is one frozen encoder plus a trained QbyT head (v2/v3/v4: 422 K trainable parameters, v4.1:
439 K; the encoder itself is frozen). Every validation is the full hard split. Aborted runs: §7.

## 2. What the data says

1. **The paper-original v2 (`gru_last`) readout wins on all three encoders** — zh-en 0.937647 vs
   0.935007 (v4.1) / 0.932327 (v4) / 0.931601 (v3); GS-base 0.927132 vs 0.919524 / 0.913148 /
   0.906355; GS-finetune 0.880924 vs 0.868674 / 0.859759 / 0.856506. Its best TPR@1%FPR (0.202073 on
   zh-en) is the highest measured anywhere in this project. v2 also has the worst calibration
   (Brier 0.1356–0.1963 vs 0.1059–0.1604 for the others).
2. **Encoder ranking (v2, streaming):** zh-en-3M **0.937647** > GS-KWS base **0.927132** > GS-KWS
   finetune **0.880924**; the same order holds for every readout and for TPR@1%FPR (0.202 / 0.137 /
   0.090). The finetuned checkpoint is consistently the worst — finetuning the base model to
   streaming KWS hurts transfer to this utterance-level query task, while the **base** checkpoint
   transfers best.
3. **GS-base operating point matters:** full context (`chunk_size=-1` / `left_context_frames=-1`)
   beats streaming 16/64 for v3/v4/v4.1 (v4: **0.919524 vs 0.913148**, EER 0.147866 vs 0.156969;
   v4.1: 0.912886 vs 0.911919; v3: 0.913611 vs 0.906355). v2 was only trained streaming and still
   beats every full-context run (0.927132).
4. **Readout ordering is encoder-dependent.** zh-en: v2 > v4.1 > v4 ≈ v3. GS-base (stream):
   v2 > v4 > v4.1 > v3. GS-finetune: v2 > v4.1 > v3 > v4. The v4.1 additions help on zh-en and
   GS-finetune but not on GS-base, and they never beat plain v2.
5. **Cost:** v2/v3/v4 take 2.8–4.2 h per 50k steps (0.20–0.30 s/step); v4.1 takes 8.3–9.8 h
   (0.60 s/step solo) and suffers most when co-located. v2 is both the best-quality and one of the
   cheapest readouts.
6. **Plateau:** all curves flatten after ~35–40 k steps; the last 10 k add at most 0.007 AUC.
7. **Noise floor:** ±0.002 AUC / ±0.005 EER between neighbouring validation points of the same run;
   TPR@0.1%FPR is noisier still (0.005–0.008). Do not read single-point gaps below that.

## 3. Readout ladder — zh-en-3M avg-2 encoder (streaming 16/64)

### 3.1 Aligned milestones (AUC, EER in brackets)

| step | v2 AUC (EER) | v3 AUC (EER) | v4 AUC (EER) | v4.1 AUC (EER) |
|---|---|---|---|---|
| 5000 | 0.8596 (0.2159) | 0.8453 (0.2289) | 0.7956 (0.2808) | — |
| 10000 | 0.8931 (0.1797) | 0.8704 (0.2039) | 0.8839 (0.1894) | 0.8854 (0.1858) |
| 20000 | 0.9196 (0.1499) | 0.9109 (0.1577) | 0.9036 (0.1656) | 0.9092 (0.1627) |
| 30000 | 0.9303 (0.1381) | 0.9230 (0.1417) | 0.9171 (0.1510) | 0.9222 (0.1440) |
| 40000 | 0.9352 (0.1318) | 0.9280 (0.1360) | 0.9309 (0.1349) | 0.9330 (0.1341) |
| 48000 | 0.9372 (0.1291) | 0.9316 (0.1322) | 0.9323 (0.1338) | 0.9350 (0.1311) |
| best step | 50000 | 48000 | 48000 | 47500 |
| **best AUC** | **0.937647** | **0.931601** | **0.932327** | **0.935007** |
| EER @ best | 0.128644 | 0.132239 | 0.133621 | 0.131075 |
| TPR@1%FPR | 0.202073 | 0.156781 | 0.170258 | 0.183210 |
| TPR@0.1%FPR | 0.023467 | 0.018331 | 0.017223 | 0.020533 |
| pAUC(<=1%) | 0.552360 | 0.538021 | 0.539058 | 0.543203 |
| Brier | 0.135602 | 0.105891 | 0.106843 | 0.108810 |
| wall time | 3.18 h | 3.38 h | 3.59 h | 8.23 h |

### 3.2 Full validation curves

**v2** — v2-musan-zhen3m-stream-50k (20 validation points)

| step | val_loss | AUC | EER | TPR@1e-2 | TPR@1e-3 | Brier | ECE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2500 | 0.6755 | 0.7914 | 0.2844 | 0.0429 | 0.0042 | 0.2166 | 0.1684 |
| 5000 | 0.6741 | 0.8596 | 0.2159 | 0.0519 | 0.0055 | 0.1814 | 0.1633 |
| 7500 | 0.6248 | 0.8911 | 0.1839 | 0.0724 | 0.0064 | 0.1729 | 0.1771 |
| 10000 | 0.6641 | 0.8931 | 0.1797 | 0.0540 | 0.0016 | 0.1703 | 0.1742 |
| 12500 | 0.6969 | 0.9017 | 0.1695 | 0.0341 | 0.0034 | 0.1682 | 0.1749 |
| 15000 | 0.6194 | 0.9038 | 0.1677 | 0.0448 | 0.0060 | 0.1515 | 0.1408 |
| 17500 | 0.6235 | 0.9097 | 0.1608 | 0.0898 | 0.0024 | 0.1591 | 0.1614 |
| 20000 | 0.5513 | 0.9196 | 0.1499 | 0.1080 | 0.0102 | 0.1435 | 0.1427 |
| 22500 | 0.6696 | 0.9199 | 0.1500 | 0.0454 | 0.0000 | 0.1579 | 0.1670 |
| 25000 | 0.6521 | 0.9217 | 0.1483 | 0.0424 | 0.0000 | 0.1569 | 0.1674 |
| 27500 | 0.7603 | 0.9192 | 0.1479 | 0.0332 | 0.0000 | 0.1789 | 0.1996 |
| 30000 | 0.6003 | 0.9303 | 0.1381 | 0.0738 | 0.0018 | 0.1441 | 0.1491 |
| 32500 | 0.5439 | 0.9321 | 0.1364 | 0.1024 | 0.0000 | 0.1356 | 0.1364 |
| 35000 | 0.6086 | 0.9320 | 0.1346 | 0.0873 | 0.0026 | 0.1446 | 0.1541 |
| 37500 | 0.6559 | 0.9347 | 0.1322 | 0.2021 | 0.0138 | 0.1474 | 0.1600 |
| 40000 | 0.6356 | 0.9352 | 0.1318 | 0.1749 | 0.0171 | 0.1425 | 0.1519 |
| 42500 | 0.6034 | 0.9368 | 0.1302 | 0.1397 | 0.0098 | 0.1393 | 0.1467 |
| 45000 | 0.6285 | 0.9370 | 0.1299 | 0.1724 | 0.0165 | 0.1421 | 0.1525 |
| 47500 | 0.6578 | 0.9372 | 0.1291 | 0.0235 | 0.0235 | 0.1439 | 0.1552 |
| 50000 | 0.6525 | 0.9376 | 0.1286 | 0.0229 | 0.0229 | 0.1433 | 0.1545 |

**v3** — v3-musan-zhen3m-50k (12 validation points)

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
| 48000 | 0.3934 | 0.9316 | 0.1322 | 0.1546 | 0.0136 | 0.1059 | 0.0787 |

**v4** — v4-musan-zhen3m-50k (16 validation points)

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
| 48000 | 0.4065 | 0.9323 | 0.1338 | 0.1568 | 0.0159 | 0.1088 | 0.0858 |

**v4.1** — v41-musan-zhen3m-50k (18 validation points)

| step | val_loss | AUC | EER | TPR@1e-2 | TPR@1e-3 | Brier | ECE |
| --- | --- | --- | --- | --- | --- | --- | --- |
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
| 47500 | 0.4109 | 0.9350 | 0.1311 | 0.1832 | 0.0199 | 0.1088 | 0.0873 |
| 50000 | 0.4173 | 0.9348 | 0.1313 | 0.1822 | 0.0205 | 0.1098 | 0.0895 |
### 3.3 Reading

v2 leads from ~15 k steps on and finishes 0.0026 AUC ahead of v4.1, 0.0053 ahead of v4 and 0.0060
ahead of v3, with a clearly better low-FPR tail (TPR@1%FPR 0.2021 vs 0.1832 / 0.1703 / 0.1568). v4.1
beats v4/v3 only in that tail while costing 2.3× the wall time; v3 ≈ v4 (final AUC within 0.0007).

## 4. Readout ladder — GigaSpeech KWS finetune encoder (streaming 16/64)

### 4.1 Aligned milestones

| step | v2 AUC (EER) | v3 AUC (EER) | v4 AUC (EER) | v4.1 AUC (EER) |
|---|---|---|---|---|
| 5000 | 0.6830 (0.3733) | 0.6320 (0.4094) | 0.5964 (0.4387) | 0.6849 (0.3707) |
| 10000 | 0.7965 (0.2771) | 0.7567 (0.3100) | 0.7720 (0.3024) | 0.7805 (0.2912) |
| 20000 | 0.8348 (0.2400) | 0.8219 (0.2543) | 0.8247 (0.2500) | 0.8352 (0.2418) |
| 30000 | 0.8657 (0.2096) | 0.8468 (0.2293) | 0.8363 (0.2386) | 0.8559 (0.2188) |
| 40000 | 0.8773 (0.1999) | 0.8560 (0.2205) | 0.8531 (0.2232) | 0.8652 (0.2114) |
| 48000 | 0.8809 (0.1958) | 0.8598 (0.2167) | 0.8565 (0.2196) | 0.8678 (0.2091) |
| best step | 48000 | 48000 | 48000 | 45000 |
| **best AUC** | **0.880924** | **0.859759** | **0.856506** | **0.868674** |
| EER @ best | 0.195756 | 0.216651 | 0.219640 | 0.207674 |
| TPR@1%FPR | 0.089869 | 0.077079 | 0.073658 | 0.081423 |
| TPR@0.1%FPR | 0.009953 | 0.012531 | 0.010307 | 0.010869 |
| pAUC(<=1%) | 0.522315 | 0.517957 | 0.517027 | 0.518470 |
| Brier | 0.188063 | 0.156922 | 0.158888 | 0.153237 |
| wall time | 3.03 h | 3.57 h | 3.85 h | 8.61 h |

### 4.2 Full validation curves

**v2** — v2-musan-gs-stream-50k (16 validation points)

| step | val_loss | AUC | EER | TPR@1e-2 | TPR@1e-3 | Brier | ECE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 3000 | 0.7328 | 0.6830 | 0.3733 | 0.0228 | 0.0033 | 0.2557 | 0.1743 |
| 6000 | 0.7984 | 0.7580 | 0.3122 | 0.0312 | 0.0030 | 0.2526 | 0.2175 |
| 9000 | 0.8342 | 0.7965 | 0.2771 | 0.0429 | 0.0025 | 0.2501 | 0.2371 |
| 12000 | 0.7439 | 0.8258 | 0.2503 | 0.0495 | 0.0040 | 0.2228 | 0.2068 |
| 15000 | 0.7696 | 0.8349 | 0.2421 | 0.0575 | 0.0049 | 0.2271 | 0.2255 |
| 18000 | 0.8138 | 0.8348 | 0.2400 | 0.0475 | 0.0037 | 0.2310 | 0.2264 |
| 21000 | 0.7158 | 0.8431 | 0.2344 | 0.0598 | 0.0050 | 0.2052 | 0.1844 |
| 24000 | 0.7993 | 0.8546 | 0.2188 | 0.0534 | 0.0028 | 0.2201 | 0.2230 |
| 27000 | 0.8256 | 0.8588 | 0.2189 | 0.0709 | 0.0009 | 0.2207 | 0.2266 |
| 30000 | 0.7514 | 0.8657 | 0.2096 | 0.0712 | 0.0082 | 0.2039 | 0.2000 |
| 33000 | 0.7472 | 0.8729 | 0.2042 | 0.0568 | 0.0100 | 0.1963 | 0.1920 |
| 36000 | 0.6896 | 0.8752 | 0.2014 | 0.0899 | 0.0027 | 0.1881 | 0.1770 |
| 39000 | 0.7241 | 0.8773 | 0.1999 | 0.0783 | 0.0076 | 0.1919 | 0.1882 |
| 42000 | 0.7425 | 0.8781 | 0.1984 | 0.0833 | 0.0093 | 0.1967 | 0.1989 |
| 45000 | 0.7415 | 0.8804 | 0.1965 | 0.0889 | 0.0002 | 0.1952 | 0.1964 |
| 48000 | 0.7517 | 0.8809 | 0.1958 | 0.0717 | 0.0003 | 0.1963 | 0.1986 |

**v3** — v3-musan-gs-50k (12 validation points)

| step | val_loss | AUC | EER | TPR@1e-2 | TPR@1e-3 | Brier | ECE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 4000 | 0.6943 | 0.6320 | 0.4094 | 0.0149 | 0.0018 | 0.2515 | 0.1357 |
| 8000 | 0.6294 | 0.7567 | 0.3100 | 0.0438 | 0.0043 | 0.2179 | 0.1147 |
| 12000 | 0.5782 | 0.7807 | 0.2913 | 0.0484 | 0.0064 | 0.1972 | 0.0734 |
| 16000 | 0.5473 | 0.8129 | 0.2612 | 0.0534 | 0.0064 | 0.1823 | 0.0731 |
| 20000 | 0.5569 | 0.8219 | 0.2543 | 0.0603 | 0.0074 | 0.1805 | 0.0946 |
| 24000 | 0.5346 | 0.8390 | 0.2371 | 0.0658 | 0.0089 | 0.1704 | 0.0907 |
| 28000 | 0.5449 | 0.8468 | 0.2293 | 0.0743 | 0.0109 | 0.1684 | 0.1010 |
| 32000 | 0.5337 | 0.8478 | 0.2280 | 0.0719 | 0.0086 | 0.1659 | 0.0934 |
| 36000 | 0.5091 | 0.8537 | 0.2233 | 0.0771 | 0.0125 | 0.1598 | 0.0804 |
| 40000 | 0.5179 | 0.8560 | 0.2205 | 0.0755 | 0.0110 | 0.1611 | 0.0937 |
| 44000 | 0.5038 | 0.8591 | 0.2177 | 0.0751 | 0.0111 | 0.1570 | 0.0837 |
| 48000 | 0.5059 | 0.8598 | 0.2167 | 0.0770 | 0.0121 | 0.1569 | 0.0847 |

**v4** — v4-musan-gs-50k (16 validation points)

| step | val_loss | AUC | EER | TPR@1e-2 | TPR@1e-3 | Brier | ECE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 3000 | 0.7272 | 0.5964 | 0.4387 | 0.0138 | 0.0016 | 0.2669 | 0.1588 |
| 6000 | 0.6335 | 0.7267 | 0.3357 | 0.0280 | 0.0025 | 0.2245 | 0.1123 |
| 9000 | 0.6355 | 0.7720 | 0.3024 | 0.0479 | 0.0055 | 0.2232 | 0.1466 |
| 12000 | 0.5870 | 0.7815 | 0.2898 | 0.0439 | 0.0050 | 0.1973 | 0.0892 |
| 15000 | 0.5660 | 0.8034 | 0.2700 | 0.0523 | 0.0056 | 0.1870 | 0.0835 |
| 18000 | 0.5166 | 0.8247 | 0.2500 | 0.0581 | 0.0062 | 0.1710 | 0.0464 |
| 21000 | 0.5475 | 0.8213 | 0.2558 | 0.0665 | 0.0075 | 0.1787 | 0.0845 |
| 24000 | 0.5500 | 0.8313 | 0.2460 | 0.0608 | 0.0069 | 0.1784 | 0.0946 |
| 27000 | 0.5204 | 0.8350 | 0.2405 | 0.0619 | 0.0084 | 0.1679 | 0.0703 |
| 30000 | 0.5277 | 0.8363 | 0.2386 | 0.0654 | 0.0088 | 0.1689 | 0.0780 |
| 33000 | 0.5468 | 0.8408 | 0.2347 | 0.0609 | 0.0084 | 0.1714 | 0.1011 |
| 36000 | 0.5345 | 0.8504 | 0.2261 | 0.0715 | 0.0093 | 0.1649 | 0.0930 |
| 39000 | 0.5214 | 0.8531 | 0.2232 | 0.0708 | 0.0103 | 0.1612 | 0.0868 |
| 42000 | 0.5196 | 0.8527 | 0.2229 | 0.0714 | 0.0094 | 0.1612 | 0.0856 |
| 45000 | 0.5124 | 0.8562 | 0.2197 | 0.0732 | 0.0103 | 0.1589 | 0.0846 |
| 48000 | 0.5218 | 0.8565 | 0.2196 | 0.0737 | 0.0103 | 0.1604 | 0.0923 |

**v4.1** — v41-musan-gs-50k (20 validation points)

| step | val_loss | AUC | EER | TPR@1e-2 | TPR@1e-3 | Brier | ECE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2500 | 0.7290 | 0.5269 | 0.4838 | 0.0106 | 0.0006 | 0.2661 | 0.1307 |
| 5000 | 0.6646 | 0.6849 | 0.3707 | 0.0256 | 0.0025 | 0.2363 | 0.1119 |
| 7500 | 0.5916 | 0.7548 | 0.3135 | 0.0396 | 0.0044 | 0.2029 | 0.0537 |
| 10000 | 0.6002 | 0.7805 | 0.2912 | 0.0454 | 0.0057 | 0.2069 | 0.1079 |
| 12500 | 0.5460 | 0.7945 | 0.2781 | 0.0390 | 0.0043 | 0.1841 | 0.0455 |
| 15000 | 0.5572 | 0.8169 | 0.2606 | 0.0608 | 0.0059 | 0.1824 | 0.0923 |
| 17500 | 0.5360 | 0.8279 | 0.2464 | 0.0538 | 0.0081 | 0.1751 | 0.0786 |
| 20000 | 0.5159 | 0.8352 | 0.2418 | 0.0512 | 0.0070 | 0.1678 | 0.0699 |
| 22500 | 0.5039 | 0.8470 | 0.2278 | 0.0644 | 0.0088 | 0.1612 | 0.0695 |
| 25000 | 0.5307 | 0.8458 | 0.2314 | 0.0753 | 0.0101 | 0.1694 | 0.0891 |
| 27500 | 0.5755 | 0.8465 | 0.2306 | 0.0610 | 0.0081 | 0.1723 | 0.1163 |
| 30000 | 0.5286 | 0.8559 | 0.2188 | 0.0693 | 0.0077 | 0.1613 | 0.0912 |
| 32500 | 0.5427 | 0.8576 | 0.2178 | 0.0686 | 0.0088 | 0.1624 | 0.1050 |
| 35000 | 0.5497 | 0.8590 | 0.2181 | 0.0704 | 0.0090 | 0.1644 | 0.1127 |
| 37500 | 0.5191 | 0.8646 | 0.2134 | 0.0778 | 0.0096 | 0.1573 | 0.0974 |
| 40000 | 0.5120 | 0.8652 | 0.2114 | 0.0808 | 0.0107 | 0.1558 | 0.0947 |
| 42500 | 0.5196 | 0.8666 | 0.2113 | 0.0755 | 0.0097 | 0.1569 | 0.1003 |
| 45000 | 0.5058 | 0.8687 | 0.2077 | 0.0803 | 0.0109 | 0.1532 | 0.0917 |
| 47500 | 0.5136 | 0.8678 | 0.2091 | 0.0810 | 0.0101 | 0.1549 | 0.0956 |
| 50000 | 0.5151 | 0.8683 | 0.2084 | 0.0814 | 0.0105 | 0.1548 | 0.0970 |
### 4.3 Reading

Everything converges 0.05–0.07 AUC below its zh-en counterpart. v2 is +0.012 over v4.1 and +0.024
over v4. The curves are still creeping up at 48–50 k (v4.1: 0.8652 at 40 k to 0.8687 at 45 k), so
this encoder is undertrained — but the size of the gap points at an encoder / pretraining mismatch
rather than a step-count issue.

## 5. GigaSpeech KWS base encoder — two operating points

The base checkpoint is causal (streaming-trained) but can be run at icefall's full-context point:
`stage1.stream.chunk_size = -1` / `left_context_frames = -1` makes `Zipformer2.get_chunk_info()`
return (-1, -1) (no chunk mask, full left context) and `ChunkCausalDepthwiseConv1d` treats the whole
sequence as one chunk. Flipping `stage1.causal` to false instead swaps in a plain `nn.Conv1d` and the
causal weights stop loading (160 missing / 64 unexpected), so the operating point — not the flag —
is the lever.

### 5.1 Full context (-1/-1)

| step | v3 AUC (EER) | v4 AUC (EER) | v4.1 AUC (EER) |
|---|---|---|---|
| 5000 | 0.6405 (0.4094) | 0.6170 (0.4278) | 0.7676 (0.3071) |
| 10000 | 0.8061 (0.2707) | 0.8364 (0.2373) | 0.8325 (0.2433) |
| 20000 | 0.8868 (0.1864) | 0.8881 (0.1844) | 0.8832 (0.1903) |
| 30000 | 0.9020 (0.1672) | 0.9076 (0.1614) | 0.9024 (0.1691) |
| 40000 | 0.9118 (0.1597) | 0.9169 (0.1517) | 0.9104 (0.1595) |
| 48000 | 0.9136 (0.1572) | 0.9195 (0.1479) | 0.9119 (0.1566) |
| best step | 48000 | 48000 | 45000 |
| **best AUC** | **0.913611** | **0.919524** | **0.912886** |
| EER @ best | 0.157224 | 0.147866 | 0.156308 |
| TPR@1%FPR | 0.125918 | 0.130580 | 0.122231 |
| TPR@0.1%FPR | 0.017282 | 0.018110 | 0.013329 |
| pAUC(<=1%) | 0.531374 | 0.532495 | 0.530486 |
| Brier | 0.128847 | 0.121307 | 0.126457 |
| wall time | 3.88 h | 4.15 h | 9.76 h |

### 5.2 Streaming 16/64

| step | v2 AUC (EER) | v3 AUC (EER) | v4 AUC (EER) | v4.1 AUC (EER) |
|---|---|---|---|---|
| 5000 | 0.7482 (0.3210) | 0.6806 (0.3777) | 0.6099 (0.4266) | 0.7488 (0.3208) |
| 10000 | 0.8576 (0.2189) | 0.8180 (0.2557) | 0.8329 (0.2425) | 0.8354 (0.2421) |
| 20000 | 0.8963 (0.1746) | 0.8759 (0.1968) | 0.8829 (0.1918) | 0.8847 (0.1888) |
| 30000 | 0.9200 (0.1493) | 0.8926 (0.1776) | 0.9008 (0.1727) | 0.9011 (0.1717) |
| 40000 | 0.9254 (0.1423) | 0.9031 (0.1662) | 0.9111 (0.1593) | 0.9093 (0.1627) |
| 48000 | 0.9271 (0.1411) | 0.9064 (0.1630) | 0.9131 (0.1570) | 0.9114 (0.1605) |
| best step | 48000 | 48000 | 48000 | 45000 |
| **best AUC** | **0.927132** | **0.906355** | **0.913148** | **0.911919** |
| EER @ best | 0.141135 | 0.163002 | 0.156969 | 0.159500 |
| TPR@1%FPR | 0.136883 | 0.105111 | 0.127388 | 0.124721 |
| TPR@0.1%FPR | 0.012745 | 0.013965 | 0.014393 | 0.015383 |
| pAUC(<=1%) | 0.536057 | 0.524005 | 0.530762 | 0.529277 |
| Brier | 0.147952 | 0.131689 | 0.125535 | 0.128110 |
| wall time | 2.82 h | 3.51 h | 3.78 h | 8.29 h |

### 5.3 Full validation curves

**v3** — v3-musan-gsbase-fullctx-50k (12 validation points)

| step | val_loss | AUC | EER | TPR@1e-2 | TPR@1e-3 | Brier | ECE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 4000 | 0.6888 | 0.6405 | 0.4094 | 0.0177 | 0.0018 | 0.2457 | 0.1371 |
| 8000 | 0.5830 | 0.8061 | 0.2707 | 0.0546 | 0.0070 | 0.1964 | 0.1144 |
| 12000 | 0.4927 | 0.8506 | 0.2246 | 0.0676 | 0.0086 | 0.1585 | 0.0664 |
| 16000 | 0.4523 | 0.8816 | 0.1916 | 0.0970 | 0.0134 | 0.1452 | 0.0846 |
| 20000 | 0.4682 | 0.8868 | 0.1864 | 0.0913 | 0.0086 | 0.1414 | 0.0885 |
| 24000 | 0.4704 | 0.8949 | 0.1791 | 0.1013 | 0.0142 | 0.1362 | 0.0921 |
| 28000 | 0.4482 | 0.9020 | 0.1672 | 0.1037 | 0.0122 | 0.1294 | 0.0884 |
| 32000 | 0.4582 | 0.9082 | 0.1613 | 0.1143 | 0.0150 | 0.1288 | 0.0963 |
| 36000 | 0.4983 | 0.9099 | 0.1602 | 0.1162 | 0.0145 | 0.1337 | 0.1110 |
| 40000 | 0.4957 | 0.9118 | 0.1597 | 0.1224 | 0.0165 | 0.1376 | 0.1193 |
| 44000 | 0.5156 | 0.9120 | 0.1583 | 0.1190 | 0.0157 | 0.1367 | 0.1187 |
| 48000 | 0.4772 | 0.9136 | 0.1572 | 0.1259 | 0.0173 | 0.1295 | 0.1051 |

**v4** — v4-musan-gsbase-fullctx-50k (16 validation points)

| step | val_loss | AUC | EER | TPR@1e-2 | TPR@1e-3 | Brier | ECE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 3000 | 0.7130 | 0.6170 | 0.4278 | 0.0144 | 0.0015 | 0.2613 | 0.1589 |
| 6000 | 0.5798 | 0.8048 | 0.2714 | 0.0482 | 0.0075 | 0.2001 | 0.1308 |
| 9000 | 0.5562 | 0.8364 | 0.2373 | 0.0730 | 0.0102 | 0.1740 | 0.1029 |
| 12000 | 0.5231 | 0.8666 | 0.2093 | 0.0837 | 0.0132 | 0.1595 | 0.1079 |
| 15000 | 0.4585 | 0.8796 | 0.1940 | 0.0789 | 0.0081 | 0.1421 | 0.0763 |
| 18000 | 0.4672 | 0.8881 | 0.1844 | 0.0879 | 0.0103 | 0.1392 | 0.0893 |
| 21000 | 0.5077 | 0.8912 | 0.1831 | 0.1040 | 0.0112 | 0.1458 | 0.1117 |
| 24000 | 0.4143 | 0.8974 | 0.1755 | 0.0987 | 0.0098 | 0.1276 | 0.0564 |
| 27000 | 0.5713 | 0.8979 | 0.1724 | 0.0943 | 0.0139 | 0.1586 | 0.1500 |
| 30000 | 0.4147 | 0.9076 | 0.1614 | 0.1153 | 0.0157 | 0.1229 | 0.0732 |
| 33000 | 0.5055 | 0.9114 | 0.1573 | 0.1254 | 0.0137 | 0.1400 | 0.1266 |
| 36000 | 0.4590 | 0.9160 | 0.1525 | 0.1255 | 0.0170 | 0.1276 | 0.1051 |
| 39000 | 0.4880 | 0.9169 | 0.1517 | 0.1263 | 0.0177 | 0.1297 | 0.1113 |
| 42000 | 0.4427 | 0.9172 | 0.1509 | 0.1287 | 0.0178 | 0.1219 | 0.0955 |
| 45000 | 0.4453 | 0.9193 | 0.1486 | 0.1295 | 0.0161 | 0.1213 | 0.0971 |
| 48000 | 0.4557 | 0.9195 | 0.1479 | 0.1306 | 0.0181 | 0.1230 | 0.1017 |

**v4.1** — v41-musan-gsbase-fullctx-50k (20 validation points)

| step | val_loss | AUC | EER | TPR@1e-2 | TPR@1e-3 | Brier | ECE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2500 | 0.7052 | 0.5493 | 0.4759 | 0.0074 | 0.0010 | 0.2565 | 0.1278 |
| 5000 | 0.5841 | 0.7676 | 0.3071 | 0.0347 | 0.0028 | 0.2037 | 0.0928 |
| 7500 | 0.5310 | 0.8227 | 0.2537 | 0.0632 | 0.0076 | 0.1755 | 0.0738 |
| 10000 | 0.5179 | 0.8325 | 0.2433 | 0.0543 | 0.0061 | 0.1715 | 0.0693 |
| 12500 | 0.5028 | 0.8646 | 0.2116 | 0.0857 | 0.0098 | 0.1636 | 0.1107 |
| 15000 | 0.4688 | 0.8719 | 0.2035 | 0.0759 | 0.0081 | 0.1452 | 0.0675 |
| 17500 | 0.4742 | 0.8785 | 0.1962 | 0.0850 | 0.0123 | 0.1449 | 0.0839 |
| 20000 | 0.4364 | 0.8832 | 0.1903 | 0.0916 | 0.0103 | 0.1358 | 0.0583 |
| 22500 | 0.4455 | 0.8876 | 0.1849 | 0.0846 | 0.0110 | 0.1368 | 0.0771 |
| 25000 | 0.4682 | 0.8988 | 0.1741 | 0.1064 | 0.0122 | 0.1386 | 0.1013 |
| 27500 | 0.4550 | 0.9002 | 0.1710 | 0.1124 | 0.0123 | 0.1335 | 0.0915 |
| 30000 | 0.4502 | 0.9024 | 0.1691 | 0.1004 | 0.0106 | 0.1300 | 0.0886 |
| 32500 | 0.4644 | 0.9048 | 0.1655 | 0.1128 | 0.0100 | 0.1312 | 0.0969 |
| 35000 | 0.4805 | 0.9074 | 0.1630 | 0.1174 | 0.0124 | 0.1338 | 0.1068 |
| 37500 | 0.5196 | 0.9033 | 0.1676 | 0.1035 | 0.0093 | 0.1396 | 0.1157 |
| 40000 | 0.4492 | 0.9104 | 0.1595 | 0.1222 | 0.0132 | 0.1265 | 0.0922 |
| 42500 | 0.4754 | 0.9113 | 0.1587 | 0.1203 | 0.0120 | 0.1307 | 0.1047 |
| 45000 | 0.4650 | 0.9129 | 0.1563 | 0.1222 | 0.0133 | 0.1283 | 0.1010 |
| 47500 | 0.4786 | 0.9119 | 0.1566 | 0.1207 | 0.0132 | 0.1301 | 0.1044 |
| 50000 | 0.4759 | 0.9121 | 0.1566 | 0.1208 | 0.0133 | 0.1296 | 0.1034 |

**v2** — v2-musan-gsbase-stream-50k (12 validation points)

| step | val_loss | AUC | EER | TPR@1e-2 | TPR@1e-3 | Brier | ECE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 4000 | 0.7456 | 0.7482 | 0.3210 | 0.0256 | 0.0025 | 0.2384 | 0.1822 |
| 8000 | 0.8305 | 0.8576 | 0.2189 | 0.0368 | 0.0045 | 0.2188 | 0.2260 |
| 12000 | 0.6891 | 0.8753 | 0.1979 | 0.0622 | 0.0025 | 0.1844 | 0.1769 |
| 16000 | 0.8097 | 0.8904 | 0.1828 | 0.0564 | 0.0092 | 0.2098 | 0.2281 |
| 20000 | 0.6567 | 0.8963 | 0.1746 | 0.0661 | 0.0007 | 0.1711 | 0.1704 |
| 24000 | 0.6676 | 0.9087 | 0.1596 | 0.0743 | 0.0003 | 0.1697 | 0.1801 |
| 28000 | 0.6704 | 0.9200 | 0.1493 | 0.1239 | 0.0000 | 0.1646 | 0.1744 |
| 32000 | 0.6946 | 0.9167 | 0.1492 | 0.0731 | 0.0031 | 0.1620 | 0.1716 |
| 36000 | 0.5877 | 0.9219 | 0.1471 | 0.1281 | 0.0014 | 0.1480 | 0.1499 |
| 40000 | 0.6643 | 0.9254 | 0.1423 | 0.1189 | 0.0085 | 0.1555 | 0.1675 |
| 44000 | 0.6978 | 0.9262 | 0.1420 | 0.1298 | 0.0127 | 0.1595 | 0.1736 |
| 48000 | 0.6850 | 0.9271 | 0.1411 | 0.1369 | 0.0112 | 0.1568 | 0.1689 |

**v3** — v3-musan-gsbase-stream-50k (12 validation points)

| step | val_loss | AUC | EER | TPR@1e-2 | TPR@1e-3 | Brier | ECE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 4000 | 0.6845 | 0.6806 | 0.3777 | 0.0220 | 0.0021 | 0.2440 | 0.1545 |
| 8000 | 0.5458 | 0.8180 | 0.2557 | 0.0575 | 0.0069 | 0.1810 | 0.0817 |
| 12000 | 0.5155 | 0.8418 | 0.2348 | 0.0600 | 0.0071 | 0.1673 | 0.0792 |
| 16000 | 0.4624 | 0.8713 | 0.2008 | 0.0710 | 0.0063 | 0.1469 | 0.0748 |
| 20000 | 0.4941 | 0.8759 | 0.1968 | 0.0735 | 0.0095 | 0.1493 | 0.0963 |
| 24000 | 0.4663 | 0.8851 | 0.1877 | 0.0885 | 0.0081 | 0.1409 | 0.0861 |
| 28000 | 0.4937 | 0.8926 | 0.1776 | 0.0925 | 0.0083 | 0.1395 | 0.1018 |
| 32000 | 0.4788 | 0.8975 | 0.1724 | 0.0996 | 0.0119 | 0.1404 | 0.1060 |
| 36000 | 0.4564 | 0.9031 | 0.1680 | 0.1051 | 0.0140 | 0.1317 | 0.0946 |
| 40000 | 0.4795 | 0.9031 | 0.1662 | 0.1029 | 0.0124 | 0.1364 | 0.1083 |
| 44000 | 0.4854 | 0.9050 | 0.1641 | 0.1018 | 0.0115 | 0.1352 | 0.1079 |
| 48000 | 0.4618 | 0.9064 | 0.1630 | 0.1033 | 0.0125 | 0.1317 | 0.1000 |

**v4** — v4-musan-gsbase-stream-50k (16 validation points)

| step | val_loss | AUC | EER | TPR@1e-2 | TPR@1e-3 | Brier | ECE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 3000 | 0.7159 | 0.6099 | 0.4266 | 0.0139 | 0.0012 | 0.2618 | 0.1408 |
| 6000 | 0.5763 | 0.8034 | 0.2727 | 0.0462 | 0.0040 | 0.1987 | 0.1172 |
| 9000 | 0.5416 | 0.8329 | 0.2425 | 0.0637 | 0.0073 | 0.1740 | 0.0939 |
| 12000 | 0.5069 | 0.8528 | 0.2242 | 0.0635 | 0.0095 | 0.1606 | 0.0861 |
| 15000 | 0.4874 | 0.8741 | 0.2008 | 0.0912 | 0.0133 | 0.1517 | 0.0930 |
| 18000 | 0.4923 | 0.8829 | 0.1918 | 0.0864 | 0.0103 | 0.1476 | 0.1023 |
| 21000 | 0.4724 | 0.8838 | 0.1923 | 0.0945 | 0.0098 | 0.1463 | 0.0983 |
| 24000 | 0.4448 | 0.8934 | 0.1795 | 0.0953 | 0.0120 | 0.1334 | 0.0745 |
| 27000 | 0.4998 | 0.8952 | 0.1773 | 0.1013 | 0.0115 | 0.1445 | 0.1131 |
| 30000 | 0.4409 | 0.9008 | 0.1727 | 0.1191 | 0.0134 | 0.1310 | 0.0828 |
| 33000 | 0.4741 | 0.9038 | 0.1664 | 0.1127 | 0.0115 | 0.1397 | 0.1123 |
| 36000 | 0.4969 | 0.9038 | 0.1673 | 0.1122 | 0.0121 | 0.1370 | 0.1081 |
| 39000 | 0.4719 | 0.9111 | 0.1593 | 0.1274 | 0.0129 | 0.1293 | 0.1002 |
| 42000 | 0.4461 | 0.9112 | 0.1584 | 0.1153 | 0.0138 | 0.1255 | 0.0904 |
| 45000 | 0.4592 | 0.9111 | 0.1586 | 0.1196 | 0.0139 | 0.1289 | 0.0995 |
| 48000 | 0.4603 | 0.9131 | 0.1570 | 0.1224 | 0.0144 | 0.1285 | 0.1009 |

**v4.1** — v41-musan-gsbase-stream-50k (20 validation points)

| step | val_loss | AUC | EER | TPR@1e-2 | TPR@1e-3 | Brier | ECE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2500 | 0.7173 | 0.5664 | 0.4590 | 0.0132 | 0.0014 | 0.2617 | 0.1254 |
| 5000 | 0.6043 | 0.7488 | 0.3208 | 0.0338 | 0.0032 | 0.2123 | 0.1040 |
| 7500 | 0.5407 | 0.8137 | 0.2600 | 0.0541 | 0.0065 | 0.1796 | 0.0773 |
| 10000 | 0.5056 | 0.8354 | 0.2421 | 0.0597 | 0.0064 | 0.1668 | 0.0583 |
| 12500 | 0.5094 | 0.8532 | 0.2237 | 0.0704 | 0.0073 | 0.1635 | 0.0951 |
| 15000 | 0.4905 | 0.8660 | 0.2117 | 0.0714 | 0.0066 | 0.1533 | 0.0861 |
| 17500 | 0.4496 | 0.8756 | 0.1985 | 0.0801 | 0.0069 | 0.1411 | 0.0595 |
| 20000 | 0.4849 | 0.8847 | 0.1888 | 0.0925 | 0.0127 | 0.1476 | 0.1027 |
| 22500 | 0.4455 | 0.8937 | 0.1822 | 0.0952 | 0.0104 | 0.1344 | 0.0798 |
| 25000 | 0.4456 | 0.8970 | 0.1776 | 0.1143 | 0.0120 | 0.1322 | 0.0790 |
| 27500 | 0.4714 | 0.8989 | 0.1729 | 0.1084 | 0.0134 | 0.1348 | 0.0961 |
| 30000 | 0.4741 | 0.9011 | 0.1717 | 0.1039 | 0.0096 | 0.1391 | 0.1082 |
| 32500 | 0.4620 | 0.9046 | 0.1680 | 0.1125 | 0.0121 | 0.1324 | 0.0979 |
| 35000 | 0.4914 | 0.9076 | 0.1635 | 0.1238 | 0.0144 | 0.1335 | 0.1056 |
| 37500 | 0.4563 | 0.9087 | 0.1649 | 0.1247 | 0.0115 | 0.1295 | 0.0957 |
| 40000 | 0.4612 | 0.9093 | 0.1627 | 0.1245 | 0.0144 | 0.1306 | 0.0997 |
| 42500 | 0.4632 | 0.9106 | 0.1616 | 0.1235 | 0.0154 | 0.1296 | 0.1001 |
| 45000 | 0.4625 | 0.9119 | 0.1595 | 0.1232 | 0.0141 | 0.1281 | 0.0993 |
| 47500 | 0.4741 | 0.9114 | 0.1605 | 0.1232 | 0.0151 | 0.1309 | 0.1046 |
| 50000 | 0.4757 | 0.9116 | 0.1600 | 0.1235 | 0.0152 | 0.1309 | 0.1050 |
### 5.4 Reading

Full context buys +0.0064 AUC / −0.0091 EER for v4 and is neutral-to-slightly-positive for v3/v4.1.
v2 (streaming only) still beats every full-context run. The base checkpoint lands ~0.035 AUC below
zh-en-3M and ~0.05 above the finetune checkpoint on the same readout.

## 6. Throughput, concurrency and data-prep lessons

### 6.1 Batch size and workers (v4.1 config, 30–40 steps per point, steady state)

| batch | workers | steady state | samples/s |
| --- | --- | --- | --- |
| 128 | 4 | 2.30 steps/s | 294 |
| 128 | 8 | 2.32 steps/s | 297 |
| 256 | 8 | 1.16 steps/s | 297 |
| 512 | 8 / 10 | 0.57 steps/s | 279–292 |
| 1024 | 10 | 0.27 steps/s | 276 |

Throughput is flat (~276–300 samples/s): the V100 saturates at batch 128 and larger batches only
lengthen the step. Batch 256 OOMs for the v7 keyword-filler readout (31.7 GiB requested).

### 6.2 Per-step cost by readout (128-dim encoder output, batch 128, fp16)

| readout | steady state | per step | GPU util | VRAM |
| --- | --- | --- | --- | --- |
| v2 `gru_last` | ~5.3 steps/s | ~0.19 s | ~100 % (co-located) | ~2 GB |
| v3 `eps_mean` | ~5.5 steps/s | ~0.18 s | ~99 % (co-located) | ~2 GB |
| v4 `eps_softmin` | ~6.2 steps/s | ~0.16 s | ~99 % (co-located) | ~2 GB |
| v4.1 (+ sink token, learned text / relative-bias audio positions) | 2.28 steps/s | 0.44 s | 88 % solo | 2.1 GB |
| v7 keyword-filler | 0.26–0.29 steps/s | 3.4–3.9 s | 20–25 % | 15–25 GB |

### 6.3 Running several readouts at once

| scenario | heavy run | light runs | aggregate | GPU util |
| --- | --- | --- | --- | --- |
| v4.1 alone | 2.28 steps/s | — | 2.28 steps/s | 87.9 % |
| v4.1 + v4 | 1.58 | 6.23 | 7.82 steps/s | 96.8 % |
| v4.1 + v4 + v3 | 1.27 | 4.89 / 4.93 | **11.09 steps/s** | **99.0 %** |
| after the light runs finish | 2.09 | done | 2.09 steps/s | ~90 % |

Light + heavy co-location fills the GPU (88 % → 99 %) and multiplies aggregate throughput; two heavy
runs would just split the same saturated GPU. The heavy run pays for it (2.28 → 1.27 steps/s) but
recovers once the light runs finish. Three concurrent runs push CPU load to ~7–9 of 10 cores.

### 6.4 Validation cost

| measurement | value |
| --- | --- |
| full hard split inside a live run (2,115 batches, 8 eval workers) | **2 min 10 s** = 16.25 it/s = 0.061 s/batch |
| 50k-step run with `val_check_interval=2500` (20 validations) | ~44 min of validation |
| 2,048-row sample (16 batches), same workers | 31 s → looks like 1.94 s/batch — fixed start-up dominates a tiny sample, never extrapolate from it |

### 6.5 Data-prep lessons (2026-10-02/03)

* **The paper Stage I needs its own fbank tree.** Wenet's `compute_fbank` is `waveform * (1 << 15)`
  then `kaldi.fbank(snip_edges=True, low_freq=20, high_freq=0, povey)` — i.e. this repo's **default**
  profile, not the icefall one. Measured on the same clip: Wenet **(50, 80), mean +16.27** vs icefall
  **(52, 80), mean −4.52**. The trees are not interchangeable, and Stage II refuses to mix profiles
  (`ValueError: fbank config mismatch …` between the MUSAN cache and the training profile).
* **The training-fbank prep can need more than 25 GB RSS** (155,619 anchors × 100 hard negatives are
  held in memory). With three GPU runs resident it was **OOM-killed** at 25.3 GB
  (`oom-kill … task=python … anon-rss:25352044kB`). Fix: run it alone, add a swap guard (this host now
  has 16 GB + 32 GB swap files), or limit anchors.
* **Parallel prep needs a scratch output directory.** The prep rewrites the parquet / clips /
  distances of its output subdir; point `prep.output_subdir` at a scratch dir while a run is reading
  the real one and share only the feature tree.
* **The data disk was grown online** from 503 GB to **787 GB** (`sudo resize2fs /dev/vdb`; ext4 sits
  directly on the device, there is no partition table), which removed the need to delete the icefall
  fbank tree to make room for the Wenet one.

### 6.6 MUSAN training-set correction (2026-10-03)

The training background pool is now **music + noise only**. The earlier runs — the 15 completed
50k runs and the first (aborted) attempt at the paper Stage I trio — sampled music, noise **and
speech**, because the eligibility list and the catalog carried all three MUSAN categories and the
sampler has no category filter.

| item | before (speech-inclusive) | after (music + noise) |
| --- | --- | --- |
| catalog recordings | 2,016 (music 660 / noise 930 / speech 426) | **1,590** (music 660 / noise 930) |
| eligible train recordings | 993 | **774** |
| crops per recording (K) | 323 | **414** |
| crops in the cache | 320,739 | **320,436** |
| icefall-profile cache id | `f1ff880b…` | `2b83f85f30ff3af8…` |
| Wenet-profile cache id | (built later, also speech-inclusive) | `3b73024a08bbdbf4…` |

Procedure:

1. a new prepare config with `category_allow: [music, noise]`
   (`data/dma-kws/processed/bg_prepare/prepare_musan_music_noise.yaml`);
2. catalog + splits rebuilt (774 train / 93 val / 723 test, all eligible);
3. both fbank-profile caches rebuilt with K=414 — the tool's own suggestion for 774 sources, which
   keeps the crop-pool size equal to the old one (320k) — via
   `scripts/rebuild_musan_music_noise_cache.sh` (~21 min, two builders in parallel);
4. `--verify-only` on both: `ok: true`;
5. the old tree was deleted and the new one renamed into its place, so every experiment overlay
   keeps pointing at `processed/background/musan/{recordings.jsonl,cache,cache_wenet}`
   without edits.

The evaluation corpus is deliberately untouched: `processed/musan_split/eval_musan.list` (902
recordings) still covers all three categories, and the FA scripts read raw audio.

Consequence for this log: the 15 completed runs in §1–§5 were trained with the speech-inclusive
pool. Their numbers stand for that recipe; re-running them with the corrected pool is a separate
campaign that has **not** been started.

## 7. Aborted / superseded runs

| run | detail |
| --- | --- |
| v7 keyword-filler (zh-en) | `icefall_zipformer_stage2_zhen3m_50k` / `qbyt-zhen3m-50k`, version 7 `one_vs_rest` emission, 160 K trainable, no background noise. Stopped at 215 / 50,000 steps (3.4 s/step, GPU 20–25 %) when the plan switched to v4.1 + MUSAN backgrounds. |
| GS ASR XL encoder (3 runs) | `gigaspeech-asr-xl-2023-10-17/pretrained.pt` (the RESULTS.md "original" non-streaming ASR Zipformer: 63.99 M encoder params, non-causal, `chunking=off`, 0/0 load) was wired up, smoke-tested and then **stopped on request** — the intended encoder was the KWS `exp/pretrained.pt`. About 1.5 k steps per run were discarded; the `*_asrxl_50k` configs remain. |

## 8. Artefacts (exported best-val_AUC weights)

| run | checkpoint | file | size |
| --- | --- | --- | --- |
| zh-en-3M / v2 (streaming 16/64) | exp/stage2_qbyt/checkpoints/v2-musan-zhen3m-stream-50k/v2-musan-zhen3m-stream-50k/version_1/stage2_step050000.pt | stage2_step050000.pt | 15.9 MB |
| zh-en-3M / v3 (streaming 16/64) | exp/stage2_qbyt/checkpoints/v3-musan-zhen3m-50k/v3-musan-zhen3m-50k/version_1/stage2_step048000.pt | stage2_step048000.pt | 15.5 MB |
| zh-en-3M / v4 (streaming 16/64) | exp/stage2_qbyt/checkpoints/v4-musan-zhen3m-50k/v4-musan-zhen3m-50k/version_1/stage2_step048000.pt | stage2_step048000.pt | 15.5 MB |
| zh-en-3M / v4.1 (streaming 16/64) | exp/stage2_qbyt/checkpoints/v41-musan-zhen3m-50k/v41-musan-zhen3m-50k/version_9/stage2_step047500.pt | stage2_step047500.pt | 15.5 MB |
| GS-KWS finetune / v2 (streaming 16/64) | exp/stage2_qbyt/checkpoints/v2-musan-gs-stream-50k/v2-musan-gs-stream-50k/version_0/stage2_step048000.pt | stage2_step048000.pt | 16.0 MB |
| GS-KWS finetune / v3 (streaming 16/64) | exp/stage2_qbyt/checkpoints/v3-musan-gs-50k/v3-musan-gs-50k/version_0/stage2_step048000.pt | stage2_step048000.pt | 15.6 MB |
| GS-KWS finetune / v4 (streaming 16/64) | exp/stage2_qbyt/checkpoints/v4-musan-gs-50k/v4-musan-gs-50k/version_0/stage2_step048000.pt | stage2_step048000.pt | 15.6 MB |
| GS-KWS finetune / v4.1 (streaming 16/64) | exp/stage2_qbyt/checkpoints/v41-musan-gs-50k/v41-musan-gs-50k/version_0/stage2_step045000.pt | stage2_step045000.pt | 15.7 MB |
| GS-KWS base / v3 (full context -1/-1) | exp/stage2_qbyt/checkpoints/v3-musan-gsbase-fullctx-50k/v3-musan-gsbase-fullctx-50k/version_0/stage2_step048000.pt | stage2_step048000.pt | 15.6 MB |
| GS-KWS base / v4 (full context -1/-1) | exp/stage2_qbyt/checkpoints/v4-musan-gsbase-fullctx-50k/v4-musan-gsbase-fullctx-50k/version_0/stage2_step048000.pt | stage2_step048000.pt | 15.6 MB |
| GS-KWS base / v4.1 (full context -1/-1) | exp/stage2_qbyt/checkpoints/v41-musan-gsbase-fullctx-50k/v41-musan-gsbase-fullctx-50k/version_1/stage2_step045000.pt | stage2_step045000.pt | 15.7 MB |
| GS-KWS base / v2 (streaming 16/64) | exp/stage2_qbyt/checkpoints/v2-musan-gsbase-stream-50k/v2-musan-gsbase-stream-50k/version_0/stage2_step048000.pt | stage2_step048000.pt | 16.0 MB |
| GS-KWS base / v3 (streaming 16/64) | exp/stage2_qbyt/checkpoints/v3-musan-gsbase-stream-50k/v3-musan-gsbase-stream-50k/version_0/stage2_step048000.pt | stage2_step048000.pt | 15.6 MB |
| GS-KWS base / v4 (streaming 16/64) | exp/stage2_qbyt/checkpoints/v4-musan-gsbase-stream-50k/v4-musan-gsbase-stream-50k/version_0/stage2_step048000.pt | stage2_step048000.pt | 15.6 MB |
| GS-KWS base / v4.1 (streaming 16/64) | exp/stage2_qbyt/checkpoints/v41-musan-gsbase-stream-50k/v41-musan-gsbase-stream-50k/version_0/stage2_step045000.pt | stage2_step045000.pt | 15.7 MB |

Each `.pt` carries `model_state_dict`, `config`, `qbyt_readout`, `qbyt_readout_version`,
`tokenizer_dict_path`, `vocab_size`, `qbyt_alignment_spec` and `dma_kws_run_context`. Per-run logs live
under `data/dma-kws/exp/stage2_qbyt/logs/<run_name>/<run_name>/version_*/` (`metrics.csv`,
`eval_history.csv`, `hparams.yaml`, tfevents); one summary row per run is appended to `runs.csv`; tmux
pane transcripts are saved as `data/dma-kws/exp/stage2_qbyt/<session>.pane.log`.

## 9. Evaluation status

| evaluation | status |
| --- | --- |
| LibriPhrase hard-split AUC / EER / TPR — all 15 runs | ✅ done (§1–§5) |
| MUSAN held-out false alarms per 24 h (902 files / 43.72 h) for the zh-en v3/v4/v4.1 trio | ✅ 0 accepts at threshold 0.5 → 0 次/24 h; 1 s-window stress grid: v4.1 29.1 < v3 35.7 < v4 45.0 FA/24 h ([False-Alarm Evaluation](False-Alarm-Evaluation)) |
| LibriSpeech train-other-500 FA (stride-6 subset, 82.71 h) for the zh-en trio | ✅ 0 accepts at 0.5 → 0 次/24 h |
| FA for the 12 remaining checkpoints (MUSAN + LibriSpeech) | ✅ done 2026-10-03: **MUSAN held-out all 0 accepts → 0 次/24 h** (51,994 windows / 43.72 h); LibriSpeech train-other-500 stride-6 subset (87,243 windows / 82.71 h): **GS-finetune v2 = 1 accept → 0.29 次/24 h**, **GS-base full-context v4.1 = 1 accept → 0.29 次/24 h**, every other model 0 (incl. zh-en v2) |
| Two-stage (QbyT + verifier) end-to-end event rate | ⏳ planned |
| Paper Stage I Wenet Conformer (non-streaming) × v2/v3/v4/v4.1 | 🔄 training started 2026-10-03 08:14 UTC (encoder 0/0, policy `chunking=off`, Wenet fbank tree + the corrected music+noise Wenet MUSAN cache). v2 and v3 were relaunched at ~12 steps/s with v4 and v4.1 paused to give them the machine; the first attempt used the speech-inclusive cache and was discarded |

## 10. Reproducing a run

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

Experiment config families (all under `configs/experiment/`): `*_v2_*`, `*_v3_*`, `*_v4_*`,
`*_eps_softmin_v41_*` (readouts); `*_zhen3m_*`, `*_gs_*`, `*_gsbase_fullctx_*`, `*_gsbase_stream_*`
(encoder + operating point); `*_paperstage1_*` (paper Stage I conformer).

Chain watchers used here: `scripts/chain_gsbase_fullctx_to_stream.sh` (phase A → B),
`scripts/chain_phase_b_v2_paper_stage1.sh` (B → v2 → phase C), `scripts/prepare_wenet_fbank_parallel.sh`
and `scripts/prepare_and_run_paper_stage1.sh` (Wenet fbank + Wenet MUSAN cache → phase C).

## 11. Template for the next record

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