---
title: Stage II 背景 fbank 缓存：使用与验收记录
description: 裁剪级 fbank 缓存的创建、验证、训练切换、内存/磁盘预算，以及本机未完成的 V100 验证。
tags:
  - qbyt
  - stage2
  - performance
---
对应实现计划：[Grok 执行计划：v4.1 背景负样本预处理与加载优化](./2026-09-05-stage2-background-data-performance-grok.md)。

代码落在分支 `refactor/hydra-config-and-training`，HEAD `3e126c5`，基线 `49b15ca`。本机定向测试 183 passed + `compileall dma_kws scripts`。**没有推送、没有跑完整 10k 训练、没有 V100 A/B JSON。**

## 行为差异

| 路径 | 默认 | 说明 |
| --- | --- | --- |
| online | `mode: online` | 仍按 dataset RNG：uniform 时长 → 抽录音 → 寻址裁剪 → fbank。旧配置不写 `mode` 也走这条。 |
| fbank_cache | 显式 `mode: fbank_cache` | 先均匀抽 source，再均匀抽该 source 的 K 个预生成裁剪。不解码音频、不跑 FbankExtractor。 |
| disabled | `enabled: false` | 不打开音频、manifest 或分片。 |

正样本仍不被替换。背景仍在 50% 负样本分支内部门控，默认条件概率 0.25，`label=0`，`query_seq` 为空。v4.1 readout spec 不变。

有限裁剪库 **不等价** 无限在线随机裁剪：记录 `seed`、K、`cache_id`与重复率，不声称逐样本轨迹相同。

## 创建与验证

先有 MUSAN split（`scripts/split_musan.py --train-categories music,noise`）。构建器只消费 `train_background.list`，不重新划分。

```bash
.venv/bin/python scripts/prepare_stage2_background.py \
  --split-dir data/dma-kws/processed/musan_split \
  --output-dir data/dma-kws/processed/stage2_background_cache \
  --experiment icefall_zipformer_stage2_eps_softmin_v41 \
  --crops-per-recording 128 --seed 2025 --workers 4 --shard-size-mib 256
```

先用 synthetic split、K=4 验证。`output-dir` 已存在则拒绝覆盖。失败不会在该路径留下可训练识别的半成品。

```bash
.venv/bin/python scripts/prepare_stage2_background.py \
  --split-dir data/dma-kws/processed/musan_split \
  --output-dir data/dma-kws/processed/stage2_background_cache \
  --verify-only
```

`--verify-only` 对索引和全部分片做 SHA256 + shape/offset/finite。**训练启动只做轻量校验**（manifest 身份、recordings/index 摘要、分片头/大小、索引范围）。完整位损坏检测靠 `--verify-only`。

K=128 是工具默认值，不是训练质量结论。构建前会打印容量估计与建议 `max(128, ceil(2*160000/N))`，**不会自动覆盖** `--crops-per-recording`。

v1 只接受 composed fbank `dither=0`。

## 训练切换

原 v4.1 overlay 仍默认 online。缓存 overlay：

```bash
.venv/bin/python scripts/train_stage2_qbyt.py \
  +experiment=icefall_zipformer_stage2_eps_softmin_v41_cached
```

该 overlay 只改 `mode=fbank_cache`、`audio_list_path=""`、`cache_manifest=${paths.processed_root}/stage2_background_cache/manifest.json`，以及独立 `run_name` / log / checkpoint（`-cached` 后缀）。`qbyt_readout_version=4` 与 v4.1 spec 不变。

`audio_list_path` 非空时只对 manifest 的来源目录记录，**不要求原音频还在训练机上**。

元数据 LRU：`stage2.metadata_cache.max_entries=128`、`max_bytes=33554432`（每 worker）。`max_entries=0` 关闭。只缓存 clips/distances 文件内容和有界 phoneme token，**不缓存语音 fbank**。元数据文件在一次运行内视为不可变；改文件后重建 Dataset。

## 后续（未改格式）

`crops.npy` 的 `source_id` 目前是 `U1024`（每条约 4 KiB）。MUSAN music+noise × K=128 时，**仅索引就可能接近 0.8 GiB**。生产缓存构建前应改成只在 `recordings.jsonl` 存 `source_id`，或把 unicode 字段缩短。这是 v1.1 跟进，本轮未改 on-disk 合同。

## 内存与磁盘

- 80 维 / 10 ms / FP32 约 32 kB/秒；平均 2 秒、160k 条约 10 GB（粗估，含量取决于实际帧数）。
- 背景抽样次数 ≈ `optimizer_steps × accumulation × batch_per_gpu × world_size × 0.5 × probability`。默认单卡 10k/128/accum=1 约 160k 次。
- 每 worker 最多 `max_open_shards` 个 mmap（默认 8）。
- 构建进程池在途结果窗口等于 `workers`，不会把全库特征收回内存。

## 测速

三层次不要混成一个「加速倍数」。

```bash
.venv/bin/python scripts/benchmark_stage2_input.py \
  --mode background \
  --manifest data/dma-kws/processed/stage2_background_cache/manifest.json \
  --split-dir data/dma-kws/processed/musan_split \
  --output outputs/background-exact-crop.json

.venv/bin/python scripts/benchmark_stage2_input.py \
  --mode loader \
  --experiment icefall_zipformer_stage2_eps_softmin_v41 \
  --override stage2.num_workers=4 \
  --override stage2.dataloader.prefetch_factor=2 \
  --output outputs/input-online-w4-p2.json

.venv/bin/python scripts/benchmark_stage2_input.py \
  --mode loader \
  --experiment icefall_zipformer_stage2_eps_softmin_v41_cached \
  --override stage2.num_workers=4 \
  --override stage2.dataloader.prefetch_factor=2 \
  --output outputs/input-cached-w4-p2.json
```

train 模式需真实 encoder 初始化权重与 CUDA；缺少时会清楚退出，不会用随机 encoder 冒充吞吐。`--device cuda` 无 CUDA 同样退出。`gpu_util` 无 nvidia-smi 时为 `unavailable`，不写 0。

## 验收分级

- **工程完成：** T1–T5 实现、定向测试、配置兼容、CPU fixture、compileall 已完成。
- **性能验证：** 未做。缺少本机真实 MUSAN 训练列表、V100 与 encoder checkpoint。不编造收益。
- **训练质量验证：** 未做。有限裁剪库需完整训练 + hard/低误报评估；不用短测速 loss 声称精度无影响，也不承诺回到 3 小时。
