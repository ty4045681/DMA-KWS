# DMA-KWS checkpoints (2026-10-08)

DMA-KWS 复现 + QbyT v1-v4.2 + Hey-Eva 适配训练出的**全部权重**：453 个文件、9.22 GiB。
权重不进 git，这个 release 是它们的托管副本；校验清单见仓库的 checkpoints/manifest.tsv。

## 内容

| 目录 | 文件 | 大小 | 说明 |
| --- | ---: | ---: | --- |
| exp/stage2_qbyt/checkpoints | 174 | 4.05 GiB | 15+ 个 50k Stage II 正式 run（v1-v4.2） |
| exp/stage2_qbyt/final | 30 | 0.66 GiB | 交付集：R1-unfreeze-hardneg100 / P1-paper-50k / C3-sinkloss-50k 等 |
| exp/stage2_qbyt/sinkhead_stage2 | 35 | 0.54 GiB | sink 头第二阶段（SS-* 7 个 encoder） |
| exp/stage2_qbyt/improve | 16 | 0.46 GiB | v4.2 improve 系列 |
| exp/stage2_qbyt/ablations + ablations_b + sinkloss_ab + v42_encoders | 51 | 0.89 GiB | 消融与 encoder 矩阵 |
| exp/stage2_adapt_v42/** | 139 | 2.24 GiB | Hey-Eva 适配 + R1 enc/qbyt/lora 共 20 个 run |
| exp/author_v1 | 2 | 38 MiB | 作者 v1 原版发布（stage1_v1.pt / stage2_v1_si.pt） |
| raw/kws-checkpoints | 6 | 0.36 GiB | 预训练 encoder：zh-en-3M、GigaSpeech-ASR-XL、paper-stage1 |

## 下载

    python scripts/checkpoint_release.py fetch --tag checkpoints-2026-10-08
    python scripts/checkpoint_release.py fetch --tag checkpoints-2026-10-08 --only final/
    python scripts/checkpoint_release.py fetch --tag checkpoints-2026-10-08 --list

资产名 = data/dma-kws 相对路径把 / 换成 __、其他非法字符换成 -，例如

    exp__stage2_qbyt__final__P1-paper-50k__checkpoints__P1-paper-50k__version_0__last.ckpt

每个文件的 sha256、大小、产出 run 和对应的训练配置快照
（run_snapshots/lightning/.../hparams.yaml）都在 checkpoints/manifest.tsv。
