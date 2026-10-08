# run_snapshots - 每次运行的实际配置快照

`outputs/` 和数据盘实验目录都被 .gitignore 排除，所以"这次跑用的确切配置"此前只存在于磁盘上。
这里保存它们的**副本**（原目录不动，脚本只读源、只写本目录）。

## 布局

    run_snapshots/
      hydra/<date>/<time>/config.yaml      # Hydra 组合后的完整配置
      hydra/<date>/<time>/hydra.yaml       # Hydra 运行元数据
      hydra/<date>/<time>/overrides.yaml   # 命令行 override，复现实验的关键
      lightning/outputs/<run>/version_N/hparams.yaml
      lightning/exp/<run>/version_N/hparams.yaml
      index.tsv                            # 快照 -> 源文件 -> sha256(前16位) -> 字节数

共 1400 个快照文件：423 个 Hydra 运行（1269 个文件）+ 131 个 Lightning hparams
（repo 侧 8 个，数据盘 `data/dma-kws/exp/` 侧 123 个，含 35 个 50k Stage II 正式 run）。

统计口径：`wc -l < run_snapshots/index.tsv`（含表头）；每次 `--check` 报漂移时重新跑一遍收集器即可。

## 刷新

    .venv/bin/python scripts/collect_run_snapshots.py          # 复制 / 更新
    .venv/bin/python scripts/collect_run_snapshots.py --check   # 只检查，有漂移退出码 1

幂等：只有字节不同的文件才会被重写。

## 找某个权重用的配置

* 数据盘 run：目录名一一对应，例如权重
  `data/dma-kws/exp/stage2_qbyt/checkpoints/v41-musan-asrxl-50k/v41-musan-asrxl-50k/version_0/last.ckpt`
  对应 `run_snapshots/lightning/exp/stage2_qbyt/checkpoints/v41-musan-asrxl-50k/v41-musan-asrxl-50k/version_0/hparams.yaml`。
* Hydra run：用 override 反查，例如

      grep -rl "v41-musan-zhen3m-50k" run_snapshots/hydra | head

  再看同目录的 `overrides.yaml` 就能拿到完整复现命令。
