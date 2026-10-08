# checkpoints - 发布在本仓库 GitHub Release 的权重清单

本项目训练出的全部权重（453 个文件，约 9.2 GiB）**不进 git**，而是作为 GitHub Release
资产发布；仓库里只保留清单和工具：

    checkpoints/manifest.tsv        资产名 <-> data/dma-kws 相对路径 <-> sha256
    checkpoints/RELEASE_NOTES.md    release 正文
    scripts/checkpoint_release.py   build / publish / fetch 三个子命令

- Release 页面: https://github.com/ty4045681/DMA-KWS/releases/tag/checkpoints-2026-10-08
- 资产名唯一且可逆：data/dma-kws 相对路径里的 / 换成 __，其他非法字符换成 -；
  以 manifest 为准，不需要手工解码。
- 仓库是公开的，下载不需要 token。

## 拉取

    .venv/bin/python scripts/checkpoint_release.py fetch                    # 全量并校验 sha256
    .venv/bin/python scripts/checkpoint_release.py fetch --only final/      # 只拉交付集
    .venv/bin/python scripts/checkpoint_release.py fetch --list             # 只列不下载

已存在且 sha256 正确的文件会跳过，可以反复执行续传。

## 更新（训练出新权重后）

    .venv/bin/python scripts/checkpoint_release.py build
    .venv/bin/python scripts/checkpoint_release.py publish --tag <新 tag>
    git add checkpoints/manifest.tsv
    git commit -m "chore(checkpoints): refresh the release manifest"

publish 只上传缺失或大小不符的资产，已存在且大小一致的不重复传；加 --dry-run 先看清单。

## manifest 列

    asset      GitHub release 资产名（唯一）
    relpath    相对 data/dma-kws 的路径，fetch 按此还原
    bytes      文件大小
    sha256     内容校验
    run        产出它的 run 目录
    snapshot   对应的 run_snapshots/lightning/.../hparams.yaml（没有则为空）
