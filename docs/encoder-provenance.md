# Encoder 来历比对

由 `scripts/compare_encoder_provenance.py` 生成。最大差 0 表示 encoder 逐元素等同于右侧，
即在本仓库里从未被训练过。

| 比对 | 公共张量 | 仅左侧 | 仅右侧 | 逐元素最大差 | 最大差位置 |
|---|---|---|---|---|---|
| SS-zh-en ← 上游 zh-en-3M | 327 | 0 | 0 | 0 | `` |
| SS-gsbase-stream ← 上游 GS-base | 327 | 0 | 0 | 0 | `` |
| SS-R1 ← 上游 zh-en-3M | 327 | 0 | 0 | 1.0997 | `encoder.encoders.3.encoder.layers.0.norm.bias` |
| SS-R1 ← R1-bare | 327 | 0 | 0 | 0 | `` |
