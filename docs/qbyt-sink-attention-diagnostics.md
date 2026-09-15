# QbyT v4.1 sink attention 诊断

短音频上观察 pooling v4.1 QbyT 的 sink attention，并对 sink **key** 做推理期屏蔽消融。入口是 `scripts/diagnose_qbyt_sink.py`。

本工具回答的是：给定已训练权重与固定 WAV，文本/音频 query 对 sink（及音频 key）的 attention 长什么样，切断 sink key 后 raw logit / 分数如何变化。它**不是** sink「学会抑噪」的证明，也不计算 FA/h 或部署事件率。`prep.sink_diagnostics.mode=windows` 未实现。

本分支**没有**随仓库附带的训练好的 v4.1 权重。下列命令里的 checkpoint / 配置 / 音频路径是占位符，请换成你本机的绝对路径。合成 fixture 演示见文末「验收状态」。

## CSV 合同

入口调用共享 `load_manifest(prep.manifest)`，后缀必须是 `.csv`。UTF-8、标准 CSV quoting、原始行序、额外列保留；相对 `audio_path` 相对清单所在目录解析。

| 列 | 必填 | 说明 |
| --- | --- | --- |
| `audio_path` | 是 | WAV 路径 |
| `keyword` | 是 | 该行匹配的关键词 |
| `label` | 否 | `1` / `0`；**空单元格 = 未标注**，不会当成 `0`（共享 loader 会丢掉空 `label`） |
| `keyword_phonemes` | 否 | 空格分隔 ARPAbet 或 JSON 数组；缺省走 G2P；显式空串不当成另一套 fallback |
| `sample_id` | 否 | 非空且全表唯一；缺省生成 `row_000002` 这类稳定行 ID |
| `condition` | 否 | 建议 `clean` / `noisy` / `noise_only` / `unrelated_speech` / `near_miss`；缺省 `unknown` |
| `pair_id` | 否 | 同一基准语音与其加噪变体的组 ID |
| `keyword_spans` / `noise_spans` | 否 | JSON `[[start,end), ...]`，源音频秒；半开区间 |
| `pronunciation_id` | 否 | 展示标签；内部身份以实际 token 序列为准 |

最低兼容清单（无诊断扩展列也能跑完）：

```csv
audio_path,keyword,label
audio/clean_001.wav,hey eva,1
audio/noise_001.wav,hey eva,0
audio/unlabeled_001.wav,hey eva,
```

扩展示例（路径示意）：

```csv
audio_path,keyword,label,sample_id,condition,pair_id,keyword_spans,noise_spans
audio/clean_001.wav,hey eva,1,clean_001,clean,p001,"[[0.8,1.6]]",[]
audio/noisy_001.wav,hey eva,1,noisy_001,noisy,p001,"[[0.8,1.6]]","[[0.5,2.0]]"
audio/noise_001.wav,hey eva,0,noise_001,noise_only,,,"[[0.0,2.0]]"
```

`pair_id` 规则：完成共享音素解析和注册后，按 `pair_id` + keyword + 实际 token 序列 分组。空格分隔 ARPAbet 与 JSON 数组若解析为同一音素序列，视为同一发音。同组须恰好一个 `condition=clean` 基准，可有多个 noisy 变体。缺 clean、多个 clean、或 keyword/token 不一致时写入 `pair_status` / `pair_reason`，单样本诊断仍继续。不要用 CSV 行序猜配对。

`prep.keyword_eval.mode=any` **不支持**（明确报错）。本工具只接受 `per_row`：每行一个关键词与一条有效发音。

不要开 `prep.audio_aug` / `prep.musan_mix`：诊断要求已导出的固定音频。

## 命令

与 `eval_stage2_clips.py` 对照时，用**同一** Hydra 配置、同一 CSV、同一校准器与 padding、`prep.amp=off`、小 batch。不要拿当前默认 v7 配置只覆盖 `version=4` 来「修好加载」；应使用该 checkpoint 的原始完整配置（或可正确组合的 Hydra overrides）。

`configs/prep/default.yaml` 已含 `sink_diagnostics` 默认块。旧保存配置若没有该命名空间，用 `++prep.sink_diagnostics...` 注入。入口会合并自身默认并拒绝未知键。

先跑原评测（可选，用于分数对照）：

```bash
.venv/bin/python scripts/eval_stage2_clips.py \
  --config-path /ABS/PATH/TO/V41_CONFIG_DIR \
  --config-name v41_run \
  prep.manifest=/ABS/PATH/TO/manifest.csv \
  prep.stage2_ckpt=/ABS/PATH/TO/v41.pt \
  prep.output_dir=/ABS/PATH/TO/baseline_clips \
  prep.keyword_eval.mode=per_row \
  prep.batch_size=1 \
  prep.num_workers=0 \
  prep.amp=off \
  run.device=cpu
```

再跑诊断：

```bash
.venv/bin/python scripts/diagnose_qbyt_sink.py \
  --config-path /ABS/PATH/TO/V41_CONFIG_DIR \
  --config-name v41_run \
  prep.manifest=/ABS/PATH/TO/manifest.csv \
  prep.stage2_ckpt=/ABS/PATH/TO/v41.pt \
  prep.output_dir=/ABS/PATH/TO/sink_diagnostics \
  prep.keyword_eval.mode=per_row \
  prep.batch_size=1 \
  prep.num_workers=0 \
  prep.amp=off \
  ++prep.sink_diagnostics.mode=clips \
  ++prep.sink_diagnostics.capture_layers=all \
  ++prep.sink_diagnostics.capture_heads=all \
  ++prep.sink_diagnostics.ablations='[block_sink_all,block_sink_each_layer]' \
  ++prep.sink_diagnostics.max_report_samples=40 \
  run.device=cpu
```

有校准文件时两个命令都加相同的 `prep.stage2_calibration=/ABS/PATH/TO/calibration.json`。padding 走共享 helper `resolve_clip_audio_padding_ms`（与 `eval_stage2_clips` 一致）。GPU 可按现有 `run.device=cuda` 约定替换，但仍须 FP32 eager（`prep.amp=off`）；**本仓库未在此分支实测 GPU 数值对照**。

诊断入口对 `prep.batch_size <= 0` 取 `1`，`prep.num_workers <= 0` 取 `0`，不会套用普通评测默认 64。安装了 `waveform_observer`（保存 traces / 报告声谱图）时会强制 `num_workers=0`：worker 进程无法回写父进程的波形缓存。

默认 `prep.sink_diagnostics`（见 `configs/prep/default.yaml`）：

```yaml
sink_diagnostics:
  mode: clips
  capture_layers: all
  capture_heads: all
  save_traces: true
  save_full_attention: false
  ablations: [block_sink_all, block_sink_each_layer]
  max_combined_tokens: 1024
  max_attention_bytes: 268435456
  parity_atol: 0.00001
  parity_rtol: 0.0001
  max_report_samples: 40
  plot_dpi: 160
  length_bins: [0, 100, 200, 400, 800]
  group_field: condition
  synthetic_fixture: false
```

`normal` 分支始终执行，不必写进 `ablations`。`block_sink_each_layer` 按实际层数展开为单层干预。`synthetic_fixture=true` 会在 `run.json` / `summary.json` / HTML 打上横幅「合成 fixture / 非训练模型结论」——仅用于 stub/fixture，不要对真实 checkpoint 打开。

## 如何读 HTML / PNG

打开输出目录下的 `report.html`（离线、无 CDN）。单样本图包括：

- **声谱图**：与模型同一条 prepared 波形；有时间映射时标 keyword/noise 区间与 padding。映射不可用时横轴标明用 `encoder_frame_index`，不要把帧号当成精确秒。
- **audio→sink**：音频 query 对 sink key 的权重，纵轴 layer×head，横轴源时间或 `encoder_frame_index`。主色标 `[0,1]`；旁侧 log 视图只是辅助。
- **text→sink+audio**：文本 query 对 key。**sink 是独立第 0 列**，后面才是音频 key；把 sink 列与音频列拼在一起展示时，**行不会再重新归一化到 1**（不是「含 sink 后仍保证行和为 1」的展示）。y 轴是实际音素。
- **EPS `position_logits`** 与消融相对 normal 的 delta。
- **ablation_scores**：normal / `block_sink_all` / 各单层屏蔽后的 `qbyt_score`。消融分数 = **原校准变换后的干预分数**（同一校准器与阈值作用在干预后的 raw logit）。这不声称干预后分数「仍已校准」。

时间轴使用前端的**名义帧中心**，不是上下文编码器的完整感受野，也不是学习权重加权的时间定位。Wenet 使用 `embed.subsampling_rate/right_context` 声明；Icefall Zipformer 校验三层卷积和末级两帧聚合结构，完整聚合对应 fbank 支撑宽度 11、步长 4，末尾重复补齐的单帧对应宽度 9。`output_frames()` 只校验输出长度，不用于反推感受野；未声明或不匹配的结构降级为 `encoder_frame_index`。实现见 [时间映射](../dma_kws/inference/qbyt_attention_report.py)。旧版本生成的时间轴和区域统计需要重新运行诊断才能修正。

配对图仅在 `pair_status=ok` 且时间网格可比较时画 attention 差值；否则只比标量。`max_report_samples` 只限制重型单样本图，不删 `records.csv` 行。

高 sink attention ≠ 抑噪成功；`delta_raw_logit > 0` 只说明切断 sink 读取后分数上升，不能单独推断机制。

## 输出文件

默认 `prep.output_dir` 为 `outputs/diagnose_qbyt_sink`。若目标目录已有 `run.json` 或 `summary.json`，入口**拒绝覆盖**，请换目录。

```text
<output_dir>/
  report.html
  run.json
  summary.json
  records.csv              # 样本 × ablation 分数 / delta / skip
  attention_metrics.csv    # layer/head × region 标量
  position_scores.csv      # EPS position logits / delta
  pairs.csv                # pair_id 标量对照
  traces/*.npz           # allow_pickle=False 可读；无 object array
  figures/                 # PNG
  .partial/                # 运行中面包屑；成功结束会删掉
```

`run.json` 含 schema、git、manifest/checkpoint SHA256、字典与校准器指纹、resolved 配置、readout spec、padding、device/dtype、展开后的 capture/ablation、时间轴方法、limits 等。对照 baseline 时按 `manifest_record_number`（或等价记录身份）对齐 `records.csv` 的 `ablation=normal` 行与 `eval_stage2_clips` 的 `results.jsonl`，不要只按 `audio_path` join。

HTML 渲染直接使用本次运行摘要；渲染成功后才写入最终 `summary.json`，并保留 `figures` 与 `num_report_selected`。渲染失败会记录 `status=failed` 和错误原因，并保留 CSV、`run.json` 及 `.partial/`。入口返回值、JSON 和 HTML 的样本计数保持一致；实现见 [输出写入](../scripts/diagnose_qbyt_sink.py)。

## 资源限制

- `max_combined_tokens`（默认 1024）：打包后文本+sink+音频 token 上限；超出则该样本 `skip_reason=resource_limit`。
- `max_attention_bytes`（默认 256 MiB）：attention **工作区**估计上限（hook 层 × `qbyt.nhead` 全部 head × packed L² × float32 × 4 倍临时张量），不是 `save_full_attention` 落盘大小，也不是所选 `capture_heads`。批内会再拆组，仍超则跳过。
- 捕获路径会暂时关掉 MHA/encoder fused **fastpath**，并在 `finally` 里恢复进入前状态。fastpath 是**进程级**开关；本工具按单线程串行设计，**不要**嵌进并发在线服务或对同一模型嵌套 capture。

## 验收状态

| 项 | 状态 |
| --- | --- |
| T01–T16（CPU FP32，小 pooling QbyT + stub encoder） | 已实现并通过测试 |
| 合成 fixture 演示（SDD `demo-run/`，gitignored） | 有；HTML/`run.json` 横幅「合成 fixture / 非训练模型结论」 |
| 真实 v4.1 训练 checkpoint | **未验证**（本分支无权重） |
| 成对干净/加噪真实语音 | **未验证** |
| GPU FP32 与 CPU 数值对照 | **未验证** |
| FA/h、`mode=windows` | **未实现 / 未验证** |

合成演示分数来自随机初始化 pooling + Linear stub encoder + 静音 WAV，**不能**当作 sink 学会（或未学会）拒噪的证据。

## 相关入口

- CLI：[`scripts/diagnose_qbyt_sink.py`](../scripts/diagnose_qbyt_sink.py)
- 与评测共享的 padding：`resolve_clip_audio_padding_ms`（`dma_kws/inference/stage2_clip.py`，`eval_stage2_clips.py` 同用）
- 采集 / 消融：`dma_kws/inference/qbyt_attention_diagnostics.py`
- CSV 扩展校验：`dma_kws/inference/qbyt_attention_manifest.py`
- HTML/PNG：`dma_kws/inference/qbyt_attention_report.py`
- 配置默认：`configs/prep/default.yaml` → `sink_diagnostics`
- 实施合同：[`docs/superpowers/plans/2026-09-13-qbyt-v41-sink-attention-diagnostics-grok.md`](./superpowers/plans/2026-09-13-qbyt-v41-sink-attention-diagnostics-grok.md)
