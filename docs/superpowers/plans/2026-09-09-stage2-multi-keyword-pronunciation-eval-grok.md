---
title: "Grok 执行计划：Stage II 多唤醒词与多发音任一命中评测"
description: "可独立实施的配置、标签、共享编码、多发音聚合、误唤醒统计、兼容性和测试合同。"
tags: [ qbyt, stage2, evaluation, implementation-plan ]
status: implementation complete / GPU validation pending
baseline_commit: 8e61517afbf019adaa5f9ded7679eb8cec1c8085
created: "2026-09-09"
---

本文件可直接交给 Grok 实现，不依赖此前对话。用户需要：每条音频匹配配置的全部唤醒词；同一个词可配置多种音素序列；任意一种发音命中任意一个词，这条音频就算唤醒，且只计一次。

本文中的新配置、接口及命令属于实施合同，尚未实现。实施者应完成代码、测试、示例与使用文档，不能只重新生成计划。编写计划时工作树干净，基线提交如上；执行时重新检查当前分支与差异，保留用户已有改动，遵守 [AGENTS.md](../../../AGENTS.md)。普通实现选择按本文执行；没有真实数据或 GPU 时完成 CPU 可验证部分，明确列出未执行的验收，不宣称完整生产验证。

**1. 目标与范围**

必做交付覆盖以下链路：

- Stage II 整段音频评测：`eval_stage2_clips.py`。
- Stage II MUSAN 滑窗误唤醒评测，以及分片启动、合并、汇总和曲线。
- 离线阈值扫描、噪声矩阵批量评测及其缓存身份。
- 同词多发音和多词统一配置、标签校验、结果与语义指纹。
- 旧 `per_row` 模式回归、当前支持的 QbyT v2–v7 数值回归。

默认仍是旧模式。新功能通过 `prep.keyword_eval.mode=any` 启用。保留已有 `prep.keyword`、`prep.keyword_phonemes`、manifest 行内 `keyword` / `keyword_phonemes` 的旧语义。

本轮不改训练、LoRA、readout 公式、checkpoint 权重或版本标记，不增加每词独立阈值，不做自动生成口音变体、不训练新校准器、不改 Stage I 候选定位。LibriPhrase 成对基准和两阶段 KWS 仍沿用旧协议；用户在这些入口显式请求 `any` 时应报“不支持此评测模式”，避免静默忽略。真实连续流的事件合并和冷却时间另属部署协议，本轮报告窗口触发率及文件级触发率，不将窗口数冒充真实唤醒事件数。

**2. 当前代码依据**

| 已核对事实 | 实现入口 |
| --- | --- |
| 整段评测按 manifest 行调用 runner、写结果、统计指标 | [eval_stage2_clips.py](../../../scripts/eval_stage2_clips.py)，`run_eval` |
| `run_batch` 一行一个 query；缓存键已经包含文本及音素 override | [stage2_clip.py](../../../dma_kws/inference/stage2_clip.py)，`Stage2ClipRunner.run_batch` |
| 现有 verifier 一批特征与一批音素一一对应；内部已有可复用音频编码入口 | [stage2_verifier.py](../../../dma_kws/inference/stage2_verifier.py)，`score_clip_feats_with_logits`、`Stage2Model.encode_for_qbyt` |
| 旧 manifest 强制 `audio_path` 和 `keyword` | [manifest.py](../../../dma_kws/inference/manifest.py)，`load_manifest`、`_validate_row` |
| 现有 `prep.keywords` 是生成 manifest 时按文件名分配单词的输入 | [prepare_two_stage_manifest.py](../../../scripts/prepare_two_stage_manifest.py)，`main` |
| MUSAN 入口只读一个 `prep.keyword`，小时数按源文件累计 | [eval_musan_fa.py](../../../scripts/eval_musan_fa.py)，`run_eval` |
| 当前结果记录强制单个 `keyword`，且严格检查有限分数 | [stage2_reporting.py](../../../dma_kws/inference/stage2_reporting.py)，`build_result_record` |
| score provenance 当前 schema 为 3，校验器只接受该版本 | [score_provenance.py](../../../dma_kws/inference/score_provenance.py) |
| MUSAN 分片合并身份中包含单词和音素等字段 | [musan_fa.py](../../../dma_kws/inference/musan_fa.py)，`_MERGE_IDENTITY_KEYS`、`merge_musan_summaries` |
| 噪声矩阵预检当前直接检查每行二值 label，且按 resolved config 等建立缓存指纹 | [batch_eval_stage2_noise_matrix.py](../../../scripts/batch_eval_stage2_noise_matrix.py)，`validate_inputs`、`build_jobs` |
| ARPAbet 重音保留为不同 token，不可擅自去掉数字 | [tokenizer.py](../../../dma_kws/tokenizer.py) |

在本会话前面的核查中，4 项现有相关测试通过；以 fake verifier 给同一音频的两个 query 返回 0.2 与 0.9，runner 输出两条独立结果。该证据只说明现有流程语义，不是新功能或真实模型验收。

**3. 固定的评分语义**

设音频为 x，关键词集合为 K，词 k 的有效发音集合为 P(k)，每个 query 的模型原始输出为 z(x,k,p)。使用现有同一个 `PositiveAffineCalibrator` 逐 query 得到 s(x,k,p)。

```text
keyword_score(x, k) = max(s(x, k, p) for p in P(k))
qbyt_score(x)       = max(keyword_score(x, k) for k in K)
detected(x)         = scored(x) and qbyt_score(x) >= threshold
```

阈值唯一来源为 `demo.qbyt_threshold`。新模式要求阈值是 [0,1] 内有限数，且拒绝 bool。内部 API 与 CLI 共用同一校验。阈值扫描另沿用其现有“拒绝所有样本”的端点表达，不为训练/推理放宽阈值配置。

- 两种发音分数 0.32、0.86，词分数为 0.86；任一词分数达到阈值即唤醒。
- 同一个词两种发音同时命中、两个词同时命中，都只增加一次音频级命中计数。
- 每个 query 保持独立音素序列，不把序列拼接、不平均 logits、不做跨 query softmax、不用概率乘积公式。
- 所有 query 使用同一个校准器，按“先逐 query 校准，再 max”实现；全局 raw logit 保存胜出 query 的真实 logit，禁止从已饱和的概率反算。
- 同分时按规范化关键词文本、再按有效 token 序列的字典序选择胜者；不能以输入列表顺序决定。完整 `matched_keywords` 保留所有过阈值的词。
- max 分数可用于排序和阈值判定；它不是已验证校准的“至少出现一个词的概率”。报告中的 ECE/Brier 等只能称聚合分数的校准诊断，不标榜校准有效。
- 加入新的有效 query，在已有 query 分数不变时 max 不会下降。因此必须在验证集重新选择阈值，再在独立测试集按固定 FPR 或固定 FA/h 比较召回，不从测试集反复选发音与阈值。

**4. 配置接口**

在 [prep 默认配置](../../../configs/prep/default.yaml) 新增：

```yaml
keyword_eval:
  mode: per_row
  targets: []
  query_batch_size: 64
```

示例 overlay 要求新建 `configs/keyword_eval/hey_eva_variants.yaml`，内容如下；这是数据配置，不绑定任何模型架构：

```yaml
# @package prep.keyword_eval
mode: any
query_batch_size: 64
targets:
  - text: "hey eva"
    pronunciations:
      - "HH EY1 IY1 V AH0"
      - "HH EY1 EY1 V AH0"
```

另新建 `configs/keyword_eval/multi_wakeup.yaml`：

```yaml
# @package prep.keyword_eval
mode: any
query_batch_size: 64
targets:
  - text: "hey eva"
    pronunciations:
      - "HH EY1 IY1 V AH0"
      - "HH EY1 EY1 V AH0"
  - text: "ok lamp"
    # 省略 pronunciations，解析成一个 G2P 发音。
```

按现有 [Hydra schema](../../../dma_kws/configs/schema.py) 的实际 prep 容器形态集成，不假设仓库已经存在 `PrepConfig`。新增小型解析/校验类型并由运行入口调用即可；需要同步 schema 默认值时保持原有 prep 其他字段。

配置规范：

| 项目 | 行为 |
| --- | --- |
| `mode` | 仅 `per_row` / `any`；默认 `per_row` |
| `targets` | `any` 要求非空对象列表；`per_row` 要求为空，避免配置被忽略 |
| `text` | 必须是非空字符串；规范化仅 trim、合并连续空白、casefold，不删除单词间空格或标点；保留输入值作为展示元数据 |
| `pronunciations` 缺失或 null | 为该词生成恰好一个 G2P 序列；必须校验 G2P 输出 |
| `pronunciations` 显式给出 | 非空字符串列表，每项是空格分隔的 ARPAbet 序列；不隐式追加 G2P |
| 空列表、空字符串、非字符串序列、非法音素 | 带 target 索引、文本和发音索引报错，不能降级成 G2P 或 `<unk>` |
| 旧讨论中的 `targets[].phonemes` | 不是已发布接口，本轮不新增这个别名；给出迁移到 `pronunciations` 的明确错误 |
| 同 text 重复 target 项 | 规范化后合并为同一个词，取发音并集；显式与 G2P 条目并存时两者都纳入；输出合并诊断 |
| 发音去重 | 按实际 token-id tuple 去重；`AH0`、`AH1` 等不得被人为合并 |
| 不同 text 对应同一 token 序列 | 仍保留两个词的身份和报告；模型计算可共享该 query，再映射到两个词 |
| `query_batch_size` | 正整数且拒绝 bool；上限是单次 QbyT 的“音频 × 有效发音”pair 数，不是词数 |
| `any` 与旧配置冲突 | 非空 `prep.keyword`、`prep.keyword_phonemes`、`prep.keywords` 同时存在时报错；默认空值允许 |

为每个规范化词和每个有效发音生成确定性 ID。不要用 Python `hash()` 或不稳定的输入索引；用规范化文本及 token tuple 的 canonical JSON / SHA-256。保存完整有效音素、token ids、G2P/显式来源，便于检查和复现。

**5. 音频级 manifest 与标签合同**

新增 `load_keyword_set_manifest(path, target_texts)`，复用原有 CSV/JSONL 读取、相对路径解析和元数据保留代码；保留旧 `load_manifest(path)` 的默认必填字段和行为。any 模式一条源音频一行，拒绝解析后重复的音频绝对路径，错误指出重复行号。不要静默合并旧 pair 行。

推荐输入形式是每条音频的逐词标签映射，JSONL 示例：

```json
{"audio_path":"audio/eva.wav","keyword_labels":{"hey eva":1,"ok lamp":0}}
{"audio_path":"audio/lamp.wav","keyword_labels":{"hey eva":0,"ok lamp":1}}
{"audio_path":"audio/both.wav","keyword_labels":{"hey eva":1,"ok lamp":1}}
{"audio_path":"audio/negative.wav","keyword_labels":{"hey eva":0,"ok lamp":0}}
```

- `keyword_labels` 描述真实音频内容，规范化其键，要求覆盖所有当前配置词；值严格为 0/1，禁止 bool、浮点截断或其他整数。JSONL 用整数，CSV 内嵌 JSON 同样如此。
- 允许映射含额外未启用词，保留为元数据；只对当前词表取 max 派生音频 `label`。这使同一份完整标注可以评测词表子集；扩大词表若缺标注必须报错。
- 行内另给 `label` 时，必须等于派生结果，否则报错。CSV 的外层 `label` 可接受精确的字符串 "0"/"1"。
- 同词不同发音共享该词真实标签，不要求口音标签；发音维度用于诊断，不能把哪个变体得分最高当作真实口音标注。
- 只有音频级标签也可评测：行必须显式带 `label_scope:"target_set"`、`target_texts:[...]` 和 `label:0|1`，其中 target_texts 规范化后必须精确等于配置词集合。此形式不生成逐词准确率。
- 完全无标签的行允许输出预测；带部分、不完整或互相冲突的标签信息必须报错。混合标注/无标注数据时汇总记录 num_labeled，metrics 只用有标签且成功评分行。
- any manifest 不接受旧的顶层 `keyword` 或 `keyword_phonemes` 字段。它们容易把单词负标签误当词表负标签；错误消息给出两种新格式示例。
- 标签字段只进入指标层，不能影响 enrollment、待评分 query 或取 max 的集合，避免按 ground truth 挑选关键词。
- CSV 的 `keyword_labels` / `target_texts` 使用标准 JSON 文本单元格，不能用 eval。文件存在性、音频格式、最短长度沿用现有音频加载逻辑。

只有“音频对 hey eva 为负”不足以推断“音频对 hey eva 和 ok lamp 的集合为负”。若提供迁移脚本，必须验证每条音频具备词表内每个词的标签后才可合并；缺少任意标签即拒绝。自动迁移工具不是本轮必交付，文档中的新格式和清晰报错必须交付。

**6. 共享编码与内存边界**

建议新增 `dma_kws/inference/keyword_set.py` 作为纯配置、enrollment、ID、聚合逻辑模块。可以用 frozen dataclass 表达 `ResolvedKeywordSet`、`ResolvedKeyword`、`ResolvedPronunciation`；名称允许按项目习惯调整，语义必须保持本文定义。不要引入一个独立于现有 verifier 的模型加载器。

在 [Stage2Verifier](../../../dma_kws/inference/stage2_verifier.py) 新增公共接口：

```python
def score_clip_feats_multi_with_logits(
    self,
    feats,
    query_ids,  # 去重后的全局发音序列，顺序由解析器固定
    *,
    query_batch_size=64,
):
    """返回每条音频、每个 query 的 (raw_logit, calibrated_score)。"""
```

接口返回 shape/映射必须明确为 `[num_clips, num_unique_queries]`，附带所需 query 身份；空 feats 返回正确空结果，空 query 列表是配置错误，不能返回所有样本为负。输出 pair 数与输入必须精确匹配，不能依赖 zip 截断而丢失结果。

执行顺序：

1. 在 main process 解析词表、运行 G2P、tokenize 和去重一次；不让 DataLoader worker 重做 G2P 或持有 GPU 模型。
2. 按音频 batch 读取/重采样/加噪/导出观察/fbank/最短长度判断；这些操作的次数不随词和发音数增加，增强使用原始音频行索引。
3. 使用现有 fbank padding、stream policy、AMP context、`eval()`、`no_grad()` 和 `encode_for_qbyt`，每个有效音频 batch 只调用一次 encoder。启用 phoneme adapter 时共享其输出，不误用 CTC 后验。
4. 将音频索引 × query 索引逻辑展开；每次只为最多 query_batch_size 个 pair gather 编码特征、长度和 anchors，然后调用现有 `self._model.qbyt`。
5. 立即回收临时 GPU pair 张量，回填 CPU 分数矩阵；不要先构造完整 `[B × P, T, D]` GPU 张量再切块。QbyT 占用由 pair chunk 控制，音频编码占用由原 `prep.batch_size` 控制。
6. 逐 query 调现有校准器一次，再由公共聚合器得到词内和词间 max。
7. runner 保持一条输入音频对应一条输出记录的稳定顺序。音频行整体打乱只改变输出行顺序，不能改变同一输入的得分。

不得把 `E(full_audio)` 的任意 slice 替代 `E(window_audio)`。MUSAN 仍按原滑窗协议生成每个窗口的模型输入，只在同一个窗口的多 query 间共享编码。不要改变左右 padding、长度传播、snip_edges、fbank_windows 模式或 crop 语义来换取速度。

按理论工作量，朴素展开是对 B×P 个输入重复编码，新路径是 B 个音频编码加 B×P 个 QbyT 查询。不得据此承诺整体耗时降低 P 倍；实际耗时和显存以小规模测量为准。

**7. Runner、跳过样本与结果结构**

在 [Stage2ClipRunner](../../../dma_kws/inference/stage2_clip.py) 新增 `run_batch_multi`，复用 `ClipFeatureDataset`、waveform augmentation/export 路径。旧 `run_batch` 接口及 per_row 记录结构保持可用。MUSAN 增加多 query 的 prepared-window 评分入口；可以把当前 `PreparedFileWindows` 的声学字段提取为公共结构，但原单词 API 的行为必须由兼容 wrapper 保持。

失败语义固定：

- 太短而没有可评分特征：一条 `skipped=true` 结果，`detected=false`，`qbyt_score=0.0`，`qbyt_raw_logit=null`，胜者 null，matched_keywords 空列表。即使 threshold=0 也不能唤醒。
- 记录 `skip_reason="too_short"`；不把 skipped 当作模型成功判负。整体 summary、扫描、MUSAN 合并要使用相同的过滤规则。
- 新模式任一实际 query 输出 NaN/Inf，带音频与 query 信息失败，不能用 nanmax 静默绕过。非法路径如何得到有限分数沿用各 readout 的现有语义，不另写“匹配失败=0”的模型替代。
- I/O 错误、配置错误、非法音素及时报告；不默认跳过坏文件继续统计。

any 结果必须使用明确的新 schema，例如下列伪 JSON 字段结构；其中分数示例仅表达层级，实际写入必须来自模型：

```text
{
  result_schema_version: 2,
  eval_protocol: "stage2_clip_keyword_set",
  keyword_eval_mode: "any",
  keyword_set_id: "...",
  audio_path: "...",
  label: 0 | 1,                        # 未标注时省略
  label_scope: "target_set",            # 有 label 时写
  keyword_labels: {...},               # 有逐词真值时写
  qbyt_raw_logit: finite | null,
  qbyt_score: finite in [0, 1],
  threshold: finite in [0, 1],
  detected: bool,
  skipped: bool,
  skip_reason: null | "too_short",
  best_keyword: "hey eva" | null,       # 无命中但有分数时仍可有 best
  best_pronunciation_id: "..." | null,
  matched_keywords: ["hey eva"],        # 每个词最多出现一次
  keyword_results: [
    {
      keyword_id: "...",
      text: "hey eva",
      qbyt_raw_logit: finite | null,
      qbyt_score: finite,
      detected: bool,
      best_pronunciation_id: "...",
      pronunciation_results: [
        {
          pronunciation_id: "...",
          phonemes: ["HH", "EY1", "IY1", "V", "AH0"],
          token_ids: [...],
          qbyt_raw_logit: finite | null,
          qbyt_score: finite,
          detected: bool
        }
      ]
    }
  ],
  clip_span_sec: {...},
  manifest_meta: {...}
}
```

不要给 any 结果伪造顶层单个 `keyword` 来冒充旧协议。输出给旧单词工具时，要么明确支持新 schema，要么以清楚错误拒绝，不能把 best_keyword 当真值。保留现有 augmented_duration、增强配方、exported_audio_path 等字段；嵌套记录同样使用有限数校验和 `json.dumps(..., allow_nan=False)`。skipped 行的完整词表可由 summary 重建，嵌套 `keyword_results` 固定为空，减少假分数。

summary 至少包含：

- 协议、模式、keyword_set_id、规范化完整词表、有效发音数、去重诊断、query_batch_size。
- num_samples、num_scored、num_skipped、num_labeled；num_samples 始终是输入音频行数，MUSAN 则是原协议生成的窗口数。
- 音频级 metrics / plots 基于总 max 分数；保持现有主键以兼容消费者。
- 有逐词真实标签时生成 per_keyword metrics；每个词每条音频只算一次，使用该词所有发音的 max。单类指标沿用项目既有约定并标明样本类别覆盖。
- 发音诊断可以给分数分布、胜出次数、阈值命中数；不宣称是真实口音分类准确率。
- `score_semantics="max_over_keywords_and_pronunciations"`、校准身份、检查点和输入协议。
- 无标签时不生成伪标签指标；旧 `per_row` 输出不强行迁移成嵌套 schema。

**8. MUSAN 统计协议**

接入 [eval_musan_fa.py](../../../scripts/eval_musan_fa.py)、[musan_fa.py](../../../dma_kws/inference/musan_fa.py) 的规则：

1. 每个源文件仍按原 window_sec / hop_sec / 尾窗规则形成窗口；每个窗口一条 max 聚合记录，`eval_protocol="stage2_window_keyword_set"`，保留 window_index、subset 和原始时间跨度。
2. 音频小时数只按原源文件实际时长累加一次；不能乘词数/发音数，不能用重叠窗口时长总和当分母。
3. `metrics.fa_per_hour` 保持窗口协议下的“过阈值窗口数 / 原始音频小时数”；summary 明确 `fa_count_unit="window"`。同一个窗口多个词或多个发音命中，FP 只增加 1。
4. 另加 `file_metrics`：total_files、num_scored_files、num_skipped_files（无有效窗口）、num_triggered_files，以及 num_triggered_files / num_scored_files。同一文件跨多个窗口命中只计一个触发文件。该比例名为 file_trigger_rate，不命名为事件 FA/h。
5. 只对有效窗口统计分数与 FP；保留 skipped 个数及分布。小时数沿用原源音频口径，同时提供 num_scored_files 等覆盖信息。全部窗口都不可评分时标记 `metrics_status="no_scored_windows"`，不能把零分母或空结果称为完美表现。
6. 分片仍按文件划分；合并后 file_metrics 应由合并的窗口与源文件清单重算，不能平均各分片比例。保存完整 source file 清单（path、subset、duration、scored_window_count），包括 0 窗口的文件，供空分片和短音频合并。
7. 全部背景音频的标签对整个配置词集合必须为负；多词/多发音不改变使用独立噪声、语音负样本、硬负样本评估的要求。语音背景若实际包含目标词，需要重新标注/排除，不能仅凭 MUSAN 名称保证标签正确。
8. any 模式分片同时校验词表、tokenization、校准器、模型、readout、窗口协议、输入目录/清单身份；同 text 增减一个发音也必须拒绝混合。拒绝重复源文件、重复 window key、重复 shard 或缺失 shard，保持原有完整性检查。
9. [分片启动脚本](../../../scripts/eval_musan_fa_shards.sh) 已透传 Hydra overrides，优先验证现有透传即可；不要另造不兼容 launcher。
10. [merge_musan_fa.py](../../../scripts/merge_musan_fa.py)、[aggregate_musan_fa.py](../../../scripts/aggregate_musan_fa.py)、[plot_musan_fa_curve.py](../../../scripts/plot_musan_fa_curve.py) 都要处理新身份及计数单位；列表汇总列显示模式、keyword_set_id、词数、发音数，不能空白单词字段下把不同词表混为一组。

**9. 语义指纹、扫描与缓存兼容**

以 [score_provenance.py](../../../dma_kws/inference/score_provenance.py) 为唯一校验/规范化入口：

- 新写出的 score provenance 升级为 schema 4，新增 `keyword_eval` 字段。它与 checkpoint 的 QBYT_READOUT_VERSION 是两个完全不同的版本号，禁止改写 checkpoint readout 版本。
- schema 4 的 `per_row` 值为 `{"mode":"per_row"}`；any 值包含 mode、aggregation、keyword_set_id、规范化词和实际有效 token 序列。
- reader 接受旧 schema 3，并且仅把它解释为 legacy per_row；语义比较时正规化为等价的 per_row 描述。未知版本仍拒绝。不能把 schema 3 自动补成 any。
- keyword_set_id 使用 canonical JSON 的 SHA-256，内容含词表 schema 版本、文本规范化规则版本、排序后的规范化文本→去重 token tuple 集合、tokenizer 字典 sha256。展示文本、来源说明、输入排列、物理路径和 query_batch_size 不放入词表身份。
- readout、stream、padding、fbank、checkpoint、校准器继续在完整 score provenance 中校验；任何现有 guard 不能因 any 而省略。
- 行内 keyword_set_id 和 summary/provenance 必须一致。扫描或合并发现混合模式、混合词表或混合结果协议时报错。
- 原始 provenance 可保留顺序、G2P 来源和执行批大小等元数据；语义比较函数只能用规范化语义内容。词/发音重新排序、等价重复项去重不能改变词表身份。
- [stage2_reporting.py](../../../dma_kws/inference/stage2_reporting.py) 生成与读取新元数据都走公共模块；不要让每个脚本各维护一份 canonicalization。

离线工具：

- [scan_stage2_thresholds.py](../../../scripts/scan_stage2_thresholds.py) 对 any 的根 `qbyt_score` 扫描，不能把嵌套 query 展平后计数；阈值改变时由分数重新判定，不复用保存时的 detected/matched_keywords。
- 新模式扫描要求对应 summary（默认同目录，允许已有 --summary 参数指定），验证词表身份、协议、跳过规则和计数单位；旧结果仍走原有兼容策略。
- 现有扫描器对未标注记录报错的行为保留；混合标签 manifest 的用户应提供完整标注的扫描数据，错误明确列出未标注行。不要默默把未标注当负。
- clips 报告 recall/FPR 曲线，MUSAN 报窗口 FA/h。曲线和选择结果带上 keyword_set_id、score semantics、协议与原始 source summary。
- [fit_stage2_calibration.py](../../../scripts/fit_stage2_calibration.py) 本轮不扩展集合级拟合。检测 any 聚合结果时明确拒绝当成旧 pair 样本拟合，防止生成没有词表绑定的新校准器；原 pair 输入仍支持。现有校准 JSON 可以在各 query 上按原方式应用。

噪声矩阵：

- [batch_eval_stage2_noise_matrix.py](../../../scripts/batch_eval_stage2_noise_matrix.py) 在校验 manifest 前先按各 model 和 common overrides 解析生效的 keyword_eval 配置，使用与 evaluator 完全相同的新 manifest/标签解析器。现有 `load_manifest` 和“原始行必须有 label”的检查不能阻挡 any 格式。
- 文件/音频身份函数同样改为消费规范化 rows，不能仅改 validate_inputs 而让 `_manifest_audio_identity` 再次用旧 loader 失败。
- 先解析有效 enrollment（用该模型实际 tokenizer）再生成 job fingerprint；把 keyword_set_id 纳入。仅 hash 原始 YAML 不足以捕获省略发音时 G2P 的实际输出变化。
- 矩阵 cache/summary 自检同时核对预期 keyword_set_id 和协议，旧缓存不得误复用；增加发音必须失效，改变模型/校准/增强输入仍沿用原有失效规则。
- 每个 model 可有不同词表配置，但矩阵列必须显示差异，不能默认其召回率可直接横比。
- 用相同音频顺序、augmentation seed 和导出规则；多 query 不影响每条音频增强配方或导出次数。已有 --dry-run 应验证配置/词表/标签而不执行 GPU 推理。

**10. 按顺序执行的实现任务**

每一步完成后运行相应测试并继续下一步；不要在只有解析器或 fake verifier 通过时停止交付。

| 步骤 | 主要文件 | 必须完成的行为与检查 |
| --- | --- | --- |
| A：词表配置与 enrollment | 新 `dma_kws/inference/keyword_set.py`；现有 prep YAML；新两个 keyword_eval overlay | 解析、严格类型校验、G2P/显式发音、同词合并、按 token 去重、跨词共享 query、确定性 ID |
| B：manifest 与标签 | [manifest.py](../../../dma_kws/inference/manifest.py)；新 `tests/test_keyword_set_manifest.py` | 两种有标签格式与无标签格式；旧 loader 不回退；重复路径、缺词标签、冲突标签、旧 pair 输入都按合同处理 |
| C：共享模型评分 | [stage2_verifier.py](../../../dma_kws/inference/stage2_verifier.py)；新 `tests/test_stage2_multi_query.py` | 单次 encoder、pair chunk、有 adapter/无 adapter、校准一次、长度和输出映射正确 |
| D：整段入口与报告 | [stage2_clip.py](../../../dma_kws/inference/stage2_clip.py)、[stage2_reporting.py](../../../dma_kws/inference/stage2_reporting.py)、[eval_stage2_clips.py](../../../scripts/eval_stage2_clips.py) | 一音频一结果；增强/导出一次；新的嵌套输出与 skip 语义；音频级/逐词 metrics |
| E：MUSAN 多词路径 | [eval_musan_fa.py](../../../scripts/eval_musan_fa.py)、[musan_fa.py](../../../dma_kws/inference/musan_fa.py)；新 `tests/test_multi_keyword_musan_fa.py` | 保持窗口特征协议；窗口 max、源小时数、文件触发率、0 窗口/空分片覆盖 |
| F：身份与离线消费者 | score provenance、scan、merge、aggregate、plot、fit 输入检查；新 `tests/test_keyword_set_provenance.py` | schema 3/4 双读、默认新写 4；混合词表拒绝；扫描 max 分数；merge 重算计数；原有路径回归 |
| G：矩阵与 CLI | [batch_eval_stage2_noise_matrix.py](../../../scripts/batch_eval_stage2_noise_matrix.py)、分片 launcher、显式不支持 any 的入口 | 预检读新 manifest、有效 enrollment 进入 fingerprint、cache 自检、overlay 透传、--dry-run 与真实执行配置一致 |
| H：数值回归与文档 | 新测试、现有 tests、[README.md](../../../README.md) 和本计划 | 完成下述验收；README 给最小示例和口径说明；在本计划末追加实际执行结果与未完成项 |

README 增加功能入口和新配置说明即可，不重新组织整个文档。项目 Markdown 按 OpenKnowledge 技能通过 MCP 读写；本计划也通过同一路径追加实施记录。不要把本文件标记 implemented，直到所有必做实现和本地可执行测试完成；真实 GPU 验收缺失时使用明确的 implementation complete / GPU validation pending 描述。

**11. 必须通过的行为验收矩阵**

测试应执行真实的解析、评分/聚合、入口或结果消费者路径，不能只用 AST 字符串查找证明功能存在。

| 类别 | 输入/条件 | 预期 |
| --- | --- | --- |
| 单词退化 | 1 个 text、1 个发音、同一模型输入 | 与旧 per_row 对应 query 的 raw logit 和 score 一致，阈值判定一致 |
| 同词多发音 | 两种发音分数 0.2、0.9，阈值 0.5 | 该词 0.9；一条 detected=true；best 指向第二种发音 |
| 多词 | A 的两发音 0.2、0.3，B 的发音 0.8 | 只记录一条音频级命中，matched_keywords=[B] |
| 全部失败 | 所有 score < threshold | detected=false，matched_keywords=[]，best 仍是最高分 query |
| 多重命中 | 多词/多发音均 >= threshold | 音频级 TP/FP 仅加 1；词内也不重复计数 |
| 边界与同分 | score=threshold；多个 score 相同 | >= 判定；胜者不依赖输入词序、发音序 |
| 空/非法配置 | 空词表、空显式发音、非法 ARPAbet、bool batch size、拼错 key | 在推理前清楚报错，不能退回默认或生成 unk |
| 合并去重 | 重复同词条目、重复 token 序列、跨词相同发音 | 同词并集正确；每个唯一 query 只算一次；跨词身份均保留 |
| G2P fallback | 省略/null、显式列表、显式空列表 | 前两者生成默认；显式列表不追加默认；空列表报错 |
| 真值不泄漏 | 改变 label/keyword_labels，音频和 targets 固定 | raw logits、scores、待查询集合完全不变 |
| 音频级标签 | A 音频与 B 音频分别只含一个词 | 两者集合 label 都为 1；缺 B 标签不能从 A=0 推导总负 |
| 标注格式 | CSV/JSONL 等价；逐词和仅集合标签；无标签 | 标签派生一致；无标签可预测；缺字段/冲突信息报错 |
| 重复音频 | 同路径的相对与绝对写法各一行 | any 模式规范化后拒绝，不能重复计样本 |
| 太短输入 | 一些或全部音频无有效帧，threshold=0 | skipped=true 且 detected=false；metrics 正确排除并记录覆盖 |
| 非有限分数 | 某一个 query 输出 NaN/Inf | 带音频/query 错误中止，不静默丢弃该 query |
| 输出完整性 | scorer 返回 pair 数少于预期 | 明确异常，不能被 zip 截断伪装成功 |
| 编码次数 | B 条有效音频，P 个 query，多次 pair chunk | encoder 每音频 batch 一次；model input 预处理/观察调用数与 P 无关 |
| 有限显存 | P 大于 query_batch_size，B 不整除 chunk | 单次 QbyT pair 数 <= 上限；结果完整，不预物化全部重复 speech |
| 数值稳定 | 不同音频长度、不同发音长度、sink token/relative bias | batched、chunked、单 query 参考得分一致；不同 padding 组合无索引漂移 |
| 校准器 | 非默认正斜率与 bias | 每 query 只校准一次；根 score 等于逐 query 校准后 max |
| Readout | pooling v2/v3/v4 legacy/v4.1、bounded v5、v6、v7；adapter 开关 | 真实 QbyT 前向通过；版本/spec guard 保持有效；原始权重严格重载 |
| MUSAN 双词 | 1 小时源音频，某窗口两个词同时误触发 | 该窗口 FP=1，分母=1 小时；不能变成 2 FP 或 2 小时 |
| 多窗口同文件 | 同文件多个窗口均触发 | 窗口各计一次，num_triggered_files=1 |
| 分片 | 单进程 vs 多分片后 merge；有空分片和 0 窗口文件 | 逐窗结果、总小时、窗口 FP、文件计数一致 |
| 混合身份 | 同 text 不同发音集合、不同词表、校准、窗口协议 | merge/scan 拒绝；重排/等价重复项不造成词表身份变化 |
| 旧记录兼容 | provenance schema 3、旧 per_row results/summary | 仍能扫描/合并；不得被解释为 any |
| 离线扫描 | 保存时阈值 0.5，扫描到 0.8 | 根据根 score 重算；不使用已保存 detected；嵌套 query 不增加样本数 |
| 矩阵缓存 | 相同权重/音频，只增加一种发音或改变 G2P 有效输出 | fingerprint 改变，旧结果不命中 cache |
| 矩阵预检 | any manifest 无顶层 label，但完整 keyword_labels | --dry-run 成功；有效标签覆盖同实际 evaluator；不调用旧 loader 误拒绝 |
| 配置组合 | 基础 experiment + 新 keyword_eval group + eval_condition | compose 后模式/targets 正确，不覆盖模型架构或丢失 prep 其他字段 |
| 功能边界 | 在明确不支持的两阶段/LibriPhrase 入口请求 any | 清楚报错，而非忽略配置继续单词评测 |

数值测试至少分两层：

- CPU 使用轻量可控 encoder + 真实 QbyT，各 query 同时跑旧单查询参考与新共享编码路径；assert raw logits、概率和阈值结果。FP32 默认 `atol=1e-5, rtol=1e-5`。因平台出现偏差先定位原因，不以放宽阈值隐藏索引/掩码问题。
- pooling v4.1 必须设置非零 relative bias，至少做一次 Adam 更新后再 `eval()+no_grad()` 比较；零初始化 bias 下通过不足以证明没有 fast-path 问题。
- CUDA/真实 Zipformer：先 FP32，同输入同版本 checkpoint 对比；再按支持情况验证部署 AMP。FP16 可用 `atol=2e-3, rtol=2e-3` 作为起始上限，记录最大误差及临界阈值附近差异；不以数值误差要求强行保证浮点临界样本完全相同。
- 除这些真实数值比较外，用 fake verifier 隔离验证 OR 和计数逻辑，两类测试不能相互替代。

**12. 验收命令与运行示例**

以下命令都是实现后的验收目标，新 overlay 和新测试由 Grok 创建。工作目录固定：

```bash
cd /Users/e4/Documents/dma-kws
```

纯 CPU / mock 的新功能测试建议固定为这些文件：

```bash
.venv/bin/python -m pytest -q \
  tests/test_keyword_set.py \
  tests/test_keyword_set_manifest.py \
  tests/test_stage2_multi_query.py \
  tests/test_multi_keyword_musan_fa.py \
  tests/test_keyword_set_provenance.py
```

随后运行受影响的现有测试：

```bash
.venv/bin/python -m pytest -q \
  tests/test_stage2_clip.py \
  tests/test_stage2_verifier.py \
  tests/test_inference_manifest.py \
  tests/test_eval_two_stage_kws.py \
  tests/test_stage2_reporting.py \
  tests/test_musan_fa.py \
  tests/test_two_stage_musan_fa.py \
  tests/test_scan_stage2_thresholds.py \
  tests/test_plot_musan_fa_curve.py \
  tests/test_batch_eval_stage2_noise_matrix.py \
  tests/test_fit_stage2_calibration.py \
  tests/test_qbyt_pooling.py \
  tests/test_qbyt_bounded.py \
  tests/test_qbyt_model.py \
  tests/test_qbyt_readout.py \
  tests/test_config.py
```

完成后按项目要求跑全套和独立编译检查。全套已有可选依赖失败见 AGENTS.md，必须区分基线失败和本次回归；不要把任何新失败随意归为缺依赖。若 smoke 在 pytest 阶段退出，编译仍需单独执行：

```bash
.venv/bin/python -m pytest tests -q
.venv/bin/python -m compileall -q dma_kws qbyt scripts
git diff --check
```

Hydra 组合检查（无需真实模型推理）：

```bash
PYTHONPATH=. .venv/bin/python scripts/eval_stage2_clips.py \
  +experiment=icefall_zipformer_stage2_eps_softmin_v41 \
  +keyword_eval=multi_wakeup \
  --cfg job --resolve
```

真实整段评测：

```bash
PYTHONPATH=. .venv/bin/python scripts/eval_stage2_clips.py \
  +experiment=icefall_zipformer_stage2_eps_softmin_v41 \
  +keyword_eval=multi_wakeup \
  prep.manifest=/path/to/keyword_set_eval.jsonl \
  prep.stage2_ckpt=/path/to/stage2_model.pt \
  prep.output_dir=outputs/eval_stage2_multi_wakeup \
  prep.batch_size=16 \
  prep.keyword_eval.query_batch_size=64 \
  demo.qbyt_threshold=0.5 \
  run.device=cuda
```

若只评测 hey eva 两种发音，将上述 `+keyword_eval=multi_wakeup` 改成 `+keyword_eval=hey_eva_variants`。model experiment 必须与实际 checkpoint readout/spec 一致；示例选择 v4.1 不代表可以给 v6/v7 权重套同一个 experiment。

真实 MUSAN 滑窗评测：

```bash
PYTHONPATH=. .venv/bin/python scripts/eval_musan_fa.py \
  +experiment=icefall_zipformer_stage2_eps_softmin_v41 \
  +keyword_eval=multi_wakeup \
  prep.stage2_ckpt=/path/to/stage2_model.pt \
  prep.musan_root=/path/to/musan_eval \
  prep.output_dir=outputs/musan_multi_wakeup \
  prep.window_sec=3.0 \
  prep.hop_sec=3.0 \
  prep.batch_size=16 \
  prep.keyword_eval.query_batch_size=64 \
  demo.qbyt_threshold=0.5 \
  run.device=cuda
```

分片一致性验收：

```bash
PYTHONPATH=. bash scripts/eval_musan_fa_shards.sh \
  --python .venv/bin/python \
  --num-shards 2 \
  --gpus 0,1 \
  +experiment=icefall_zipformer_stage2_eps_softmin_v41 \
  +keyword_eval=multi_wakeup \
  prep.stage2_ckpt=/path/to/stage2_model.pt \
  prep.musan_root=/path/to/musan_eval \
  prep.output_dir=outputs/musan_multi_wakeup_sharded \
  prep.window_sec=3.0 \
  prep.hop_sec=3.0 \
  prep.batch_size=16 \
  prep.keyword_eval.query_batch_size=64
```

离线扫描直接消费聚合结果，不重新推理；以下预算仅用于展示参数，实际阈值须在独立验证集选择：

```bash
PYTHONPATH=. .venv/bin/python scripts/scan_stage2_thresholds.py \
  outputs/eval_stage2_multi_wakeup --mode clips --max-fpr 0.001

PYTHONPATH=. .venv/bin/python scripts/scan_stage2_thresholds.py \
  outputs/musan_multi_wakeup --mode musan --max-fa-per-hour 0.1
```

噪声矩阵先 dry-run，再移除 --dry-run 跑实际评测：

```bash
PYTHONPATH=. .venv/bin/python scripts/batch_eval_stage2_noise_matrix.py \
  --manifest /path/to/keyword_set_eval.jsonl \
  --model v41=/path/to/stage2_model.pt::icefall_zipformer_stage2_eps_softmin_v41 \
  --condition clean \
  --condition stationary_snr10 \
  --override '+keyword_eval=multi_wakeup' \
  --override prep.keyword_eval.query_batch_size=64 \
  --output-root outputs/noise_matrix_multi_wakeup \
  --dry-run
```

Grok 应实际验证这些命令的参数解析/配置组合；不能把本计划的示例未加验证地复制到最终 README。真实路径缺失时用小型临时 fixture 验证 CLI 入口，明确真实推理未执行。

**13. 小规模性能与召回验收**

先保证数值正确，再测性能：

- 固定模型、同一批真实音频、fbank/stream/padding、precision、batch size，比较逐 query 独立评分参考与共享 encoder 路径。
- 使用 P=1、2、8 个有效发音序列；分别记录实际有效 query 数、音频数、encoder 调用次数、墙钟耗时、clips/s、CUDA peak allocated memory。
- GPU 测量在预热后开始，计时段前后同步；不要每个 pair 强制同步再宣称真实吞吐。计时范围包括/不包括 fbank 必须明示，独立对照使用同一范围。
- 当显存不足，分别调小音频 batch 和 query_batch_size；不能缩短/改写音素、修改输入 padding、关闭某个词来伪装完成。
- 功能验收不要求召回率一定提高；报告两种 hey eva 发音相对于单发音在固定 FPR/FA/h 下的召回，以及每个词的结果。测试样本不足时明确说明统计范围，不宣称覆盖所有口音。
- 拓展词表后发现误触发上升属于需要通过阈值/数据衡量的行为，不应擅自改成平均分来压低误触发。

**14. 完成定义与 Grok 最终交付格式**

只有配置、评分、标签、结果、MUSAN、离线工具和矩阵均贯通，才能称功能完成。禁止以“runner 支持多查询”替代整个评测链路完成。

最终交付应包含：

1. 改动文件与最终接口；从同 text 的两种发音到一条音频结果的最小示例。
2. 实际执行过的命令、测试通过/失败/跳过数量；基线缺依赖与新问题分开列出。
3. 旧单词路径数值回归、真实 QbyT 各 family 回归、encoder 调用次数和分片一致性证据。
4. 一组可运行 CLI 命令、示例配置、示例 JSONL/CSV；README 和本计划的实施记录。
5. 真实 checkpoint / GPU / AMP / 真实口音数据验收状态。缺少资源时给精确补验命令，不能写“全部验证通过”。
6. 若实现偏离本文，逐条说明触发原因、替代实现和验证证据；不得静默降低边界条件或验收要求。

文档创建状态：计划已完成，功能尚未实施。上文提及的先前 4 项测试是旧路径核查，不能计入本计划新功能验收。

**15. 计划编写阶段的验证记录（非功能验收）**

2026-09-09，Codex 仅新增本计划 Markdown 文件，没有改动实现代码、训练配置或模型权重。

- OpenKnowledge 文档 lint：0 error、0 warning；文档引用的现有本地文件链接均解析成功。
- 3 段 YAML 示例经 OmegaConf 解析通过，4 条 JSONL 样例经 JSON 解析通过，10 段 shell 示例经 bash -n 语法检查通过。
- 在系统临时目录创建 keyword_eval overlay，使用现有 compose_config 验证 `icefall_zipformer_stage2_eps_softmin_v41 + keyword_eval=multi_wakeup + eval_condition=clean` 组合通过；检查 mode、两个词、hey eva 两种发音、v4 readout/sink_token 和原 prep.manifest 字段保留。临时目录退出后清理，未把拟新增配置提前写入仓库。
- 这些检查不代表新功能存在，也不代表命令已完成真实推理；新功能测试、checkpoint/GPU、真实音频召回与 FA/h 均由实施阶段完成。

**16. 实施记录（2026-09-09）**

implementation complete / GPU validation pending.

Shipped path: `prep.keyword_eval.mode=any` enrolls targets in `dma_kws/inference/keyword_set.py`, scores unique pronunciations with one encoder pass (`Stage2Verifier.score_clip_feats_multi_with_logits`), then max-aggregates. Clip eval writes `eval_protocol=stage2_clip_keyword_set`; MUSAN windows write `stage2_window_keyword_set`. Overlays: `configs/keyword_eval/hey_eva_variants.yaml`, `configs/keyword_eval/multi_wakeup.yaml`.

Minimal clip result: two hey-eva pronunciations scoring 0.2 and 0.9 at threshold 0.5 → one row, `qbyt_score=0.9`, `detected=true`, `matched_keywords=["hey eva"]`, `best` is the second pronunciation, no top-level `keyword`.

CPU commands run:

- `.venv/bin/python -m pytest -q tests/test_keyword_set.py tests/test_keyword_set_manifest.py tests/test_stage2_multi_query.py tests/test_multi_keyword_musan_fa.py tests/test_keyword_set_provenance.py` → **77 passed**.
- Affected existing suite from this plan → **354 passed**, **1 failed**: `tests/test_stage2_verifier.py::test_stage2_verifier_resamples_before_candidate_slicing` (`lhotse` missing; AGENTS.md baseline).
- Full `.venv/bin/python -m pytest tests -q` → **1769 passed**, **6 skipped**, **4 failed**, all four AGENTS.md optional-dep baselines (`whisper` ×3, `lhotse` ×1). `transformers` freeze-encoder and `test_musan_fa.py::test_run_file_windows_counts_and_spans` did not fail here. No new any-mode failures.
- Hydra compose (twice): `PYTHONPATH=. .venv/bin/python scripts/eval_stage2_clips.py +experiment=icefall_zipformer_stage2_eps_softmin_v41 +keyword_eval=multi_wakeup --cfg job --resolve` → `prep.keyword_eval.mode=any`, targets `hey eva` (two pronunciations) and `ok lamp`, `stage2.qbyt_readout.mode=eps_softmin` / `sink_token: true`, `prep.manifest` retained.
- `.venv/bin/python -m compileall -q dma_kws qbyt scripts` and `git diff --check` → exit 0.

QbyT CPU numerical layer: pooling v2/v3/v4/v4.1, bounded v5, v6, v7 vs per-query reference at `atol=1e-5, rtol=1e-5`, including v4.1 after a non-zero relative-bias Adam step. Encoder is called once per audio batch; QbyT pair chunks respect `query_batch_size`. Fake-verifier tests lock OR/counting, skip-at-threshold-0, label non-leakage, and P=1 match to `per_row`.

Not run here (no local Zipformer checkpoint / MUSAN corpus / CUDA job):

```bash
PYTHONPATH=. .venv/bin/python scripts/eval_stage2_clips.py \
  +experiment=icefall_zipformer_stage2_eps_softmin_v41 \
  +keyword_eval=multi_wakeup \
  prep.manifest=/path/to/keyword_set_eval.jsonl \
  prep.stage2_ckpt=/path/to/stage2_model.pt \
  prep.output_dir=outputs/eval_stage2_multi_wakeup \
  demo.qbyt_threshold=0.5 run.device=cuda

PYTHONPATH=. .venv/bin/python scripts/eval_musan_fa.py \
  +experiment=icefall_zipformer_stage2_eps_softmin_v41 \
  +keyword_eval=multi_wakeup \
  prep.stage2_ckpt=/path/to/stage2_model.pt \
  prep.musan_root=/path/to/musan_eval \
  prep.output_dir=outputs/musan_multi_wakeup \
  demo.qbyt_threshold=0.5 run.device=cuda
```

AMP, real Zipformer CUDA, and accent/MUSAN production FA/h remain GPU validation pending. Implementation note: G2P is constructed only when a target omits `pronunciations`.

Follow-up CPU-contract tests (same session): scan at threshold 0.8 ignores saved `detected`; matrix fingerprint changes when a pronunciation is added to the same overlay; `build_keyword_set_result_record` rejects a forged top-level `keyword`; clip runner stamps `eval_protocol=stage2_clip_keyword_set`; LibriPhrase `main` rejects `mode=any`; plot_musan carries/rejects mixed `keyword_set_id`.

Codex P2 follow-up (same branch, 2026-09-09): any-mode scan/plot refuse a `per_row` or identity-incomplete summary before using its hours; matrix precheck/fingerprint compose uses the same later-wins override order as the executed command (`common` then `model`); empty CSV cells and JSONL `null` label fields are stripped so mixed labeled/unlabeled manifests write; schema-3 and schema-4 per_row provenance compare equal after migration, including `mine_hey_eva_hard_negatives._validate_comparable_results`. Keyword-set modules **81 passed**; plan-affected existing suite **367 passed**, **1 failed** (`lhotse` verifier resample; AGENTS.md baseline). `compileall` and `git diff --check` exit 0. GPU/real Zipformer/MUSAN still pending.

