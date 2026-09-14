---
title: Grok 执行计划：v4.1 QbyT sink attention 可视化与消融诊断
description: CSV 兼容、attention 采集、时间轴、sink 干预、本地报告及数值验收的完整实施合同。
tags:
  - qbyt
  - stage2
  - attention
  - implementation-plan
status: implemented
baseline_commit: 6d81fae7e306057a83f69ff0e37b2b607eeef937
created: 2026-09-13
---
本文可直接交给 Grok 实现，不依赖此前对话。目标是用真实 v4.1 QbyT checkpoint 观察噪声与 sink attention 的关系，并通过同权重的推理干预检验其对唤醒分数的影响。

**交付要求：完成实现、测试、CSV 示例和使用说明，不要只重新生成计划。本文中的新增文件、接口、配置和命令是待实现合同。**

基线：2026-09-13，Git HEAD 为 `6d81fae7e306057a83f69ff0e37b2b607eeef937`。编写时已有未跟踪 `output/`，应保留。执行前重新核查工作树，遵守 [AGENTS.md](../../../AGENTS.md)，代码定位优先 CodeGraph。当前文档没有运行真实模型实验，也没有证明 sink 已学会拒绝噪声。

**1. 范围、里程碑和完成定义**

必做版本包含 A、B、C：

| 阶段 | 工作 | 完成门槛 |
| --- | --- | --- |
| A | CSV 兼容、checkpoint 校验、逐层逐头 attention 采集 | 普通与采集前向的 logits 数值一致，mask 和样本索引正确 |
| B | 短音频图、干净/加噪配对、sink 屏蔽实验、本地 HTML | 可以沿输入、attention、EPS position logits 到最终分数追踪单样本 |
| C | 批量标量统计、长度分层、错误样本筛选、运行身份和使用说明 | 原有 per-row 评测 CSV 可直接诊断，无诊断列也能完整输出 |
| D，后续扩展 | 长录音滑窗、部署事件协议、FA/h | 本次不作为完成门槛；不得给未实现选项返回伪结果 |

本次只支持 `prep.keyword_eval.mode=per_row`、完整短音频和 FP32 eager 推理。每行一个关键词和一条有效发音；同词多发音可以分行诊断，保留身份，不能在本工具中暗中做 max 聚合。`any` 模式明确报错，不能把目标集合标签解释成某一个词的标签。现有 any-mode 的真实区别见 [manifest.py](../../../dma_kws/inference/manifest.py) 的 `load_keyword_set_manifest`；相关历史方案为 [多关键词评测计划](./2026-09-09-stage2-multi-keyword-pronunciation-eval-grok.md)，当前代码优先于历史计划。

本次不修改训练 loss、优化器、sink 参数、readout 公式和 checkpoint 版本，不重新训练或拟合校准器。使用用户准备好的干净/加噪 WAV；不在线随机增强。若配置启用 `audio_aug` 或 `musan_mix`，报明确信息要求使用已导出的固定音频，不能静默忽略。异常不能用虚构低分代替。不实现后台服务、网站部署或上传音频。

**2. 已核实的代码依据**

| 当前事实 | 权威源及入口 |
| --- | --- |
| 评测用 Hydra，per-row 调用共享 manifest loader；支持 calibration、padding、stream policy | [eval_stage2_clips.py](../../../scripts/eval_stage2_clips.py)：`run_eval`、`_resolve_audio_padding_ms` |
| CSV 必填列为 audio_path、keyword；相对路径按 CSV 父目录解析；label 可选，空 label 被删除；额外列保留 | [manifest.py](../../../dma_kws/inference/manifest.py)：`load_manifest`、`_normalize_row` |
| 已有离线诊断采用 runner 注册音素、ClipFeatureDataset、verifier 诊断入口 | [probe_qbyt_emissions.py](../../../scripts/probe_qbyt_emissions.py)：`_probe_clips` |
| 音素解析接受空格分隔 ARPAbet，也接受 JSON 字符串数组；显式空序列报错 | [stage2_clip.py](../../../dma_kws/inference/stage2_clip.py)：`parse_phoneme_sequence`、`resolve_keyword_phonemes`、`enroll_phonemes` |
| verifier 已传完整 qbyt_score 给加载检查；内部 encode_for_qbyt 包含 encoder 和可选 adapter | [stage2_verifier.py](../../../dma_kws/inference/stage2_verifier.py)：`Stage2Verifier.__init__`、`Stage2Model.encode_for_qbyt` |
| sink 可学习，联合序列重排后是有效文本、sink、有效音频、padding | [pooling.py](../../../qbyt/pooling.py)：`QbyT._forward_impl`、`repacked_modality_and_index` |
| attention 使用 nn.TransformerEncoder；relative bias 含模态 pair table 和音频相对位置项 | [pooling.py](../../../qbyt/pooling.py)：`QbyT.__init__`、`RelativeAttentionBias.compute` |
| EPS 最终 position_logits 来自 final_pos_fc，而 seq_fc 是另一辅助头 | [pooling.py](../../../qbyt/pooling.py)：`forward_with_readout_details`、`_forward_impl` |
| pooling 的 spec 解码、旧 checkpoint 兼容和 spec 比较已经有公共实现 | [checkpoint_io.py](../../../dma_kws/training/checkpoint_io.py)：`checkpoint_qbyt_readout_spec`、`assert_qbyt_readout_version` |
| 当前默认 readout version 是 7，诊断必须加载对应的真实 v4 配置 | [stage2 默认配置](../../../configs/stage2/default.yaml)、[prep 默认配置](../../../configs/prep/default.yaml) |

复用上述接口。不要误修改 `qbyt/models/attention.py`：本任务的采集对象是 pooling 模型里的 PyTorch `self_attn`。现有 emission 诊断模块针对其他 readout 机制，新 attention 诊断独立放置。

**3. CSV 输入合同**

主入口直接调用 `load_manifest(prep.manifest)`，本工具要求后缀为 `.csv`。不得另写一套 CSV 基础解析器。保留 UTF-8、标准 CSV quoting、原始行序、额外字段和相对路径规则。支持无标签数据；label 有值时诊断入口额外限定为 0/1，拒绝其他整数。不能默认 label=0。

| 列 | 必填 | 定义 |
| --- | --- | --- |
| audio_path | 是 | 与原评测相同；相对路径相对清单目录 |
| keyword | 是 | 当前行匹配的关键词 |
| label | 否 | 1=包含该行关键词，0=不包含；空值为未标注 |
| keyword_phonemes | 否 | 原有空格分隔 ARPAbet 格式；直接走共享 parser |
| sample_id | 否 | 提供时必须非空且唯一；空/缺失时自动生成稳定行 ID |
| condition | 否 | 建议 clean/noisy/noise_only/unrelated_speech/near_miss；允许自定义字符串，缺失归 unknown |
| pair_id | 否 | 同一基准语音与其不同加噪版本的组标识 |
| keyword_spans | 否 | JSON 数组，源音频秒坐标，区间采用 [start,end) |
| noise_spans | 否 | 同上；指已知加噪/噪声区间，不等价于严格的非语音标注 |
| pronunciation_id | 否 | 展示标签；内部身份另以实际 token 序列计算 |

最低兼容清单：

```csv
audio_path,keyword,label
audio/clean_001.wav,hey eva,1
audio/noise_001.wav,hey eva,0
audio/unlabeled_001.wav,hey eva,
```

扩展清单（路径为示意，不宣称音频存在；此例所有行用 G2P，省略 keyword_phonemes 整列）：

```csv
audio_path,keyword,label,sample_id,condition,pair_id,keyword_spans,noise_spans
audio/clean_001.wav,hey eva,1,clean_001,clean,p001,"[[0.8,1.6]]",[]
audio/noisy_001.wav,hey eva,1,noisy_001,noisy,p001,"[[0.8,1.6]]","[[0.5,2.0]]"
audio/noise_001.wav,hey eva,0,noise_001,noise_only,,,"[[0.0,2.0]]"
audio/speech_001.wav,hey eva,0,speech_001,unrelated_speech,,,
```

发音注意：缺少 override 用现有 G2P；显式提供的空 `keyword_phonemes` 不可自行改成另一套 fallback。核对原 `run_batch` 的分支，并让诊断与原评测对同一行采取相同处理。JSON 空数组、非法音素和丢失重音标记不能静默修正。

诊断扩展列由 `validate_attention_manifest_rows` 在共享 loader 后解析；不改变其他评测消费者。时间区间必须是有限数对，满足 0 <= start < end <= 源音频时长。空单元格表示无标注，`[]` 表示空集合；两者可显示区别，但都不能证明整段无噪声。非标准 JSON NaN/Infinity 必须拒绝。源时长需解码后再校验。重叠区间做集合并集统计，不重复计帧。

自动 ID 建议 `row_000002`（CSV 记录号，报告同时记录 manifest 指纹）；重复音频路径允许，因为不同关键词/发音是独立样本。用户 sample_id 不直接作为路径：用内部安全文件 ID 生成输出，防止特殊字符或同名覆盖。

配对键至少包含 pair_id、keyword、实际 token 序列；每组要求唯一 clean 基准，允许多个 noisy 变体。缺基准、多个 clean 或序列不一致时输出 pair_status 与原因，正常单样本诊断仍继续，不按 CSV 顺序猜配对。

**4. 配置与 CLI 合同**

沿用 Hydra 的 `config_name=config` 和现有 `resolved_config`、`run.device`、`prep.manifest`、`prep.stage2_ckpt`、`prep.stage2_calibration`、`prep.output_dir`、`prep.batch_size`、`prep.num_workers`、padding、`demo.qbyt_threshold`。

在 [prep 默认配置](../../../configs/prep/default.yaml) 新增以下命名空间，其他入口不因新增默认字段改变行为：

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
```

正常分支 normal 永远执行，不需要出现在 ablations 列表。空列表只采集正常行为。`block_sink_each_layer` 展开为实际层数个单层干预。采集层/head 选择只影响采集与展示；不能意外限制消融范围。传层号/head 号时用 0-based 整数列表，拒绝 bool、负数、重复和越界，运行记录保存展开后的明确列表。

诊断专用入口遇到 prep.batch_size <= 0 取 1，num_workers <= 0 取 0；不把普通评测的默认 64 复制进来。`prep.amp=off` 是本版要求；显式请求 fp16/bf16 报 unsupported，而非静默改配置。`mode` 只接受 clips；D 阶段实现之前拒绝 windows。已有 prep 下没有的 limit 等字段不随意添加：本版处理全清单，报告只限制可视化样本数。

group_field 默认 condition 且列缺失时归 unknown，以兼容最低 CSV；用户指定其他不存在的列应报错。拒绝阈值非有限数、bool 或不在 [0,1]。

padding 必须与 eval_stage2_clips 相同，包括默认值推导。若需要共享解析 helper，将其小范围下沉至 inference 模块，并让旧入口调用，添加回归；不要从 scripts 互相 import。

**5. 模型和分数兼容性**

按 `Stage2ClipRunner.from_config -> Stage2Verifier -> build_qbyt` 加载，保留 strict state_dict 和既有 stream/readout guard。已有完整 qbyt_score 已传入 guard，无须另建加载器。

通过公共 checkpoint spec 解码后要求真实 version=4、family=pooling、mode=eps_softmin、sink_token=true；从模型核对 sink 参数存在。完整比较 temperature、text_position、audio_position、relative_num_buckets、relative_max_distance。允许 spec 合法的不同位置配置，将其显式显示在报告；不要把架构变体统称成已验证默认配置。默认 v7 配置遇到 v4 checkpoint 必须失败，不能悄悄补标 v4 或按权重形状猜 temperature。

旧 checkpoint 先走仓库公共兼容解码；能确定的字段沿用规范，不另造强制迁移。仍缺少可确定的必要分数语义时停止，提示提供原配置/规范转换产物。LoRA 单独增量权重不在本次加载范围；需要提供现有转换链路导出的完整可推理 checkpoint。

运行中不写 checkpoint、不 stamp 7/8、不重置 sink。单次预处理和 encoder/adapter 输出供 normal、全部消融及 parity 对比共用；不同音频变体仍各自编码。跨多行相同音频缓存不作为首版硬要求。

normal 的 raw logit、原校准器输出、原阈值独立保存。消融仍使用同一个校准器和阈值，报告称“原校准变换后的干预分数”，不声称它在干预后仍已校准。EPS 位置分数取 final_pos_fc，不能用 seq_fc 替代。

**6. Attention 采集实现合同**

拟新增 `dma_kws/inference/qbyt_attention_diagnostics.py`。优先临时实例 hook，不复制整个 Transformer 前向，也不全局 monkey-patch PyTorch 类。

建议 API（命名可在小范围调整，数据合同不可省略）：

```python
@dataclass(frozen=True)
class AttentionCaptureSpec:
    layers: tuple[int, ...]
    heads: tuple[int, ...]
    save_full_attention: bool = False
    max_combined_tokens: int = 1024
    max_attention_bytes: int = 268435456

@dataclass(frozen=True)
class SinkAblationSpec:
    name: str
    blocked_layers: tuple[int, ...] = ()

def capture_pooling_attention(
    qbyt, speech, anchors, speech_lengths, anchor_lengths,
    *, capture_spec, ablation_spec
) -> list[SampleAttentionTrace]:
    ...

# Stage2Verifier 新增，负责 padding、一次编码、parity、normal/消融编排和校准
def attention_diagnostics(
    self, feats, keyword_ids_batch, *, capture_spec, ablations,
    parity_atol=1e-5, parity_rtol=1e-4
) -> list[SampleAttentionDiagnostics]:
    ...
```

采集协议：

1. 输入模型必须是 eval、FP32 eager。低层 API 对 training 模型报错，不能悄悄切换训练状态。
2. 在进入 QbyT 前向之前安装 `register_forward_pre_hook(..., with_kwargs=True)` 与 forward hook。包装器正确处理 positional/keyword 参数，避免同一参数重复传入。
3. pre-hook 对采集层设置 `need_weights=True, average_attn_weights=False`。预先核对本地 PyTorch 的实际调用签名。forward hook 读取第二个输出，不更换 attention output。
4. 禁用 MHA/encoder fused fastpath，确保 hook 未被跳过；选中每层必须恰好采到一次。不要通过 train() 禁用 fastpath。与现有 `_additive_float_mask_encoder` 嵌套兼容。
5. 使用原始 additive relative bias 和 padding mask；不能只根据 QK 点积重算一张“近似 attention”替代实际前向。
6. 权重在有效 query 行上的 key 维求和应约为 1（eval，无 dropout）。无效 query 的输出不用于可视化或统计。
7. 每层立即按样本裁剪、detach、转 CPU；没有 save_full_attention 时不留存完整矩阵。
8. `finally` 删除本次安装的全部 hook，恢复进入前 fastpath 状态；部分安装后异常也清理。禁止删除调用者已有 hook。
9. 本工具模型推理单线程串行；禁止同一模型嵌套 capture 和并发诊断。fastpath 是进程级状态，文档说明不适合嵌入并发在线服务。
10. 不新增 parameter/buffer，不更换 module/state_dict key；测试进入前后 state_dict 数值完全不变。

对样本 b，令 U=anchor_lengths[b]，T=speech_lengths[b]（未加 sink 的原始长度）：

```text
text queries/keys = [0, U)
sink key index   = U
audio            = [U+1, U+1+T)
valid keys       = [0, U+1+T)
padding          = 后续位置
```

索引不能使用 batch 最大文本长度；不能将模型内部加了 1 的 speech_lengths 再误当成真实音频长度。

每条 SampleAttentionTrace 至少包含：

| 字段 | shape / 说明 |
| --- | --- |
| layer_ids、head_ids | 实际原始索引，不能用切片后的下标替代 |
| audio_to_sink | [selected_layers, selected_heads, T] |
| text_to_sink | [selected_layers, selected_heads, U] |
| text_to_audio | [selected_layers, selected_heads, U, T] |
| position_logits | [U]，有效文本位置的最终 EPS raw logits |
| audio_frame_index、audio_time_sec（可选） | [T]；时间不可验证时只存帧索引 |
| report_spectrogram、spectrogram_time、spectrogram_frequency（选中样本） | 实际 prepared 波形的绘图数据，供离线重绘；不重新调用另一套前端 |
| raw_logit | scalar |
| text_length、audio_length、sink_index | 正确的逐样本边界 |
| row_sum_max_error、padding_mass_max | 截取前计算的归一化诊断 |
| full_attention，可选 | 每条样本每层/head 裁到有效 L×L，不含 batch padding |

文本视图只展示 sink+audio 是原 attention 的切片；不要再归一化。缺少的 text→text 概率质量正常存在，可额外显示 `text_key_mass`，避免读者误以为图上每行应总和为 1。

**7. Parity 与 sink 消融**

每批执行次序：

```text
同一预处理输入
  -> encode_for_qbyt 一次
  -> 无 hook 的 forward_with_readout_details：基准
  -> normal capture：与基准比较
  -> block_sink_all
  -> block_sink_layer_0 ... layer_N-1
  -> 保存/汇总并释放本批特征
```

normal 采集失败或 raw logit/有效 position_logits 不满足 `abs(a-b) <= atol + rtol*abs(b)` 时，本次运行标为 failed，停止解释性汇总；记录最大误差和样本身份。测试可额外验证原普通 forward 与 details 接口 logits 一致。比较仅覆盖有效文本位置；不拿 padding 占位值当 EPS。

消融必须在实际 self-attn softmax 前屏蔽选定层的 sink key 列。将原 mask 克隆后按样本、所有 head、所有 query 把该列设置 -inf；兼容原 2D/3D mask 和 bool/additive 表示，保持 key_padding_mask 语义。不得原地污染原始 shared attn_bias、下一层、后续 normal 或其他样本。不要将 sink 参数清零，不删除 token，不移动位置，也不屏蔽 sink query 整行。

block_sink_all = 所有 Transformer 层、所有 head 屏蔽 sink key。block_sink_layer_i = 只在第 i 层屏蔽。未直接屏蔽的下游层可能因上游表示改变而改变 attention，不要求其数值等于 normal。

主指标：

```text
delta_raw_logit = ablated_raw_logit - normal_raw_logit
delta_qbyt_score = ablated_qbyt_score - normal_qbyt_score
delta_position_logits = ablated_position_logits - normal_position_logits
detected = qbyt_score >= demo.qbyt_threshold
```

标注为负的样本若 delta_raw_logit > 0，只能说切断 sink 读取后分数上升。结合多条样本、正样本损失和层级干预判断，不能仅凭 attention 较高推断 sink 在“吸收噪声”。本实验不能证明单个 sink 容量足够，也不等价于重新训练无 sink 模型的对比。

本版拒绝空关键词，encoder 有效帧为 0 按不可评分处理，避免屏蔽后全 key 被 mask 的 NaN。单测仍应对非法 mask 状态明确报错。

**8. 时间坐标、预处理和配对差值**

复用 `ClipFeatureDataset` 的实际重采样、prepare_waveform、fbank 和 padding 链路，见 [stage2_clip.py](../../../dma_kws/inference/stage2_clip.py)。声谱图应来自同一个实际输入波形，不能另载音频后使用另一套重采样/归一化参数。可利用已有 waveform_observer 保存预 padding 的 prepared 波形并记录左右 padding。

每条记录保存 source sample rate/duration、模型 sample rate、fbank frame_shift/frame_length/snip_edges、padding、encoder 有效长度、stream policy 和时间轴推导方法。

严禁用 duration/T 均匀铺满来假装精确时间轴。实现时从实际 encoder/subsampling 的定义和既有时间辅助函数推导 nominal frame center 或支持区间，覆盖前端边界偏移和右裁剪。无法验证的 encoder 用 encoder_frame_index 展示；标为 time_axis_status=unavailable，此时不计算 noise_spans 区间统计和配对时间差值。模型分数和 attention 仍可输出。

映射成功后：

- 源坐标 = 模型输入坐标 - 左 padding 秒数（clips 模式）。
- 人工 padding 对应的位置另外标识；其参与模型 key 的实际行为不能擅自改变。
- 对 noise_spans 用 nominal center 的区间包含规则生成 mask，明确这是位置对应，不是精确感受野归因。
- 无匹配帧的区间指标为缺失，输出 valid_frame_count=0；不能填 0 当作 sink 不关注。
- 噪声区间的补集称 outside_noise_annotation，不称 clean_speech，因为其中可能有静音或未标注噪声。

同 pair 的干净和加噪样本只有在源时长、采样/前端、token 序列、有效帧数和 nominal 时间网格相容时，才计算逐帧 `noisy - clean`。否则只比较标量，保存 pair_status 原因；不自动插值拉伸来制造对齐。用户的 pair_id 是对应关系声明，报告注明来源，不当成算法验证的语音对齐。

**9. 图表、统计与本地 HTML**

新增 `qbyt_attention_report.py`；静态科学图使用 Matplotlib 导出 PNG，HTML 使用少量本地 JS 做样本/层/head 选择和表格排序。无需 React 或在线 CDN；双击 report.html 离线可读，数据内嵌或本地静态资源加载，不依赖 file:// 下被浏览器禁止的 fetch。转义文本/路径，不能把 CSV 内容当 HTML 注入。

每条可视化样本展示：

- 标题：sample_id、关键词及实际音素、condition、label、readout spec、normal 分数/阈值。
- 声谱图：关键词/噪声区间、padding 标记；如时间映射不可用，明确分离时间轴与帧轴。
- audio→sink 热力图：横轴时间/encoder 帧，纵轴 layer×head。
- text→sink+audio 热力图：可切换层/head，按实际音素标签排列，sink 独立一列。
- EPS position_logits 及消融变化。
- normal/all/single-layer 消融配对分数。
- 有效配对样本的 noisy-clean attention 差值，正负色标对称。

统一主 attention 色标 [0,1]。小权重允许附加缩放视图，但要注明上下限，不能取代主图；配对差值使用共享对称限值。保留逐头图，均值作为辅助视图。PNG 标签不重叠、不裁断，中文字体缺失时可用英文图内标签，音素必须保持可读。导出 DPI 可配置。

统计：

- 对每条样本、每层/head：S_audio=有效音频 query 对 sink 权重均值；S_text=有效文本 query 对 sink 权重均值。
- 有标注且映射可用时：S_noise、S_outside_noise_annotation、各区间 valid_count。
- 分组箱线图与 attention 对 normal 分数的散点图，标签缺失样本不参与误判判断。
- 按 condition/label、音频长度 bin、文本长度分层；length_bins 最后一个边界以上还有溢出 bin。分组时先算每样本均值，避免长音频贡献更多权重。
- sink 是一个 key，单列 attention 天然受有效 key 数影响；不设通用“sink attention > 0.5 即成功”标准。
- 不把相关性写成因果证明；自动报告只陈述观察量和干预分数变化。

max_report_samples 只限制重型单样本图，不限制评分、CSV 或组统计。选择规则确定且写入 run.json：优先有标签的误判，再 valid pair 成组选择，再各 condition 按输入顺序补齐；小配额无法容纳配对组时明确不展示配对图。完整 records 中记录 report_selected，报告显示全部样本数和入选数。

**10. 输出数据格式与可追溯性**

```text
outputs/diagnose_qbyt_sink/<run>/
  report.html
  records.csv
  attention_metrics.csv
  position_scores.csv
  pairs.csv
  summary.json
  run.json
  traces/<internal_sample_id>__<ablation>.npz
  figures/...
```

run.json 或按 sample_id 索引的内嵌 metadata 必须保存每样本时间映射、标注和绘图轴信息；选中样本 NPZ 保存重绘所需声谱图，离线重绘不依赖再次取得原始音频。若 save_traces=false，说明仅可查看本次已生成图。

采用多张关系表，不在一张表里混用“样本粒度”和“head 粒度”。CSV 使用 csv.DictWriter、newline=""；JSON 数组字段用 json.dumps 后交由 writer 转义。NaN/Inf 不落盘，缺失标量为 CSV 空单元格、JSON null。NPZ 只存 numeric/string arrays，不使用 object array，读入 allow_pickle=False。

records.csv：一条输入样本 × 一个 ablation 一行。至少包含 run_id、sample_id、manifest_record_number、audio_path、keyword、实际 keyword_phonemes、query_id、condition、pair_id、label、ablation、blocked_layers、qbyt_raw_logit、qbyt_score、threshold、detected、delta_raw_logit、delta_qbyt_score、text_length、audio_length、status、skip_reason、trace_path、report_selected。normal 的 delta=0；skipped 的所有分数字段为空，detected 为空。

attention_metrics.csv：sample_id × ablation × layer × head × region，一行保存 mean/min/max、query_count、key_count、row_sum_error；region 为 audio/text/noise/outside_noise_annotation。未采集层/head 不伪造值。

position_scores.csv：sample_id × ablation × position，保存音素、position_logit、与 normal 差值。pairs.csv：基准 sample_id、变体 sample_id、匹配键、pair_status、正常分数差、时间网格可比性；无效配对仍留原因。

run.json 至少保存 schema_version、created_at、git_commit/dirty、manifest SHA256、checkpoint SHA256、字典与校准器指纹、实际 token 序列、完整 resolved 配置/readout spec、stream policy、padding、设备/dtype/PyTorch 版本、展开后的采集/消融参数、时间映射方法、容差、随机种子（若有）、limits、输入/配置指纹。元数据不要包含凭据或无关环境变量。

summary.json：输入数、成功/跳过/失败数、未标注数、最大 parity 误差、配对有效性统计、分组指标、图表与输出索引、status=complete/failed。空清单、全跳过也生成可读报告，不能除零。

输出目录已有本工具结果时默认拒绝覆盖，提示换目录；不复用来源不明的 NPZ。逐批保存临时产物后原子 rename；失败保留已完成记录与失败状态，不生成貌似完整的 summary。渲染阶段应支持从本工具保存的 run.json/CSV/NPZ 重新生成图，不重跑模型（建议 Python API，非强制新增 CLI）。

**11. 资源边界与错误处理**

attention 请求权重会产生 O(B×H×L²) 临时张量，保存裁剪结果不等于消除计算峰值。预检按 batch padding 后的 L 计算 float32 权重大小，并至少乘 4 的临时张量系数；max_attention_bytes 表示本诊断 attention 工作区的保守预算估计，不是总 GPU 显存保证。总内存还包含 encoder 激活、bias 和 CPU traces，运行日志分别记录估计量与实测峰值（设备支持时）。

- 超 max_combined_tokens 或预算：若批次拆分能解决则按样本拆分；单样本仍超限则保留 skipped=resource_limit，不能裁短音频却仍记原标签/时长。
- 完整 attention 只用于小样本，保存前继续检查预算。
- 一批完成后释放大矩阵；全清单只保留标量索引，NPZ 增量写入。max_report_samples 不控制模型评分数量。
- save_traces=false 时仍能本次渲染选中样本；之后缺少矩阵的离线重绘应明确拒绝相应图。
- 配置错误、checkpoint 不兼容、非法 CSV、非有限 logits、hook 未触发、parity 失败：退出非零并标记失败。
- 音频太短和资源上限：按带原因的 skipped 处理。文件缺失/损坏默认退出失败，防止悄悄改变样本集。
- Matplotlib/绘图依赖在需要绘图时惰性导入；不因诊断功能改变普通训练/推理依赖。遵守 [pyproject.toml](../../../pyproject.toml) 的现有依赖组织；若缺绘图库，添加最小可选 extra 并给出安装命令。
- CPU 支持是必需，GPU 是相同数值合同下的额外验收。不要承诺固定速度提升。

**12. 文件级实施清单**

下表为拟新增或改动范围，实施后更新实际路径和函数名。新增路径当前不存在，不应在实现前引用为现有能力。

| 文件 | 具体工作 |
| --- | --- |
| dma_kws/inference/qbyt_attention_diagnostics.py，新建 | dataclass、context hook、索引裁剪、mask 干预、parity、指标汇总；对真实 pooling QbyT 可直接 CPU 测试 |
| dma_kws/inference/qbyt_attention_manifest.py，新建 | 共享 load_manifest 后的诊断列校验、ID、pair、区间解析；只处理新增规则 |
| dma_kws/inference/qbyt_attention_report.py，新建 | 时间元数据消费、PNG、本地 HTML、确定性图表选择、离线重绘 API |
| dma_kws/inference/stage2_verifier.py，修改 | 增加 attention_diagnostics 公共方法，复用编码、readout 和校准；旧评分接口保持返回结构 |
| dma_kws/inference/stage2_clip.py，必要时小改 | 只为复用 padding/时间信息或注册 API 做小范围 helper 提取，不另写 fbank/音素流程 |
| scripts/diagnose_qbyt_sink.py，新建 | Hydra 入口、预检、dataset 编排、CSV/NPZ/JSON 输出、报告调用 |
| configs/prep/default.yaml，修改 | 加 sink_diagnostics 配置和注释，不改变既有入口默认行为 |
| tests/test_qbyt_attention_diagnostics.py，新建 | 真实小 QbyT 的数值采集/干预/恢复测试 |
| tests/test_qbyt_attention_manifest.py，新建 | CSV 和扩展列、配对、标签、路径兼容 |
| tests/test_qbyt_attention_report.py，新建 | 时间轴、区域统计、输出/重绘、转义与空样本 |
| tests/test_diagnose_qbyt_sink.py，新建 | 使用真实小 QbyT + 假 encoder 的 CLI/编排集成测试 |
| tests/test_stage2_verifier.py，增补 | 新方法编码次数、校准和旧接口回归 |
| tests/fixtures/qbyt_sink_manifest_minimal.csv，新建 | 最低原格式示例；单测生成临时对应音频 |
| tests/fixtures/qbyt_sink_manifest_extended.csv，新建 | 扩展 CSV quoting 示例；不得把示例音频路径当真实数据 |
| docs 下使用说明，新增 | 实现后写实际命令、图表解释、CSV 示例、资源限制及真实验收状态 |

原则上不需要改 qbyt/pooling.py；若 hook API 在目标 PyTorch 无法满足合同，可在 pooling 内增加显式 analysis-only 入口，但必须说明原因，复用同一前向、保持所有旧调用及 state_dict 完全兼容，并通过相同 parity 测试。不得复制另一份 Transformer 计算链作为捷径。

**13. 必须通过的测试**

数值测试使用小型真实 pooling QbyT，不以 AST 或 mock 返回固定 attention 代替。fixture 采用 version=4、eps_softmin、sink=true、learned text position、relative audio bias；relative pair table/audio bucket 赋非零值。固定种子、eval、CPU FP32；同时增加合法 sinusoidal 变体验证 mask 分支。

| 编号 | 测试 | 验收标准 |
| --- | --- | --- |
| T01 | 普通 forward、details、normal capture | raw logits 与有效 EPS logits 在指定容差内一致 |
| T02 | 短文本/短音频单独与混合 batch | 有效 attention 切片和 logits 一致，sink 索引取逐样本 U |
| T03 | 非零相对偏置、padding | hook 不遗漏 bias；padding key 质量约 0，有效 query 行和约 1 |
| T04 | 采集层/head 选择 | 正确保留原 layer/head ID；每层恰好一次，无遗漏 |
| T05 | sink_all / sink_layer_i | 被干预列约 0，行和约 1；只干预指定层；不错误要求下游不变 |
| T06 | mask 克隆与恢复 | 干预后再次 normal 与初次 normal 一致，其他样本未被污染 |
| T07 | 生命周期 | 安装/forward/保存抛异常均移除本次 hook；fastpath 初始开/关均恢复，既有 hook 保留 |
| T08 | checkpoint/状态 | 不支持版本、无 sink、spec mismatch 报错；state_dict 参数与 buffer 完全未变 |
| T09 | 编码/校准 | 一批各消融共用一次 encoder/adapter 输出；raw logit、校准器、阈值及 delta 符号正确 |
| T10 | CSV 兼容 | 同一原格式清单，两入口音频绝对路径、row 序、音素 token、label、正常分数一致 |
| T11 | 缺失/错误字段 | 空 label 不转负例；非法 label/音素/NaN 区间/重复 ID/any 模式报清晰错误 |
| T12 | 区间/配对 | 左 padding、前端 offset、边界、重叠区间只计一次；无有效帧不填 0；不匹配 pair 不插值 |
| T13 | 空/短/超限 | 空 CSV、全跳过、零 encoder 帧和超预算有完整状态，无除零/伪分数 |
| T14 | 输出合同 | CSV 正确 quoting，NPZ allow_pickle=False 可读，JSON 严格有限值，run 指纹完整 |
| T15 | 报告 | 图显示正确方向/标签；HTML 离线可用、文本转义、选择确定，图数量限制不减少 records |
| T16 | 回归 | 原 pooling、manifest、verifier、其他 readout 的相关既有测试通过或给出基线失败证据 |

不要写“噪声输入必须提高 sink attention”这样的单测。合成随机权重只用于实现正确性；真实 checkpoint 的机制结论来自独立数据。

定向验收（在仓库根目录，新增文件完成后执行）：

```bash
.venv/bin/python -m pytest \
  tests/test_qbyt_attention_diagnostics.py \
  tests/test_qbyt_attention_manifest.py \
  tests/test_qbyt_attention_report.py \
  tests/test_diagnose_qbyt_sink.py \
  tests/test_qbyt_pooling.py \
  tests/test_qbyt_model.py \
  tests/test_qbyt_readout.py \
  tests/test_inference_manifest.py \
  tests/test_keyword_set_manifest.py \
  tests/test_stage2_verifier.py \
  tests/test_checkpoint_io.py -q

.venv/bin/python -m py_compile \
  dma_kws/inference/qbyt_attention_diagnostics.py \
  dma_kws/inference/qbyt_attention_manifest.py \
  dma_kws/inference/qbyt_attention_report.py \
  dma_kws/inference/stage2_verifier.py \
  scripts/diagnose_qbyt_sink.py
```

随后按 [AGENTS.md](../../../AGENTS.md) 执行 `.venv/bin/python -m pytest tests -q`。缺少 whisper/transformers/lhotse 等可选依赖的已知失败应按项目要求验证基线，不能仅凭清单断言是旧问题，不安装大依赖来掩盖不相关问题。保留用户改动，不自动 stash 用户文件。smoke 早停不能代替 compile 检查。

**14. 真实运行与对照命令**

以下命令在实现完成后使用；ABS 路径和 v41_run 是用户实际配置/数据的占位符。v41_run.yaml 应为该 checkpoint 的原始完整配置或可正确组合的 Hydra 配置。不要拿当前默认 v7 配置仅覆盖 version=4 来“修好加载”。若原配置需额外 experiment 等 overrides，普通评测与诊断使用同一组 overrides。

先运行原评测，固定 FP32、batch=1、同一 CSV、校准器和 padding：

```bash
.venv/bin/python scripts/eval_stage2_clips.py \
  --config-path /ABS/PATH/TO/V41_CONFIG_DIR \
  --config-name v41_run \
  prep.manifest=/ABS/PATH/TO/manifest.csv \
  prep.stage2_ckpt=/ABS/PATH/TO/v41.pt \
  prep.output_dir=/ABS/PATH/TO/baseline_clips \
  prep.keyword_eval.mode=per_row \
  prep.batch_size=1 prep.num_workers=0 prep.amp=off \
  run.device=cpu
```

再运行诊断：

```bash
.venv/bin/python scripts/diagnose_qbyt_sink.py \
  --config-path /ABS/PATH/TO/V41_CONFIG_DIR \
  --config-name v41_run \
  prep.manifest=/ABS/PATH/TO/manifest.csv \
  prep.stage2_ckpt=/ABS/PATH/TO/v41.pt \
  prep.output_dir=/ABS/PATH/TO/sink_diagnostics \
  prep.keyword_eval.mode=per_row \
  prep.batch_size=1 prep.num_workers=0 prep.amp=off \
  ++prep.sink_diagnostics.mode=clips \
  ++prep.sink_diagnostics.capture_layers=all \
  ++prep.sink_diagnostics.capture_heads=all \
  ++prep.sink_diagnostics.ablations='[block_sink_all,block_sink_each_layer]' \
  ++prep.sink_diagnostics.max_report_samples=40 \
  run.device=cpu
```

`++` 兼容旧保存配置尚无新命名空间；入口合并自身默认诊断参数并严格校验未知键。如有校准文件，两个命令都加相同 `prep.stage2_calibration=/ABS/PATH/TO/calibration.json`；padding 同理。GPU 命令按现有 run.device 约定替换，并保留 FP32；不要声称本计划已实测 GPU 命令。

Grok 应提供实际执行后的最终可复制命令，而非继续保留占位路径。若没有真实 checkpoint 或音频：完成临时 WAV、真实小 QbyT、假 encoder 的端到端 smoke，输出演示报告并显著标“合成 fixture / 非训练模型结论”；列出真实 CPU/GPU 验收待补，不编造路径或结果。

独立检查脚本或测试将 baseline 的 results.jsonl 与 records.csv 中 normal 按输入记录身份一一对齐。比较 qbyt_raw_logit、qbyt_score、detected、有效音素 token；重复 audio_path 不能仅按路径 join。需记录与 baseline 的最大误差和任何阈值翻转；raw parity 容差内的临界阈值翻转也必须报告，不能因容差通过隐藏。

真实样本验收建议先 40–60 条：干净正例、其固定加噪配对、纯噪声、无关语音；再加入近音词与真实误唤醒。样本数是启动建议，不是统计置信度保证。人工检查至少一组 pair、一个噪声误报、一个加噪正例、一个无标注样本和一个被跳过样本。

小样本报告只给样本分数/判定与分组差异，不计算 FA/h；D 阶段必须复用窗口划分、触发合并、冷却、keyword_set/protocol 身份和有效负音频时长，不能以重叠窗口数除源音频小时数声称部署误唤醒事件率。

**15. Grok 执行顺序与最终交付清单**

按依赖完成，可分提交但不要在完成 A 后将整项任务标完成：

1. 重新核对源码/配置、工作树和测试基线；确认本计划中的接口没有变化。
2. 实现 CSV 扩展校验和最小真实 QbyT capture；先通过 T01–T04。
3. 加入 mask 消融、恢复和编码复用；通过 T05–T09。
4. 接入 Hydra、ClipFeatureDataset、CSV/NPZ/JSON；通过 T10–T14。
5. 实现时间映射和本地报告，人工查看 PNG/HTML；通过 T15。
6. 完成相关回归与全套测试、compile；记录失败边界。
7. 有真实模型/数据则执行第 14 节对照；无资源则交付 fixture 验证与真实验证待办。
8. 更新使用说明，最终提供改动文件列表、实际命令、测试结果、报告位置、最大 parity 误差、未验证事项。

最终验收必须同时满足：

- 原 CSV 不增加必填列即可使用，标签和音素含义一致。
- hook 开启前后数值一致，干预后可恢复普通评分。
- sink key 列与 attention 方向正确，逐层逐头可读。
- 图表和记录能追溯到真实配置、音频、token、层/head。
- 不以高 sink attention 自动宣判抑噪成功；不虚构 GPU/真实数据结果。
- 此次交付包含可运行实现及文档，训练与常规评测语义保持原合同。
