---
title: "Grok 执行计划：v4.1 背景负样本预处理与加载优化"
description: "可独立执行的实现合同：裁剪级 fbank 缓存、分片读取、采样优化、测速和兼容性验收。"
tags: [ qbyt, stage2, performance, implementation-plan ]
status: implemented
baseline_commit: 49b15ca
---

本文件是给 Grok 的实施任务，不依赖此前对话。目标是在 V100 32GB 单卡上降低 v4.1 背景负样本的数据等待和重复特征计算开销。用户报告约 0.54 it/s，旧 v4 一次训练约 3 小时；这两项是用户提供的背景，尚未经本机复测。执行时先核对当前代码，再按本计划完成实现、测试和可运行的短测速。

实现记录与使用说明：[Stage II 背景 fbank 缓存：使用与验收记录](./2026-09-05-stage2-background-cache-notes.md)。

**1. 执行边界与已确认事实**

- 实施数据预处理、数据加载、元数据缓存和测速工具。保留在线路径作为默认，新增显式启用的有限裁剪 fbank 缓存。
- 保持 v4.1 的 QbyT 架构、读出版本与 spec、loss、优化器、学习率、有效 batch、正负样本门控、MUSAN 划分。长度分桶、encoder 输出缓存、attention 改写、FP16 特征压缩均不在本轮范围。
- 当前背景路径已按需读取片段，并在 worker 中复用特征提取器；不要把它们当成待新增能力。参考 [背景采样与读取](../../../dma_kws/stage2/features.py)。
- 背景 gate 位于约 50% 的负样本分支，默认条件概率 0.25；保持正样本不被替换、背景 label=0、query_seq 为空及原序列标签逻辑。参考 [Dataset](../../../dma_kws/stage2/dataset.py)。
- 缓存的是“具体波形裁剪的 fbank”，不是整条录音 fbank 的任意切片。当前 Icefall 配置为 80 维、16 kHz、Lhotse、dither=0、snip_edges=false；所有解析仍走公共 [fbank 配置](../../../configs/fbank/icefall_kws.yaml) 和 [FbankExtractor](../../../dma_kws/stage2/fbank.py)。
- 有限缓存不等同于无限在线随机裁剪。保持录音抽样权重、时长生成规则和标签语义，明确记录有限裁剪库的种子、容量和重复率，不宣称逐样本训练轨迹相同。
- 执行前遵守 [AGENTS.md](../../../AGENTS.md)；有 CodeGraph 时先用它定位代码。保留用户现有改动；编写计划时工作区已有 .gitignore 修改和未跟踪的 .superpowers/。
- 执行者应完成下述必做任务，不停留在重新规划；普通实现选择按本文默认值处理。缺少 GPU 或真实数据时仍完成实现和 CPU fixture 测试，准确列出未执行的硬件验收。不启动完整训练，不自动推送或发布。

**2. 配置合同：先固定接口再实现**

在 [schema.py](../../../dma_kws/configs/schema.py) 中扩展 Stage2BackgroundNegativeConfig，新增 Stage2MetadataCacheConfig 并挂到 Stage2Config；同步修改 [默认 YAML](../../../configs/stage2/default.yaml)、[Dataset 配置白名单](../../../dma_kws/stage2/dataset.py)、训练构造以及 [LoRA replay 构造](../../../dma_kws/stage2/adapt.py)。后者必须能透传新增配置，但本轮不修改 LoRA 算法。

```yaml
stage2:
  background_negative:
    enabled: false
    probability: 0.25
    audio_list_path: ""
    duration_seconds_min: 1.0
    duration_seconds_max: 3.0
    mode: online                 # online | fbank_cache
    cache_manifest: ""
    max_open_shards: 8            # 正整数；每个 worker 独立
  metadata_cache:
    max_entries: 128              # clips/distances 共用 LRU；0 禁用
    max_bytes: 33554432           # 32 MiB / worker，和条目上限同时生效
```

- 缺省字段保持旧配置可用；online 行为保留原始 RNG 调用顺序。enabled=false 时不打开音频、manifest 或分片。max_entries 为非负整数，max_bytes 与 max_open_shards 为正整数；max_entries=0 禁用元数据缓存。拒绝未知 mode、非法概率/时长、非法容量；bool 不视为有效整数容量。
- online 要求非空且有效的 audio_list_path。fbank_cache 要求 cache_manifest，允许 audio_list_path 为空；若非空，只读取列表并与 manifest 的来源目录记录匹配，不要求原音频仍在训练机器上，也不调用旧的逐音频存在性检查。
- fbank_cache v1 只接受 dither=0；非零显式报错，不能偷偷设为 0 或冻结随机 dither 后声称语义一致。
- 启动时验证缓存版本、训练 split 身份、规范化 fbank 参数、时长范围、FP32 dtype、特征维度、索引/分片边界。缺文件或不匹配报清晰错误；不得退回 online。
- 不把 source_mode 写入 qbyt_readout，不改变 checkpoint 模型兼容规则。将 cache_id、manifest 路径、缓存格式、裁剪数和规范化特征配置写入已有 resolved config / run record；复用 [运行记录入口](../../../dma_kws/stage2/train.py)，不要另造一套实验日志。
- 在 configs/experiment/ 新增 icefall_zipformer_stage2_eps_softmin_v41_cached.yaml，继承 [现有 v4.1 overlay](../../../configs/experiment/icefall_zipformer_stage2_eps_softmin_v41.yaml)，只设置 mode=fbank_cache、audio_list_path=""、manifest 的 processed_root 默认路径，以及独立 run_name/log/checkpoint 路径。原 v4.1 overlay 继续默认在线。
- 调参仍使用已有 [DataLoader 字段](../../../dma_kws/training/loaders.py)，不要增加冗余线程池、异步后台生成器或新的存储依赖。

**3. T1：提取背景裁剪合同，保持在线兼容**

修改 [features.py](../../../dma_kws/stage2/features.py)，新增 dma_kws/stage2/background_sampling.py。后者仅负责源信息、裁剪描述和裁剪物化，不依赖 Lightning 或 QbyT。

建议公开接口（名称照用，类型细节可按项目风格补全）：

```python
@dataclass(frozen=True)
class BackgroundSourceInfo:
    source_id: str
    path: Path
    sample_rate: int
    num_frames: int
    channels: int

@dataclass(frozen=True)
class BackgroundCropSpec:
    source_id: str
    duration_seconds: float
    read_start_frame: int
    read_num_frames: int
    target_num_samples: int
    final_offset: int

def draw_crop_spec(source, duration_seconds, *, rng) -> BackgroundCropSpec: ...
def materialize_crop(source, spec) -> tuple[torch.Tensor, int]: ...
```

实现要求：

1. TrainingBackgroundSampler.extract 继续按“uniform 时长 → choice 录音 → 抽读取起点 → 最终截取/重复相位”消耗传入的 dataset RNG。在线分支通过公共函数物化裁剪，再调用原 FbankExtractor。
2. 保留原实现中的 duration 微秒取整、源采样率下 ceil 读取长度、round 目标样本数、长录音 seek、短录音周期重复、final_offset 抽样行为；不要把它们简化为看似等价的整毫秒裁剪。final_offset 同时描述读取结果的最终截取起点或短录音重复相位。
3. 多声道仍先取均值成 mono，然后在原裁剪完成后由 FbankExtractor.prepare_waveform 重采样；不要先重采样整条录音。
4. 源 header 信息按 worker 惰性缓存；验证实际读取与 spec 相容。不得为抽一个片段整段读取长录音。
5. 预生成库时 source 已确定，因此调用 draw_crop_spec 的单源路径；它保持同一条件采样规则，但无需模拟在线选择其他录音的 RNG 消耗。
6. TrainingNoiseAugmenter 与普通语音读取继续可用。公共 _load_audio 若改签名，只增加可选参数，保留现有调用合同。
7. 先从当前在线算法建立独立 fixture/golden 对照，验证重构后同 seed 的波形、长度和 RNG 后续状态；不要用两次调用同一个新函数作为唯一“等价测试”。

T1 完成标准：旧背景/加噪测试通过，长/短录音及非整毫秒时长的真实波形对照通过；此时 online 仍可训练。

**4. T2：预生成有限裁剪库，确定存储格式**

新增 dma_kws/data_prep/stage2_background.py 和 scripts/prepare_stage2_background.py。脚本使用 argparse 包装现有 [compose_config/config_to_dict](../../../dma_kws/config.py)，通过 --experiment 和可重复 --override 解析训练 fbank 及背景时长，禁止复制一套特征默认值。

固定 CLI：

```text
--split-dir PATH                 required；读取已有 split.json / train_background.list / eval_musan.list
--output-dir PATH                required；新缓存目录
--experiment NAME                default icefall_zipformer_stage2_eps_softmin_v41
--override KEY=VALUE              可重复；交给 compose_config
--crops-per-recording INT         default 128；>=1
--seed INT                       default 2025
--workers INT                    default 4；>=1
--shard-size-mib INT              default 256；>=1
--verify-only                    校验已有 output-dir 后退出，不生成
```

来源约束依据 [MUSAN split 实现](../../../dma_kws/data_prep/musan_split.py)：

- 消费现有 split，不能重新划分；只取 train_background.list。确认 train 与 eval 录音不相交，列表 catalog/count 与 split.json 一致。v4.1 构建入口要求 train_categories 仅 music/noise；speech 仍全部在评估侧。
- 不对 music/noise 再做 50/50 采样；保留按录音均匀抽样时的类别权重。不静默跳过损坏录音或重复源，错误时报告源路径并失败。
- 对 N 条源录音各生成 K 个裁剪，总数 N*K；运行时先均匀抽 source，再均匀抽该 source 的裁剪，有放回。不同源的时长不会改变被选概率。
- 用 SHA256(seed, source_id, crop_ordinal) 派生每个裁剪的局部 RNG 种子，禁止 Python hash()。任务分配和完成顺序不能改变裁剪或索引；按稳定 source_id、ordinal 顺序写出。
- source_id 使用 MUSAN 根目录下相对路径的稳定标识，manifest 保存来源相对路径和构建时的 canonical list entries；缓存内部路径始终相对 manifest。运行时移动整个缓存目录仍可加载。
- 预生成任务按录音组织，worker 内惰性创建 FbankExtractor；worker 的 Torch CPU 线程先设为 1。使用现有 Python 多进程能力，限制在途任务/结果数量，不把全部特征收集回内存。
- 每个 crop 按 T1 物化后计算 fbank，检查二维、80 维（以实际配置为准）、非空、有限值，保存原始 FP32。
- 不默认生成源波形缓存，不改成整条音频 fbank 后切片，不启用 FP16 或有损压缩。

缓存格式 v1：

```text
background_cache/
  manifest.json
  recordings.jsonl
  crops.npy
  features-00000.npy
  features-00001.npy
  ...
```

| 文件 | 必需合同 |
| --- | --- |
| manifest.json | format_version=1、split_role=train、cache_id、seed、K、source/crop 总数、时长范围、规范化 fbank 字典、dtype=float32、feature_dim、特征库/库依赖版本、split/list/source/索引/分片 SHA256 与大小 |
| recordings.jsonl | 稳定排序；source_index、source_id、来源标识、源采样率/帧数/声道、源内容 SHA256、该源 crop 起始行和数量 |
| crops.npy | allow_pickle=False 的结构化数组；crop_id、source_index、ordinal、T1 的完整 crop spec、shard_index、frame_offset、num_frames；索引单位明确为特征帧，不是字节 |
| features-*.npy | allow_pickle=False；C contiguous FP32 数组，shape=[total_frames_in_shard, feature_dim]，多个变长 crop 首尾连接 |

cache_id 为规范化 manifest 内容（排除 cache_id 自身与构建时间等非语义字段）的 SHA256，包含引用的 source/index/shard 摘要，保证身份随输入、裁剪和特征配置变化。只把最末一片写成实际长度，不保留未初始化区域。

创建与验证规则：

- 构建只写同一文件系统中的新 staging 目录，完成检查后以原子 rename 发布；已存在的 output-dir 默认拒绝覆盖，错误不得破坏已有缓存。
- 本轮不做断点续建；失败的本次 staging 可清理，重新运行使用新目录。--verify-only 必须支持对完整索引及全部分片做 SHA256、形状/offset/finite 校验。
- 构建时完整计算摘要；训练启动只校验 manifest 身份、recordings/index 摘要、分片头/大小和全部索引范围，避免每次训练扫描数 GB 分片内容。文档明确：完整位损坏检测由 --verify-only 提供。
- K=128 是工具默认值，不是训练质量结论。文档给出容量估计：背景抽样数约为 optimizer_steps × accumulation × batch_per_gpu × world_size × 0.5 × probability。默认单卡 10k/128/accumulation=1 时约 160k 次；按 N 和可用磁盘选择 K，初步可取 max(128, ceil(2*160000/N))。记录实际重复率，不声称这个估计能消除重复。
- 当前 80 维/10 ms shift 的 FP32 特征每秒约 32 kB，平均 2 秒时 160k 条约 10 GB（粗估，含量取决于实际帧数；不含额外索引）。预处理开始前打印预计容量。

T2 完成标准：小型真实音频 fixture 串行/多进程生成得到相同语义 manifest、crop index 和特征内容；构建失败无半成品可被训练误认，verify-only 可识别破损产物。

**5. T3：分片读取与训练集成**

新增 dma_kws/stage2/background_cache.py，提供：

```python
class BackgroundFeatureCache:
    def __init__(self, manifest_path, *, expected_fbank_kwargs,
                 duration_seconds_min, duration_seconds_max,
                 audio_list_path="", max_open_shards=8): ...
    def extract(self, *, rng: random.Random) -> torch.Tensor: ...
    def read_crop(self, crop_id: int) -> torch.Tensor: ...
    def close(self) -> None: ...
```

- __init__ 完成 T2 定义的轻量启动校验，不创建 GPU Tensor，不打开全部 feature mmap；source/index 可保留紧凑只读数组。
- 每次 extract 只使用调用者传入的 RNG 做 source/crop 选择；不在 reader 中私自创建同 seed 的 RNG，不在不同 worker 间共享游标。
- 每个 worker 惰性 np.load(..., mmap_mode="r", allow_pickle=False)，用 LRU 限制最多 max_open_shards 个打开映射。用 PID guard 和 __getstate__/__setstate__ 清除父进程映射/句柄，兼容 fork 与 spawn。
- 只复制选中 crop 到可写 contiguous CPU Tensor 再返回；不复制整片。返回 Tensor 不得引用随后会被 LRU 关闭的 mmap，否则会产生失效内存或只读写入问题。
- missing shard、非法 index、dtype/维度不符直接报带文件名和 crop_id 的错误，无 online fallback。实现显式 close 和异常路径清理。
- 在 LibriPhraseTrainDataset 初始化中选择 sampler，__getitem__ 的正负 gate、空 query_seq、seq_label 保持原逻辑；常规 batch key/类型保持现有合同。缓存路径不调用音频解码器，也不调用 FbankExtractor。
- 修改 schema、默认 YAML、cached overlay、训练 run record 和 LoRA replay 配置透传。验证配置序列化后仍包含原 qbyt_readout_version=4 和完整 v4.1 spec。
- 保持同 seed、同配置、同 rank/worker 拓扑下可复现；不同 worker/rank RNG 不重放。明确现有 worker-based RNG 不保证改变 worker 数量或重启中途训练后逐样本完全一致，本轮不新增这种承诺。

T3 完成标准：删去测试 fixture 原音频后，仅凭缓存仍能产生合法背景样本；缓存的配置白名单、disabled/online/cached 三条路径和 LoRA replay 全部覆盖。

**6. T4：消除非背景元数据的重复工作**

修改 [dataset.py](../../../dma_kws/stage2/dataset.py) 及相关构造点。

1. get_negative 使用 r=rng.randrange(N-1)，index=r + int(r>=excluded_index)，保持排除自身的均匀分布；N<2 时在真正需要普通负采样时给出明确错误，不能造一个自身负例。预计算 anchor 数量。
2. clips/distances 共用 worker 内 LRU，受 max_entries 和 max_bytes 双重限制；0 entries 为禁用。对象数组的字节预算必须包含字符串/dict 等 payload，不能只用 object-array.nbytes。转换为只读紧凑结构或实现去重的递归大小估计，超大单项直接读取使用但不纳入缓存。
3. 元数据缓存只缓存文件内容，不缓存每次随机抽到的 clip。命中/未命中均在相同位置消耗 RNG；禁止为测试提速关闭包含完整关键词的负样本重抽逻辑。
4. 预存 ngram_g2p、clips_file、distances_file 等常用列，减少热路径 df.iloc；phoneme token 缓存也应有容量上限，避免多 worker 复制所有大对象。复用与 metadata_cache 相同的总预算或在其内部共享预算，不能创建第二个无界字典。
5. 语音 fbank 不额外做无限 RAM 缓存。元数据文件按一次运行内不可变处理；变更文件后需重建 Dataset，这点写入使用文档。
6. 更新 train 和 adaptation 的构造透传，不修改共享 DataLoader helper 默认行为。max_entries=0 提供基线对照和回退。

T4 完成标准：固定 seed 的 cache on/off 采样结果一致、重复文件减少真实 np.load 次数、LRU 淘汰后输出仍正确、容量超限不驻留、N=1/2 边界和普通/困难负样本规则通过。

**7. T5：统一构造与可复现测速**

在 [train.py](../../../dma_kws/stage2/train.py) 提取 build_stage2_train_dataset(config, tokenizer) 和 build_stage2_train_dataloader(config, dataset)，正式训练与 benchmark 调用同一构造路径。保留现有 rank seed、worker_init_fn、shuffle、drop_last、persistent_workers 语义。可放入新增 dma_kws/stage2/train_data.py 以避免 benchmark 引入整个训练流程，但只保留一个实现源。

新增 scripts/benchmark_stage2_input.py 和 dma_kws/stage2/input_benchmark.py。CLI 使用与 preparer 相同的 --experiment/--override：

```text
--mode background|loader|train   default loader
--warmup-batches INT             default 20
--measure-batches INT            default 200
--device cpu|cuda                default cpu；train/cuda 时必须真有 CUDA
--manifest PATH                 background 模式必填
--split-dir PATH                background 模式必填，定位原始源音频
--output PATH                   required；JSON
--profile                       可选单独诊断模式，输出 trace；不计入正常测速结果
```

三个层次不能混为一个“加速倍数”：

- background：按确定顺序选择同一组 manifest crop_id，比较在线重算这些 exact crop 与 read_crop，先校验特征/帧数一致，再报告读取/裁剪/重采样/fbank 分项时间。这里不混入随机长度差异。
- loader：只迭代真实 DataLoader，不建模型；计时 next(iterator) 的真实等待与 CPU 端有效帧/最大长度/样本组成。首次 next 与预热单列，以免 sample_lens 的大 shuffle 成本混入稳态。
- train：单 GPU 短跑真实 Stage2LightningModule 和已有 Trainer 精度/优化器配置；benchmark 单独关闭验证、保存 checkpoint 和外部 logger，保留实际 loss、AMP、梯度累积/裁剪设置；不能用一个简化 toy loss 替代。正式 train 入口行为不变。

输出合同：

- 记录 git commit、完整解析配置、cache_id、PyTorch/CUDA/设备、CPU 核数、workers、prefetch、batch、precision、accumulation、warmup/measure 实际完成数。
- wait_p50_ms、wait_p95_ms、warmup_seconds、measured_wall_seconds、samples_per_second、有效音频秒/s、帧长分布、padding_ratio=1-sum(lengths)/(B*max_length)、背景/正负计数、process RSS、cache hit/miss/open-shard 高水位。
- cache 模式可在诊断下附 crop_id/source_id，计算选中裁剪重复率；元信息保留 CPU，不将文件名、耗时字典送进 GPU。常规训练关闭诊断时不增加统计开销。实现 profile-only 的 Dataset/Collate 包装器或可关闭观察器即可，不改常规 batch 合同。
- warmup-batches/measure-batches 均指 DataLoader microbatch。train 模式将两个窗口边界向上对齐到 accumulate_grad_batches 的整数倍，在完整 optimizer step 后结束，并报告请求数、实际数和 optimizer step 数；不把 Trainer.max_steps 直接当作 microbatch 数。background 模式按配置 batch_size_per_gpu 个 crop 组成一个计量批次。
- GPU 利用率可通过可用的 nvidia-smi 采样；不可用时标为 unavailable，不能写 0。普通端到端计时只在测量窗口边界 synchronize；CPU next 等待用真实迭代包装/Profiler 事件测量，不以 batch hook 间隔冒充数据等待。逐算子 profiler 和正常吞吐实验分开运行。
- 同 seed 不代表不同 source_mode 消耗相同 RNG；loader/train 对照必须报告各自的样本构成与长度。只有 background exact-crop 对照可声称输入相同。
- workers=[4,8]、prefetch=[2,4]，资源充足再测 12 workers；先单项 A/B，再组合。无需为此新增配置项。
- 测速使用独立结果路径，不覆盖用户训练 checkpoint，也不自动恢复某次已有训练。

T5 完成标准：background/loader 模式可在 CPU fixture 上运行并生成合同字段；有真实 V100、源数据和 encoder 权重时再完成 train 200 batch 测速。没有硬件不伪造收益。

**8. 任务依赖、文件清单与测试**

推荐执行顺序：T0 基线快照与现有定向测试 → T1 裁剪合同 → T2 缓存构建 → T3 读取/接入 → T4 元数据优化 → T5 最终 A/B。T5 的共享构造/最小测速骨架应在 T0 先落下，后续补齐缓存模式，避免事后无法重跑原路径。阶段对应可独立审阅的 diff；无需主动 git commit/push。

已有文件改动范围：

| 文件 | 修改目的 |
| --- | --- |
| [features.py](../../../dma_kws/stage2/features.py) | 共用背景裁剪，保留 online/加噪兼容 |
| [dataset.py](../../../dma_kws/stage2/dataset.py) | source mode、配置校验、LRU、O(1) 负采样、热路径列访问 |
| [train.py](../../../dma_kws/stage2/train.py) | 共用数据构造、缓存身份/run record |
| [adapt.py](../../../dma_kws/stage2/adapt.py) | replay 透传配置 |
| [schema.py](../../../dma_kws/configs/schema.py) / [默认 YAML](../../../configs/stage2/default.yaml) | typed fields 与默认值 |
| [training/loaders.py](../../../dma_kws/training/loaders.py) | 原则上复用；只在实际需要验证参数时最小改动 |
| [collate.py](../../../dma_kws/stage2/collate.py) | 原则上保持；诊断元数据由 benchmark 包装器处理 |
| [fbank.py](../../../dma_kws/stage2/fbank.py) / [musan_split.py](../../../dma_kws/data_prep/musan_split.py) | 复用；本轮不改特征算法与划分策略 |

新文件：dma_kws/stage2/background_sampling.py、background_cache.py、input_benchmark.py；dma_kws/data_prep/stage2_background.py；scripts/prepare_stage2_background.py、benchmark_stage2_input.py；cached experiment overlay；对应新测试。train_data.py 仅在提取公共构造需要时新增。

测试必须验证行为而非源码 AST。扩展既有 [背景测试](../../../tests/test_stage2_background_negative.py)、[Dataset 测试](../../../tests/test_stage2_dataset.py)、[加噪测试](../../../tests/test_stage2_training_noise.py)、[训练 helper](../../../tests/test_stage2_training_helpers.py)、[配置测试](../../../tests/test_config.py)，新增 tests/test_stage2_background_cache.py、test_prepare_stage2_background.py、test_stage2_input_benchmark.py。

必须覆盖：

1. online 重构前后的长/短/立体声/8k→16k/非整毫秒裁剪；具体 wave 样本值、shape、下一次 RNG 结果。独立 oracle 或重构前固定输出。
2. 同 exact crop 的 offline/online fbank 在同环境中 shape 相同、torch.testing.assert_close(rtol=1e-5, atol=1e-5)；失败先找边界/重采样差异，不能放宽容差掩盖错误。
3. 串行与多进程产物语义、索引和特征相同；source 均匀选择、每 source K、短源可用，不额外改变类别权重。
4. disabled/online/cache 配置、未知键、非零 dither、参数不符、缺分片、offset 越界、错误 dtype、verify-only SHA 不符、失败原子性。
5. reader 无 raw audio 依赖；把 _load_audio/FbankExtractor 替换成抛错函数，cache 提取仍成功。
6. 多 worker spawn 的 reader 可序列化；访问超过 max_open_shards，句柄高水位不越界；先前返回的 Tensor 在映射淘汰后仍正确。
7. worker/rank RNG 独立，固定拓扑 seed 可复现；元数据 cache on/off 不改变 clip 抽样与标签。
8. LRU 同时遵守 entries/bytes 上限，object payload 被计入；大项不驻留；普通负采样排除自己，N=1/2 有定义行为。
9. 背景不替换正样本、背景 query_seq 空及 CTC skip 原合同；v4.1 checkpoint spec 保持；LoRA replay 接受新增字段。
10. CPU benchmark 字段/单位/padding 计算正确；不得在普通训练热路径加入强制 CUDA 同步。

先运行与变更相关的测试，再按 [AGENTS.md](../../../AGENTS.md) 跑全套。已知可选依赖失败仍需核对实际栈并证明基线同样失败。保护现有用户改动，遵守项目关于定向 stash 复核的要求，记录 stash 标识并确保恢复；不要为测试清空用户工作区。

**9. 可直接执行的命令合同**

以下新命令在实现后必须能运行。示例路径均相对仓库根；执行者用本机真实路径覆盖 split、parquet、feature_root 和 init_checkpoint。已有入口支持 run.limit_steps，但本计划短测速使用独立 benchmark，不自动启动完整 10k 训练。

```bash
.venv/bin/python scripts/prepare_stage2_background.py \
  --split-dir data/dma-kws/processed/musan_split \
  --output-dir data/dma-kws/processed/stage2_background_cache \
  --experiment icefall_zipformer_stage2_eps_softmin_v41 \
  --crops-per-recording 128 --seed 2025 --workers 4 --shard-size-mib 256
```

先用 synthetic split 和 K=4 验证；生产缓存的 K 按 T2 的容量估计决定，记录实际选择及空间。命令默认 output-dir 已存在时失败。

```bash
.venv/bin/python scripts/prepare_stage2_background.py \
  --split-dir data/dma-kws/processed/musan_split \
  --output-dir data/dma-kws/processed/stage2_background_cache \
  --verify-only

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

train 模式须设置正确 encoder 初始化权重/特征路径；不能用随机 encoder 冒充正式配置吞吐。V100 对照双方都显式使用同一精度，示例为 16-mixed：

```bash
.venv/bin/python scripts/benchmark_stage2_input.py \
  --mode train --device cuda \
  --experiment icefall_zipformer_stage2_eps_softmin_v41_cached \
  --override stage2.precision=16-mixed \
  --override stage2.batch_size_per_gpu=128 \
  --warmup-batches 20 --measure-batches 200 \
  --output outputs/train-cached-v100.json
```

对 online 重复同一 train 命令，仅替换 experiment 和输出文件。cache 容量/元数据开关/workers/prefetch 的消融同样只改单一变量。

```bash
.venv/bin/python -m pytest -q \
  tests/test_stage2_background_negative.py \
  tests/test_stage2_background_cache.py \
  tests/test_prepare_stage2_background.py \
  tests/test_stage2_input_benchmark.py \
  tests/test_stage2_dataset.py \
  tests/test_stage2_training_noise.py \
  tests/test_stage2_training_helpers.py \
  tests/test_config.py

.venv/bin/python -m pytest tests -q
.venv/bin/python -m compileall -q dma_kws scripts
```

即使全套 pytest 因已确认的可选依赖失败，compileall 也要单独执行；不要只依赖会提前退出的 smoke shell。

**10. 最终交付与停止条件**

交付代码、测试、使用说明及 JSON 测速结果；说明文件放在本计划同目录或项目已有文档位置，通过项目要求的 Markdown 工具写入。产物至少包含：实际修改文件、已执行命令/结果、online/cache 行为差异、缓存创建/验证/切换方法、内存与磁盘预算、无法完成的硬件验证。

验收分三级，分别标记，不可混淆：

- 工程完成：T1–T5 必做实现、定向测试、公共配置兼容、CPU fixture、静态编译完成，且相关新代码测试不因缺少可选依赖而全部被跳过。
- 性能验证完成：真实数据/V100 的同配置 A/B JSON 已生成，报告稳态样本吞吐、数据等待、padding 和一次性预处理成本。只有数据等待确实下降才认定输入侧优化有效；如果 GPU 计算主导，报告事实并停止扩展范围，不临时改模型。
- 训练质量验证：有限裁剪库需要后续完整训练和既有 hard/低误报评估；本轮不自动启动，准确列为待验证。不得用短测速 loss 或只跑单元测试宣称精度无影响，也不承诺恢复到 3 小时。

如果缺真实数据、依赖或 CUDA，仅阻塞对应测试层，继续完成其余实现。只有无法推断的源路径/环境访问才请求一次必要信息，附具体缺失项；不重复询问本文已经固定的设计决策。
