# Stage II 直接消费 LibriPhrase-100 decoded parquet 音频

日期:2026-06-22
分支:`codex/two-stage-real-demo`

## 背景与问题

Stage II 的数据准备脚本 `scripts/prepare_stage2_libriphrase.py` 从 LibriPhrase 风格
聚合 parquet 的 `clips` 列读取相对音频路径(如 `LP-100/missus_rachel/103-1240-0000_000.wav`),
写进 `train.jsonl` 的 `wav_path` 字段。训练脚本 `scripts/train_stage2_qbyt.py` 在
`resolve_wav_path()` 里把相对路径拼到 `paths.libriphrase100_root` 下,再用
`torchaudio.load()` 读磁盘上的散装 `.wav` 文件。

但 Hugging Face 上 `ZhiqiAi/LibriPhrase-100` 仓库并不提供散装 `.wav` 文件:

- 聚合元数据 parquet(`aggregated_segments_by_ngram.parquet`、
  `aggregated_segments_with_g2p.parquet`、
  `aggregated_segments_with_g2p_distance.parquet`)只有 `ngram` / `clips` /
  `ngram_g2p` 等列,`clips` 仅是路径字符串映射,不含波形。
- 实际音频波形打包在 6 个 `LP-100-decoded-*.parquet` 分片(约 5 GB)里。

因此照现状直接跑两个脚本,会在训练时 `torchaudio.load` 找不到文件而失败。
本设计改造 prep 与 train 脚本,使 Stage II 直接消费 decoded parquet 的音频,
不再依赖散装 wav 文件。

## 关键数据事实(已核实)

decoded parquet 行结构(取自 HF datasets-server 样本):

```json
{
  "audio_rel": "the_free_press/7264-92316-0025_000.wav",
  "text": "the_free_press",
  "audio": [0.00323486, 0.00399780, ...],   // 已解码的 float 采样数组,非编码字节
  "sampling_rate": 16000,
  "num_samples": 10880,
  "duration": 0.68
}
```

聚合 parquet 的 `clips` 与 decoded 的 `audio_rel` 对应关系:

- `clips` 路径:`LP-100/the_free_press/7264-92316-0025_000.wav`
- `audio_rel`:`the_free_press/7264-92316-0025_000.wav`
- **`clips` 去掉 `LP-100/` 前缀 == `audio_rel`**

约束:

1. `audio` 是已解码 float 数组(已是 16 kHz、[-1, 1] 范围),不能用 `torchaudio.load`,
   需直接构 tensor。
2. 实际只用到被选中的 clip(正样本每 ngram 取 `clips[0]`,负样本每锚点随机取 1 个),
   远少于全量 5 GB —— 只需物化用到的 clip。

## 设计决策

经讨论确定:

- **桥接策略:prep 时落盘**(而非 train 时建内存索引或把数组内联进 JSONL)。
  重复训练快、训练时内存低、JSONL 仍可读。
- **落盘格式:`.npy`**(而非 `.wav`)。读写最快、无编解码损耗;代价是 trainer
  的 Dataset 需改成 `np.load`。

## 改动方案

### 1. prep 脚本 (`scripts/prepare_stage2_libriphrase.py` + `dma_kws/stage2/pairs.py`)

- 新增参数 `--decoded-parquet-root`(目录或单文件;默认取 `paths.libriphrase100_root`),
  指向 decoded 分片。
- **内存策略:先收集需要的 clips,再流式抽取。**
  1. 先按现有逻辑从聚合 parquet 构建 anchors 与 pair 列表,得到所有被引用的 clip
     相对路径集合。
  2. 把每个 clip 路径用 `removeprefix("LP-100/")` 规整为 `audio_rel` 键。
  3. 逐个扫 decoded 分片,只抽取命中集合的行,命中即把其 `audio` 数组以 float32
     落盘为 `.npy`,不让全量波形驻留内存。
- 落盘路径:`<processed_root>/stage2_qbyt/audio/<audio_rel>.npy`(float32,16 kHz)。
- 匹配不到的 clip:跳过并计数,结束时打印告警(不静默丢弃)。
- `pairs.py`:
  - `PairRecord.wav_path` 改写成对应的 `.npy` 相对路径(相对 `stage2_qbyt` 目录)。
  - 新增字段 `sample_rate`(写入 JSONL,便于 trainer 校验)。
  - `make_pair_records` 的正/负样本选择逻辑不变。

### 2. train 脚本 (`scripts/train_stage2_qbyt.py`)

- `resolve_wav_path` 的基准目录从 `libriphrase100_root` 调整为
  `<processed_root>/stage2_qbyt`(与 prep 落盘目录一致);绝对路径仍直接返回。
- `Stage2Dataset.__getitem__`:
  - `torchaudio.load(wav_path)` → `np.load(wav_path)` 得 float32 数组 →
    `torch.from_numpy(...)`,reshape 成 `(1, T)`。
  - 删除声道合并与 resample 分支(落盘已是 16 kHz 单声道)。
  - 保留 `kaldi.fbank` 特征提取。实现时核对:decoded 的 float 已是 [-1, 1],与原
    `torchaudio.load` 返回的 float 范围一致,故 `kaldi.fbank` 行为不变。
- 不再依赖 torchaudio 的解码路径,但 `torchaudio.compliance.kaldi` 仍需保留。

### 3. 测试

- `dma_kws/stage2/pairs.py` 纯函数:新增「clips 前缀规整 + audio_rel 匹配」单测
  (假 DataFrame,沿用 stage1 monkeypatch parquet 读取的风格)。
- prep 烟雾测试:小样本(`--limit-anchors`)跑通,断言生成 `.npy` 与 JSONL,
  且数组 shape 合理、`wav_path` 指向存在的 `.npy`。
- 不引入新依赖(`numpy` / `pandas` / `pyarrow` 已在 `requirements.txt`)。

### 4. 文档

更新 README 3.2 节:

- 说明需下载 decoded 分片(`LP-100-decoded-*.parquet`)。
- 新参数 `--decoded-parquet-root` 用法。
- 落盘目录 `<processed_root>/stage2_qbyt/audio/`。
- trainer 现在读 `.npy`(不再读散装 wav)。

## 验收标准

1. `prepare_stage2_libriphrase.py` 能从聚合 parquet + decoded 分片生成
   `train.jsonl` 和对应 `.npy` 文件,无需任何散装 wav。
2. `train_stage2_qbyt.py` 能用生成的 JSONL + `.npy` 跑通若干步训练(`--limit-steps`)。
3. 未匹配的 clip 有告警计数,不静默吞掉。
4. 新增/现有单测通过,无新依赖。
5. README 反映新流程。

## 不做(YAGNI)

- 不支持 `.wav` 落盘(已选 `.npy`)。
- 不做 train 时 parquet 内存索引或 JSONL 内联数组。
- 不做无关重构。
