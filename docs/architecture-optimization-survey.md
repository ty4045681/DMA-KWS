# DMA-KWS Stage II 架构继续优化：调研与路线图

> 生成日期：2026-10-03（本机数据快照）；文献部分由 5 路并行调研汇总。
> 数据来源：`docs/wiki/Experiment-Log.md`、`docs/wiki/False-Alarm-Evaluation.md`、
> `outputs/qbyt_review_2026-10-03/`、`outputs/qbyt_v41_audit_2026-10-03/`、
> DMA-KWS / DS-KWS / PLCL / ParallelKWS / KFC-KWS / CORD-KWS 等论文（见 §6）。
> 只记录**可复核**的数字；标“估算/摘要级”的是外推或未读全文，不是测量值。

---

## 0. TL;DR

1. **现有 15 个 50k run 的方差分解：encoder 因素解释 94.0% 的 AUC 方差，readout 只有 4.6%，
   残差 1.4%。** 继续微调 readout（v2/v3/v4/v4.1/v7）是投入产出比最低的方向。

| 因子 | 主效应均值极差 | 方差占比 |
|---|---|---|
| encoder（zh-en-3M / GS-base / GS-finetune，streaming 16/64） | **0.0677 AUC** | **94.0%** |
| readout（v2 / v3 / v4 / v4.1） | 0.0160 AUC | 4.6% |

2. **当前最好 frozen-encoder 头：AUC 0.9376 / EER 12.86%**（v2 × zh-en-3M，50k 步，155,619 anchors）。
   同任务的 frozen/可训练设置：DS-KWS M1 **0.9577/10.02**、M2 **0.9785/6.13**、
   KFC-KWS（frozen）**0.9765/7.75**、作者联合训练 v1 release 在本机评测 **0.9595/9.86**。
   差距的大头在 encoder 表征 + 训练目标/负样本，不在 readout 结构。

3. **作者发布的 1460h Stage-1 encoder 本机已在使用且未带来优势**：
   `raw/kws-checkpoints/paper-stage1/ls-gs-1460.pt` 与 `exp/author_v1/stage1_v1.pt` 的
   encoder 张量哈希完全一致（sha `94e3d65e3b411be42b45`，249 个张量）。
   在这同一个 encoder 上：
   * 现有 progress 目标（paperstage1 run）@34,472 步：**0.9254 / 13.91**；
   * 论文 v1 目标 + 背景负样本、仅 5,000 步（`v1-ft-wenet-bg`）：**0.9472 / 11.34**；
   * 作者联合训练的 v1 head：**0.9595 / 9.86**。
   → **换 encoder 不是答案；目标函数/训练方式是。**

4. **最便宜、证据最强的一步：目标换成论文 v1 配方（membership + token 归一化 + 含末位监督），
   并把 hard negative 从 1:1 提到论文的 100:1。** 本机 `v1-ft-wenet-bg` 只用 2,684 anchors / 5k 步。

5. **第二步让 encoder 参与训练（PEFT 优先）**：SUPERB KS 上 **BitFit（bias-only，0.10M 参数）
   97.33 > Houlsby adapter 97.17 > Prefix 97.05 > 全量 FT 95.87 > frozen 95.32**；
   DS-KWS M1→M2 消融 = **+2.08 AUC / −3.89 EER**。

6. **第三步才是 head 结构**：CTC keyframe + cross-attention（KFC-KWS）、
   parallel cross-attention + PDA 时长对齐（ParallelKWS）、phoneme-level contrastive + memory bank（PLCL）、
   query-conditioned attentive pooling / MaxSim、prefix-overlap 校正（POB/EPS）。
   v7 bounded segmental 方向正确但 3.4 s/step 成本要先解决。

7. **低 FPR 尾部是部署指标且文献没有基线**：TPR@0.1%FPR 只有 2.3%、pAUC(≤1%) 0.552；
   调研的 ~20 篇同任务论文**没有任何一篇报告 TPR@1%FPR / TPR@0.1%FPR / pAUC / Brier / ECE**。
   建议 pAUC/AUC-margin 代理损失 + 近音词分组 c-AUC，并把尾部指标当成可发表的贡献点。

8. **校准、阈值与分数融合几乎免费**：仓库已有 `PositiveAffineCalibrator`；
   可再做 AS-norm 队列归一化、LTT/保形阈值选择、EVT 尾部外推、Stage I/II 分数 Platt 融合。

9. **误唤醒对上下文长度极敏感**：官方 3 s/3 s 网格全部 0；1 s/1 s 网格
   v4.1 29.1 < v3 35.7 < v4 45.0 FA/24h，且几乎全来自 speech。需要多尺度聚合 + 长度增强 + speech 负样本消融。

10. **单项收益最大的三类“非架构”改动**（都已有公开消融）：
    * **FA-aware precision 损失**：MALEFA 去掉后 hard EER 13.91→21.10、AMI FAR 0.007%→14.5%；
    * **长度约束 + 子序列监督**：SLiCK 使 LP-hard AUC 88.52→94.90、EER 18.82→11.10；
    * **负样本构造**：AdaKWS 的“同句其他词 / 字符替换 / batch-nearest”在 frozen Whisper 上达 LP-hard 95.09/11.48。

---

## 1. 现有测试数据盘点

### 1.1 评测协议与噪声底

| 项目 | 值 |
|---|---|
| 主评测 | LibriPhrase-460 **hard split**：270,684 个 (text, audio) 对，每次验证全量（2,115 batch） |
| 主指标 | AUC / EER / TPR@1%FPR / TPR@0.1%FPR / pAUC(≤1%) / Brier / ECE |
| 噪声底（相邻验证点） | **±0.002 AUC / ±0.005 EER**；TPR@0.1%FPR 更吵（0.005–0.008） |
| 验证成本 | 全量 hard split ≈ 2 min 10 s（8 workers） |
| 方法学限制（审计原文） | hard split 被反复用于 checkpoint 选择；每配方单 seed；v2 的 fp16 sigmoid 日志可能损失排序分辨率 |

### 1.2 15 个 50k run 的结果矩阵（AUC / EER）

全部 frozen encoder + 422K（v4.1 439K）trainable head；训练集 LS-GS-1460（155,619 anchors / 5.84M clips）；
random + G2P hard negative **1:1**；MUSAN 背景 crop p=0.25；batch 128；50k 步。

| encoder（frozen） | 操作点 | v2 | v3 | v4 | v4.1 |
|---|---|---|---|---|---|
| zh-en-3M | streaming 16/64 | **0.9376 / 12.86** | 0.9316 / 13.22 | 0.9323 / 13.36 | 0.9350 / 13.11 |
| GS-KWS base | streaming 16/64 | 0.9271 / 14.11 | 0.9064 / 16.30 | 0.9131 / 15.70 | 0.9119 / 15.95 |
| GS-KWS finetune | streaming 16/64 | 0.8809 / 19.58 | 0.8598 / 21.67 | 0.8565 / 21.96 | 0.8687 / 20.77 |
| GS-KWS base | full context −1/−1 | （未跑） | 0.9136 / 15.72 | 0.9195 / 14.79 | 0.9129 / 15.63 |

* **v2（GRU-last pooling）在 3 个 encoder 上都赢**，TPR@1%FPR 最高（zh-en 0.2021）。
* v2 校准最差（Brier 0.136–0.196）；v3/v4/v4.1 更好（0.106–0.160），但尾部更差。
* v4.1 比 v4 慢 2.2–2.35×（0.44–0.60 s/step vs 0.16–0.19 s/step），只在 zh-en / GS-finetune 上小幅领先 v4。
* full context（−1/−1）对 GS-base 的 v3/v4/v4.1 有 +0.0064 AUC（v4），但 v2 只有 streaming 仍全面更好。
* 所有曲线 ~35–40k 步后趋平，最后 10k 步最多 +0.007 AUC。

### 1.3 与论文 / 作者权重的对照（同一 hard split；可比性见 §3.0）

| 系统 | encoder 是否训练 | 参数量 | AUC | EER | 备注 |
|---|---|---|---|---|---|
| 本仓库最好 frozen（v2 × zh-en-3M） | 否 | 422K head | 0.9376 | 12.86 | 155,619 anchors，50k 步 |
| 论文 DS-KWS **M1**（frozen） | 否 | ~0.5M head | **0.9577** | **10.02** | 音素 CTC Conformer（LS-GS-1460），155k anchors |
| 论文 DS-KWS **M2**（trainable encoder） | 是 | ~4.1M | **0.9785** | **6.13** | 用时间戳重切原始音频 |
| KFC-KWS（Interspeech 2026，frozen XLS-R） | 否 | — | **0.9765** | **7.75** | CTC keyframe + cross-attention；摘要级 |
| 作者发布 v1 SI（`155k-v2-ft.ckpt`） | 联合训练 | — | **0.9595** | **9.86** | TPR@1%FPR 0.2877；MUSAN FA 2266.8 次/24h |
| `v1-ft-wenet-bg`（本机，5k 步） | 否（冻作者 1460h encoder） | — | 0.9472 | 11.34 | 2,684 anchors；MUSAN FA **0** |
| `v1-ft-icefall-bg`（本机，5k 步） | 否（冻 Zipformer BPE encoder） | — | 0.5698 | 45.81 | 同一 v1 目标，encoder 换掉就崩 |

DS-KWS anchor 规模消融（frozen encoder）：12k → 20k → 40k → 78k → 155k anchors，
LP-hard AUC **93.22 → 93.95 → 94.75 → 95.33 → 95.45**，EER 13.38 → 10.65。
**我们用 155k anchors 却停在它 12k 档的水平**——phoneme matcher 没有把 anchor 多样性转成判别力。

> **训练中快照（2026-10-03 19:5x，作者 1460h encoder × frozen + progress loss）**
> v2 @34,472 步：0.9254 / 13.91；v3 @44,019 步：0.9220 / 14.51。
> 同配方 zh-en-3M：30k 步 0.9303、40k 步 0.9352。
> 即：**换成作者 1460h 音素 encoder、保留 progress 目标，并没有变好**（呼应 §0-3）。

### 1.4 误差结构

| 观察 | 数字 |
|---|---|
| 低 FPR 尾部弱 | 最好 TPR@1%FPR = 0.2021；TPR@0.1%FPR = 0.0235 |
| pAUC(≤1%) 天花板低 | 0.5524（最好） |
| 校准差 | Brier 0.1059–0.1963；ECE 0.0787–0.2266 |
| 部署阈值 0.5 处不可用 | v2 zh-en 在 0.5 阈值 deploy_fpr≈0.21–0.27 |
| FA 与窗口长度强相关 | 3 s 网格：MUSAN 全 0、LS-other 仅 2 个模型各 1 次（0.29/24h）；1 s 网格：v4.1 29.1 < v3 35.7 < v4 45.0 FA/24h |
| FA 集中在 speech | 1 s 网格 v4.1 的 53 次 accept 中 36 次来自 speech（24.2h），17 次 music（17.0h），0 次 noise（2.5h） |

### 1.5 成本实测

| readout | steady state | s/step | 50k 步墙钟 | GPU util | VRAM |
|---|---|---|---|---|---|
| v2 `gru_last` | ~5.3 steps/s（共置） | ~0.19 | 2.8–3.2 h | ~100%（共置） | ~2 GB |
| v3 `eps_mean` | ~5.5 steps/s | ~0.18 | 3.4–3.9 h | ~99% | ~2 GB |
| v4 `eps_softmin` | ~6.2 steps/s | ~0.16 | 3.6–4.2 h | ~99% | ~2 GB |
| v4.1 | 2.28 steps/s | 0.44（独跑） | 8.2–9.8 h | 88% | 2.1 GB |
| v7 bounded segmental | 0.26–0.29 steps/s | 3.4–3.9 | 未跑完（215/50k 中止） | 20–25% | 15–25 GB |

数据加载吞吐在 276–300 samples/s 饱和（V100），batch >128 只拉长 step。

---

## 2. 诊断：瓶颈假设、证据与优先级

### H1（最高）encoder 表征与训练目标的组合问题，而非单纯“encoder 不够强”

* README §7d：Zipformer KWS encoder 是 transducer/BPE 目标；QbyT 文本分支是 71 音素 `nn.Embedding`，
  两者之间只有一层 `Linear`，仅靠 utterance/seq BCE 监督。
* 同一 v1 目标下，encoder 从 Wenet phoneme（0.9472）换到 Icefall Zipformer BPE（0.5698）→ 空间不匹配会崩。
* 但**把 encoder 换成作者 1460h 音素 encoder、保留 progress 目标仍不涨**（0.9254 @34k）。
* KFC-KWS 证明 frozen encoder（XLS-R 0.3B）+ 好的对齐 head 可到 0.9765；
  → 特征是“encoder 强度 × 目标匹配 × head 对齐”三者的乘积，不能只换一个。

### H2（次高）目标函数与负样本配比

* 论文 v1 目标 = utterance BCE + **membership 序列 BCE（每位置含末位、token 归一化）**；
  当前 v2/v3/v4 默认 = ordered-prefix progress(0.3) + completion(0.5) + sample 归一化。
* 论文第二阶段 hard-neg **100:1**；当前 frozen run **1:1**。
* 本机对照（同一 1460h encoder）：progress 50k run ≈0.925@34k vs v1 目标 0.9472@5k。
* 负样本从一开始就是 hard（1:1），所有 run ~40k 就平——符合“hard negative 早期饱和”的特征（CurricularFace/CHNS 文献）。

### H3（高）frozen encoder 封顶

DS-KWS M1→M2 = +2.08 AUC / −3.89 EER；SUPERB KS 上 PEFT（BitFit/Adapter/Prefix）反超全量微调。
本机所有 15 run + paperstage1 run 都是 frozen，且没有用 PEFT。

### H4（中）head 交互方式过于简单，但优先级靠后

v2/v3/v4 = `cat([text,audio]) → 2 层 self-attention → GRU-last/EPS pooling`；
无显式 cross-attention、无单调对齐、无 query-conditioned pooling、无 keyframe 选择。
readout ladder 只差 ≤0.006 AUC → 先修 H1–H3。

### H5（中）低 FPR 尾部与校准

`negative_tail_cvar_loss`（top 10% 负样本）开启后 v3/v4 的 TPR@1%FPR 反而低于 v2
（0.157/0.170 vs 0.202），只改善 Brier/ECE。CVaR 没有对准“固定 FPR 下的召回”。

### H6（中）训练/部署 segment 分布不一致 + 背景类别

训练背景为 1–3 s MUSAN crop（music+noise，speech 被排除）；评测用 3 s 与 1 s 窗口；
1 s 网格误唤醒几乎全来自 speech。可检验，成本低。

---

## 3. 论文调研

> 每条都带 URL；标注“摘要级”表示只看过摘要，未读实验表。未标注的为可复核事实。

### 3.0 ⚠️ 可比性警告

* **LibriPhrase “Hard” 在不同论文中不是同一份 trial list**：同批论文出现 73.9–84.2、84.21、
  91.29、92.7、95.56、96.59、97.65、97.85 等 AUC；enrollment 模式（text-only / audio+text /
  speaker-dependent）与负样本构造也可能不同。跨论文只做方向性参考。
* **没有一篇同任务论文报告 TPR@1%FPR / TPR@0.1%FPR / pAUC / Brier / ECE**。
  我们的低 FPR 弱点是真实短板，但**没有文献基线**，也意味着这是一个可以自己定义并发表的评测维度。
* DMA-KWS 与 DS-KWS 都报 97.85/6.13；DS-KWS 明确是 text enrollment + M2（可训练 encoder）。
* 本仓库 270,684 对 hard split 与论文是否逐对一致**未核实**。

### 3.1 同任务系统（user-defined / open-vocabulary KWS）

| 论文 | 核心机制 | 关键数字 | 启示 |
|---|---|---|---|
| **DMA-KWS** TASLP 在审 [arXiv:2605.22120](https://arxiv.org/abs/2605.22120)；[官方 repo](https://github.com/aizhiqi-work/DMA-KWS) | Stage I 流式音素搜索 + Stage II QbyT phoneme matcher（phoneme+utterance 两级 BCE）；多模态 enrollment（MAM）；LoRA 只调 matcher attention QKV（187K 参数） | LP-H 97.85 AUC / 6.13 EER | 我们复现的对象；发布 1460h Stage-1 与 155k Stage-2 权重 |
| **DS-KWS**（5 页版）[arXiv:2510.10740](https://arxiv.org/abs/2510.10740) | 双数据 scaling：ASR 460→1460h；anchor 12k→155k；M1 frozen / M2 可训练重编码 | M1 95.77/10.02；M2 **97.85/6.13**；anchor 消融 12k→78k = 93.22→95.33 | M1→M2 消融 + anchor 异常的解释 |
| **KFC-KWS** Interspeech 2026 [arXiv:2606.10365](https://arxiv.org/abs/2606.10365) | CTC 峰值后验选 keyframe（5 帧窗）→ cross-attention 融合全句；audio/phoneme/text 三模态 | balanced 98.73；LP-H **97.65/7.75**（frozen XLS-R，摘要级） | frozen encoder + 对齐 head 的存在性证明 |
| **CORD-KWS** [arXiv:2609.31869](https://arxiv.org/abs/2609.31869) | 保持 cosine 打分与 O(1) enrollment，加“绝对分数监督 + 帧级 CTC 顺序目标”的 calibrated head | EER 0.43%（easy）/ **9.64%（hard）** | 直指我们的 ECE/Brier 与低 FPR 尾部；不改 encoder |
| **PLCL** ICASSP 2025 [arXiv:2412.20805](https://arxiv.org/abs/2412.20805) | phoneme-level contrastive；phoneme memory bank（insert/replace/delete 合成近音负样本）；第三类判别器 | LP-H 95.56（text）/ **96.59**（audio+text） | 针对 confusable words 的最系统方案 |
| **ParallelKWS** [arXiv:2408.03593](https://arxiv.org/abs/2408.03593) | parallel cross-attention + self-attention；三路 max-pool；PDA 时长对齐损失 | LP-H 91.29/14.80 → +PDA 91.68/14.36 | 最小改动 head 升级 |
| **U2-KWS** ASRU 2023 [arXiv:2312.09760](https://arxiv.org/abs/2312.09760) | CTC 第一遍 + decoder 第二遍统一训练；两分支都注入 keyword cross-attention | 固定 0.5 FA/h 相对 **+41%** 唤醒率 | Stage I 也可 keyword-conditioned；两级分数融合 |
| **MM-KWS** Interspeech 2024 [arXiv:2406.07310](https://arxiv.org/abs/2406.07310) | 多模态 prompt 注册 | LP-H AT 96.25/9.30 | SD 场景上限参考 |
| **PhonMatchNet** Interspeech 2023 [arXiv:2308.16511](https://arxiv.org/abs/2308.16511) | 双流 encoder + self-attention pattern extractor + **音素级 detection loss** | 平均相对 EER −67% / AUC +80% | 音素级监督直接打 confusable words |
| **Homogeneous Audio-Text Embedding** [arXiv:2308.06472](https://arxiv.org/abs/2308.06472) | G2P→音素→embedding，音素向量取自**配对 audio encoder**；合成 confusable keyword | LP-H AUC 84.21→92.7，EER 23.36→14.4 | 我们的 93.76/12.86 已超过它；说明差距不在 head 基础结构 |
| **No Word Left Behind / POB+EPS** ICASSP 2026 [arXiv:2602.08930](https://arxiv.org/abs/2602.08930) | OV-KWS 偏重关键词前几个音素 → 共享前缀误触发；Partial Overlap Benchmark + Equal-weighting Position Scoring | EER 64.4→29.3（POB-Spark），acc 87.6→96.8（POB-LP） | 负样本需要**前缀重叠**专项；GRU-last 也天然偏尾部/前缀 |
| **MALEFA** ICASSP 2026 [arXiv:2604.03689](https://arxiv.org/abs/2604.03689) | utterance+phoneme 双级 cross-attention 对比 + **FA-aware precision-constrained loss** | LP-hard 93.58/13.91（0.7M）；AMI FAR 0.007% vs PhonMatchNet 17.879%；**去掉 FA loss → AMI FAR 14.542%、LP-hard EER 21.10** | 目前找到的**最大单消融**：一个只管误接受的损失项 |
| **MFA-KWS** [arXiv:2505.19577](https://arxiv.org/abs/2505.19577) | 流式 CTC+transducer 多 head 帧异步解码；keyword-specific phone-synchronous CTC；分数融合 | Snips SOTA | 作者 Stage-I 血统；替代我们的 ContextGraph |
| **Synth4Kws** Interspeech 2024 [arXiv:2407.16840](https://arxiv.org/abs/2407.16840) | TTS 生成 custom KWS 数据；增大 phrase 多样性与采样量 | EER −30.1% / AUC +46.7%（相对 50k 真实语音基线） | TTS 数据是被验证的独立增益轴 |
| **LLM-Synth4KWS** [arXiv:2505.22995](https://arxiv.org/abs/2505.22995) | LLM 生成近音词分组 + TTS 多风格合成；定义 c-AUC | Speech Commands AUC +3.7%，**c-AUC +11.3%** | 数据侧最便宜的低 FPR 手段 + 新指标 |
| **GE2E-KWS** [arXiv:2410.16647](https://arxiv.org/abs/2410.16647) | 419KB 量化 conformer vs 7.5GB ASR encoder | 相对 AUC +23.6%（摘要级） | 小模型也能赢大 ASR 特征 |
| **Massive OV-KWS** [arXiv:2606.11279](https://arxiv.org/abs/2606.11279) | 大规模开放词表 | 内存 ×1/128（摘要级） | 规模化词表的工程约束 |
| **KWS-Whisper** Interspeech 2024 [arXiv:2309.09552](https://arxiv.org/abs/2309.09552) | Whisper encoder 隐状态上做 OV-KWS + 上下文 ASR 多任务 | 热词 recall 大幅提升 | frozen Whisper-family encoder 可被关键词探测 |
| **Learning Audio-Text Agreement** Interspeech 2022 [arXiv:2206.15400](https://arxiv.org/abs/2206.15400) | cross-modal attention matching + monotonic matching + keyword classification + denoising | 提出 **LibriPhrase** | hard split 的来源，文本注册范式 |
| **NN end-to-end QbE-STD** [arXiv:1911.08332](https://arxiv.org/abs/1911.08332)；经典 posteriorgram DTW [ISCA 2011](https://www.isca-archive.org/interspeech_2011/muscariello11_interspeech.html) | CNN matching 联合训练 vs 模板 DTW | — | 后验图质量比匹配器更重要 |
| 端侧/流式补充：CTC-aligned audio-text [arXiv:2406.07923](https://arxiv.org/abs/2406.07923)（155K 参数、O(U)）；W-CTC [ISCA 2025](https://www.isca-archive.org/interspeech_2025/kim25d_interspeech.html)；Lattice-Free OV-KWS [IEEE 10485678](https://ieeexplore.ieee.org/document/10485678)；Duration-Aware Phone Upsampling [IEEE 10983304](https://ieeexplore.ieee.org/document/10983304)；Efficient Audio-Text KWS [ACL ICON 2024](https://aclanthology.org/anthology-files/anthology-files/pdf/icon-2024/2024.icon2024-1.31.pdf) | — | — | 流式可行性与端侧效率 |

### 3.2 融合与对齐结构

**显式跨模态注意力**
* **ParallelKWS**：parallel self/cross attention + 三路 max-pool；PDA 用 CTC 连续同音素帧数构造时长矩阵。
* **Multi-task Cross Attention KWS** [arXiv:2107.07634](https://arxiv.org/abs/2107.07634)：可训练 query 序列 cross-attend 音素 encoder。
* **Hyper-Matched Filters** [arXiv:2508.04857](https://arxiv.org/abs/2508.04857)：hyper-network 把关键词变成 keyword-specific 卷积核（matched filter），
  引导 Perceiver cross-attention；**4.2M 参数**即匹配更大模型，泛化到 L2 语音。与我们的 422K head 同一量级。

**late interaction / MaxSim**
* **ColBERT** [arXiv:2004.12832](https://arxiv.org/abs/2004.12832) / **ColBERTv2** [arXiv:2112.01488](https://arxiv.org/abs/2112.01488)：query 每 token 对 doc 取 max 相似再求和。
  71 音素 token × 冻结帧序列天然适配；**未发现用于音素-CTC KWS 的先例（开放实验）**。
* **CLaMR** [arXiv:2506.06144](https://arxiv.org/abs/2506.06144)（含语音转写模态）；
  **FLASH-MAXSIM** [arXiv:2605.29517](https://arxiv.org/abs/2605.29517)（不 materialize 分数矩阵，峰值显存 −1.4~2.6×）。
* **MATE** [arXiv:2601.14012](https://arxiv.org/abs/2601.14012)：Matryoshka audio-text embedding + PCA 前缀对齐（摘要级）。

**单调 / segmental / OT 对齐**
* **MoChA** [arXiv:1712.05382](https://arxiv.org/abs/1712.05382) + **CTC-synchronous training** [arXiv:2005.04712](https://arxiv.org/abs/2005.04712)；
  **CIF** [arXiv:1905.11235](https://arxiv.org/abs/1905.11235)（可微音素-帧边界）；
  **m-LTM** [arXiv:2405.10084](https://arxiv.org/abs/2405.10084)、**OTReg** [arXiv:2508.08131](https://arxiv.org/abs/2508.08131)（对齐即正则项）。
* 本仓库 **v7 bounded segmental aligner**（`qbyt/monotonic_alignment.py`）：
  精确 keyword-vs-filler 图 partition LLR + duration/gap/span 约束 + one-edit 近音图；方向最结构化、成本最高。

**pooling / query 条件化**
* **KWT** [arXiv:2104.00769](https://arxiv.org/abs/2104.00769)；**ST-AttNet** [arXiv:2108.12146](https://arxiv.org/abs/2108.12146)；
  **ECAPA-TDNN** [arXiv:2005.07143](https://arxiv.org/abs/2005.07143)（query-conditioned attentive statistics pooling，mean+std）。
* **AdaKWS** [arXiv:2309.08561](https://arxiv.org/abs/2309.08561)、**FiLM on frozen encoder** [arXiv:2606.06211](https://arxiv.org/abs/2606.06211)（<50K 参数的文本条件化）。
* **Perceiver** [arXiv:2103.03206](https://arxiv.org/abs/2103.03206)、**USM+Perceiver** [arXiv:2310.13010](https://arxiv.org/abs/2310.13010)。

**对比音频-文本空间**
* **CLAP** [arXiv:2206.04769](https://arxiv.org/abs/2206.04769)；
  **“Discriminative Axis”** [arXiv:2608.01560](https://arxiv.org/abs/2608.01560)（摘要级）：对比 embedding 只编码“同 batch 负样本迫使它编码的轴”。
  **负样本结构（而不是语料规模）决定表征学什么。**

### 3.3 Encoder、表征与迁移

**SSL / 预训练 encoder**
* **WavLM** [arXiv:2110.13900](https://arxiv.org/abs/2110.13900)；**HuBERT** [arXiv:2106.07447](https://arxiv.org/abs/2106.07447)；
  **wav2vec 2.0** [arXiv:2006.11477](https://arxiv.org/abs/2006.11477)；**XLS-R** [arXiv:2111.09296](https://arxiv.org/abs/2111.09296)（KFC-KWS 用 0.3B 档）；
  **w2v-BERT 2.0 / SeamlessM4T** [arXiv:2308.11596](https://arxiv.org/abs/2308.11596)；**Whisper** [arXiv:2212.04356](https://arxiv.org/abs/2212.04356)；
  **SUPERB** [arXiv:2105.01051](https://arxiv.org/abs/2105.01051)。
* 无 LibriPhrase KWS 数字的：WavLM / w2v-BERT 2.0 / Whisper encoder。

**encoder 架构**
* **Zipformer** [arXiv:2310.11230](https://arxiv.org/abs/2310.11230)（U-Net 多速率 stack；我们 Stage I/II 的 encoder）；
  **Conformer** [arXiv:2005.08100](https://arxiv.org/abs/2005.08100)、**Branchformer** [arXiv:2207.02971](https://arxiv.org/abs/2207.02971)、
  **E-Branchformer** [arXiv:2210.00077](https://arxiv.org/abs/2210.00077)（LibriSpeech 1.81/3.65 WER）。
  结论：encoder 架构不是当前瓶颈（head Δ≤0.006），但 parallel merge 值得抄到 head。

**PEFT / 适配（H3 最直接的解法）**
* **Efficient adapter transfer** [arXiv:2202.03218](https://arxiv.org/abs/2202.03218)：wav2vec2 adapter <10% 参数；
  **只加在顶部若干层即可匹配全量迁移**。
* **Exploring efficient-tuning in SSL speech** SLT 2022 [arXiv:2210.06175](https://arxiv.org/abs/2210.06175)：
  HuBERT-base SUPERB **KS**：frozen 95.32 / weighted-sum 96.30 / full FT 95.87 /
  Houlsby 97.17 / AdapterBias 97.30 / **BitFit 97.33（0.10M）** / Prefix 97.05 / LoRA 96.59。
  → **PEFT 超过全量微调；bias-only / 顶部 adapter 就够。**
* **SURE** [arXiv:2303.03267](https://arxiv.org/abs/2303.03267)（ConvAdapter，0.94% 参数）；
  **LoRA** [arXiv:2106.09685](https://arxiv.org/abs/2106.09685)、**BitFit** [arXiv:2106.10199](https://arxiv.org/abs/2106.10199)、
  **Prefix-tuning** [arXiv:2101.00190](https://arxiv.org/abs/2101.00190)。

**蒸馏**
* **DistilHuBERT** [arXiv:2110.01900](https://arxiv.org/abs/2110.01900)、**FitHuBERT** [arXiv:2207.00555](https://arxiv.org/abs/2207.00555)、
  端侧 KD-S3RL KWS [arXiv:2307.02720](https://arxiv.org/abs/2307.02720)；双视角互相关蒸馏 [ISCA 2023](https://www.isca-archive.org/interspeech_2023/yang23y_interspeech.html)。
  **只有 teacher 足够强时蒸馏才有效。**

**SSL 用于 KWS**
* 轻量 transformer on S3RL [arXiv:2303.04255](https://arxiv.org/abs/2303.04255)：330K 参数，固定 FRR 下误接受相对 −6%~−23.7%。
* Few-shot KWS + SSL [arXiv:2506.17686](https://arxiv.org/abs/2506.17686)：wav2vec2 + Sub-center ArcFace teacher + ResNet15；
  10-shot 1% FA 下 acc 33.4%→**74.1%**。
* layer 选择：Pasad ASRU 2021 [arXiv:2107.04734](https://arxiv.org/abs/2107.04734)、ICASSP 2023 [arXiv:2211.03929](https://arxiv.org/abs/2211.03929)；
  儿童 zero-shot KWS [arXiv:2508.21248](https://arxiv.org/abs/2508.21248)（第 22 层最好）。

### 3.4 目标函数、难负样本与校准

**pAUC / 排序代理**
* When AUC meets DRO (ICML 2022) [arXiv:2203.00176](https://arxiv.org/abs/2203.00176)：pAUC 即 DRO（CVaR 精确、KL 平滑）。
* Large-scale optimization of range-pAUC [arXiv:2203.01505](https://arxiv.org/abs/2203.01505)：nonsmooth DC + Moreau 平滑。
* instance-wise 无偏估计 [arXiv:2210.03967](https://arxiv.org/abs/2210.03967)；
  **AUC-margin**（O(1) 内存）[arXiv:2012.03173](https://arxiv.org/abs/2012.03173)。
* KWS 专用：AUC optimization for robust KWS [ar5iv 2107.05859](https://ar5iv.labs.arxiv.org/html/2107.05859)；
  multi-class AUC [Interspeech 2022](https://www.isca-archive.org/interspeech_2022/xu22h_interspeech.html)；
  **Optimize what matters** [arXiv:2011.01151](https://arxiv.org/abs/2011.01151)：直接按检测指标训练，固定 FA 下相对 FRR −70%+。

**margin / metric learning**
* **ArcFace** [arXiv:1801.07698](https://arxiv.org/abs/1801.07698)、**CosFace** [arXiv:1801.09414](https://arxiv.org/abs/1801.09414)、
  **AM-Softmax** [arXiv:1801.05599](https://arxiv.org/abs/1801.05599)、**Sub-center ArcFace** [ECCV 2020](https://www.ecva.net/papers/eccv_2020/papers_ECCV/html/1445_ECCV_2020_paper.php)。
* **CurricularFace** [arXiv:2004.00288](https://arxiv.org/abs/2004.00288)：易样本先、难样本后 —— 对应我们“从一开始就 hard、40k 平”的现象。
* **Multi-Similarity** [arXiv:1904.06627](https://arxiv.org/abs/1904.06627)、**In Defense of the Triplet Loss** [arXiv:1703.07737](https://arxiv.org/abs/1703.07737)。

**对比 / 不平衡目标**
* **InfoNCE / CPC** [arXiv:1807.03748](https://arxiv.org/abs/1807.03748)：128 batch = 每 query 127 个免费负样本。
* **PLCL**（见 §3.1）；**Focal Loss** [arXiv:1708.02002](https://arxiv.org/abs/1708.02002)；
  **Asymmetric Loss** [arXiv:2009.14119](https://arxiv.org/abs/2009.14119)（负样本比例提高后用于抑制易负样本/噪声标签）。

**难负样本挖掘**
* **Regional Hard-Example mining for KWS** ICASSP 2020 [链接](https://pure.nwpu.edu.cn/zh/publications/mining-effective-negative-training-samples-for-keyword-spotting/)：
  逐帧选非连续 hard 帧、控制 neg:pos 比；**1 FA/h 下相对 FRR −45%~−58%**。最贴近我们的 KWS 误报方案。
* **Adversarial Retriever-Ranker** [arXiv:2110.03611](https://arxiv.org/abs/2110.03611)：Stage I 可对抗式挖 Stage II 的难候选。
* 对抗 confusable 合成 [arXiv:2201.00167](https://arxiv.org/abs/2201.00167)；解耦对抗训练 [arXiv:2408.13355](https://arxiv.org/abs/2408.13355)（1% FAR 下相对 FRR −40.31%）。
* 比例调度：Clustering-based Hard Negative Sampling [Interspeech 2025](https://www.isca-archive.org/interspeech_2025/masztalski25_interspeech.html)（相对 EER/minDCF −18%）；
  **When Hard Negatives Hurt** [arXiv:2606.01304](https://arxiv.org/abs/2606.01304)（生成式 hard negative 可能因分布差/假阳性污染而伤检索）。

**校准、分数标准化与阈值**
* **On Calibration of Modern NNs** [arXiv:1706.04599](https://arxiv.org/abs/1706.04599)：单温度即可修好大部分误校准。
* **Speaker verification 校准对比** [arXiv:2203.15106](https://arxiv.org/abs/2203.15106)（logistic / MagNetO / SONet；Cllr）；
  Brummer & du Preez CSL 2006 [链接](https://www.sciencedirect.com/science/article/pii/S0885230805000483)（Cllr 定义）。
* **AS-norm** [Interspeech 2017](https://www.isca-archive.org/interspeech_2017/matejka17_interspeech.html)：队列归一化，相对 +30%；训练无关。
* **Learn then Test** [arXiv:2110.01052](https://arxiv.org/abs/2110.01052)、**Conformal Risk Control** [arXiv:2208.02814](https://arxiv.org/abs/2208.02814)：
  在 holdout 上选阈值并给有限样本 FPR 保证。
* **EVT 尾部外推** [arXiv:1808.09902](https://arxiv.org/abs/1808.09902) + GPD 生物特征精度估计：
  我们的 TPR@0.1%FPR=2.3% 只基于约 271 个负样本，可用 GPD 给出带置信区间的估计。
* **Quantization-Based Score Calibration** [arXiv:2510.15432](https://arxiv.org/abs/2510.15432)：用量化误差归一化，使单一阈值跨噪声条件有效。

### 3.5 数据、负样本、增广与鲁棒性

**最有决策价值的四条证据**
* **FA-aware 损失**：MALEFA [arXiv:2604.03689](https://arxiv.org/abs/2604.03689) 去掉 precision 约束项后
  AMI FAR 从 0.007% 恶化到 14.542%、LP-hard EER 从 13.91 恶化到 **21.10**（+7.2）。
  现有 `negative_tail_cvar_loss` 是“抬高难负样本损失”，不是“压低接受数”的 precision 目标，这解释了
  为什么开启后 TPR@1%FPR 反而下降（§2-H5）。
* **长度约束 + 子序列监督**：SLiCK [arXiv:2409.09067](https://arxiv.org/abs/2409.09067) 不做变长文本聚合、
  改为子序列匹配，LP-hard AUC 88.52→**94.90**、EER 18.82→**11.10**。我们的 v1 读出按构造依赖 padding，
  v2 pooling 仍看整段拼接，尾部虚假高分与此一致。
* **负样本构造方式**（AdaKWS [arXiv:2309.08561](https://arxiv.org/abs/2309.08561)，frozen Whisper，LP-hard 95.09/11.48）：
  对比 random / 字符替换（s↔z、p↔b）/ 关键词拼接 / **batch 内最近文本 embedding**；整句 ≤30 s，负样本来自同句其他词。
  GraphemeAug [arXiv:2505.14814](https://arxiv.org/abs/2505.14814) 用字形插入/删除/替换生成边界负样本，
  在合成 hard-negative 集 AUC +61%（真实音频收益未证实）。
* **真实误报来自近音语音，而不是噪声**：Howl [NLP-OSS 2020](https://aclanthology.org/2020.nlposs-1.9/)
  的误报 **90% 来自同域 substring 单词**（"fire"/"fox" 类）；训练负样本只占 Common Voice 的 10%，运行点 4 FA/h。
  与本机 1 s 网格“speech 子集占 36/53”完全一致。

**其余数据/鲁棒性**
* **TTS / LLM 合成**：Synth4Kws [arXiv:2407.16840](https://arxiv.org/abs/2407.16840)（TTS 混合使 EER −30.1% / AUC +46.7%）；
  MM-KWS 用 G2P 编辑距离 + 语义 + LLM 生成 confusable、ZS-TTS 渲染 1.5M EN/2.4M ZH 音频（LP-hard 96.25/9.30）；
  LLM-Synth4KWS [arXiv:2505.22995](https://arxiv.org/abs/2505.22995)（c-AUC +11.3%）。
  **合成数据要和真实数据混合**（ICASSP 2021 报告混合训练使 DET AUC 改善 >11%），
  且朴素合成/真实混合会掉点（ZeSTA [arXiv:2603.04219](https://arxiv.org/abs/2603.04219) 用 domain embedding + 真实过采样修复）。
  本仓库 §8 已有 TTS→real + LibriPhrase 1:1 的 LoRA 适配流程。
* **对抗样本 + 解耦训练** [arXiv:2408.13355](https://arxiv.org/abs/2408.13355)：clean / noise(0–20 dB) / SpecAugment 分源 BN + 对抗样本，
  1% FAR 下 FRR 相对 −40.31%。
* **回放/干扰混合**（Interspeech 2018 [arXiv:1808.00563](https://arxiv.org/abs/1808.00563)）：
  音乐与 TV/电影按 SIR 混合，固定 FA 下相对 FRR −30%~−45%；比单纯 music/noise 更有效。
* **前缀重叠专项负样本 + EPS**（POB）[arXiv:2602.08930](https://arxiv.org/abs/2602.08930)。
* **SpecAugment** [arXiv:1904.08779](https://arxiv.org/abs/1904.08779)、**MUSAN** [arXiv:1510.08484](https://arxiv.org/abs/1510.08484)：
  标准配置；本仓库已有 MUSAN 背景、`dma_kws/inference/audio_aug.py` 与合成噪声工具链。
  **注意**：SpecAugment 在低 FPR 下是否有 KWS 级收益没有找到受控实验。
* **噪声 wildcard 图**：NTC-KWS [arXiv:2412.12614](https://arxiv.org/abs/2412.12614)。
* **决策层抗误报**：Cross-layer Discrimination Consistency [arXiv:2412.12635](https://arxiv.org/abs/2412.12635)
  用层间一致性区分正例与误触发，0.05 FA/h 下相对 miss rate −46.3%、绝对 recall +6.8%。
* **滑窗聚合**：Max-pooling-loss LSTM [arXiv:1705.02411](https://ar5iv.labs.arxiv.org/html/1705.02411)（平滑后取阈值，相对 AUC +67.6%）；
  两级多分辨率集成 [arXiv:2310.11379](https://arxiv.org/abs/2310.11379)（13 个分类器 log-odds 融合，在每个噪声条件下都超过最优单模型）。
* **长音频 / 多尺度决策**：本机 1 s vs 3 s FA 差异（§1.4）；连续 accept 折叠后事件数几乎不变
  （52 events vs 53 windows），说明是短促孤立误触发，适合用持续性/一致性门控。
* **frozen encoder 的天花板（DS-KWS 补充）**：M1（frozen + 0.5M 投影）在 Hey-Snips 零样本上
  **不如 Stage-I CTC 单独**；M2（3.6M 可训练 re-encoder）才拿到 headline。460h 时 M1 95.33/10.78 vs M2 97.03/7.97。
  → 冻结表征是主约束项，负样本/增广只能在剩余空间里改善。

---

## 4. 建议路线（按 ROI 排序）

> 判据统一：与当前最好 frozen 头 0.9376 / 12.86 在同一 hard split 比较；
> 单点差异小于噪声底（±0.002 AUC / ±0.005 EER）不算赢；每个改动配证伪实验。

### P0 —— 目标换成论文 v1 配方（1–2 个 run，1 天）★最高优先级

**机制**：membership 目标对每个 anchor 位置监督成员关系，梯度更密；token 归一化避免长 anchor 主导；
hard-neg 100:1 把训练分布推向近音词。

**证据（全部本机、同一 1460h encoder）**：
progress 目标 50k-run @34k = 0.9254；v1 目标 5k = 0.9472；作者联合训练 = 0.9595；
DS-KWS frozen M1 = 0.9577。

**实现**（现有代码即可）：

```bash
# E1: v2 readout + v1 目标 + 100:1，在已冻结的作者 1460h encoder 上
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/train_stage2_qbyt.py \
  +experiment=icefall_zipformer_stage2_v2_musan_cached_paperstage1_50k \
  stage2.sequence_loss.target_mode=membership \
  stage2.sequence_loss.progress_weight=1.0 \
  stage2.sequence_loss.normalization=token \
  stage2.sequence_loss.completion_weight=0.0 \
  stage2.hard_negative_ratio=100 \
  stage2.run_name=v2-membership-paperstage1-100neg-50k
```

注意：pooling readout 的 `completion_weight` 必须置 0（membership 目标下末位不是“整词完成”指示）。

**判据**：AUC ≥ 0.947；若 < 0.94，则 frozen encoder + 新目标不成立，转 P2。

### P1 —— 隔离 readout vs objective（1 个 run，1 天）

同样 v1 目标，readout 换成 v2 的 batch-invariant GRU-last（即 P0 本身），
再加 v1 readout 原样（E2）与 progress 目标基线（已在跑），构成 2×2 部分因子设计。

### P2 —— encoder 适配（1–3 个 run，1–3 天）

1. **PEFT（首选）**：在 encoder 顶部 2–4 个 stack 注入 rank-8/16 LoRA，或只调 bias（BitFit）。
   依据：SUPERB KS 上 frozen 95.32 → BitFit 97.33 / Houlsby 97.17，**超过全量 FT 95.87**；
   顶部若干层 adapter 即可匹配全量迁移。
2. **全量解冻**：`stage2.freeze_encoder=false`，encoder LR 取 head 的 1/10，梯度累积模拟 batch 512。
   预计 0.5–1.0 s/step（估算），V100 单卡 50k 步约 8–14 h。
3. **CORD-KWS 式 calibrated head + 帧级 CTC 顺序目标**：保留 cosine/GRU 打分，加绝对分数监督与 CTC order loss，
   直接针对 ECE/Brier 与尾部（hard EER 9.64）。
4. **XLS-R 0.3B 探针**：先 5% 数据 / 5k 步；dev AUC <0.95 就停（复制 KFC-KWS 的前置检查）。
5. **phoneme CTC adapter trunk**（保留现有 Zipformer）：Step A `ctc_adapter_icefall.yaml`，
   Step B `iceflall_zipformer_stage2_adapter.yaml`，`ctc_weight=0.2`。15 个 run 全部关闭了它，是空白点。

### P3 —— 对齐式 head（改代码，2–4 天）

优先级/成本从低到高：
1. **query-conditioned attentive pooling**（ECAPA 风格 mean+std；phoneme 均值作 query 对帧做单调掩码 attention）
   替代 GRU 末状态，顺带消除 padding 依赖。
2. **长度约束 + 子序列监督**（SLiCK）：按 query 期望时长（音素数 × 40 ms）过滤候选 span，
   用 best-sub-span 损失替代“整段拼接池化”；LP-hard 上这是公开记录里最大的匹配类提升（88.52→94.90 AUC）。
3. **CTC keyframe + cross-attention**（KFC-KWS）：用 Stage I CTC 峰值选 keyframe，与全句表征 cross-attend。
4. **Parallel cross-attention + max-pool**（ParallelKWS）+ **PDA 时长对齐损失**。
5. **POB/EPS**：为前缀重叠构造专项负样本，并加 Equal-weighting Position Scoring 层。
6. **MaxSim late interaction**（ColBERT 风格；无 KWS 先例，高风险高回报）。
7. **FiLM / keyword-conditioned affine**（AdaKWS / Hyper-Matched Filters；<50K 参数）。
8. **Perceiver latent bottleneck**；**v7 bounded segmental 性能修复**（目标 3.4 s/step → <0.5 s/step）。

**版本约束**：AGENTS.md 明确“不要 stamp 版本 8，也不要把 7 写到 pooling checkpoint 上”；
新 readout 需先与维护者确定版本号/兼容键。

### P4 —— 损失与负样本（改代码，2–4 天）

* **FA-aware precision 损失（新增，优先级最高）**：MALEFA 去掉它后 hard EER +7.2、AMI FAR 从 0.007% → 14.5%；
  现有 CVaR 尾损失不是这个目标。实现成一个可微的“被接受负样本数”惩罚项（按 batch 内负样本分数超阈值的比例），
  与 utt BCE 并联，权重从 0.1 起扫。
* **pAUC / AUC-margin 代理**：α=1%，O(1) 内存优先；checkpoint 选择也改用 pAUC。
* **负样本构造换成 AdaKWS 式**：同句其他词、字符替换（s↔z、p↔b）、batch 内最近文本 embedding、~25% random；
  再加 GraphemeAug 的字形插入/删除/替换边界负样本与 POB 的前缀重叠负样本。
* **RHE 式逐帧 hard 负样本 + 比例阶梯**：1:1 → 5:1 → 20:1 → 100:1，配合 ASL 阈值丢弃易负样本/疑似真阳性。
* **课程式调度**：0–20k 易（random 1:1），20k–50k 难（分数挖掘 10:1），CurricularFace 式自适应强调。
* **PLCL 对比 + phoneme memory bank**：音素 insert/replace/delete 合成近音负样本 + 第三类判别器。
* **TTS/LLM confusable 负样本（~30% 占比）**：MM-KWS 路线；配 ZeSTA 式合成域嵌入或对抗判别器防过拟合；
  必须同时报 LP-hard 与 MUSAN accept/day。
* **新指标**：c-AUC（近音词分组）与尾部指标一起报告。

### P5 —— 校准、阈值、分数融合与多尺度决策（不改架构，1 天）

1. **Platt/温度校准** + **AS-norm 队列归一化**（训练无关，SV 上相对 +30%）。
2. **时长感知校准**：按候选音素长度 × 40 ms 分桶拟合 isotonic/logistic 校准（短词系统性低分是 TPR@1%FPR 被压的一个候选原因）。
3. **LTT / 保形风险控制**在 holdout 上选 1% FPR 阈值；**GPD/EVT** 给出 TPR@0.1%FPR 的置信区间。
4. **Stage I span score 与 Stage II logit 的 Platt 融合**（U2-KWS 两遍式相对 +41%）；
   **决策层持续性/一致性门控**：要求 ≥2 个重叠窗口确认，或层间一致性（CDC 在 0.05 FA/h 下相对 miss −46.3%）。
   本机 1 s 网格的 accept 是孤立短尖峰（52 events/53 windows），持续门控的预期收益最大。
5. FA 同时跑 1/2/3 s 网格并折叠连续 accept 为事件率；**平滑后阈值**（max-pooling loss 配方）与
   多分辨率 log-odds 集成（13 模型融合在每种噪声条件下都超过最优单模型）。
6. 报告 **Cllr**（Brummer & du Preez），补齐现有 Brier/ECE。

### P6 —— 背景池消融（1 个 run，半天）

1 s 网格误唤醒几乎全来自 speech，而训练背景池是 music+noise；Howl 也显示 90% 的真误报来自同域近音语音。
把 MUSAN **train** split 的 speech 重新加入背景（与 eval split 不重叠），并做**回放/干扰混合**
（TV/电影、音乐，SIR 0–20 dB；Interspeech 2018 在固定 FA 下相对 FRR −30%~−45%），
观察 hard-split AUC 与 1 s FA/24h 的 trade-off。
（LibriSpeech train-other-500 与 LibriPhrase 同源，只能作同域诊断。）

### P7 —— 多模态 enrollment / MAM（按产品需要，1–2 周）

SD 场景（注册音频）可用 DMA-KWS 的 MAM + LoRA-on-QKV；SI hard-split 指标不因此变化。

### P8 —— 蒸馏与集成（兜底）

* **蒸馏**：先用同一 frozen 特征训 4M 参数 teacher；teacher AUC <0.945 就停止。
* **v1 → v2 的 false-alarm-aware 蒸馏**：v1 在 1% FPR 下排序更好（0.2877 vs 0.2021）但 MUSAN 崩；
  对正样本与高置信挖掘负样本匹配 v1 logits，并屏蔽 v1 在 MUSAN 上的假接受。
* **多 seed / 多 readout 集成 + 温度缩放**：头部差异 ≤0.006 AUC，集成主要稳住尾部。

---

## 5. 建议的实验队列

| # | 实验 | 依赖 | 预计成本 | 成功判据 |
|---|---|---|---|---|
| E1 | paperstage1(1460h) + v2 + membership + 100:1（P0） | 无 | ~3 h | AUC ≥ 0.947 |
| E2 | 同 encoder + v1 readout + membership + 100:1（作者目标原样） | 无 | ~3 h | AUC ≥ 0.955 |
| E3 | zh-en-3M + phoneme adapter + membership + 100:1（P2.5） | Step A trunk | 3 h + 3 h | 追平 E1/E2 |
| E4 | encoder 顶部 2–4 层 LoRA/BitFit（P2.1） | E1 配方 | 4–8 h（估算） | 相对 E1 +≥0.005 且尾部不降 |
| E5 | CORD-KWS 式 calibrated head + CTC order loss（P2.3） | 71 token tokenizer | 1–2 天开发 + 3 h | ECE/Brier 相对 −20% 且 TPR@1%FPR +≥2 点 |
| E6 | XLS-R 0.3B frozen 探针（5% 数据 / 5k 步，P2.4） | 特征预计算 | 1–2 天 | dev AUC ≥ 0.95 才继续 |
| E7 | pAUC / AUC-margin 代理（P4，需改 `dma_kws/stage2/losses.py`） | E1 配方 | 1–2 天开发 + 3 h | TPR@1%FPR +≥1.5 点、AUC 降 ≤0.002 |
| E8 | RHE 逐帧挖掘 + 5:1 → 100:1 阶梯（P4） | 一次全量打分 | 1 天 + 2×3 h | TPR@1%FPR 与 MUSAN accept/day 同时改善 |
| E9 | PLCL 对比 + memory bank + 前缀重叠负样本（P4） | 音素 tokenizer | 2–3 天开发 + 3 h | c-AUC / TPR@0.1%FPR 提升 |
| E10 | CTC keyframe + cross-attn head（P3.3，需确认版本号） | Stage I CTC | 3–4 天开发 + 3 h | AUC ≥ E1，长音频更稳 |
| E11 | 校准 + AS-norm + LTT 阈值 + EVT 尾部 + 两级分数融合（P5） | E1 checkpoint | 1 天 | 1% FPR 阈值在 holdout 落在 [0.8%,1.2%]；给出 Cllr/EVT CI |
| E12 | 多尺度 FA（1/2/3 s）+ 事件率报告（P5.5） | 任意 checkpoint | 半天 | 输出可比较的事件率曲线 |
| E13 | MUSAN speech 背景消融（P6） | 无 | 3 h | 1 s FA/24h 降且 AUC 不掉出噪声底 |
| E14 | v1 → v2 false-alarm-aware 蒸馏（P8） | v1 checkpoint | 3 h | TPR@1%FPR ≥25% 且 MUSAN accept/day 不高于 v2 |
| E15 | MALEFA 式 FA-aware precision 损失（P4，与 E7 可合并） | E1 配方 | 1–2 天开发 + 3 h | 固定 FPR 下接受数下降、hard EER 不升 |
| E16 | AdaKWS 式负样本（同句/字符替换/batch-nearest）+ c-AUC（P4） | E1 配方 | 1–2 天开发 + 3 h | TPR@1%FPR +≥2 点；c-AUC +≥3 点 |
| E17 | 持续性门控 + 时长校准 + 多分辨率聚合（P5.2/5.4/5.5） | 任意 checkpoint | 1 天 | 1 s FA/24h 相对 −20% 且 TPR 损失 ≤1 点 |

排期：**E1/E2/E11 三天内给答案 → E3/E4/E5/E6 → E7/E8/E15/E16 → E9/E10/E17 → E12–E14**。
不要在 E1–E6 之前先动 head 结构，否则无法归因。

---

## 6. 参考文献

**同任务 / UDKWS**
1. DMA-KWS. arXiv:2605.22120. https://arxiv.org/abs/2605.22120 ；官方 repo https://github.com/aizhiqi-work/DMA-KWS
2. DS-KWS: Dual Data Scaling. arXiv:2510.10740. https://arxiv.org/abs/2510.10740
3. KFC-KWS: Keyframe Fusion with CTC. arXiv:2606.10365. https://arxiv.org/abs/2606.10365
4. CORD-KWS. arXiv:2609.31869. https://arxiv.org/abs/2609.31869
5. PLCL. arXiv:2412.20805. https://arxiv.org/abs/2412.20805
6. ParallelKWS. arXiv:2408.03593. https://arxiv.org/abs/2408.03593
7. U2-KWS. arXiv:2312.09760. https://arxiv.org/abs/2312.09760
8. MM-KWS. arXiv:2406.07310. https://arxiv.org/abs/2406.07310
9. PhonMatchNet. arXiv:2308.16511. https://arxiv.org/abs/2308.16511
10. Homogeneous Audio-Text Embedding. arXiv:2308.06472. https://arxiv.org/abs/2308.06472
11. No Word Left Behind (POB+EPS). arXiv:2602.08930. https://arxiv.org/abs/2602.08930
12. MALEFA. arXiv:2604.03689. https://arxiv.org/abs/2604.03689
13. MFA-KWS. arXiv:2505.19577. https://arxiv.org/abs/2505.19577
14. Learning Audio-Text Agreement (LibriPhrase). arXiv:2206.15400. https://arxiv.org/abs/2206.15400
15. KWS-Whisper. arXiv:2309.09552. https://arxiv.org/abs/2309.09552
16. GE2E-KWS. arXiv:2410.16647. https://arxiv.org/abs/2410.16647
17. Massive Open-Vocabulary Keyword Spotting. arXiv:2606.11279. https://arxiv.org/abs/2606.11279
18. CTC-aligned Audio-Text Embedding. arXiv:2406.07923. https://arxiv.org/abs/2406.07923
19. W-CTC. ISCA Interspeech 2025. https://www.isca-archive.org/interspeech_2025/kim25d_interspeech.html
20. Lattice-Free Open Vocabulary KWS. IEEE 10485678. https://ieeexplore.ieee.org/document/10485678
21. Duration-Aware Phone Embedding Upsampling. IEEE 10983304. https://ieeexplore.ieee.org/document/10983304
22. Efficient Audio-Text KWS. ACL ICON 2024. https://aclanthology.org/anthology-files/anthology-files/pdf/icon-2024/2024.icon2024-1.31.pdf
23. NN end-to-end QbE-STD. arXiv:1911.08332. https://arxiv.org/abs/1911.08332
24. Zero-resource audio-only STD by template matching. ISCA Interspeech 2011. https://www.isca-archive.org/interspeech_2011/muscariello11_interspeech.html

**融合 / 对齐**
25. ColBERT. arXiv:2004.12832. https://arxiv.org/abs/2004.12832
26. ColBERTv2. arXiv:2112.01488. https://arxiv.org/abs/2112.01488
27. CLaMR. arXiv:2506.06144. https://arxiv.org/abs/2506.06144
28. Hyper-Matched Filters. arXiv:2508.04857. https://arxiv.org/abs/2508.04857
29. Multi-task Cross Attention KWS. arXiv:2107.07634. https://arxiv.org/abs/2107.07634
30. MoChA. arXiv:1712.05382. https://arxiv.org/abs/1712.05382
31. CTC-synchronous Training. arXiv:2005.04712. https://arxiv.org/abs/2005.04712
32. CIF. arXiv:1905.11235. https://arxiv.org/abs/1905.11235
33. DONUT. arXiv:1811.10736. https://arxiv.org/abs/1811.10736
34. NTC-KWS. arXiv:2412.12614. https://arxiv.org/abs/2412.12614
35. KWT. arXiv:2104.00769. https://arxiv.org/abs/2104.00769
36. ST-AttNet. arXiv:2108.12146. https://arxiv.org/abs/2108.12146
37. ECAPA-TDNN. arXiv:2005.07143. https://arxiv.org/abs/2005.07143
38. AdaKWS. arXiv:2309.08561. https://arxiv.org/abs/2309.08561
39. CLAP. arXiv:2206.04769. https://arxiv.org/abs/2206.04769
40. m-LTM. arXiv:2405.10084. https://arxiv.org/abs/2405.10084
41. OTReg. arXiv:2508.08131. https://arxiv.org/abs/2508.08131
42. SLAM. arXiv:2110.10329. https://arxiv.org/abs/2110.10329
43. Perceiver. arXiv:2103.03206. https://arxiv.org/abs/2103.03206
44. USM + Perceiver classifier. arXiv:2310.13010. https://arxiv.org/abs/2310.13010
45. FLASH-MAXSIM. arXiv:2605.29517. https://arxiv.org/abs/2605.29517
46. MATE. arXiv:2601.14012. https://arxiv.org/abs/2601.14012

**encoder / 迁移 / PEFT**
47. WavLM. arXiv:2110.13900. https://arxiv.org/abs/2110.13900
48. HuBERT. arXiv:2106.07447. https://arxiv.org/abs/2106.07447
49. wav2vec 2.0. arXiv:2006.11477. https://arxiv.org/abs/2006.11477
50. XLS-R. arXiv:2111.09296. https://arxiv.org/abs/2111.09296
51. w2v-BERT 2.0 / SeamlessM4T. arXiv:2308.11596. https://arxiv.org/abs/2308.11596
52. Whisper. arXiv:2212.04356. https://arxiv.org/abs/2212.04356
53. SUPERB. arXiv:2105.01051. https://arxiv.org/abs/2105.01051
54. Zipformer. arXiv:2310.11230. https://arxiv.org/abs/2310.11230
55. Conformer. arXiv:2005.08100. https://arxiv.org/abs/2005.08100
56. Branchformer. arXiv:2207.02971. https://arxiv.org/abs/2207.02971
57. E-Branchformer. arXiv:2210.00077. https://arxiv.org/abs/2210.00077
58. Efficient Adapter Transfer. arXiv:2202.03218. https://arxiv.org/abs/2202.03218
59. Exploring Efficient-Tuning in SSL Speech. arXiv:2210.06175. https://arxiv.org/abs/2210.06175
60. SURE. arXiv:2303.03267. https://arxiv.org/abs/2303.03267
61. LoRA. arXiv:2106.09685. https://arxiv.org/abs/2106.09685
62. BitFit. arXiv:2106.10199. https://arxiv.org/abs/2106.10199
63. Prefix-Tuning. arXiv:2101.00190. https://arxiv.org/abs/2101.00190
64. DistilHuBERT. arXiv:2110.01900. https://arxiv.org/abs/2110.01900
65. FitHuBERT. arXiv:2207.00555. https://arxiv.org/abs/2207.00555
66. On-device KD-S3RL KWS. arXiv:2307.02720. https://arxiv.org/abs/2307.02720
67. SSL for KWS with light-weight transformers. arXiv:2303.04255. https://arxiv.org/abs/2303.04255
68. Few-shot KWS with SSL. arXiv:2506.17686. https://arxiv.org/abs/2506.17686
69. Pasad et al. layer-wise analysis. arXiv:2107.04734. https://arxiv.org/abs/2107.04734
70. Layer-wise analysis ICASSP 2023. arXiv:2211.03929. https://arxiv.org/abs/2211.03929
71. Children zero-shot KWS layer selection. arXiv:2508.21248. https://arxiv.org/abs/2508.21248

**目标 / 难负样本 / 校准**
72. When AUC meets DRO. arXiv:2203.00176. https://arxiv.org/abs/2203.00176
73. Range-pAUC optimization. arXiv:2203.01505. https://arxiv.org/abs/2203.01505
74. Instance-wise pAUC estimator. arXiv:2210.03967. https://arxiv.org/abs/2210.03967
75. AUC-margin surrogate. arXiv:2012.03173. https://arxiv.org/abs/2012.03173
76. AUC optimization for robust KWS. arXiv:2107.05859. https://arxiv.org/abs/2107.05859
77. Multi-class AUC for KWS. ISCA Interspeech 2022. https://www.isca-archive.org/interspeech_2022/xu22h_interspeech.html
78. Optimize what matters. arXiv:2011.01151. https://arxiv.org/abs/2011.01151
79. ArcFace. arXiv:1801.07698. https://arxiv.org/abs/1801.07698
80. CosFace. arXiv:1801.09414. https://arxiv.org/abs/1801.09414
81. AM-Softmax. arXiv:1801.05599. https://arxiv.org/abs/1801.05599
82. Sub-center ArcFace. ECCV 2020. https://www.ecva.net/papers/eccv_2020/papers_ECCV/html/1445_ECCV_2020_paper.php
83. CurricularFace. arXiv:2004.00288. https://arxiv.org/abs/2004.00288
84. Multi-Similarity. arXiv:1904.06627. https://arxiv.org/abs/1904.06627
85. In Defense of the Triplet Loss. arXiv:1703.07737. https://arxiv.org/abs/1703.07737
86. InfoNCE / CPC. arXiv:1807.03748. https://arxiv.org/abs/1807.03748
87. Focal Loss. arXiv:1708.02002. https://arxiv.org/abs/1708.02002
88. Asymmetric Loss. arXiv:2009.14119. https://arxiv.org/abs/2009.14119
89. Regional Hard-Example mining for KWS. ICASSP 2020. https://pure.nwpu.edu.cn/zh/publications/mining-effective-negative-training-samples-for-keyword-spotting/
90. Adversarial Retriever-Ranker. arXiv:2110.03611. https://arxiv.org/abs/2110.03611
91. Generating adversarial confusables. arXiv:2201.00167. https://arxiv.org/abs/2201.00167
92. Disentangled adversarial training. arXiv:2408.13355. https://arxiv.org/abs/2408.13355
93. Clustering-based hard negative sampling. ISCA Interspeech 2025. https://www.isca-archive.org/interspeech_2025/masztalski25_interspeech.html
94. When Hard Negatives Hurt. arXiv:2606.01304. https://arxiv.org/abs/2606.01304
95. On Calibration of Modern NNs. arXiv:1706.04599. https://arxiv.org/abs/1706.04599
96. Calibration for speaker verification. arXiv:2203.15106. https://arxiv.org/abs/2203.15106
97. Brummer & du Preez, Cllr. Computer Speech & Language 2006. https://www.sciencedirect.com/science/article/pii/S0885230805000483
98. AS-norm. ISCA Interspeech 2017. https://www.isca-archive.org/interspeech_2017/matejka17_interspeech.html
99. Learn then Test. arXiv:2110.01052. https://arxiv.org/abs/2110.01052
100. Conformal Risk Control. arXiv:2208.02814. https://arxiv.org/abs/2208.02814
101. EVT for open-set scores. arXiv:1808.09902. https://arxiv.org/abs/1808.09902
102. Quantization-based score calibration. arXiv:2510.15432. https://arxiv.org/abs/2510.15432

**数据 / 增广 / 长音频**
103. Synth4Kws. arXiv:2407.16840. https://arxiv.org/abs/2407.16840
104. LLM-Synth4KWS. arXiv:2505.22995. https://arxiv.org/abs/2505.22995
105. Synthetic audio in training keyword spotters (ICASSP 2021). https://www.2021.ieeeicassp.org/2021.ieeeicassp.org/Papers/ViewPaper1bf7.html
106. GraphemeAug. arXiv:2505.14814. https://arxiv.org/abs/2505.14814
107. Howl. NLP-OSS 2020. https://aclanthology.org/2020.nlposs-1.9/
108. Playback interference augmentation. arXiv:1808.00563. https://arxiv.org/abs/1808.00563
109. Cross-layer Discrimination Consistency. arXiv:2412.12635. https://arxiv.org/abs/2412.12635
110. ZeSTA (synthetic/real domain adaptation). arXiv:2603.04219. https://arxiv.org/abs/2603.04219
111. Personalized TTS for dysarthric ASR. arXiv:2508.06391. https://arxiv.org/abs/2508.06391
112. SLiCK: Length-constrained keyword spotting. arXiv:2409.09067. https://arxiv.org/abs/2409.09067
113. Max-pooling-loss LSTM. arXiv:1705.02411. https://ar5iv.labs.arxiv.org/html/1705.02411
114. Two-stage multi-resolution ensembles. arXiv:2310.11379. https://arxiv.org/abs/2310.11379
115. SpecAugment. arXiv:1904.08779. https://arxiv.org/abs/1904.08779
116. MUSAN. arXiv:1510.08484. https://ar5iv.labs.arxiv.org/html/1510.08484
117. The Taste of IPA / CLAP-IPA. NAACL 2024. https://aclanthology.org/2024.naacl-long.43/

---

## 7. 附录：数字是怎么算出来的

* §0/§1.2 方差分解：对 Experiment-Log §1 的 **12 个 streaming run** 做 encoder（3 水平）× readout（4 水平）
  两因子主效应分解：SS_encoder = 94.0%，SS_readout = 4.6%，残差 1.4%（残差含操作点、seed 与交互）。
* §1.3 DS-KWS 数字来自其 Table 1/2/3；作者 v1 数字来自 `outputs/qbyt_review_2026-10-03/v1_evaluations.csv`。
* §1.4 FA 数字来自 `docs/wiki/False-Alarm-Evaluation.md` §7.1–7.3（阈值 0.5）。
* **encoder 同一性验证**：`paper-stage1/ls-gs-1460.pt` 与 `exp/author_v1/stage1_v1.pt` 的 249 个 encoder 张量
  按名字+float32 字节重算 sha256，均为 `94e3d65e3b411be42b45…`。
* `v1-wenet-subset` 的 2,684 anchors 由 `pyarrow.parquet.read_table(...)` 现场统计。
* 复算脚本见 `outputs/qbyt_v41_audit_2026-10-03/audit.py` 与 `outputs/qbyt_review_2026-10-03/`。
