# QbyT v4.1（保留 sink token）的小改动优化方案（v2）

> 生成日期：2026-10-03。目标：**保留 v4.1 主架构（sink token + eps_softmin readout）**，
> 只做小改动，让它的优势从“不明显 / encoder 相关”变成稳定为正。
> v2 版新增：① 对“远距离桶解钳”假设做了**零训练证伪实验**（结果是否定的，见 §3）；
> ② 四路文献调研已并入（§4）。

---

## 0. TL;DR

1. **v4.1 目前只在“低 FPR 尾部 + 弱 encoder”上赢**：相对 v4，zh-en-3M 上 TPR@1%FPR +0.0264、
   GS-finetune 上 AUC +0.0122，但 GS-base full-context 上 AUC −0.0066；相对 v2 全面落后
   （0.9350 vs 0.9376 等），却贵 2.3×。

2. **两个来自训练后权重的结构性事实**（v41-musan-zhen3m-50k / step 47,500）：
   * **相对位置 bias 在 ≥49.35 帧（2.0 s）处被钳进最后一个桶**（该桶学习值 +1.64/+2.16/−0.86/−0.41），
     3 s 窗口有 **11.6%** 的帧对吃它，而训练 anchor 是 0.5–2.0 s。
   * **sink 状态从未被读出**（所有 logit 来自 text 位置）；学习到的 text→sink 偏置
     在 4 头里 3 头是负的，sink→text 是正的，sink 实际是“读 query 的 register”。

3. **“把远距离桶置零”这个最省事的修法已被零训练实验证伪**（§3）：
   在真正的 0.409 最差窗口上，置零只把 0.409 降到 0.386，却把 3 s 网格的中间尾部**抬高**
   （≥0.1 的窗口 4 → 6，≥0.2 的 1 → 2）。学习到的远距离 bias 至少在背景上是**有保护作用**的，
   不能简单删掉。→ **长度问题应该用“长度条件校准/归一化 + 让训练见到 3 s”解决，而不是删桶。**

4. **第 0 步已实测：两条零成本改动就能全面超过 v2**（详见 §4.5）：
   **softmin 温度 1.0 → 0.05–0.1**（AUC +0.0019、TPR@1%FPR +0.028，零训练）
   ＋ **读出 sink 状态**（半拟合/半评估 AUC 0.94057 / EER 0.12723 / TPR@1% 0.23621 / TPR@0.1% 0.04282，
   全面超过 v2 的 0.93765 / 0.12864 / 0.20207 / 0.02347）。sink 单独线性探针 AUC 0.93595，
   与整个 pooled readout（0.93589）相当——信息一直在 sink 里，只是没被读出。

5. **最值得做的小改动**（全部保留 sink）：
   * **S3 sink 状态读出**（约 129 参数；文献上 register 读出在异常检测上降 FPR 2–3%，且它是我们唯一
     未被使用的全局状态）；
   * **S4 sink 身份补全**（modality embedding + 独立 index；目前 sink 在 relative bias 里与第 0 帧共享 index）；
   * **S8 可学习温度**（1 参数；先做固定 T 扫描）；
   * **S11 长度条件分数归一化**（0 参数，直接针对 1 s/3 s 分布差异：1 s 网格 p99.99=0.688、53 个 ≥0.5；
     3 s 网格 p99.99=0.099、1 个 ≥0.2）；
   * **S12 滑窗平滑 + N-of-M 持续门控**（0 训练，针对孤立尖峰）；
   * **S9 成本修复**（FlexAttention score_mod / batch-independent bias，目标回到约 0.2 s/step）。

6. **learned text position：看起来像随机码，但不能零训练替换（重要更正）**。
   它有 128×128 = 16,384 参数（占 v4.1 相对 v4 参数增量的约 93%），位移 std 0.074、相邻行 cosine 0.017
   （都说明它没有可解释的位置结构）；但**零训练换成 sinusoidal 后 AUC 从 0.93501 崩到 0.70264**（§4.5-1c）。
   整个头已经围着这张“固定随机码”适配，换码必须重训才能判断。默认**保留 learned**，S7 降级为需要从头验证的可选项。

---

## 1. v4.1 现状数据

### 1.1 v4.1 vs v4（审计 v41_vs_v4.csv）

| group | ΔAUC | ΔEER | ΔTPR@1%FPR | ΔTPR@0.1%FPR | ΔpAUC | ΔBrier | 墙钟比 |
|---|---|---|---|---|---|---|---|
| zh-en-3M streaming | +0.0027 | −0.0027 | **+0.0264** | +0.0040 | +0.0090 | ~0 | 2.29× |
| GS-finetune streaming | **+0.0122** | **−0.0120** | +0.0066 | +0.0006 | +0.0013 | −0.0071 | 2.24× |
| GS-base streaming | −0.0012 | +0.0025 | +0.0008 | −0.0003 | −0.0015 | −0.0003 | 2.20× |
| GS-base full context | **−0.0066** | +0.0084 | −0.0084 | −0.0048 | −0.0030 | +0.0053 | 2.35× |

与 v2（422K，GRU-last，无 sink）相比：zh-en 0.9350 vs 0.9376、GS-finetune 0.8687 vs 0.8809、
GS-base stream 0.9119 vs 0.9271。

### 1.2 背景分数分布强烈依赖窗口长度（本机同一 checkpoint 的两套网格）

| 网格 | 窗口数 | max | p99.99 | p99.9 | p99 | ≥0.05 | ≥0.1 | ≥0.2 | ≥0.5 |
|---|---|---|---|---|---|---|---|---|---|
| 3 s/3 s | 51,994 | 0.4090 | 0.0985 | 0.0324 | 0.0075 | 26 | 4 | 1 | 0 |
| 1 s/1 s | 156,929 | 0.9606 | 0.6877 | 0.2383 | 0.0111 | 532 | 341 | 188 | 53 |

* 3 s 的“0.409”是一个孤立极值（p99.99 只有 0.0985）；1 s 是一条宽尾。
* → 单一全局阈值在两个网格上不可能同时正确，**长度条件归一化/校准是零成本的第一优先级**（S11）。

---

## 2. 训练后权重取证（v41-musan-zhen3m-50k / step 47,500）

训练配置：Adam，lr 5e-4，weight_decay 0，warmup 500，50k 步，batch 128，fp16，encoder frozen。

### 2.1 相对位置 bias 在 2.0 s 处钳制

T5 bidirectional bucket（32 桶 / max_distance 64）：每方向 16 桶，max_exact=8；
解 val_if_large ≥ 15 得距离 **n ≥ 49.35 帧（2.0 s）**落入最后桶。

| 音频长度 | 帧数 | 最后桶占比 | head0 mean/std | head1 mean/std | head2 | head3 |
|---|---|---|---|---|---|---|
| 1.0 s | 25 | 0.0% | −0.17 / 0.21 | −0.14 / 0.23 | +0.07 / 0.09 | −0.13 / 0.14 |
| 2.0 s | 50 | 0.0% | −0.06 / 0.34 | −0.01 / 0.37 | +0.03 / 0.13 | −0.10 / 0.13 |
| 3.0 s | 75 | **11.6%** | +0.24 / 0.65 | +0.35 / 0.79 | −0.11 / 0.32 | −0.13 / 0.15 |
| 4.0 s | 100 | **25.5%** | +0.49 / 0.78 | +0.67 / 0.97 | −0.24 / 0.40 | −0.17 / 0.17 |

最后桶学习值：+1.638 / +2.156 / −0.858 / −0.408。

### 2.2 sink 的模态偏置与“只进不出”

pair_table[q_mod, k_mod, head]（text / audio / sink）：

| q → k | text | audio | sink |
|---|---|---|---|
| text | −0.124 −0.189 −0.287 −0.180 | **+0.154 +0.203 +0.076 +0.195** | **−0.349 −0.239 +0.073 −0.167** |
| audio | −0.071 −0.085 −0.008 −0.053 | +0.068 +0.086 +0.000 +0.028 | +0.037 −0.091 +0.149 +0.332 |
| sink | **+0.196 +0.193 +0.009 +0.008** | −0.198 −0.197 +0.001 −0.002 | +0.243 +0.311 −0.371 −0.133 |

* text→audio 正偏置（合理）；**text→sink 3/4 头为负**；sink→text 正偏置。
* sink_token std 0.118、absmax 0.363；**sink 最终状态不参与任何 logit**。
* sink 无 modality embedding、无 position；在 repacked_modality_and_index 里 **sink 的 index 与第 0 个音频帧同为 0**
  （audio bias 只作用于 audio–audio 对，所以 sink 的 logits 实际只由 pair_table 决定）。

### 2.3 learned text positional embedding 无结构

| 参数 | step 1 std | step 47.5k std | 元素位移 std |
|---|---|---|---|
| qbyt.text_pos_emb.weight（128×128） | 0.9924 | 0.9892 | **0.074** |
| qbyt.relative_bias.audio_buckets | 0 | 0.4175 | **0.418** |
| qbyt.relative_bias.pair_table | 0 | 0.1787 | 0.179 |
| qbyt.text_projection.weight | 0.9970 | 1.0137 | 0.206 |

相邻行 cosine 0.017 ± 0.088（随机水平）。

**更正（2026-10-03 实测）**：虽然这张表“看起来是随机码、且几乎没被训练”，但它**不能零训练替换**。
把 text_position 换成 sinusoidal（其余权重不动）后，hard split AUC 从 0.93501 掉到 **0.70264**，
EER 从 0.131 掉到 0.355（§4.5-1c）。原因：它是一份**固定的随机位置码**，整个头（注意力与读出）
已经围着这串具体向量适配；换一套码等于换了输入分布。因此“learned 是死重”只对**新训练**成立
（新 run 可以从头学 sinusoidal），对**已训练 checkpoint** 不成立。

### 2.4 2.3× 成来源（文献补充：很可能是 mask 构造，不全是 kernel）

RelativeAttentionBias.compute 物化 [B, L, L] int → audio_buckets[buckets] → [B, L, L, H] float
（B=64, L=200, H=4 约 41 MB），再 permute/float/masked_fill(-inf)/reshape(B*H, L, L)；
另有一个 pair_table 的 [B, L, L, H] 张量。T5 官方实现是 **batch-independent 的 [1, H, Lq, Lk]、只在第 0 层计算**，
我们完全可以在保持数值一致的前提下复刻（见 S9）。

---

## 3. “解钳”假设的零训练证伪（本报告最重要的实验）

**做法**：加载训练好的 v4.1 checkpoint，把 audio_buckets 的两个最后桶（正方向 15、负方向 31）置零，
其余权重不动；在 MUSAN 上重新评测。先跑 60 文件（3.54 h）子集，再跑官方 3 s 网格里分数最高的 5 个文件（845 窗口）。

**结果（top-5 文件，845 窗口）**：

| 指标 | 原始 v4.1 | 置零后 |
|---|---|---|
| 最差窗口（music-jamendo-0143 @21–24 s） | **0.4090** | 0.3862 |
| ≥0.4 的窗口 | 1 | 0 |
| ≥0.2 的窗口 | 1 | **2** |
| ≥0.1 的窗口 | 4 | **6** |
| ≥0.05 的窗口 | 6 | 6 |
| 其它典型窗口 | us-gov-0029 0.167 | **0.240** |
| | us-gov-0035 0.099 | **0.179** |
| | us-gov-0070 0.105 | **0.177** |

**结论**：置零只轻微削掉单个极值，却把中间尾部整体抬高。**远距离桶的偏置在背景上起保护作用**，
“把 bias 解钳/置零”不是可行的小改动。长度问题应转向：
① 长度条件校准/归一化（S11）；② 让训练分布覆盖 3 s（长度抖动，S13）；
③ 若要改 bias，只能在**重训**中让新桶从数据里学（S2/S14），不能零训练删。

*说明*：这是 5 文件 / 845 窗口的快速证伪，不是全量 43.7 h 结论；但方向（中间尾部变差）在多个文件上一致。

---

## 4. 文献调研（四路汇总）

### 4.1 Attention sink / register / aggregator token

| 论文 | 机制/结论 | 对我们的启示 |
|---|---|---|
| StreamingLLM（ICLR 2024）[arXiv:2309.17453](https://arxiv.org/abs/2309.17453) | sink = softmax 的“不 attend 任何东西”槽位，等价 SoftMax-1；**Zero-Sink 几乎不可用**（ppl 27.9 → 29214），可学习 sink 才有效；两个 sink 不比一个好 | 保留 1 个 sink；**零初始化 + 只做 key 不够**，要给它可学的 key bias / 值通道 |
| When Attention Sink Emerges（ICLR 2025）[arXiv:2410.10781](https://arxiv.org/abs/2410.10781) | sink 由优化产生，作用是**key 侧偏置**存多余的 attention 分数；可学习 sink 会长出 massive activation | 给 sink 加**每头 key-logit bias**；约束 sink 状态范数 |
| Massive Activations（ICML 2024）[arXiv:2402.17762](https://arxiv.org/abs/2402.17762) | 极少数的巨大激活是关键常数偏置 | 监控/限制 sink 范数 |
| ViT registers（ICLR 2024）[arXiv:2309.16588](https://arxiv.org/abs/2309.16588) | 1 个 register 即可消除高范数伪影，更多可再涨下游指标；register 不参与输出 | 1 个 sink 保留；2–4 个可试（S5） |
| Registers for robust adaptation（ICASSP 2025）[arXiv:2501.04784](https://arxiv.org/abs/2501.04784) | **读出 register 状态**（CLS ⊕ mean register）：OOD +2–4%，**异常检测 FPR −2–3%** | 我们最相关的一条：sink 状态没被读出（S3） |
| Mamba-R（CVPR 2025）[arXiv:2405.14858](https://arxiv.org/abs/2405.14858) | register 均匀插入序列并**回收进最终决策** | sink 状态并入 readout（S3） |
| Memory Transformer（2020）[arXiv:2006.11527](https://arxiv.org/abs/2006.11527)；Set Transformer/PMA（ICML 2019）[arXiv:1810.00825](https://arxiv.org/abs/1810.00825)；Perceiver（ICML 2021）[arXiv:2103.03206](https://arxiv.org/abs/2103.03206) | 可学习 seed 对集合做 cross-attention 产生聚合输出 | sink = 一个从不产生输出的 PMA seed；**读出它就是最自然的小改动** |
| AST（Interspeech 2021）[arXiv:2104.01778](https://arxiv.org/abs/2104.01778)；KWT（Interspeech 2021）[arXiv:2104.00769](https://arxiv.org/abs/2104.00769) | 音频/KWS 里加一个可学习 CLS/distillation token 是标准操作 | 我们的 sink 与它们同源 |
| AVSR sink decorrelation（ICASSP 2026）[arXiv:2510.22603](https://arxiv.org/abs/2510.22603) | 低语义 sink 与 BOS 高 cosine 会放大激活；**decorrelation loss** 可减少 sink | 加 sink–audio 去相关正则（S6） |
| Sink gating（2026）[arXiv:2604.03316](https://arxiv.org/abs/2604.03316) | sink 编码全局先验但压制局部证据；**逐层 sink gate** 可调 | 逐层/逐头 sink 门控（S5/S6） |
| SinkProbe（2026）[arXiv:2609.08574](https://arxiv.org/pdf/2609.08574) | sink 由训练目标产生，架构类修复在小模型上不总是复现 | 期望值放低，先做诊断 |

### 4.2 Pooling / no-match token / 校准与验证决策

| 论文 | 机制 | 启示 |
|---|---|---|
| ECAPA-TDNN（Interspeech 2020）[ISCA](https://www.isca-archive.org/interspeech_2020/desplanques20_interspeech.html)；Attentive Statistics Pooling（2018）[arXiv:1803.10963](https://arxiv.org/abs/1803.10963) | 注意力加权 mean+std，比 mean pooling EER −7.5%/−8.1% | 备选读出：对 audio 段做统计池化（S3 的替代） |
| MQMHA + inter-topK（VoxSRC-21）[arXiv:2110.05042](https://arxiv.org/abs/2110.05042) | 多头多 query 池化 + 近邻惩罚 | 多头 sink query + 尾部惩罚 |
| Dummy Prototypical Networks（Interspeech 2022）[arXiv:2206.13691](https://arxiv.org/abs/2206.13691) | 加“dummy prototype”显式 null 类，开放集 few-shot KWS AUROC 提升 | **sink_logit 作为显式 null 假设**（S3） |
| OpenMax（CVPR 2016）[CVF](https://openaccess.thecvf.com/content_cvpr_2016/html/Bendale_Towards_Open_Set_CVPR_2016_paper.html)；ARPL（TPAMI 2022）[arXiv:2103.00953](https://arxiv.org/abs/2103.00953) | 极值理论重标定未知类分数；可学习 reciprocal points | 背景原型 + 尾部拟合 |
| Energy-based OOD（NeurIPS 2020）[arXiv:2010.03759](https://arxiv.org/abs/2010.03759)；LogitNorm（ICML 2022）[arXiv:2205.09310](https://arxiv.org/abs/2205.09310)；ReAct（NeurIPS 2021）[arXiv:2111.12797](https://arxiv.org/abs/2111.12797) | energy = −T·logsumexp(logits/T)；logit 归一化/激活裁剪 | 我们的 softmin 与 energy 同型；可做 null 混合（S3）与 logit 裁剪（S10） |
| 温度缩放（ICML 2017）[arXiv:1706.04599](https://arxiv.org/abs/1706.04599)；SV 校准（2022）[arXiv:2203.15106](https://arxiv.org/abs/2203.15106)；MagnetO（2020）[arXiv:2102.01760](https://arxiv.org/abs/2102.01760) | 单温度/Platt；**用池化层激活与时长做条件校准** | S8 可学习温度 + S11 长度条件校准 |
| AS-norm / cohort（Interspeech 2017）[ISCA](https://www.isca-archive.org/interspeech_2017/matejka17_interspeech.html)；condition-matched cohort（2015）[ISCA](https://www.isca-archive.org/interspeech_2015/nautsch15_interspeech.html) | 用 top-K cohort 做 per-trial z-norm；条件匹配 cohort 更好（相对 +30%） | S11 的直接依据 |
| Speech Commands（2018）[arXiv:1804.03209](https://arxiv.org/abs/1804.03209)；keyword-filler RNN-T（ASRU 2017）[arXiv:1710.09617](https://arxiv.org/abs/1710.09617) | 显式 Unknown/Silence 类；filler = null 假设 | 把背景/无匹配当一等标签 |

**关键提醒（子代理分析）**：把常数 null logit 并入 softmin 只是分数的单调压缩，**AUC/EER 不变**；
只有**输入相关**的 null（w·h_sink + c）才可能改变排序（S3 必须用输入相关的形式）。

### 4.3 高效 relative bias 与 attention 温度

| 论文/实现 | 要点 | 启示 |
|---|---|---|
| T5（JMLR 2020）[arXiv:1910.10683](https://arxiv.org/abs/1910.10683)；[HF 实现](https://github.com/huggingface/transformers/blob/main/src/transformers/models/t5/modeling_t5.py) | bias 形状 **[1, H, Lq, Lk]（batch-independent）**、**只在第 0 层计算**、max_distance=128、scaling=1.0 | S9：复刻这三点即可去掉大部分 [B,L,L,H] 流量 |
| ALiBi（ICLR 2022）[arXiv:2108.12409](https://arxiv.org/abs/2108.12409) | 只有 per-head 斜率，无表，长度外推好 | 若训练后 audio_buckets 近似线性，可换 ALiBi（本机桶值非线性，故优先级低） |
| Shaw（NAACL 2018）[arXiv:1803.02155](https://arxiv.org/abs/1803.02155) | 相对 + 绝对位置无额外收益 | 对新头：text 侧用 sinusoidal 即可；**对已训练 v4.1：不能换**（§4.5-1c） |
| Conformer（Interspeech 2020）[arXiv:2005.08100](https://arxiv.org/abs/2005.08100)；Sperber（Interspeech 2020）[arXiv:2005.09940](https://arxiv.org/abs/2005.09940)；语音增强 PE 对比（Interspeech 2024）[ISCA](https://www.isca-archive.org/interspeech_2024/zhang24n_interspeech.html) | 语音 encoder 里相对位置对长度泛化更好 | audio 侧保留 relative bias |
| Kazemnejad（NeurIPS 2023）[arXiv:2305.19466](https://arxiv.org/abs/2305.19466) | **NoPE 超过所有显式位置编码**；相对 bias 不等于自动长度鲁棒 | 值得做“第 1 层无 bias”消融 |
| FlexAttention（2024）[arXiv:2412.05496](https://arxiv.org/abs/2412.05496) | score_mod 把 bias 编进 fused kernel | S9 的首选实现 |
| HyPE（2023）[arXiv:2310.19676](https://arxiv.org/abs/2310.19676)；RIB（2026）[arXiv:2603.06738](https://arxiv.org/abs/2603.06738) | 把 bias 变成 q/k 的额外通道（rank-factorized），无需物化 O(L²) | S9 的备选 |
| Swin v2（CVPR 2022）[arXiv:2111.09883](https://arxiv.org/abs/2111.09883) | cos(q,k)/τ + B，τ 逐头可学且 >0.01 | 可学习 attention 温度 |
| QKNorm（2020）[arXiv:2010.04245](https://arxiv.org/abs/2010.04245) | L2-normalize q/k + 可学习 scale | 稳定 logits |
| ViT-22B（2023）[arXiv:2302.05442](https://arxiv.org/abs/2302.05442) | QK-norm 限制 logit 增长 | 与 sink 范数问题相关 |

### 4.4 长度鲁棒性、长音频聚合与误唤醒

* **长度条件归一化**：AS-norm / condition-matched cohort / QMF（TASLP 2013）[DOI](https://doi.org/10.1109/tasl.2013.2279332) /
  Trial-Based Calibration（Odyssey 2014）[DOI](https://doi.org/10.21437/odyssey.2014-4) /
  长度-噪声感知训练（Interspeech 2020）[arXiv:2008.12218](https://arxiv.org/abs/2008.12218)：
  用与测试条件（时长/SNR）匹配的背景 cohort 做 per-trial 归一化。
* **池化随长度漂移**：Adaptive pooling（TASLP 2018）[arXiv:1804.10070](https://arxiv.org/abs/1804.10070)：
  softmax pooling 随 L 增大趋向 mean pooling；max vs noisy-OR（ICASSP 2018）[arXiv:1804.01146](https://arxiv.org/abs/1804.01146)：
  noisy-OR 会随实例数饱和 → 我们的 softmin 也有 log-C 项，query 长度会带来 τ·log C 的漂移。
* **滑窗聚合**：max-pooling-loss LSTM（SLT 2016）[arXiv:1705.02411](https://arxiv.org/abs/1705.02411)
  （30 帧平滑 + 40 帧 lockout）；端到端流式 KWS（2018）[arXiv:1812.02802](https://arxiv.org/abs/1812.02802)
  （100 帧平滑，0.1 FA/h）；Howl（NLP-OSS 2020）[ACL](https://aclanthology.org/2020.nlposs-1.9.pdf)
  （5 FA/h @ 16% FRR，FA/hour ROC 选点）。
* **持续/一致性门控**：Cross-layer Discrimination Consistency（ICASSP 2025）
  [arXiv:2412.12635](https://arxiv.org/abs/2412.12635)（0.05 FA/h 下相对 miss −46.3%）。
* **多分辨率集成**：Multi-Scale Convolution（Interspeech 2020）[ISCA](https://www.isca-archive.org/interspeech_2020/yang20d_interspeech.html)；
  两级多分辨率集成（2023）[arXiv:2310.11379](https://arxiv.org/abs/2310.11379)。
* **长度抖动**：Length/noise-aware training（Interspeech 2020）[arXiv:2008.12218](https://arxiv.org/abs/2008.12218)。
* **注意力熵稀释**：Information Entropy Invariance（2025）[arXiv:2501.08570](https://arxiv.org/abs/2501.08570)；
  Preventing Attention Entropy Collapse（ICML 2023）[arXiv:2303.06296](https://arxiv.org/abs/2303.06296)。
* 现状缺口：**没有论文报告“phoneme-query 验证器”的 FA/hour 随窗口长度的曲线**；
  sink/register 也没有任何 KWS 的先例（arXiv 元数据搜索 0 命中）。我们的 1 s vs 3 s 数据本身就是空白点。

---

## 4.5 第 0 步实测结果（2026-10-03）

**协议**：脚本 scripts/probe_qbyt_v41_offline.py + scripts/analyze_v41_offline.py，
v4.1（v41-musan-zhen3m-50k / step 47,500）在 LibriPhrase hard split 全量 **270,684 对**上做 fp16 推理，
导出 pooled logit / 每位置 logits / sink 隐状态 / 长度；sink 读出与组合权重在**一半数据上拟合、另一半上评估**。
基线完全复现日志数字：AUC 0.935007 / EER 0.13106 / TPR@1% 0.18323 / TPR@0.1% 0.019935 / pAUC 0.54320
（日志：0.935007 / 0.131075 / 0.183210 / 0.0199347 / 0.543203）。

### （1）温度扫描：免费 +0.0019 AUC / +0.028 TPR@1%FPR

| T | AUC | EER | TPR@1%FPR | TPR@0.1%FPR | pAUC(≤1%) |
|---|---|---|---|---|---|
| **0.05** | **0.93695** | 0.12944 | **0.21090** | 0.02061 | **0.55380** |
| 0.10 | 0.93682 | 0.12944 | 0.20937 | 0.02056 | 0.55293 |
| 0.20 | 0.93658 | 0.12927 | 0.20603 | 0.01998 | 0.55113 |
| 0.50 | 0.93592 | 0.13003 | 0.19440 | 0.02149 | 0.54657 |
| 1.00（现状） | 0.93501 | 0.13106 | 0.18323 | 0.01993 | 0.54320 |
| 4.00 | 0.93283 | 0.13389 | 0.17038 | 0.01866 | 0.53932 |

**T=0.05 相对现状：AUC +0.00194、EER −0.00162、TPR@1%FPR +0.0277、pAUC +0.0106，零训练成本。**
（趋势在 T→0 方向仍在改善，说明 softmin 偏好接近 min 的池化；重训时可把 T 做成可学习或直接扫更小值。）

### （1b）top-k 与 log-C 修正：没有额外收益，log-C 修正有害

| 池化 | AUC | EER | TPR@1%FPR | TPR@0.1%FPR | pAUC |
|---|---|---|---|---|---|
| k=1（硬 min） | 0.93708 | 0.12928 | 0.21269 | 0.02245 | 0.55458 |
| k=2 平均 | 0.93710 | 0.12937 | 0.21374 | 0.02191 | 0.55431 |
| k=3 平均 | 0.93663 | 0.13024 | 0.21042 | 0.02239 | 0.55343 |
| 全部平均（v3 风格） | 0.93136 | 0.13614 | 0.16274 | 0.01682 | 0.53771 |
| softmin T=0.05 | 0.93695 | 0.12944 | 0.21090 | 0.02061 | 0.55380 |
| softmin T=0.05 + T·logC | 0.93680 | 0.12938 | 0.20807 | 0.02028 | 0.55297 |
| softmin T=1.0 + T·logC | 0.92981 | 0.13531 | 0.13858 | 0.01459 | 0.53153 |

* k=1 / k=2 / softmin T=0.05 三者在噪声底内等价（AUC 0.937 一档），**top-k 没有超过“接近 min 的 softmin”**。
* 去掉 softmin 的 log-C 归一化（加 T·logC）在所有温度下都变差，T=1.0 时 AUC 掉到 0.92981。
* 结论：S10 的这两个变体**放弃**；池化就用“温度约 0.05–0.1 的 softmin”。

### （1c）文本位置零训练替换 sinusoidal：崩了（重要更正）

| 文本位置 | AUC | EER | TPR@1%FPR | TPR@0.1%FPR | pAUC |
|---|---|---|---|---|---|
| learned（现状） | 0.93501 | 0.13106 | 0.18323 | 0.019935 | 0.54320 |
| 零训练换 sinusoidal | **0.70264** | 0.35532 | 0.03083 | 0.004300 | 0.50605 |

* 在 sinusoidal 版本上，温度扫描与 sink 探针同样大幅变差（sink-only AUC 0.80365 vs 0.93595）。
* 结论：learned 表“看起来是随机码”，但它是整个头适配过的**固定随机码**，**推理期不能换**；
  “是否改用 sinusoidal”只能靠重训回答，而且不能用 2k 步热身消融判断（会先撞上分布冲击）。
* 行动：D6 默认**保留 learned**；S7 降级为可选，需要 10k 步从头单独验证。

### （2）sink 读出：sink-only 就顶得上整个 pooled 头；组合后全面超过 v2

| 设置 | AUC | EER | TPR@1%FPR | TPR@0.1%FPR | pAUC |
|---|---|---|---|---|---|
| sink 线性探针（单独） | 0.93595 | — | — | — | — |
| pooled-only（T=1，eval 半集） | 0.93589 | 0.13087 | 0.19686 | 0.02375 | 0.54670 |
| pooled+sink（T=1，eval 半集） | 0.93962 | 0.12889 | 0.22543 | 0.03885 | 0.56513 |
| **pooled(T=0.05)+sink（eval 半集）** | **0.94057** | **0.12723** | **0.23621** | **0.04282** | **0.56800** |
| v2 参考（日志） | 0.93765 | 0.12864 | 0.20207 | 0.02347 | 0.55236 |

* **sink 状态单独一个线性探针的 AUC（0.93595）几乎等于整个 pooled readout（0.93589）**——
  说明模型其实已经把全部判别信息写在 sink 里了，只是从来不读它。
* 把 sink 读出与 pooled 组合（2 特征 logistic，在一半数据拟合）：**AUC 0.94057 > v2 0.93765（+0.0029，
  超过 ±0.002 噪声底），EER、TPR@1%FPR、TPR@0.1%FPR、pAUC 全部超过 v2**，其中
  TPR@1%FPR +0.0341、TPR@0.1%FPR +0.0194（相对 +82%）。
* 另一种实现（null 混合：把 sink_logit 作为 softmin 的额外候选，T=0.4）：
  AUC 0.93756 / EER 0.12870 / TPR@1% 0.21768 / TPR@0.1% 0.03871 / pAUC 0.55886 ——
  AUC/EER 与 v2 持平，尾部大幅超过 v2。两种实现都可行；重训时建议都 A/B。

### （3）长度分析：hard split 上不是杠杆，但解释了 FA 的长度敏感性

* hard-split 的音频长度几乎全在 **10–20 帧（0.4–0.8 s）**，根本没有触及 relative bias 的 2.0 s 钳制桶；
  所以“解钳”不可能改变 hard-split AUC（与 §3 的证伪一致）。
* 按音频长度分箱 z-norm：AUC 0.93589 → 0.93671（+0.0008，噪声级）；按文本长度分箱：无增益。
* 文本长度对难度影响很大：≤3 token 的 AUC 0.9773，>9 token 只有 0.9071；分数与长度正相关
  （正样本 Spearman 0.227）。
* 结论：长度条件归一化（S11）**不是 hard-split 指标杠杆**，保留为部署/FA 的阈值稳定手段。

### （4）滑窗平滑 / N-of-M 去尖峰：1 s 压力网格的孤立误触发几乎可全消

MUSAN 1 s/1 s 网格（156,929 窗 / 43.6 h），事件已折叠连续 accept：

| 阈值 | 原始事件/24h | 3 窗平滑后/24h | 2-of-2 持续后/24h |
|---|---|---|---|
| 0.5 | 28.6 | **0.0** | 0.6 |
| 0.3 | 66.1 | 3.3 | 0.6 |
| 0.2 | 103.0 | 16.5 | 0.6 |
| 0.1 | 182.8 | 69.9 | 4.4 |

3 s 网格本来就干净（0.5 阈值 0 事件），平滑后仍为 0。
**注意**：这会引入 1 个窗口的延迟、可能损失部分短关键词的召回；上线前必须在 hard split / 两级事件率上核对 TPR。

### （5）第 0 步结论

1. **两条零成本改动就能让 v4.1 全面超过 v2：softmin 温度降到约 0.05–0.1 + 读出 sink 状态。**
   离线评估（半拟合/半评估）AUC 0.94057 / EER 0.12723 / TPR@1% 0.23621 / TPR@0.1% 0.04282 / pAUC 0.56800。
2. sink 单独就有 0.93595 AUC：它本来就在编码判别信息，读出来是“捡回来的分”。
3. 长度归一化与去尖峰不改变 hard-split 指标，但能显著改善部署误唤醒（1 s 网格 28.6 → 0 事件/24h）。
4. 下一步：把 S3（sink 读出，两种实现都试）+ S8（温度，初始化为 0.1 并让它可以学）做成重训 run D6；
   跑完用同一套离线探针复算，再决定是否加 S4/S7。

## 5. 小改动清单（全部保留 sink token；按“证据强度 ÷ 成本”排序）

### S3 sink 状态读出（约 129 参数）★最高优先

**机制**：sink_fc = Linear(128 → 1) 作用在 sink 的最终隐状态；utterance logit 用**输入相关**的方式合并，
两种实现（可都试）：
(a) 加法：utt = pooled_text_logit + α · sink_logit，α 为 Parameter(0)；
(b) 软上限（null 混合）：把 sink_logit 作为一个额外元素并入 softmin：
pooled = −T·[logsumexp(文本 logits ∪ sink_logit) − log(n+1)]。

**依据**：sink 状态目前完全没被读出；sink→text 正偏置说明它已经在读 query；
register 读出在异常检测上 FPR −2–3%（ICASSP 2025）；PMA/Memory Transformer 的 seed 本就是要读出；
Dummy ProtoNets 的 dummy prototype、energy-based OOD 都指向“显式 null 假设”。
注意：常数 null 不改排序，所以必须输入相关。

**实现**：qbyt/pooling.py QbyT._forward_impl；packed 布局下 sink 在 index = text_lengths（或 text_width 前一个位置，
按当前 repack 规则确认）；α 初始化 0 → step 0 与 v4.1 逐位一致。

**预期**：尾部 TPR@1%FPR / FA‑24h 改善，AUC 约持平；风险：sink_logit 与 text logit 共线。

**零训练证伪（强烈建议先做）**：用 hook 取现有 v4.1 的 sink 状态，在 dev 上单独评估 sink_logit 的 AUC，
以及把 sink_logit 以 (a)/(b) 方式并入后的 TPR@1%FPR。若 sink_logit 单独 AUC < 0.6 或并入后尾部无变化，放弃。

### S4 sink 身份补全（2–3 行）

* ModalityEmbedding 增加 sink 行（目前 sink 没有任何模态向量）；
* repacked_modality_and_index 给 sink 独立 index（目前与第 0 帧同为 0）；
* 可选：用投影后音频帧的均值初始化 sink（StreamingLLM：Zero-Sink 几乎不可用），并加范数约束。
**零训练证伪**：在现有 checkpoint 上 hook 出 sink_state.norm / audio_state.norm、以及 final_pos_fc(sink_state)
在正样本 vs 背景上的差异；若完全无区分度，说明 sink 表征本身没信息，优先补初始化/监督。

### S8b 训练/打分温度解耦（零训练，A 轮新增结论）

**实测**：离线只改推理温度（T=0.1，不训练）在原始 v4.1 上 = AUC 0.93682 / TPR@1% 0.2094；
但把 T=0.1 写进训练配置（A3/A4）只有 0.93455 / 0.1799——**训练时改温度会毁掉推理期的收益**，
而且 TPR@0.1%FPR 反而变差（0.0181 → 0.0148）。
**做法**：训练保持 T=1（权重不动），在打分/校验路径上用独立的“打分温度”（0.05–0.1）重新池化。
**实现**：在 qbyt_readout spec 里把“训练温度”和“打分温度”分成两个字段（默认相等，兼容旧 checkpoint）；
或先在评测脚本里做离线重池化验证，再决定是否落进 spec。
**判据**：打分温度 0.05–0.1 相对 T=1 在 hard split 上 AUC +≥0.0015 且 TPR@1%FPR +≥0.02（离线已验证），
并且 MUSAN 3 s 最高分不升。

### S8 可学习温度：训练侧不做（A 轮已证否，2026-10-04 修订）

**A 轮实测（A3 vs A4）**：固定 T=0.1 与可学习 T（初值 0.1）在 2k 步后**完全等价**——
AUC 0.93455 vs 0.93453、TPR@1%FPR 0.17989 vs 0.18014；可学习标量只移动了 +0.0036（log 空间）。
而且离线扫描已覆盖全部敏感区间（0.05–0.2 等价），训练期改温度还会毁掉推理期收益（见 S8b）。

**结论**：**不在训练里学温度**。替代做法是 S8b（训练 T=1，打分温度后验拟合/离线扫描）；
若将来希望“自动选温度”，把它做成 dev 上拟合的**后验打分参数**（或复用仓库已有的 affine 校准器），
而不是训练参数。代码里的 temperature_learnable 开关保留（默认关闭、有 parity 测试），但不排 GPU 实验。

### S11 长度条件分数归一化（0 参数）★零成本

**机制**：per-trial z-norm：s' = (s − μ_bin)/σ_bin，bin 由（音频长度，query 长度）决定；
μ/σ 用背景 cohort 统计（可离线从现有 MUSAN/训练背景分数算）。
**依据**：本机 1 s vs 3 s 分布差异巨大（§1.2）；AS-norm / cohort / QMF / TBC 都支持条件匹配归一化。
**零训练证伪**：直接用现有两套 results.jsonl 做离线分析：把 3 s 分数按长度分箱 z-norm 后，
看是否能用一个阈值同时满足 1 s/3 s 的 FA 目标；若分箱后分布仍重叠严重，收益有限。

### S12 滑窗平滑 + N-of-M 持续门控（0 训练）

**机制**：后处理，不改 head：对相邻窗口分数做 mean/median 平滑（3–5 窗），要求 k≥2 连续超阈；
可选 lockout。**依据**：SLT 2016 / 流式 KWS / Howl / CDC 层间一致性；本机 1 s 网格 53 个 accept
是孤立尖峰（连续 accept 折叠后事件数几乎不变）。
**零训练证伪**：在现有 1 s 网格分数上离线扫 k=1/2/3，比较固定 TPR 下的 FA/24h。

### S9 成本修复（工程，1–2 天）

优先级：(1) 复刻 T5 的 batch-independent [1,H,Lq,Lk]、只算音频 tile、缓存 bucket 索引；
(2) FlexAttention score_mod 把 bucket+pair_table 编进 fused kernel；
(3) pair_table（rank ≤3）折进 q/k 的额外通道；
(4) 只在第 0 层加 bias（Kazemnejad 的 NoPE 结论支持做这个消融）。
**注意**：必须保持 phone_matchor.layers.N.self_attn 的 state_dict 名字（LoRA 目标和 attention 诊断依赖它）。
**判据**：新旧实现逐样本 logit 差 < 1e-4，step 时间从 0.44–0.60 回到 0.20–0.25 s。

### S7 text position：默认保留 learned；换 sinusoidal 必须重训验证（修订）

**实测（零训练替换）**：AUC 0.93501 → **0.70264**（§4.5-1c）。learned 是一份“固定随机码”，
整个头围着它适配，推理期换码等于换输入分布。
**因此**：
* D6 默认**保留 learned**（16K 参数只占头的约 4%，不是成本瓶颈）；
* “换 sinusoidal”只作为可选项，且**不能**用 2k 步热身消融判断（会被分布冲击误判），
  需要 10k 步从头跑，或热身续训给位置表更长适应期 + 更大 LR；
* 将来若要从零设计新头，可以直接用 sinusoidal（省 16K、可长度外推），但必须与新头一起训练。

### S2 窗口化 bias + S13 长度抖动训练（重训，二选一或一起）

* S2：相对 bias 只在 |Δ| ≤ W（如 32 帧）内有效，超出共享一个可学习标量（或 0）。
  **注意**：§3 证明“远端置零”会抬高中间尾部，所以 S2 也必须**重训**并同时做 S13，不能推理期改。
* S13：训练样本做长度抖动（把候选拼到 1–4 s、随机 crop/extend），加 |s(x) − s(crop(x))| 一致性惩罚，
  让 3 s 进入训练分布。依据：Length/noise-aware training（Interspeech 2020）。
**判据**：3 s 网格 p99.99 与 ≥0.1 计数下降，同时 hard-split TPR@1%FPR 不降。

### S5 多 sink / 每头 sink bias / S6 sink 监督（重训，中等）

* S5：sink_token 从 [d] 扩到 [k, d]（k=2–4），各自独立 index，加 cos² 去相关惩罚；
  或加每头可学习 sink key bias（When Sinks Emerge 的 key-bias 视角）。Darcet 用 4，StreamingLLM 说 2 个没用——
  矛盾，必须 A/B。
* S6：对负样本提高、对正样本降低 sink attention 质量；或 sink–audio 去相关的正则。
  依据：AVSR decorrelation；CDC 一致性。
**判据**：k=1/2/4 与正则 on/off，比 3 s p99.99/最差背景与 TPR@1%FPR，而不是只看 AUC。

### S10 其它一参数级候选

logit 裁剪 / LogitNorm（ICML 2022）、QK-norm（ViT-22B）、count-aware top-k softmin（用 top-k 均值替代 min，
消除 query 长度漂移）、per-length softmin 温度 + log-C 修正（子代理指出 pooled = min + τ·log C 当单点主导时）。
这些都可以在**现有 checkpoint 的 logits 上离线验证**（top-k、log-C 修正），先离线再决定是否重训。

---

## 5.6 D6 细化：快速消融 + 全量训练（2026-10-03 新增）

目标：把 S3（sink 读出）+ S8（温度）+ S4（sink 身份）落到代码里，先用**最快速度**摸清规律，
再跑全量训练，验收线是 **AUC > 0.9376（v2）且 TPR@1%FPR ≥ 0.2021**。
**S7（换 sinusoidal）已降级为可选项**：零训练替换会让 AUC 崩到 0.70264（§4.5-1c），
只能靠 10k 从头单独跑来判断，不进入快速消融。

### 已被离线分析钉死的事实（消融不必重复验证）

| 事实 | 数字 | 结论 |
|---|---|---|
| sink 单独线性探针 | AUC 0.93595（pooled 0.93589） | sink 里有全部信息，必须读出 |
| 复用 final_pos_fc 读 sink | 单独 0.92033；与 pooled 组合 0.93586（无增益，系数为负） | **必须给 sink 自己的线性层**（约 129 参数），不能省 |
| 温度 | T=0.05 最好；0.05/0.1/0.2 差异在噪声内 | 初值取 0.1，做成可学习 |
| top-k / 去掉 log-C 归一化 | 无增益 / 有害（T=1 时 AUC 0.92981） | 池化只用“接近 min 的 softmin” |
| additive vs null-mixture | 半拟合/半评估：0.94057 vs 0.93756（后者尾部更好） | 两种读出都要 A/B |
| 文本位置 learned → sinusoidal | 零训练替换 AUC **0.93501 → 0.70264** | **推理期不能换码**；默认保留 learned，S7 降级为需 10k 从头验证的可选项 |
| 组合上限（离线） | AUC 0.94057 / EER 0.12723 / TPR@1% 0.23621 / TPR@0.1% 0.04282 | 全量训练的目标区间是 AUC 0.939–0.941 |

### D6.0 代码改动（约 0.5–1 天）

1. **qbyt/pooling.py** 新增只影响 v4 家族的读出开关（默认值全部等于现状）：
   * sink_readout: none（默认）/ additive / mixture
     - additive：sink_fc = Linear(128, 1)；utt = pooled_text_logit + alpha * sink_logit；alpha = Parameter(0)；
     - mixture：把 sink_logit 当作 softmin 的**额外候选**（与文本位置一起做 masked 归一化），
       即 pooled = softmin(text positions 并集 sink_logit)；
   * sink_modality: bool（给 sink 加模态向量）、sink_index: bool（relative bias 里给 sink 独立 index）；
   * temperature_learnable: bool（log_temperature = Parameter(log(0.1))，clamp 到 [0.02, 4] 倍初值）。
   新增参数总计：sink_fc 129 + alpha 1 + 温度 1。
2. **加载器**：新增 stage2.init_allow_readout_mismatch（默认 false）。只作用于
   stage2.init_checkpoint 的**权重热身**路径：spec 不一致时打大字警告并继续 strict=False；
   推理/评测路径仍然严格匹配 spec，保持“checkpoint 自描述”的纪律。
3. **测试**（补进 tests/）：
   * 新开关全关时，forward 与 v4.1 逐位一致（parity 1e-6）；
   * additive 且 alpha=0 时，logits 与基线逐位一致；
   * sink 读出后 batch 不变性仍成立（不同 padding 下分数漂移为 0）；
   * mixture 的 mask/长度处理；温度 clamp 边界。
4. **热身基线确认**：用 v4.1 47.5k checkpoint 在同一验证流程下复算一次，确认 0.935007
   （防止“改了代码把基线弄坏”）。

### D6.1 第一轮消融：单因子，热身续训（7 组 × 2k 步，约 1–1.5 h 墙钟）

**协议**：全部从 v4.1 的 47.5k 权重**热身续训**（不是从零），这样 2k 步就能分辨 0.001–0.002 级差异；
LR 5e-5（warmup 100，cosine 到 0），batch 128，固定 seed，2,000 步，结束时在全 hard split 验证（约 2 min）；
用 stage2.init_allow_readout_mismatch=true 加载。

| 组 | 改动 | 用途 |
|---|---|---|
| **A0** | 什么都不改，续训 2k 步 | 对照：测“继续训练本身的漂移”，作为所有 Δ 的基准 |
| A1 | sink_readout=additive | S3 的一种实现 |
| A2 | sink_readout=mixture | S3 的另一种实现 |
| A3 | 固定 T=0.1 | S8 的“不学、直接调”版本 |
| A4 | temperature_learnable（init 0.1） | S8 的可学习版本 |
| A5 | sink_modality + sink_index | S4 |
| A6 | text_position=sinusoidal **（不能用 2k 热身判断；改为 10k 从头单跑，或从快消融中移除）** | S7（可选） |

命令示意（A1）：

    CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/train_stage2_qbyt.py
      +experiment=icefall_zipformer_stage2_eps_softmin_v41_musan_cached_zhen3m_50k
      stage2.qbyt_readout.sink_readout=additive
      stage2.init_allow_readout_mismatch=true
      stage2.init_checkpoint=<v4.1 stage2_step047500.pt>
      stage2.learning_rate=5e-5 stage2.warmup_steps=100
      stage2.max_steps=2000 stage2.total_scheduler_steps=2000
      stage2.validation.val_check_interval=2000 stage2.val_check_interval=2000
      stage2.run_name=A1-sink-additive
      stage2.checkpoint_dir=... stage2.log_dir=...

**判据**：与 A0 比，ΔAUC > 0.0015 或 ΔTPR@1%FPR > 0.01 才算赢；落在噪声带内的组若想保留，
用第二个 seed 复核。**并发**：3 组共置在同一张 V100（仓库实测共置只让单个 run 变慢，但总吞吐更高）。

### D6.1 A 轮结果（已完成 2026-10-04，热身 2k 步 / LR 5e-5）

| 组 | AUC | ΔAUC vs A0 | EER | TPR@1%FPR | ΔTPR@1% | TPR@0.1%FPR | pAUC |
|---|---|---|---|---|---|---|---|
| A0 对照（只续训） | 0.93422 | — | 0.13263 | 0.17442 | — | 0.018117 | 0.53994 |
| A1 sink additive | 0.93419 | −0.00003 | 0.13234 | 0.17629 | +0.0019 | 0.017984 | 0.54102 |
| A2 sink mixture | 0.92956 | **−0.00466** | 0.13374 | 0.13362 | **−0.0408** | 0.014526 | 0.52953 |
| A3 温度固定 0.1 | 0.93455 | +0.00033 | 0.13258 | 0.17989 | +0.0055 | 0.014822 | 0.54523 |
| A4 温度可学习 | 0.93453 | +0.00031 | 0.13252 | 0.18014 | +0.0057 | 0.014792 | 0.54525 |
| A5 sink 身份 | 0.93395 | −0.00028 | 0.13264 | 0.17714 | +0.0027 | 0.015405 | 0.54175 |

参考：v4.1 原始（47.5k，T=1）= 0.93501；v2（50k）= 0.937647。

**三条结论：**

1. **续训本身会掉点**：A0 从 0.93501 → 0.93422（−0.0008）。所以任何消融都必须以 A0 为基准，Δ 才有意义。
2. **新参数确实没学到（A1/A4/A5）**：与 §5.6 的权重取证一致——A1 的 α=+0.003 几乎为零，A5 的 identity 向量范数 0.02。
   A1/A5 与 A0 的差异都在噪声内，属于“优化不足造成的假阴性”，不能据此否定想法。
3. **A2（mixture）明显有害（−0.0047 AUC / −0.041 TPR@1%）**：随机初始化的 sink_fc 让一个噪声候选长期待在 softmin 池里，
   污染了池化分数。→ **mixture 必须配零初始化**（B2 已按此修）。
4. **A4（可学习温度）与 A3（固定 0.1）在噪声内完全等价**（0.93453 vs 0.93455，标量只动了 +0.0036）
   → **训练里学温度没有额外价值**，温度改用 S8b 的后验打分方式决定。

**最重要的新发现（关于温度）：**
* 离线“只改推理温度”（不训练）在原始 checkpoint 上给 +0.0018 AUC / +0.026 TPR@1%FPR（T=0.1）；
* 但 **训练时就把 T 设成 0.1（A3/A4）只比 A0 好 0.0003 AUC**，远没有兑现离线增益，而且 TPR@0.1%FPR 变差
  （0.0181 → 0.0148）。
* 结论：**训练与打分共用温度是把两件事绑死了**。正确做法是解耦——训练仍用 T=1，打分时用 T≈0.05–0.1 重新池化。
  这是一条**零训练**就能拿的 +0.0018 AUC / +0.026 TPR@1%FPR（详见 S8b）。

### D6.2 第二轮消融：已并入 B 轮（不再单独做）

原计划的四组组合消融已被重排后的 **B 轮**取代，理由：

* S7（换 sinusoidal）已证伪（零训练替换 AUC 崩到 0.70264），退出组合；
* α / T 的初值敏感性不再需要单独测：α 已改成“sink_fc 零初始化 + α=1”的标准残差参数化；
  温度的敏感区间已由离线扫描覆盖（0.05–0.2 等价）；
* “sink 读出 + 温度”的组合**就是 B3**（additive-zero + T=0.1），已经在跑；
* 唯一可能残留的补充臂：若 B1/B2 里 **mixture 胜出**，则补一组 “mixture-zero + T=0.1”（约 50 min）；
  若 B 轮显示 sink 读出有效但想收口 S4，可再加一组 “胜者 + sink_identity”。

**B 轮实测（2026-10-04 启动，4 组 × 2k 步 / LR 2e-4）**：B0 对照、B1 additive-zero、
B2 mixture-zero、B3 additive-zero + T=0.1。

**判据**：B 轮胜者（相对 B0 控制）达到 AUC ≥ 0.939 且 TPR@1%FPR ≥ 0.22 → 直接进 D6.3；
否则说明“热身 2k 步”不足以让新头收敛，需要一次 10k 步从头复核后再决定。

### D6.3 全量训练（确认，8–9 h；若先做 D5 成本修复约 3 h）

| 组 | 配置 | 验收 |
|---|---|---|
| **C1（必做）** | 胜者组合，zh-en-3M 冻结 encoder，**从零 50k 步**（与 v2 同预算、同数据、同 seed） | AUC ≥ 0.938；TPR@1% ≥ 0.2021；TPR@0.1% ≥ 0.0235；pAUC ≥ 0.5524；MUSAN 3 s 最高分 ≤ 0.41、1 s FA/24h ≤ 29.1 |
| C2（可选） | 同配置换 paperstage1（作者 1460h）encoder，50k | 检查 encoder 泛化；目标同上 |
| C3（可选） | 在 C1 基础上加 S13 长度抖动 + S2 窗口 bias（即 D7） | 3 s 尾部下降且 hard 指标不掉 |

C1 完成后：用 scripts/probe_qbyt_v41_offline.py + scripts/analyze_v41_offline.py 复算同一套离线指标
（温度扫描 / sink 探针 / 长度分析），确认“模型自己学到的”与“离线后验拟合的”一致；
再跑 eval_musan_fa.py 的 3 s 与 1 s 网格。

### 决策树（避免无效投入）

* A1/A2 相对 A0 的增益 < 0.0015 AUC → sink 读出在**真实训练**里不成立：
  只保留温度（A3/A4，白捡 +0.0019 AUC / +0.028 TPR@1%FPR），
  把主攻方向转回训练配方（membership 目标 + 100:1 hard negative，见前一份架构调研 §4-P0）。
* A 组有效但 B1 达不到 0.939 → 加一次 **10k 步从头跑**（约 50 min）复核，排除“热身续训太乐观”。
* C1 达标 → 更新 Experiment Log 与 wiki，把 v4.1+sink 作为新的默认候选；
  若只是尾部更好但 AUC 没到 0.938，则定位为“低误报优先”的部署模型，继续 C3。

### 统计与避坑

* 噪声底：单点 ±0.002 AUC / ±0.005 EER；消融必须**配对**（同一热身权重、同一 seed、同一验证集）。
* 验证只用**全 hard split**（2 min）；不要用 20k 子集下结论（AUC 标准误约 0.003，会淹没 0.001–0.002 的效应）。
* 消融期间**不要改动 relative bias / 成本相关实现**，否则混淆因素；成本修复（D5）留在 D6.3 之后或作为并行工程。
* 每组都必须导出 stage2 .pt（带 spec 与新字段），否则事后无法用离线探针复算。
* 如果时间只够一件事：**A1 + A4 + C1**（sink additive 读出 + 可学习温度 + 从零 50k）。
  离线预测 AUC 0.9406、TPR@1%FPR 0.236，是超过 v2 的最短路径。

> **后续工作索引（2026-10-05）**：v4.2 的第二段/写回、5 个 encoder 成绩表、TTS 唤醒评测、
> 论文原仓复现（R1/R2）与批大小基准，全部归档在 `docs/qbyt-v4.2-design.md`（§7.3-§7.5、§9）。

## 5.7 C1/C2 全量 50k 结果（2026-10-04）

### C1 = v4.1 配方 + additive-zero sink 读出，从零 50k（已完成）

| 指标 | C1（训练 T=1，部署分） | v2 | Δ |
|---|---|---|---|
| AUC | 0.93785 | 0.93765 | +0.0002 |
| EER | 0.12730 | 0.12864 | −0.0013 |
| TPR@1%FPR | 0.20385 | 0.20207 | +0.0018 |
| TPR@0.1%FPR | 0.028919 | 0.023467 | +0.0055 |
| pAUC(≤1%) | 0.55118 | 0.55236 | −0.0012 |

**首个在 AUC/EER/两个尾部指标上都不输 v2 的 v4.1 变体**（pAUC 略低）。
sink 分支确实被训练了：α 从 1.0 学到 **0.640**，sink_fc 权重范数 0 → **0.368**。

### 重要发现：训练出来的 sink 分支是轻微负收益，但 sink 状态本身有信息

用 C1 最终 checkpoint 做离线探针（半拟合 / 半评估，保守协议）：

| 打分方式 | AUC | EER | TPR@1%FPR | TPR@0.1%FPR | pAUC |
|---|---|---|---|---|---|
| 模型自带分数（text pooling + α·sink，T=1） | 0.93785 | 0.12730 | 0.20385 | 0.028919 | 0.55118 |
| 只用 text pooling，T=1（忽略 sink 分支） | 0.93937 | 0.12717 | 0.22720 | 0.019654 | 0.55952 |
| text pooling，T=0.1（S8b 打分温度） | **0.93978** | 0.12622 | **0.22415** | 0.023208 | 0.55841 |
| text pooling，T=0.3 | **0.93986** | 0.12603 | 0.22579 | 0.018531 | 0.55854 |
| text + **后验拟合** sink_logit（logistic，eval 半集） | **0.94168** | 0.12485 | 0.23661 | 0.04352 | 0.56619 |
| text ∪ 后验拟合 sink_logit 的 null 混合，T=1（eval 半集） | **0.94221** | 0.12393 | 0.25088 | 0.041966 | 0.56887 |

结论：
1. **联合训练的 sink 分支在 C1 里是轻微负收益**（−0.0015 AUC）：优化器主动把 α 从 1.0 降到 0.64，说明 BCE 目标并不奖励它；
2. **但 sink 状态携带 +0.003 的排序信息**：在 dev 上后验拟合一个线性头后，组合分数到 **0.9417–0.9422**（比 v2 高 +0.004–0.0046，尾部 +0.035–0.049 TPR@1%FPR）。
3. 因此正确的落地方式是**后验拟合 sink 头**（像校准器一样，在 dev 上拟合、固定后部署），而不是指望它跟主目标一起被端到端训出来。

### S8b（打分温度）在 C1 上得到确认

C1 只用 text pooling、把打分温度从 1.0 降到 0.1–0.3：AUC 0.9394 → **0.9398**、EER 0.1272 → **0.1261**、
TPR@1%FPR 0.2272 → **0.2242–0.2258**、pAUC 0.5595 → 0.5584。
相对 v2：AUC **+0.0021**、EER **−0.0024**、TPR@1%FPR **+0.022**、pAUC **+0.006**。
零训练、只需在打分路径加一个温度参数。

### C2（sinusoidal 文本位置）进行中，明显落后

图（outputs/v41_final/figures/）：c1_curves.png（训练四指标）、c1_vs_c2.png（C1 vs C2）、
ablations_ab.png（A/B 消融柱状）、c1_temperature.png（打分温度扫描 + 后验 sink 头）。
生成脚本 scripts/plot_v41_results.py。

| step | C1 AUC | C2 AUC | Δ |
|---|---|---|---|
| 9,999 | 0.89248 | 0.87882 | −0.0137 |
| 12,499 | 0.90162 | 0.88818 | −0.0134 |
| 17,499 | 0.90939 | 0.90236 | −0.0070 |
| 19,999 | 0.91715 | 0.90777 | −0.0094 |
| 22,499 | 0.91889 | 0.90665 | −0.0122 |
| 24,999 | 0.92172 | 0.91399 | −0.0077 |
| 49,999（终值） | 0.93785 | **0.92973** | **−0.0081** |

C2 全程低于 C1（−0.0065 到 −0.0134 波动，终值 −0.0081），EER 也一直更高（0.13612 vs 0.12730），
未触发 0.02 熔断阈值。终值探针：部署 0.92973；打分温度 T=0.05 也仅 0.93199；拟合 sink 头 0.93331——
三条打分路径全部输给 C1（0.93785 / 0.93985 / 0.94168）→
**learned 位置表优于 sinusoidal，"换回 sinusoidal" 在这套配方下不成立**（与 §4.5-1c 的零训练替换结论方向一致）。

## 5.8 v4.2 的损失设计（草案，2026-10-04）

> v4.2 = v4.1 + additive sink 读出（zero_init + α 可学习）+ 损失重组。
> 一切先经 2k 步热身消融验证，胜者进全量 50k（排在 C2 之后，称为 C1'）。

### 决定 1：旧“尾部难样本损失”（CVaR）不保留，改由新排序损失接管

理由：
* 实测是反面教材：v3/v4/v4.1 开启它后 Brier/ECE 变好，但 **TPR@1%FPR 反而下降**（0.202 → 0.157/0.170）；
* 它优化的是“平均对错”的加权，不是“固定 FPR 下的召回”；
* 它的岗位由新的 pairwise 排序损失接管（作用对象换到部署分上）。
**兜底**：保留一个 2k 步消融臂（E1：仅把它关掉），防止“C1 的成绩其实靠它”这种意外。

**E1 已执行（2026-10-05）**：C1 权重续训 2k 步（LR 2e-4）关闭 CVaR，配对对照 S0（CVaR 开，同协议）：

| 臂 | AUC | EER | TPR@1%FPR | TPR@0.1%FPR | pAUC |
|---|---|---|---|---|---|
| C1 基座 | 0.93785 | 0.12730 | 0.20385 | 0.028919 | 0.55118 |
| S0（CVaR 开，2k 续训） | 0.93645 | 0.12811 | 0.19802 | 0.029340 | 0.55142 |
| **E1（CVaR 关，2k 续训）** | **0.93802** | **0.12767** | **0.21529** | **0.033264** | **0.55674** |

Δ(E1−S0)：AUC **+0.00157**、TPR@1%FPR **+0.01727**、TPR@0.1%FPR +0.00392、pAUC +0.00532。

**后续：C1-notail-50k 从零实测（2026-10-05）推翻上面的乐观判断**：

| 运行 | AUC | EER | TPR@1%FPR | TPR@0.1%FPR |
|---|---|---|---|---|
| C1（有 CVaR，从零） | 0.93785 | 0.12730 | 0.20385 | 0.028919 |
| C1-notail（无 CVaR，从零） | 0.93450 | 0.13350 | 0.18775 | 0.018191 |
| C1-notail + 第二段 + 写回 | 0.93807 | 0.13080 | 0.22131 | 0.033197 |
| C1 + 第二段 + 写回（v4.2 正式版） | **0.94258** | 0.12360 | 0.24595 | 0.039921 |

**结论修正**：E1 的 2k 热身增益是短程效应（已收敛模型上关 CVaR 立即松绑尾部）；
但从零训练里 CVaR 是有效正则——去掉后整条曲线低 −0.0034。**主训练配方保留 CVaR，C1 维持 v4.2 主训练基座**；
无 CVaR 版即使补上 sink 头（0.93807）也追不上正式版（0.94258）。

### 决定 2：sink 分加专属目标；pairwise 排序加在最终分上

| 损失 | 公式 | 作用对象 | 目的 |
|---|---|---|---|
| sink 专属 BCE | BCE(sink_logit, 1 − label)，权重 λ_sink | **sink 分** | 给草稿格一份对口考纲：学会“这条不匹配”（解决 B2 的“无人管→纯噪声”） |
| pairwise 排序 | softplus(margin − (s_pos − s_neg))，只对批内分数最高的 α≈1% 负样本，权重 λ_rank | **最终分**（主分 + α·sink分） | 直接优化“固定 FPR 下的召回”，替代旧 CVaR |

分工：**sink 头 = 独立的无匹配探测器；最终分 = 负责与部署直接挂钩的尾部排序。**
（排序理论上对应 pAUC/AUC-margin 代理：arXiv:2203.01505、2012.03173；实现取批内 top-α 负样本即可，O(1) 内存。）

### 消融臂（2k 步 / LR 2e-4 / 每个 ~40 min，4 组并行）

| 臂 | 配置 | 回答的问题 |
|---|---|---|
| E0 | C1 配方（对照） | 基准 |
| E1 | E0 + 关掉 CVaR | 旧尾部损失到底有没有用 |
| E2 | E0 + λ_sink=0.5 | sink 专属目标是否生效 |
| E3 | E0 + λ_rank=0.5 | 排序损失是否提升尾部 |
| E4 | E2 + E3 合体（视前三个结果再加） | 组合是否相加 |

判据：相对 E0，ΔAUC > 0.0015 或 ΔTPR@1%FPR > 0.01 才算有效；尾部看 ΔTPR@0.1%FPR 与 ΔpAUC。
胜者组合 = v4.2；随后启动全量 50k（C1'），评测带 S8b 打分温度。

## 6. 建议的实验顺序（更新版）

| # | 实验 | 类型 | 成本 | 状态/判据 |
|---|---|---|---|---|
| D1 | 远距离桶置零 | 零训练 | 已完成 | **已证伪**：最差 0.409→0.386，但 ≥0.2 窗口 1→2、≥0.1 4→6 |
| D2 | sink_logit 离线评估（单独 AUC + 并入 readout 的尾部） | 零训练 | 已完成 | sink-only AUC 0.93595 ≈ pooled 0.93589；组合后 0.94057 / EER 0.12723 / TPR@1% 23.6% / TPR@0.1% 4.3%，**全面超 v2** → **S3 值得做** |
| D3 | 固定 T 扫描（T ∈ [0.05,4]）+ top-k / log-C 修正 | 零训练 | 已完成 | T=0.05 最佳（AUC 0.93695 / TPR@1% 21.1%）；top-k 无额外增益、log-C 修正有害 → **S8 做，S10 放弃** |
| D4 | 长度条件 z-norm + 持续门控离线评估（用现有 results.jsonl） | 零训练 | 部分完成 | hard-split 长度 z-norm 仅 +0.0008（噪声级，S11 不做指标杠杆）；MUSAN 去尖峰 FA 侧有效（1 s：28.6 → 0 事件/24h @0.5），**TPR 侧需两级事件评测后才能定 S12** |
| D5 | 成本修复 S9（先 profile 再改） | 工程 | 1–2 天 | parity <1e-4 且 <0.25 s/step |
| D6 | 重训：**S3+S8（+S4）**（sink 读出 + 温度 + 身份；S7 降级为可选，需 10k 从头）；**已细化为 D6.0–D6.3（见 §5.6）** | 训练 | 消融 2 h + 全量 8–9 h | **AUC 超过 v2 的 0.9376**，TPR@1%FPR ≥ 20.2%，3 s p99.99 不升 |
| D7 | 重训：S13 长度抖动 + S2 窗口 bias | 训练 | 3–8 h | 3 s 尾部下降且 hard AUC 不掉 |
| D8 | 重训：S5/S6（多 sink / sink 监督） | 训练 | 各 3 h | 只按尾部指标判 |
| D9 | D0 诊断：scripts/diagnose_qbyt_sink.py 跑 1/2/3 s 正负样本 | 诊断 | 数小时 | sink attention mass vs 长度/正负性 |

顺序：**D2/D3/D4 已完成（结论见 §4.5）**；下一步是 **D6 重训（S3+S8，S4/S7 一起带上）**，
D9 诊断可与 D6 并行，D5（成本修复）排在 D6 之后或并行，D7/D8 视 D6 结果再定。
D4 剩下的“去尖峰对 TPR 的影响”留到两级事件级评测时补（与本次重训无耦合）。

---

## 7. 主要参考文献

**sink / register / aggregator**
1. StreamingLLM. arXiv:2309.17453. https://arxiv.org/abs/2309.17453
2. When Attention Sink Emerges in LMs. arXiv:2410.10781. https://arxiv.org/abs/2410.10781
3. Massive Activations in LLMs. arXiv:2402.17762. https://arxiv.org/abs/2402.17762
4. Vision Transformers Need Registers. arXiv:2309.16588. https://arxiv.org/abs/2309.16588
5. Leveraging Registers in ViTs for Robust Adaptation. arXiv:2501.04784. https://arxiv.org/abs/2501.04784
6. Mamba-R: Vision Mamba ALSO Needs Registers. arXiv:2405.14858. https://arxiv.org/abs/2405.14858
7. Memory Transformer. arXiv:2006.11527. https://arxiv.org/abs/2006.11527
8. Set Transformer. arXiv:1810.00825. https://arxiv.org/abs/1810.00825
9. Perceiver. arXiv:2103.03206. https://arxiv.org/abs/2103.03206
10. AST. arXiv:2104.01778. https://arxiv.org/abs/2104.01778
11. Keyword Transformer. arXiv:2104.00769. https://arxiv.org/abs/2104.00769
12. AVSR sink decorrelation. arXiv:2510.22603. https://arxiv.org/abs/2510.22603
13. When Sinks Help or Hurt. arXiv:2604.03316. https://arxiv.org/abs/2604.03316
14. SinkProbe. arXiv:2609.08574. https://arxiv.org/pdf/2609.08574

**pooling / null / 校准**
15. ECAPA-TDNN. https://www.isca-archive.org/interspeech_2020/desplanques20_interspeech.html
16. Attentive Statistics Pooling. arXiv:1803.10963. https://arxiv.org/abs/1803.10963
17. MQMHA pooling + inter-topK. arXiv:2110.05042. https://arxiv.org/abs/2110.05042
18. Dummy Prototypical Networks. arXiv:2206.13691. https://arxiv.org/abs/2206.13691
19. OpenMax. https://openaccess.thecvf.com/content_cvpr_2016/html/Bendale_Towards_Open_Set_CVPR_2016_paper.html
20. ARPL / Reciprocal Points. arXiv:2103.00953. https://arxiv.org/abs/2103.00953
21. Energy-based OOD. arXiv:2010.03759. https://arxiv.org/abs/2010.03759
22. LogitNorm. arXiv:2205.09310. https://arxiv.org/abs/2205.09310
23. ReAct. arXiv:2111.12797. https://arxiv.org/abs/2111.12797
24. On Calibration of Modern NNs. arXiv:1706.04599. https://arxiv.org/abs/1706.04599
25. Calibration for speaker verification. arXiv:2203.15106. https://arxiv.org/abs/2203.15106
26. MagnetO calibration. arXiv:2102.01760. https://arxiv.org/abs/2102.01760
27. AS-norm. https://www.isca-archive.org/interspeech_2017/matejka17_interspeech.html
28. Condition-matched cohorts. https://www.isca-archive.org/interspeech_2015/nautsch15_interspeech.html
29. Speech Commands. arXiv:1804.03209. https://arxiv.org/abs/1804.03209
30. Streaming RNN-T KWS (keyword-filler). arXiv:1710.09617. https://arxiv.org/abs/1710.09617

**位置 bias / 温度 / 实现**
31. T5. arXiv:1910.10683. https://arxiv.org/abs/1910.10683 ；HF 实现 https://github.com/huggingface/transformers/blob/main/src/transformers/models/t5/modeling_t5.py
32. ALiBi. arXiv:2108.12409. https://arxiv.org/abs/2108.12409
33. Shaw et al. relative position. arXiv:1803.02155. https://arxiv.org/abs/1803.02155
34. Conformer. arXiv:2005.08100. https://arxiv.org/abs/2005.08100
35. Sperber et al. arXiv:2005.09940. https://arxiv.org/abs/2005.09940
36. Speech-enhancement PE comparison. https://www.isca-archive.org/interspeech_2024/zhang24n_interspeech.html
37. Kazemnejad et al. length generalization. arXiv:2305.19466. https://arxiv.org/abs/2305.19466
38. FlexAttention. arXiv:2412.05496. https://arxiv.org/abs/2412.05496
39. HyPE. arXiv:2310.19676. https://arxiv.org/abs/2310.19676
40. RIB. arXiv:2603.06738. https://arxiv.org/abs/2603.06738
41. Swin v2. arXiv:2111.09883. https://arxiv.org/abs/2111.09883
42. QKNorm. arXiv:2010.04245. https://arxiv.org/abs/2010.04245
43. ViT-22B. arXiv:2302.05442. https://arxiv.org/abs/2302.05442

**长度 / 长音频 / FA**
44. Adaptive pooling for weakly labeled SED. arXiv:1804.10070. https://arxiv.org/abs/1804.10070
45. Max vs noisy-OR pooling. arXiv:1804.01146. https://arxiv.org/abs/1804.01146
46. Max-pooling-loss LSTM KWS. arXiv:1705.02411. https://arxiv.org/abs/1705.02411
47. End-to-end streaming KWS. arXiv:1812.02802. https://arxiv.org/abs/1812.02802
48. Howl. https://aclanthology.org/2020.nlposs-1.9.pdf
49. Cross-layer discrimination consistency. arXiv:2412.12635. https://arxiv.org/abs/2412.12635
50. Multi-scale convolution KWS. https://www.isca-archive.org/interspeech_2020/yang20d_interspeech.html
51. Two-stage multi-resolution ensembles. arXiv:2310.11379. https://arxiv.org/abs/2310.11379
52. Length/noise-aware training. arXiv:2008.12218. https://arxiv.org/abs/2008.12218
53. Information Entropy Invariance. arXiv:2501.08570. https://arxiv.org/abs/2501.08570
54. Preventing Attention Entropy Collapse. arXiv:2303.06296. https://arxiv.org/abs/2303.06296
55. QMF duration calibration. https://doi.org/10.1109/tasl.2013.2279332
56. Trial-based calibration. https://doi.org/10.21437/odyssey.2014-4
57. Quantization-based score calibration for KWS. arXiv:2510.15432. https://arxiv.org/abs/2510.15432
58. MALEFA. arXiv:2604.03689. https://arxiv.org/abs/2604.03689

---

## 8. 复算说明

* 权重取证：torch.load(ckpt)['model_state_dict'] 的 qbyt.*；
  bucket 公式与 qbyt/pooling.py 的 _relative_position_bucket 完全一致（32 桶 / 64）。
* 钳制阈值：解 max_exact + log(n/max_exact)/log(max_distance/max_exact)*(num_buckets−max_exact) ≥ 15 → n ≥ 49.35 帧。
* D1 证伪：把 audio_buckets[15] 与 [31] 置零后另存 checkpoint（outputs/v41_bucket_probe/v41_nofarbucket.pt），
  用 scripts/eval_musan_fa.py 在 outputs/v41_bucket_probe/musan_top5.list（官方 3 s 网格分数最高的 5 个文件，845 窗口）
  上对比；结果见 §3。
* 1 s vs 3 s 分布：data/dma-kws/exp/stage2_qbyt/fa/v41-musan{,-1s0}/results.jsonl（51,994 / 156,929 窗口）。
* v4.1 vs v4：outputs/qbyt_v41_audit_2026-10-03/v41_vs_v4.csv。
