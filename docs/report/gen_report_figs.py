
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import numpy as np

OUT = "/home/ubuntu/dma-kws/docs/report/figures"
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Noto Serif CJK SC", "Noto Sans CJK SC", "DejaVu Serif"],
    "axes.unicode_minus": False,
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9.5,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 8,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.8,
    "figure.dpi": 110, "savefig.dpi": 300, "savefig.bbox": "tight",
})
RED="#C1272D"; BLUE="#0072B2"; GRAY="#8C8C8C"; MGRAY="#A6A6A6"; LGRAY="#C9C9C9"
ORANGE="#E69F00"; DARK="#2B2B2B"

def plabel(ax, s, x=-0.20, y=1.05):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=10.5, fontweight="bold", va="bottom", visible=True)

def check(fig, name, boxpairs=None):
    fig.canvas.draw(); r = fig.canvas.get_renderer()
    items = []
    for ax in fig.axes:
        cand = [ax.title, ax.xaxis.label, ax.yaxis.label] + list(ax.texts)
        if ax.axison:
            abb = ax.get_window_extent(r)
            for t in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
                tb = t.get_window_extent(r)
                cx, cy = (tb.x0+tb.x1)/2, (tb.y0+tb.y1)/2
                if abb.x0-60 <= cx <= abb.x1+60 and abb.y0-14 <= cy <= abb.y1+14:
                    cand.append(t)
        lg = ax.get_legend()
        if lg is not None:
            cand += list(lg.get_texts())
        for t in cand:
            if t is None or not t.get_text().strip() or not t.get_visible():
                continue
            items.append((t.get_text().replace("\n", "/")[:18], t.get_window_extent(r)))
    for t in fig.texts:
        if t.get_text().strip() and t.get_visible():
            items.append((t.get_text().replace("\n", "/")[:18], t.get_window_extent(r)))
    out = [lab for lab, bb in items
           if bb.x0 < fig.bbox.x0-1 or bb.x1 > fig.bbox.x1+1 or bb.y0 < fig.bbox.y0-1 or bb.y1 > fig.bbox.y1+1]
    bad = []
    for i in range(len(items)):
        for j in range(i+1, len(items)):
            a, b = items[i][1], items[j][1]
            ix = min(a.x1,b.x1)-max(a.x0,b.x0); iy = min(a.y1,b.y1)-max(a.y0,b.y0)
            if ix > 1.5 and iy > 1.5:
                bad.append((round(ix*iy), items[i][0], items[j][0]))
    bad.sort(reverse=True)
    print(f"[检查] {name}: 文本 {len(items)}, 越界 {len(out)}, 重叠 {len(bad)}")
    for o in out[:6]: print("    越界:", o)
    for b in bad[:8]: print(f"    重叠 {b[0]}px2: {b[1]} <-> {b[2]}")
    if boxpairs:
        for text_artist, patch in boxpairs:
            tb = text_artist.get_window_extent(r); pb = patch.get_window_extent(r)
            if tb.x0 < pb.x0 or tb.x1 > pb.x1 or tb.y0 < pb.y0 or tb.y1 > pb.y1:
                print(f"    文字超出框: {text_artist.get_text().replace(chr(10),'/')[:20]}")

def save(fig, name, boxpairs=None):
    fig.savefig(f"{OUT}/{name}.png"); check(fig, name, boxpairs); plt.close(fig)

def fig0_1():
    steps = [("v2\n修批依赖", 0.937647, False, 11), ("v3\n修校准", 0.931601, False, -15),
             ("v4", 0.932327, False, 11), ("v4.1\n加 sink", 0.935007, False, -15),
             ("C1\nv4.2 头", 0.937854, False, 11), ("v4.2\n两段式", 0.94258, False, -15),
             ("R1\n解冻续训", 0.946298, True, 11), ("SS-R1", 0.94779, True, -15),
             ("写回终版", 0.95028, True, 11)]
    fig, ax = plt.subplots(figsize=(7.8, 4.1))
    x = np.arange(len(steps)); ys = [s[1] for s in steps]
    ax.plot(x, ys, "-", color=LGRAY, lw=1.2, zorder=1)
    ax.scatter(x, ys, c=[RED if s[2] else GRAY for s in steps], s=42, zorder=3)
    for i, (lab, v, is_new, dy) in enumerate(steps):
        ax.annotate(f"{v:.4f}", (x[i], v), textcoords="offset points", xytext=(0, dy), ha="center",
                    fontsize=8.2, color=RED if is_new else DARK, fontweight="bold" if is_new else "normal")
    for i in range(1, len(steps)):
        ax.annotate(f"{(ys[i]-ys[i-1])*100:+.2f}", (x[i]-0.5, (ys[i]+ys[i-1])/2), ha="center", va="center",
                    fontsize=7.4, color=DARK)
    ax.axhline(0.95028, color=RED, ls="--", lw=1.1)
    ax.text(9.05, 0.9509, "当前最好\n0.9503", fontsize=8.4, color=RED, va="bottom", ha="right", fontweight="bold")
    ax.axhline(0.937647, color=GRAY, ls="--", lw=1.0)
    ax.text(1.6, 0.93805, "v2 强基线 0.9376", fontsize=8.4, color=GRAY, va="bottom", ha="left")
    ax.axhline(0.9595, color=BLUE, ls=":", lw=1.0)
    ax.text(-0.55, 0.9600, "作者发布模型 0.9595（联合训练，误唤醒 2266 次/24 小时）", fontsize=8.4, color=BLUE, va="bottom")
    ax.text(0.05, 0.9452, "起点：论文 v1 读出\n分数依赖批组成，无法安全批量推理", fontsize=8.2, color=DARK)
    ax.annotate("", xy=(0, 0.9382), xytext=(1.05, 0.9446),
                arrowprops=dict(arrowstyle="->", lw=0.9, color=DARK))
    ax.annotate("", xy=(9.0, 0.95028), xytext=(9.0, 0.937647),
                arrowprops=dict(arrowstyle="-", lw=1.2, color=RED))
    ax.text(9.08, 0.9439, "+1.26 pp\n对比 v2", fontsize=8.6, color=RED, fontweight="bold", va="center")
    ax.set_xticks(x); ax.set_xticklabels([s[0] for s in steps], fontsize=8.4)
    ax.set_ylabel("hard 验证集 AUC"); ax.set_xlabel("读出与训练配方演进（每一级修复上一级的实测缺陷）")
    ax.set_ylim(0.9255, 0.9645); ax.set_xlim(-0.7, 10.2)
    ax.set_title("从 v1 到 v4.2-R1：九级演进把 AUC 从 0.9376 推到 0.9503，每级都有实测依据", pad=12)
    fig.subplots_adjust(left=0.085, right=0.985, top=0.90, bottom=0.19)
    save(fig, "fig0-1_ladder_overview")

def fig1_1():
    fig, ax = plt.subplots(figsize=(7.8, 4.3))
    ax.set_xlim(0, 100); ax.set_ylim(0, 62); ax.axis("off")
    pairs = []
    def box(x, y, w, h, text, fc="white", ec=DARK, fs=8.4, bold=False):
        p = mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.7,rounding_size=1.2",
                                     fc=fc, ec=ec, lw=1.0)
        ax.add_patch(p)
        t = ax.text(x+w/2, y+h/2, text, ha="center", va="center", fontsize=fs,
                    fontweight="bold" if bold else "normal", linespacing=1.6)
        pairs.append((t, p))
    def arrow(x1, y1, x2, y2, color=DARK, lw=1.2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, shrinkA=0, shrinkB=0))
    ax.add_patch(mpatches.FancyBboxPatch((1, 2), 40, 56, boxstyle="round,pad=1,rounding_size=1.5",
                 fc="#F5F5F5", ec=GRAY, lw=0.9))
    ax.text(21, 56.5, "第一阶段  声学编码器（冻结）", fontsize=9.5, fontweight="bold", ha="center")
    box(5, 45, 32, 6.5, "音频 16 kHz")
    box(5, 35.5, 32, 6.5, "fbank 特征 80 维")
    box(5, 26, 32, 6.5, "声学编码器\nZipformer 或 Wenet")
    box(5, 16.5, 32, 6.5, "128 维帧特征")
    box(5, 6, 32, 7, "R1 路线：编码器解冻\n续训 13.3k 步", fc="#FBE9E7", ec=RED, fs=8.0)
    arrow(21, 45, 21, 42); arrow(21, 35.5, 21, 32.5); arrow(21, 26, 21, 23)
    ax.add_patch(mpatches.FancyBboxPatch((43, 2), 56, 56, boxstyle="round,pad=1,rounding_size=1.5",
                 fc="white", ec=DARK, lw=1.0))
    ax.text(71, 56.5, "第二阶段  QbyT 匹配器（可训 422K + 130 参数）", fontsize=9.5, fontweight="bold", ha="center")
    box(45.5, 45, 24, 6.5, "关键词文本\nG2P 音素序列")
    box(72.5, 45, 24, 6.5, "音素嵌入\n加文本位置编码")
    box(45.5, 35.5, 24, 6.5, "sink 令牌\n加 sink 头 130 参数", fc="#FBE9E7", ec=RED, fs=8.0)
    box(72.5, 35.5, 24, 6.5, "音频投影\n加相对偏置编码")
    box(51.5, 25, 40, 7, "拼接  文本 / sink / 音频\n2 层 Transformer，128 维，4 头")
    box(51.5, 14.5, 40, 7, "读出：文本位置 eps-softmin\n加 additive sink 头")
    box(51.5, 4.5, 40, 7, "整句分数 到 阈值判决 到 唤醒", fc="#FBE9E7", ec=RED, fs=8.6, bold=True)
    arrow(57.5, 45, 57.5, 42); arrow(84.5, 45, 84.5, 42)
    arrow(57.5, 35.5, 63, 32); arrow(84.5, 35.5, 79, 32)
    arrow(71, 25, 71, 21.5); arrow(71, 14.5, 71, 11.5)
    arrow(37, 19.75, 51.5, 19.75)
    ax.text(44.2, 21.4, "128 维特征", fontsize=8.2, ha="center")
    ax.set_title("开放词表唤醒的两阶段结构：编码器冻结，只训一个很小的文本查询匹配器", pad=12)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.90, bottom=0.03)
    save(fig, "fig1-1_architecture", pairs)

def fig3_1():
    fig = plt.figure(figsize=(8.0, 3.8))
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1.05, 1.0, 1.0], wspace=0.68)
    names = ["v2", "v3", "v4", "v4.1"]
    auc = [0.937647, 0.931601, 0.932327, 0.935007]
    brier = [0.135602, 0.105891, 0.106843, 0.108810]
    tpr1 = [0.202073, 0.156781, 0.170258, 0.183210]
    trig = [32.4, 20.3, 21.5, 21.5]
    margin = [0.304, 0.184, 0.200, -0.194]
    ax = fig.add_subplot(gs[0]); plabel(ax, "(a)", x=-0.30, y=1.06)
    ax.bar(range(4), auc, color=[GRAY, MGRAY, MGRAY, LGRAY], width=0.58)
    ax.axhline(0.937647, color=DARK, ls="--", lw=1.0)
    ax.text(2.35, 0.9383, "v2 基线", fontsize=8.2, color=DARK)
    for i, v in enumerate(auc):
        ax.text(i, v+0.0008, f"{v:.4f}", ha="center", fontsize=8.4, fontweight="bold" if i == 0 else "normal")
    ax.set_xticks(range(4))
    ax.set_xticklabels([f"{n}\n{t:.1f}%" for n, t in zip(names, trig)], fontsize=8.6)
    ax.set_ylim(0.9288, 0.9422); ax.set_ylabel("hard 验证集 AUC")
    ax.set_title("四级读出的 AUC", fontsize=9, pad=8)
    ax = fig.add_subplot(gs[1]); plabel(ax, "(b)", x=-0.34, y=1.06)
    ax.scatter(brier[0], tpr1[0], c=RED, s=55, zorder=3)
    for i in (1, 2, 3): ax.scatter(brier[i], tpr1[i], c=GRAY, s=45, zorder=3)
    offs = {"v2": ((13, 1), "left"), "v3": ((-3, -19), "center"),
            "v4": ((9, 5), "center"), "v4.1": ((9, 5), "center")}
    for i, n in enumerate(names):
        (dx, dy), ha = offs[n]
        ax.annotate(n, (brier[i], tpr1[i]), textcoords="offset points", xytext=(dx, dy),
                    fontsize=8.4, ha=ha)
    ax.plot(brier, tpr1, ls="--", lw=1.0, color=LGRAY, zorder=1)
    ax.annotate("", xy=(brier[1], tpr1[1]), xytext=(brier[0], tpr1[0]),
                arrowprops=dict(arrowstyle="-|>", lw=1.1, color=GRAY, shrinkA=10, shrinkB=10))
    ax.text(0.0955, 0.2165, "虚线＝演进顺序\n向左：校准变好\n向上：尾部召回变好",
            fontsize=7.4, color=DARK, va="top")
    ax.set_xlabel("Brier 分数（越低越好）"); ax.set_ylabel("TPR@1%FPR")
    ax.set_xlim(0.094, 0.152); ax.set_ylim(0.136, 0.228)
    ax.set_title("尾部与校准的取舍", fontsize=9, pad=8)
    ax = fig.add_subplot(gs[2]); plabel(ax, "(c)", x=-0.30, y=1.06)
    ax.bar(range(4), margin, color=[GRAY, GRAY, GRAY, RED], width=0.58)
    ax.axhline(0, color=DARK, lw=1.0)
    for i, v in enumerate(margin):
        ax.text(i, v + (0.024 if v >= 0 else -0.060), f"{v:+.2f}", ha="center", fontsize=8.4,
                color=RED if v < 0 else DARK, fontweight="bold" if v < 0 else "normal")
    ax.set_xticks(range(4)); ax.set_xticklabels(names, fontsize=8.6)
    ax.set_ylim(-0.30, 0.40); ax.set_xlim(-0.65, 3.6); ax.set_ylabel("EER 阈值 减 背景最高分")
    ax.text(-0.55, -0.118, "1 个背景窗口越线\n0.55 次/24 小时", fontsize=7.8, color=RED, ha="left", va="center")
    ax.set_title("各自 EER 工作点的安全余度", fontsize=9, pad=8)
    fig.suptitle("读出演进：每一级修掉一个实测缺陷，v4.1 换来尾部却花掉了安全余度", y=0.99, fontsize=10.2)
    fig.text(0.5, 0.035, "x 轴下行数字：固定阈值 0.5 时验证集负样本的误触发率。安全余度＝EER 阈值减去 MUSAN 背景最高分。",
             ha="center", fontsize=8.2, color=DARK)
    fig.subplots_adjust(left=0.075, right=0.985, top=0.84, bottom=0.26)
    save(fig, "fig3-1_readout_ladder")

def fig3_2():
    fig = plt.figure(figsize=(8.0, 3.4))
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1.75, 1.0], wspace=0.48)
    enc = ["zh-en-3M\n（流式）", "GS-base\n（流式）", "GS-finetune\n（流式）"]
    data = {"v2": [0.9376, 0.9271, 0.8809], "v3": [0.9316, 0.9064, 0.8598],
            "v4": [0.9323, 0.9131, 0.8565], "v4.1": [0.9350, 0.9119, 0.8687]}
    ax = fig.add_subplot(gs[0]); plabel(ax, "(a)", x=-0.15, y=1.10)
    x = np.arange(3); w = 0.19
    cols = {"v2": RED, "v3": MGRAY, "v4": GRAY, "v4.1": LGRAY}
    for i, (k, v) in enumerate(data.items()):
        pos = x + (i-1.5)*w
        ax.bar(pos, v, w, color=cols[k], label=k)
        for xi, vi in zip(pos, v):
            ax.text(xi, vi+0.0018, f"{vi:.3f}", rotation=90, ha="center", va="bottom", fontsize=6.6,
                    fontweight="bold" if k == "v2" else "normal", color=RED if k == "v2" else DARK)
    ax.set_xticks(x); ax.set_xticklabels(enc, fontsize=8.4)
    ax.set_ylim(0.838, 0.9495); ax.set_ylabel("hard 验证集 AUC")
    ax.legend(ncol=4, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.16))
    ax.set_title("15 次全量对照", fontsize=9, pad=8)
    ax = fig.add_subplot(gs[1]); plabel(ax, "(b)", x=-0.42, y=1.10)
    parts = [("编码器", 94.0, RED), ("读出", 4.6, GRAY), ("残差", 1.4, LGRAY)]
    for i, (nm, v, c) in enumerate(parts):
        ax.barh([2-i], [v], color=c, height=0.5)
        ax.text(v + (1.6 if v > 5 else 1.6), 2-i, f"{v:.1f}%", va="center", fontsize=8.6,
                color=RED if c == RED else DARK, fontweight="bold" if c == RED else "normal")
    ax.set_yticks([2, 1, 0]); ax.set_yticklabels(["编码器", "读出", "残差"], fontsize=9)
    ax.set_xlim(0, 112); ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("AUC 方差占比（%）")
    ax.set_title("方差分解", fontsize=9, pad=8)
    fig.suptitle("决定上限的是编码器而不是读出：15 次对照中编码器解释 94% 的 AUC 方差", y=0.995, fontsize=10.2)
    fig.subplots_adjust(left=0.085, right=0.97, top=0.83, bottom=0.25)
    save(fig, "fig3-2_encoder_matrix")

def fig4_1():
    fig = plt.figure(figsize=(8.0, 3.9))
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1.0, 1.45], wspace=0.42)
    T = [0.05,0.1,0.15,0.2,0.25,0.3,0.4,0.5,0.75,1.0,1.5,2.0,3.0,4.0]
    A = [0.93973,0.93978,0.93982,0.93984,0.93985,0.93986,0.93984,0.93979,0.93960,0.93937,0.93894,0.93860,0.93811,0.93780]
    ax = fig.add_subplot(gs[0]); plabel(ax, "(a)", x=-0.28, y=1.06)
    ax.axvspan(0.1, 0.3, color=RED, alpha=0.10, lw=0)
    ax.plot(T, A, "o-", color=DARK, ms=4, lw=1.2)
    ax.axhline(0.937647, color=GRAY, ls="--", lw=1.0)
    ax.text(1.55, 0.93782, "v2 基线", fontsize=8.2, color=GRAY)
    ax.scatter([1.0], [0.93937], c=BLUE, s=50, zorder=4)
    ax.annotate("训练温度 1.0", (1.0, 0.93937), textcoords="offset points", xytext=(-14, -22),
                fontsize=8.2, color=BLUE, ha="center")
    ax.annotate("", xy=(0.25, 0.93985), xytext=(1.0, 0.93937),
                arrowprops=dict(arrowstyle="->", lw=1.3, color=RED))
    ax.text(0.058, 0.94025, "0.1 至 0.3 平台\n零训练 +0.21 pp", fontsize=8.2, color=RED)
    ax.set_xscale("log"); ax.set_xlabel("打分温度 T（对数刻度）"); ax.set_ylabel("hard 验证集 AUC（C1）")
    ax.set_ylim(0.9370, 0.9409); ax.set_xticks([0.05, 0.1, 0.25, 0.5, 1, 2, 4])
    ax.set_xticklabels(["0.05", "0.1", "0.25", "0.5", "1", "2", "4"])
    ax.set_title("打分温度扫描：免费增益", fontsize=9, pad=8)
    ax = fig.add_subplot(gs[1]); plabel(ax, "(b)", x=-0.22, y=1.06)
    models = ["C1", "S1", "S2", "C3"]
    text_only = [0.93861, 0.93552, 0.93552, 0.93111]
    sink_only = [0.93552, 0.93841, 0.93903, 0.93732]
    combo = [0.94168, 0.94131, 0.94134, 0.93781]
    x = np.arange(4)
    for i in range(4):
        ax.plot([i, i], [text_only[i], sink_only[i]], color=LGRAY, lw=2.6, zorder=1, solid_capstyle="round")
    ax.scatter(x, text_only, c=GRAY, s=48, zorder=3, label="文本头单用")
    ax.scatter(x, sink_only, c=ORANGE, s=48, zorder=3, label="sink 头单用（事后拟合）")
    ax.scatter(x, combo, c=RED, s=72, marker="D", zorder=4, label="两路组合")
    for i in range(4):
        ax.annotate(f"{text_only[i]:.3f}", (i, text_only[i]), textcoords="offset points", xytext=(-14, 0),
                    fontsize=6.8, color=GRAY, ha="right", va="center")
        ax.annotate(f"{sink_only[i]:.3f}", (i, sink_only[i]), textcoords="offset points", xytext=(-14, -17),
                    fontsize=6.8, color="#B36B00", ha="right", va="center")
        ax.annotate(f"{combo[i]:.4f}", (i, combo[i]), textcoords="offset points", xytext=(0, 11),
                    fontsize=7.8, color=RED, ha="center")
    ax.set_xticks(x); ax.set_xticklabels(models, fontsize=9.0)
    ax.set_xlim(-0.85, 3.5)
    ax.set_ylim(0.9285, 0.9455); ax.set_ylabel("AUC（验证集后半区）")
    hnd, lab = ax.get_legend_handles_labels()
    fig.legend(hnd, lab, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.005))
    ax.set_title("头分解：共享主干上的目标冲突", fontsize=9, pad=8)
    fig.suptitle("sink 状态本来就带判别信息，读出它近乎免费；把 sink 损失塞进主训练则会互相伤害", y=0.995, fontsize=10.2)
    fig.text(0.5, 0.115, "连线两端：同一模型的文本头与 sink 头。C1 无 sink 损失；S1、S2 加 bce 损失；C3 从零训练。",
             ha="center", fontsize=8.2, color=DARK)
    fig.text(0.5, 0.075, "加上 sink 损失后 sink 头从 0.9355 升到 0.9390，文本头从 0.9386 降到 0.9311。",
             ha="center", fontsize=8.2, color=DARK)
    fig.subplots_adjust(left=0.10, right=0.98, top=0.83, bottom=0.30)
    save(fig, "fig4-1_probe")

def fig5_1():
    fig = plt.figure(figsize=(8.0, 4.3))
    gs = GridSpec(2, 5, figure=fig, height_ratios=[1.1, 1.0], hspace=0.95, wspace=0.8)
    metrics = [("AUC", 0.937647, 0.94258, "+0.49 pp", (0.930, 0.952), 4),
               ("EER（%）", 12.864, 12.360, "−0.50 pp", (10.0, 14.8), 2),
               ("TPR@1%（%）", 20.207, 24.595, "+4.39 pp", (12, 31), 2),
               ("TPR@0.1%（%）", 2.347, 3.992, "＋70%", (0, 6.6), 2),
               ("pAUC", 0.55236, 0.56721, "+1.48 pp", (0.530, 0.590), 4)]
    for i, (name, v2, v42, d, (ylo, yhi), nd) in enumerate(metrics):
        ax = fig.add_subplot(gs[0, i])
        ax.bar([0], [v2], 0.6, color=GRAY); ax.bar([1], [v42], 0.6, color=RED)
        ax.set_xlim(-0.8, 1.8); ax.set_ylim(ylo, yhi); ax.set_xticks([])
        ax.text(0, v2 + (yhi-ylo)*0.04, f"{v2:.{nd}f}", ha="center", fontsize=7.8)
        ax.text(1, v42 + (yhi-ylo)*0.04, f"{v42:.{nd}f}", ha="center", fontsize=7.8, fontweight="bold", color=RED)
        ax.text(0.5, yhi - (yhi-ylo)*0.03, d, ha="center", va="top", fontsize=8.8, color=RED, fontweight="bold")
        ax.set_title(name, fontsize=8.8, pad=6); ax.tick_params(labelsize=7.6)
    fig.text(0.5, 0.905, "灰柱：v2 强基线　　红柱：v4.2（两段式加写回，仅多 130 个参数）", ha="center", fontsize=8.8, color=DARK)
    gsb = gs[1, :].subgridspec(1, 2, wspace=0.30)
    ax = fig.add_subplot(gsb[0]); plabel(ax, "(b)", x=0.0, y=1.06)
    bars = ax.bar(["v4.1", "v4.2"], [0.409, 0.225], 0.85, color=[GRAY, RED])
    ax.axhline(0.22, color=BLUE, ls=":", lw=1.2)
    pass
    for b, v in zip(bars, [0.409, 0.225]):
        ax.text(b.get_x()+b.get_width()/2, v+0.016, f"{v:.3f}", ha="center", fontsize=8.6)
    ax.set_ylim(0, 0.55); ax.set_ylabel("MUSAN 背景最高分")
    ax.set_title("背景分尾巴（虚线＝自身 EER 阈值 0.22）", fontsize=9, pad=8)
    ax = fig.add_subplot(gsb[1]); plabel(ax, "(c)", x=0.0, y=1.06)
    bars = ax.bar(["v4.1", "v4.2"], [26.4, 3.2], 0.85, color=[GRAY, RED])
    for b, v in zip(bars, [26.4, 3.2]):
        ax.text(b.get_x()+b.get_width()/2, v+0.9, f"{v:.1f}", ha="center", fontsize=8.6)
    ax.text(0.5, 16, "−88%", ha="center", fontsize=11, color=RED, fontweight="bold")
    ax.set_ylim(0, 33); ax.set_ylabel("误唤醒（次/24 小时）")
    ax.set_title("语音背景误唤醒，阈值 0.05", fontsize=9, pad=8)
    fig.suptitle("v4.2 五项指标全面超过 v2 强基线，并把 v4.1 花掉的安全余度补了回来", y=0.995, fontsize=10.2)
    fig.subplots_adjust(left=0.075, right=0.98, top=0.80, bottom=0.09)
    save(fig, "fig5-1_v42_vs_v2")

def fig5_2():
    encs = ["zh-en-3M", "GS-base\n流式", "GS-base\n全上下文", "GS-KWS\n微调版", "paper-1460h\n从零训练"]
    base = [0.93501, 0.91190, 0.91290, 0.86870, 0.92639]
    v42 = [0.94258, 0.91849, 0.91994, 0.87467, 0.93439]
    fig, ax = plt.subplots(figsize=(8.0, 3.2))
    x = np.arange(5); w = 0.33
    ax.bar(x-w/2, base, w, color=GRAY, label="v4.1 基线")
    ax.bar(x+w/2, v42, w, color=RED, label="v4.2（两段式加写回）")
    for xi, b, v in zip(x, base, v42):
        ax.text(xi+w/2, v-0.0025, f"{v:.4f}", ha="center", va="top", fontsize=8.2, fontweight="bold", color="white")
        ax.text(xi-w/2, b-0.0025, f"{b:.4f}", ha="center", va="top", fontsize=7.8, color="white")
        ax.text(xi, max(b, v)+0.0045, f"提升 {(v-b)*100:+.2f} pp", ha="center", fontsize=8.6, color=RED, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(encs, fontsize=8.4)
    ax.set_ylim(0.852, 0.960); ax.set_ylabel("hard 验证集 AUC")
    ax.legend(frameon=False, ncol=2, loc="upper right")
    ax.set_title("同一套加 130 参数的配方可以迁移：五种编码器全部提升 0.60 到 0.80 个百分点", pad=12)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.88, bottom=0.19)
    save(fig, "fig5-2_encoder_gains")

def fig6_1():
    fig = plt.figure(figsize=(8.0, 5.6))
    gs = GridSpec(2, 2, figure=fig, hspace=0.85, wspace=0.45)
    ax = fig.add_subplot(gs[0, 0]); plabel(ax, "(a)", x=-0.10, y=1.12)
    names = ["基线\nSS-zh-en", "U1 解冻\nlr 2e-5", "U2 解冻\nlr 5e-5", "H1 冻结\n100:1", "H2 解冻\n100:1"]
    vals = [0.94106, 0.93423, 0.93966, 0.93948, 0.93767]
    ax.bar(names, vals, 0.58, color=[DARK, MGRAY, MGRAY, MGRAY, MGRAY])
    ax.axhline(0.94106, color=DARK, ls="--", lw=1.0)
    for i, v in enumerate(vals):
        ax.text(i, v+0.0006, f"{v:.4f}", ha="center", fontsize=7.8)
    ax.set_ylim(0.9325, 0.9435); ax.set_ylabel("hard 验证集 AUC"); ax.tick_params(axis="x", labelsize=7.8)
    ax.set_title("3k 步热启动：四组全部低于起点", fontsize=9, pad=8)
    ax.text(0.5, -0.26, "U＝解冻 encoder（lr 2e-5 / 5e-5）；H＝100:1 硬负样本（H1 冻结、H2 解冻）",
            transform=ax.transAxes, ha="center", fontsize=7.4, color=DARK)
    ax.text(0.5, -0.38, "从零跑 100:1 在 5k 步熔断（AUC 0.672），该配方必须从收敛点续训",
            transform=ax.transAxes, ha="center", fontsize=7.4, color=DARK)
    ax = fig.add_subplot(gs[0, 1]); plabel(ax, "(b)", x=-0.10, y=1.12)
    names = ["C1\n起点", "R1\n解冻加 100:1\n13.3k 步", "SS-R1\nsink 头 3k", "写回\ndev 拟合"]
    vals = [0.93785, 0.94630, 0.94779, 0.95028]
    ax.bar(names, vals, 0.56, color=[LGRAY, GRAY, MGRAY, RED])
    ax.axhline(0.94258, color=BLUE, ls=":", lw=1.2)
    ax.text(-0.55, 0.9429, "旧最好", fontsize=7.6, color=BLUE, ha="left")
    for i, v in enumerate(vals):
        ax.text(i, v+0.0006, f"{v:.4f}", ha="center", fontsize=8.2,
                fontweight="bold" if i == 3 else "normal", color=RED if i == 3 else DARK)
    ax.set_ylim(0.9350, 0.9540); ax.set_ylabel("hard 验证集 AUC"); ax.tick_params(axis="x", labelsize=7.4)
    ax.set_xlim(-0.65, 4.15)
    ax.annotate("", xy=(3.38, 0.95028), xytext=(3.38, 0.94258),
                arrowprops=dict(arrowstyle="<->", lw=1.0, color=BLUE))
    ax.text(3.50, 0.9464, "+0.77 pp 对旧最好", fontsize=7.0, color=BLUE, va="center", rotation=90)
    ax.annotate("", xy=(3.72, 0.95028), xytext=(3.72, 0.93785),
                arrowprops=dict(arrowstyle="<->", lw=1.0, color=DARK))
    ax.text(3.84, 0.9440, "+1.25 pp 对 C1 起点", fontsize=7.0, color=DARK, va="center", rotation=90)
    ax.set_title("R1 链：论文式续训的四级台阶", fontsize=9, pad=8)
    ax = fig.add_subplot(gs[1, 0]); plabel(ax, "(c)", x=-0.10, y=1.12)
    names = ["v4.1", "v4.2-zh", "v4.2-R1"]
    musan = [0.55, 0.55, 0.0]; ls = [0.29, 0.58, 0.0]
    x = np.arange(3); w = 0.28
    b1 = ax.bar(x-w/2, musan, w, color=GRAY, label="MUSAN")
    b2 = ax.bar(x+w/2, ls, w, color=BLUE, label="LibriSpeech")
    for b, v in list(zip(b1, musan)) + list(zip(b2, ls)):
        ax.text(b.get_x()+b.get_width()/2, v+0.022, f"{v:.2f}", ha="center", fontsize=8.2)
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=8.6)
    ax.set_ylim(0, 0.80); ax.set_ylabel("误唤醒（次/24 小时）")
    ax.legend(frameon=False, ncol=2, loc="upper right")
    ax.set_title("各自 EER 阈值下，3 秒窗：R1 版归零", fontsize=9, pad=8)
    ax = fig.add_subplot(gs[1, 1]); plabel(ax, "(d)", x=-0.10, y=1.12)
    vals = [29.1, 80.7, 22.5]
    ax.bar(names, vals, 0.5, color=[GRAY, MGRAY, RED])
    for i, v in enumerate(vals):
        ax.text(i, v+1.8, f"{v:.1f}", ha="center", fontsize=8.4,
                fontweight="bold" if i == 2 else "normal", color=RED if i == 2 else DARK)
    ax.text(2.45, 86, "比 v4.2-zh\n干净 3.6 倍", ha="right", fontsize=7.8, color=RED)
    ax.set_ylim(0, 100); ax.set_ylabel("误唤醒（次/24 小时，1 秒窗）")
    ax.set_title("1 秒窗压力测试：R1 版最干净", fontsize=9, pad=8)
    fig.suptitle("编码器微调只有按长程续训才有效；跑通之后，第 3 章遗留的误唤醒退化也一并修好", y=0.99, fontsize=10.2)
    fig.subplots_adjust(left=0.085, right=0.98, top=0.90, bottom=0.07)
    save(fig, "fig6-1_r1_chain")


# ================================================ 图 7-1 增训效果
def fig7_1():
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(8.0, 3.6))
    gs = GridSpec(1, 4, figure=fig, wspace=0.75)
    names = ["增训前", "LoRA\n24k", "QbyT\n439k", "enc\n3.19M"]
    real = [0.6085, 0.9849, 0.9922, 0.9991]
    tts  = [0.8228, 0.9860, 0.9807, 0.9956]
    lph  = [0.9411, 0.9300, 0.9300, 0.9413]
    rec  = [0.2172, 0.9041, 0.9616, 0.9911]
    trig = [0.1657, 0.0414, 0.0444, 0.0163]
    cols = [LGRAY, MGRAY, GRAY, RED]
    axs = []
    for k, (vals, title, ylo, yhi, fmt) in enumerate([
            (real, "真人留出说话人 AUC", 0.55, 1.18, "%.4f"),
            (tts,  "TTS 留出音色 AUC",   0.78, 1.14, "%.4f"),
            (lph,  "通用能力 LibriPhrase AUC", 0.90, 0.99, "%.4f")]):
        ax = fig.add_subplot(gs[0, k]); plabel(ax, f"({chr(97+k)})", x=-0.34, y=1.06); axs.append(ax)
        ax.bar(range(4), vals, 0.62, color=cols)
        for i, v in enumerate(vals):
            ax.text(i, v + (yhi - ylo) * 0.02, fmt % v, ha="center", va="bottom", rotation=90,
                    fontsize=6.8, fontweight="bold" if i == 3 else "normal",
                    color=RED if i == 3 else DARK)
        ax.set_xticks(range(4)); ax.set_xticklabels(names, fontsize=7.4)
        ax.set_ylim(ylo, yhi); ax.set_title(title, fontsize=8.8, pad=8)
        ax.tick_params(axis="y", labelsize=7.6)
    axs[0].annotate("", xy=(0.52, 0.9849), xytext=(0.52, 0.6085),
                    arrowprops=dict(arrowstyle="<->", lw=1.1, color=RED))
    axs[0].text(0.30, 0.80, "+0.376", fontsize=7.6, color=RED, fontweight="bold",
                ha="center", va="center", rotation=90)
    ax = fig.add_subplot(gs[0, 3]); plabel(ax, "(d)", x=-0.34, y=1.14)
    x = np.arange(4); w = 0.34
    ax.bar(x - w/2, rec, w, color=RED)
    ax.bar(x + w/2, trig, w, color=GRAY)
    for xi, a, b in zip(x, rec, trig):
        ax.text(xi - w/2, a + 0.02, f"{a:.2f}", ha="center", va="bottom", rotation=90,
                fontsize=6.6, color=RED)
        ax.text(xi + w/2, b + 0.02, f"{b:.3f}", ha="center", va="bottom", rotation=90,
                fontsize=6.6)
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=7.0)
    ax.set_ylim(0, 1.22)
    ax.set_title("阈值 0.5\n红＝召回，灰＝误触发", fontsize=8.0, pad=8)
    fig.suptitle("增训把真人唤醒从 0.6085 拉到 0.9991，通用能力不但没丢反而略升", y=0.99, fontsize=10.2)
    fig.subplots_adjust(left=0.075, right=0.985, top=0.83, bottom=0.19)
    save(fig, "fig7-1_adaptation_effect")

# ================================================ 图 7-2 方法选择
def fig7_2():
    fig = plt.figure(figsize=(8.0, 3.4))
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1.0, 1.15, 1.0], wspace=0.55)
    scopes = ["LoRA\n24k", "QbyT 全参\n439k", "encoder 全参\n3.19M"]
    ax = fig.add_subplot(gs[0]); plabel(ax, "(a)", x=-0.30, y=1.06)
    vals = [0.9849, 0.9922, 0.9991]; losses = [0.270, 0.164, 0.040]
    ax.bar(range(3), vals, 0.6, color=[MGRAY, GRAY, RED])
    for i, (v, l) in enumerate(zip(vals, losses)):
        ax.text(i, v + 0.0018, f"{v:.4f}", ha="center", fontsize=8.0,
                fontweight="bold" if i == 2 else "normal", color=RED if i == 2 else DARK)
        ax.text(i, 0.9765, f"损失\n{l:.3f}", ha="center", fontsize=7.0, color=DARK)
    ax.set_xticks(range(3)); ax.set_xticklabels(scopes, fontsize=7.6)
    ax.set_ylim(0.974, 1.004); ax.set_ylabel("真人留出 AUC")
    ax.set_title("放开可训范围的收益", fontsize=9, pad=8)
    ax = fig.add_subplot(gs[1]); plabel(ax, "(b)", x=-0.26, y=1.06)
    x = np.arange(3); w = 0.33
    c1 = [0.9837, 0.9922, 0.9991]; r1 = [0.8938, 0.9192, 0.9558]
    ax.bar(x - w/2, c1, w, color=RED, label="C1 基座")
    ax.bar(x + w/2, r1, w, color=GRAY, label="R1 基座")
    for xi, a, b in zip(x, c1, r1):
        ax.text(xi - w/2, a + 0.002, f"{a:.4f}", ha="center", fontsize=7.0, color=RED)
        ax.text(xi + w/2, b + 0.002, f"{b:.4f}", ha="center", fontsize=7.0, color=DARK)
    ax.set_xticks(x); ax.set_xticklabels(scopes, fontsize=7.4)
    ax.set_ylim(0.86, 1.02); ax.set_ylabel("真人留出 AUC")
    ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.20), fontsize=7.6)
    ax.set_title("同一配方换基座：C1 全面领先", fontsize=9, pad=8)
    ax = fig.add_subplot(gs[2]); plabel(ax, "(c)", x=-0.30, y=1.06)
    vals = [3.363, 67.869, 16.424, 4.049]
    bars = ax.bar(range(4), vals, 0.6, color=[LGRAY, MGRAY, GRAY, RED])
    for i, v in enumerate(vals):
        ax.text(i, v * 1.12, f"{v:.1f}", ha="center", fontsize=7.4,
                fontweight="bold" if i == 3 else "normal", color=RED if i == 3 else DARK)
    ax.set_yscale("log"); ax.set_ylim(1.8, 220)
    ax.set_xticks(range(4)); ax.set_xticklabels(["增训前", "LoRA", "QbyT\n全参", "encoder\n全参"], fontsize=7.6)
    ax.set_ylabel("误唤醒（次/小时，1 秒窗，对数刻度）")
    ax.set_title("误唤醒：从 67.9 压回 4.0", fontsize=9, pad=8)
    fig.suptitle("结论：放开可训范围到 encoder 全参，且基座留在 C1，不要换 R1", y=0.99, fontsize=10.2)
    fig.subplots_adjust(left=0.085, right=0.98, top=0.82, bottom=0.26)
    save(fig, "fig7-2_scope_and_base")

# ================================================ 图 7-3 部署工作点
def fig7_3():
    fig = plt.figure(figsize=(8.0, 4.5))
    gs = GridSpec(2, 3, figure=fig, width_ratios=[1.32, 1.0, 1.0], hspace=0.72, wspace=0.46)
    fa = [0.05, 0.1, 0.5, 1.0]
    curves = [
        ("增训前 SS-zh-en", [0.360, 0.384, 0.432, 0.454], LGRAY, "o"),
        ("C1 + LoRA", [0.746, 0.822, 0.904, 0.923], ORANGE, "s"),
        ("C1 + QbyT 全参", [0.926, 0.933, 0.970, 0.977], MGRAY, "^"),
        ("C1 + encoder 全参（交付）", [0.996, 0.996, 0.997, 0.998], RED, "D"),
        ("R1 + encoder 5e-5", [0.965, 0.967, 0.970, 0.971], BLUE, "v"),
    ]
    ax = fig.add_subplot(gs[:, 0]); plabel(ax, "(a)", x=-0.20, y=1.03)
    for name, ys, c, m in curves:
        ax.plot(fa, ys, marker=m, color=c, lw=1.5, ms=5, label=name, zorder=4 if c == RED else 3)
    ax.set_xscale("log"); ax.set_xticks(fa); ax.set_xticklabels(["0.05", "0.1", "0.5", "1.0"])
    ax.set_xlim(0.04, 1.35); ax.set_ylim(0.30, 1.08)
    ax.set_xlabel("目标误唤醒预算（次/小时，对数刻度）"); ax.set_ylabel("真人留出唤醒率")
    ax.legend(frameon=False, fontsize=6.8, loc="center right", bbox_to_anchor=(1.02, 0.42))
    ax.set_title("标准口径：唤醒率与误唤醒预算的交换曲线", fontsize=8.8, pad=8)

    names = ["增训前", "LoRA", "QbyT", "R1+enc", "enc\n全参"]
    cols = [LGRAY, MGRAY, GRAY, BLUE, RED]
    panels = [
        (0, 1, "(b)", "TTS 唤醒率（预算 0.05）", [0.985, 0.848, 0.891, 0.948, 0.970], (0.80, 1.04)),
        (1, 1, "(c)", "近音词误触发（预算 0.05）", [0.247, 0.000, 0.031, 0.260, 0.024], (0, 0.31)),
        (0, 2, "(d)", "TTS 唤醒率（预算 1.0）", [0.991, 0.936, 0.909, 0.964, 0.970], (0.80, 1.04)),
        (1, 2, "(e)", "近音词误触发（预算 1.0）", [0.305, 0.061, 0.065, 0.283, 0.039], (0, 0.37)),
    ]
    for (r, c, letter, title, vals, (ylo, yhi)) in panels:
        ax = fig.add_subplot(gs[r, c]); plabel(ax, letter, x=-0.30, y=1.05)
        ax.bar(range(5), vals, 0.62, color=cols)
        for i, v in enumerate(vals):
            ax.text(i, v + (yhi - ylo) * 0.035, f"{v:.3f}", ha="center", fontsize=6.8,
                    color=RED if i == 4 else DARK, fontweight="bold" if i == 4 else "normal")
        ax.set_xticks(range(5)); ax.set_xticklabels(names, fontsize=7.0)
        ax.set_ylim(ylo, yhi); ax.tick_params(axis="y", labelsize=7.4)
        ax.set_title(title, fontsize=8.2, pad=7)
    fig.suptitle("两个误唤醒预算档的对照：交付模型在两档都是唤醒率最高、近音词误触发最低", y=1.0, fontsize=10.2)
    fig.text(0.5, 0.015, "读图提示：增训前模型的 TTS 唤醒率看着高，是因为它对近音词也点头；请与同预算的近音词柱一起读。",
             ha="center", fontsize=8.0, color=DARK)
    fig.subplots_adjust(left=0.08, right=0.985, top=0.87, bottom=0.14)
    save(fig, "fig7-3_operating_points")

# ================================================ 图 7-4 配方与消融
def fig7_4():
    fig = plt.figure(figsize=(8.0, 4.6))
    gs = GridSpec(2, 3, figure=fig, height_ratios=[1.0, 1.15], hspace=0.62, wspace=0.5)
    # (a) joint vs sequential：TTS AUC
    ax = fig.add_subplot(gs[0, 0]); plabel(ax, "(a)", x=-0.30, y=1.08)
    ax.bar([0, 1], [0.9099, 0.9860], 0.55, color=[GRAY, RED])
    for i, v in enumerate([0.9099, 0.9860]):
        ax.text(i, v + 0.004, f"{v:.4f}", ha="center", fontsize=7.6)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["sequential", "joint"], fontsize=8.4)
    ax.set_ylim(0.88, 1.01); ax.set_ylabel("TTS 留出 AUC")
    ax.set_title("TTS 唤醒能力", fontsize=8.8, pad=8)
    # (b) joint vs sequential：音乐噪声误唤醒（对数）
    ax = fig.add_subplot(gs[0, 1]); plabel(ax, "(b)", x=-0.30, y=1.08)
    ax.bar([0, 1], [27.14, 1.48], 0.55, color=[GRAY, RED])
    for i, v in enumerate([27.14, 1.48]):
        ax.text(i, v * 1.15, f"{v:.2f}", ha="center", fontsize=7.8)
    ax.set_yscale("log"); ax.set_ylim(0.8, 90)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["sequential", "joint"], fontsize=8.4)
    ax.set_ylabel("误唤醒（次/小时）")
    ax.set_title("背景误唤醒，低 18 倍", fontsize=8.8, pad=8)
    # (c) joint vs sequential：1 FA/h 召回
    ax = fig.add_subplot(gs[0, 2]); plabel(ax, "(c)", x=-0.30, y=1.08)
    ax.bar([0, 1, 2], [0.498, 0.698, 0.885], 0.55, color=[LGRAY, GRAY, RED])
    for i, v in enumerate([0.498, 0.698, 0.885]):
        ax.text(i, v + 0.012, f"{v:.3f}", ha="center", fontsize=7.6,
                fontweight="bold" if i == 2 else "normal", color=RED if i == 2 else DARK)
    ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["增训前", "sequential", "joint"], fontsize=7.8)
    ax.set_ylim(0, 1.02); ax.set_ylabel("1 次/小时预算真人召回")
    ax.set_title("配额方式的差别", fontsize=8.8, pad=8)
    # (d) TTS 预热消融
    ax = fig.add_subplot(gs[1, 0]); plabel(ax, "(d)", x=-0.30, y=1.06)
    pairs = [("真人 AUC", 0.9709, 0.9870), ("TPR@1%", 0.6967, 0.7690), ("近负误触发", 0.1775, 0.0651)]
    for yi, (lab, a, b) in enumerate(pairs):
        y = 2 - yi
        ax.plot([a, b], [y, y], "-", color=LGRAY, lw=2.8, zorder=1, solid_capstyle="round")
        ax.scatter([a], [y], c=GRAY, s=46, zorder=3, label="直接上真人" if yi == 0 else None)
        ax.scatter([b], [y], c=RED, s=46, zorder=3, label="TTS 预热 600 步" if yi == 0 else None)
        ax.text(a - 0.16 if a < b else a + 0.16, y, f"{a:.3f}",
                ha="right" if a < b else "left", va="center", fontsize=7.2, color=GRAY)
        ax.text(b + 0.16 if a < b else b - 0.16, y, f"{b:.3f}",
                ha="left" if a < b else "right", va="center", fontsize=7.2, color=RED)
        ax.text((a + b) / 2, y + 0.20, f"{b - a:+.3f}", ha="center", fontsize=7.6, color=RED, fontweight="bold")
    ax.set_yticks([2, 1, 0]); ax.set_yticklabels(["真人 AUC", "TPR@1%", "近负误触发"], fontsize=7.6)
    ax.set_xlim(-0.48, 1.34); ax.set_ylim(-0.55, 2.80)
    ax.legend(frameon=False, fontsize=6.8, loc="upper left", bbox_to_anchor=(-0.02, 1.04))
    ax.set_title("消融：TTS 预热值这 600 步", fontsize=8.8, pad=8)
    # (e) 负结果一：背景加压
    ax = fig.add_subplot(gs[1, 1]); plabel(ax, "(e)", x=-0.08, y=1.16)
    ax.bar([0, 1], [0.369, 0.252], 0.55, color=[GRAY, MGRAY])
    for i, v in enumerate([0.369, 0.252]):
        ax.text(i, v + 0.010, f"{v:.3f}", ha="center", fontsize=7.6)
    ax.set_ylim(0, 0.45); ax.set_xticks([0, 1]); ax.set_xticklabels(["增训后", "背景加压后"], fontsize=7.8)
    ax.set_ylabel("0.05 次/小时预算真人召回")
    ax.set_title("负结果：多喂随机背景没用", fontsize=8.8, pad=8)
    # (f) 负结果二：sink 写回
    ax = fig.add_subplot(gs[1, 2]); plabel(ax, "(f)", x=-0.08, y=1.16)
    ax.bar([0, 1], [1.464, 187.96], 0.55, color=[GRAY, MGRAY])
    for i, v in enumerate([1.464, 187.96]):
        ax.text(i, v * 1.18, f"{v:.2f}", ha="center", fontsize=7.6)
    ax.set_yscale("log"); ax.set_ylim(0.8, 700)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["不写回", "域内拟合写回"], fontsize=7.8)
    ax.set_ylabel("误唤醒（次/小时）")
    ax.set_title("负结果：sink 写回放大 128 倍", fontsize=8.8, pad=8)
    fig.suptitle("配方与消融：joint 配额是主配方，TTS 预热值得做，随机加压与 sink 写回都不要做", y=0.985, fontsize=10.2)
    fig.subplots_adjust(left=0.085, right=0.98, top=0.86, bottom=0.10)
    save(fig, "fig7-4_recipe_and_ablations")

# ================================================ 图 8-1 收益总表
def fig8_1():
    import textwrap
    rows = [
        ("阶段一\n复现与平台", "论文代码不能直接用于产品词训练",
         "编码器复现校验一致（249 张量哈希）；v1 目标 5k 步达 0.9472 且误唤醒 0；99 项测试全绿",
         "版本化 checkpoint 与可复现评测协议"),
        ("阶段二\n读出演进 v1 到 v4.1", "分数依赖批组成；校准差导致固定阈值误触发 32.4%；尾部弱；安全余度薄",
         "修好批不变性；误触发 32.4% 降到 20.3%；近音词尾部 +2.5 个百分点；确认编码器解释 94% 方差",
         "强基线 v2；目标冲突的认知；跨编码器对照矩阵"),
        ("阶段三\n审计 v4.1", "不重训还有没有空间",
         "打分温度零训练 +0.21 个百分点；事后拟合 sink 头 +0.31 个百分点；证否四条路",
         "v4.2 的设计依据：sink 状态带判别信息"),
        ("阶段四\nv4.2", "把零训练增益固化进模型",
         "五项指标全面超过 v2（AUC 0.9426，EER 12.36%）；五种编码器全部 +0.60 到 +0.80 个百分点；只多 130 参数",
         "新主基座 C1 与 SS；负结果清单"),
        ("阶段五\nencoder 微调 R1", "通用指标还能不能再提一档",
         "hard AUC 0.9463 提到 0.9503（项目最好）；1 秒窗误唤醒 80.7 降到 22.5；EER 11.20%",
         "新通用基座 v4.2-R1 链（供后续增训选型）"),
        ("阶段六\nHey Eva 增训", "通用指标好不等于产品词好（真人 AUC 只有 0.6085）",
         "真人 AUC 0.9991；EER 1.05%；1 次/小时预算召回 0.998；1 秒窗误唤醒 4.05",
         "交付 c1_encfull 模型；标准评测口径与部署清单"),
    ]
    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")
    xs = [0.6, 16.2, 36.2, 76.2]
    ws = [15.0, 19.4, 39.4, 23.0]
    head = ["阶段", "解决的缺陷", "量化收益", "留下的资产"]
    y = 94
    hh = 5.0
    allpairs = []
    for x, w, t in zip(xs, ws, head):
        ax.add_patch(mpatches.Rectangle((x, y), w, hh, fc="#F0F0F0", ec=DARK, lw=0.8))
        ax.text(x + w/2, y + hh/2, t, ha="center", va="center", fontsize=9, fontweight="bold")
    y -= hh
    rh = 14.5
    for i, r in enumerate(rows):
        fc = "#FDF3EE" if i == len(rows) - 1 else "white"
        cells = []
        for x, w, t in zip(xs, ws, r):
            ax.add_patch(mpatches.Rectangle((x, y - rh), w, rh, fc=fc, ec=DARK, lw=0.7))
            wrapped = "\n".join(textwrap.wrap(t, width=max(6, int(w * 0.70))))
            cell = ax.text(x + w/2, y - rh/2, wrapped, ha="center", va="center", fontsize=7.1,
                           linespacing=1.6)
            cells.append((cell, ax.patches[-1]))
        allpairs.extend(cells)
        y -= rh
    fig.suptitle("收益总账：六个阶段，每一阶段都有可复核的数字", y=0.965, fontsize=10.5)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.90, bottom=0.02)
    save(fig, "fig8-1_gains_table", allpairs)

for f in [fig0_1, fig1_1, fig3_1, fig3_2, fig4_1, fig5_1, fig5_2, fig6_1, fig7_1, fig7_2, fig7_3, fig7_4, fig8_1]:
    try:
        f()
    except Exception:
        import traceback; traceback.print_exc()
print("done")
