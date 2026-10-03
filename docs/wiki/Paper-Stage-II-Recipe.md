# Paper Stage II Recipe — negative mixing and the finetune lineage

What the authors' code actually does, read from
[aizhiqi-work/DMA-KWS](https://github.com/aizhiqi-work/DMA-KWS) `main`
(`train/two_stage/…`), plus how this repository mirrors it and what our completed runs did
instead.

**Last updated:** 2026-10-03.

## 1. How one training sample is built

All three Stage II dataset variants (`dataset/libriphrase_train.py`,
`libriphrase_train_npy.py`, `libriphrase_train_new_npy.py`) implement the same
`__getitem__`:

```python
if random.random() < 0.5:                        # 50 % positive sample
    label = 1
    query_wav = random.choice(anchor_clips)['audio_path']
else:                                            # 50 % negative sample
    label = 0
    negative_type = random.choices([1, 2],
        weights=[self.negative_ratio, self.hard_negative_ratio], k=1)[0]
    if negative_type == 1 or len(hard_neg_lists) == 0:
        ...   # random ("normal" / easy) negative: another anchor drawn at random
    else:
        ...   # hard negative: drawn from this anchor's distances list
```

* one sample is **one query against one anchor** (not "1 positive + N negatives"), label 1/0;
* **50 % of the draws are positives and 50 % are negatives**;
* inside the negative half the split is `negative_ratio : hard_negative_ratio` — the
  defaults in every dataset class are **1 : 1**;
* an empty hard-negative list falls back to a random negative.

With the default 1:1 the effective mix is therefore
**positive : easy negative : hard negative = 50 % : 25 % : 25 % = 2 : 1 : 1**, i.e.
**hard : normal = 1 : 1**.

## 2. Ratios and training settings per script

| script (paper) | dataset | `negative_ratio` | `hard_negative_ratio` | negative mix | steps | lr / warmup | optimiser covers |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `train.py` (`init-ls-460`) | `ls-460/aggregated_segments_with_g2p_distance.parquet` | 1 | **1** | 1 : 1 | 50 k | 1e-3 / 2500 | qbyt + encoder |
| `train_2.py` | `ls-gs-1460/processed_data.parquet` | 1 | **1** | 1 : 1 | 100 k (scheduler) | 1e-3 / 2500 | qbyt + encoder |
| `train_frozen.py` | `ls-gs-1460/processed_data.parquet` | 1 | **1** | 1 : 1 | 50 k | 1e-3 / 2500 | **qbyt only** (encoder frozen) |
| **`train_2_2ft.py`** | `ls-gs-1460/processed_data.parquet` | 1 | **100** | **1 : 100** | 100 k | **5e-4** / 2500 | qbyt + encoder |

All four use batch 512 × 4 GPUs, `accumulate_grad_batches=2`, `gradient_clip_val=1.0`,
`val_check_interval=1000` and a checkpoint every 1 000 steps, with a cosine schedule.
(`train_2.py` sets `max_steps=1000000` while its scheduler is 100 k — almost certainly a
typo in the released script.)

## 3. Initialisation / finetune lineage

| script | how the model is created | encoder weights | encoder frozen? | ckpt / log directory |
| --- | --- | --- | --- | --- |
| `train.py` | `Wrapper()` — **nothing loaded**, the Conformer encoder is randomly initialised | none | no | `qbyt_460/ckpts/init-ls-460` |
| `train_2.py` | `Wrapper()` fresh QbyT, then `self.encoder.load_state_dict(<Wenet Stage I ls-gs-1460 avg_10.pt>['encoder*'])` (the freeze lines are commented out) | Wenet Stage I | no | `qbyt_1460/ckpts/ft-ls-gs-1460->ls-gs-1460` |
| `train_frozen.py` | `Wrapper.load_from_checkpoint("qbyt_460/ckpts/ls-gs-1460->ls-460/avg_10.ckpt")` — **a previously trained QbyT** — plus the Stage I encoder with `requires_grad=False` | Wenet Stage I | **yes** | `qbyt_1460/ckpts/ls-gs-1460->ls-gs-1460` |
| **`train_2_2ft.py`** | `Wrapper.load_from_checkpoint("qbyt_1460/ckpts/ft-ls-gs-1460->ls-gs-1460/step_step=064000_auc_val_auc=0.975285.ckpt")` — **an already finetuned QbyT (step 64 k)** | restored from that checkpoint | no | `qbyt_1460/ckpts/ft-ls-gs-1460->ls-gs-1460-100:1-ft` |

**Yes — the paper's "two-stage finetune" starts from an already trained QbyT**, specifically
from a QbyT that had *already* been finetuned once (the `ft-…->…-ft` directory name says so).
`load_from_checkpoint` restores the whole Lightning state — encoder + QbyT + optimiser +
scheduler — so it is a warm restart rather than a weight-only cold start, and the encoder is
**not** frozen in that stage.

Read as a chain: `init-ls-460` (from scratch, 1:1) → swap in the Wenet Stage I encoder and
train 100 k on LS-GS-1460 (1:1) → frozen-encoder QbyT finetune 50 k → **another 100 k finetune
on LS-GS-1460 with hard negatives at 1:100** (the stage that is called the two-stage
finetune).

## 4. Hard negatives in our data

* source: the `distances` artefacts of the prepared parquet
  (`processed/stage2_qbyt/ls-gs-1460/aggregated_segments_with_g2p_distance.parquet` →
  `distances/*.npy`);
* **155 619 anchors, 15 561 900 hard-negative entries** (up to 100 per anchor, chosen by G2P
  distance);
* easy negatives are not precomputed — they are drawn from other anchors at training time.

## 5. How this repository mirrors it

| paper element | this repository |
| --- | --- |
| `load_from_checkpoint(...)` (full-state restart) | `stage2.resume_checkpoint` → `Stage2LightningModule.load_from_checkpoint(...)` |
| weight-only init from an encoder checkpoint | `stage2.init_checkpoint` (auto-detects the icefall/Wenet encoder format and loads `encoder_embed` + `encoder`) |
| `init-ls-460` (50 k, 1:1) | `+experiment=paper_ls460` — 50 k, `hard_negative_ratio: 1` |
| `ft-ls-gs-1460` (100 k, 1:100, warm start from the init stage) | `+experiment=paper_ls_gs1460` — 100 k, `hard_negative_ratio: 100`, `init_checkpoint` / `resume_checkpoint` = `init-ls-460/avg_10.ckpt` |
| negative sampling | `dma_kws/stage2/dataset.py` implements the same `random.choices([1, 2], weights=[negative_ratio, hard_negative_ratio])` and the same 50/50 positive switch |

**The 15 runs logged in the [Experiment Log](Experiment-Log) are single-stage**: their
`init_checkpoint` is only the frozen Stage I encoder (icefall KWS or the paper conformer),
`resume_checkpoint` is empty, and the QbyT head is trained from scratch. All of them used the
icefall-family preset `hard_negative_ratio: 1` — the 2:1:1 mix.

## 6. Reproducing the paper's `train_2_2ft` stage here

```bash
# warm restart from an existing Stage II checkpoint, hard-negative dominant
EXTRA_OVERRIDES="run.resume_from=data/dma-kws/exp/stage2_qbyt/checkpoints/<run>/<run>/version_N/last.ckpt" \
  bash scripts/start_stage2_qbyt_tmux.sh <experiment> <session>

# on top of that experiment overlay add (paper values):
#   stage2.hard_negative_ratio=100
#   stage2.max_steps=100000 stage2.total_scheduler_steps=100000
#   stage2.learning_rate=5e-4 stage2.warmup_steps=2500
#   stage2.accumulate_grad_batches=2
```

Budget note: the paper trains 4 GPUs × batch 512 × `accumulate_grad_batches=2` = 4 096
samples per optimizer step, while a local single-V100 run uses 128 samples per step. The step
count is therefore not directly comparable between the two setups — scale it if you want to
match the paper's sample budget.
