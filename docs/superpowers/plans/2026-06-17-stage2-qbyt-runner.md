# Stage II QbyT Runner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the minimal real Stage II pipeline needed to prepare positive/negative phrase pairs and start QbyT training.

**Architecture:** Keep pair generation independent of torch in `dma_kws.stage2.pairs`; keep dataset/audio/model training in `scripts/train_stage2_qbyt.py` so remote GPU dependencies are only needed when training.

**Tech Stack:** Python standard library, pandas/pyarrow for parquet preparation, optional g2p_en, PyTorch/torchaudio for remote training.

---

## Tasks

### Task 1: Pair builder helper

- [ ] Write one smoke test for making positive and random-negative pair records.
- [ ] Implement `AnchorExample`, `PairRecord`, and `make_pair_records`.
- [ ] Verify the smoke test passes.

### Task 2: Preparation CLI

- [ ] Implement `scripts/prepare_stage2_libriphrase.py`.
- [ ] Support author-style parquet columns `ngram`, `ngram_g2p`, and `clips`.
- [ ] Write `processed/stage2_qbyt/train.jsonl`.
- [ ] Verify `--help` runs locally.

### Task 3: QbyT training CLI

- [ ] Implement `scripts/train_stage2_qbyt.py`.
- [ ] Train ConformerEncoder + QbyT using BCE loss from prepared pair JSONL.
- [ ] Optionally load Stage I encoder weights if `--stage1-ckpt` is provided.
- [ ] Verify `--help` runs locally without importing torch at module import time.

### Task 4: Verification and commit

- [ ] Run simplified tests.
- [ ] Run new scripts with `--help`.
- [ ] Commit the Stage II runner slice.
