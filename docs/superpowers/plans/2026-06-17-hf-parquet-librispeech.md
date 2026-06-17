# HF Parquet LibriSpeech Stage I Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Allow Stage I phoneme CTC preparation to consume HuggingFace `openslr/librispeech_asr` parquet shards such as `clean/train.360`.

**Architecture:** Keep the existing LibriSpeech directory reader unchanged and add a focused parquet reader in `dma_kws/stage1/librispeech.py`. Extend `scripts/prepare_stage1_librispeech.py` with an `--input-format hf-parquet` path that writes extractable parquet audio bytes to a processed audio cache and emits the same JSONL manifest shape as the current directory mode.

**Tech Stack:** Python standard library, optional pandas/pyarrow for parquet reading, existing pytest tests.

---

### Task 1: Add parquet record conversion utilities

**Files:**
- Modify: `dma_kws/stage1/librispeech.py`
- Test: `tests/test_stage1_librispeech.py`

- [ ] Write tests for converting HuggingFace parquet records with `audio.bytes` into utterances.
- [ ] Run targeted test and confirm it fails because the parquet helpers do not exist.
- [ ] Implement a `ParquetAudioUtterance` dataclass and `iter_librispeech_parquet_utterances()` helper.
- [ ] Run targeted tests and confirm they pass.

### Task 2: Add prepare-script parquet mode

**Files:**
- Modify: `scripts/prepare_stage1_librispeech.py`
- Test: `tests/test_prepare_stage1_librispeech.py`

- [ ] Write tests for preparing a parquet utterance into JSONL and extracted audio cache.
- [ ] Run targeted test and confirm it fails because the prepare helper does not support parquet inputs.
- [ ] Implement `--input-format`, `--parquet-root`, and `--parquet-split` CLI options plus extraction helpers.
- [ ] Run targeted tests and confirm they pass.

### Task 3: Verify existing behavior

**Files:**
- Test: `tests/test_stage1_librispeech.py`
- Test: `tests/test_prepare_stage1_librispeech.py`

- [ ] Run `PYTHONPATH=. pytest tests -q`.
- [ ] Confirm existing LibriSpeech directory parsing still passes.
- [ ] Document usage command for the remote parquet directory.
