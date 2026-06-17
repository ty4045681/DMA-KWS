# Stage I CTC Runner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the minimal real Stage I pipeline needed to prepare LibriSpeech phoneme manifests and start phoneme CTC training.

**Architecture:** Keep reusable parsing/G2P helpers in `dma_kws.stage1.librispeech`, and keep runnable CLIs in `scripts/`. Heavy dependencies such as torch/torchaudio are imported inside training code so local smoke checks can run without GPU dependencies.

**Tech Stack:** Python standard library, PyYAML, optional `g2p_en` for preparation, PyTorch/torchaudio for remote training.

---

## Files

- Create: `dma_kws/stage1/librispeech.py` — LibriSpeech transcript parsing, audio path resolution, stress stripping.
- Create: `tests/test_stage1_librispeech.py` — one smoke test for transcript parsing.
- Create: `scripts/prepare_stage1_librispeech.py` — manifest/vocab generation CLI.
- Create: `scripts/train_stage1_ctc.py` — minimal phoneme CTC training CLI.
- Modify: `configs/demo_librispeech100.yaml` — add Stage I split and training knobs.
- Create: `requirements.txt` — lightweight project requirements with a note that CUDA PyTorch should be installed separately on the remote machine.

## Tasks

### Task 1: LibriSpeech parsing helper

- [ ] Write one failing smoke test for parsing a `.trans.txt` line and locating `.flac`.
- [ ] Implement `TranscriptUtterance`, `iter_librispeech_utterances`, `strip_stress_marker`.
- [ ] Run the smoke test and verify it passes.

### Task 2: Stage I preparation CLI

- [ ] Implement `scripts/prepare_stage1_librispeech.py`.
- [ ] It reads configured splits, runs `g2p_en`, writes JSONL manifests, and writes `phoneme_vocab.txt`.
- [ ] Verify `--help` runs locally without importing `g2p_en` at module import time.

### Task 3: Stage I CTC training CLI

- [ ] Implement `scripts/train_stage1_ctc.py`.
- [ ] It reads a manifest/vocab, computes fbank with torchaudio, trains ConformerEncoder + CTC, and saves a checkpoint.
- [ ] Verify `--help` runs locally without importing torch at module import time.

### Task 4: Config and requirements

- [ ] Add `train_splits`, `dev_splits`, `max_train_steps`, `num_workers`, and `sample_rate` to config.
- [ ] Add root `requirements.txt` for non-CUDA Python dependencies.

### Task 5: Verification and commit

- [ ] Run simplified tests with `python3 -m pytest tests/test_config.py tests/test_phonemes.py tests/test_stage1_candidates.py tests/test_stage1_librispeech.py -q`.
- [ ] Run both new scripts with `--help`.
- [ ] Commit the Stage I runner slice.
