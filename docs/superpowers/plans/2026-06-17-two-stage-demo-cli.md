# Two-Stage Demo CLI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a runnable two-stage inference CLI that uses trained Stage I and Stage II checkpoints.

**Architecture:** Keep the first demo CLI self-contained and conservative. It rebuilds the same model shapes used by the training scripts, performs Stage I greedy CTC decoding, finds approximate keyword candidates, then scores candidate audio spans with the Stage II QbyT checkpoint.

**Tech Stack:** Python, PyTorch/torchaudio on the remote GPU machine, existing `qbyt` Conformer/QbyT modules.

---

## Tasks

### Task 1: Demo CLI

- [ ] Implement `scripts/run_two_stage_demo.py` with lazy torch imports.
- [ ] Support `--config`, `--stage1-ckpt`, `--stage2-ckpt`, `--audio`, `--keyword`, and `--vocab`.
- [ ] Print JSON containing keyword phonemes, Stage I candidates, Stage II scores, and detected.

### Task 2: Verification and commit

- [ ] Verify `python3 scripts/run_two_stage_demo.py --help` runs locally.
- [ ] Run simplified tests.
- [ ] Commit the demo CLI slice.
