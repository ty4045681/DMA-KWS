---
title: QbyT readout compatibility implementation plan
description: Restore v2-v7 QbyT readouts as first-class families with era-aware checkpoint I/O, losses, and eval defaults.
tags:
  - qbyt
  - stage2
  - checkpoint
  - compatibility
---
# QbyT Readout Compatibility Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans (this work is tightly coupled; do not fan out independent subagents that edit the same factory/checkpoint files). Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Stage II load, eval, resume, adapt, and train v2-v7 QbyT readouts in one tree, reproducing each era's score, without silently mixing formulas.

**Architecture:** Three first-class QbyT families restored from git (pooling v2-v4, bounded v5, keyword-filler v6/v7). A version discriminator on Hydra plus era-specific spec classes decode checkpoints exactly as they were written. `build_qbyt` is the only constructor. Losses, LoRA targets, and clip-eval padding follow the family. v1 stays refused.

**Tech Stack:** PyTorch, Hydra structured configs, existing Stage II Lightning module, pytest.

## Global Constraints

- Restore model/loss/spec/LoRA inject code from git. Do not reimplement scoring. SHAs: pooling `79628e862a2f1a88eb7dc080bb9be65793a1d0a5`, bounded `f0e261af2d96219676401bd35501218a6191ccbc`, v6 filler ablation already in `dma_kws/inference/qbyt_diagnostics.py`.
- `qbyt/` must not import `dma_kws`.
- All families expose `forward(speech, text, speech_lengths=None, text_lengths=None) -> (utterance_logits [B], seq_logits [B, U])`.
- Do not add `emission` to stamped `qbyt_alignment_spec`. v6 vs v7 is the version integer.
- Stamp the run's actual version, never a hardcoded 7 onto pooling/v5/v6 weights.
- Strict match: checkpoint spec must equal Hydra spec. Pooling keeps 79628e's v2/v3/v4 same-score compatibility. v5, v6, v7 require equal version numbers. No `allow_legacy_qbyt_readout` cross-family warm start.
- `CURRENT_QBYT_READOUT_VERSION = 7`. `SUPPORTED_QBYT_READOUT_VERSIONS = {2, 3, 4, 5, 6, 7}`.
- New runs default to version 7. Missing yaml version infers 4 / 5 / 7 as designed; v6 must be explicit.
- Do not commit unless the user asks.
- Verify with `.venv/bin/python -m pytest tests -q`. Ignore the six optional-dep failures listed in `Agents.md`.

---

## File map

**Create (restored, then path-adjusted):**

- `qbyt/pooling.py` from `79628e:qbyt/model.py` (drop `if __name__` demo).
- `qbyt/bounded.py` from `f0e261a:qbyt/model.py` (import aligner from `qbyt.bounded_alignment`).
- `qbyt/bounded_alignment.py` from `f0e261a:qbyt/monotonic_alignment.py`.
- `dma_kws/stage2/scoring.py` from `79628e:dma_kws/stage2/scoring.py`.
- `dma_kws/stage2/readout_pooling.py` from `79628e:dma_kws/stage2/readout.py`.
- `dma_kws/stage2/readout_bounded.py` from `f0e261a:dma_kws/stage2/readout.py`.
- `tests/test_qbyt_pooling.py` from `79628e:tests/test_qbyt_model.py`.
- `tests/test_qbyt_bounded.py` from `f0e261a:tests/test_qbyt_model.py`.
- `tests/test_bounded_alignment.py` from `f0e261a:tests/test_monotonic_alignment.py`.
- `tests/test_stage2_losses_pooling.py` from `79628e:tests/test_stage2_losses.py`.

**Modify:**

- `qbyt/model.py` - add `emission` in `{one_vs_rest, query_relative}`.
- `dma_kws/stage2/readout.py` - facade over pooling/bounded/current specs.
- `dma_kws/stage2/model_factory.py` - dispatch `build_qbyt`.
- `dma_kws/stage2/losses.py` - keep v6/v7; add pooling/v5 entry points restored from git.
- `dma_kws/stage2/module.py` - resolve spec, dispatch losses, freeze unused pooling heads.
- `dma_kws/stage2/objective.py` - era-aware `sequence_loss` including pooling `completion_weight` and `LEGACY_SEQUENCE_OBJECTIVE`.
- `dma_kws/training/checkpoint_io.py` - era-aware stamp/decode/assert.
- `dma_kws/training/checkpoint_convert.py`, `checkpoint_avg.py` - same-family only; convert preserves original version.
- `dma_kws/training/lora.py` - pooling inject from 79628e plus current v5+ targets.
- `dma_kws/configs/schema.py`, `configs/stage2/default.yaml` - `qbyt_readout_version: 7`, restore `qbyt_readout`, make `qbyt_alignment` version-gated.
- `dma_kws/inference/stage2_reporting.py`, clip eval scripts - padding default and provenance.
- `tests/test_qbyt_readout.py`, `tests/test_checkpoint_io.py`, convert/avg tests, `AGENTS.md`.

---

### Task 1: Restore pooling QbyT (v2-v4)

**Files:**
- Create: `qbyt/pooling.py`
- Create: `tests/test_qbyt_pooling.py`

**Interfaces:**
- Produces: `qbyt.pooling.QbyT` with `forward(...) -> (utterance_logits, seq_logits)` and `forward_with_readout_details(...)` as in 79628e.

- [ ] **Step 1: Extract and path-adjust**

```bash
git show 79628e862a2f1a88eb7dc080bb9be65793a1d0a5:qbyt/model.py > qbyt/pooling.py
```

Remove the `if __name__ == "__main__"` demo. Keep local `GRU_LAST_READOUT` / `EPS_*` constants (qbyt must not import dma_kws). Public `forward` already returns two tensors.

- [ ] **Step 2: Port 79628e model tests**

Copy `79628e:tests/test_qbyt_model.py` to `tests/test_qbyt_pooling.py`. Import `QbyT` from `qbyt.pooling`. Keep numerical padding / EPS / GRU tests unchanged.

- [ ] **Step 3: Run**

```bash
.venv/bin/python -m pytest tests/test_qbyt_pooling.py -q
```

Expected: PASS.

---

### Task 2: Restore bounded QbyT (v5)

**Files:**
- Create: `qbyt/bounded_alignment.py`, `qbyt/bounded.py`
- Create: `tests/test_bounded_alignment.py`, `tests/test_qbyt_bounded.py`

**Interfaces:**
- Produces: `qbyt.bounded.QbyT` using `qbyt.bounded_alignment.BoundedSegmentalAligner` (not the current v6 DP).

- [ ] **Step 1: Extract aligner snapshot**

```bash
git show f0e261af2d96219676401bd35501218a6191ccbc:qbyt/monotonic_alignment.py > qbyt/bounded_alignment.py
git show f0e261af2d96219676401bd35501218a6191ccbc:qbyt/model.py > qbyt/bounded.py
```

In `bounded.py` change the aligner import to `from qbyt.bounded_alignment import BoundedSegmentalAligner`. Leave current `qbyt/monotonic_alignment.py` untouched.

- [ ] **Step 2: Port v5 tests**

Copy `f0e261a:tests/test_monotonic_alignment.py` to `tests/test_bounded_alignment.py` (import from `qbyt.bounded_alignment`). Copy `f0e261a:tests/test_qbyt_model.py` to `tests/test_qbyt_bounded.py` (import `qbyt.bounded.QbyT`).

- [ ] **Step 3: Run**

```bash
.venv/bin/python -m pytest tests/test_qbyt_bounded.py tests/test_bounded_alignment.py -q
```

Expected: PASS. Current `tests/test_monotonic_alignment.py` and `tests/test_qbyt_model.py` still pass.

---

### Task 3: v6/v7 emission switch on current QbyT

**Files:**
- Modify: `qbyt/model.py`
- Modify: `tests/test_qbyt_model.py`

**Interfaces:**
- Consumes: `query_relative` formula from `dma_kws/inference/qbyt_diagnostics.py::_target_llr`.
- Produces: `QbyT(..., emission="one_vs_rest"|"query_relative")`. Default `one_vs_rest` (v7).

- [ ] **Step 1: Write failing test**

Same weights, `emission="query_relative"` vs `"one_vs_rest"` must disagree on a near-miss-like batch (two query phones that overlap inventory). Illegal `emission` raises `ValueError`.

- [ ] **Step 2: Implement**

Store `self.emission`. In `_encode_lattice`, branch with the diagnostics `query_relative` vs current one-vs-rest code. Do not change parameter names or shapes.

- [ ] **Step 3: Run**

```bash
.venv/bin/python -m pytest tests/test_qbyt_model.py -q
```

Expected: PASS. Existing v7 numerical tests stay on the default emission.

---

### Task 4: Era-aware readout spec + Hydra

**Files:**
- Create: `dma_kws/stage2/readout_pooling.py`, `dma_kws/stage2/readout_bounded.py`
- Modify: `dma_kws/stage2/readout.py`, `dma_kws/configs/schema.py`, `configs/stage2/default.yaml`
- Modify: `tests/test_qbyt_readout.py`

**Interfaces:**
- Produces: `resolve_qbyt_score_spec(stage2_config) -> (version: int, spec)` where spec is `QbyTReadoutConfig` (v2-4), v5 `QbyTAlignmentSpec`, or current v6/v7 `QbyTAlignmentSpec`.
- `qbyt_readout_specs_equal` compares within one era only.

- [ ] **Step 1: Restore spec modules from git**

```bash
git show 79628e862a2f1a88eb7dc080bb9be65793a1d0a5:dma_kws/stage2/readout.py > dma_kws/stage2/readout_pooling.py
git show f0e261af2d96219676401bd35501218a6191ccbc:dma_kws/stage2/readout.py > dma_kws/stage2/readout_bounded.py
```

Keep current `readout.py` classes as the v6/v7 spec. Turn `readout.py` into the facade: infer/require `qbyt_readout_version`, dispatch, reject inactive-section conflicts (`qbyt_readout` as a v7 topology still errors).

Missing version inference: `qbyt_readout` present without keyword-filler/bounded topology -> 4; `bounded_segmental_v1` -> 5; else -> 7. Training v6 requires explicit `6`.

- [ ] **Step 2: Hydra**

Add `qbyt_readout_version: int = 7` and restore `Stage2QbyTReadoutConfig` from 79628e. Do not keep a single alignment dataclass that injects v7 `weakest_phone_*` into v5 runs. Validate `qbyt_alignment` as a mapping through the era spec (exact fields).

Default yaml adds `qbyt_readout_version: 7` and leaves the current `qbyt_alignment` block.

- [ ] **Step 3: Replace refusal tests**

`gru_last` / `eps_*` resolve when version is 2-4. `bounded_segmental_v1` resolves when version is 5. Those topologies still fail at version 7. `QBYT_READOUT_VERSION == 7` becomes current-default plus supported set.

- [ ] **Step 4: Run**

```bash
.venv/bin/python -m pytest tests/test_qbyt_readout.py tests/test_config.py -q
```

---

### Task 5: Checkpoint I/O

**Files:**
- Modify: `dma_kws/training/checkpoint_io.py`
- Modify: `tests/test_qbyt_readout.py`, `tests/test_checkpoint_io.py`

**Interfaces:**
- `stamp_qbyt_readout_version(payload, *, spec)` writes `payload[QBYT_READOUT_VERSION_KEY] = spec.version`. Pooling does not write `qbyt_alignment_spec`. v5+ writes era `as_dict()`.
- `checkpoint_qbyt_readout_spec` uses 79628e decode for 2/3/4 (including head-key inference) and exact-field alignment specs for 5/6/7.
- `assert_qbyt_readout_version(..., expected=...)` succeeds iff decode matches expected. v6 vs v7 never match.

- [ ] **Step 1: Failing tests first**

Port 79628e readout checkpoint tests. Change `test_readout_version_6_qbyt_checkpoint_is_refused_despite_a_valid_spec` so v6+v6 config passes and v6+v7 config still exits. Stamp of a pooling spec must not write alignment spec or version 7.

- [ ] **Step 2: Implement decode/stamp/assert**

Rename constant to keep `QBYT_READOUT_VERSION = 7` as the current default alias so existing imports compile, and add `SUPPORTED_QBYT_READOUT_VERSIONS`.

- [ ] **Step 3: Run**

```bash
.venv/bin/python -m pytest tests/test_qbyt_readout.py tests/test_checkpoint_io.py -q
```

---

### Task 6: Factory, losses, module, LoRA

**Files:**
- Modify: `dma_kws/stage2/model_factory.py`, `losses.py`, `module.py`, `objective.py`, `dma_kws/training/lora.py`, `dma_kws/stage2/adapt.py`
- Create: `dma_kws/stage2/scoring.py`, `tests/test_stage2_losses_pooling.py`
- Modify: `tests/test_stage2_losses.py`, `tests/test_lora.py`, `tests/test_stage2_module.py`

**Interfaces:**
- `build_qbyt(stage2_cfg, *, input_dim, vocab_size)` constructs pooling / bounded / current QbyT from the resolved spec. Version 6 -> `emission="query_relative"`; 7 -> `one_vs_rest`.
- Pooling losses: restore 79628e `compute_stage2_losses` (completion BCE, all valid positions) via scoring helpers.
- v5 losses: 79628e-less completion, drop last prefix, no `valid_path_mask` / negative-tail.
- v6/v7: current `compute_stage2_losses`.
- `negative_tail` / `background_negative` enabled on pooling or v5 raises.
- Pooling unused heads (`gru` vs `final_pos_fc`) get `requires_grad=False`.
- LoRA: 79628e `phone_matchor` inject when pooling; current `audio_key`/`text_query` otherwise. Default targets follow family.

- [ ] **Step 1: Restore scoring + pooling losses tests from 79628e, watch them fail against current losses**
- [ ] **Step 2: Dispatch in factory/module/lora**
- [ ] **Step 3: Run**

```bash
.venv/bin/python -m pytest tests/test_stage2_losses.py tests/test_stage2_losses_pooling.py tests/test_lora.py tests/test_stage2_module.py tests/test_qbyt_pooling.py tests/test_qbyt_bounded.py tests/test_qbyt_model.py -q
```

---

### Task 7: Eval, convert/avg, provenance, docs

**Files:**
- Modify: `scripts/eval_stage2_clips.py` (and other clip-eval DEFAULT_PADDING_MS sites), `dma_kws/inference/stage2_reporting.py`, `dma_kws/inference/score_provenance.py`, `dma_kws/training/checkpoint_convert.py`, `dma_kws/training/checkpoint_avg.py`, `dma_kws/training/callbacks.py`, `dma_kws/training/metrics_history.py`, `AGENTS.md`
- Modify matching tests.

**Interfaces:**
- Clip eval default padding: 160 ms iff resolved checkpoint/config version is 6 and yaml omitted the keys; else 0. Explicit yaml wins. MUSAN paths stay 0.
- Provenance: pooling records `qbyt_readout` + version; v5+ records `qbyt_alignment` + version. Bump provenance schema version.
- Convert preserves original readout version (79628e behavior) and refuses cross-family.
- Avg requires equal decoded specs.
- Emission probe refuses pooling/v5.
- `AGENTS.md`: supported versions, stamp-the-actual-version, v6 vs v7 same shapes.

- [ ] **Step 1: Tests for v6 default 160 / v7 default 0 / convert preserve v2 / avg mixed-version refuse**
- [ ] **Step 2: Implement**
- [ ] **Step 3: Full suite**

```bash
.venv/bin/python -m pytest tests -q
```

Expected: only the six optional-dep failures listed in `Agents.md`.

---

## Spec coverage

- v2-v7 load/eval/train/adapt: Tasks 1-6.
- Strict match + pooling v2/v3/v4 same-score: Task 5.
- No new stamp fields / no emission in alignment spec: Tasks 3-5.
- Losses/LoRA/padding by family: Tasks 6-7.
- v1 refused: Task 5.
- Numerical locks from git tests: Tasks 1-3, 6.
