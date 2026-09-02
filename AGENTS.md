# Agent notes

## Verification

```bash
.venv/bin/python -m pytest tests -q      # or: bash scripts/run_smoke.sh
```

`scripts/run_smoke.sh` runs `set -e`, so it aborts at the first pytest failure and
never reaches its script-compile checks. Run the compile step yourself when the
suite is already red.

### Tests that fail without the optional dependencies

Six tests need extras that a lightweight local venv usually lacks. They fail on a
clean tree too, so do not chase them:

| Test | Missing |
| --- | --- |
| `test_cmvn.py::test_build_encoder_passes_cmvn_and_wenet_options` | `openai-whisper` (via `qbyt/models/encoder.py`) |
| `test_stage1_wenet_ctc.py::test_export_stage1_encoder_pt_roundtrip` | `openai-whisper` |
| `test_streaming_search.py::test_decode_keyword_candidates_smoke_without_gpu` | `openai-whisper` |
| `test_stage2_module.py::test_freeze_encoder_disables_encoder_gradients` | `transformers` (via `dma_kws/training/scheduler.py`) |
| `test_musan_fa.py::test_run_file_windows_counts_and_spans` | `openai-whisper` |
| `test_stage2_verifier.py::test_stage2_verifier_resamples_before_candidate_slicing` | `lhotse` (`pip install -e ".[icefall]"`) |

Confirm a failure is pre-existing with `git stash push <file>` before assuming you
caused it.

## QbyT readout version

`QBYT_READOUT_VERSION` in `dma_kws/training/checkpoint_io.py` guards Stage II
checkpoints whose QbyT weights were trained against a different pooled readout.
Parameter shapes do not change across such a fix, so nothing else would notice.

If you change what `QbyT.forward` reads out, bump the constant. Checkpoints are
stamped by `Stage2LightningModule.on_save_checkpoint` and the explicit
`torch.save` calls in `dma_kws/stage2/train.py` and `dma_kws/stage2/adapt.py`;
loaders check it via `assert_qbyt_readout_version`. Test fixtures that simulate a
current-build checkpoint must call `stamp_qbyt_readout_version`.

Version 7 replaced the query-relative filler with one-vs-rest per-phone emission
log-odds (`target_llr = log p_u - log(1 - p_u)`). Parameter shapes are identical
to version 6, but 6-era checkpoints score differently on the same weights and
are refused by the guard; they must be retrained.

## QbyT batch invariance

`QbyT.forward` concatenates padded text and padded audio, so anything expressed in
per-sample lengths must account for the batch padding between the two blocks. The
sequence is re-packed to `[valid text][valid audio][padding]` for exactly this
reason. `tests/test_qbyt_model.py` locks the invariants down numerically; a
structural/AST assertion cannot catch a wrong index.
