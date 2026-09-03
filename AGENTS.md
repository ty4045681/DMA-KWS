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

`QBYT_READOUT_VERSION` (`CURRENT_QBYT_READOUT_VERSION`) is 7, the default for
new Stage II runs. `SUPPORTED_QBYT_READOUT_VERSIONS` is `{2, 3, 4, 5, 6, 7}`.
Stamp the version that actually produced the weights; do not write 7 onto a
pooling or v6 checkpoint.

Loaders (`assert_qbyt_readout_version`) accept any supported checkpoint that
decodes. When a run config is supplied, pooling v2/v3/v4 may match on
mode/temperature; v5/v6/v7 require an equal version and spec. v6 and v7 share
parameter shapes but not the emission formula (query-relative vs one-vs-rest),
so a v6 file cannot be scored as v7.

Families live in `qbyt/pooling.py` (v2-v4), `qbyt/bounded.py` (v5), and
`qbyt/model.py` (v6/v7). `build_qbyt` is the only constructor. Checkpoints are
stamped by `Stage2LightningModule.on_save_checkpoint` and the explicit
`torch.save` calls in `dma_kws/stage2/train.py` and `dma_kws/stage2/adapt.py`.
Test fixtures that simulate a current-build checkpoint must call
`stamp_qbyt_readout_version`.

## QbyT batch invariance

Pooling `QbyT.forward` concatenates padded text and padded audio, so anything
expressed in per-sample lengths must account for the batch padding between the
two blocks. The sequence is re-packed to `[valid text][valid audio][padding]`
for exactly this reason. `tests/test_qbyt_pooling.py` and
`tests/test_qbyt_model.py` lock the invariants down numerically; a
structural/AST assertion cannot catch a wrong index.
