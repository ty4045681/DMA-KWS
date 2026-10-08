# Run On A New Machine

End-to-end route from a bare V100 box to a working inference call and a training run, using
only this repository plus the checkpoint release.

**Last verified:** 2026-10-08 — Ubuntu 22.04, Python 3.10.12, 1x Tesla V100-SXM2-32GB,
driver 580 (CUDA 13.0), torch 2.7.1+cu126, icefall checkout at /home/ubuntu/icefall. Both
inference commands in section 4 were run end to end on that machine with weights fetched
from the release.

## 0. What ships and what does not

| Item | Where |
| --- | --- |
| code, configs, tests | this repository |
| per-run training config snapshots | `run_snapshots/` |
| weights: 453 files, 9.22 GiB | GitHub release [checkpoints-2026-10-08](https://github.com/ty4045681/DMA-KWS/releases/tag/checkpoints-2026-10-08), indexed by `checkpoints/manifest.tsv` |
| datasets, prepared parquet, fbank features | not shipped; download and prepare (section 6) |
| icefall + k2 | clone/install separately (section 2) |

There is **no self-trained Stage I phoneme CTC checkpoint** in the release. The released
Stage I model is the author's paper-original `exp/author_v1/stage1_v1.pt` (written by
`scripts/import_author_v1_checkpoints.py`); `raw/kws-checkpoints/paper-stage1/ls-gs-1460.pt`
is only an encoder, not a CTC model.

## 1. Environment

Follow [Environment Setup](Environment-Setup) for the uv commands, the cu126 pin and the V100
traps. The three that bite:

* install **cu126** wheels explicitly. On a V100, `--torch-backend=auto` can select cu130,
  which dropped sm_70, and every kernel then fails with "no kernel image is available";
* `pip install -e ".[icefall]"` brings lhotse 1.33 but **not** icefall/k2, and cuDNN 9.5.1
  is required or V100 convolutions fail — see [Stage II Training Pipeline](Stage-II-Training-Pipeline) §2;
* `PYTHONPATH` must contain the repository root; the vendored `qbyt/` and `wenet/`
  are namespace packages.

```bash
git clone https://github.com/ty4045681/DMA-KWS && cd DMA-KWS
uv venv --python 3.10 .venv
uv pip install --python .venv/bin/python torch==2.7.1 torchaudio==2.7.1 \
  --index-url https://download.pytorch.org/whl/cu126
uv pip install --python .venv/bin/python -e ".[icefall]"
export ICEFALL_ROOT=/path/to/icefall
export PYTHONPATH=$ICEFALL_ROOT:$PWD
```

## 2. Fetch the weights

```bash
python scripts/checkpoint_release.py fetch --list            # 453 rows: asset, path, size
python scripts/checkpoint_release.py fetch --only final/     # delivery set (~0.66 GiB)
python scripts/checkpoint_release.py fetch --only author_v1  # author v1 pair (~38 MiB)
python scripts/checkpoint_release.py fetch                   # everything, 9.22 GiB
```

It restores into `data/dma-kws/`, verifies every sha256 with the manifest and skips files
that already match. The release is public, so no token is needed. `data/dma-kws/` is
created on demand (in the original workspace it was a symlink to a data disk).

## 3. Inference: Stage II only (cropped keyword clip) — verified

```bash
python scripts/checkpoint_release.py fetch --only final/C1-sink-50k

PYTHONPATH=.:$ICEFALL_ROOT ICEFALL_ROOT=$ICEFALL_ROOT python scripts/run_stage2_demo.py \
  +experiment=icefall_zipformer_stage2_eps_softmin_v41_musan_cached_zhen3m_50k \
  +stage2.qbyt_readout.sink_readout=additive \
  +stage2.qbyt_readout.sink_zero_init=true \
  prep.stage2_ckpt=data/dma-kws/exp/stage2_qbyt/final/C1-sink-50k/checkpoints/C1-sink-50k/version_0/stage2_step050000.pt \
  prep.audio=wenet/test/resources/librispeech-1995-1837-0001.wav \
  prep.keyword="hello world"
```

Expected output is JSON; with the LibriSpeech utterance above the keyword is absent, so
`"detected": false` and a low score are correct:

```json
{ "qbyt_score": 0.00043, "qbyt_raw_logit": -7.75, "threshold": 0.5, "detected": false }
```

The two `+` overrides are the readout spec this checkpoint carries; section 5 shows how to
read them for any checkpoint.

## 4. Inference: two-stage (long audio + keyword) — verified

```bash
python scripts/checkpoint_release.py fetch --only author_v1

PYTHONPATH=.:$ICEFALL_ROOT python scripts/run_two_stage_demo.py \
  +experiment=v1_paper_ls460 \
  prep.stage1_ckpt=data/dma-kws/exp/author_v1/stage1_v1.pt \
  prep.stage2_ckpt=data/dma-kws/exp/author_v1/stage2_v1_si.pt \
  prep.audio=wenet/test/resources/librispeech-1995-1837-0001.wav \
  prep.keyword="hello world"
```

`v1_paper_ls460` also selects the author's 73-symbol tokenizer
(`data/dict/lang_char_v1_73.txt`), which the v1 weights require.

A self-trained Stage I can **not** simply be combined with the icefall Stage II presets: those
presets override `/stage1` with `encoder_only`, so they describe the Stage II encoder and
not a CTC locator. Two-stage inference with a self-trained Stage I needs a new overlay that
carries both.

## 5. Matching a checkpoint to a preset

Two things must match, and both fail loudly instead of scoring with different semantics:

1. **encoder family.** `*_zhen3m*` presets set `cnn_module_kernel: 15` for the
   zh-en-3M encoder; the GigaSpeech KWS / ASR-XL / paper-stage1 encoders need the plain
   `icefall_zipformer_stage2*` presets (31 in stacks 0/1/5). A mismatch appears as
   `size mismatch for encoder...` during load.
2. **readout spec.** Read it straight from the checkpoint:

```bash
python scripts/checkpoint_release.py show-readout <path/to/checkpoint>
```

It prints the full spec plus ready-to-paste Hydra overrides. Prefix a key with `+` when the
preset does not define it (`sink_readout` and `sink_zero_init` are not in the base
`configs/stage2/default.yaml`).

`checkpoints/manifest.tsv` has a `readout` column for all 446 stamped checkpoints
(version distribution: 347x v4, 40x v3, 30x v2, 15x v1, 14x v7), plus `run` and
`snapshot` columns pointing at the producing run and its mirrored training config.

### Recommended checkpoints

| Purpose | Checkpoint | Preset | Extra overrides |
| --- | --- | --- | --- |
| delivery base (verified) | `.../final/C1-sink-50k/.../stage2_step050000.pt` | `icefall_zipformer_stage2_eps_softmin_v41_musan_cached_zhen3m_50k` | `+stage2.qbyt_readout.sink_readout=additive +stage2.qbyt_readout.sink_zero_init=true` |
| best general, R1 (verified) | `.../final/R1-unfreeze-hardneg100/.../stage2_step012500.pt` | same | same |
| Hey-Eva sink head | `.../sinkhead_stage2/SS-zh-en/.../stage2_step003000.pt` | same | same |
| Hey-Eva delivery (report §7) | `.../stage2_adapt_v42/c1_encfull/stage2_adapted.pt` | same | same |
| author paper v1 pair | `author_v1/stage1_v1.pt` + `author_v1/stage2_v1_si.pt` | `v1_paper_ls460` | none |
| pretrained encoders | `raw/kws-checkpoints/{zh-en-3M-2025-12-20,gigaspeech-20240219,gigaspeech-asr-xl-2023-10-17,paper-stage1}/...` | training init only | none |

Only the first two rows are freshly verified end to end here; the others carry the same readout
spec — confirm with `show-readout` before use.

**A Lightning `.ckpt` is not accepted by `prep.stage2_ckpt`.** Many release files are
`.ckpt`; either pick the run's exported `.pt` (`stage2_stepNNNNN.pt`,
`stage2_adapted.pt`) or export it first with `scripts/convert_stage2_checkpoints.py`
(README §8 shows the LoRA usage of the same converter).

## 6. Training

Minimum demo-scale data is LibriSpeech `train-clean-100` plus LibriPhrase-100 (README §3),
then the prep chains in README §4 (Stage I) and §6 (Stage II). The paper recipe needs
LibriSpeech-460, LibriPhrase-460 and GigaPhrase-1000 and is documented in
[docs/paper-reproduction.md](../paper-reproduction.md).

Verified smoke shape (it still needs the prepared parquet and fbank features — there is no
zero-data training smoke; `bash scripts/run_smoke.sh` only runs the unit tests):

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_stage2_qbyt.py \
  +experiment=icefall_zipformer_stage2 \
  stage2.parquet_file=data/dma-kws/processed/stage2_qbyt/ls-gs-1460/aggregated_segments_with_g2p_distance.parquet \
  stage2.precision=16-mixed \
  run.limit_steps=1 run.devices=1
```

On V100, `stage2.precision=16-mixed` is required: the presets default to `bf16-mixed`,
which has no hardware support there. The released encoders replace the external
`ICEFALL_CHECKPOINT` download, e.g.
`data/dma-kws/raw/kws-checkpoints/gigaspeech-20240219/exp-finetune/pretrained.pt`.

## 7. Known rough edges

* No overlay yet combines a **self-trained Stage I CTC model** with an icefall Stage II encoder,
  so two-stage inference on self-trained weights needs a new preset (section 4).
* `configs/experiment/ctc_adapter_icefall_accent.yaml` and
  `configs/experiment/adapt_hey_eva_icefall_adapter_v2.yaml` target the Chinese-accent
  dataset tree, which is not published; their manifests are now env-overridable
  (`CHINESE_ACCENT_ADAPTER_*_MANIFEST`, `HEY_EVA_ADAPT_MANIFEST`, `HEY_EVA_ADAPT_DATA_ROOT`).
* `dma_kws/data_prep/chinese_accent_english.py` deliberately writes the shared corpus'
  historical `/home/q00931063/...` prefix into generated manifests (provenance, so the tree
  can be copied without rewriting rows); override with `DMA_KWS_REMOTE_DATASET_ROOT` when
  regenerating for another host.
* `raw/kws-checkpoints/*` are raw encoder state dicts and carry no QbyT readout stamp, so
  their `readout` column is empty by design.
