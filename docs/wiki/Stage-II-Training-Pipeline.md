# Stage II QbyT Training Pipeline

Verified, end-to-end recipe for getting from a bare GPU host to a **working single-step
Stage II QbyT training run** with an Icefall Zipformer encoder, and the measured
time budget for longer runs.

**Last verified:** 2026-10-01 — Ubuntu 22.04, 1× Tesla V100-SXM2-32GB (sm_70),
torch 2.7.1+cu126, torchaudio 2.7.1+cu126, k2 1.24.4 (cu126/torch2.7.1), cuDNN 9.5.1,
icefall master `3f848bb`, 10 CPU cores, 38 GB RAM, 512 GB data disk.

Environment creation itself is covered by [Environment Setup](Environment-Setup).

## 1. Data layout

Everything lives on the data disk; `data/dma-kws` is a symlink to it:

```
data/dma-kws -> /mnt/datadisk0/dma-kws
```

| Artifact | Path | Size / count |
| --- | --- | --- |
| GigaPhrase-1000 (GP audio only) | `raw/GigaPhrase-1000/` | 76 files, 79.00 GB, 71 decoded shards |
| LibriPhrase-460 (LP audio) | `raw/LibriPhrase-460/` | 66 files, 49.58 GB, 46 decoded shards |
| Combined decoded root (hardlinks) | `raw/LibriPhrase-GS-1460/decoded/` | 117 shards: GP `0000–0070`, LP-460 `1000–1045` |
| LibriPhrase eval set | `raw/LibriPhrase-100/eval/` | 136,467 files, 2.93 GB |
| MUSAN | `raw/musan/{music,noise,speech}` | 2,016 wav, 12 GB |
| LibriSpeech train-other-500 | `raw/LibriPhrase...` → `raw/LibriSpeech/train-other-500/` | 148,688 flac, 30 GB |
| Paper-format training parquet | `processed/stage2_qbyt/ls-gs-1460/aggregated_segments_with_g2p_distance.parquet` | 155,619 anchors / 5,837,186 clips |
| Training fbank (Icefall profile) | `features/fbank_icefall_kws/` | **5,837,186** `.npy`, 133 GB |
| Eval fbank | `features/fbank_icefall_kws_eval/` | 136,461 `.npy`, 3.0 GB |
| MUSAN split + catalog + cache | `processed/musan_split/`, `processed/background/musan/` | cache: 320,739 crops, 77 shards, 20 GB |

### 1.1 GigaPhrase-1000 is only half of the LS-GS-1460 finetune set

The HF dataset `ZhiqiAi/GigaPhrase-1000` ships **one aggregated parquet that describes the
combined LS-GS-1460 set** (GP-1000 **and** LP-460 clips) but **decoded shards that contain
only the GP-1000 audio**. Verified:

```text
71 decoded shards -> 3,548,973 distinct audio_rel   (== exactly the GP-1000 clip count)
3,000 sampled GP-1000 clips found in shards : 3,000 / 3,000
3,000 sampled LP-460 clips found in shards  : 0 / 3,000
```

Without `ZhiqiAi/LibriPhrase-460` the prep stops at ~61 % with
`missing_in_decoded = 2288213` (= the LP-460 clip count). Download both, then merge them
into one decoded root with non-colliding names (hardlinks, no extra disk):

```bash
cd data/dma-kws/raw
export HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 HF_HUB_DISABLE_PROGRESS_BARS=1
hf download ZhiqiAi/GigaPhrase-1000  --repo-type dataset --local-dir GigaPhrase-1000  --max-workers 8
hf download ZhiqiAi/LibriPhrase-460  --repo-type dataset --local-dir LibriPhrase-460  --max-workers 4

mkdir -p LibriPhrase-GS-1460/decoded
for f in GigaPhrase-1000/decoded/LP-100-decoded-*.parquet; do
  ln -f "$f" LibriPhrase-GS-1460/decoded/$(basename "$f")            # 0000–0070
done
i=0
for f in LibriPhrase-460/LP-100-decoded-*.parquet; do
  ln -f "$f" "LibriPhrase-GS-1460/decoded/LP-100-decoded-$(printf %04d $((1000+i))).parquet"
  i=$((i+1))
done                                                            # 1000–1045
```

Both repositories name their shards `LP-100-decoded-*.parquet`, hence the renumbering.

### 1.2 Paper parquet + fbank (Icefall profile)

```bash
python3 scripts/prepare_stage2_paper.py \
  +experiment=icefall_zipformer_stage2 \
  prep.input_parquet=data/dma-kws/raw/GigaPhrase-1000/aggregated_segments_with_g2p_distance.parquet \
  prep.decoded_parquet_root=data/dma-kws/raw/LibriPhrase-GS-1460/decoded \
  prep.output_subdir=stage2_qbyt/ls-gs-1460 \
  prep.num_workers=8
```

Result (measured):

```text
anchors: 155619   clips_total: 5837186
fbank_written: 2288213   fbank_skipped: 3548973
missing_in_decoded: 0
```

`features/fbank_icefall_kws/` then holds exactly 5,837,186 `.npy` files (133 GB).
`configs/fbank/icefall_kws.yaml` reproduces icefall's on-the-fly extractor exactly
(`Fbank(FbankConfig(num_mel_bins=80))` = lhotse defaults: 16 kHz, 25/10 ms, povey,
dither 0.0, `snip_edges=False`, low_freq 20, high_freq −400); a bit-exact comparison on a
real clip gave max |diff| = 0.0.

Do **not** point `stage2.wav_dir` at a tree produced with the default Wenet profile — the
frame count and value scale differ (snip_edges and the 2^15 factor).

### 1.3 Eval set (required — training refuses to start without it)

The eval CSVs are not in either ZhiqiAi repository. `E4rris/LibriPhrase_evalset`
(a verbatim copy of the official LibriPhrase test data, per its README) has exactly the
expected layout:

```bash
hf download E4rris/LibriPhrase_evalset --repo-type dataset \
  --local-dir data/dma-kws/raw/LibriPhrase-100/eval --max-workers 32
python3 scripts/prepare_stage2_eval_fbank.py +experiment=icefall_zipformer_stage2 prep.from_csv=true
```

136,461 wav → 136,461 eval fbank files (`written=136461, skipped=0, failed=0`).
The 13.6k-file download is per-file latency bound: 8 workers ≈ 1.8 files/s, 32 workers
trips HTTP 429 on huggingface.co. A single `aria2c -j 16` over a filtered URL list with
`--max-tries=50 --retry-wait=20 --save-session` ran at ~15 files/s; re-running the loop
until the session file is empty is the robust pattern.

## 2. icefall + k2 (the part that bites)

```bash
git clone --depth 1 https://github.com/k2-fsa/icefall.git /home/ubuntu/icefall
```

### 2.1 k2 has no Linux wheels on PyPI

The official docs (`k2-fsa.github.io/k2/installation`) point at pre-compiled CUDA wheels
hosted on HuggingFace, per torch version. Pick the wheel matching **torch, CUDA and
Python** and install it via `hf-mirror` (no GitHub/k2-fsa index needed):

```bash
W=k2-1.24.4.dev20260625+cuda12.6.torch2.7.1-cp310-cp310-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl
curl -L -o "$W" "https://hf-mirror.com/csukuangfj2/k2/resolve/main/ubuntu-cuda/1.24.4.dev20260625/$W"
uv pip install --python .venv/bin/python --no-deps "$W"
```

That wheel works on the V100 (sm_70 kernels present): `k2.swoosh_l_forward` matched a
plain torch implementation with max |diff| 0.0 on CPU and 1.5e-8 GPU-vs-CPU.

Also required by icefall's import chain: `kaldialign`, `sentencepiece`, `pypinyin`
(`uv pip install ...`).

### 2.2 cuDNN: force the pip 9.5.1 or V100 convs fail

With the image's default cuDNN (torch reports **9.2.0**) the Zipformer `Conv2dSubsampling`
raises on GPU:

```text
RuntimeError: GET was unable to find an engine to execute this computation
```

`TORCH_CUDNN_V8_API_DISABLED=1` and `cudnn.benchmark=True` do **not** help;
`cudnn.enabled=False` works but is slow. The fix is to force the pip-installed cuDNN
9.5.1 via `LD_LIBRARY_PATH` (torch then reports 90501 and forward+backward succeed):

```bash
export LD_LIBRARY_PATH=/home/ubuntu/dma-kws/.venv/lib/python3.10/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH
```

### 2.3 PYTHONPATH must contain the repository root

The v4.1 `qbyt/` code uses package-qualified imports (`from qbyt.monotonic_alignment
import ...`), while `dma_kws/pathing.py` only puts `qbyt/` itself on `sys.path`.
Running a script directly therefore fails with `ModuleNotFoundError: No module named
'qbyt'` — the README's `PYTHONPATH=.` workaround is required:

```bash
export PYTHONPATH=/home/ubuntu/icefall:/home/ubuntu/dma-kws:$PYTHONPATH
```

### 2.4 One env file for all of it

`~/.dma-kws-env.sh` (sourced by `~/.bashrc` **and** `~/.profile`, so login,
non-interactive and interactive shells all get it):

```bash
export ICEFALL_ROOT=/home/ubuntu/icefall
export ICEFALL_CHECKPOINT=/home/ubuntu/dma-kws/data/dma-kws/raw/kws-checkpoints/gigaspeech-20240219/exp-finetune/pretrained.pt
export PYTHONPATH=/home/ubuntu/icefall:/home/ubuntu/dma-kws:${PYTHONPATH:-}
export LD_LIBRARY_PATH=/home/ubuntu/dma-kws/.venv/lib/python3.10/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}
```

## 3. Encoder checkpoints

| Checkpoint | Path | Status |
| --- | --- | --- |
| GigaSpeech KWS finetune | `raw/kws-checkpoints/gigaspeech-20240219/exp-finetune/pretrained.pt` | loads into `icefall_zipformer_stage2` with `encoder_embed missing=0 unexpected=0`, `encoder missing=0 unexpected=0` |
| GigaSpeech KWS base | `.../gigaspeech-20240219/exp/pretrained.pt` | same |
| zh-en 3M (bilingual) | `raw/kws-checkpoints/zh-en-3M-2025-12-20/pretrained-epoch-13-avg-2.pt` | needs `cnn_module_kernel="15,15,15,15,15,15"`; then 0/0 |

The zh-en checkpoint's own `config.json` (HuggingFace conversion of the original
ModelScope release) records `cnn_module_kernel [15×6]`, `encoder_chunk_size 16`,
`left_context_frames 64`; every other encoder hyper-parameter equals the GigaSpeech
preset. Its sha256 (`62188d07…f6eb`) matches the hash documented upstream.

A ready-to-run overlay is committed as `configs/experiment/icefall_zipformer_stage2_zhen3m.yaml`
(only the kernel differs, plus run/checkpoint names and the prepared parquet paths).

## 4. Verified single-step smoke

```bash
CUDA_VISIBLE_DEVICES=0 python3 scripts/train_stage2_qbyt.py \
  +experiment=icefall_zipformer_stage2 \
  stage2.parquet_file=data/dma-kws/processed/stage2_qbyt/ls-gs-1460/aggregated_segments_with_g2p_distance.parquet \
  stage2.precision=16-mixed \
  run.limit_steps=1 run.devices=1
```

Observed on the V100 (`rc=0`, 38 s wall, startup ≈ 25 s):

```text
Loaded encoder_embed weights ...: missing=0 unexpected=0
Loaded encoder weights ...      : missing=0 unexpected=0
GPU available: True (cuda), used: True | precision: 16-mixed
Gradient diagnostics at step 0: all trainable parameters have gradients
Trainer.fit stopped: max_steps=1 reached
metrics.csv: loss_total=20.16  loss_utt_raw=17.26  loss_seq_weighted=2.90  illegal_path_rate=0.0078
artifacts: stage2_step000001.pt (12 MB, qbyt_readout_version=7), last.ckpt, metrics.csv, hparams.yaml, tfevents
```

The same smoke passes for the two overlays:

* `+experiment=icefall_zipformer_stage2_eps_softmin_v41_musan_cached`
  (v4.1 readout, MUSAN fbank-cache backgrounds) — giga ckpt 0/0, rc=0;
* `+experiment=icefall_zipformer_stage2_zhen3m` — zh-en ckpt 0/0, rc=0.

`stage2.precision=16-mixed` is required on V100 (the preset defaults to `bf16-mixed`,
which has no hardware support there).

## 5. Measured throughput and time budget

Measured with `icefall_zipformer_stage2` on **1× V100** — batch 128/GPU,
`accumulate_grad_batches=1`, `precision=16-mixed`, frozen Zipformer + QbyT,
`num_workers=4` (from `logs/.../runs.csv` and `eval_history.csv`):

| Quantity | Measured |
| --- | --- |
| Training | **3.9 s/step** (`duration_s=194.8` for 50 steps) ≈ 924 steps/h |
| Validation (2,048-sample sample, 16 batches) | 31.1 s → **1.94 s/batch** |
| Full hard-split validation (270,684 rows = 2,115 batches) | **≈ 69 min** |

Validation is data-loading bound (per-item G2P + fbank `.npy` read with 4 workers), not
GPU bound.

Projection for **50,000 steps**:

| `stage2.validation.val_check_interval` | validations | train | validation | total |
| --- | --- | --- | --- | --- |
| 500 (preset default) | 100 | 54.1 h | 114.2 h | **168 h ≈ 7.0 days** |
| 1000 | 50 | 54.1 h | 57.1 h | 111 h ≈ 4.6 days |
| 2000 | 25 | 54.1 h | 28.5 h | 83 h ≈ 3.4 days |
| 5000 | 10 | 54.1 h | 11.4 h | 66 h ≈ 2.7 days |
| no validation | 0 | 54.1 h | — | 54 h ≈ 2.3 days |

Levers, in order of value: raise `val_check_interval`; raise `stage2.eval.num_workers`
(4 → 16; validation is CPU bound); evaluate offline after training with
`scripts/batch_eval_stage2_clips.sh`; add GPUs (`run.devices=2` halves both training and
the sharded validation); consider batch 256 (memory allows — ~12 GB used of 32 GB).

## 6. MUSAN backgrounds (optional, v4.1)

```bash
# 1) split MUSAN into train/eval lists (60 % train by duration)
python3 scripts/split_musan.py --musan-root data/dma-kws/raw/musan \
  --output-dir data/dma-kws/processed/musan_split

# 2) catalog (needs an eligibility list; the MUSAN adapter marks nothing eligible by itself)
python3 scripts/prepare_background_sources.py --config <prepare.yaml>

# 3) fbank crop cache (builder prints its own capacity estimate; K=323 was suggested for
#    the default 10k steps x 128 batch x 0.25 probability ≈ 160k background draws)
python3 scripts/prepare_stage2_background.py \
  --source-manifest data/dma-kws/processed/background/musan/recordings.jsonl \
  --output-dir data/dma-kws/processed/background/musan/cache \
  --experiment icefall_zipformer_stage2_eps_softmin_v41_multisource_cached \
  --crops-per-recording 323 --seed 2025 --workers 8

python3 scripts/prepare_stage2_background.py --verify-only \
  --output-dir data/dma-kws/processed/background/musan/cache     # -> "ok": true
```

Measured: split train 1,114 / eval 902 recordings; catalog 993 train / 121 val / 902 test;
cache **320,739 crops in 77 shards, 20 GB**, `--verify-only` → `"ok": true`.
MUSAN-only overlay: `configs/experiment/icefall_zipformer_stage2_eps_softmin_v41_musan_cached.yaml`.
DNS / FSD50K are **not** downloaded, so the three-source overlay is not runnable yet.

`eval_musan.list` (902 recordings) doubles as a held-out MUSAN allowlist for FA evaluation
(`prep.musan_audio_list_path`). For speech-background FA/hour the same knob accepts any
audio list — e.g. `raw/LibriSpeech/train-other-500.list` (148,688 flac paths, validated
with `dma_kws.inference.manifest.load_audio_file_list`).

## 7. Troubleshooting

| Symptom | Cause / fix |
| --- | --- |
| `ModuleNotFoundError: No module named 'qbyt'` | put the repo root on `PYTHONPATH` (§2.3) |
| `No module named 'k2'` / `kaldialign` / `sentencepiece` / `pypinyin` | install the matching k2 wheel + the three packages (§2.1) |
| `GET was unable to find an engine to execute this computation` on GPU | cuDNN 9.2 from the image; force pip cuDNN 9.5.1 (§2.2) |
| `size mismatch ... depthwise_conv.causal_conv` | wrong `cnn_module_kernel` for the checkpoint (15 vs 31) |
| `missing_in_decoded` ≈ 2,288,213 | LP-460 audio missing; add the LibriPhrase-460 shards (§1.1) |
| Validation never finishes | full hard split is 270,684 rows ≈ 69 min on 1 GPU; schedule it deliberately (§5) |
| `bf16-mixed` on V100 | use `stage2.precision=16-mixed` |
| HTTP 429 while downloading 100k small files | `aria2c -j 16` + filtered list + session file (§1.3) |

## 8. Artifact checklist

```text
raw/kws-checkpoints/                    encoder .pt files (+ word lists)
raw/LibriPhrase-GS-1460/decoded/        117 hardlinked shards (GP + LP-460)
raw/LibriPhrase-100/eval/               LibriPhrase eval wavs + evaluation_set/*.csv
raw/musan/                              MUSAN tree
raw/LibriSpeech/train-other-500/        FA evaluation audio
processed/stage2_qbyt/ls-gs-1460/       paper parquet + clips/ + distances/
processed/musan_split/                  train_background.list / eval_musan.list
processed/background/musan/             recordings.jsonl + cache/ (77 shards)
features/fbank_icefall_kws/             5,837,186 training fbank .npy
features/fbank_icefall_kws_eval/        136,461 eval fbank .npy
configs/experiment/icefall_zipformer_stage2_zhen3m.yaml
configs/experiment/icefall_zipformer_stage2_eps_softmin_v41_musan_cached.yaml
```
