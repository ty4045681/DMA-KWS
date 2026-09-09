# DMA-KWS

**DMA-KWS**: Effective User-defined Keyword Spotting with Dual-stage Matching, Multi-modal Enrollment, and Continual Adaptation.

This repository implements a two-stage keyword spotting pipeline with a **single algorithm and shared codebase**. Scale differences between the small demo and paper reproduction come from Hydra experiment presets and dataset size, not separate implementations.

1. **Stage I** (training + inference): phoneme CTC decoding proposes candidate keyword spans in audio.
2. **Stage II** (training + inference): QbyT phoneme matching verifies each candidate.
3. **Continual adaptation** (optional): LoRA-tune the Stage II phoneme matcher on user keyword data (TTS → real, LibriPhrase 1:1 anti-forgetting). See [§8](#8-stage-ii-lora-continual-adaptation).

## Table of contents

**Overview**

- [Terminology](#terminology)
- [Path conventions](#path-conventions)
- [Stage II training scripts](#stage-ii-training-scripts)
- [Continual adaptation (Stage II LoRA)](#continual-adaptation-stage-ii-lora)
- [Checkpoints and resume](#checkpoints-and-resume)
- [What is implemented now](#what-is-implemented-now)
- [Recommended hardware](#recommended-hardware)

**Workflow**

1. [Clone and create environment](#1-clone-and-create-environment)
2. [Prepare data directories](#2-prepare-data-directories)
3. [Download datasets](#3-download-datasets)
4. [Stage I: prepare phoneme CTC data](#4-stage-i-prepare-phoneme-ctc-data)
5. [Stage I: train phoneme CTC model](#5-stage-i-train-phoneme-ctc-model)
6. [Stage II: prepare training data](#6-stage-ii-prepare-training-data)
7. [Stage II: train QbyT verifier](#7-stage-ii-train-qbyt-verifier)
7b. [Stage II: initialize from external Wenet ASR encoder](#7b-stage-ii-initialize-from-external-wenet-asr-encoder)
7c. [Stage II: initialize from Icefall Zipformer encoder](#7c-stage-ii-initialize-from-icefall-zipformer-encoder)
7d. [Phoneme CTC adapter on a frozen encoder](#7d-phoneme-ctc-adapter-on-a-frozen-encoder)
8. [Stage II LoRA continual adaptation](#8-stage-ii-lora-continual-adaptation)
9. [Run the two-stage demo](#9-run-the-two-stage-demo)
10. [Stage II-only clip inference](#10-stage-ii-only-clip-inference)
11. [MUSAN false-accept evaluation](#11-musan-false-accept-evaluation)

**Reference**

- [External locator (Zipformer / WeKws+Wenet)](#external-locator-zipformer--wekwswenet)
- [Troubleshooting](#troubleshooting)
- [Datasets](#datasets)
- [Paper reproduction](#paper-reproduction)

### Terminology

| Term | Meaning |
|------|---------|
| **experiment** | Hydra config overlay selected on the CLI, e.g. `+experiment=demo_librispeech100` |
| **recipe** | `training.recipe` label for multi-phase Stage II chains: `init-ls-460`, `ft-ls-gs-1460`, `frozen-wenet-encoder`, `wenet-asr-init`, `icefall-zipformer-frozen`, `phoneme-adapter-ctc`, `icefall-zipformer-frozen-adapter` |
| **trunk** | The trainable module the phoneme CTC loss and QbyT share on top of a frozen encoder. See [§7d](#7d-phoneme-ctc-adapter-on-a-frozen-encoder) |
| **demo preset** | Scale-only experiment (`demo_librispeech100`). Same algorithm as the paper, smaller data and step counts |

Do not overload "recipe" to mean the whole codebase; the repo is one shared pipeline with multiple experiment/recipe presets.

### Path conventions

Defaults in `configs/paths/default.yaml` are **repo-relative** paths such as `data/dma-kws/raw`, `data/dma-kws/processed`, etc. Run scripts from the repository root so these resolve correctly.

```text
data/dma-kws/
├── raw/
│   ├── LibriSpeech/
│   └── LibriPhrase-100/
├── processed/
├── features/
└── exp/
```

On a shared server you may mount data at `/data/dma-kws`. Override without editing yaml:

```bash
python3 scripts/train_stage1_ctc.py +experiment=demo_librispeech100 \
  paths.processed_root=/data/dma-kws/processed \
  paths.exp_root=/data/dma-kws/exp
```

Or edit `paths:` in `configs/experiment/<name>.yaml` or `configs/paths/default.yaml`. `+experiment=...` is a **CLI selector**, not a file you edit in place.

### Stage II training scripts

| Script | Use |
|--------|-----|
| `scripts/train_stage2_qbyt.py` | Demo single-phase training, Wenet ASR encoder init (`wenet_asr_stage2`, §7b), and Icefall Zipformer encoder init (`icefall_zipformer_stage2`, §7c) |
| `scripts/train_ctc_adapter.py` | Step A for a frozen encoder: train the phoneme CTC trunk that Stage II then reads; see [§7d](#7d-phoneme-ctc-adapter-on-a-frozen-encoder) |
| `scripts/train_stage2_recipe.py` | Paper multi-phase chain only: `init-ls-460`, `ft-ls-gs-1460`, `frozen-wenet-encoder` |
| `scripts/adapt_stage2_keyword.py` | Stage II LoRA continual adaptation for a user keyword (phoneme matcher QKV); see [§8](#8-stage-ii-lora-continual-adaptation) |
| `scripts/run_keyword_adaptation.py` | One-command prepare → sweep → train → eval for keyword adaptation; see [§8](#8-stage-ii-lora-continual-adaptation) |

For the full paper chain (init → avg → finetune → eval), see [docs/paper-reproduction.md](docs/paper-reproduction.md).

### Continual adaptation (Stage II LoRA)

After a speaker-independent Stage II checkpoint is trained ([§7](#7-stage-ii-train-qbyt-verifier)), you can adapt the **QbyT phoneme matcher** to a user-defined wake word with LoRA (paper Section III-E): freeze the encoder and base QbyT, fine-tune matcher attention on **TTS → real** keyword data mixed 1:1 with LibriPhrase for anti-forgetting. Swap `adapt.keyword` to target another phrase without code changes.

**Quick start:**

```bash
pip install -e '.[adapt]'   # optuna for optional hyperparameter sweep

python3 scripts/run_keyword_adaptation.py \
  adapt.keyword="hey eva" \
  prep.stage2_ckpt=/path/to/stage2_si.pt \
  adapt.stage=all
```

Outputs land under `exp/stage2_adapt/<slug>/` (`adapter_<slug>.pt`, merged `stage2_adapted.pt`, eval report). Full data layout, config tables, and step-by-step commands: [§8. Stage II LoRA continual adaptation](#8-stage-ii-lora-continual-adaptation).

### Checkpoints and resume

| Config key | Format | Purpose |
|------------|--------|---------|
| `run.resume_from` | Lightning `.ckpt` | Full training-state resume (weights, optimizer, scheduler, step) |
| `run.init_checkpoint` / `stage2.init_checkpoint` | Exported `.pt` | Weight-only seed for a **new** run (e.g. Stage I encoder → Stage II, or external Wenet ASR) |
| `stage2.resume_checkpoint` | `.ckpt` or exported weights | Weight seed for finetune from prior Stage II average (`avg_10.ckpt`) |
| `stage2.phoneme_adapter.init_checkpoint` | Exported `.pt` | Step A trunk weights (`adapter_step*.pt`); loaded with `strict=True`. See [§7d](#7d-phoneme-ctc-adapter-on-a-frozen-encoder) |

Exported weight files use names like `stage1_step000020.pt` and `stage2_step000020.pt`. Stage I training also writes `stage1_avg.pt` when `stage1.checkpoint_avg.enabled=true` (demo default).

`scripts/average_checkpoints.py` averages Lightning checkpoints. Use `prep.pattern="*.ckpt"` for checkpoint directories. Demo Stage I auto-averages to `avg_10.ckpt` and exports `stage1_avg.pt` at end of training.

If you started a smoke run with `run.limit_steps=20`, pass the same limit again when resuming with `run.resume_from=last`.

---

## What is implemented now

The runnable path (demo and paper share the same code):

**Training**

```text
LibriSpeech
  -> Stage I CharTokenizer manifest (phoneme vocabulary in data/dict/lang_char.txt)
  -> Stage I Conformer + CTC training

LibriPhrase (paper-format parquet + precomputed fbank .npy)
  -> Stage II utt_loss + seq_loss, random + hard negatives
  -> Stage II Conformer + QbyT training
     demo: single phase (train_stage2_qbyt.py)
     paper: init -> avg -> finetune chain only (train_stage2_recipe.py)

Frozen external encoder (optional, §7d)
  -> Step A: phoneme CTC trunk on the frozen encoder (train_ctc_adapter.py)
  -> Step B: QbyT reads the trunk instead of the raw encoder output

User keyword (optional, after SI Stage II)
  -> LoRA on QbyT matcher, TTS -> real, LibriPhrase 1:1 mix
  -> merged stage2_adapted.pt (see §8)
```

**Inference** (Stage II required; Stage I locator swappable)

```text
input audio + keyword text
  -> Stage I locator: propose keyword time spans (default: phoneme CTC + ContextGraph)
  -> Stage II QbyT verification: score each candidate on cropped audio
  -> detected / not detected
```

**Stage II-only clip inference** is also available for pre-cropped keyword clips:

```text
keyword clip + keyword text
  -> Stage II QbyT verification: score the full clip as one candidate
  -> detected / not detected
```

Default Stage I is the trained phoneme-CTC model. You can swap it for external locators (Zipformer sherpa-onnx, WeKws+Wenet ASR) that only provide `start_sec`/`end_sec`; Stage II still uses this repo's Conformer+QbyT encoder. See [External locator](#external-locator-zipformer--wekwswenet).

`+experiment=wenet_asr_stage2` seeds the Stage II **encoder** from an external Wenet ASR checkpoint; with the default `phoneme_ctc` locator, inference also needs a Stage I CTC checkpoint (`prep.stage1_ckpt`). That preset is distinct from `frozen-wenet-encoder` (`configs/experiment/frozen_wenet_encoder.yaml`), which freezes a **self-trained** Stage I encoder during paper-scale Stage II. See [docs/paper-reproduction.md](docs/paper-reproduction.md).

`+experiment=icefall_zipformer_stage2` seeds the Stage II **encoder** from an Icefall Zipformer KWS checkpoint (requires `ICEFALL_ROOT` + `ICEFALL_CHECKPOINT` env vars). The Conformer encoder is replaced by Icefall's `Zipformer2`; `freeze_encoder=true` by default so only the QbyT head is trained. See [§7c](#7c-stage-ii-initialize-from-icefall-zipformer-encoder).

An external transducer/BPE encoder was never trained against a phoneme target, so QbyT's phoneme text embedding has nothing comparable to match against. `stage2.phoneme_adapter` inserts a small phoneme-CTC-supervised trunk between the frozen encoder and QbyT so both read the same tensor. It is off by default; see [§7d](#7d-phoneme-ctc-adapter-on-a-frozen-encoder).

Main entry points:

```text
scripts/prepare_stage1_wenet.sh        # Stage I manifest + fbank prep chain
scripts/prepare_stage1_librispeech.py
scripts/prepare_stage1_fbank.py
scripts/train_stage1_ctc.py
scripts/average_checkpoints.py
scripts/prepare_stage2_paper.py
scripts/recompute_stage2_distances.py  # phoneme hard-negative distances from g2p parquet
scripts/prepare_stage2_libriphrase.py  # alias for prepare_stage2_paper.py
scripts/prepare_stage2_eval_fbank.py   # precompute LibriPhrase eval fbank .npy for Stage II validation
scripts/train_stage2_qbyt.py           # demo + wenet-asr-init (single phase)
scripts/train_ctc_adapter.py           # Step A: phoneme CTC trunk on a frozen encoder (§7d)
scripts/train_stage2_recipe.py         # paper multi-phase chain only
scripts/prepare_keyword_adaptation.py  # keyword LoRA: fbank + manifests (§8)
scripts/adapt_stage2_keyword.py        # keyword LoRA: per-phase training (§8)
scripts/sweep_adapt_lora.py            # keyword LoRA: Optuna hyperparameter search (§8)
scripts/run_keyword_adaptation.py      # keyword LoRA: prepare/sweep/train/eval orchestrator (§8)
scripts/eval_stage2_libriphrase.py
scripts/run_stage2_demo.py             # Stage II-only single clip inference
scripts/eval_stage2_clips.py           # Stage II-only batch clip eval (manifest)
scripts/eval_two_stage_kws.py
scripts/run_two_stage_demo.py
```

The repo vendors the Wenet toolkit under `wenet/` for encoder/tokenizer utilities. The legacy `qbyt/` reference scripts are not used by the Hydra training pipeline.

Paper-scale configs: `+experiment=paper_ls460`, `+experiment=paper_ls_gs1460`, `+experiment=frozen_wenet_encoder`. Wenet ASR encoder init preset: `+experiment=wenet_asr_stage2`. Icefall Zipformer encoder init preset: `+experiment=icefall_zipformer_stage2`. Phoneme CTC adapter presets: `+experiment=ctc_adapter_icefall` (Step A), `+experiment=icefall_zipformer_stage2_adapter` (Step B). See [docs/paper-reproduction.md](docs/paper-reproduction.md) for the full recipe chain.

Configs use [Hydra](https://hydra.cc/): base groups live under `configs/` (paths, stage1, stage2, …) and experiments are overlays in `configs/experiment/`. Select one with `+experiment=<name>` and override any leaf with dotlist syntax, e.g. `run.devices=2 run.limit_steps=20 stage2.learning_rate=0.001`.

### Shared `fbank` config

Stage I/II prep scripts read a top-level `fbank:` section for shared fbank settings. The default profile preserves the Wenet-compatible torchaudio path:

```yaml
fbank:
  num_mel_bins: 80
  frame_length: 25
  frame_shift: 10
  dither: 0.1
  window_type: povey
  backend: torchaudio_kaldi
  target_sample_rate: null
  snip_edges: true
  low_freq: 20.0
  high_freq: 0.0
```

`+experiment=icefall_zipformer_stage2` overrides this group with `configs/fbank/icefall_kws.yaml`: Lhotse `Fbank`, 16 kHz, 80 mel bins, 25/10 ms frames, Povey window, `dither: 0.0`, `snip_edges: false`, and a 20 Hz to Nyquist-minus-400 Hz passband. The shared extractor applies the same profile to training prep, eval prep, and online Stage II verification. Eval-only overrides can still go under `stage2.eval.fbank:`.

---

## Recommended hardware

The small-scale real demo is intended for a Linux training machine with:

- 2 × NVIDIA V100 GPUs, or similar;
- CUDA-enabled PyTorch + torchaudio;
- enough disk for LibriSpeech and LibriPhrase features/checkpoints.

CPU/Mac local runs are useful for code checks, but not for realistic training.

---

## 1. Clone and create environment

Requires **Python ≥ 3.10**.

```bash
git clone https://github.com/ty4045681/DMA-KWS.git
cd DMA-KWS
```

**Conda (recommended on GPU servers):**

```bash
conda create -n dma-kws python=3.10 -y
conda activate dma-kws
```

**venv (local / Mac):**

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

### Install dependencies

Dependencies are declared in [`pyproject.toml`](pyproject.toml) (package metadata + optional extras) and mirrored in [`requirements.txt`](requirements.txt) for the README workflow. **`torch` and `torchaudio` are not pinned** in either file, so install them separately first.

**Step 1: PyTorch / torchaudio** (match your CUDA driver on GPU machines):

```bash
# Linux + CUDA 12.1 example
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121

# Mac or CPU-only
pip install torch torchaudio
```

**Step 2: project core** (pick one approach):

```bash
# README workflow: requirements.txt + editable install
pip install -r requirements.txt
pip install -e .

# Or install core deps from pyproject.toml only
pip install -e .
```

**Step 3: optional extras** (from `pyproject.toml`):

| Extra | Command | When you need it |
|-------|---------|------------------|
| `dev` | `pip install -e ".[dev]"` | Run `pytest` locally |
| `locator` | `pip install -e ".[locator]"` | `+locator=sherpa_zipformer_kws` (sherpa-onnx) |
| `wekws` | `pip install -e ".[wekws]"` | `+locator=wekws_wenet` (librosa for wav I/O) |
| `icefall` | `pip install -e ".[icefall]"` | Lhotse 1.33.0 fbank for `+experiment=icefall_zipformer_stage2` |

Install everything at once:

```bash
pip install -e ".[dev,locator,wekws,icefall]"
```

**Minimal inference/training install:**

```bash
pip install torch torchaudio
pip install -r requirements.txt
pip install -e .
```

Core packages include `hydra-core`, `g2p_en`, `pandas`, `pyarrow`, `rapidfuzz` (Stage II prep), `soundfile` (FLAC), and `openai-whisper` (vendored QbyT/Wenet encoder utilities). After `pip install -e .`, scripts resolve `dma_kws` without setting `PYTHONPATH`. If you run scripts without an editable install, prefix with `PYTHONPATH=.`.

On some Linux/conda systems, `torchaudio` may still need system audio libraries for `.flac` files; install them before training if `torchaudio.load()` cannot read LibriSpeech audio:

```bash
conda install -c conda-forge libsndfile ffmpeg -y
pip install -U soundfile
```

`g2p_en` uses NLTK data files at runtime. Download them once inside the same conda environment before preparing
phoneme manifests:

```bash
python3 - <<'PY'
import nltk

for pkg in [
    "averaged_perceptron_tagger_eng",
    "averaged_perceptron_tagger",
    "cmudict",
    "punkt",
    "punkt_tab",
]:
    print(f"Downloading {pkg} ...")
    nltk.download(pkg)
PY
```

If the training server cannot access the internet, download the same NLTK data on another machine and copy the
`nltk_data` directory to one of the paths shown by the error message, for example `/root/nltk_data`, or set:

```bash
export NLTK_DATA=/path/to/nltk_data
```

Check lightweight local tests and script entry points:

```bash
bash scripts/run_smoke.sh
```

---

## 2. Prepare data directories

Default experiment: `configs/experiment/demo_librispeech100.yaml` (select with `+experiment=demo_librispeech100`).

By default it expects this layout relative to the repo root (see `configs/paths/default.yaml`):

```text
data/dma-kws/
├── raw/
│   ├── LibriSpeech/
│   │   ├── train-clean-100/
│   │   └── dev-clean/
│   └── LibriPhrase-100/
├── processed/
├── features/
└── exp/
```

If your data lives elsewhere, override on the CLI (`paths.processed_root=...`, etc.) or edit `paths:` in `configs/experiment/<name>.yaml` or `configs/paths/default.yaml`.

Create directories:

```bash
mkdir -p data/dma-kws/raw data/dma-kws/processed data/dma-kws/features data/dma-kws/exp
```

---

## 3. Download datasets

### 3.1 LibriSpeech for Stage I

Download at least:

- `train-clean-100`
- `dev-clean`

Example:

```bash
cd data/dma-kws/raw
mkdir -p LibriSpeech
cd LibriSpeech

wget https://www.openslr.org/resources/12/train-clean-100.tar.gz
wget https://www.openslr.org/resources/12/dev-clean.tar.gz

tar -xzf train-clean-100.tar.gz --strip-components=1
tar -xzf dev-clean.tar.gz --strip-components=1
```

After extraction, verify:

```bash
ls data/dma-kws/raw/LibriSpeech/train-clean-100
ls data/dma-kws/raw/LibriSpeech/dev-clean
```

### 3.2 LibriPhrase-100 for Stage II

For the small real demo, start with:

```text
ZhiqiAi/LibriPhrase-100
```

Download with the current Hugging Face Hub CLI. The newer CLI entrypoint is `hf`; if your environment only has
`huggingface-cli` and it reports `invalid choice: 'download'`, upgrade `huggingface_hub` first.

```bash
python3 -m pip install -U "huggingface_hub"

hf download ZhiqiAi/LibriPhrase-100 \
  --repo-type dataset \
  --local-dir data/dma-kws/raw/LibriPhrase-100
```

Fallback if the `hf` command is still unavailable after upgrading:

```bash
python3 - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="ZhiqiAi/LibriPhrase-100",
    repo_type="dataset",
    local_dir="data/dma-kws/raw/LibriPhrase-100",
)
PY
```

The Stage II preparation script reads two kinds of files from the download:

1. An **aggregated** parquet for phrase metadata, with columns `ngram`, `clips`, and
   (for hard negatives) per-anchor `distances` plus `ngram_g2p`. Use
   `aggregated_segments_with_g2p_distance.parquet` when the download includes it.
   If you only have `aggregated_segments_with_g2p.parquet` (`ngram`, `clips`,
   `ngram_g2p`), run `scripts/recompute_stage2_distances.py` first (see §6) to add
   phoneme-level hard-negative distances before `prepare_stage2_paper.py`.

2. The **decoded** audio shards `LP-100-decoded-*.parquet` (columns
   `audio_rel`, `audio`, `sampling_rate`, ...). The script loads referenced
   clips from these shards, writes per-phrase `clips` / `distances` `.npy` under
   `processed/stage2_qbyt/clips/` and `processed/stage2_qbyt/distances/`, and
   computes 80-dim fbank `.npy` files under `features/fbank/LP-100-fbank/`.

Inspect columns if needed:

```bash
python3 - <<'PY'
import pandas as pd
path = 'data/dma-kws/raw/LibriPhrase-100/path/to/file.parquet'
df = pd.read_parquet(path)
print(df.columns)
print(df.head(1))
PY
```

---

## 4. Stage I: prepare phoneme CTC data

Run a smoke preparation first:

```bash
python3 scripts/prepare_stage1_librispeech.py \
  +experiment=demo_librispeech100 \
  prep.limit=100
```

Expected outputs:

```text
data/dma-kws/processed/stage1_phoneme_ctc/train.jsonl
data/dma-kws/processed/stage1_phoneme_ctc/dev.jsonl
```

Manifests include `phonemes_g2p` targets for Wenet `CharTokenizer`, a phoneme vocabulary over `data/dict/lang_char.txt` (stress-marked ARPAbet: `AH0`, `AH1`, `AH2` are three distinct symbols). `validate_lang_char_dict` enforces the whole contract on load: exactly 71 tokens, contiguous ids `0-70`, the full stress-marked inventory present, and `<blank>` at id 0. That last point matters because every CTC consumer here (Stage I, the `phoneme_ctc` locator, `collapse_ctc`, the search helpers, the §7d adapter) hardcodes blank 0.

If the smoke run works, prepare the full LibriSpeech-100 split:

```bash
python3 scripts/prepare_stage1_librispeech.py \
  +experiment=demo_librispeech100
```

If your LibriSpeech data was downloaded from HuggingFace as parquet shards, for example:

```text
/home/h00513998/librispeech_train_clean_360/
├── 0000.parquet
├── 0001.parquet
└── ...
```

prepare the training manifest directly from those shards:

```bash
python3 scripts/prepare_stage1_librispeech.py \
  +experiment=demo_librispeech100 \
  prep.limit=100 \
  prep.input_format=hf-parquet \
  prep.num_workers=4 \
  prep.parquet_root=/home/h00513998/librispeech_train_clean_360 \
  prep.parquet_split=train-clean-360 \
  prep.dev_parquet_root=/home/h00513998/librispeech_dev_clean \
  prep.dev_parquet_split=dev-clean
```

Remove `prep.limit=100` for the full run after the smoke run completes:

```bash
python3 scripts/prepare_stage1_librispeech.py \
  +experiment=demo_librispeech100 \
  prep.input_format=hf-parquet \
  prep.num_workers=4 \
  prep.parquet_root=/home/h00513998/librispeech_train_clean_360 \
  prep.parquet_split=train-clean-360 \
  prep.dev_parquet_root=/home/h00513998/librispeech_dev_clean \
  prep.dev_parquet_split=dev-clean
```

This mode extracts `audio.bytes` from the parquet records into:

```text
data/dma-kws/processed/stage1_phoneme_ctc/audio/train-clean-360/
```

and writes `train.jsonl` / `dev.jsonl` with `wav_path` values pointing at the extracted audio files. If you omit
`prep.dev_parquet_root`, `paths.librispeech_root` must contain the configured dev split in the official LibriSpeech
directory layout; otherwise the script stops instead of silently writing an empty dev manifest.

Notes:

- The script uses `g2p_en` to convert English transcripts to ARPAbet phonemes.
- Stress markers are **kept** everywhere (`AH0` stays `AH0`): the 71-token vocabulary spells out every stress variant, and all stages must tokenize text through `dma_kws.g2p.text_to_phonemes` so their symbols match. Phonemes outside the vocabulary abort data preparation instead of becoming `<unk>`.
- Stage I and Stage II share the Wenet CharTokenizer phoneme vocabulary at `data/dict/lang_char.txt`.
- `prep.num_workers` controls **hf-parquet mode only** (`prep.input_format=hf-parquet`) for both train and dev manifests:
  - `0` = auto (`min(8, cpu_count())`)
  - `1` = serial processing
  - `>1` = multi-process processing by parquet shard
- In hf-parquet mode with `prep.num_workers>1`, each worker initializes its own `g2p_en` instance, writes a per-shard temporary JSONL, and the main process merges shard outputs by shard filename order so `train.jsonl` / `dev.jsonl` stay deterministic across runs.
- `prep.limit` still caps the final manifest size. In parallel hf-parquet mode the cap is applied during stable merge (same deterministic ordering each run).

Optional: precompute Stage I fbank features for faster training (reads the JSONL manifests above):

```bash
python3 scripts/prepare_stage1_fbank.py \
  +experiment=demo_librispeech100
```

---

## 5. Stage I: train phoneme CTC model

Smoke training run:

```bash
CUDA_VISIBLE_DEVICES=0,1 python3 scripts/train_stage1_ctc.py \
  +experiment=demo_librispeech100 \
  run.devices=2 \
  run.limit_steps=20
```

Full first run:

```bash
CUDA_VISIBLE_DEVICES=0,1 python3 scripts/train_stage1_ctc.py \
  +experiment=demo_librispeech100 \
  run.devices=2
```

Resume an interrupted run (re-pass `run.limit_steps` if the original run was a smoke test):

```bash
CUDA_VISIBLE_DEVICES=0,1 python3 scripts/train_stage1_ctc.py \
  +experiment=demo_librispeech100 \
  run.devices=2 \
  run.resume_from=last \
  run.limit_steps=20
```

`run.resume_from=last` restores the full training state (weights + optimizer + scheduler + step/epoch) from `<checkpoint_dir>/last.ckpt` and continues to the configured limit. Pass an explicit `.ckpt` path instead of `last` to resume from a specific checkpoint. If the original run used `run.limit_steps`, pass the same value again on resume.

Checkpoints are saved under:

```text
data/dma-kws/exp/stage1_phoneme_ctc/checkpoints/
```

Lightning checkpoints use `.ckpt` (including `last.ckpt`). After training, exported encoder weights are written as `stage1_step{step:06d}.pt` and, when averaging is enabled, `stage1_avg.pt`.

Examples:

```bash
ls data/dma-kws/exp/stage1_phoneme_ctc/checkpoints/*.ckpt
ls data/dma-kws/exp/stage1_phoneme_ctc/checkpoints/stage1_avg.pt
```

Optionally average the last few Lightning checkpoints before Stage II init or demo inference:

```bash
python3 scripts/average_checkpoints.py \
  +experiment=demo_librispeech100 \
  prep.input_dir=data/dma-kws/exp/stage1_phoneme_ctc/checkpoints \
  prep.pattern="*.ckpt" \
  prep.last_k=10 \
  prep.output=data/dma-kws/exp/stage1_phoneme_ctc/checkpoints/avg_10.ckpt
```

Demo config enables auto-averaging at end of Stage I training (`stage1.checkpoint_avg`), producing `avg_10.ckpt` and `stage1_avg.pt` without a separate averaging step.

If you hit out-of-memory, reduce batch size via CLI override (demo defaults are 48 per GPU):

```bash
python3 scripts/train_stage1_ctc.py +experiment=demo_librispeech100 stage1.batch_size_per_gpu=24 stage1.num_workers=2
```

---

## 6. Stage II: prepare training data

Stage II training uses the **paper pipeline** (`LibriPhraseTrainDataset`): a parquet with columns `ngram`, `ngram_g2p`, `clips_file`, `distances_file`, plus precomputed fbank `.npy` under `features/fbank/`. This is the same format as the paper configs; the demo differs only in dataset size and step counts.

Prepare that layout with `scripts/prepare_stage2_paper.py` (or the identical alias `scripts/prepare_stage2_libriphrase.py`). Training reads the paper parquet paths from `stage2.parquet_file` and `stage2.wav_dir` in your config (defaults under `data/dma-kws/processed/stage2_qbyt/` and `data/dma-kws/features/fbank/`).

**Hydra prep overrides:** All preparation scripts share the `prep:` group from `configs/prep/default.yaml`. Override any leaf on the command line with dotlist syntax, e.g. `prep.input_parquet=/path/to/file.parquet prep.limit_anchors=50`. Keys not set on the CLI use the yaml defaults (often empty / zero meaning "all" or "auto").

Pass the **aggregated** LibriPhrase parquet explicitly with `prep.input_parquet`. If omitted, `prepare_stage2_paper.py` auto-finds `aggregated_segments_with_g2p*.parquet` under `paths.libriphrase100_root` only; for `+experiment=paper_ls460` you must pass `prep.input_parquet` explicitly. Use `aggregated_segments_with_g2p_distance.parquet` (includes per-clip G2P distances for hard negatives), **not** `aggregated_segments_by_ngram.parquet`.

### Hard negatives and G2P

Hard-negative mining requires a `distances` column per anchor. When your raw download
has `aggregated_segments_with_g2p.parquet` but no `distances` column, recompute them
first with `scripts/recompute_stage2_distances.py` (helper: `dma_kws/stage2/distances.py`;
config: `prep.recompute_distances` in `configs/prep/default.yaml`):

```bash
python3 scripts/recompute_stage2_distances.py \
  +experiment=demo_librispeech100 \
  prep.recompute_distances.input_parquet=data/dma-kws/raw/LibriPhrase-100/aggregated_segments_with_g2p.parquet
```

Default output is alongside the input with the `_g2p_distance.parquet` suffix. Pass
that file to `prepare_stage2_paper.py` via `prep.input_parquet`. Useful overrides:
`prep.recompute_distances.output_parquet`, `top_k` (default 100), `block_size`,
`workers` (-1 = auto), `strip_stress` (default true; hard-negative ranking compares phone identity, so
`AH0`/`AH1`/`AH2` collapse here even though the model itself trains on stress-marked symbols). When `prep.recompute_distances.input_parquet` is unset,
the script looks for `aggregated_segments_with_g2p.parquet` under
`paths.libriphrase460_root`, `paths.libriphrase100_root`, or `paths.libriphrase_root`.

If `ngram_g2p` is missing from the aggregated parquet, or its phonemes carry no stress
markers (i.e. they predate the stress-marked vocabulary), `prepare_stage2_paper.py` runs
G2P via `g2p_en` for every anchor and reports `g2p_recomputed=True`. Force that with
`prep.force_g2p_recompute=true`. When per-anchor `distances` is empty, it falls back to
phoneme edit-distance confusables within the anchor set (top-5 by default).

Demo smoke run:

```bash
python3 scripts/prepare_stage2_paper.py \
  +experiment=demo_librispeech100 \
  prep.input_parquet=data/dma-kws/raw/LibriPhrase-100/aggregated_segments_with_g2p_distance.parquet \
  prep.limit_anchors=50
```

Full demo prep (omit `prep.limit_anchors` or set `prep.limit_anchors=0`):

```bash
python3 scripts/prepare_stage2_paper.py \
  +experiment=demo_librispeech100 \
  prep.input_parquet=data/dma-kws/raw/LibriPhrase-100/aggregated_segments_with_g2p_distance.parquet
```

Same input parquet for the Wenet-init recipe:

```bash
python3 scripts/prepare_stage2_paper.py \
  +experiment=wenet_asr_stage2 \
  prep.input_parquet=data/dma-kws/raw/LibriPhrase-100/aggregated_segments_with_g2p_distance.parquet
```

Prepare features for a frozen Icefall Zipformer KWS encoder with its matching input distribution:

```bash
pip install -e ".[icefall]"

python3 scripts/prepare_stage2_paper.py \
  +experiment=icefall_zipformer_stage2 \
  prep.input_parquet=data/dma-kws/raw/LibriPhrase-100/aggregated_segments_with_g2p_distance.parquet
```

This preset resamples decoded clips to 16 kHz and writes them under `data/dma-kws/features/fbank_icefall_kws/`. The separate root prevents an existing Wenet `.npy` from being reused for the frozen Zipformer encoder.

GigaPhrase-1000 (clips use `GP-1000/` prefix; decoded shards `GP-1000-decoded-*.parquet` under the gigaphrase root; dataset is auto-detected from clip paths in `pairs.py`):

```bash
python3 scripts/prepare_stage2_paper.py \
  +experiment=wenet_asr_stage2 \
  prep.input_parquet=data/dma-kws/raw/GigaPhrase-1000/aggregated_segments_with_g2p_distance.parquet \
  prep.decoded_parquet_root=data/dma-kws/raw/GigaPhrase-1000
```

If `paths.gigaphrase1000_root` is set in your config, you can omit `prep.decoded_parquet_root`.
Use `prep.output_subdir=stage2_qbyt/gp1000` (or `stage2.prep.output_subdir` in config) to avoid
overwriting LibriPhrase outputs. When you change `prep.output_subdir`, also set `stage2.parquet_file` to the new parquet path and verify `stage2.wav_dir` still points at your fbank tree.

LibriPhrase-460 (paper scale; auto-detected from `LP-460/` clip prefixes, decoded shards `LP-460-decoded-*.parquet`, config root `paths.libriphrase460_root`):

```bash
python3 scripts/prepare_stage2_paper.py \
  +experiment=paper_ls460 \
  prep.input_parquet=data/dma-kws/raw/LibriPhrase-460/aggregated_segments_with_g2p_distance.parquet
```

Set `paths.libriphrase460_root` in `+experiment=paper_ls460` (or pass `prep.decoded_parquet_root=...`) if the decoded shards are not under the default path. See [docs/paper-reproduction.md](docs/paper-reproduction.md) for the full paper prep and training chain (`train_stage2_recipe.py`, `eval_stage2_libriphrase.py`).

Parallelism and console output (defaults in `configs/prep/default.yaml`):

- `prep.num_workers` parallelizes fbank extraction **within** each decoded shard via `stream_fbank_from_decoded()`; shards are scanned **serially** (one shard in memory at a time). `0` = auto (`min(8, cpu_count())`); `1` = serial fbank jobs (useful for debugging).
- `prep.use_rich` controls Rich tables and multi-task progress bars when stdout is a TTY; falls back to plain `print` / `tqdm` when redirected or non-interactive (even if `prep.use_rich=true`).

```bash
# Full prep with explicit parallelism
python3 scripts/prepare_stage2_paper.py \
  +experiment=wenet_asr_stage2 \
  prep.input_parquet=data/dma-kws/raw/LibriPhrase-100/aggregated_segments_with_g2p_distance.parquet \
  prep.num_workers=4
```

The script prints staged progress (plan → load parquet → scan decoded shards → build anchors / fbank → summary) and a final stats table. Key fields: `anchors`, `clips_total`, `fbank_written`, `fbank_skipped`, `missing_in_decoded` (referenced clip keys not found in decoded parquet shards during streaming fbank).

### Operational notes

- Fbank uses `stream_fbank_from_decoded()`. Decoded shards are processed one at a time, so peak memory is bounded by a single shard, not the full anchor set.
- Fbank extraction skips existing `.npy` files by default, which makes incremental reruns after fixing a subset of clips safe.
- Use `prep.num_workers=1` when debugging fbank parallelism or reproducing ordering issues within a shard.

Expected outputs:

```text
data/dma-kws/processed/stage2_qbyt/aggregated_segments_with_g2p_distance.parquet
data/dma-kws/processed/stage2_qbyt/clips/
data/dma-kws/processed/stage2_qbyt/distances/
data/dma-kws/features/fbank/LP-100-fbank/
# Zipformer profile instead:
data/dma-kws/features/fbank_icefall_kws/LP-100-fbank/
```

Fbank parameters (`backend`, `target_sample_rate`, `num_mel_bins`, frame settings, and frequency bounds) are read from the top-level `fbank:` block in your config YAML. `stage2.wav_dir` is both the preparation output root and the training input root.

Training consumes the paper parquet + fbank layout directly via `LibriPhraseTrainDataset` (random + hard negatives, utt + seq loss).

### Optional Stage II waveform noise augmentation

Stage II can mix additive noise into training clips before fbank extraction. The
feature cache remains the clean fast path: only samples selected by
`stage2.noise_augmentation.probability` load a waveform, mix noise at a random
SNR, and recompute fbank. Validation and evaluation never use this augmentation.
This switch applies to the base `train_stage2_qbyt.py`/`train_stage2_recipe.py`
flow; keyword LoRA adaptation keeps its separately prepared feature manifests.

The decoded LibriPhrase shards do not provide loose WAV files to the training
dataset, so first build the optional waveform cache while preparing Stage II:

```bash
python3 scripts/prepare_stage2_paper.py \
  +experiment=paper_ls460 \
  prep.input_parquet=/data/dma-kws/raw/LibriPhrase-460/aggregated_segments_with_g2p_distance.parquet \
  prep.waveform_dir=/data/dma-kws/features/stage2_waveforms \
  prep.num_workers=4
```

This is incremental: if clean fbank files already exist, the prep reads decoded
audio only to fill missing WAV cache entries and does not recompute those fbank
files. WAVs are written as PCM16 under the original clip layout, for example
`stage2_waveforms/LP-460/<phrase>/<clip>.wav`. The waveform cache currently
requires zero training padding; prep fails explicitly if it is combined with
non-zero `prep.left_padding_ms`/`prep.right_padding_ms`, rather than silently
producing clean and noise-augmented features with different boundaries.

Create a UTF-8 noise list containing one WAV/FLAC path per line. Blank lines and
`#` comments are ignored; relative entries are resolved from the list file's
directory. Then enable augmentation for training:

```bash
python3 scripts/train_stage2_recipe.py \
  +experiment=paper_ls460 \
  stage2.noise_augmentation.enabled=true \
  stage2.noise_augmentation.probability=0.3 \
  stage2.noise_augmentation.waveform_dir=/data/dma-kws/features/stage2_waveforms \
  stage2.noise_augmentation.noise_list_path=/data/musan/noise.list \
  stage2.noise_augmentation.snr_db_min=10 \
  stage2.noise_augmentation.snr_db_max=20 \
  run.devices=4
```

The probability gate, noise choice, crop and SNR draw use the Stage II dataset's
worker/DDP-aware RNG. The mixed waveform is converted with the same top-level
`fbank:` backend and parameters as the clean training features. A missing
waveform/noise file, empty list, invalid probability, or invalid SNR interval
fails before or at the first affected sample with a concrete path in the error.

### LibriPhrase eval data (required for validation)

Stage II training runs LibriPhrase validation on a schedule (`stage2.validation.val_check_interval`). You need the official eval set under `stage2.eval.test_dir` (default `data/dma-kws/raw/LibriPhrase-100/eval`). It is **not** included in the LibriPhrase-100 training download; obtain it from the LibriPhrase-460 Hugging Face eval assets or symlink a shared eval tree. See [docs/paper-reproduction.md](docs/paper-reproduction.md) for LP-460 hard eval used in paper metrics.

Expected layout:

```text
<test_dir>/
  evaluation_set/libriphrase_diffspk_all_1word.csv
  evaluation_set/libriphrase_diffspk_all_2word.csv
  evaluation_set/libriphrase_diffspk_all_3word.csv
  evaluation_set/libriphrase_diffspk_all_4word.csv
  train-other-500/train-other-500/<spk>/<chap>/*.wav
```

`stage2.eval.split` defaults to `hard`; validation CSVs must include the columns expected for that split. Training reads `stage2.eval.test_dir`; `prep.test_dir` applies only to `prepare_stage2_eval_fbank.py`.

Precompute validation features with `scripts/prepare_stage2_eval_fbank.py`. When `stage2.eval.fbank_dir` is empty, each `.npy` remains next to its `.wav` for backward compatibility. When it is set, the script mirrors each WAV-relative path under that separate root, and validation reads from the same location. The Zipformer preset uses `data/dma-kws/features/fbank_icefall_kws_eval/` so old Wenet eval features cannot be reused. Recommend `prep.from_csv=true` (default `false` walks every wav under `test_dir`):

```bash
python3 scripts/prepare_stage2_eval_fbank.py \
  +experiment=demo_librispeech100 \
  prep.from_csv=true
```

```bash
python3 scripts/prepare_stage2_eval_fbank.py \
  +experiment=wenet_asr_stage2 \
  prep.from_csv=true
```

```bash
python3 scripts/prepare_stage2_eval_fbank.py \
  +experiment=icefall_zipformer_stage2 \
  prep.from_csv=true
```

`prep.from_csv=true` converts only wav files referenced by the eval CSV `anchor` / `comparison` columns. With a separate `fbank_dir`, `train-other-500/.../clip.wav` maps to `<fbank_dir>/train-other-500/.../clip.npy`; without one, it maps to a sibling `clip.npy`.

Useful overrides (also in `configs/prep/default.yaml`):

- `prep.test_dir=/path` sets the eval root for the prep script when it differs from `stage2.eval.test_dir` in config.
- `prep.limit=100` runs a smoke test (first N wav paths only).
- `prep.log_interval=1000` sets the progress print frequency.
- `prep.no_skip_existing=true` recomputes fbank even when `.npy` already exists (default skips existing files for incremental reruns).

---

## 7. Stage II: train QbyT verifier

Use `scripts/train_stage2_qbyt.py` for the demo single-phase run. Optionally initialize from a Stage I exported weight file:

```bash
STAGE1_CKPT=data/dma-kws/exp/stage1_phoneme_ctc/checkpoints/stage1_step000020.pt
# or: stage1_avg.pt / avg_10.ckpt converted via training export
```

Smoke training run:

```bash
CUDA_VISIBLE_DEVICES=0,1 python3 scripts/train_stage2_qbyt.py \
  +experiment=demo_librispeech100 \
  run.devices=2 \
  run.init_checkpoint="$STAGE1_CKPT" \
  run.limit_steps=20
```

Full first run:

```bash
CUDA_VISIBLE_DEVICES=0,1 python3 scripts/train_stage2_qbyt.py \
  +experiment=demo_librispeech100 \
  run.devices=2 \
  run.init_checkpoint="$STAGE1_CKPT"
```

Resume an interrupted run:

```bash
CUDA_VISIBLE_DEVICES=0,1 python3 scripts/train_stage2_qbyt.py \
  +experiment=demo_librispeech100 \
  run.devices=2 \
  run.resume_from=last \
  run.limit_steps=20
```

`run.resume_from=last` restores the full training state from `<checkpoint_dir>/last.ckpt`, or pass an explicit `.ckpt` path. If the original run used `run.limit_steps`, pass the same value again on resume. This is distinct from `run.init_checkpoint` / `stage2.resume_checkpoint`, which only load weights to seed a fresh finetune recipe.

Checkpoints are saved under:

```text
data/dma-kws/exp/stage2_qbyt/checkpoints/
```

Lightning checkpoints use `.ckpt`; exported weights are written as `stage2_step{step:06d}.pt`.

Example:

```bash
ls data/dma-kws/exp/stage2_qbyt/checkpoints/*.ckpt
ls data/dma-kws/exp/stage2_qbyt/checkpoints/stage2_step000020.pt
```

If you hit out-of-memory, reduce batch size via CLI override (demo default is 128 per GPU):

```bash
python3 scripts/train_stage2_qbyt.py +experiment=demo_librispeech100 stage2.batch_size_per_gpu=64 stage2.num_workers=2
```

For the paper multi-phase chain (`init-ls-460` → avg → `ft-ls-gs-1460`), use `scripts/train_stage2_recipe.py`. See [docs/paper-reproduction.md](docs/paper-reproduction.md).

---

## 7b. Stage II: initialize from external Wenet ASR encoder

Alternative to Stage I init: seed the Stage II Conformer **encoder** from a pretrained Wenet ASR checkpoint. Use `+experiment=wenet_asr_stage2` with `scripts/train_stage2_qbyt.py` (single phase). Two-stage **inference** still requires a separately trained Stage I CTC model for candidate proposal.

This is distinct from `+experiment=frozen_wenet_encoder` (`configs/experiment/frozen_wenet_encoder.yaml`), which freezes a self-trained Stage I encoder during the paper `frozen-wenet-encoder` recipe. See [docs/paper-reproduction.md](docs/paper-reproduction.md).

Prerequisites:

- A Wenet ASR `.pt` checkpoint whose state dict contains `encoder.*` weights (for example `model_dir/avg_10.pt` or `model_dir/step_247499.pt`).
- The matching `global_cmvn` JSON file from the same Wenet training run.
- `stage1` encoder settings in the config must match the Wenet ASR training yaml (`causal`, `cnn_module_norm`, `cmvn`, block sizes, and related fields). Mismatches show up as high missing/unexpected counts at startup.

Set environment variables referenced in the config:

```bash
export WENET_CMVN_FILE=/path/to/global_cmvn
export WENET_ASR_CHECKPOINT=/path/to/wenet_asr.pt
```

Full data prep and training chain:

```bash
python3 scripts/prepare_stage2_paper.py \
  +experiment=wenet_asr_stage2 \
  prep.input_parquet=data/dma-kws/raw/LibriPhrase-100/aggregated_segments_with_g2p_distance.parquet \
  prep.decoded_parquet_root=data/dma-kws/raw/LibriPhrase-100

python3 scripts/prepare_stage2_eval_fbank.py \
  +experiment=wenet_asr_stage2 \
  prep.from_csv=true

CUDA_VISIBLE_DEVICES=0,1 python3 scripts/train_stage2_qbyt.py \
  +experiment=wenet_asr_stage2 \
  run.devices=2
```

Checkpoints are written under `data/dma-kws/exp/stage2_qbyt/checkpoints/wenet-asr-init/`.

Override the init checkpoint without editing the yaml:

```bash
CUDA_VISIBLE_DEVICES=0,1 python3 scripts/train_stage2_qbyt.py \
  +experiment=wenet_asr_stage2 \
  run.devices=2 \
  run.init_checkpoint=/path/to/wenet_asr.pt
```

Notes:

- QbyT still uses the phoneme CharTokenizer at `data/dict/lang_char.txt` (same as the paper recipe). This path does **not** switch to Wenet BPE/subword tokenization.
- On startup, look for `Loaded encoder weights from ...: missing=N unexpected=M`. Both counts should be low (ideally zero). High values usually mean the encoder yaml does not match the Wenet checkpoint or the checkpoint path is wrong.

---

## 7c. Stage II: initialize from Icefall Zipformer encoder

Alternative to Wenet Conformer: seed the Stage II encoder from a pretrained [Icefall Zipformer KWS](https://github.com/k2-fsa/icefall/tree/master/egs/gigaspeech/KWS/zipformer) checkpoint. Use `+experiment=icefall_zipformer_stage2` with `scripts/train_stage2_qbyt.py`.

The experiment selects `configs/fbank/icefall_kws.yaml`, which matches the final KWS fine-tuning input path: normalized waveform samples passed directly to Lhotse `Fbank`, without the Wenet path's `32768` scale factor or CMVN. It does not select the GigaSpeech `KaldifeatFbank` pretraining profile. Training prep, eval prep, and online verification all force non-16 kHz audio to 16 kHz before extracting features.

By default `freeze_encoder=true`, so only the QbyT phoneme matcher head is trained. Two-stage **inference** still requires a separately trained Stage I CTC model for candidate proposal.

Prerequisites:

- An Icefall Zipformer KWS `.pt` checkpoint saved by `train.py` (contains a `model` key with `encoder_embed.*` and `encoder.*` weights).
- The Icefall repository cloned locally.
- The `icefall` dependency extra installed with `pip install -e ".[icefall]"`.
- `stage1` encoder parameters in the config must match the Icefall checkpoint (encoder_dim, num_encoder_layers, downsampling_factor, causal, etc.).

Set environment variables referenced in the config:

```bash
export ICEFALL_ROOT=/path/to/icefall          # icefall repo root
export ICEFALL_CHECKPOINT=/path/to/checkpoint.pt  # Zipformer KWS checkpoint
```

Full data prep and training chain:

```bash
pip install -e ".[icefall]"

python3 scripts/prepare_stage2_paper.py \
  +experiment=icefall_zipformer_stage2 \
  prep.input_parquet=data/dma-kws/raw/LibriPhrase-100/aggregated_segments_with_g2p_distance.parquet

python3 scripts/prepare_stage2_eval_fbank.py \
  +experiment=icefall_zipformer_stage2 \
  prep.from_csv=true

CUDA_VISIBLE_DEVICES=0,1 python3 scripts/train_stage2_qbyt.py \
  +experiment=icefall_zipformer_stage2 \
  run.devices=2
```

Checkpoints are written under `data/dma-kws/exp/stage2_qbyt/checkpoints/icefall-zipformer-frozen/`.

The prep commands write training features under `data/dma-kws/features/fbank_icefall_kws/` and eval features under `data/dma-kws/features/fbank_icefall_kws_eval/`. Do not point `stage2.wav_dir` or `stage2.eval.fbank_dir` at a tree generated with the default Wenet profile.

Smoke run:

```bash
python3 scripts/train_stage2_qbyt.py \
  +experiment=icefall_zipformer_stage2 \
  run.limit_steps=5 \
  run.device=cpu
```

Override the checkpoint without editing the yaml:

```bash
CUDA_VISIBLE_DEVICES=0,1 python3 scripts/train_stage2_qbyt.py \
  +experiment=icefall_zipformer_stage2 \
  run.devices=2 \
  run.init_checkpoint=/path/to/zipformer_kws.pt
```

Notes:

- The Icefall encoder adapter (`dma_kws/stage2/icefall_encoder.py`) exposes the same `forward(feat, feat_lengths) → (encoder_out, encoder_mask)` interface as the Wenet Conformer, so the rest of the Stage II training pipeline is unchanged.
- QbyT still uses the phoneme CharTokenizer at `data/dict/lang_char.txt`.
- Default encoder output dimension is `max(encoder_dim) = 128` (Zipformer KWS recipe default). `stage2.qbyt_embed_dim` does **not** have to match it: `QbyT.audio_projection` is `Linear(encoder_output_size, embed_dim)`, which is also what lets the [§7d](#7d-phoneme-ctc-adapter-on-a-frozen-encoder) trunk change the width.
- This preset trains QbyT directly on the frozen encoder output. That output lives in the space a BPE transducer objective produced, not a phoneme space, so the single `audio_projection` layer carries the whole cross-space mapping under utterance/sequence BCE supervision only. [§7d](#7d-phoneme-ctc-adapter-on-a-frozen-encoder) is the alternative that adds phoneme CTC supervision to that mapping.
- `causal` must match the checkpoint. The `icefall_zipformer_stage2` preset sets `causal: true` for the target KWS checkpoint; override it only when loading a checkpoint trained with `--causal=false`. Flipping it swaps `ChunkCausalDepthwiseConv1d` for a plain `nn.Conv1d`, and because the icefall state dict is loaded with `strict=False` the convolution weights would be dropped silently.
- Online Stage II verification uses the same Hydra fbank profile as precomputation and resamples the full validation waveform before slicing candidate spans.
- On startup, look for `Loaded encoder_embed weights from ...: missing=N unexpected=M` and `Loaded encoder weights from ...`. Both counts should be low. High unexpected counts usually mean a config mismatch (wrong `encoder_dim`, `num_encoder_layers`, etc.).

### Streaming operating point (`stage1.stream`)

Chunked attention is configured in one place and applied consistently to every phase. `chunk_size` / `left_context_frames` are the **deployment operating point**: they must be single values and they drive validation, offline eval and inference. Randomized multi-latency training is opt-in via `train_policy`.

```yaml
stage1:
  causal: true
  stream:
    chunk_size: 16              # 50 Hz frames -> 320 ms
    left_context_frames: 64     # 50 Hz frames -> 1.28 s
    train_policy: match         # match | multi
    train_chunk_size: "16,32,64,-1"
    train_left_context_frames: "64,128,256,-1"
```

| Phase | Chunk config |
|-------|--------------|
| Stage I icefall decode | `--causal/--chunk-size/--left-context-frames` generated from `stage1.stream` |
| Stage II training, encoder frozen | `train_policy` (`match` reuses the operating point) |
| Stage II training, encoder fine-tuned | same; use `multi` only if you ship several latencies |
| Validation inside training | operating point |
| Offline eval / test | operating point |
| Inference / demo | operating point |

Units are backend specific. For `icefall_zipformer` they are frames at 50 Hz (fbank 100 Hz halved by `Conv2dSubsampling`); for the Wenet Conformer they are frames at 25 Hz (after the 4x conv2d subsampling), and `train_policy: multi` delegates to wenet's own dynamic-chunk sampler. `chunk_size: -1` with `left_context_frames: -1` selects offline full context on both backends.

This matters because neither backend gates its chunk sampling on training mode: icefall's `Zipformer2.get_chunk_info` calls `random.choice` unconditionally, and wenet's `add_optional_chunk_mask` takes its `torch.randint` branch whenever `decoding_chunk_size == 0`. Without a declared operating point, eval scores are not reproducible and do not correspond to any deployable configuration. Every eval script therefore records the resolved point in its output (`"stream": "backend=... eval=16/64 ..."`), and loading a `.pt` checkpoint trained at a different point raises.

The operating point lives in the experiment overlay, so comparing points is a shell loop:

```bash
for point in 16/64 32/128 64/256 -1/-1; do
  python3 scripts/eval_stage2_libriphrase.py \
    +experiment=icefall_zipformer_stage2 \
    stage1.stream.chunk_size=${point%/*} \
    stage1.stream.left_context_frames=${point#*/} \
    prep.checkpoint=/path/to/stage2.pt
done
```

**Migration.** `stage1.chunk_size` / `stage1.left_context_frames` were removed; setting either raises with the replacement snippet. Previous runs left the encoder on icefall's `16,32,64,-1` list in *all* phases, so their metrics are a mixture over randomly drawn operating points and are not comparable with post-migration numbers. Re-measure your baseline before starting a new sweep.

Known gaps that a fixed operating point does **not** close: training and eval encode isolated clips, so the first chunk has no left context and the clip always starts on a chunk boundary, whereas a streaming deployment carries real preceding audio and an arbitrary chunk phase. `Stage2Verifier` also crops the waveform and re-encodes, while a shared streaming encoder would slice already-computed output frames.

---

## 7d. Phoneme CTC adapter on a frozen encoder

Optional addition to §7c. **Off by default** (`stage2.phoneme_adapter.enabled: false`), so §7c reproduces unchanged and stays the baseline any adapter run is compared against.

### Why

QbyT's text branch is `nn.Embedding` over the 71-symbol phoneme table, so the audio it is matched against has to live in a comparable space. In the paper that was automatic: the Stage I encoder was trained with phoneme CTC, so its frames are already phoneme-discriminative. An Icefall Zipformer KWS encoder was trained with a **transducer/BPE** objective instead, and §7c hands QbyT its raw output. The entire cross-space mapping then rests on one `Linear` layer supervised only by the two BCE terms.

That supervision is weak in a specific way: `seq_label` is per-anchor-phoneme **set membership** (`build_seq_label` in `dma_kws/tokenizer.py`), with no ordering, position or count information. Nothing in the objective forces compositional phoneme matching over whole-phrase acoustic templates, and the latter does not generalize to unseen keywords.

The fix is a trunk that the CTC loss and QbyT **share**:

```text
frozen encoder ──▶ h = trunk(encoder_out)      <- the only shared trainable module
                        ├─▶ Linear(d, 71) ──▶ CTC loss        (Step A, and optionally Step B)
                        └─▶ QbyT ──────────▶ utt BCE + seq BCE
```

A CTC head bolted on as a **separate branch** would not do this. It would only give Stage I phoneme search; QbyT would still read the untouched encoder output and the cross-space problem would be exactly where it was.

### Step A: train the trunk

```bash
export ICEFALL_ROOT=/path/to/icefall
export ICEFALL_CHECKPOINT=/path/to/zipformer_kws.pt

python3 scripts/train_ctc_adapter.py +experiment=ctc_adapter_icefall
```

Reuses the Stage I manifests from `scripts/prepare_stage1_librispeech.py` (`${paths.processed_root}/stage1_phoneme_ctc/{train,dev}.jsonl`); no new data prep. Validates with `val/per` on the dev manifest and exports `adapter_step*.pt` to `phoneme_adapter.checkpoint_dir`. The presets give each trunk its own subdirectory (`${paths.exp_root}/phoneme_adapter/checkpoints/{conv,linear,mlp,conformer}/`) so the control runs do not overwrite each other.

A dev manifest is required, not optional: `val/per` on the frozen representation is the only signal that says which trunk to use.

The encoder is frozen and forced into `eval()`. That is not cosmetic: `use_icefall_dropout_schedule` builds a `ScheduledFloat((0,0.3),(20000,0.1))` and icefall's `set_batch_count()` is never called here, so a train-mode encoder would leave dropout pinned at `0.3` and the trunk would learn against a representation the deployed encoder never produces.

**Features must be the same profile as Stage II.** The runner forces `FbankExtractor` with the resolved `fbank:` group, and `ctc_adapter_icefall` overrides that group to `icefall_kws`. Do not substitute `scripts/prepare_stage1_fbank.py` output here: `dma_kws/audio.py:extract_fbank` calls `kaldi.fbank(waveform, …)` while `FbankExtractor` calls `kaldi.fbank(waveform * (1 << 15), …)`, a 32768× amplitude difference on top of differing `dither`/`snip_edges`.

### Trunk types

`phoneme_adapter.trunk.type` selects how much temporal context the trunk has. `linear` and `mlp` are pointwise; they re-mix channels within a frame but cannot move evidence between frames, which is what an RNN-T-trained encoder needs because transducer emission is systematically delayed relative to the acoustics.

| `trunk.type` | Structure | Purpose | Preset |
|---|---|---|---|
| `linear` | `Linear(128, d)` | Control: how linearly separable phonemes already are | `+experiment=ctc_adapter_probe_linear` |
| `mlp` | `Linear → LayerNorm → SiLU → Linear` | Control: separates "needs capacity" from "needs context" | `+experiment=ctc_adapter_probe_mlp` |
| `conv` | N × depthwise-separable conv + residual | **Default.** Has a receptive field | `+experiment=ctc_adapter_icefall` |
| `conformer` | 1 to 2 Wenet Conformer blocks | Expressiveness upper bound | `+experiment=ctc_adapter_probe_conformer` |

`conv` padding follows `stage1.causal`: left-only when causal, so the trunk cannot train with lookahead the streaming deployment has no way to provide. For a `conformer` trunk on a causal encoder, `trunk.chunk_size` is **required** (units are trunk-input frames, i.e. encoder output frames at 25 Hz). Unrestricted attention there would be the same silent lookahead, so it raises instead of defaulting.

### Step B: Stage II reads the trunk

```bash
python3 scripts/train_stage2_qbyt.py \
  +experiment=icefall_zipformer_stage2_adapter \
  stage2.phoneme_adapter.init_checkpoint=/path/to/adapter_step060000.pt \
  run.devices=2
```

Two modes:

| Mode | Config | Behaviour |
|---|---|---|
| **B1** | `freeze: true`, `ctc_weight: 0.0` | Trunk fixed after Step A; only QbyT trains. Reproducible, cache-friendly |
| **B2** | `freeze: false`, `ctc_weight: 0.2` | Trunk keeps training, with CTC holding it in phoneme space against the two BCE terms. Preset default |

B2's CTC target is each clip's **own** phoneme sequence (`query_seq`), not the anchor's. For a negative pair the audio is a different phrase, so supervising with the anchor would teach the trunk the wrong transcript. Samples whose label is longer than their frame count are dropped and counted in `train/ctc_skipped`; at 25 Hz a 0.5 s clip has only ~12 frames, so watch that counter. A high skip rate means the auxiliary loss only ever sees the long clips.

`trunk.*` in Step B must match what Step A trained. The checkpoint loads with `strict=True`, and the recorded `blank_id` is compared, so a mismatch fails at construction rather than scoring against a space CTC never supervised.

### LoRA (§8) with an adapter

Set `stage2.phoneme_adapter.enabled=true` and mirror the trunk shape. The trunk stays frozen with `ctc_weight: 0.0` during adaptation. It is part of the forward pass Stage I and Stage II share, so letting LoRA move it would break the single-encoder-pass premise.

### Adapter config keys

Step A reads the top-level `phoneme_adapter:` group (`configs/phoneme_adapter/default.yaml`); Step B reads `stage2.phoneme_adapter:`. The `trunk` block has the same shape in both and must agree.

| Key | Default | Meaning |
|-----|---------|---------|
| `phoneme_adapter.init_checkpoint` | `""` | Icefall/Stage I checkpoint the frozen encoder starts from |
| `phoneme_adapter.train_manifest` / `.dev_manifest` | `""` | Empty → `${paths.processed_root}/stage1_phoneme_ctc/{train,dev}.jsonl` |
| `phoneme_adapter.trunk.type` | `conv` | `linear` / `mlp` / `conv` / `conformer` |
| `phoneme_adapter.trunk.output_dim` | `192` | Trunk width; becomes QbyT's `encoder_output_size` |
| `phoneme_adapter.trunk.num_layers` | `2` | Blocks (`conv`) or Conformer layers |
| `phoneme_adapter.trunk.kernel_size` | `7` | Conv kernel (`conv`) or CNN module kernel (`conformer`) |
| `phoneme_adapter.trunk.chunk_size` | `0` | `conformer` only, in 25 Hz frames; **required** when `stage1.causal` |
| `phoneme_adapter.trunk.left_context_chunks` | `-1` | `conformer` only; `<0` = all left chunks |
| `phoneme_adapter.expose_posterior` | `false` | Concatenate CTC log-posteriors onto the QbyT input (ablation) |
| `phoneme_adapter.checkpoint.every_n_train_steps` | `2000` | Must be a multiple of the validation interval |
| `stage2.phoneme_adapter.enabled` | `false` | Insert the trunk between encoder and QbyT |
| `stage2.phoneme_adapter.init_checkpoint` | `""` | Step A `adapter_step*.pt`; `strict=True` |
| `stage2.phoneme_adapter.freeze` | `false` | `true` = B1 |
| `stage2.phoneme_adapter.ctc_weight` | `0.0` | Auxiliary CTC weight; `0.0` reproduces the pre-adapter loss exactly |

### What to measure

`val/per` picks the trunk; it does not by itself say whether the adapter is worth keeping. The decision metric is Stage II **LibriPhrase `hard` AUC/EER against the §7c baseline checkpoint**, since `enabled=false` is always available as the fallback.

### Notes

- Feed QbyT the trunk hidden state, not the 71-dim posteriors. `expose_posterior: true` concatenates `[h ; log_softmax]` as an ablation, but a posteriorgram alone is a hard information bottleneck and its blank-dominated peaks are a poor dense sequence representation.
- With `ctc_weight: 0.0` the CTC projection is frozen and skipped in the forward pass. It stays in the state dict so checkpoints load strictly, but leaving it trainable would make DDP abort, because a parameter that requires grad and never receives one fails the reduction check unless `find_unused_parameters` happens to be on.
- Loading a checkpoint that was supposed to carry trunk weights but does not raises. A random trunk is only accepted when no checkpoint was given at all (training the trunk from scratch inside Stage II is a legitimate variant).
- `data/dict/lang_char.txt` must keep `<blank>` at id 0; `validate_lang_char_dict` now enforces it, because every CTC consumer here hardcodes it.
- `phoneme_adapter.checkpoint.every_n_train_steps` must be a multiple of the validation interval, otherwise `ModelCheckpoint` cannot read `val/per` at save time and silently stops writing a best checkpoint.

---

## 8. Stage II LoRA continual adaptation

Paper Section III-E: freeze the full model, LoRA-tune **QbyT phoneme matcher attention QKV** on **TTS → real** data with **LibriPhrase : keyword = 1:1** anti-forgetting mix. Requires a trained SI Stage II checkpoint ([§7](#7-stage-ii-train-qbyt-verifier)).

**Data layout** (slug from `adapt.keyword`, e.g. `hey eva` → `hey_eva`; default root `${paths.processed_root}/adapt/<slug>`):

```text
data/dma-kws/processed/adapt/<slug>/raw/{tts,real}/{positive,negative/<neg_text_slug>}/...       # train
data/dma-kws/processed/adapt/<slug>/raw/{tts,real}/eval/{positive,negative/<neg_text_slug>}/...  # eval
data/dma-kws/processed/adapt/<slug>/fbank/...
data/dma-kws/processed/adapt/<slug>/manifests/{tts,real}_{train,eval}.csv
```

Alternatively, configure external source directories for each phase. Every wav under `positive_dir` is a positive example for `adapt.keyword`; each direct child directory under `negative_root` is a negative phrase, with all of its wavs assigned to that phrase. The name is case-insensitive and separators such as `_`, spaces, and `-` normalize to spaces, so `Hi_Eva`, `HI EVA`, and `hi-eva` all map to `hi eva`.

```yaml
adapt:
  keyword: hey eva
  sources:
    tts:
      positive_dir: /data/tts/hey_eva
      negative_root: /data/tts/confusable
    real:
      positive_dir: /data/real/hey_eva
      negative_root: /data/real/confusable
```

In this mode the TTS and real samples are each split into train/eval sets with `adapt.eval_fraction` and `adapt.eval_seed`, then fbank features and standard phase manifests are written under `adapt.data_root`.

**One command** (prepare → optional Optuna sweep → TTS phase → real phase → eval report):

```bash
pip install -e '.[adapt]'   # adds optuna for sweep

python3 scripts/run_keyword_adaptation.py \
  adapt.keyword="hey eva" \
  prep.stage2_ckpt=/path/to/stage2_si.pt \
  adapt.stage=all
```

**Step-by-step:**

```bash
# 1. Prepare fbank + manifests
python3 scripts/prepare_keyword_adaptation.py adapt.keyword="hey eva"

# 2. Optional hyperparameter search (TPE + MedianPruner)
python3 scripts/sweep_adapt_lora.py adapt.keyword="hey eva" prep.stage2_ckpt=/path/to/stage2_si.pt

# 3. Two-phase LoRA training (real resumes TTS adapter on same SI base)
python3 scripts/adapt_stage2_keyword.py adapt.keyword="hey eva" adapt.phase=tts prep.stage2_ckpt=/path/to/stage2_si.pt
python3 scripts/adapt_stage2_keyword.py adapt.keyword="hey eva" adapt.phase=real prep.stage2_ckpt=/path/to/stage2_si.pt

# 4. Eval: base vs adapted under exp/stage2_adapt/<slug>/reports/
python3 scripts/run_keyword_adaptation.py adapt.keyword="hey eva" prep.stage2_ckpt=/path/to/stage2_si.pt adapt.stage=eval
```

Outputs: `exp/stage2_adapt/<slug>/adapter_<slug>.pt` (small LoRA only), `stage2_adapted.pt` (merged, loadable by `Stage2Verifier`). To adapt another wake word, change `adapt.keyword` and place data under `data/dma-kws/processed/adapt/<new_slug>/raw/...`.

### Joint LoRA: real + TTS + LibriPhrase + MUSAN

By default (`adapt.method=lora`), the [joint overlay](configs/experiment/adapt_joint.yaml) runs one LoRA adapter and one optimizer schedule from the original Stage II base. Compose it after the experiment matching the checkpoint; it preserves the encoder, phoneme adapter and QbyT readout configuration. Existing TTS → real runs remain available.

```bash
PYTHONPATH=. python scripts/run_keyword_adaptation.py \
  '+experiment=[adapt_hey_eva_icefall_adapter_v2,adapt_joint]' \
  adapt.stage=train \
  adapt.data_root=/path/to/prepared/adapt \
  prep.stage2_ckpt=/path/to/stage2_base.pt \
  stage2.background_negative.audio_list_path=/path/to/train_background.list
```

The data root must contain `manifests/{real,tts}_{train,eval}.csv` and the matching `fbank/` tree. There is no `joint_train.csv`: the [joint dataset](dma_kws/stage2/joint_dataset.py) reads the two keyword sources separately and reuses LibriPhrase replay. Each keyword source/split must contain positives and negatives. For preparation with `adapt.stage=all`, provide a combined `prep.manifest_csv` with `phase=real` or `phase=tts` on every row, or configure both raw source trees. The adapter-v2 experiment's default real-only source CSV is insufficient for joint preparation.

Default sample shares are real **30%**, TTS **20%**, LibriPhrase **40%**, MUSAN **10%**. The underlying strata are real positive/negative 15% each, TTS positive/negative 10% each, LibriPhrase positive 25% / speech negative 15%, and background 10%. The [sampler](dma_kws/stage2/joint_dataset.py) carries fractional quotas across batches and epochs; it balances real speakers and TTS voices/negative phrases where metadata is available. Background examples use the target keyword for half their queries and LibriPhrase queries for the other half.

Adjust `adapt.mix_ratio` (all keyword data), `adapt.joint.real_fraction` (real share within keyword data), and `stage2.background_negative.probability` (background share within replay negatives). Thus background's global share is `(1 - mix_ratio) * 0.5 * probability`. These are sampling settings; the [training entry point](dma_kws/stage2/adapt.py) retains the existing readout-specific losses across all three methods. LoRA keeps the whole base frozen; `qbyt_full` trains QbyT with a frozen encoder; `encoder_qbyt_full` trains both encoder and QbyT. Any phoneme adapter stays frozen. Online additive noise augmentation remains unsupported for these adaptation manifests.

The [manifest checks](dma_kws/stage2/joint_manifest.py) reject real speaker leakage, shared audio/recording identities across train/eval, and inconsistent keyword pronunciations. Joint preparation groups related recordings before splitting; explicit splits are preserved. Supply `speaker_id`, `voice_id`, and original-recording metadata when available: relationships cannot be inferred from missing metadata. MUSAN training and validation lists must refer to disjoint original recordings; background cache source records are checked too.

Training logs separate `real`, `tts`, `lph` and `musan` source fractions and BCE losses. Validation keeps real, TTS and LibriPhrase separate; `val_target_auc` and the default checkpoint selection refer to **real speech**. Set `adapt.joint.background_eval_list=/path/to/validation_background.list` to add fixed background-crop validation (`val/musan_deploy_fpr`). Clip FPR is not continuous-audio FA/h. Run `adapt.stage=eval` with that list and `prep.musan_root=/path/to/musan` for the [evaluation report](scripts/run_keyword_adaptation.py), including base/adapted real, TTS, LibriPhrase and continuous MUSAN FA/h. Use a validation list here; keep final blind-test recordings separate. This change does not automatically enforce a FA/h or forgetting constraint during checkpoint selection.

The default joint output root is `paths.exp_root/stage2_adapt_joint/<slug>`, isolated from sequential runs. Full Lightning resumes (`run.resume_from`) restore the [consumed-batch cursor](dma_kws/stage2/joint_loader.py), optimizer and scheduler. Data manifests, replay parquet, tokenizer, sampling policy, batch size and world size must match. With gradient accumulation, both batches per rank per epoch and the validation interval must be divisible by `stage2.accumulate_grad_batches`; incomplete-accumulation checkpoints are rejected. Adapter-only initialization (`run.resume_checkpoint`) starts a new optimizer schedule. A first joint run starts directly from the base without automatically loading a prior TTS adapter.

#### Full QbyT adaptation with a frozen encoder

The [adaptation trainer](dma_kws/stage2/adapt.py) supports `adapt.method=qbyt_full` for joint and sequential data. This trains all QbyT parameters, freezes the encoder and any phoneme adapter, and disables auxiliary CTC. It does not inject LoRA. The existing `lora` method remains the default.

For an initial experiment on the same joint data, compose the [full-QbyT overlay](configs/experiment/adapt_qbyt_full.yaml) last:

```bash
PYTHONPATH=. .venv/bin/python scripts/run_keyword_adaptation.py \
  '+experiment=[icefall_zipformer_stage2_eps_softmin_v41,adapt_joint,adapt_qbyt_full]' \
  adapt.stage=train \
  prep.stage2_ckpt=/path/to/current/joint/stage2_adapted.pt \
  adapt.data_root=/path/to/existing/adapt/hey_eva
```

Use a complete Stage II checkpoint or a merged LoRA `stage2_adapted.pt` as the starting model. Loading a merged model preserves the LoRA update in the ordinary QbyT weights and starts a new optimizer/scheduler. Raw adapter files and unmerged LoRA Lightning checkpoints are rejected; merge a full LoRA checkpoint first with the [checkpoint converter](scripts/convert_stage2_checkpoints.py). Keep the experiment's readout and streaming configuration consistent with the starting checkpoint, including the v4.1 pooling fields.

The [overlay](configs/experiment/adapt_qbyt_full.yaml) starts at learning rate `3e-5`, 100 warmup steps and 1000 optimizer steps, with validation every 100 training microbatches per rank. These are experiment starting values, not measured optimal settings. The validation interval must remain divisible by `stage2.accumulate_grad_batches`. Joint data, sampling proportions, losses and source diagnostics are shared with LoRA. Compare checkpoints at a fixed false-positive budget on held-out validation data; the default checkpoint monitor remains real-speech AUC and does not enforce a FA/h constraint.

The [path resolver](dma_kws/stage2/adapt_paths.py) uses `paths.exp_root/stage2_adapt_joint_qbyt_full/<slug>` for joint full-QbyT runs and `paths.exp_root/stage2_adapt_qbyt_full/<slug>` for sequential full-QbyT runs. Data manifests are shared with LoRA, while default model and sweep outputs are separate. The trainer saves a complete model at `stage2_adapted.pt` and a per-phase copy at `<phase>/stage2_adapted.pt`; it produces no LoRA adapter file. In sequential TTS → real training, the real phase starts from the TTS full model.

Resume an interrupted full-QbyT run using its Lightning `.ckpt`, preserving the original data and experiment settings:

```bash
PYTHONPATH=. .venv/bin/python scripts/adapt_stage2_keyword.py \
  '+experiment=[icefall_zipformer_stage2_eps_softmin_v41,adapt_joint,adapt_qbyt_full]' \
  adapt.data_root=/path/to/existing/adapt/hey_eva \
  run.resume_from=/path/to/full_qbyt_run/last.ckpt
```

The [resume hooks](dma_kws/stage2/adapt.py) restore the complete training state, validate the frozen feature trunk and reject cross-method resumes. Full-state resume requires the same optimizer, learning rate, weight decay, warmup and schedule horizon; change these through a new weights-only run. To switch from LoRA to full QbyT, start a new run from merged model weights instead. The joint sampler's existing data-signature and accumulation-boundary checks still apply. The [sweep implementation](dma_kws/stage2/sweep_adapt.py) supports all three methods. Full-QbyT trials search learning rate and update budget without rank/alpha; encoder/full-QbyT trials additionally search an independent encoder learning rate. Method-tagged studies and parameter files prevent accidental mixing of results.


#### Full encoder and QbyT adaptation

The [adaptation trainer](dma_kws/stage2/adapt.py) also supports `adapt.method=encoder_qbyt_full`. It trains the complete Stage II encoder (including input subsampling) and all QbyT parameters without LoRA. An enabled phoneme adapter remains frozen in evaluation mode with auxiliary CTC disabled; gradients still pass through it to the encoder. Existing joint fbank manifests and background fbank caches remain usable because they contain input features, not frozen encoder outputs.

Compose the [encoder/full-QbyT experiment overlay](configs/experiment/adapt_encoder_qbyt_full.yaml) last:

```bash
PYTHONPATH=. .venv/bin/python scripts/run_keyword_adaptation.py \
  '+experiment=[icefall_zipformer_stage2_eps_softmin_v41,adapt_joint,adapt_encoder_qbyt_full]' \
  adapt.stage=train \
  prep.stage2_ckpt=/path/to/current/joint/stage2_adapted.pt \
  adapt.data_root=/path/to/existing/adapt/hey_eva
```

`adapt.learning_rate` controls QbyT; `adapt.encoder_learning_rate` controls the encoder. Initial experiment values are `3e-5` and `3e-6`, with 100 warmup steps, 1000 optimizer steps and validation every 100 training microbatches per rank. These are unvalidated starting values. Keep data, sampling and update budget fixed when comparing against `qbyt_full`; select thresholds on validation data and compare real-speech recall at the same false-positive budget on separate test data. The default checkpoint monitor remains real-speech AUC.

For Icefall, the [encoder schedule](dma_kws/stage2/encoder_schedule.py) drives the [wrapper's batch-count hook](dma_kws/stage2/icefall_encoder.py). `adapt.encoder_schedule.start_batch_count=100000` starts in the fine-tuning portion of the schedules; `reference_duration=600` sets the seconds per normalized batch. Progress uses globally summed, unpadded fbank frames and the configured frame shift. Every consumed training microbatch advances this duration clock, independently of optimizer accumulation; validation does not. This is the adaptation duration-clock contract, not a reproduction of the upstream maximum-batch-duration estimate.

New-mode Lightning checkpoints retain the encoder schedule and [per-rank Python, NumPy, Torch and CUDA random state](dma_kws/training/random_state.py). Resume rejects incompatible optimizer groups, schedule settings, world size or accelerator type (CPU versus CUDA), and checkpoints must be taken after completed optimizer updates. Use weights-only initialization when changing accelerator type. Joint mode additionally restores the deterministic data cursor. Sequential mode retains its existing data-loader behavior; restoring model and optimizer state there does not promise the identical subsequent sample sequence. Inference `.pt` exports contain model weights and provenance rather than end-of-run training counters.

The [path resolver](dma_kws/stage2/adapt_paths.py) isolates joint outputs under `paths.exp_root/stage2_adapt_joint_encoder_qbyt_full/<slug>` and sequential outputs under `paths.exp_root/stage2_adapt_encoder_qbyt_full/<slug>`. Both the run checkpoint directory and stable `stage2_adapted.pt` outputs contain the updated encoder and QbyT. Deployment must load this complete model, as [Stage2Verifier](dma_kws/inference/stage2_verifier.py) does; retaining the old encoder would discard part of the adaptation.

Initialize from complete Stage II weights, a merged LoRA model, or a full-QbyT model. Switching methods starts a new optimizer/scheduler. Full-state resume uses `run.resume_from=/path/to/last.ckpt` with the same method and configuration, rather than `prep.stage2_ckpt`; it restores both optimizer parameter groups. Sequential TTS → real training passes the complete trained model to the next phase. The [checkpoint converter](scripts/convert_stage2_checkpoints.py) accepts this mode's Lightning checkpoints and preserves all trained model tensors.

### Console output

All four adaptation scripts print rich progress and summary tables: a plan table before any heavy work, G2P/fbank progress bars during preparation, dataset composition and trainable parameter budget before training, the resolved run summary (`Stage II LoRA Adaptation Run`, `Stage II Full QbyT Adaptation Run`, or `Stage II Encoder + QbyT Adaptation Run`), per-trial sweep scores, and a base-vs-adapted metric comparison with deltas at eval time. Each table is followed by the machine-readable JSON/YAML line the scripts have always emitted.

Set `prep.use_rich=false` for plain text; output also degrades to plain text automatically when stdout is not a TTY (piped output, CI logs, captured subprocesses).

### Metrics logs (CSV / TensorBoard)

Stage II training and LoRA adaptation write metrics through the backends in `stage2.logging.backends` (default `[csv, tensorboard]`; W&B and Trackio optional). Per run (`logs/<run_name>/version_N/`):

- `metrics.csv` / TensorBoard events: Lightning's native step-level stream, with `train/loss`, `train/utt_loss`, `train/seq_loss`, `train/lr`, `train/grad_norm`, and all `val/*` metrics. Adaptation additionally logs per-source training metrics: `train/keyword_utt_loss`, `train/libri_utt_loss`, and `train/keyword_frac` (actual keyword share per batch). [Hyperparameters](dma_kws/training/metrics_history.py) (lr, batch size, max_steps, seed; plus adapt_method/mix_ratio/keyword/phase for adaptation and rank/alpha for LoRA only) are logged once at startup, so the TensorBoard HPARAMS tab is populated. Set `stage2.logging.grad_norm=false` to disable gradient-norm logging.
- `eval_history.csv`: one dense row per validation pass (no sparse columns), with step, epoch, wall time, steps/sec, the latest train metrics, and every val metric. Use this for within-run comparison and plotting.

Cross-run comparison: every completed run appends one row (timestamp, run name, hyperparameters, final and best val metrics, step count, duration) to `exp/stage2_qbyt/runs.csv` (Stage II) or `exp/stage2_adapt/runs.csv` (adaptation). Best-metric direction is inferred per metric (AUC-like → max, EER/loss → min).

Note: `val_auc` (an alias of `val/auc` used only for checkpoint filenames) is no longer written to CSV/TensorBoard, and the redundant `lr-Adam` column from `LearningRateMonitor` was removed in favor of `train/lr`.

### Data preparation

**Prerequisites**

| Item | Purpose |
|------|---------|
| SI Stage II checkpoint | Base weights to adapt (`prep.stage2_ckpt` or `adapt.init_checkpoint`); same format as normal Stage II export (`stage2_*.pt`) |
| Keyword audio | TTS synthetic + real recordings, positives and confusable negatives |
| LibriPhrase parquet + fbank | Anti-forgetting mix during training (`stage2.parquet_file`, `stage2.wav_dir`); same as standard Stage II training |
| LibriPhrase eval (optional) | LPH hard-split monitoring during training and in `eval_report.json` |

**Directory layout (recommended)**

Place train wav files directly under each `raw/<phase>/` using polarity and negative phrase slug. Place held-out wav files under the phase's `eval/` subtree with the same polarity layout. The slug is derived automatically from `adapt.keyword` unless you override `adapt.data_root`.

Example for `adapt.keyword="hey eva"`:

```text
data/dma-kws/processed/adapt/hey_eva/
├── raw/
│   ├── tts/
│   │   ├── positive/              # train: label=1, text="hey eva"
│   │   │   └── *.wav
│   │   ├── negative/
│   │   │   ├── hey_ava/           # train: label=0, text="hey ava"
│   │   │   │   └── *.wav
│   │   │   └── hey_eve/
│   │   │       └── *.wav
│   │   └── eval/
│   │       ├── positive/          # eval: label=1, text="hey eva"
│   │       │   └── *.wav
│   │       └── negative/
│   │           └── hey_ava/       # eval: label=0, text="hey ava"
│   │               └── *.wav
│   └── real/
│       ├── positive/              # train
│       │   └── *.wav
│       ├── negative/
│       │   └── hey_ava/
│       │       └── *.wav
│       └── eval/
│           ├── positive/          # eval
│           │   └── *.wav
│           └── negative/
│               └── hey_ava/
│                   └── *.wav
├── fbank/                         # written by prepare script (mirrors raw tree, .npy)
└── manifests/                     # written by prepare script
    ├── tts_train.csv
    ├── tts_eval.csv
    ├── real_train.csv
    └── real_eval.csv
```

**Label and text rules**

| Location | `label` | `text` (G2P query) |
|----------|---------|---------------------|
| `<phase>/positive/*.wav` and `<phase>/eval/positive/*.wav` | `1` | `adapt.keyword` exactly (e.g. `hey eva`) |
| `<phase>/negative/<slug>/*.wav` and `<phase>/eval/negative/<slug>/*.wav` | `0` | folder slug with `_` → spaces (e.g. `hey_ava` → `hey ava`) |

Files directly under each phase's `positive/` and `negative/` trees are written to the train manifest. Files under the phase's `eval/` tree are written only to the eval manifest. All `text` values are validated through G2P at prepare time; invalid phrases fail early.

**Prepare command**

Runs explicit directory split → G2P check → Kaldi fbank (from `configs/fbank/default.yaml`, 80-dim, dither 0.1) → train/eval manifests:

```bash
python3 scripts/prepare_keyword_adaptation.py adapt.keyword="hey eva"
```

Useful overrides:

| Override | Default | Meaning |
|----------|---------|---------|
| `adapt.eval_fraction` | `0.2` | Held-out fraction per phase only when using `prep.manifest_csv` |
| `adapt.eval_seed` | `2025` | Shuffle seed only when using `prep.manifest_csv` |
| `prep.no_skip_existing=true` | off | Recompute all fbank even if `.npy` exists |
| `adapt.data_root=...` | `${paths.processed_root}/adapt/<slug>` | Custom data root |
| `prep.manifest_csv=...` | — | Skip directory scan; CSV with columns `audio_path,text,label` |

**Manifest columns**

- **Train** (`{phase}_train.csv`): `audio_path,text,label`, with `audio_path` relative to `data_root`.
- **Eval** (`{phase}_eval.csv`): `audio_path,text,keyword,label`, used for target-word validation and `eval_stage2_clips`.

**Training phase order (paper III-E)**

1. `adapt.phase=tts`: LoRA on synthetic data (+ LibriPhrase 1:1 mix).
2. `adapt.phase=real`: continue LoRA on real data; loads TTS adapter weights on top of the **same SI base checkpoint** (do not pass the merged TTS checkpoint as `prep.stage2_ckpt`).

### Config reference

Defaults live in `configs/adapt/default.yaml`. Override on the CLI with `adapt.<key>=...` or `prep.<key>=...`.

**Identity and paths**

| Key | Default | Description |
|-----|---------|-------------|
| `adapt.keyword` | `hey eva` | Wake phrase; drives anchor G2P and slug |
| `adapt.slug` | `""` | Filesystem slug override; empty → auto from keyword |
| `adapt.data_root` | `""` | Data directory; empty → `paths.processed_root/adapt/<slug>` |
| `adapt.exp_root` | `""` | Experiment output; empty → `paths.exp_root/stage2_adapt/<slug>` |
| `adapt.phase` | `tts` | Training phase: `tts` or `real` |
| `adapt.stage` | `all` | Orchestrator stage: `prepare`, `sweep`, `train`, `eval`, or `all` |

**LoRA and optimizer** (paper defaults: rank 16, lr 4e-4, alpha = 2×rank)

| Key | Default | Description |
|-----|---------|-------------|
| `adapt.rank` | `16` | LoRA rank on QbyT matching projections |
| `adapt.alpha` | `32` | LoRA scaling (`alpha/rank` applied to `B@A`) |
| `adapt.learning_rate` | `4e-4` | Adam learning rate (LoRA params only); canonical key |
| `adapt.lr` | unset | Alias for `adapt.learning_rate`; wins when explicitly set |
| `adapt.optimizer` | `adam` | `adam` or `adamw` |
| `adapt.weight_decay` | `0` | Weight decay (AdamW only) |
| `adapt.warmup_steps` | `100` | Cosine schedule warmup |
| `adapt.max_steps` | `3000` | Training steps per phase |
| `adapt.lora_targets` | `audio_key.weight`, `text_query.weight` | Projections that build the phone/frame emission lattice |

**Data mixing and loading**

| Key | Default | Description |
|-----|---------|-------------|
| `adapt.mix_ratio` | `0.5` | Keyword : LibriPhrase sampling ratio (0.5 = 1:1 anti-forgetting) |
| `adapt.sample_lens` | `3000` | Virtual epoch length (random resampling) |
| `adapt.batch_size_per_gpu` | `64` | Train batch size |
| `adapt.val_batch_size` | `64` | Target-keyword val batch size |
| `adapt.num_workers` | `2` | Train DataLoader workers |
| `adapt.val_num_workers` | `null` | Target-keyword val workers (`null` = `min(num_workers, 4)`) |

**Checkpoint inputs**

| Key | Description |
|-----|-------------|
| `prep.stage2_ckpt` | **Required** SI Stage II base checkpoint for adaptation |
| `adapt.init_checkpoint` | Alternative to `prep.stage2_ckpt` |
| `adapt.params_file` | YAML with sweep best params (e.g. `exp/stage2_adapt/<slug>/sweep/best_params.yaml`) |
| `run.limit_steps` | Cap steps (smoke tests); overrides `adapt.max_steps` when set |

**Optuna sweep** (`adapt.sweep.*`; install `pip install -e '.[adapt]'`)

| Key | Default | Description |
|-----|---------|-------------|
| `adapt.sweep.enabled` | `false` | Run sweep before training in `adapt.stage=all` |
| `adapt.sweep.n_trials` | `20` | Number of Optuna trials |
| `adapt.sweep.lambda_forget` | `1.0` | Penalty weight: `score = target_auc − λ × max(0, lph_base − lph_adapted)` |
| `adapt.sweep.lph_subset` | `2000` | LibriPhrase pairs for fast LPH eval during sweep |
| `adapt.sweep.single_phase` | `false` | TTS-only trials for quick search |
| `adapt.sweep.search_mix` | `false` | Also search `mix_ratio` |
| `adapt.sweep.storage` | auto | SQLite path (`exp/stage2_adapt/<slug>/sweep/optuna.db`) |

The sweep searches `rank`, `alpha_ratio` (`alpha = alpha_ratio × rank`), `learning_rate`, `max_steps`, and optionally `mix_ratio`. Best params are written to `exp/stage2_adapt/<slug>/sweep/best_params.yaml` using `adapt.*` config keys, and are picked up automatically by `adapt_stage2_keyword.py` (or explicitly via `adapt.params_file=...`).

> **Re-sweep after upgrading:** earlier revisions dropped the searched learning rate (shadowed by the `adapt.lr` default) and the derived `alpha` (persisted as `alpha_ratio`), so every trial effectively trained at `lr=4e-4, alpha=32`. Both are fixed, and old `best_params.yaml` files are normalized on load, but sweep results produced before the fix only reflect `rank`/`max_steps` and are worth re-running.

**Example: adapt a new wake word on a shared server**

```bash
python3 scripts/run_keyword_adaptation.py \
  adapt.keyword="hi lumina" \
  prep.stage2_ckpt=/data/exp/stage2_si.pt \
  paths.processed_root=/data/dma-kws/processed \
  paths.exp_root=/data/dma-kws/exp \
  stage2.parquet_file=/data/dma-kws/processed/stage2_qbyt/aggregated_segments_with_g2p_distance.parquet \
  stage2.wav_dir=/data/dma-kws/features/fbank \
  adapt.stage=all
```

Place wavs under `/data/dma-kws/processed/adapt/hi_lumina/raw/...` before running. After training, use `exp/stage2_adapt/hi_lumina/stage2_adapted.pt` as `prep.stage2_ckpt` in [§9](#9-run-the-two-stage-demo) or [§10](#10-stage-ii-only-clip-inference).

---

## 9. Run the two-stage demo

After both stages have checkpoints:

```bash
STAGE1_CKPT=data/dma-kws/exp/stage1_phoneme_ctc/checkpoints/stage1_step000020.pt
STAGE2_CKPT=data/dma-kws/exp/stage2_qbyt/checkpoints/stage2_step000020.pt
AUDIO=/path/to/test.wav
KEYWORD="hello world"

python3 scripts/run_two_stage_demo.py \
  +experiment=demo_librispeech100 \
  prep.stage1_ckpt="$STAGE1_CKPT" \
  prep.stage2_ckpt="$STAGE2_CKPT" \
  prep.audio="$AUDIO" \
  prep.keyword="$KEYWORD" \
  demo.qbyt_threshold=0.6
```

The decision threshold defaults to `0.5` in config; override on the CLI as above or in yaml:

```yaml
demo:
  qbyt_threshold: 0.5
  # Shortest candidate that gets scored, in *encoder* frames. The fbank length
  # this implies is derived from the configured encoder, because each backend
  # subsamples differently: one encoder frame needs 7 fbank frames on the Wenet
  # Conformer but 9 on the icefall Zipformer.
  min_stage2_encoder_frames: 1
```

`demo.min_stage2_fbank_frames` was removed and now raises. It hardcoded the Wenet figure, so an 65 to 85 ms candidate passed the guard and then subsampled to zero frames inside icefall's `Conv2dSubsampling` (`(T-7)//2`), crashing the convolution.

Output is JSON:

```json
{
  "audio": "/path/to/test.wav",
  "keyword": "hello world",
  "keyword_phonemes": ["HH", "AH", "L", "OW", "W", "ER", "L", "D"],
  "decoded_phonemes": ["..."],
  "stage1_candidates": [
    {
      "start_sec": 1.24,
      "end_sec": 2.03,
      "stage1_score": -3.82,
      "phonemes": ["HH", "AH", "L", "OW"]
    }
  ],
  "stage2_scores": [
    {
      "start_sec": 1.24,
      "end_sec": 2.03,
      "stage1_score": -3.82,
      "qbyt_score": 0.87
    }
  ],
  "threshold": 0.5,
  "detected": true
}
```

---

## 10. Stage II-only clip inference

Use this path when each input audio file is already a cropped keyword clip and you want to test the Stage II QbyT verifier without any Stage I locator. This is useful for LibriPhrase-style positive clips, Stage II ablations, and quick checkpoint smoke tests when no Stage I checkpoint is available.

This is not keyword localization. If you pass a long utterance, the script scores the whole utterance against the keyword text; it does not search for the keyword inside the utterance.

| Script | Input | Output |
|--------|-------|--------|
| `scripts/eval_stage2_libriphrase.py` | LibriPhrase anchor/comparison pairs with precomputed fbank | AUC/EER benchmark metrics |
| `scripts/run_two_stage_demo.py` / `scripts/eval_two_stage_kws.py` | Long audio + keyword, with a Stage I locator | Candidate spans plus Stage II scores |
| `scripts/run_stage2_demo.py` / `scripts/eval_stage2_clips.py` | Pre-cropped keyword clips + keyword | Full-clip `qbyt_score` and detected flag |

Single clip:

```bash
python3 scripts/run_stage2_demo.py \
  +experiment=wenet_asr_stage2 \
  prep.stage2_ckpt=data/dma-kws/exp/stage2_qbyt/checkpoints/stage2_step050000.pt \
  prep.audio=/path/keyword_clip.wav \
  prep.keyword="hey eva" \
  demo.qbyt_threshold=0.5
```

Only `prep.stage2_ckpt`, `prep.audio`, and `prep.keyword` are required. Do not pass `prep.stage1_ckpt` or `+locator=...`; this path bypasses Stage I entirely.

Output is JSON:

```json
{
  "audio": "/path/keyword_clip.wav",
  "keyword": "hey eva",
  "keyword_phonemes": ["HH", "EY", "IY", "V", "AH"],
  "clip_span_sec": {
    "start_sec": 0.0,
    "end_sec": 1.23
  },
  "qbyt_score": 0.87,
  "threshold": 0.5,
  "detected": true,
  "skipped": false
}
```

Batch evaluation uses the same manifest format as the two-stage batch runner: `audio_path`, `keyword`, and optional `label`. In this mode, each `audio_path` must point to a cropped clip.

`scripts/prepare_two_stage_manifest.py` now supports two manifest build modes:

- **single** (default): one `prep.keyword` shared by all audio files.
- **auto_assign**: infer per-file keyword from filename, then assign labels via `prep.keyword_labels`.

`auto_assign` filename rule:

1. Take file stem (without extension).
2. Split by `_`.
3. Drop the last segment.
4. Join remaining segments and match against `prep.keywords` after normalization (lowercase, ignore spaces/underscores).

Example: `hey_eva_001.wav` -> candidate `heyeva`, matches `"hey eva"`.

```bash
# Single keyword mode (backward-compatible)
python3 scripts/prepare_two_stage_manifest.py \
  prep.input_dir=/path/to/keyword_clips \
  prep.keyword="hey eva" \
  prep.output=/path/stage2_clip_manifest.csv \
  prep.label=1

# Multi-keyword auto assignment mode
python3 scripts/prepare_two_stage_manifest.py \
  prep.input_dir=/path/to/keyword_clips \
  prep.keyword_mode=auto_assign \
  prep.keywords='["hey eva","ok lamp","wake up"]' \
  prep.keyword_labels='{"hey eva":1,"ok lamp":0,"wake up":1}' \
  prep.skip_unmatched=true \
  prep.output=/path/stage2_clip_manifest.csv

python3 scripts/eval_stage2_clips.py \
  +experiment=wenet_asr_stage2 \
  prep.manifest=/path/stage2_clip_manifest.csv \
  prep.stage2_ckpt=data/dma-kws/exp/stage2_qbyt/checkpoints/stage2_step050000.pt
```

In `auto_assign` mode, unmatched files are skipped when `prep.skip_unmatched=true` (default), and the script summary reports counts and examples of skipped files.

Default output directory: `outputs/eval_stage2_clips` (`prep.stage2_clip_output_dir`). Override with `prep.output_dir=/path/to/output` if needed. The script writes `results.jsonl` with per-clip scores and `summary.json` with accuracy, precision, recall, f1, auc, and eer when labels are present. With both classes present and `prep.plot_curves=true` (the default) it also writes `roc_curve.png`, `det_curve.png`, and `roc_curve.csv` (`threshold`, `tpr`, `fpr` for every empirical operating point). If a manifest row contains extra columns beyond `audio_path`, `keyword`, and optional `label`, those key-value pairs are copied into `results.jsonl` under `manifest_meta`.

### Multi-keyword / multi-pronunciation (`prep.keyword_eval.mode=any`)

Default clip and MUSAN eval is still one query per row (`mode=per_row`). Opt in with the `keyword_eval` Hydra group. `+keyword_eval=hey_eva_variants` enrolls two hey-eva pronunciations; `+keyword_eval=multi_wakeup` adds `ok lamp` with G2P. Each clip or MUSAN window is encoded once, each pronunciation is scored, then `qbyt_score` is the max over keywords and pronunciations. Several hits on one audio still count as one detection. Too-short clips are `skipped=true` and never detect, even at threshold 0. LibriPhrase and two-stage KWS reject `mode=any`.

Labeled any-mode manifests use one source-audio row, not a keyword pair:

```json
{"audio_path":"audio/eva.wav","keyword_labels":{"hey eva":1,"ok lamp":0}}
{"audio_path":"audio/neg.wav","label_scope":"target_set","target_texts":["hey eva","ok lamp"],"label":0}
```

CSV cells for `keyword_labels` / `target_texts` are JSON. Unlabeled rows may omit every label field. Do not set `prep.keyword`, `prep.keyword_phonemes`, or `prep.keywords` together with `mode=any`.

Config compose (checked locally; no checkpoint required):

```bash
PYTHONPATH=. python3 scripts/eval_stage2_clips.py \
  +experiment=icefall_zipformer_stage2_eps_softmin_v41 \
  +keyword_eval=multi_wakeup \
  --cfg job --resolve
```

That resolve keeps the v4.1 readout (`eps_softmin`, `sink_token: true`) and other prep fields, with `prep.keyword_eval.mode=any` and both target texts. Real scoring still needs `prep.manifest` / `prep.stage2_ckpt` (clips) or `prep.musan_root` (MUSAN). MUSAN `metrics.fa_per_hour` is over-threshold windows / source-file hours (`fa_count_unit=window`). `file_metrics.file_trigger_rate` is triggered files / scored files, not event FA/h.

Clips are scored in padded GPU batches, with audio loading and fbank extraction parallelized across DataLoader workers and G2P/tokenization cached per unique keyword. Tune with:

- `prep.batch_size`: clips per forward pass (default 64 when unset/0).
- `prep.num_workers`: feature-extraction workers (default `min(8, cpu_count)` when unset/0).

```bash
python3 scripts/eval_stage2_clips.py \
  +experiment=wenet_asr_stage2 \
  prep.manifest=/path/stage2_clip_manifest.csv \
  prep.stage2_ckpt=data/dma-kws/exp/stage2_qbyt/checkpoints/stage2_step050000.pt \
  prep.batch_size=128 \
  prep.num_workers=8
```
### Configurable waveform robustness with MUSAN and synthetic noise

`scripts/eval_stage2_clips.py` can independently enable continuous MUSAN noise, music, overlapping speech, synthetic stationary noise, MUSAN burst noise, and clean-signal volume variation. Every effect is disabled by default, so existing clean evaluation is unchanged. Continuous noise/music/speech and burst noise read only their matching `<musan_root>/noise/**`, `<musan_root>/music/**`, or `<musan_root>/speech/**` subset. Synthetic stationary noise and volume variation are source-free and do not require `prep.musan_root`.

Noise-only evaluation at 10 dB SNR:

```bash
python3 scripts/eval_stage2_clips.py \
  +experiment=wenet_asr_stage2 \
  prep.manifest=/path/stage2_clip_manifest.csv \
  prep.stage2_ckpt=/path/stage2.pt \
  prep.musan_root=/path/to/musan \
  prep.musan_mix.noise.enabled=true \
  prep.musan_mix.noise.snr_db=10
```

Set `prep.musan_mix.noise.snr_db=20` for a 20 dB condition. The value is `20*log10(RMS(clean)/RMS(noise))`, so a larger SNR means quieter noise. Music uses the same definition through `prep.musan_mix.music.snr_db`.

Noise and music together:

```bash
python3 scripts/eval_stage2_clips.py \
  +experiment=wenet_asr_stage2 \
  prep.manifest=/path/stage2_clip_manifest.csv \
  prep.stage2_ckpt=/path/stage2.pt \
  prep.musan_root=/path/to/musan \
  prep.musan_mix.seed=2025 \
  prep.musan_mix.noise.enabled=true \
  prep.musan_mix.noise.snr_db=10 \
  prep.musan_mix.music.enabled=true \
  prep.musan_mix.music.snr_db=10
```

Noise and music SNR values are applied independently against the same clean RMS after optional volume variation. The mixer does not force their combined interference to a separate total SNR.

Speech-only evaluation:

```bash
python3 scripts/eval_stage2_clips.py \
  +experiment=wenet_asr_stage2 \
  prep.manifest=/path/stage2_clip_manifest.csv \
  prep.stage2_ckpt=/path/stage2.pt \
  prep.musan_root=/path/to/musan \
  prep.musan_mix.speech.enabled=true \
  prep.musan_mix.speech.relative_db=-6
```

`prep.musan_mix.speech.relative_db` controls speech RMS relative to the clean clip: a negative value is quieter, `0` is equal, and a positive value is louder. For example, `-6`, `0`, and `+6` dB are approximately `0.5x`, `1x`, and `2x` the clean RMS.

Enable noise and overlapping speech together when needed:

```bash
python3 scripts/eval_stage2_clips.py \
  +experiment=wenet_asr_stage2 \
  prep.manifest=/path/stage2_clip_manifest.csv \
  prep.stage2_ckpt=/path/stage2.pt \
  prep.musan_root=/path/to/musan \
  prep.musan_mix.seed=2025 \
  prep.musan_mix.noise.enabled=true \
  prep.musan_mix.noise.snr_db=20 \
  prep.musan_mix.speech.enabled=true \
  prep.musan_mix.speech.relative_db=0
```

Continuous MUSAN noise, music, and speech are each scaled against the same clean RMS after optional volume variation, then summed without clipping. Long MUSAN sources are cropped and short sources are repeated to the clip length. For negative-set experiments, avoid overlapping speech that contains the enrolled wake word; [the MUSAN WeNet filter](scripts/filter_musan_by_wenet_asr.py) can build a filtered corpus tree.

#### Synthetic stationary noise

Enable deterministic white Gaussian noise at 10 dB SNR without a MUSAN corpus:

```bash
python3 scripts/eval_stage2_clips.py \
  +experiment=wenet_asr_stage2 \
  prep.manifest=/path/stage2_clip_manifest.csv \
  prep.stage2_ckpt=/path/stage2.pt \
  prep.musan_mix.stationary_noise.enabled=true \
  prep.musan_mix.stationary_noise.snr_db=10
```

Use `prep.musan_mix.stationary_noise.snr_db=20` for the corresponding 20 dB run. The first implementation intentionally supports only `kind=white_gaussian`. Each row gets a deterministic zero-mean realization, scaled to the exact configured RMS ratio. Stationary-noise-only evaluation does not require `prep.musan_root`.

#### MUSAN burst noise

A single 100–400 ms burst at 10 dB uses the declared defaults:

```bash
python3 scripts/eval_stage2_clips.py \
  +experiment=wenet_asr_stage2 \
  prep.manifest=/path/stage2_clip_manifest.csv \
  prep.stage2_ckpt=/path/stage2.pt \
  prep.musan_root=/path/to/musan \
  prep.musan_mix.burst_noise.enabled=true \
  prep.musan_mix.burst_noise.snr_db=10
```

For two to four non-overlapping events with an explicit gap and duration range:

```bash
python3 scripts/eval_stage2_clips.py \
  +experiment=wenet_asr_stage2 \
  prep.manifest=/path/stage2_clip_manifest.csv \
  prep.stage2_ckpt=/path/stage2.pt \
  prep.musan_root=/path/to/musan \
  prep.musan_mix.burst_noise.enabled=true \
  prep.musan_mix.burst_noise.event_count_min=2 \
  prep.musan_mix.burst_noise.event_count_max=4 \
  prep.musan_mix.burst_noise.duration_ms_min=80 \
  prep.musan_mix.burst_noise.duration_ms_max=250 \
  prep.musan_mix.burst_noise.fade_ms=10 \
  prep.musan_mix.burst_noise.allow_overlap=false \
  prep.musan_mix.burst_noise.min_gap_ms=50
```

`snr_scope=active_event` scales every faded event so its event-interval RMS has the configured ratio to the whole post-volume clean RMS. Event duration therefore does not change its instantaneous level. `snr_scope=whole_clip` first combines all events and intervening silence, then scales that full burst bed to the exact whole-clip SNR; sparse events will consequently be louder. With `allow_overlap=false`, `min_gap_ms` is enforced and an impossible placement fails explicitly instead of silently changing the recipe.

#### Alternating clean-signal volume

```bash
python3 scripts/eval_stage2_clips.py \
  +experiment=wenet_asr_stage2 \
  prep.manifest=/path/stage2_clip_manifest.csv \
  prep.stage2_ckpt=/path/stage2.pt \
  prep.musan_mix.volume_variation.enabled=true \
  prep.musan_mix.volume_variation.low_gain_db=-12 \
  prep.musan_mix.volume_variation.high_gain_db=6 \
  prep.musan_mix.volume_variation.segment_ms_min=250 \
  prep.musan_mix.volume_variation.segment_ms_max=750 \
  prep.musan_mix.volume_variation.transition_ms=50
```

The clean signal alternates between the low and high dB gains in deterministic variable-length segments. Segment boundaries use a raised-cosine transition in the dB domain; `transition_ms` must not exceed `segment_ms_min`, which keeps every transition complete and continuous. The sample count does not change, and this effect also works without `prep.musan_root`.

#### Combining the new effects

All three new effects, and the existing continuous MUSAN components, can be enabled independently or together. For example:

```bash
python3 scripts/eval_stage2_clips.py \
  +experiment=wenet_asr_stage2 \
  prep.manifest=/path/stage2_clip_manifest.csv \
  prep.stage2_ckpt=/path/stage2.pt \
  prep.musan_root=/path/to/musan \
  prep.musan_mix.seed=2025 \
  prep.musan_mix.volume_variation.enabled=true \
  prep.musan_mix.stationary_noise.enabled=true \
  prep.musan_mix.stationary_noise.snr_db=20 \
  prep.musan_mix.burst_noise.enabled=true \
  prep.musan_mix.burst_noise.snr_db=10 \
  prep.musan_mix.burst_noise.event_count_min=2 \
  prep.musan_mix.burst_noise.event_count_max=3
```

The fixed MUSAN-mixer order is:

```text
prepared mono clean -> volume_variation
                    -> independently scale continuous noise/music/speech,
                       stationary noise, and burst noise against that same clean
                    -> sum once -> existing zero-valued context padding
```

Every additive component independently satisfies its configured ratio, so two 10 dB components do not imply a 10 dB total-interference SNR. The mixer intentionally applies no hard clipping or limiter because either would change those ratios; temporary floating-point values outside `[-1, 1]` are retained.

All recipe seeds derive from `prep.musan_mix.seed`, manifest row, audio path, component, and burst-event index. Enabling one component therefore does not change another component's source or placement, and results are stable across batch sizes and DataLoader worker counts. Mixing happens in memory after mono conversion and resampling, before the existing zero padding; source files are never modified. When an SNR-controlled additive component is active, a zero-RMS clean reference or selected MUSAN segment fails explicitly because its requested dB ratio is undefined.

`summary.json` records the resolved configuration, SNR scope, processing order, and MUSAN source-pool sizes. Each augmented `results.jsonl` row contains a top-level `musan_mix` recipe. Burst recipes include source, start fraction, duration, source-offset fraction, and event seed; volume variation records its gain/segment range and envelope seed.

### audio_aug-compatible waveform transformations

`eval_stage2_clips.py` also provides the nine waveform transformations from [`audio_aug` commit `58410f27`](https://github.com/ty4045681/audio_aug/tree/58410f27c5beecdb9fa438e98833daccb69db895), configured by the [audio augmentation configuration](configs/prep/default.yaml) and implemented by the [in-memory evaluator adapter](dma_kws/inference/audio_aug.py). Each method has its own `enabled` switch under `prep.audio_aug.transforms`; all nine default to `false`, so clean and MUSAN-only evaluations keep their previous waveform path.

| Method | Pipeline phase | Main parameters |
| --- | --- | --- |
| `speed_change` | pre-mix | `speed_factor` |
| `volume_gain` | pre-mix | `gain_db` |
| `noise_mix` | additive | `snr_db` |
| `amp_distortion` | post-mix | `distortion_type`, `rate`, subtype parameters |
| `subband_eq` | post-mix | `low_min_gain_db`, `high_min_gain_db` |
| `band_limit` | post-mix | `mode`, `cutoff_hz`, `filter_order`, `target_sample_rate` |
| `narrowband` | post-mix | `target_sample_rate` |
| `spectral_mask` | post-mix | mask counts and gain range |
| `signal_mimic` | post-mix | five child-stage probabilities |

Stationary Gaussian noise at an exact in-memory 10 dB RMS ratio:

```bash
python3 scripts/eval_stage2_clips.py \
  +experiment=wenet_asr_stage2 \
  prep.manifest=/path/stage2_clip_manifest.csv \
  prep.stage2_ckpt=/path/stage2.pt \
  prep.audio_aug.pcm_policy=float_unclipped \
  prep.audio_aug.transforms.noise_mix.enabled=true \
  prep.audio_aug.transforms.noise_mix.snr_db=10
```

The default `prep.audio_aug.transforms.noise_mix.snr_mode=exact_rms` scales the generated noise delta to the requested empirical RMS ratio before the PCM policy is applied. Set `snr_mode=upstream_std` to reproduce the source method's Gaussian standard-deviation semantics instead. The default `prep.audio_aug.pcm_policy=clip_round_each_stage` rounds and clips after every enabled stage to model the upstream PCM16 script workflow; clipping can change the final measured ratio. Use `float_unclipped` for controlled SNR sweeps without quantization or clipping.

Speed and whole-clip gain can be combined independently:

```bash
python3 scripts/eval_stage2_clips.py \
  +experiment=wenet_asr_stage2 \
  prep.manifest=/path/stage2_clip_manifest.csv \
  prep.stage2_ckpt=/path/stage2.pt \
  prep.audio_aug.transforms.speed_change.enabled=true \
  prep.audio_aug.transforms.speed_change.speed_factor=1.10 \
  prep.audio_aug.transforms.volume_gain.enabled=true \
  prep.audio_aug.transforms.volume_gain.gain_db=-3
```

The default `prep.audio_aug.speed_length_policy=variable` preserves the real speed-adjusted sample count. `results.jsonl` then records both the original `clip_span_sec` and padding-free `augmented_duration_sec`. Set `center_crop_or_zero_pad` when a fixed input duration is required; it center-crops slower/longer output and symmetrically zero-pads faster/shorter output instead of resampling away the speed and pitch change.

The fixed multi-effect order is:

```text
speed_change -> audio_aug volume_gain -> musan_mix volume_variation
             -> parallel audio_aug noise_mix and MUSAN continuous/stationary/burst deltas
             -> amp_distortion -> subband_eq -> band_limit -> narrowband
             -> spectral_mask -> signal_mimic
             -> existing zero-valued context padding
```

Every additive branch uses the same clean waveform after all enabled pre-mix volume transforms as its level reference. `signal_mimic` may internally invoke subband EQ, band limiting, narrowband conversion, or spectral masking, so enabling it together with those explicit methods is rejected by default. Set `prep.audio_aug.allow_signal_mimic_overlap=true` only when repeated degradation is intentional.

`prep.audio_aug.seed` is combined with the manifest row, audio path, and method name, making each method reproducible without coupling its recipe to other enabled methods. `summary.json` records the compatibility revision, NumPy/SciPy versions, resolved parameters, and execution order. Each augmented result records its method-local seed and resolved recipe under `audio_aug`.

To evaluate exported `.pt` checkpoints against a manifest, use `scripts/batch_eval_stage2_clips.sh`. Directory inputs scan immediate `*.pt` children; an optional `:OUT_DIR` sets that directory source's output root, otherwise the script writes to `<base-out>/<checkpoint-directory-name>/<checkpoint-stem>/`:

```bash
bash scripts/batch_eval_stage2_clips.sh \
  --dir /path/to/export1:/path/to/out1 \
  --dir /path/to/export2 \
  --manifest /path/to/merged.csv \
  --base-out /path/to/outputs \
  --experiment icefall_zipformer_stage2
```

To evaluate selected checkpoints, pass repeatable `--pt PT:OUT_DIR` arguments. Each explicit checkpoint writes directly to its declared `OUT_DIR`, without adding its filename:

```bash
bash scripts/batch_eval_stage2_clips.sh \
  --pt /path/to/export1/stage2_step010000.pt:/path/to/outputs/step10000 \
  --pt /path/to/export2/stage2_step020000.pt:/path/to/outputs/step20000 \
  --manifest /path/to/merged.csv \
  --experiment icefall_zipformer_stage2
```

For long lists, use `--dirs-file` (one `CKPT_DIR[:OUT_DIR]` per non-comment line) and `--pts-file` (one `PT:OUT_DIR` per non-comment line). Directory and explicit inputs may be combined; `--manifest` and `--experiment` apply to every checkpoint, while `--base-out` only supplies the fallback output root for directory entries without `:OUT_DIR`:

```bash
bash scripts/batch_eval_stage2_clips.sh \
  --dirs-file checkpoint_dirs.txt \
  --pts-file selected_checkpoints.txt \
  --manifest /path/to/merged.csv \
  --base-out /path/to/outputs \
  --experiment icefall_zipformer_stage2
```

Checkpoints run sequentially on one GPU; to use multiple GPUs, split the checkpoint sources and launch one invocation per GPU with `CUDA_VISIBLE_DEVICES`.

---

## 11. MUSAN false-accept evaluation

Hold speech out of training background negatives with a leakage-safe split:

```bash
python scripts/split_musan.py \
  --musan-root /path/to/musan \
  --output-dir /path/to/musan_split \
  --train-categories music,noise
```

`--train-categories` defaults to `music,noise,speech` (all three still split).
Omitting `speech` sends every speech file to eval so Stage II never trains on
it. The two lists must stay on opposite sides of the train/eval cut:

- `train_background.list` → `stage2.background_negative.audio_list_path`
- `eval_musan.list` → `prep.musan_audio_list_path` for `eval_musan_fa.py` and
  `batch_eval_musan_fa.sh --audio-list`

Do not pass `train_background.list` to eval. `split.json` records
`policy.train_categories` and catalog hashes so the cut is auditable. When
`prep.musan_audio_list_path` is set, noisy-speech `eval_condition` MUSAN mixing
also uses only those eval files.

`scripts/eval_musan_fa.py` and `scripts/batch_eval_musan_fa.sh` score Stage II
only: they slide a fixed window over every MUSAN file and treat every window as
a negative. The official grid is `prep.window_sec=3.0` / `prep.hop_sec=3.0`.
Equal window and hop means no overlap. FA/hour is **not** comparable across hops:
the numerator is the number of windows over the threshold, the denominator is
audio hours.

Each run writes `results.jsonl` and `summary.json`. With
`prep.plot_curves=true` (the default) it also writes a matching pair:

- `fa_per_hour_curve.png` — threshold versus FA/hour, with the deployment
  threshold marked
- `fa_per_hour_curve.csv` — the same points (`threshold`, `false_accepts`,
  `fa_per_hour`, `fa_per_24_hours`, `fa_per_1000_hours`)

`summary.json` reports overall and per-subset (`music`/`noise`/`speech`)
FA/hour. Each result row keeps the keyword phonemes and the single deployed
Stage-II score.

Single keyword, single checkpoint:

```bash
python3 scripts/eval_musan_fa.py \
  +experiment=icefall_zipformer_stage2 \
  prep.keyword="hey eva" \
  'prep.keyword_phonemes=HH EY1 IY1 V AH0' \
  prep.musan_root=/path/to/musan \
  prep.stage2_ckpt=/path/to/stage2_step020000.pt \
  prep.window_sec=3.0 \
  prep.hop_sec=3.0 \
  prep.batch_size=64 \
  prep.output_dir=/path/to/out
```

Omit `prep.keyword_phonemes` (or leave it blank) to keep automatic G2P.

### Two-stage MUSAN FA (Zipformer locate + QbyT)

`scripts/eval_two_stage_musan_fa.py` and
`scripts/batch_eval_two_stage_musan_fa.sh` run the deployed two-stage
pipeline on whole MUSAN files. Stage I locates on the file, Stage II
scores each remaining span, and FA/hour is **wake-ups / audio hours**:
every span that meets `demo.qbyt_threshold` counts, including multiple
hits in one file. Files with no scored Stage I spans still contribute
duration. This is **not** comparable to the official 3s Stage-II-only
grid above. `prep.window_sec` / `prep.hop_sec` are unused.

```bash
python3 scripts/eval_two_stage_musan_fa.py \
  +experiment=icefall_zipformer_stage2 \
  +locator=sherpa_zipformer_kws \
  locator.tokens=/path/tokens.txt \
  locator.encoder=/path/encoder.onnx \
  locator.decoder=/path/decoder.onnx \
  locator.joiner=/path/joiner.onnx \
  locator.keywords_file=/path/keywords.txt \
  prep.keyword="hey eva" \
  'prep.keyword_phonemes=HH EY1 IY1 V AH0' \
  prep.musan_root=/path/to/musan \
  prep.stage2_ckpt=/path/to/stage2.pt \
  prep.output_dir=/path/to/out
```

Use the same `+experiment` as Stage II training so encoder, fbank, stream,
and adapter match the checkpoint. `+locator=phoneme_ctc` still needs
`prep.stage1_ckpt`; that locator searches the same ARPAbet sequence as QbyT,
including `prep.keyword_phonemes`. sherpa-onnx does **not** accept ARPAbet:
Stage I reads only `locator.keywords_file` (official sherpa-onnx
`keywords.txt`, one phrase per line). The caller must put this evaluation's
keyword in that file. `prep.keyword` still labels the run; `prep.keyword_phonemes`
only changes Stage II scoring. icefall / WeKws still locate from keyword text.

Batch and multi-GPU wrappers:

```bash
bash scripts/batch_eval_two_stage_musan_fa.sh \
  --keyword "hey eva" \
  --keyword-phonemes "HH EY1 IY1 V AH0" \
  --musan-root /path/to/musan \
  --pt /path/to/stage2.pt:/path/to/out \
  +locator=sherpa_zipformer_kws \
  locator.tokens=/path/tokens.txt \
  locator.encoder=/path/encoder.onnx \
  locator.decoder=/path/decoder.onnx \
  locator.joiner=/path/joiner.onnx \
  locator.keywords_file=/path/keywords.txt

bash scripts/eval_two_stage_musan_fa_shards.sh \
  +experiment=icefall_zipformer_stage2 \
  +locator=sherpa_zipformer_kws \
  locator.keywords_file=/path/keywords.txt \
  'prep.keyword=hey eva' \
  prep.musan_root=/path/to/musan \
  prep.stage2_ckpt=/path/to/stage2.pt \
  prep.output_dir=/path/to/out
```

### Throughput knobs

These do not change the hop grid. Defaults keep same-grid scores identical to
the historical per-window fp32 path:

| Override | Default | Effect |
| --- | --- | --- |
| `prep.batch_size` | `64` | GPU windows per forward. Caps memory on long speech files. |
| `prep.num_workers` | `0` | CPU prefetch of the next files' fbank. `0` is serial. |
| `prep.amp` | `off` | `fp16` is the V100 option and **changes logits**. |
| `prep.fbank_windows` | `independent` | `file` extracts fbank once per file and slices frames. Faster, but **not** bit-identical when fbank uses `snip_edges=false` (Icefall). |

Faster and not bit-identical: add `prep.amp=fp16 prep.fbank_windows=file`. Copy
`musan_root` to local disk when two processes would otherwise share a network
filesystem.

### Two GPUs

Do not point each process at `music/`, `noise/`, or `speech/` as
`prep.musan_root`. That breaks subset names. Shard files by duration, then merge:

```bash
bash scripts/eval_musan_fa_shards.sh \
  --num-shards 2 \
  --gpus 0,1 \
  +experiment=icefall_zipformer_stage2 \
  prep.keyword="hey eva" \
  'prep.keyword_phonemes=HH EY1 IY1 V AH0' \
  prep.musan_root=/path/to/musan \
  prep.stage2_ckpt=/path/to/stage2_step020000.pt \
  prep.output_dir=/path/to/out \
  prep.batch_size=64 \
  prep.num_workers=8
```

That writes `shard_0/`, `shard_1/`, and a pooled `merged/` whose FA/hour is
total FP / total hours (never the average of shard FA/hour). Merge later with
`python3 scripts/merge_musan_fa.py /path/to/out /path/to/out/merged`.

### Plot an existing eval directory

`scripts/plot_musan_fa_curve.py` does not run inference. It reads
`results.jsonl` and, when present, `summary.json` for `total_hours` and the
deployment threshold, then rewrites the PNG/CSV pair:

```bash
python3 scripts/plot_musan_fa_curve.py /path/to/musan_test
```

Override with `--total-hours`, `--threshold`, `--summary`, or `--output-dir`.
`scripts/scan_stage2_thresholds.py` is the related table scan over the same
`qbyt_score` values. Point it at the eval directory. Mixed labels keep the
usual metric table; a positive-only file also writes a threshold-versus-recall
plot, and a negative-only file writes threshold versus FPR.

### Many checkpoints or keywords

```bash
bash scripts/batch_eval_musan_fa.sh \
  --keyword "hey eva" \
  --keyword-phonemes "HH EY1 IY1 V AH0" \
  --keyword "hey android" \
  --musan-root /path/to/musan \
  --pt /path/to/stage2_step020000.pt \
  --base-out /path/to/musan_fa_outputs \
  --window-sec 3.0 \
  --hop-sec 3.0
```

`--keyword-phonemes` applies to the immediately preceding `--keyword`. Keywords
without that option use automatic G2P. For many keywords, use a one- or two-column
TAB-separated file (`keywords.txt`):

```text
# keyword<TAB>optional ARPAbet phonemes; blank lines and # comments are ignored.
hey eva	HH EY1 IY1 V AH0
hey android
hi galaxy	HH AY1 G AE1 L AH0 K S IY0
```

then run:

```bash
bash scripts/batch_eval_musan_fa.sh \
  --keywords-file keywords.txt \
  --musan-root /path/to/musan \
  --pts-file checkpoints.txt \
  --base-out /path/to/musan_fa_outputs
```

Each checkpoint × keyword combination writes its own `results.jsonl`,
`summary.json`, and (by default) the PNG/CSV curve pair. The batch wrapper does
not build a cross-run table. For that, use:

```bash
python3 scripts/aggregate_musan_fa.py \
  /path/to/musan_fa_outputs \
  /path/to/musan_fa_outputs/musan_fa_summary.tsv
```

### Remove wake-word audio from MUSAN with WeNet ASR

Use `scripts/filter_musan_by_wenet_asr.py` to build a clean MUSAN copy before
false-accept evaluation. It decodes 30-second windows with one second of overlap,
uses normalized RapidFuzz `partial_ratio` matching, and excludes the complete
source file when any window matches any configured wake word. The default WeNet
mode is `attention_rescoring`, and the default match threshold is 85.

A model directory can contain:

```text
wenet-model/
├── final.pt                  # any single *.pt name is accepted
├── train.yaml
├── global_cmvn               # global_cvmn is also accepted
├── unigram5000.model
└── units.txt
```

The WeNet Python source package must be installed or supplied through
`--wenet-root`/`WENET_ROOT`. The root must contain `wenet/bin/recognize.py`.
Both current (`--modes`/`--result_dir`) and legacy
(`--mode`/`--result_file`) recognition CLIs are detected automatically.

```bash
python3 scripts/filter_musan_by_wenet_asr.py \
  --musan-root /path/to/musan \
  --output-root /path/to/musan-filtered \
  --model-dir /path/to/wenet-model \
  --wenet-root /path/to/wenet \
  --keyword "hey eva" \
  --keyword "hey android" \
  --device cuda
```

For a longer list, pass `--keywords-file keywords.txt`. Individual
`--checkpoint`, `--config`, `--cmvn`, `--bpe-model`, and `--units` arguments
override model-directory discovery. `--threshold`, `--window-sec`,
`--overlap-sec`, `--batch-size`, `--beam-size`, and `--mode` are also
configurable. The output path must not already exist.

By default, audit files are written beside the filtered dataset under
`<output-name>_asr_filter_report/`:

- `results.jsonl` records each source audio file, every ASR window transcript,
  the highest-scoring wake word, score, and keep/remove decision.
- `summary.json` records overall, per-subset, and per-keyword counts together
  with the resolved model paths and decode settings.

Window audio is materialized temporarily as FLAC and removed after decoding, so
ensure the system temporary directory has enough free space. For short wake
words or noisy transcripts, inspect `results.jsonl` and adjust `--threshold` to
control the trade-off between missed matches and over-filtering.

## External locator (Zipformer / WeKws+Wenet)

Stage II verification stays on the existing Conformer+QbyT checkpoint (`prep.stage2_ckpt`). You can swap the Stage I candidate proposer with an external locator that only returns time spans; Stage II re-extracts fbank from cropped wav with its own encoder.

Hydra presets live under `configs/locator/`. Select with `+locator=<name>` (default group: `phoneme_ctc` in `configs/config.yaml`).

| Locator | Hydra preset | Extra install | Required overrides |
|---------|--------------|---------------|-------------------|
| **phoneme_ctc** (default) | `+locator=phoneme_ctc` | core only | `prep.stage1_ckpt` |
| **sherpa_zipformer_kws** | `+locator=sherpa_zipformer_kws` | `pip install -e ".[locator]"` | `locator.tokens`, `encoder`, `decoder`, `joiner`, `keywords_file` |
| **wekws_wenet** | `+locator=wekws_wenet` | `pip install -e ".[wekws]"` + local wekws checkout | `locator.wekws.config`, `checkpoint`, `symbol_table` |
| **icefall_pt_kws** (optional) | `+locator=icefall_pt_kws` | core only | `locator.root`, `decode_script`, `checkpoint` |

**WeKws+Wenet setup:** clone your wekws fork and point the runtime at it:

```bash
export WEKWS_ROOT=/path/to/wekws   # default in code: ~/Documents/myfork/wekws
pip install -e ".[wekws]"
```

Single-file demo (WeKws+Wenet locator + Stage II):

```bash
python3 scripts/run_two_stage_demo.py \
  +experiment=wenet_asr_stage2 \
  +locator=wekws_wenet \
  prep.stage2_ckpt=data/dma-kws/exp/stage2_qbyt/checkpoints/stage2_step050000.pt \
  prep.audio=/path/test.wav \
  prep.keyword="hey eva" \
  locator.wekws.config=/path/train.yaml \
  locator.wekws.checkpoint=/path/step247499.pt \
  locator.wekws.symbol_table=/path/units.txt
```

With the default `phoneme_ctc` locator, also pass `prep.stage1_ckpt=...`.

Batch evaluation reads a manifest with columns `audio_path`, `keyword`, and optional `label`. Both CSV and JSONL are supported; relative `audio_path` values are resolved against the manifest's own directory.

Step 1 (optional): generate a manifest from a folder of audio.

- Use **single** mode when all files share one keyword.
- Use **auto_assign** mode when different files map to different keywords by filename.

```bash
# single mode
python3 scripts/prepare_two_stage_manifest.py \
  prep.input_dir=/path/to/audio_folder \
  prep.keyword="hey eva" \
  prep.output=/path/manifest.csv \
  prep.label=1

# auto_assign mode (per-keyword labels)
python3 scripts/prepare_two_stage_manifest.py \
  prep.input_dir=/path/to/audio_folder \
  prep.keyword_mode=auto_assign \
  prep.keywords='["hey eva","ok lamp"]' \
  prep.keyword_labels='{"hey eva":1,"ok lamp":0}' \
  prep.skip_unmatched=true \
  prep.output=/path/manifest.csv
```

Extra options: `prep.recursive=false` limits scanning to the top-level directory, `prep.manifest_format=jsonl` (or a `.jsonl` output suffix) writes JSONL, and `prep.limit=N` caps the number of files for a quick smoke run. The scanner picks up `.wav`, `.flac`, `.mp3`, and `.m4a` files.

Step 2: run the two-stage pipeline over the manifest.

```bash
python3 scripts/eval_two_stage_kws.py \
  +experiment=wenet_asr_stage2 \
  +locator=wekws_wenet \
  prep.manifest=/path/manifest.csv \
  prep.output_dir=outputs/two_stage_eval \
  prep.stage2_ckpt=data/dma-kws/exp/stage2_qbyt/checkpoints/stage2_step050000.pt \
  locator.wekws.config=/path/train.yaml \
  locator.wekws.checkpoint=/path/step247499.pt \
  locator.wekws.symbol_table=/path/units.txt
```

Default output directory: `outputs/eval_two_stage_kws` (`prep.output_dir`). Writes `results.jsonl` (per-utterance detections and scores) and `summary.json` (accuracy, precision, recall, f1, auc, eer when labels are present). If a manifest row contains extra columns beyond `audio_path`, `keyword`, and optional `label`, those key-value pairs are copied into `results.jsonl` under `manifest_meta`.

---

## Troubleshooting

### `ModuleNotFoundError: dma_kws`

Install the package in editable mode from the repo root:

```bash
cd /path/to/DMA-KWS
pip install -e .
```

Or run scripts with `PYTHONPATH=.`:

```bash
cd /path/to/DMA-KWS
PYTHONPATH=. python3 scripts/prepare_stage1_librispeech.py +experiment=demo_librispeech100 --help
```

### `Missing torch/torchaudio`

Install CUDA PyTorch and torchaudio on the remote GPU machine before training:

```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Use the wheel index that matches your CUDA driver.

### `ModuleNotFoundError: No module named 'whisper'`

The vendored QbyT/Wenet encoder utilities import `whisper.tokenizer`, which is provided by the `openai-whisper`
package. Install core dependencies after pulling the latest repo:

```bash
pip install -r requirements.txt
pip install -e .
```

If you installed a different package named `whisper`, remove it and install OpenAI Whisper:

```bash
pip uninstall -y whisper
pip install -U openai-whisper
```

Verify:

```bash
python3 - <<'PY'
from whisper.tokenizer import LANGUAGES

print("whisper tokenizer OK, languages:", len(LANGUAGES))
PY
```

### `Couldn't find appropriate backend to handle uri ... .flac`

This means `torchaudio.load()` cannot decode the LibriSpeech FLAC file in the current environment, or the manifest
points at a missing/corrupt file. First verify the path from the traceback:

```bash
WAV=/path/to/LibriSpeech/train-clean-100/5652/39938/5652-39938-0026.flac

ls -lh "$WAV"
file "$WAV"
```

If the file is missing, check `paths.librispeech_root` / `paths.processed_root` in your config and regenerate the
Stage I manifest. If the file exists, inspect the available torchaudio backends:

```bash
python3 - <<'PY'
import torch
import torchaudio

p = "/path/to/LibriSpeech/train-clean-100/5652/39938/5652-39938-0026.flac"

print("torch:", torch.__version__)
print("torchaudio:", torchaudio.__version__)
print("audio backends:", torchaudio.list_audio_backends())
print(torchaudio.info(p))
waveform, sr = torchaudio.load(p)
print(waveform.shape, sr)
PY
```

Install FLAC-capable audio libraries and retry:

```bash
conda install -c conda-forge libsndfile ffmpeg -y
pip install -U soundfile
```

Then rerun the small training smoke command.

### `Missing dependency g2p_en`

Install core dependencies:

```bash
pip install -r requirements.txt
pip install -e .
```

### `ImportError: sherpa-onnx` or `ImportError: librosa`

Install the optional locator extras:

```bash
pip install -e ".[locator]"   # sherpa-onnx for Zipformer KWS
pip install -e ".[wekws]"     # librosa for WeKws+Wenet locator
```

For WeKws+Wenet, also set `WEKWS_ROOT` to your local wekws checkout.

### `Resource 'averaged_perceptron_tagger_eng' not found`

This comes from `g2p_en` calling NLTK's POS tagger while converting text to phonemes. Download the required NLTK
runtime data in the active environment:

```bash
python3 - <<'PY'
import nltk

for pkg in [
    "averaged_perceptron_tagger_eng",
    "averaged_perceptron_tagger",
    "cmudict",
    "punkt",
    "punkt_tab",
]:
    nltk.download(pkg)
PY
```

To install into a specific shared directory:

```bash
mkdir -p /root/nltk_data

python3 -m nltk.downloader \
  -d /root/nltk_data \
  averaged_perceptron_tagger_eng averaged_perceptron_tagger cmudict punkt punkt_tab
```

Verify:

```bash
python3 - <<'PY'
from nltk.data import find

for path in [
    "taggers/averaged_perceptron_tagger_eng/",
    "corpora/cmudict",
]:
    print(path, "=>", find(path))
PY
```

For offline servers, copy a prepared `nltk_data` directory to one of NLTK's search paths, or set `NLTK_DATA` to
that directory before running the preparation scripts.

### Stage II parquet columns do not match

Check columns on your source aggregated parquet:

```bash
python3 - <<'PY'
import pandas as pd
path = 'your_file.parquet'
df = pd.read_parquet(path)
print(df.columns)
PY
```

The preparation script expects `ngram`, `clips`, and optional `ngram_g2p`. If your HuggingFace dump has different column names, add a small converter or adapt `scripts/prepare_stage2_paper.py`.

### Training is too slow or OOM

Start with smoke overrides on the relevant script (always include `+experiment=...`):

```bash
# Stage I prep
python3 scripts/prepare_stage1_librispeech.py +experiment=demo_librispeech100 prep.limit=100

# Stage II prep
python3 scripts/prepare_stage2_paper.py +experiment=demo_librispeech100 prep.limit_anchors=50

# Stage I train
python3 scripts/train_stage1_ctc.py +experiment=demo_librispeech100 run.limit_steps=20 run.devices=1

# Stage II train
python3 scripts/train_stage2_qbyt.py +experiment=demo_librispeech100 run.limit_steps=20 stage2.batch_size_per_gpu=64
```

Then increase data and step limits gradually. Demo defaults are `stage1.batch_size_per_gpu=48` and `stage2.batch_size_per_gpu=128`.

### `LibriPhrase eval data is required`

Stage II validation reads LibriPhrase eval wav/CSV files under `stage2.eval.test_dir`. The raw LibriPhrase-100 training download does not include eval assets. Obtain them from LibriPhrase-460 Hugging Face eval files or symlink a shared eval tree; set `stage2.eval.test_dir` in your config. See [docs/paper-reproduction.md](docs/paper-reproduction.md) for LP-460 hard eval used in paper metrics. The demo preset validates on LP-100 eval; paper metrics require LP-460 hard eval.

### `FileNotFoundError` for `.npy` during validation

Validation reads from `stage2.eval.fbank_dir` when configured; otherwise it expects each `.npy` next to its eval WAV. Generate features with the same experiment used for training:

```bash
python3 scripts/prepare_stage2_eval_fbank.py +experiment=wenet_asr_stage2 prep.from_csv=true
```

For Zipformer, replace the experiment with `icefall_zipformer_stage2`; its output goes to `features/fbank_icefall_kws_eval`. Use `prep.from_csv=true` to convert only WAV files referenced by the eval CSVs, or set `prep.test_dir=/path` if your eval root differs from the config.

---

## Datasets

- **LibriSpeech**: [https://www.openslr.org/12/](https://www.openslr.org/12/)
- **LibriPhrase-100**: `ZhiqiAi/LibriPhrase-100` on HuggingFace
- **LibriPhrase-460**: `ZhiqiAi/LibriPhrase-460` on HuggingFace
- **GigaPhrase-1000**: [https://github.com/aizhiqi-work/GigaPhrase-1000](https://github.com/aizhiqi-work/GigaPhrase-1000), HuggingFace ID `ZhiqiAi/GigaPhrase-1000`

Recommended first target: **LibriSpeech train-clean-100 + LibriPhrase-100**.

---

## Paper reproduction

For full-scale reproduction (LibriSpeech-460, LibriPhrase-460, GigaPhrase-1460 finetune, hard/easy eval), follow **[docs/paper-reproduction.md](docs/paper-reproduction.md)**.

Key points:

- **Same algorithm as demo**: architecture, utt+seq loss, hard negatives, CharTokenizer phoneme vocabulary, streaming Stage I search, checkpoint averaging, and LibriPhrase eval are all implemented in the shared codebase.
- **Demo preset**: `+experiment=demo_librispeech100` uses smaller data and fewer steps; it validates the pipeline on LP-100 eval but does not produce paper metrics.
- **Paper metrics** require the full recipe chain on LibriPhrase-460 hard eval: `init-ls-460` → avg → `ft-ls-gs-1460` via `train_stage2_recipe.py`, then `eval_stage2_libriphrase.py` with `prep.split=hard` (see [docs/paper-reproduction.md](docs/paper-reproduction.md)).
- **`frozen-wenet-encoder`**: optional paper recipe with `stage2.freeze_encoder=true` (`+experiment=frozen_wenet_encoder`); distinct from external Wenet ASR init (`wenet_asr_stage2`).
- Reported paper numbers: **97.85% AUC**, **6.13% EER** on LibriPhrase hard (target: within 1% absolute of main logs).
