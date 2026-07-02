# DMA-KWS

**DMA-KWS**: Effective User-defined Keyword Spotting with Dual-stage Matching, Multi-modal Enrollment, and Continual Adaptation.

This repository implements a two-stage keyword spotting pipeline with a **single algorithm and shared codebase**. Scale differences between the small demo and paper reproduction come from Hydra experiment presets and dataset size, not separate implementations.

1. **Stage I** (training + inference): phoneme CTC decoding proposes candidate keyword spans in audio.
2. **Stage II** (training + inference): QbyT phoneme matching verifies each candidate.

### Terminology

| Term | Meaning |
|------|---------|
| **experiment** | Hydra config overlay selected on the CLI, e.g. `+experiment=demo_librispeech100` |
| **recipe** | `training.recipe` label for multi-phase Stage II chains: `init-ls-460`, `ft-ls-gs-1460`, `frozen-wenet-encoder`, `wenet-asr-init` |
| **demo preset** | Scale-only experiment (`demo_librispeech100`) — same algorithm as the paper, smaller data and step counts |

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
| `scripts/train_stage2_qbyt.py` | Demo single-phase training and Wenet ASR encoder init (`wenet_asr_stage2`, §7b) |
| `scripts/train_stage2_recipe.py` | Paper multi-phase chain only: `init-ls-460`, `ft-ls-gs-1460`, `frozen-wenet-encoder` |

For the full paper chain (init → avg → finetune → eval), see [docs/paper-reproduction.md](docs/paper-reproduction.md).

### Checkpoints and resume

| Config key | Format | Purpose |
|------------|--------|---------|
| `run.resume_from` | Lightning `.ckpt` | Full training-state resume (weights, optimizer, scheduler, step) |
| `run.init_checkpoint` / `stage2.init_checkpoint` | Exported `.pt` | Weight-only seed for a **new** run (e.g. Stage I encoder → Stage II, or external Wenet ASR) |
| `stage2.resume_checkpoint` | `.ckpt` or exported weights | Weight seed for finetune from prior Stage II average (`avg_10.ckpt`) |

Exported weight files use names like `stage1_step000020.pt` and `stage2_step000020.pt`. Stage I training also writes `stage1_avg.pt` when `stage1.checkpoint_avg.enabled=true` (demo default).

`scripts/average_checkpoints.py` averages Lightning checkpoints — use `prep.pattern="*.ckpt"` for checkpoint directories. Demo Stage I auto-averages to `avg_10.ckpt` and exports `stage1_avg.pt` at end of training.

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
```

**Inference** (Stage II required; Stage I locator swappable)

```text
input audio + keyword text
  -> Stage I locator: propose keyword time spans (default: phoneme CTC + ContextGraph)
  -> Stage II QbyT verification: score each candidate on cropped audio
  -> detected / not detected
```

Default Stage I is the trained phoneme-CTC model. You can swap it for external locators (Zipformer sherpa-onnx, WeKws+Wenet ASR) that only provide `start_sec`/`end_sec`; Stage II still uses this repo's Conformer+QbyT encoder — see [External locator](#external-locator-zipformer--wekwswenet).

`+experiment=wenet_asr_stage2` seeds the Stage II **encoder** from an external Wenet ASR checkpoint; with the default `phoneme_ctc` locator, inference also needs a Stage I CTC checkpoint (`prep.stage1_ckpt`). That preset is distinct from `frozen-wenet-encoder` (`configs/experiment/frozen_wenet_encoder.yaml`), which freezes a **self-trained** Stage I encoder during paper-scale Stage II — see [docs/paper-reproduction.md](docs/paper-reproduction.md).

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
scripts/train_stage2_recipe.py         # paper multi-phase chain only
scripts/eval_stage2_libriphrase.py
scripts/eval_two_stage_kws.py
scripts/run_two_stage_demo.py
```

The repo vendors the Wenet toolkit under `wenet/` for encoder/tokenizer utilities. The legacy `qbyt/` reference scripts are not used by the Hydra training pipeline.

Paper-scale configs: `+experiment=paper_ls460`, `+experiment=paper_ls_gs1460`, `+experiment=frozen_wenet_encoder`. Wenet ASR encoder init preset: `+experiment=wenet_asr_stage2`. See [docs/paper-reproduction.md](docs/paper-reproduction.md) for the full recipe chain.

Configs use [Hydra](https://hydra.cc/): base groups live under `configs/` (paths, stage1, stage2, …) and experiments are overlays in `configs/experiment/`. Select one with `+experiment=<name>` and override any leaf with dotlist syntax, e.g. `run.devices=2 run.limit_steps=20 stage2.learning_rate=0.001`.

### Shared `fbank` config

Stage I/II prep scripts read a top-level `fbank:` section for shared Kaldi fbank settings (mel bins, frame length/shift, dither, window type). Example:

```yaml
fbank:
  num_mel_bins: 80
  frame_length: 25
  frame_shift: 10
  dither: 0.1
  window_type: povey
```

Eval-only overrides (e.g. `dither: 0.0` for deterministic validation features) can go under `stage2.eval.fbank:` without changing training prep.

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

Dependencies are declared in [`pyproject.toml`](pyproject.toml) (package metadata + optional extras) and mirrored in [`requirements.txt`](requirements.txt) for the README workflow. **`torch` and `torchaudio` are not pinned** in either file — install them separately first.

**Step 1 — PyTorch / torchaudio** (match your CUDA driver on GPU machines):

```bash
# Linux + CUDA 12.1 example
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121

# Mac or CPU-only
pip install torch torchaudio
```

**Step 2 — project core** (pick one approach):

```bash
# README workflow: requirements.txt + editable install
pip install -r requirements.txt
pip install -e .

# Or install core deps from pyproject.toml only
pip install -e .
```

**Step 3 — optional extras** (from `pyproject.toml`):

| Extra | Command | When you need it |
|-------|---------|------------------|
| `dev` | `pip install -e ".[dev]"` | Run `pytest` locally |
| `locator` | `pip install -e ".[locator]"` | `+locator=sherpa_zipformer_kws` (sherpa-onnx) |
| `wekws` | `pip install -e ".[wekws]"` | `+locator=wekws_wenet` (librosa for wav I/O) |

Install everything at once:

```bash
pip install -e ".[dev,locator,wekws]"
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

1. An **aggregated** parquet for phrase metadata — columns `ngram`, `clips`, and
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

Manifests include `phonemes_g2p` targets for Wenet `CharTokenizer` — a phoneme vocabulary over `data/dict/lang_char.txt` (ARPAbet-like symbols; stress markers such as `AH0` are normalized to `AH` everywhere).

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

- The script uses `g2p_en` to convert English transcripts to ARPAbet-like phonemes.
- Stress markers such as `AH0` are normalized to `AH` in manifests and downstream G2P.
- Stage I and Stage II share the Wenet CharTokenizer phoneme vocabulary at `data/dict/lang_char.txt`.

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

Stage II training uses the **paper pipeline** (`LibriPhraseTrainDataset`): a parquet with columns `ngram`, `ngram_g2p`, `clips_file`, `distances_file`, plus precomputed fbank `.npy` under `features/fbank/`. This is the same format as the paper configs — the demo differs only in dataset size and step counts.

Prepare that layout with `scripts/prepare_stage2_paper.py` (or the identical alias `scripts/prepare_stage2_libriphrase.py`). Training reads the paper parquet paths from `stage2.parquet_file` and `stage2.wav_dir` in your config (defaults under `data/dma-kws/processed/stage2_qbyt/` and `data/dma-kws/features/fbank/`).

**Hydra prep overrides:** All preparation scripts share the `prep:` group from `configs/prep/default.yaml`. Override any leaf on the command line with dotlist syntax, e.g. `prep.input_parquet=/path/to/file.parquet prep.limit_anchors=50`. Keys not set on the CLI use the yaml defaults (often empty / zero meaning “all” or “auto”).

Pass the **aggregated** LibriPhrase parquet explicitly with `prep.input_parquet`. If omitted, `prepare_stage2_paper.py` auto-finds `aggregated_segments_with_g2p*.parquet` under `paths.libriphrase100_root` only — for `+experiment=paper_ls460` you must pass `prep.input_parquet` explicitly. Use `aggregated_segments_with_g2p_distance.parquet` (includes per-clip G2P distances for hard negatives) — **not** `aggregated_segments_by_ngram.parquet`.

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
`workers` (-1 = auto), `strip_stress` (default true). When `prep.recompute_distances.input_parquet` is unset,
the script looks for `aggregated_segments_with_g2p.parquet` under
`paths.libriphrase460_root`, `paths.libriphrase100_root`, or `paths.libriphrase_root`.

If `ngram_g2p` is missing from the aggregated parquet, `prepare_stage2_paper.py` runs
on-the-fly G2P via `g2p_en`. When per-anchor `distances` is empty, it falls back to
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

- `prep.num_workers` — parallelizes fbank extraction **within** each decoded shard via `stream_fbank_from_decoded()`; shards are scanned **serially** (one shard in memory at a time). `0` = auto (`min(8, cpu_count())`); `1` = serial fbank jobs (useful for debugging).
- `prep.use_rich` — Rich tables and multi-task progress bars when stdout is a TTY; falls back to plain `print` / `tqdm` when redirected or non-interactive (even if `prep.use_rich=true`).

```bash
# Full prep with explicit parallelism
python3 scripts/prepare_stage2_paper.py \
  +experiment=wenet_asr_stage2 \
  prep.input_parquet=data/dma-kws/raw/LibriPhrase-100/aggregated_segments_with_g2p_distance.parquet \
  prep.num_workers=4
```

The script prints staged progress (plan → load parquet → scan decoded shards → build anchors / fbank → summary) and a final stats table. Key fields: `anchors`, `clips_total`, `fbank_written`, `fbank_skipped`, `missing_in_decoded` (referenced clip keys not found in decoded parquet shards during streaming fbank).

### Operational notes

- Fbank uses `stream_fbank_from_decoded()` — decoded shards are processed one at a time, so peak memory is bounded by a single shard, not the full anchor set.
- Fbank extraction skips existing `.npy` files by default — safe for incremental reruns after fixing a subset of clips.
- Use `prep.num_workers=1` when debugging fbank parallelism or reproducing ordering issues within a shard.

Expected outputs:

```text
data/dma-kws/processed/stage2_qbyt/aggregated_segments_with_g2p_distance.parquet
data/dma-kws/processed/stage2_qbyt/clips/
data/dma-kws/processed/stage2_qbyt/distances/
data/dma-kws/features/fbank/LP-100-fbank/
```

Fbank parameters (`num_mel_bins`, `frame_length`, `frame_shift`, etc.) are read from the top-level `fbank:` block in your config YAML.

Training consumes the paper parquet + fbank layout directly via `LibriPhraseTrainDataset` (random + hard negatives, utt + seq loss).

### LibriPhrase eval data (required for validation)

Stage II training runs LibriPhrase validation on a schedule (`stage2.validation.val_check_interval`). You need the official eval set under `stage2.eval.test_dir` (default `data/dma-kws/raw/LibriPhrase-100/eval`). It is **not** included in the LibriPhrase-100 training download — obtain it from the LibriPhrase-460 Hugging Face eval assets or symlink a shared eval tree. See [docs/paper-reproduction.md](docs/paper-reproduction.md) for LP-460 hard eval used in paper metrics.

Expected layout:

```text
<test_dir>/
  evaluation_set/libriphrase_diffspk_all_1word.csv
  evaluation_set/libriphrase_diffspk_all_2word.csv
  evaluation_set/libriphrase_diffspk_all_3word.csv
  evaluation_set/libriphrase_diffspk_all_4word.csv
  train-other-500/train-other-500/<spk>/<chap>/*.wav
```

`stage2.eval.split` defaults to `hard`; validation CSVs must include the columns expected for that split. Training reads `stage2.eval.test_dir` — `prep.test_dir` applies only to `prepare_stage2_eval_fbank.py`.

Validation reads fbank `.npy` files as **siblings** of each `.wav` (same directory), not under `features/fbank/`. Precompute them with `scripts/prepare_stage2_eval_fbank.py` (uses the same `fbank:` settings as training prep; eval dither defaults come from `stage2.eval.fbank` when set). Recommend `prep.from_csv=true` (default `false` walks every wav under `test_dir`):

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

`prep.from_csv=true` converts only wav files referenced by the eval CSV `anchor` / `comparison` columns (faster than scanning all wav under `test_dir`). Each `train-other-500/.../clip.wav` gets a sibling `clip.npy` in the same folder.

Useful overrides (also in `configs/prep/default.yaml`):

- `prep.test_dir=/path` — eval root for the prep script when it differs from `stage2.eval.test_dir` in config.
- `prep.limit=100` — smoke test (first N wav paths only).
- `prep.log_interval=1000` — progress print frequency.
- `prep.no_skip_existing=true` — recompute fbank even when `.npy` already exists (default skips existing files for incremental reruns).

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

For the paper multi-phase chain (`init-ls-460` → avg → `ft-ls-gs-1460`), use `scripts/train_stage2_recipe.py` — see [docs/paper-reproduction.md](docs/paper-reproduction.md).

---

## 7b. Stage II: initialize from external Wenet ASR encoder

Alternative to Stage I init: seed the Stage II Conformer **encoder** from a pretrained Wenet ASR checkpoint. Use `+experiment=wenet_asr_stage2` with `scripts/train_stage2_qbyt.py` (single phase). Two-stage **inference** still requires a separately trained Stage I CTC model for candidate proposal.

This is distinct from `+experiment=frozen_wenet_encoder` (`configs/experiment/frozen_wenet_encoder.yaml`), which freezes a self-trained Stage I encoder during the paper `frozen-wenet-encoder` recipe — see [docs/paper-reproduction.md](docs/paper-reproduction.md).

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

## 8. Run the two-stage demo

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
```

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

## External locator (Zipformer / WeKws+Wenet)

Stage II verification stays on the existing Conformer+QbyT checkpoint (`prep.stage2_ckpt`). You can swap the Stage I candidate proposer with an external locator that only returns time spans; Stage II re-extracts fbank from cropped wav with its own encoder.

Hydra presets live under `configs/locator/`. Select with `+locator=<name>` (default group: `phoneme_ctc` in `configs/config.yaml`).

| Locator | Hydra preset | Extra install | Required overrides |
|---------|--------------|---------------|-------------------|
| **phoneme_ctc** (default) | `+locator=phoneme_ctc` | core only | `prep.stage1_ckpt` |
| **sherpa_zipformer_kws** | `+locator=sherpa_zipformer_kws` | `pip install -e ".[locator]"` | `locator.tokens`, `encoder`, `decoder`, `joiner` |
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

Step 1 (optional): generate a manifest from a folder of audio that all share one keyword. Pass `prep.label=1` to mark them all as positives (enables precision/recall/f1 in the summary), or omit it to leave the manifest unlabeled.

```bash
python3 scripts/prepare_two_stage_manifest.py \
  prep.input_dir=/path/to/audio_folder \
  prep.keyword="hey eva" \
  prep.output=/path/manifest.csv \
  prep.label=1
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

Default output directory: `outputs/eval_two_stage_kws` (`prep.output_dir`). Writes `results.jsonl` (per-utterance detections and scores) and `summary.json` (accuracy, precision, recall, f1, auc, eer when labels are present).

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

Validation expects precomputed fbank `.npy` next to eval wav files. Run:

```bash
python3 scripts/prepare_stage2_eval_fbank.py +experiment=wenet_asr_stage2 prep.from_csv=true
```

Use `prep.from_csv=true` to convert only wav files referenced by the eval CSVs, or set `prep.test_dir=/path` if your eval root differs from the config.

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

- **Same algorithm as demo** — architecture, utt+seq loss, hard negatives, CharTokenizer phoneme vocabulary, streaming Stage I search, checkpoint averaging, and LibriPhrase eval are all implemented in the shared codebase.
- **Demo preset** — `+experiment=demo_librispeech100` uses smaller data and fewer steps; it validates the pipeline on LP-100 eval but does not produce paper metrics.
- **Paper metrics** require the full recipe chain on LibriPhrase-460 hard eval: `init-ls-460` → avg → `ft-ls-gs-1460` via `train_stage2_recipe.py`, then `eval_stage2_libriphrase.py` with `prep.split=hard` (see [docs/paper-reproduction.md](docs/paper-reproduction.md)).
- **`frozen-wenet-encoder`** — optional paper recipe with `stage2.freeze_encoder=true` (`+experiment=frozen_wenet_encoder`); distinct from external Wenet ASR init (`wenet_asr_stage2`).
- Reported paper numbers: **97.85% AUC**, **6.13% EER** on LibriPhrase hard (target: within 1% absolute of main logs).
