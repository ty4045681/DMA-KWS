# DMA-KWS

**DMA-KWS**: Effective User-defined Keyword Spotting with Dual-stage Matching, Multi-modal Enrollment, and Continual Adaptation.

This repository contains code for a two-stage keyword spotting pipeline:

1. **Stage I**: phoneme CTC decoding to find candidate keyword regions.
2. **Stage II**: QbyT phoneme matching to verify each candidate.

> The repository uses a **single training recipe** aligned with the paper/main codebase. `configs/demo_librispeech100.yaml` is a **scale-only smoke preset** (LibriSpeech-100 / LibriPhrase-100, fewer steps) — same architecture, losses, negative mining, tokenizer, and streaming Stage I search as the paper configs. It does **not** define a separate demo algorithm.

---

## What is implemented now

The runnable path (demo and paper share the same code):

```text
LibriSpeech
  -> Stage I CharTokenizer manifest (data/dict/lang_char.txt)
  -> Stage I Conformer + CTC training

LibriPhrase (paper-format parquet + precomputed fbank .npy)
  -> Stage II utt_loss + seq_loss, random + hard negatives
  -> Stage II Conformer + QbyT training (init -> avg -> finetune)

input audio + keyword text
  -> Stage I prefix beam + ContextGraph candidates
  -> Stage II QbyT verification
  -> detected / not detected
```

Main entry points:

```text
scripts/prepare_stage1_librispeech.py
scripts/prepare_stage1_fbank.py
scripts/train_stage1_ctc.py
scripts/average_checkpoints.py
scripts/prepare_stage2_paper.py
scripts/train_stage2_qbyt.py
scripts/train_stage2_recipe.py
scripts/eval_stage2_libriphrase.py
scripts/run_two_stage_demo.py
```

Paper-scale configs: `configs/paper_ls460.yaml`, `configs/paper_ls_gs1460.yaml`. See [docs/paper-reproduction.md](docs/paper-reproduction.md) for the full recipe chain.

---

## Recommended hardware

The small-scale real demo is intended for a Linux training machine with:

- 2 × NVIDIA V100 GPUs, or similar;
- CUDA-enabled PyTorch + torchaudio;
- enough disk for LibriSpeech and LibriPhrase features/checkpoints.

CPU/Mac local runs are useful for code checks, but not for realistic training.

---

## 1. Clone and create environment

```bash
git clone https://github.com/ty4045681/DMA-KWS.git
cd DMA-KWS

conda create -n dma-kws python=3.10 -y
conda activate dma-kws
```

Install CUDA PyTorch/torchaudio according to your server CUDA driver. Example for CUDA 12.1 wheels:

```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Then install the project Python dependencies:

```bash
pip install -r requirements.txt
```

The requirements include `soundfile` for FLAC decoding support and `openai-whisper` for the vendored QbyT/Wenet
encoder utilities. On some Linux/conda systems, `torchaudio` may still need system audio libraries for `.flac`
files; install them before training if `torchaudio.load()` cannot read LibriSpeech audio:

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

Default config:

```text
configs/demo_librispeech100.yaml
```

By default it expects this layout:

```text
/data/dma-kws/
├── raw/
│   ├── LibriSpeech/
│   │   ├── train-clean-100/
│   │   └── dev-clean/
│   └── LibriPhrase-100/
├── processed/
├── features/
└── exp/
```

If your data root is not `/data/dma-kws`, edit `configs/demo_librispeech100.yaml` and update the `paths:` section before running scripts.

Create directories:

```bash
mkdir -p /data/dma-kws/raw /data/dma-kws/processed /data/dma-kws/features /data/dma-kws/exp
```

---

## 3. Download datasets

### 3.1 LibriSpeech for Stage I

Download at least:

- `train-clean-100`
- `dev-clean`

Example:

```bash
cd /data/dma-kws/raw
mkdir -p LibriSpeech
cd LibriSpeech

wget https://www.openslr.org/resources/12/train-clean-100.tar.gz
wget https://www.openslr.org/resources/12/dev-clean.tar.gz

tar -xzf train-clean-100.tar.gz --strip-components=1
tar -xzf dev-clean.tar.gz --strip-components=1
```

After extraction, verify:

```bash
ls /data/dma-kws/raw/LibriSpeech/train-clean-100
ls /data/dma-kws/raw/LibriSpeech/dev-clean
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
  --local-dir /data/dma-kws/raw/LibriPhrase-100
```

Fallback if the `hf` command is still unavailable after upgrading:

```bash
python3 - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="ZhiqiAi/LibriPhrase-100",
    repo_type="dataset",
    local_dir="/data/dma-kws/raw/LibriPhrase-100",
)
PY
```

The Stage II preparation script reads two kinds of files from the download:

1. An **aggregated** parquet for phrase metadata — at least columns:

   ```text
   ngram
   clips
   ```

   and optionally `ngram_g2p` (skips on-the-fly G2P if present). Recommended:
   `aggregated_segments_with_g2p.parquet`.

2. The **decoded** audio shards `LP-100-decoded-*.parquet` (columns
   `audio_rel`, `audio`, `sampling_rate`, ...). The script extracts referenced
   clips, writes per-phrase `clips` / `distances` `.npy` shards, and computes
   80-dim fbank `.npy` files under `<feature_root>/fbank/`.

Inspect columns if needed:

```bash
python3 - <<'PY'
import pandas as pd
path = '/data/dma-kws/raw/LibriPhrase-100/path/to/file.parquet'
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
  --config configs/demo_librispeech100.yaml \
  --limit 100
```

Expected outputs:

```text
/data/dma-kws/processed/stage1_phoneme_ctc/train.jsonl
/data/dma-kws/processed/stage1_phoneme_ctc/dev.jsonl
```

Manifests include `phonemes_g2p` targets for Wenet `CharTokenizer` (`data/dict/lang_char.txt`).

If the smoke run works, prepare the full LibriSpeech-100 split:

```bash
python3 scripts/prepare_stage1_librispeech.py \
  --config configs/demo_librispeech100.yaml
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
  --config configs/demo_librispeech100.yaml \
  --limit 100 \
  --input-format hf-parquet \
  --parquet-root /home/h00513998/librispeech_train_clean_360 \
  --parquet-split train-clean-360 \
  --dev-parquet-root /home/h00513998/librispeech_dev_clean \
  --dev-parquet-split dev-clean
```

Remove `--limit 100` for the full run after the smoke run completes:

```bash
python3 scripts/prepare_stage1_librispeech.py \
  --config configs/demo_librispeech100.yaml \
  --input-format hf-parquet \
  --parquet-root /home/h00513998/librispeech_train_clean_360 \
  --parquet-split train-clean-360 \
  --dev-parquet-root /home/h00513998/librispeech_dev_clean \
  --dev-parquet-split dev-clean
```

This mode extracts `audio.bytes` from the parquet records into:

```text
/data/dma-kws/processed/stage1_phoneme_ctc/audio/train-clean-360/
```

and writes `train.jsonl` / `dev.jsonl` with `wav_path` values pointing at the extracted audio files. If you omit
`--dev-parquet-root`, `paths.librispeech_root` must contain the configured dev split in the official LibriSpeech
directory layout; otherwise the script stops instead of silently writing an empty dev manifest.

Notes:

- The script uses `g2p_en` to convert English transcripts to ARPAbet-like phonemes.
- Stress markers such as `AH0` are normalized to `AH`.
- Stage I and Stage II share the Wenet CharTokenizer vocabulary at `data/dict/lang_char.txt`.

Optional: precompute Stage I fbank features for faster training (reads the JSONL manifests above):

```bash
python3 scripts/prepare_stage1_fbank.py \
  --config configs/demo_librispeech100.yaml
```

---

## 5. Stage I: train phoneme CTC model

Smoke training run:

```bash
CUDA_VISIBLE_DEVICES=0,1 python3 scripts/train_stage1_ctc.py \
  --config configs/demo_librispeech100.yaml \
  --devices 2 \
  --limit-steps 20
```

Full first run:

```bash
CUDA_VISIBLE_DEVICES=0,1 python3 scripts/train_stage1_ctc.py \
  --config configs/demo_librispeech100.yaml \
  --devices 2
```

Checkpoints are saved under:

```text
/data/dma-kws/exp/stage1_phoneme_ctc/checkpoints/
```

Example:

```bash
ls /data/dma-kws/exp/stage1_phoneme_ctc/checkpoints/*.pt
```

Optionally average the last few checkpoints before Stage II init or demo inference:

```bash
python3 scripts/average_checkpoints.py \
  --input-dir /data/dma-kws/exp/stage1_phoneme_ctc/checkpoints \
  --pattern "*.pt" \
  --last-k 10 \
  --output /data/dma-kws/exp/stage1_phoneme_ctc/checkpoints/avg_10.pt
```

If you hit out-of-memory, reduce these values in `configs/demo_librispeech100.yaml`:

```yaml
stage1:
  batch_size_per_gpu: 24
  num_workers: 2
```

---

## 6. Stage II: prepare training data

Stage II training uses the **paper pipeline** (`LibriPhraseTrainDataset`): a parquet with columns `ngram`, `ngram_g2p`, `clips_file`, `distances_file`, plus precomputed fbank `.npy` under `features/fbank/`. This is the same format as the paper configs — the demo differs only in dataset size and step counts.

Prepare that layout with `scripts/prepare_stage2_paper.py`. Training reads the paper parquet paths from `stage2.parquet_file` and `stage2.wav_dir` in your config (defaults under `/data/dma-kws/processed/stage2_qbyt/` and `/data/dma-kws/features/fbank/`).

Smoke example (adjust paths to your LibriPhrase-100 download):

```bash
python3 scripts/prepare_stage2_paper.py \
  --config configs/demo_librispeech100.yaml \
  --limit-anchors 50
```

Expected outputs:

```text
/data/dma-kws/processed/stage2_qbyt/aggregated_segments_with_g2p_distance.parquet
/data/dma-kws/processed/stage2_qbyt/clips/
/data/dma-kws/processed/stage2_qbyt/distances/
/data/dma-kws/features/fbank/LP-100-fbank/
```

Training consumes the paper parquet + fbank layout directly via `LibriPhraseTrainDataset` (random + hard negatives, utt + seq loss).

---

## 7. Stage II: train QbyT verifier

Optionally initialize from a Stage I checkpoint:

```bash
STAGE1_CKPT=/data/dma-kws/exp/stage1_phoneme_ctc/checkpoints/stage1_epoch000_step000020.pt
```

Smoke training run:

```bash
CUDA_VISIBLE_DEVICES=0,1 python3 scripts/train_stage2_qbyt.py \
  --config configs/demo_librispeech100.yaml \
  --devices 2 \
  --init-checkpoint "$STAGE1_CKPT" \
  --limit-steps 20
```

Full first run:

```bash
CUDA_VISIBLE_DEVICES=0,1 python3 scripts/train_stage2_qbyt.py \
  --config configs/demo_librispeech100.yaml \
  --devices 2 \
  --init-checkpoint "$STAGE1_CKPT"
```

Checkpoints are saved under:

```text
/data/dma-kws/exp/stage2_qbyt/checkpoints/
```

Example:

```bash
ls /data/dma-kws/exp/stage2_qbyt/checkpoints/*.pt
```

If you hit out-of-memory, reduce:

```yaml
stage2:
  batch_size_per_gpu: 64
  num_workers: 2
```

---

## 8. Run the two-stage demo

After both stages have checkpoints:

```bash
STAGE1_CKPT=/data/dma-kws/exp/stage1_phoneme_ctc/checkpoints/stage1_epoch000_step000020.pt
STAGE2_CKPT=/data/dma-kws/exp/stage2_qbyt/checkpoints/stage2_step000020.pt
AUDIO=/path/to/test.wav
KEYWORD="hello world"

python3 scripts/run_two_stage_demo.py \
  --config configs/demo_librispeech100.yaml \
  --stage1-ckpt "$STAGE1_CKPT" \
  --stage2-ckpt "$STAGE2_CKPT" \
  --audio "$AUDIO" \
  --keyword "$KEYWORD"
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

The decision threshold is configurable:

```yaml
demo:
  qbyt_threshold: 0.5
```

---

## Troubleshooting

### `ModuleNotFoundError: dma_kws`

Run scripts from the repository root:

```bash
cd /path/to/DMA-KWS
python3 scripts/prepare_stage1_librispeech.py --help
```

### `Missing torch/torchaudio`

Install CUDA PyTorch and torchaudio on the remote GPU machine before training:

```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Use the wheel index that matches your CUDA driver.

### `ModuleNotFoundError: No module named 'whisper'`

The vendored QbyT/Wenet encoder utilities import `whisper.tokenizer`, which is provided by the `openai-whisper`
package. Install the project requirements after pulling the latest repo:

```bash
pip install -r requirements.txt
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

Install requirements:

```bash
pip install -r requirements.txt
```

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

Start with smoke flags:

```bash
--limit 100
--limit-steps 20
--limit-anchors 100
```

Then increase data/steps gradually.

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

- **Same recipe as demo** — architecture, utt+seq loss, hard negatives, CharTokenizer, streaming Stage I search, checkpoint averaging, and LibriPhrase eval are all implemented.
- **Scale-only demo** — `demo_librispeech100.yaml` uses smaller data and fewer steps; it validates the pipeline but does not produce paper metrics.
- **Paper metrics** require the full recipe chain on LibriPhrase-460 hard eval: init → avg → finetune → `eval_stage2_libriphrase.py --split hard`.
- Reported paper numbers: **97.85% AUC**, **6.13% EER** on LibriPhrase hard (target: within 1% absolute of main logs).
