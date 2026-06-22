# DMA-KWS

**DMA-KWS**: Effective User-defined Keyword Spotting with Dual-stage Matching, Multi-modal Enrollment, and Continual Adaptation.

This repository contains code for a two-stage keyword spotting pipeline:

1. **Stage I**: phoneme CTC decoding to find candidate keyword regions.
2. **Stage II**: QbyT phoneme matching to verify each candidate.

> Current training scripts are optimized for a **small-scale real training demo** first: train both stages yourself on LibriSpeech-100 / LibriPhrase-100, get a real two-stage demo running, then scale to larger LibriPhrase/GigaPhrase settings.

---

## What is implemented now

The current runnable path is:

```text
LibriSpeech train-clean-100
  -> Stage I phoneme manifest/vocab
  -> Stage I Conformer + CTC training

LibriPhrase-100 style phrase data
  -> Stage II positive/negative phrase pairs
  -> Stage II Conformer + QbyT training

input audio + keyword text
  -> Stage I candidate regions
  -> Stage II QbyT scores
  -> detected / not detected
```

Main entry points:

```text
scripts/prepare_stage1_librispeech.py
scripts/train_stage1_ctc.py
scripts/prepare_stage2_libriphrase.py
scripts/train_stage2_qbyt.py
scripts/run_two_stage_demo.py
```

The first version uses **offline** Stage I decoding. Streaming search and paper-scale reproduction can be added after the small training pipeline is stable.

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
python3 -m pytest tests -q
python3 scripts/prepare_stage1_librispeech.py --help
python3 scripts/train_stage1_ctc.py --help
python3 scripts/prepare_stage2_libriphrase.py --help
python3 scripts/train_stage2_qbyt.py --help
python3 scripts/run_two_stage_demo.py --help
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

The Stage II preparation reads two kinds of files from the download:

1. An **aggregated** parquet for phrase metadata — at least columns:

   ```text
   ngram
   clips
   ```

   and optionally `ngram_g2p` (skips on-the-fly G2P if present). Recommended:
   `aggregated_segments_with_g2p.parquet`.

2. The **decoded** audio shards `LP-100-decoded-*.parquet` (columns
   `audio_rel`, `audio`, `sampling_rate`, ...). The script extracts only the
   clips actually referenced by the pairs and writes them as float32 `.npy`
   files under `<processed_root>/stage2_qbyt/audio/`. No loose `.wav` files
   are needed.

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
/data/dma-kws/processed/stage1_phoneme_ctc/phoneme_vocab.txt
```

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
- The phoneme vocabulary produced here is reused by Stage II.

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

If you hit out-of-memory, reduce these values in `configs/demo_librispeech100.yaml`:

```yaml
stage1:
  batch_size_per_gpu: 24
  num_workers: 2
```

---

## 6. Stage II: prepare QbyT pairs

Pick the aggregated LibriPhrase parquet with phrase metadata and G2P, and specify the decoded audio root:

```bash
LP_AGG=/data/dma-kws/raw/LibriPhrase-100/aggregated_segments_with_g2p.parquet
```

Smoke pair preparation:

```bash
python3 scripts/prepare_stage2_libriphrase.py \
  --config configs/demo_librispeech100.yaml \
  --input-parquet "$LP_AGG" \
  --decoded-parquet-root /data/dma-kws/raw/LibriPhrase-100 \
  --limit-anchors 50
```

The `--decoded-parquet-root` defaults to `paths.libriphrase100_root`, so it can be omitted if the decoded shards live there.

Full first run on the selected parquet:

```bash
python3 scripts/prepare_stage2_libriphrase.py \
  --config configs/demo_librispeech100.yaml \
  --input-parquet "$LP_AGG" \
  --decoded-parquet-root /data/dma-kws/raw/LibriPhrase-100
```

Expected output:

```text
/data/dma-kws/processed/stage2_qbyt/train.jsonl
```

The generated file contains pair records like:

```json
{
  "anchor_text": "hello world",
  "anchor_phonemes": ["HH", "AH", "L", "OW", "W", "ER", "L", "D"],
  "wav_path": "relative/audio/path.npy",
  "sample_rate": 16000,
  "label": 1
}
```

Note: `wav_path` points to `.npy` waveforms (float32) produced by the preparation script, relative to `<processed_root>/stage2_qbyt`. The trainer loads these `.npy` files and no longer reads `.wav` from `libriphrase100_root`.

---

## 7. Stage II: train QbyT verifier

Use the Stage I checkpoint to initialize the Stage II encoder if possible:

```bash
STAGE1_CKPT=/data/dma-kws/exp/stage1_phoneme_ctc/checkpoints/stage1_epoch000_step000020.pt
```

Smoke training run:

```bash
CUDA_VISIBLE_DEVICES=0,1 python3 scripts/train_stage2_qbyt.py \
  --config configs/demo_librispeech100.yaml \
  --devices 2 \
  --stage1-ckpt "$STAGE1_CKPT" \
  --limit-steps 20
```

Full first run:

```bash
CUDA_VISIBLE_DEVICES=0,1 python3 scripts/train_stage2_qbyt.py \
  --config configs/demo_librispeech100.yaml \
  --devices 2 \
  --stage1-ckpt "$STAGE1_CKPT"
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

Check columns:

```bash
python3 - <<'PY'
import pandas as pd
path = 'your_file.parquet'
df = pd.read_parquet(path)
print(df.columns)
PY
```

The current preparation script supports `ngram`, `clips`, and optional `ngram_g2p`. If your HuggingFace dump has different column names, add a small converter or adapt `scripts/prepare_stage2_libriphrase.py`.

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

## Notes on paper reproduction

The original README reports LibriPhrase hard-set performance of 97.85% AUC and 6.13% EER. The scripts in this branch are intended to get a real two-stage training/demo pipeline running first. To approach paper-scale results, next steps include:

1. scale Stage I beyond LibriSpeech-100;
2. scale Stage II to LibriPhrase-460 / GigaPhrase-1000;
3. improve hard-negative mining;
4. add checkpoint averaging;
5. add streaming Stage I candidate search;
6. add full LibriPhrase hard/easy evaluation scripts.
