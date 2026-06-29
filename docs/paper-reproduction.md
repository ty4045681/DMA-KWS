# Paper reproduction guide

This document describes how to reproduce the DMA-KWS paper pipeline on full-scale data. The repository uses a **single training recipe** shared with the small demo: architecture, losses, negative mining, tokenizer, and Stage I streaming search are identical. Only **scale** differs (dataset size, steps, batch size, GPU count).

**Important:** LibriSpeech-100 / LibriPhrase-100 smoke runs validate that the pipeline works end-to-end. They do **not** produce paper-level metrics. Claiming paper numbers requires the full recipe chain below on LibriPhrase-460 hard eval.

## Prerequisites

- Linux machine with 4× NVIDIA V100 (or similar) for paper-scale Stage II DDP training
- Python 3.10, CUDA PyTorch, torchaudio, PyTorch Lightning
- Dependencies from `requirements.txt` (includes `g2p_en`, `openai-whisper`, `torchmetrics`)
- NLTK data for G2P (see [README](../README.md#1-clone-and-create-environment))

Configs:

| Config | Purpose |
|--------|---------|
| `+experiment=paper_ls460` | Stage II init on LibriPhrase-460 (`init-ls-460`, 50k steps) |
| `+experiment=paper_ls_gs1460` | Stage II finetune on LS+GigaPhrase-1460 (`ft-ls-gs-1460`, 100k steps) |
| `+experiment=demo_librispeech100` | Same recipe, smaller data — smoke only |

Edit the `paths:` section in each YAML if your data root is not `/data/dma-kws`.

Expected directory layout:

```text
/data/dma-kws/
├── raw/
│   ├── LibriSpeech/          # train-clean-460, dev-clean
│   ├── LibriPhrase-460/      # metadata, eval CSV, decoded audio
│   └── LibriPhrase-GS-1460/  # finetune parquet + fbank shards
├── processed/
│   └── stage2_qbyt/
│       └── aggregated_segments_with_g2p_distance.parquet
├── features/
│   └── fbank/                # precomputed 80-dim fbank .npy (LP-460-fbank/, GP-1000-fbank/)
└── exp/
    ├── stage1_phoneme_ctc/
    └── stage2_qbyt/
```

Tokenizer (shared across stages):

```text
data/dict/lang_char.txt   # repo-canonical Wenet CharTokenizer phoneme vocabulary (73 tokens)
```

See [`data/dict/README.md`](../data/dict/README.md). The vendored dict is the single source of truth for both stages; the original author dictionary is unavailable. Do not replace it without retraining Stage I and Stage II. Runtime validation: `dma_kws.tokenizer.validate_lang_char_dict` (also called from `load_char_tokenizer`).

---

## 1. Download datasets

### 1.1 LibriSpeech (Stage I)

Download **train-clean-460** and **dev-clean** from [OpenSLR 12](https://www.openslr.org/12/):

```bash
mkdir -p /data/dma-kws/raw/LibriSpeech
cd /data/dma-kws/raw/LibriSpeech

wget https://www.openslr.org/resources/12/train-clean-460.tar.gz
wget https://www.openslr.org/resources/12/dev-clean.tar.gz
tar -xzf train-clean-460.tar.gz
tar -xzf dev-clean.tar.gz
```

### 1.2 LibriPhrase-460 (Stage II init + eval)

Download from HuggingFace:

```bash
hf download ZhiqiAi/LibriPhrase-460 \
  --repo-type dataset \
  --local-dir /data/dma-kws/raw/LibriPhrase-460
```

Required artifacts:

- `aggregated_segments_with_g2p_distance.parquet` — training metadata with `ngram`, `ngram_g2p`, `clips_file`, `distances_file` columns
- Per-phrase `clips_file` and `distances_file` `.npy` shards referenced by the parquet
- `eval/evaluation_set/*.csv` and `eval/evaluation_set/test_all_phrase.csv` for hard/easy splits
- Precomputed fbank features under `features/fbank/` (see §3)

Symlink or copy the aggregated parquet to the path expected by config:

```bash
mkdir -p /data/dma-kws/processed/stage2_qbyt
cp /data/dma-kws/raw/LibriPhrase-460/aggregated_segments_with_g2p_distance.parquet \
   /data/dma-kws/processed/stage2_qbyt/
```

### 1.3 GigaPhrase / LS-GS-1460 (Stage II finetune)

Download the LibriPhrase + GigaSpeech combined finetune set referenced in `+experiment=paper_ls_gs1460`:

```bash
hf download ZhiqiAi/GigaPhrase-1000 \
  --repo-type dataset \
  --local-dir /data/dma-kws/raw/GigaPhrase-1000
```

Place the finetune parquet at:

```text
/data/dma-kws/processed/stage2_qbyt/ls-gs-1460/processed_data.parquet
```

(Adjust `stage2.parquet_file` in `paper_ls_gs1460.yaml` if your layout differs.)

---

## 2. Stage I: phoneme CTC (self-trained)

Stage I uses Wenet-aligned **CharTokenizer** targets (`data/dict/lang_char.txt`), ConformerEncoder (80→144, 6 blocks), and CTC loss. **Train Stage I yourself** on LibriSpeech; do not use external author checkpoints (e.g. `/nvme01/.../avg_10.pt`). The averaged Stage I checkpoint from your run initializes Stage II when using `frozen-wenet-encoder` or `--init-checkpoint`.

### Prepare manifests + optional offline fbank

In-repo prep (recommended — no external Wenet submodule):

```bash
bash scripts/prepare_stage1_wenet.sh +experiment=paper_ls460
```

This runs:

1. `scripts/prepare_stage1_librispeech.py` — JSONL manifests with `phonemes_g2p`
2. `scripts/prepare_stage1_fbank.py` — optional precomputed fbank under `features/stage1_fbank/`

Outputs:

```text
/data/dma-kws/processed/stage1_phoneme_ctc/train.jsonl
/data/dma-kws/processed/stage1_phoneme_ctc/dev.jsonl
/data/dma-kws/features/stage1_fbank/          # mirror LibriSpeech tree as .npy
```

Manifest-only prep (online fbank at train time):

```bash
python3 scripts/prepare_stage1_librispeech.py +experiment=paper_ls460
```

Fbank-only (after manifests exist):

```bash
python3 scripts/prepare_stage1_fbank.py +experiment=paper_ls460
```

### Train Stage I

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 scripts/train_stage1_ctc.py \
  +experiment=paper_ls460 \
  run.devices=4
```

Checkpoints: `/data/dma-kws/exp/stage1_phoneme_ctc/checkpoints/`

When `stage1.checkpoint_avg.enabled: true` (set in `paper_ls460.yaml`), training auto-averages the last 10 Lightning checkpoints to `avg_10.ckpt` and also writes `stage1_avg.pt` (encoder weights loadable by Stage II).

Manual averaging:

```bash
python3 scripts/average_checkpoints.py \
  --input-dir /data/dma-kws/exp/stage1_phoneme_ctc/checkpoints \
  --pattern "*.ckpt" \
  --last-k 10 \
  --output /data/dma-kws/exp/stage1_phoneme_ctc/checkpoints/avg_10.ckpt
```

Use your averaged Stage I checkpoint for Stage II init (frozen encoder recipe):

```text
/data/dma-kws/exp/stage1_phoneme_ctc/checkpoints/avg_10.ckpt
```

---

## 3. Stage II: precomputed fbank features

Stage II training reads **precomputed 80-dim fbank `.npy`** files (not online fbank). Paths are resolved as:

```text
features/fbank/LP-460-fbank/<clip>.npy
features/fbank/GP-1000-fbank/<clip>.npy
```

These must be produced with the same Kaldi fbank settings as `dma_kws/stage2/features.py` (80 mel bins, 25 ms frame / 10 ms shift, Povey window). For LibriPhrase-460 you can download precomputed shards from HuggingFace, or generate them from decoded audio using `scripts/prepare_stage2_paper.py` (LibriPhrase-100 demo) or the Wenet/main feature extraction pipeline.

Set `stage2.wav_dir` in config (default: `/data/dma-kws/features/fbank`).

For the LibriPhrase-100 smoke demo, run:

```bash
python3 scripts/prepare_stage2_paper.py \
  +experiment=demo_librispeech100 \
  --limit-anchors 100
```

This writes `processed/stage2_qbyt/aggregated_segments_with_g2p_distance.parquet` plus `clips/`, `distances/`, and fbank `.npy` shards under `features/fbank/`.

---

## 4. Stage II: init training (`init-ls-460`)

Recipe: LibriPhrase-460, random + hard negatives (1:1), `utt_loss + seq_loss`, Adam + cosine warmup, DDP.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 scripts/train_stage2_recipe.py \
  +experiment=paper_ls460 \
  training.recipe=init-ls-460 +experiment=paper_ls460 \
  run.devices=4
```

Equivalent direct entry point:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 scripts/train_stage2_qbyt.py \
  +experiment=paper_ls460 \
  run.devices=4
```

Checkpoints: `/data/dma-kws/exp/stage2_qbyt/checkpoints/init-ls-460/`

### Average init checkpoints

```bash
python3 scripts/average_checkpoints.py \
  --input-dir /data/dma-kws/exp/stage2_qbyt/checkpoints/init-ls-460 \
  --pattern "step_step=*.ckpt" \
  --last-k 10 \
  --output /data/dma-kws/exp/stage2_qbyt/checkpoints/init-ls-460/avg_10.ckpt
```

---

## 5. Stage II: finetune (`ft-ls-gs-1460`)

Finetune from averaged init checkpoint on LS+GigaPhrase-1460 with hard-negative ratio 100:1:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 scripts/train_stage2_recipe.py \
  +experiment=paper_ls_gs1460 \
  training.recipe=ft-ls-gs-1460 +experiment=paper_ls_gs1460 \
  run.devices=4
```

`paper_ls_gs1460.yaml` sets `stage2.init_checkpoint` to the averaged init weights. Checkpoints: `/data/dma-kws/exp/stage2_qbyt/checkpoints/ft-ls-gs-1460/`

Average the last 10 finetune checkpoints the same way as init.

### Optional: frozen self-trained Stage I encoder (`frozen-wenet-encoder`)

Train QbyT only (~187k params) with **your** averaged Stage I encoder frozen (not an external Wenet checkpoint):

```bash
python3 scripts/train_stage2_recipe.py \
  +experiment=paper_ls460 \
  --recipe frozen-wenet-encoder \
  run.devices=4 \
  run.init_checkpoint=/data/dma-kws/exp/stage1_phoneme_ctc/checkpoints/avg_10.ckpt
```

If `--init-checkpoint` is omitted, the recipe resolves `stage1.checkpoint_avg` output under your `exp_root`.

## 6. Evaluation: LibriPhrase hard / easy

Evaluate on the official LibriPhrase-460 eval splits (AUC / EER):

```bash
python3 scripts/eval_stage2_libriphrase.py \
  +experiment=paper_ls460 \
  --checkpoint /data/dma-kws/exp/stage2_qbyt/checkpoints/init-ls-460/avg_10.ckpt \
  --split hard
```

Splits: `easy`, `hard`, or `all`. Example output:

```json
{"split": "hard", "auc": 0.9785, "eer": 0.0613}
```

Eval paths come from `stage2.eval` in config (`test_dir`, `aggregate_csv`).

---

## 7. Two-stage inference demo (streaming Stage I)

Stage I candidate search uses **prefix beam search + ContextGraph** (not greedy substring matching):

```bash
STAGE1_CKPT=/data/dma-kws/exp/stage1_phoneme_ctc/checkpoints/avg_10.ckpt
STAGE2_CKPT=/data/dma-kws/exp/stage2_qbyt/checkpoints/init-ls-460/avg_10.ckpt

python3 scripts/run_two_stage_demo.py \
  +experiment=paper_ls460 \
  --stage1-ckpt "$STAGE1_CKPT" \
  --stage2-ckpt "$STAGE2_CKPT" \
  --audio /path/to/test.wav \
  --keyword "hello world"
```

---

## 8. Expected metrics

Paper-reported LibriPhrase **hard** split (from original README / main logs):

| Split | AUC | EER |
|-------|-----|-----|
| hard  | 97.85% | 6.13% |

Acceptance criterion for this alignment work: reproduced AUC/EER within **1% absolute** of main on the same config and data.

Small-scale presets (`demo_librispeech100.yaml`, `run.limit_steps=20`) verify the pipeline runs but will **not** match these numbers.

---

## 9. Full recipe chain (summary)

```text
LibriSpeech-460 + LibriPhrase-460 + GigaPhrase
  → Stage I prep (prepare_stage1_wenet.sh: manifests + optional fbank)
  → Stage I train → avg last 10 ckpts → avg_10.ckpt (self-trained)
  → Stage II prep (prepare_stage2_paper.py) + fbank features
  → init-ls-460 (50k steps) → avg_10.ckpt
  → ft-ls-gs-1460 (100k steps, hard neg 100:1) → avg_10.ckpt
  → eval_stage2_libriphrase.py --split hard
  → run_two_stage_demo.py (streaming Stage I)
```

Smoke validation without full data:

```bash
bash scripts/run_smoke.sh
```
