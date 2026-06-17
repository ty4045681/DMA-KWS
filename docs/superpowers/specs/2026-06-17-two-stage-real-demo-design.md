# Two-Stage Real Demo Design

## Goal

Build a small-scale but genuinely trained two-stage DMA-KWS demo on a remote Linux machine with 2× NVIDIA V100 GPUs.

The demo will train both stages from data rather than using toy/random checkpoints:

1. **Stage I:** train a phoneme CTC model that maps speech to phoneme posteriors/decoded phoneme hypotheses and produces keyword candidate regions.
2. **Stage II:** train a QbyT phoneme matching model that verifies each candidate against the keyword phoneme sequence.
3. **Demo:** run an end-to-end command that takes an audio file and a keyword text string, then returns candidates, QbyT scores, and a final detected/not-detected decision.

The first version prioritizes a real working pipeline over paper-level SOTA metrics.

## Scope

### In scope

- Make paths configurable instead of relying on hard-coded `/nvme01/openkws/...` paths.
- Define a reproducible local/remote data layout.
- Add data preparation for a small real training setup.
- Train Stage I from LibriSpeech `train-clean-100` transcripts converted to phonemes.
- Train Stage II from `ZhiqiAi/LibriPhrase-100` or equivalent small LibriPhrase-derived data.
- Reuse the repository's existing Conformer, CTC, search, and QbyT components where practical.
- Provide CLI commands for preparation, training, evaluation, and demo inference.
- Produce a demo with real detection behavior on held-out examples.

### Out of scope for the first version

- Full paper-scale reproduction on `LibriPhrase-460` / `GigaPhrase-1000`.
- Paper-level AUC/EER claims.
- Streaming Stage I inference as the default path.
- Parameter-efficient continual adaptation claims such as exactly 187k trainable parameters.
- Production deployment/runtime optimization.

## Recommended approach

Use **LibriSpeech-100 + LibriPhrase-100 small-scale real training**.

This is the best first target because it is large enough to train non-toy models, but small enough to iterate on 2× V100 GPUs. Once the pipeline works, the same interfaces can scale to `LibriPhrase-460` and `GigaPhrase-1000`.

Alternative approaches considered:

1. **Toy/smoke-test demo:** fastest to run, but does not satisfy the requirement to train both stages with useful detection behavior.
2. **Full paper-scale reproduction:** closest to the paper, but current repository lacks the complete Stage I recipe, preprocessing pipeline, and checkpoints; starting there would make debugging slow and brittle.

## Target user workflow

The intended remote Linux workflow should look like this:

```bash
# 1. Prepare environment
conda create -n dma-kws python=3.10 -y
conda activate dma-kws
pip install -r requirements.txt

# 2. Download raw datasets outside this repo
# LibriSpeech train-clean-100, dev-clean/test-clean as needed
# ZhiqiAi/LibriPhrase-100

# 3. Prepare Stage I data
python scripts/prepare_stage1_librispeech.py \
  --config configs/demo_librispeech100.yaml

# 4. Train Stage I phoneme CTC
python scripts/train_stage1_ctc.py \
  --config configs/demo_librispeech100.yaml \
  --devices 2

# 5. Prepare Stage II data
python scripts/prepare_stage2_libriphrase.py \
  --config configs/demo_librispeech100.yaml

# 6. Train Stage II QbyT
python scripts/train_stage2_qbyt.py \
  --config configs/demo_librispeech100.yaml \
  --devices 2

# 7. Run two-stage demo
python scripts/run_two_stage_demo.py \
  --config configs/demo_librispeech100.yaml \
  --audio examples/demo.wav \
  --keyword "hello world"
```

The exact script names may change during implementation if keeping compatibility with existing files is cleaner, but the workflow should remain this clear.

## Data layout

Use a configurable project data root, for example:

```text
/data/dma-kws/
├── raw/
│   ├── LibriSpeech/
│   │   ├── train-clean-100/
│   │   ├── dev-clean/
│   │   └── test-clean/
│   └── LibriPhrase-100/
├── processed/
│   ├── stage1_phoneme_ctc/
│   │   ├── train.jsonl
│   │   ├── dev.jsonl
│   │   ├── test.jsonl
│   │   └── phoneme_vocab.txt
│   └── stage2_qbyt/
│       ├── train.parquet
│       ├── dev.parquet
│       ├── test.parquet
│       └── phoneme_vocab.txt
├── features/
│   ├── librispeech_fbank/
│   └── libriphrase_fbank/
└── exp/
    ├── stage1_phoneme_ctc/
    └── stage2_qbyt/
```

The repository should not require this exact absolute root. It should come from YAML config.

## Stage I design: phoneme CTC

### Inputs

- LibriSpeech audio files.
- LibriSpeech transcript text.
- A G2P converter, initially `g2p_en`, to convert transcript text to ARPAbet-like phoneme tokens.

### Preparation

The Stage I preparation step will:

1. Scan LibriSpeech transcript files.
2. Normalize text consistently.
3. Convert text to phonemes.
4. Build `phoneme_vocab.txt` with reserved tokens:
   - `<blank>` id 0
   - `<unk>` id 1
5. Extract or lazily compute 80-dim fbank features.
6. Save train/dev/test manifests as JSONL.

Manifest records should include:

```json
{
  "utt_id": "103-1240-0000",
  "wav_path": "/data/dma-kws/raw/LibriSpeech/train-clean-100/...flac",
  "feature_path": "/data/dma-kws/features/librispeech_fbank/...npy",
  "text": "chapter one",
  "phonemes": ["CH", "AE", "P", "T", "ER", "W", "AH", "N"],
  "phoneme_ids": [12, 5, 44, 51, 18, 57, 4, 39]
}
```

### Model

Reuse the existing modules:

- `qbyt/models/encoder.py` for `ConformerEncoder`.
- `qbyt/models/ctc.py` for CTC projection/loss.
- `qbyt/models/search.py` for CTC decoding utilities.

The first Stage I model should be offline CTC:

```text
fbank -> ConformerEncoder -> CTC phoneme logits
```

Streaming/chunked inference can be added after offline inference works.

### Training

Use PyTorch Lightning or a simple PyTorch training loop. Prefer consistency with the existing `qbyt/train.py` Lightning style unless it introduces unnecessary complexity.

Recommended starting parameters for 2× V100:

- input feature dim: 80
- encoder output dim: 144, matching current QbyT expectations
- encoder blocks: 6, matching current code
- batch size: start with 32–64 utterances per GPU for Stage I, then tune
- precision: optional mixed precision if stable
- validation metric: phoneme error rate or token error rate
- checkpoint output: `exp/stage1_phoneme_ctc/checkpoints/best.pt`

### Stage I candidate generation

The first version should generate candidates offline:

1. Decode audio into a phoneme sequence with approximate frame/peak times.
2. Convert keyword text to keyword phoneme sequence.
3. Search for approximate phoneme subsequence matches in the decoded stream.
4. Expand matching regions with a small left/right margin.
5. Return candidate time spans.

Candidate record:

```json
{
  "start_sec": 1.24,
  "end_sec": 2.03,
  "stage1_score": -3.82,
  "decoded_phonemes": ["HH", "AH", "L", "OW"]
}
```

This is sufficient for the first real demo. Streaming search can later reuse the same scoring interface.

## Stage II design: QbyT matching

### Inputs

- `LibriPhrase-100` phrase-level training data.
- Phrase audio clips or paths to clips.
- Anchor phrase text/phoneme sequence.
- Positive and negative pairs.

### Preparation

The Stage II preparation step will:

1. Load `LibriPhrase-100` data.
2. Normalize anchor phrase text.
3. Convert anchors to phoneme sequences using the same phoneme vocabulary where possible.
4. Generate positive pairs from matching phrase/audio clips.
5. Generate random negative pairs.
6. Add simple hard negatives if the dataset provides distance/confusability metadata; otherwise keep this optional for version one.
7. Extract fbank `.npy` files for phrase clips.
8. Write a compact training index compatible with a cleaned-up dataset loader.

The initial prepared training file can be parquet or JSONL. Parquet is preferred for larger data, but JSONL is acceptable for easier debugging.

### Model

Reuse and minimally improve:

- `qbyt/model.py::QbyT`
- `qbyt/models/encoder.py::ConformerEncoder`

The current QbyT architecture is:

```text
anchor phoneme ids -> text embedding + position + modality embedding
candidate fbank -> ConformerEncoder -> audio projection + position + modality embedding
concat(text, audio) -> TransformerEncoder -> GRU -> binary score
```

The first implementation should keep the architecture stable to avoid changing too many variables at once.

However, the design should include a targeted fix for padding/masks because the current `QbyT.forward()` does not mask padded text/audio frames and uses the last GRU state directly. This can be implemented in a compatibility-preserving way by passing lengths/masks from the wrapper.

### Training

Recommended first run:

- initialize Stage II encoder from the trained Stage I encoder if state dicts match cleanly;
- otherwise train Stage II encoder + QbyT jointly from scratch on LibriPhrase-100;
- train with 2 GPUs using DDP;
- start with batch size 128–256 total and tune upward;
- evaluate on a held-out split and/or official LibriPhrase eval if available.

Outputs:

```text
/data/dma-kws/exp/stage2_qbyt/
├── checkpoints/
│   └── best.ckpt
├── metrics.json
└── config.yaml
```

## Two-stage demo design

The demo CLI should:

1. Load config.
2. Load Stage I checkpoint and phoneme vocabulary.
3. Load Stage II checkpoint.
4. Convert keyword text to phonemes.
5. Run Stage I over the input audio.
6. Generate candidate spans.
7. Extract fbank for each candidate span.
8. Score each candidate with QbyT.
9. Apply a configurable threshold.
10. Print JSON output.

Example output:

```json
{
  "audio": "examples/demo.wav",
  "keyword": "hello world",
  "keyword_phonemes": ["HH", "AH", "L", "OW", "W", "ER", "L", "D"],
  "stage1_candidates": [
    {
      "start_sec": 1.24,
      "end_sec": 2.03,
      "stage1_score": -3.82
    }
  ],
  "stage2_scores": [
    {
      "start_sec": 1.24,
      "end_sec": 2.03,
      "qbyt_score": 0.87
    }
  ],
  "threshold": 0.5,
  "detected": true
}
```

## Configuration design

A YAML config should define all external paths and hyperparameters:

```yaml
paths:
  data_root: /data/dma-kws
  librispeech_root: /data/dma-kws/raw/LibriSpeech
  libriphrase100_root: /data/dma-kws/raw/LibriPhrase-100
  exp_root: /data/dma-kws/exp

stage1:
  input_dim: 80
  encoder_output_dim: 144
  num_blocks: 6
  batch_size_per_gpu: 48
  max_epochs: 30
  learning_rate: 0.001
  checkpoint_dir: /data/dma-kws/exp/stage1_phoneme_ctc/checkpoints

stage2:
  encoder_output_dim: 144
  qbyt_embed_dim: 128
  qbyt_layers: 2
  batch_size_per_gpu: 128
  max_steps: 50000
  learning_rate: 0.001
  checkpoint_dir: /data/dma-kws/exp/stage2_qbyt/checkpoints

demo:
  stage1_candidate_margin_sec: 0.15
  qbyt_threshold: 0.5
```

The implementation may use a smaller default for laptop smoke tests, but the main config should target the 2× V100 Linux machine.

## Success criteria

The first version is successful when all of the following are true:

1. The repository has documented dataset download requirements.
2. Stage I data preparation runs on LibriSpeech-100 and produces manifests/vocab/features.
3. Stage I training runs on 2× V100 and saves a checkpoint.
4. Stage I decoding produces plausible phoneme sequences and candidate spans.
5. Stage II data preparation runs on LibriPhrase-100 and produces train/eval indexes/features.
6. Stage II training runs on 2× V100 and saves a checkpoint.
7. The two-stage demo command runs on a real audio file and keyword.
8. The demo output includes Stage I candidates, Stage II scores, threshold, and final decision.
9. No required script depends on `/nvme01/openkws/...`.
10. The README explains the small-scale real training workflow.

## Risks and mitigations

### Risk: HuggingFace dataset fields differ from current loader assumptions

Mitigation: write a dedicated inspection/preparation script for `LibriPhrase-100` instead of forcing current hard-coded loaders to work unchanged.

### Risk: Stage I phoneme CTC quality is weak with `train-clean-100`

Mitigation: start with offline decoding and tolerant candidate generation; later scale to more LibriSpeech data or improve G2P/text normalization.

### Risk: Stage II QbyT depends on author-specific parquet/distances files

Mitigation: define a new minimal pair-index format and adapt the dataset loader to it. Hard negatives can start as random negatives and become optional enhanced metadata.

### Risk: Current QbyT ignores padding masks

Mitigation: add length-aware masking/gathering during implementation. This is a focused model correctness fix that supports the demo goal.

### Risk: Full streaming Stage I adds complexity early

Mitigation: implement offline Stage I first. Preserve interfaces so streaming can be added later without changing Stage II or the demo output schema.

## Future extensions

After the small-scale real demo works:

1. Scale Stage I from LibriSpeech-100 to LibriSpeech-460 or larger.
2. Scale Stage II from LibriPhrase-100 to LibriPhrase-460.
3. Add GigaPhrase-1000.
4. Add hard negative mining based on phoneme edit distance or Stage I confusions.
5. Add checkpoint averaging.
6. Add streaming Stage I candidate generation.
7. Add continual adaptation experiments and parameter-count reporting.
8. Reproduce paper-level LibriPhrase hard AUC/EER.
