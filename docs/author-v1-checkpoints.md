# Paper-original v1 (SI) checkpoints

This page covers the author's paper-original speaker-independent QbyT
("readout version 1"): the released `155k-v2-ft.ckpt` plus the
`ls-gs-1460.pt` Wenet ASR Stage I model. The multimodal enrolment releases
(`155k-mm-f1.ckpt`, `155k-mm-f2.ckpt`) are **not** supported: they add
`modality_enc.enr_audio_emb` and `cross_attn.*`, and the importer refuses them.

## 1. Import

The raw releases are unversioned and every loader rejects unversioned QbyT
weights on purpose, because v1's deployed score (the GRU state at the final
position of the padded text-then-audio concatenation) is not the same function
as any later readout. Import once, with the author checkout available:

```bash
python scripts/import_author_v1_checkpoints.py \
  --stage1 /path/to/ckpts/stage1/ls-gs-1460.pt \
  --stage2 /path/to/ckpts/stage2/155k-v2-ft.ckpt \
  --output-dir data/dma-kws/exp/author_v1
```

Outputs:

* `stage1_v1.pt` - encoder + CTC head (the 110 ASR `decoder.*` tensors are
  dropped), 73-symbol vocabulary, usable as `prep.stage1_ckpt`.
* `stage2_v1_si.pt` - encoder + v1 QbyT, stamped `qbyt_readout_version=1`,
  strict-loadable.

The importer proves the architecture with a strict state-dict match and prints
the matched tensor counts (stage1 249 + 2; stage2 249 + 38).

## 2. Evaluate

```bash
python scripts/eval_stage2_clips.py +experiment=v1_eval_author_si \
  prep.stage2_ckpt=data/dma-kws/exp/author_v1/stage2_v1_si.pt \
  prep.manifest=/path/to/manifest.csv
```

The manifest is CSV or JSONL with `audio_path`, `keyword` and optional
`label` / `keyword_phonemes`. `prep.batch_size` only controls data loading:
v1 readout scores clips one at a time regardless, because batching would pad
clips together and change the deployed score.

## 3. Fidelity check against the author's code

```bash
python scripts/parity_check_author_v1.py \
  --upstream-root /path/to/DMA-KWS \
  --stage2-ckpt ckpts/stage2/155k-v2-ft.ckpt \
  --repo-ckpt data/dma-kws/exp/author_v1/stage2_v1_si.pt \
  --audio path/to/clip.wav --keyword-phonemes "K AE1 R IH0 JH"
```

It runs the author's `decode/models/encoder.py` + `decode/stage2/model.py` and
this repo's imported payload on identical fbank features and requires
agreement within 1e-5. The first run on LibriPhrase-100 audio matched to
0.000e+00.

## 4. Training

Three Hydra experiments mirror the author's train/two_stage/ scripts. All of
them pin the 73-token dictionary, readout version 1, the membership sequence
target and the paper's fixed objective (utterance BCE plus full-prefix sequence
BCE at 1:1, token-normalized), and disable the features v1 never had
(negative-tail loss, background negatives, phoneme adapter):

| Config | Author script | Encoder | Data | LR | Steps |
| --- | --- | --- | --- | --- | --- |
| +experiment=v1_paper_ls460 | train.py | trained from scratch | LibriPhrase-460 | 1e-3 | 50k |
| +experiment=v1_paper_ls_gs1460 | train_2.py + train_2_2ft.py | initialized from the v1 init average, unfrozen | LS+GigaPhrase-1460 (155k anchors), hard negatives 100:1 | 5e-4 | 100k |
| +experiment=v1_frozen_wenet_encoder | train_frozen.py | frozen | LibriPhrase-460 | 1e-3 | 50k |

Example: finetune from the imported author release instead of a local init:

    python scripts/train_stage2_qbyt.py +experiment=v1_paper_ls_gs1460 \
      run.init_checkpoint=data/dma-kws/exp/author_v1/stage2_v1_si.pt

The module refuses to start when a v1 run asks for a different sequence
objective or enables an unsupported feature, because the dataset derives its
sequence targets from the same config: a mismatch would train membership-shaped
labels with progress semantics. Every saved checkpoint is stamped with
qbyt_readout_version=1 and the membership objective, so the exported .pt file
loads straight into Stage2Verifier and eval_stage2_clips with a strict
state-dict match.

## 5. Known differences from the paper numbers

* `155k-v2-ft.ckpt` is a single step-3000 snapshot; the author's own scripts
  evaluate `155k-v2-ft/avg_10.ckpt` (average of the last ten). Expect small
  score differences from the published AUC/EER.
* The v1 models use the 73-symbol dictionary
  (`data/dict/lang_char_v1_73.txt`, vendored): ids differ from the repo's
  71-symbol canonical dictionary. Mixing them fails on a shape mismatch, by
  design.
* Inference runs full-context (`stage1.stream` disabled), matching the
  author's `decode/` scripts. Enabling chunked attention changes the scores.
