# Wenet phoneme vocabulary (`lang_char.txt`)

This directory holds the repo-canonical Wenet-format phoneme dictionary used by **both** training stages.

## Contents

- **`lang_char.txt`** — 71-token `CharTokenizer` vocabulary (ids 0–70): CTC `<blank>`, `<unk>`, and the full **stress-marked** CMU ARPAbet inventory — 15 vowels × 3 stress levels (`AA0`–`UW2`) plus 24 consonants (`B`–`ZH`). This matches the 71-symbol phoneme inventory reported in the paper, and is the default for both stages.
- **`lang_char_v1_73.txt`** — the author's paper-original 73-token dictionary (ids 0–72), vendored verbatim from the upstream `librispeech-g2p` recipe: the canonical inventory plus a reserved `<sos/eos>` (id 2) and a stress-less `UW`. Every id after those insertions shifts, so it is **not** interchangeable with `lang_char.txt`. It is required only by the paper-original v1 QbyT checkpoints (`tokenizer.dict_path: data/dict/lang_char_v1_73.txt`, see `docs/author-v1-checkpoints.md`).

`g2p_en` emits stress digits on every vowel, so training, evaluation and inference all tokenize those symbols directly — nothing strips stress. The one exception is hard-negative mining, which collapses `AH0`/`AH1`/`AH2` before ranking phoneme edit distances because confusability is about phone identity; both mining paths do this — `prep.recompute_distances.strip_stress` and the in-prep fallback `compute_hard_negatives_from_phonemes`.

Symbols outside this inventory are rejected at data-preparation time (`dma_kws.tokenizer.unsupported_phones`) instead of silently becoming `<unk>`. Punctuation that `g2p_en` echoes verbatim for letterless tokens (the `'` in `boys'`) is not a phoneme and is dropped by `clean_phoneme_tokens` before that check.

## Canonical source of truth

Both files are supported vocabularies, but only one applies to a given checkpoint: the canonical 71-token dict is the default for Stage I (phoneme CTC) and Stage II (QbyT), and the 73-token author dict is required by the paper-original v1 QbyT releases (`docs/author-v1-checkpoints.md`). Validation is enforced in code via `dma_kws.tokenizer.validate_lang_char_dict(path, profile=...)`, which auto-detects the profile from the token count; loading a checkpoint with the wrong dict fails on the embedding/CTC shape instead of silently mis-tokenizing.

Do **not** replace or edit this file without retraining **both** stages. Checkpoints, CTC graphs, and Stage II embeddings all assume exactly this token inventory and id assignment.

## Migration from the 73-token stress-stripped vocabulary

This section is about the repo's retired *stress-stripped* 73-token dict; it is not the author dictionary vendored as `lang_char_v1_73.txt`, which is stress-marked and served by the v1 profile.

Checkpoints trained against the old 73-token dict are **not loadable** against the canonical vocabulary: the QbyT text embedding and the Stage I CTC head are both sized by the vocabulary. Rebuild in this order:

1. Re-run `scripts/prepare_stage2_paper.py` — it regenerates `ngram_g2p` when the parquet column has no stress markers and reports `g2p_recomputed=True`. Precomputed fbank `.npy` features are **unaffected** and do not need recomputing.
2. Re-run `scripts/prepare_stage1_librispeech.py` if you train Stage I: the old `train.jsonl`/`dev.jsonl` carry stress-stripped `phonemes_g2p`. `Stage1Dataset` refuses to load such a manifest rather than tokenizing every vowel to `<unk>`.
3. Retrain Stage II (and Stage I if you use the repo's phoneme-CTC locator), then re-measure the LibriPhrase baseline before starting a new adaptation sweep.

## Validation

```bash
python -m pytest tests/test_lang_char_dict.py -q
```

Any custom dict path passed to `load_char_tokenizer` must pass the same checks.
