# Wenet phoneme vocabulary (`lang_char.txt`)

This directory holds the repo-canonical Wenet-format phoneme dictionary used by **both** training stages.

## Contents

- **`lang_char.txt`** — 71-token `CharTokenizer` vocabulary (ids 0–70): CTC `<blank>`, `<unk>`, and the full **stress-marked** CMU ARPAbet inventory — 15 vowels × 3 stress levels (`AA0`–`UW2`) plus 24 consonants (`B`–`ZH`). This matches the 71-symbol phoneme inventory reported in the paper.

`g2p_en` emits stress digits on every vowel, so training, evaluation and inference all tokenize those symbols directly — nothing strips stress. The one exception is hard-negative mining (`prep.recompute_distances.strip_stress`), which collapses `AH0`/`AH1`/`AH2` before ranking phoneme edit distances because confusability is about phone identity.

Symbols outside this inventory are rejected at data-preparation time (`dma_kws.tokenizer.unsupported_phones`) instead of silently becoming `<unk>`.

## Canonical source of truth

The file vendored here is the **only** supported vocabulary for Stage I (phoneme CTC) and Stage II (QbyT). The original author checkpoint dictionary is not available for comparison; validation is enforced in code via `dma_kws.tokenizer.validate_lang_char_dict`.

Do **not** replace or edit this file without retraining **both** stages. Checkpoints, CTC graphs, and Stage II embeddings all assume exactly this token inventory and id assignment.

## Migration from the 73-token stress-stripped vocabulary

Checkpoints trained against the old 73-token dict are **not loadable**: the QbyT text embedding and the Stage I CTC head are both sized by the vocabulary. Rebuild in this order:

1. Re-run `scripts/prepare_stage2_paper.py` — it regenerates `ngram_g2p` when the parquet column has no stress markers and reports `g2p_recomputed=True`. Precomputed fbank `.npy` features are **unaffected** and do not need recomputing.
2. Retrain Stage II (and Stage I if you use the repo's phoneme-CTC locator), then re-measure the LibriPhrase baseline before starting a new adaptation sweep.

## Validation

```bash
python -m pytest tests/test_lang_char_dict.py -q
```

Any custom dict path passed to `load_char_tokenizer` must pass the same checks.
