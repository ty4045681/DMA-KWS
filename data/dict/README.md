# Wenet phoneme vocabulary (`lang_char.txt`)

This directory holds the repo-canonical Wenet-format phoneme dictionary used by **both** training stages.

## Contents

- **`lang_char.txt`** — 73-token `CharTokenizer` vocabulary (ids 0–72): CTC blank/unk, stress-stripped CMU ARPAbet phones (AA–ZH), punctuation and non-speech symbols, and `<sos/eos>`.

## Canonical source of truth

The file vendored here is the **only** supported vocabulary for Stage I (phoneme CTC) and Stage II (QbyT). The original author checkpoint dictionary is not available for comparison; validation is enforced in code via `dma_kws.tokenizer.validate_lang_char_dict`.

Do **not** replace or edit this file without retraining **both** stages. Checkpoints, CTC graphs, and Stage II embeddings all assume exactly this token inventory and id assignment.

## Validation

```bash
python -m pytest tests/test_lang_char_dict.py -q
```

Any custom dict path passed to `load_char_tokenizer` must pass the same checks.
