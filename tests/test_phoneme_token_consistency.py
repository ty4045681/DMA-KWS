"""Training, evaluation and inference must tokenize text into identical ids.

Regression guard for the stress-marker split that silently mapped every eval
anchor vowel to ``<unk>``: the LibriPhrase eval dataset tokenized raw g2p_en
output (``AH0``) while training tokenized stress-stripped phonemes (``AH``).
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dma_kws.g2p import text_to_phonemes
from dma_kws.stage2.dataset import LibriPhraseEvalDataset
from dma_kws.stage2.prepare_paper import _resolve_g2p, needs_g2p_recompute
from dma_kws.tokenizer import (
    CANONICAL_DICT_PATH,
    load_char_tokenizer,
    tokenize_phoneme_string,
    unsupported_phones,
)

KEYWORD = "hello world"
STRESS_MARKED_G2P = "HH AH0 L OW1 W ER1 L D"
LEGACY_STRESS_STRIPPED_G2P = "HH AH L OW W ER L D"


def _fake_g2p(text: str) -> list[str]:
    """Mimic g2p_en output: stress digits on vowels, spaces between words."""
    assert text == KEYWORD, text
    return ["HH", "AH0", "L", "OW1", " ", "W", "ER1", "L", "D"]


@pytest.fixture
def tokenizer():
    return load_char_tokenizer(CANONICAL_DICT_PATH, split_with_space=" ")


@pytest.fixture
def eval_dataset(monkeypatch, tokenizer):
    monkeypatch.setattr(
        "dma_kws.stage2.dataset.np.load",
        lambda path, allow_pickle=False: np.ones((4, 80), dtype=np.float32),
    )
    df = pd.DataFrame(
        {
            "anchor_text": [KEYWORD],
            "anchor": ["a.wav"],
            "anchor_dur": [1.0],
            "comparison_text": [KEYWORD],
            "comparison": ["pos.wav"],
            "comparison_dur": [1.0],
            "target": [1],
            "type": ["diffspk_positive"],
        }
    )
    return LibriPhraseEvalDataset(
        test_dir="/data/eval",
        split="all",
        df=df,
        tokenizer=tokenizer,
        g2p=_fake_g2p,
    )


def test_eval_anchor_ids_contain_no_unknown_tokens(eval_dataset, tokenizer):
    unk_id = tokenizer.symbol_table["<unk>"]

    anchor_ids = eval_dataset[0]["anchor_seq"].tolist()

    assert anchor_ids
    assert unk_id not in anchor_ids


def test_train_eval_inference_produce_identical_anchor_ids(eval_dataset, tokenizer):
    # Training reads the precomputed parquet column.
    train_g2p, _ = _resolve_g2p({"ngram": KEYWORD, "ngram_g2p": STRESS_MARKED_G2P}, None)
    train_ids = tokenize_phoneme_string(tokenizer, train_g2p)
    # Inference (Stage2ClipRunner, two-stage pipeline, adaptation datasets).
    inference_ids = tokenize_phoneme_string(
        tokenizer, " ".join(text_to_phonemes(_fake_g2p, KEYWORD))
    )
    eval_ids = eval_dataset[0]["anchor_seq"].tolist()

    assert train_ids == eval_ids == inference_ids


def test_resolve_g2p_rejects_phonemes_outside_the_vocabulary():
    with pytest.raises(ValueError, match="outside the vocabulary"):
        _resolve_g2p({"ngram": KEYWORD, "ngram_g2p": LEGACY_STRESS_STRIPPED_G2P}, None)


def test_needs_g2p_recompute_detects_legacy_columns():
    stress_marked = pd.DataFrame({"ngram_g2p": [STRESS_MARKED_G2P]})
    legacy = pd.DataFrame({"ngram_g2p": [LEGACY_STRESS_STRIPPED_G2P]})

    assert not needs_g2p_recompute(stress_marked)
    assert needs_g2p_recompute(legacy)
    assert needs_g2p_recompute(stress_marked, force=True)
    assert needs_g2p_recompute(pd.DataFrame({"ngram": [KEYWORD]}))


def test_unsupported_phones_flags_stress_stripped_vowels():
    assert unsupported_phones(["HH", "AH0", "L"]) == []
    assert unsupported_phones(["HH", "AH", "L", "OW"]) == ["AH", "OW"]


def test_repo_dict_matches_the_paper_inventory():
    lines = Path(CANONICAL_DICT_PATH).read_text(encoding="utf-8").splitlines()

    assert len(lines) == 71
    assert lines[0] == "<blank> 0"
    assert lines[1] == "<unk> 1"
