from pathlib import Path

import pytest

from dma_kws.tokenizer import build_seq_label, load_char_tokenizer, tokenize_phoneme_string

REPO_ROOT = Path(__file__).resolve().parents[1]
DICT_PATH = REPO_ROOT / "data" / "dict" / "lang_char.txt"


def test_tokenize_phoneme_string_returns_int_list():
    tok = load_char_tokenizer(DICT_PATH)
    ids = tokenize_phoneme_string(tok, "HH AH0 L OW1")

    assert isinstance(ids, list)
    assert ids
    assert all(isinstance(token_id, int) for token_id in ids)


def test_stress_stripped_phonemes_are_out_of_vocabulary():
    """Regression: stress-stripped vowels used to silently become <unk>."""
    tok = load_char_tokenizer(DICT_PATH)
    unk_id = tok.symbol_table["<unk>"]

    assert unk_id not in tokenize_phoneme_string(tok, "HH AH0 L OW1")
    assert tokenize_phoneme_string(tok, "HH AH L OW") == [35, unk_id, 44, unk_id]


HEY_GOOGLE = [10, 11, 12, 13, 12, 14, 15]


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        (HEY_GOOGLE, [1, 1, 1, 1, 1, 1, 1]),
        ([99, *HEY_GOOGLE], [1, 1, 1, 1, 1, 1, 1]),
        ([*HEY_GOOGLE, 99], [1, 1, 1, 1, 1, 1, 1]),
        # Play/A/OK Google share most of the suffix but never start the anchor.
        ([20, 21, 11, 12, 13, 12, 14, 15], [0, 0, 0, 0, 0, 0, 0]),
        ([14, 12, 13, 12, 14, 15], [0, 0, 0, 0, 0, 0, 0]),
        ([22, 23, 11, 12, 13, 12, 14, 15], [0, 0, 0, 0, 0, 0, 0]),
        (HEY_GOOGLE[:4], [1, 1, 1, 1, 0, 0, 0]),
        ([10, 99, *HEY_GOOGLE[1:]], [1, 0, 0, 0, 0, 0, 0]),
        ([], [0, 0, 0, 0, 0, 0, 0]),
    ],
)
def test_build_seq_label_uses_ordered_contiguous_prefix_progress(query, expected):
    assert build_seq_label(HEY_GOOGLE, query) == expected


def test_build_seq_label_respects_repeated_phone_count_and_position():
    assert build_seq_label([10, 10, 11], [10, 11]) == [1, 0, 0]
    assert build_seq_label([10, 10, 11], [99, 10, 10, 11, 98]) == [1, 1, 1]


def test_build_seq_label_uses_best_contiguous_query_start():
    assert build_seq_label([10, 11, 12], [10, 99, 10, 11, 88]) == [1, 1, 0]


def test_build_seq_label_rejects_unknown_mode():
    with pytest.raises(ValueError, match="Unsupported seq label mode"):
        build_seq_label([10], [10], mode="edit_distance")


def test_build_seq_label_rejects_empty_anchor():
    with pytest.raises(ValueError, match="anchor_ids must contain"):
        build_seq_label([], [10])


def test_load_char_tokenizer_loads_lang_char_dict():
    tok = load_char_tokenizer(DICT_PATH)

    assert tok.vocab_size() == 71
    assert tok.symbol_table["<blank>"] == 0
    assert tok.symbol_table["<unk>"] == 1
    assert tok.symbol_table["AH0"] == 8
    assert tok.symbol_table["HH"] == 35
    assert tok.symbol_table["ZH"] == 70
    assert "AH1" in tok.symbol_table
    assert "AH2" in tok.symbol_table
    assert "AH" not in tok.symbol_table
