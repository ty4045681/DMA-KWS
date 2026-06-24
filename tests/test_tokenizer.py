from pathlib import Path

from dma_kws.tokenizer import build_seq_label, load_char_tokenizer, tokenize_phoneme_string

REPO_ROOT = Path(__file__).resolve().parents[1]
DICT_PATH = REPO_ROOT / "data" / "dict" / "lang_char.txt"


def test_tokenize_phoneme_string_returns_int_list():
    tok = load_char_tokenizer(DICT_PATH)
    ids = tokenize_phoneme_string(tok, "HH AH L OW")

    assert isinstance(ids, list)
    assert ids
    assert all(isinstance(token_id, int) for token_id in ids)


def test_build_seq_label_membership():
    assert build_seq_label([10, 11, 12], [11, 99]) == [0, 1, 0]


def test_load_char_tokenizer_loads_lang_char_dict():
    tok = load_char_tokenizer(DICT_PATH)

    assert tok.vocab_size() == 73
    assert tok.symbol_table["<blank>"] == 0
    assert tok.symbol_table["<unk>"] == 1
    assert tok.symbol_table["HH"] == 17
    assert tok.symbol_table["<sos/eos>"] == 72
    assert "HH" in tok.symbol_table
    assert "AH" in tok.symbol_table
