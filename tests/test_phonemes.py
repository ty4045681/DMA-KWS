from dma_kws.phonemes import PhonemeVocabulary, normalize_english_text


def test_normalize_english_text_lowercases_and_removes_punctuation():
    assert normalize_english_text("Hello, WORLD! It's 2026.") == "hello world it's 2026"


def test_phoneme_vocabulary_round_trip(tmp_path):
    vocab = PhonemeVocabulary.build(["HH", "AH", "L", "OW", "HH"], reserved=("<blank>", "<unk>"))

    assert vocab.token_to_id["<blank>"] == 0
    assert vocab.token_to_id["<unk>"] == 1
    assert vocab.encode(["HH", "MISSING", "OW"]) == [2, 1, 5]
    assert vocab.decode([2, 1, 5]) == ["HH", "<unk>", "OW"]

    path = tmp_path / "phoneme_vocab.txt"
    vocab.write(path)
    loaded = PhonemeVocabulary.read(path)

    assert loaded.token_to_id == vocab.token_to_id
    assert loaded.id_to_token == vocab.id_to_token
