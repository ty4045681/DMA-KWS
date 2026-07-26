from dma_kws.g2p import clean_phoneme_tokens, has_stress_markers, text_to_phonemes
from dma_kws.tokenizer import unsupported_phones


def test_clean_phoneme_tokens_keeps_stress_and_drops_spaces():
    tokens = ["HH", "AH0", " ", "L", "OW1", ""]

    assert clean_phoneme_tokens(tokens) == ["HH", "AH0", "L", "OW1"]


def test_text_to_phonemes_uses_stub_g2p_and_normalizes():
    captured = {}

    def fake_g2p(normalized):
        captured["text"] = normalized
        return ["HH", "AH0", " ", "L", "OW1"]

    phonemes = text_to_phonemes(fake_g2p, "Hello!")

    assert captured["text"] == "hello"
    assert phonemes == ["HH", "AH0", "L", "OW1"]


def test_has_stress_markers_detects_legacy_stress_stripped_strings():
    assert has_stress_markers("HH AH0 L OW1")
    assert not has_stress_markers("HH AH L OW")


def test_clean_phoneme_tokens_drops_punctuation_g2p_en_echoes():
    # g2p_en returns any letterless token unchanged, so "boys' school" yields a
    # bare apostrophe that is not a phoneme and is not in the vocabulary.
    tokens = ["B", "OY1", "Z", "'", " ", "S", "K", "UW1", "L"]

    assert clean_phoneme_tokens(tokens) == ["B", "OY1", "Z", "S", "K", "UW1", "L"]


def test_text_to_phonemes_survives_possessive_apostrophes():
    def fake_g2p(normalized):
        assert normalized == "boys' school"
        return ["B", "OY1", "Z", "'", " ", "S", "K", "UW1", "L"]

    phonemes = text_to_phonemes(fake_g2p, "Boys' School")

    assert "'" not in phonemes
    assert unsupported_phones(phonemes) == []
