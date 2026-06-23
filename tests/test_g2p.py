from dma_kws.g2p import clean_phoneme_tokens, text_to_phonemes


def test_clean_phoneme_tokens_strips_stress_and_drops_spaces():
    tokens = ["HH", "AH0", " ", "L", "OW1", ""]

    assert clean_phoneme_tokens(tokens) == ["HH", "AH", "L", "OW"]


def test_clean_phoneme_tokens_can_keep_stress():
    tokens = ["AH0", "OW1"]

    assert clean_phoneme_tokens(tokens, strip_stress=False) == ["AH0", "OW1"]


def test_text_to_phonemes_uses_stub_g2p_and_normalizes():
    captured = {}

    def fake_g2p(normalized):
        captured["text"] = normalized
        return ["HH", "AH0", " ", "L", "OW1"]

    phonemes = text_to_phonemes(fake_g2p, "Hello!")

    assert captured["text"] == "hello"
    assert phonemes == ["HH", "AH", "L", "OW"]


def test_text_to_phonemes_preserves_stress_when_disabled():
    phonemes = text_to_phonemes(lambda _: ["AH0", " ", "OW1"], "x", strip_stress=False)

    assert phonemes == ["AH0", "OW1"]
