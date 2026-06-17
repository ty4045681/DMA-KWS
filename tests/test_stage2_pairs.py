from dma_kws.stage2.pairs import AnchorExample, make_pair_records


def test_make_pair_records_creates_positive_and_negative_pair():
    anchors = [
        AnchorExample(text="hello", phonemes=["HH", "AH", "L", "OW"], clips=["a.wav"]),
        AnchorExample(text="world", phonemes=["W", "ER", "L", "D"], clips=["b.wav"]),
    ]

    pairs = make_pair_records(anchors, negatives_per_anchor=1, seed=7)

    assert len(pairs) == 4
    assert pairs[0].label == 1
    assert pairs[0].wav_path == "a.wav"
    assert any(pair.label == 0 and pair.anchor_text == "hello" and pair.wav_path == "b.wav" for pair in pairs)
