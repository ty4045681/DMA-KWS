from dma_kws.stage2.pairs import AnchorExample, PairRecord, clip_to_audio_rel, make_pair_records


def test_clip_to_audio_rel_strips_lp100_prefix():
    assert clip_to_audio_rel("LP-100/missus_rachel/103-1240-0000_000.wav") == "missus_rachel/103-1240-0000_000.wav"


def test_clip_to_audio_rel_without_prefix_is_identity():
    assert clip_to_audio_rel("missus_rachel/103-1240-0000_000.wav") == "missus_rachel/103-1240-0000_000.wav"


def test_pair_record_carries_sample_rate():
    record = PairRecord(
        anchor_text="hi",
        anchor_phonemes=["HH", "AY"],
        wav_path="audio/a.npy",
        label=1,
        sample_rate=16000,
    )
    assert record.to_json_dict()["sample_rate"] == 16000


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
