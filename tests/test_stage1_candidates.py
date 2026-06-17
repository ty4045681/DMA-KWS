from dma_kws.stage1.candidates import PhonemeFrame, find_keyword_candidates


def test_find_keyword_candidates_returns_time_span_with_margin():
    frames = [
        PhonemeFrame("SIL", 0.00, 0.10, -0.1),
        PhonemeFrame("HH", 0.10, 0.20, -0.2),
        PhonemeFrame("AH", 0.20, 0.30, -0.3),
        PhonemeFrame("L", 0.30, 0.40, -0.4),
        PhonemeFrame("OW", 0.40, 0.50, -0.5),
        PhonemeFrame("SIL", 0.50, 0.60, -0.1),
    ]

    candidates = find_keyword_candidates(frames, ["HH", "AH", "L", "OW"], margin_sec=0.05)

    assert len(candidates) == 1
    assert candidates[0].start_sec == 0.05
    assert candidates[0].end_sec == 0.55
    assert candidates[0].phonemes == ["HH", "AH", "L", "OW"]
