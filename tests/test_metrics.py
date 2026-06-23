from dma_kws.metrics import collapse_ctc, edit_distance, per


def test_edit_distance_identical_sequences_is_zero():
    assert edit_distance(["AH", "L", "OW"], ["AH", "L", "OW"]) == 0


def test_edit_distance_empty_sequences():
    assert edit_distance([], []) == 0
    assert edit_distance([], ["A", "B"]) == 2
    assert edit_distance(["A", "B", "C"], []) == 3


def test_edit_distance_single_substitution():
    assert edit_distance(["A", "B", "C"], ["A", "X", "C"]) == 1


def test_edit_distance_single_insertion():
    assert edit_distance(["A", "C"], ["A", "B", "C"]) == 1


def test_edit_distance_single_deletion():
    assert edit_distance(["A", "B", "C"], ["A", "C"]) == 1


def test_edit_distance_mixed_edits():
    # substitute B->X, insert D, delete nothing else
    assert edit_distance(["A", "B", "C"], ["A", "X", "C", "D"]) == 2


def test_edit_distance_is_symmetric():
    a = ["HH", "AH", "L", "OW"]
    b = ["HH", "EH", "L"]
    assert edit_distance(a, b) == edit_distance(b, a)


def test_per_returns_distance_and_ref_length():
    dist, ref_len = per(["A", "B", "C"], ["A", "X", "C"])
    assert dist == 1
    assert ref_len == 3


def test_per_corpus_level_aggregation():
    refs = [["A", "B", "C"], ["D", "E"]]
    hyps = [["A", "X", "C"], ["D", "E"]]
    pieces = [per(r, h) for r, h in zip(refs, hyps)]
    total_dist = sum(d for d, _ in pieces)
    total_len = sum(n for _, n in pieces)
    assert total_dist == 1
    assert total_len == 5
    assert total_dist / total_len == 0.2


def test_collapse_ctc_removes_blanks_and_consecutive_dups():
    # blank_id=0: 0 1 1 0 2 2 2 0 1 -> 1 2 1
    assert collapse_ctc([0, 1, 1, 0, 2, 2, 2, 0, 1]) == [1, 2, 1]


def test_collapse_ctc_keeps_repeats_separated_by_blank():
    # 1 0 1 -> 1 1 (blank between identical ids preserves both)
    assert collapse_ctc([1, 0, 1]) == [1, 1]


def test_collapse_ctc_all_blank_is_empty():
    assert collapse_ctc([0, 0, 0]) == []


def test_collapse_ctc_empty_input():
    assert collapse_ctc([]) == []


def test_collapse_ctc_custom_blank_id():
    # blank_id=5: 5 1 1 5 2 -> 1 2
    assert collapse_ctc([5, 1, 1, 5, 2], blank_id=5) == [1, 2]
