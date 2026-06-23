"""Pure-Python metric helpers for Stage I PER validation.

These helpers are intentionally torch-free (and numpy-free) so they unit-test
in a CPU-only environment. They mirror the paper code:
- edit_distance / per adapt wenet/runtime/gpu/client/utils.py:_levenshtein_distance
- collapse_ctc mirrors qbyt/models/utils/ctc_utils.py:remove_duplicates_and_blank
"""

from __future__ import annotations


def edit_distance(ref: list, hyp: list) -> int:
    """Levenshtein distance between two token sequences.

    The minimum number of single-token edits (substitutions, insertions or
    deletions) required to change ``ref`` into ``hyp``. Adapted from
    wenet ``_levenshtein_distance`` using O(min(m, n)) space, pure python.
    """
    m = len(ref)
    n = len(hyp)

    # special case
    if ref == hyp:
        return 0
    if m == 0:
        return n
    if n == 0:
        return m

    if m < n:
        ref, hyp = hyp, ref
        m, n = n, m

    # use O(min(m, n)) space: two rolling rows
    distance = [[0] * (n + 1) for _ in range(2)]

    # initialize distance matrix
    for j in range(n + 1):
        distance[0][j] = j

    # calculate levenshtein distance
    for i in range(1, m + 1):
        prev_row_idx = (i - 1) % 2
        cur_row_idx = i % 2
        distance[cur_row_idx][0] = i
        for j in range(1, n + 1):
            if ref[i - 1] == hyp[j - 1]:
                distance[cur_row_idx][j] = distance[prev_row_idx][j - 1]
            else:
                s_num = distance[prev_row_idx][j - 1] + 1
                i_num = distance[cur_row_idx][j - 1] + 1
                d_num = distance[prev_row_idx][j] + 1
                distance[cur_row_idx][j] = min(s_num, i_num, d_num)

    return distance[m % 2][n]


def per(ref_tokens: list[str], hyp_tokens: list[str]) -> tuple[int, int]:
    """Per-utterance phoneme error pieces.

    Returns ``(edit_distance, len(ref_tokens))`` so callers can aggregate a
    corpus-level PER = ``sum(dists) / sum(ref_lens)``.
    """
    return edit_distance(ref_tokens, hyp_tokens), len(ref_tokens)


def collapse_ctc(ids: list[int], blank_id: int = 0) -> list[int]:
    """Collapse a raw CTC id sequence to its phoneme tokens.

    Removes consecutive duplicate ids and drops blanks, mirroring
    qbyt ``remove_duplicates_and_blank`` (pure python, no torch).
    """
    new_hyp: list[int] = []
    cur = 0
    while cur < len(ids):
        if ids[cur] != blank_id:
            new_hyp.append(ids[cur])
        prev = cur
        while cur < len(ids) and ids[cur] == ids[prev]:
            cur += 1
    return new_hyp
