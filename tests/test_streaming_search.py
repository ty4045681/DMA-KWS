import torch

from dma_kws.stage1.streaming_search import (
    build_keyword_context_graph,
    decode_keyword_candidates,
)


def test_build_keyword_context_graph_creates_keyword_trie():
    symbol_table = {"<blank>": 0, "A": 2, "B": 3, "C": 4}
    keyword_ids = [2, 3, 4]

    graph = build_keyword_context_graph(keyword_ids, symbol_table)

    assert graph.context_list == [keyword_ids]
    assert graph.root.token == -1
    assert 2 in graph.root.next
    assert 3 in graph.root.next[2].next
    assert graph.root.next[2].next[3].next[4].is_end


def test_decode_keyword_candidates_smoke_without_gpu():
    symbol_table = {"<blank>": 0, "A": 2, "B": 3}
    keyword_ids = [2, 3]
    graph = build_keyword_context_graph(keyword_ids, symbol_table)

    vocab_size = 5
    num_frames = 12
    log_probs = torch.full((1, num_frames, vocab_size), -20.0)
    log_probs[:, :, 0] = -1.0
    log_probs[0, 4, 2] = 0.0
    log_probs[0, 6, 3] = 0.0
    encoder_lens = torch.tensor([num_frames], dtype=torch.long)

    candidates = decode_keyword_candidates(
        log_probs,
        encoder_lens,
        graph,
        frame_shift_sec=0.04,
        symbol_table=symbol_table,
        beam_size=5,
    )

    assert isinstance(candidates, list)
    if candidates:
        candidate = candidates[0]
        assert candidate.start_sec >= 0.0
        assert candidate.end_sec > candidate.start_sec
        assert candidate.phonemes == ["A", "B"]
