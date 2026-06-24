"""Stage I keyword search with CTC prefix beam search and ContextGraph."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

import torch

from dma_kws.stage1.candidates import KeywordCandidate

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FRAME_SHIFT_SEC = 0.04


def _ensure_qbyt_on_path() -> None:
    qbyt_root = REPO_ROOT / "qbyt"
    qbyt_path = str(qbyt_root)
    if qbyt_path not in sys.path:
        sys.path.insert(0, qbyt_path)


def build_keyword_context_graph(
    keyword_token_ids: list[int],
    symbol_table: dict,
    *,
    context_score: float = 6.0,
):
    """Build a ``ContextGraph`` biased toward ``keyword_token_ids``."""
    _ensure_qbyt_on_path()
    from models.utils.context_graph import ContextGraph, ContextState

    known_ids = set(symbol_table.values())
    for token_id in keyword_token_ids:
        if token_id not in known_ids:
            raise ValueError(f"Keyword token id {token_id} is absent from symbol_table")

    graph = ContextGraph.__new__(ContextGraph)
    graph.context_score = context_score
    graph.context_list = [keyword_token_ids]
    graph.num_nodes = 0
    graph.root = ContextState(
        id=graph.num_nodes,
        token=-1,
        token_score=0,
        node_score=0,
        output_score=0,
        is_end=False,
    )
    graph.root.fail = graph.root
    graph.build_graph([keyword_token_ids])
    return graph


def _invert_symbol_table(symbol_table: dict) -> dict[int, str]:
    return {token_id: symbol for symbol, token_id in symbol_table.items()}


def _keyword_subsequence_span(
    tokens: Sequence[int],
    keyword_token_ids: Sequence[int],
) -> tuple[int, int] | None:
    if not keyword_token_ids:
        return None

    keyword_len = len(keyword_token_ids)
    for start in range(len(tokens) - keyword_len + 1):
        if list(tokens[start : start + keyword_len]) == list(keyword_token_ids):
            return start, start + keyword_len - 1
    return None


def _token_ids_to_phonemes(token_ids: Sequence[int], id_to_symbol: dict[int, str]) -> list[str]:
    phonemes: list[str] = []
    for token_id in token_ids:
        symbol = id_to_symbol.get(token_id)
        if symbol and symbol not in {"<blank>", "<unk>", "<sos/eos>"}:
            phonemes.append(symbol)
    return phonemes


def decode_keyword_candidates(
    log_probs: torch.Tensor,
    encoder_lens: torch.Tensor,
    context_graph,
    *,
    frame_shift_sec: float = DEFAULT_FRAME_SHIFT_SEC,
    beam_size: int = 10,
    margin_sec: float = 0.0,
    blank_id: int = 0,
    symbol_table: dict | None = None,
) -> list[KeywordCandidate]:
    """Run prefix beam search with ``context_graph`` and map hits to candidates."""
    _ensure_qbyt_on_path()
    from models.search import ctc_prefix_beam_search

    keyword_token_ids = context_graph.context_list[0]
    id_to_symbol = _invert_symbol_table(symbol_table) if symbol_table is not None else {}

    decode_results = ctc_prefix_beam_search(
        log_probs,
        encoder_lens,
        beam_size,
        context_graph,
        blank_id=blank_id,
    )

    candidates: list[KeywordCandidate] = []
    seen_spans: set[tuple[float, float]] = set()

    for result in decode_results:
        hypotheses = result.nbest or [result.tokens]
        hypothesis_times = result.nbest_times or [result.times or []]
        hypothesis_scores = result.nbest_scores or [result.score]

        for tokens, times, score in zip(hypotheses, hypothesis_times, hypothesis_scores):
            if not tokens or not times:
                continue

            span = _keyword_subsequence_span(tokens, keyword_token_ids)
            if span is None:
                continue

            start_index, end_index = span
            if end_index >= len(times):
                continue

            start_sec = max(0.0, times[start_index] * frame_shift_sec - margin_sec)
            end_sec = (times[end_index] + 1) * frame_shift_sec + margin_sec
            span_key = (round(start_sec, 6), round(end_sec, 6))
            if span_key in seen_spans:
                continue
            seen_spans.add(span_key)

            keyword_slice = tokens[start_index : end_index + 1]
            candidates.append(
                KeywordCandidate(
                    start_sec=round(start_sec, 6),
                    end_sec=round(end_sec, 6),
                    stage1_score=float(score),
                    phonemes=_token_ids_to_phonemes(keyword_slice, id_to_symbol),
                )
            )

    candidates.sort(key=lambda candidate: candidate.start_sec)
    return candidates
