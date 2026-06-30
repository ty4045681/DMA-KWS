"""Phoneme-level hard-negative distance helpers for Stage II parquet prep."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from dma_kws.g2p import clean_phoneme_tokens

_PHONEME_CHAR_BASE = 0xE000
_EXCLUDE_DIST = int(np.iinfo(np.uint8).max)


def phoneme_tokens_from_g2p(g2p_text: str, *, strip_stress: bool = True) -> list[str]:
    """Parse a space-separated G2P string into phoneme tokens."""
    return clean_phoneme_tokens(str(g2p_text).split(), strip_stress=strip_stress)


def encode_phoneme_strings(
    phones_list: list[list[str]],
) -> tuple[list[str], dict[str, str]]:
    """Map phoneme token lists to single-character strings for fast Levenshtein.

    Each distinct phoneme is assigned a unique character from the Unicode
    private-use area so ``rapidfuzz`` can operate on compact strings.
    """
    vocab: dict[str, str] = {}
    next_ord = _PHONEME_CHAR_BASE

    def char_for(phone: str) -> str:
        nonlocal next_ord
        if phone not in vocab:
            if next_ord > 0xF8FF:
                raise ValueError("Too many distinct phonemes for private-use encoding")
            vocab[phone] = chr(next_ord)
            next_ord += 1
        return vocab[phone]

    encoded = ["".join(char_for(phone) for phone in phones) for phones in phones_list]
    reverse = {phone: char for phone, char in vocab.items()}
    return encoded, reverse


def _neighbors_for_row(
    row_dist: np.ndarray,
    global_idx: int,
    encoded: list[str],
    ngrams: list[str],
    top_k: int,
) -> list[dict[str, Any]]:
    dists = np.asarray(row_dist, dtype=np.int64).copy()
    dists[global_idx] = _EXCLUDE_DIST

    phone_lens = [len(text) for text in encoded]
    len_a = phone_lens[global_idx]

    valid = np.flatnonzero((dists > 0) & (np.arange(len(dists)) != global_idx))
    if valid.size == 0:
        return []

    valid_dists = dists[valid]
    k = min(top_k, valid.size)
    if k < valid.size:
        part = np.argpartition(valid_dists, k - 1)[:k]
        top_idx = valid[part]
    else:
        top_idx = valid

    pairs = sorted(
        ((int(dists[idx]), idx) for idx in top_idx),
        key=lambda item: (item[0], ngrams[item[1]]),
    )

    results: list[dict[str, Any]] = []
    for dist, idx in pairs[:top_k]:
        denom = max(len_a, phone_lens[idx])
        sim = round(1.0 - dist / denom, 3) if denom > 0 else 0.0
        results.append({"distance": sim, "ngram": ngrams[idx]})

    results.sort(key=lambda item: (-item["distance"], item["ngram"]))
    return results


def top_k_hard_negatives(
    phones: list[list[str]],
    ngrams: list[str],
    *,
    top_k: int = 100,
    block_size: int = 1000,
    workers: int = -1,
    progress_callback: Callable[[int, int], None] | None = None,
) -> list[list[dict[str, Any]]]:
    """Return top-K phonetically closest anchors for each query anchor."""
    from rapidfuzz import process
    from rapidfuzz.distance import Levenshtein

    if len(phones) != len(ngrams):
        raise ValueError("phones and ngrams must have the same length")

    encoded, _ = encode_phoneme_strings(phones)
    total = len(encoded)
    if total == 0:
        return []

    all_results: list[list[dict[str, Any]]] = []
    for start in range(0, total, block_size):
        end = min(start + block_size, total)
        block = encoded[start:end]
        matrix = process.cdist(
            block,
            encoded,
            scorer=Levenshtein.distance,
            workers=workers,
            dtype=np.uint8,
        )
        for local_i, global_i in enumerate(range(start, end)):
            all_results.append(
                _neighbors_for_row(matrix[local_i], global_i, encoded, ngrams, top_k)
            )
        if progress_callback is not None:
            progress_callback(end, total)

    return all_results


def build_distances_column(
    df,
    *,
    top_k: int = 100,
    strip_stress: bool = True,
    block_size: int = 1000,
    workers: int = -1,
    progress_callback: Callable[[int, int], None] | None = None,
):
    """Attach a ``distances`` column to an aggregated G2P parquet dataframe."""
    phones = [
        phoneme_tokens_from_g2p(row["ngram_g2p"], strip_stress=strip_stress)
        for _, row in df.iterrows()
    ]
    ngrams = df["ngram"].astype(str).tolist()
    distances = top_k_hard_negatives(
        phones,
        ngrams,
        top_k=top_k,
        block_size=block_size,
        workers=workers,
        progress_callback=progress_callback,
    )
    out = df.copy()
    out["distances"] = distances
    return out
