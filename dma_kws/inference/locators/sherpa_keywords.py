"""Helpers for formatting keywords for sherpa-onnx KeywordSpotter."""

from __future__ import annotations

from pathlib import Path


def _load_token_vocab(tokens_path: str) -> set[str]:
    vocab: set[str] = set()
    with Path(tokens_path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            token = line.split(maxsplit=1)[0]
            vocab.add(token)
    return vocab


def _english_char_keyword_line(keyword: str, tokens_path: str) -> str:
    """Space-separate per-character tokens that exist in ``tokens.txt``."""
    vocab = _load_token_vocab(tokens_path)
    display = keyword.strip()
    chars = [ch for ch in display if not ch.isspace()]
    if not chars:
        raise ValueError("keyword must contain at least one non-space character")

    tokens: list[str] = []
    for char in chars:
        if char in vocab:
            tokens.append(char)
        elif char.lower() in vocab:
            tokens.append(char.lower())
        elif char.upper() in vocab:
            tokens.append(char.upper())
        else:
            tokens.append(char)

    return f"{' '.join(tokens)} @{display}"


def format_sherpa_keyword(
    keyword: str,
    *,
    modeling_unit: str = "cjkchar",
    tokens_path: str | None = None,
) -> str:
    """Format a keyword line for ``KeywordSpotter.create_stream``.

    Pass-through lines that already contain ``@`` (e.g. pinyin ``x iǎo ài @小爱``).
    For simple English char models with ``tokens.txt``, emit space-separated
    character tokens. Otherwise fall back to ``keyword@keyword``.
    """
    keyword = keyword.strip()
    if not keyword:
        raise ValueError("keyword must be non-empty")

    if "@" in keyword:
        return keyword

    unit = modeling_unit.lower()
    if unit in {"en", "english", "char", "bpe"} and tokens_path:
        return _english_char_keyword_line(keyword, tokens_path)

    return f"{keyword}@{keyword}"
