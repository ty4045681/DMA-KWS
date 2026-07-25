"""Grapheme-to-phoneme helpers shared across DMA-KWS scripts.

Consolidates the near-identical ``make_g2p`` / ``text_to_phonemes`` /
``clean_phoneme_tokens`` copies in prepare_stage1_librispeech.py
and run_two_stage_demo.py.
"""

from __future__ import annotations

import re

from dma_kws.phonemes import normalize_english_text

_STRESS_DIGIT_RE = re.compile(r"[0-2]")


def make_g2p():
    """Construct a g2p_en.G2p converter, with a friendly error if missing."""
    try:
        from g2p_en import G2p
    except ImportError as exc:
        raise SystemExit("Missing dependency g2p_en. Install it with: pip install g2p_en") from exc
    return G2p()


def text_to_phonemes(g2p, text: str) -> list[str]:
    """Normalize ``text`` and convert it to a clean phoneme token list.

    ARPAbet stress digits are **kept** (``AH0`` stays ``AH0``): the phoneme
    vocabulary spells out every stress variant, and training, evaluation and
    inference must all tokenize text through this one function so the symbols
    they produce are identical.
    """
    return clean_phoneme_tokens(g2p(normalize_english_text(text)))


def clean_phoneme_tokens(tokens) -> list[str]:
    """Drop spaces and empty tokens from a raw G2P token sequence."""
    phonemes: list[str] = []
    for phone in tokens:
        cleaned = str(phone).strip()
        if cleaned:
            phonemes.append(cleaned)
    return phonemes


def has_stress_markers(g2p_text: str) -> bool:
    """Return True when a space-separated G2P string carries stress digits.

    Used to detect legacy stress-stripped ``ngram_g2p`` columns, which must be
    recomputed before they can be tokenized against the current vocabulary.
    """
    return bool(_STRESS_DIGIT_RE.search(str(g2p_text)))
