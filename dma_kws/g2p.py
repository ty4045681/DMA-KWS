"""Grapheme-to-phoneme helpers shared across DMA-KWS scripts.

Consolidates the near-identical ``make_g2p`` / ``text_to_phonemes`` /
``clean_phoneme_tokens`` copies in prepare_stage1_librispeech.py,
prepare_stage2_libriphrase.py, and run_two_stage_demo.py.
"""

from __future__ import annotations

from dma_kws.phonemes import normalize_english_text
from dma_kws.stage1.librispeech import strip_stress_marker


def make_g2p():
    """Construct a g2p_en.G2p converter, with a friendly error if missing."""
    try:
        from g2p_en import G2p
    except ImportError as exc:
        raise SystemExit("Missing dependency g2p_en. Install it with: pip install g2p_en") from exc
    return G2p()


def text_to_phonemes(g2p, text: str, *, strip_stress: bool = True) -> list[str]:
    """Normalize ``text`` and convert it to a clean phoneme token list.

    When ``strip_stress`` is True (the default, matching Stage I), ARPAbet
    stress digits are removed (e.g. ``AH0`` -> ``AH``).
    """
    normalized = normalize_english_text(text)
    return clean_phoneme_tokens(g2p(normalized), strip_stress=strip_stress)


def clean_phoneme_tokens(tokens, *, strip_stress: bool = True) -> list[str]:
    """Drop spaces/empties and optionally strip stress markers from tokens."""
    phonemes: list[str] = []
    for phone in tokens:
        if phone == " ":
            continue
        cleaned = str(phone)
        if strip_stress:
            cleaned = strip_stress_marker(cleaned)
        cleaned = cleaned.strip()
        if cleaned:
            phonemes.append(cleaned)
    return phonemes
