"""Wenet CharTokenizer helpers aligned with main-branch QbyT training."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable

CANONICAL_DICT_PATH = Path(__file__).resolve().parents[1] / "data" / "dict" / "lang_char.txt"

_REQUIRED_SPECIAL_TOKENS = frozenset({"<blank>", "<unk>"})

# CMU ARPAbet vowels carry a stress digit in g2p_en output (AH0/AH1/AH2);
# consonants never do. The model is trained and evaluated on these exact
# stress-marked symbols, so the vocabulary must spell them all out.
ARPABET_STRESS_LEVELS = ("0", "1", "2")
ARPABET_VOWELS = (
    "AA",
    "AE",
    "AH",
    "AO",
    "AW",
    "AY",
    "EH",
    "ER",
    "EY",
    "IH",
    "IY",
    "OW",
    "OY",
    "UH",
    "UW",
)
ARPABET_CONSONANTS = (
    "B",
    "CH",
    "D",
    "DH",
    "F",
    "G",
    "HH",
    "JH",
    "K",
    "L",
    "M",
    "N",
    "NG",
    "P",
    "R",
    "S",
    "SH",
    "T",
    "TH",
    "V",
    "W",
    "Y",
    "Z",
    "ZH",
)
_REQUIRED_ARPABET_PHONES = frozenset(
    {f"{vowel}{stress}" for vowel in ARPABET_VOWELS for stress in ARPABET_STRESS_LEVELS}
    | set(ARPABET_CONSONANTS)
)

# 2 special tokens + 45 stress-marked vowels + 24 consonants, matching the
# 71-symbol phoneme inventory reported in the paper.
CANONICAL_VOCAB_SIZE = len(_REQUIRED_SPECIAL_TOKENS) + len(_REQUIRED_ARPABET_PHONES)


def unsupported_phones(phones: Iterable[str]) -> list[str]:
    """Return sorted phones that fall outside the canonical ARPAbet inventory.

    Data preparation calls this so an unexpected symbol fails loudly instead of
    being tokenized to ``<unk>`` and quietly poisoning training.
    """
    return sorted({str(phone) for phone in phones} - _REQUIRED_ARPABET_PHONES)


def _parse_dict_line(line: str, line_no: int, dict_path: Path) -> tuple[str, int]:
    stripped = line.strip()
    if not stripped:
        raise ValueError(f"{dict_path}:{line_no}: empty line")
    parts = stripped.rsplit(None, 1)
    if len(parts) != 2:
        raise ValueError(f"{dict_path}:{line_no}: expected '<token> <id>', got {stripped!r}")
    token, id_str = parts
    try:
        token_id = int(id_str)
    except ValueError as exc:
        raise ValueError(
            f"{dict_path}:{line_no}: token id must be an integer, got {id_str!r}"
        ) from exc
    return token, token_id


def validate_lang_char_dict(dict_path: Path) -> None:
    """Validate a Wenet-format ``lang_char.txt`` phoneme vocabulary.

    Raises ``ValueError`` when the file does not match the repo-canonical
    constraints: 71 tokens with contiguous ids 0-70, ``<blank>``/``<unk>``, and
    the full stress-marked ARPAbet phone inventory (AA0-ZH).
    """
    dict_path = Path(dict_path)
    if not dict_path.is_file():
        raise ValueError(f"lang_char dict not found: {dict_path}")

    symbol_table: dict[str, int] = {}
    for line_no, line in enumerate(dict_path.read_text(encoding="utf-8").splitlines(), start=1):
        token, token_id = _parse_dict_line(line, line_no, dict_path)
        if token in symbol_table:
            raise ValueError(f"{dict_path}:{line_no}: duplicate token {token!r}")
        symbol_table[token] = token_id

    missing_special = _REQUIRED_SPECIAL_TOKENS - symbol_table.keys()
    if missing_special:
        missing = ", ".join(sorted(missing_special))
        raise ValueError(f"{dict_path}: missing required special tokens: {missing}")

    missing_phones = _REQUIRED_ARPABET_PHONES - symbol_table.keys()
    if missing_phones:
        missing = ", ".join(sorted(missing_phones))
        raise ValueError(f"{dict_path}: missing required ARPAbet phones: {missing}")

    if len(symbol_table) != CANONICAL_VOCAB_SIZE:
        raise ValueError(
            f"{dict_path}: expected {CANONICAL_VOCAB_SIZE} tokens, got {len(symbol_table)}"
        )

    ids = sorted(symbol_table.values())
    expected_ids = list(range(CANONICAL_VOCAB_SIZE))
    if ids != expected_ids:
        raise ValueError(
            f"{dict_path}: token ids must be contiguous 0-{CANONICAL_VOCAB_SIZE - 1}, "
            f"got ids {ids}"
        )

    # Every CTC consumer in the repo hardcodes blank 0 (Stage I module, the
    # phoneme_ctc locator, collapse_ctc, the search helpers, and the adapter's
    # default). Make that assumption a checked precondition rather than a
    # coincidence of the shipped dict.
    if symbol_table["<blank>"] != 0:
        raise ValueError(
            f"{dict_path}: <blank> must have id 0, got {symbol_table['<blank>']}; "
            "CTC collapsing and keyword search assume blank is 0 throughout."
        )


def _ensure_qbyt_on_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    qbyt_root = repo_root / "qbyt"
    qbyt_path = str(qbyt_root)
    if qbyt_path not in sys.path:
        sys.path.insert(0, qbyt_path)


def load_char_tokenizer(dict_path: Path, split_with_space: str = " "):
    """Load Wenet ``CharTokenizer`` from ``dict_path``."""
    validate_lang_char_dict(dict_path)
    _ensure_qbyt_on_path()
    from models.text.char_tokenizer import CharTokenizer

    return CharTokenizer(str(dict_path), None, split_with_space=split_with_space)


def tokenize_phoneme_string(tokenizer, g2p_text: str) -> list[int]:
    """Tokenize a space-separated G2P phoneme string to integer ids."""
    _, token_ids = tokenizer.tokenize(g2p_text)
    return token_ids


SEQ_LABEL_MEMBERSHIP = "membership"
SEQ_LABEL_ORDERED_CONTIGUOUS_PREFIX = "ordered_contiguous_prefix"
DEFAULT_SEQ_LABEL_MODE = SEQ_LABEL_ORDERED_CONTIGUOUS_PREFIX
SEQ_LABEL_MODES = frozenset(
    {
        SEQ_LABEL_MEMBERSHIP,
        SEQ_LABEL_ORDERED_CONTIGUOUS_PREFIX,
    }
)


def normalize_seq_label_mode(mode: str) -> str:
    """Validate and normalize a Stage II sequence-target mode."""
    normalized = str(mode).strip().lower()
    if normalized not in SEQ_LABEL_MODES:
        choices = ", ".join(sorted(SEQ_LABEL_MODES))
        raise ValueError(
            f"Unsupported seq label mode {mode!r}; expected one of: {choices}"
        )
    return normalized


def _longest_contiguous_anchor_prefix(
    anchor_ids: list[int],
    query_ids: list[int],
) -> int:
    """Length of the longest anchor prefix occurring contiguously in the query."""
    best = 0
    for query_start in range(len(query_ids)):
        matched = 0
        while (
            matched < len(anchor_ids)
            and query_start + matched < len(query_ids)
            and anchor_ids[matched] == query_ids[query_start + matched]
        ):
            matched += 1
        best = max(best, matched)
        if best == len(anchor_ids):
            break
    return best


def build_seq_label(
    anchor_ids: list[int],
    query_ids: list[int],
    *,
    mode: str = DEFAULT_SEQ_LABEL_MODE,
) -> list[int]:
    """Build one binary sequence target per anchor phoneme.

    The default target describes ordered, contiguous progress: if the longest
    prefix of ``anchor_ids`` found anywhere in ``query_ids`` has length ``r``,
    the result is ``r`` ones followed by zeros. Query-side leading/trailing
    context is allowed, but insertions, reordering and reusing one occurrence of
    a repeated phoneme cannot advance the target.

    ``membership`` retains the released DMA/PhonMatchNet heuristic for explicit
    legacy ablations. Ids remain stress-marked, so e.g. ``AH0`` and ``AH1`` are
    distinct in both modes.
    """
    mode = normalize_seq_label_mode(mode)
    if not anchor_ids:
        raise ValueError("anchor_ids must contain at least one phoneme")
    if mode == SEQ_LABEL_MEMBERSHIP:
        return [1 if anchor_id in query_ids else 0 for anchor_id in anchor_ids]

    matched = _longest_contiguous_anchor_prefix(anchor_ids, query_ids)
    return [1] * matched + [0] * (len(anchor_ids) - matched)
