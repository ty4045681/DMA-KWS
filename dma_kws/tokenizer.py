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

#: The author's paper-original v1 release uses a 73-symbol dictionary: the
#: canonical inventory plus a reserved <sos/eos> id and a stress-less UW. Ids
#: shift after both insertions, so the two dictionaries are not interchangeable:
#: every checkpoint must be loaded with the one it was trained on.
VOCAB_PROFILE_CANONICAL = "canonical"
VOCAB_PROFILE_V1_73 = "v1_73"
VOCAB_PROFILES = (VOCAB_PROFILE_CANONICAL, VOCAB_PROFILE_V1_73)
V1_VOCAB_SIZE = 73
V1_DICT_PATH = Path(__file__).resolve().parents[1] / "data" / "dict" / "lang_char_v1_73.txt"
V1_EXTRA_SPECIAL_TOKENS = frozenset({"<sos/eos>"})
V1_EXTRA_PHONES = frozenset({"UW"})

#: profile -> (expected token count, required specials, required extra phones)
_PROFILE_REQUIREMENTS: dict[str, tuple[int, frozenset, frozenset]] = {
    VOCAB_PROFILE_CANONICAL: (
        CANONICAL_VOCAB_SIZE,
        _REQUIRED_SPECIAL_TOKENS,
        frozenset(),
    ),
    VOCAB_PROFILE_V1_73: (
        V1_VOCAB_SIZE,
        _REQUIRED_SPECIAL_TOKENS | V1_EXTRA_SPECIAL_TOKENS,
        V1_EXTRA_PHONES,
    ),
}
_VOCAB_SIZE_TO_PROFILE = {
    size: name for name, (size, _specials, _extras) in _PROFILE_REQUIREMENTS.items()
}


def resolve_vocab_profile(dict_path: Path, symbol_table: dict[str, int], profile: str | None) -> str:
    """Return the vocabulary profile, inferring it from the size when unset."""

    if profile is None:
        inferred = _VOCAB_SIZE_TO_PROFILE.get(len(symbol_table))
        if inferred is None:
            # Unknown size: fall back to canonical so the size / missing-token
            # checks below report the concrete defect instead of a bare profile
            # mismatch. Callers that need the v1 dictionary pass v1_73
            # explicitly (or use the size-73 file, which infers it).
            return VOCAB_PROFILE_CANONICAL
        return inferred
    normalized = str(profile).strip().lower()
    if normalized not in VOCAB_PROFILES:
        raise ValueError(
            f"Unsupported tokenizer vocab profile {profile!r}; expected one of "
            f"{', '.join(VOCAB_PROFILES)}"
        )
    return normalized


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


def validate_lang_char_dict(dict_path: Path, *, profile: str | None = None) -> None:
    """Validate a Wenet-format phoneme vocabulary against a known profile.

    Two profiles exist. "canonical" is the repo's 71-token dictionary
    (contiguous ids 0-70). "v1_73" is the author's paper-original dictionary
    (73 tokens, adding a reserved <sos/eos> and a stress-less UW). Passing
    profile=None infers the profile from the token count and then applies the
    same strict checks. Raises ValueError on any mismatch.
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

    resolved_profile = resolve_vocab_profile(dict_path, symbol_table, profile)
    expected_size, required_specials, required_extra_phones = _PROFILE_REQUIREMENTS[
        resolved_profile
    ]

    missing_special = required_specials - symbol_table.keys()
    if missing_special:
        missing = ", ".join(sorted(missing_special))
        raise ValueError(
            f"{dict_path}: profile {resolved_profile!r} is missing required "
            f"special tokens: {missing}"
        )

    required_phones = _REQUIRED_ARPABET_PHONES | required_extra_phones
    missing_phones = required_phones - symbol_table.keys()
    if missing_phones:
        missing = ", ".join(sorted(missing_phones))
        raise ValueError(
            f"{dict_path}: profile {resolved_profile!r} is missing required "
            f"ARPAbet phones: {missing}"
        )

    if len(symbol_table) != expected_size:
        raise ValueError(
            f"{dict_path}: profile {resolved_profile!r} expects {expected_size} "
            f"tokens, got {len(symbol_table)}"
        )

    ids = sorted(symbol_table.values())
    expected_ids = list(range(expected_size))
    if ids != expected_ids:
        raise ValueError(
            f"{dict_path}: token ids must be contiguous 0-{expected_size - 1}, "
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


def load_char_tokenizer(
    dict_path: Path,
    split_with_space: str = " ",
    *,
    profile: str | None = None,
):
    """Load Wenet CharTokenizer from dict_path under a validated vocab profile."""

    validate_lang_char_dict(dict_path, profile=profile)
    _ensure_qbyt_on_path()
    from models.text.char_tokenizer import CharTokenizer

    return CharTokenizer(str(dict_path), None, split_with_space=split_with_space)


def tokenize_phoneme_string(tokenizer, g2p_text: str) -> list[int]:
    """Tokenize a space-separated G2P phoneme string to integer ids."""
    _, token_ids = tokenizer.tokenize(g2p_text)
    return token_ids


SEQ_LABEL_ORDERED_CONTIGUOUS_PREFIX = "ordered_contiguous_prefix"
#: Paper-original v1 target: 1 when the anchor phone occurs anywhere in the
#: clip's phoneme sequence. It has no progress ordering and no keyword-
#: containment gate, matching the author's membership formula
#: (1 if x in query_seq else 0, for x in anchor_seq).
SEQ_LABEL_MEMBERSHIP = "membership"
DEFAULT_SEQ_LABEL_MODE = SEQ_LABEL_ORDERED_CONTIGUOUS_PREFIX
SEQ_LABEL_MODES = frozenset(
    {SEQ_LABEL_ORDERED_CONTIGUOUS_PREFIX, SEQ_LABEL_MEMBERSHIP}
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

    With mode="membership" (the paper-original v1 target) every position is
    independent: it is 1 exactly when that anchor phone occurs anywhere in the
    query. That is what the v1 checkpoint was trained on, so its training runs
    must use the membership mode rather than the ordered-prefix default.

    Ids remain stress-marked, so for example AH0 and AH1 are distinct.
    """
    mode = normalize_seq_label_mode(mode)
    if not anchor_ids:
        raise ValueError("anchor_ids must contain at least one phoneme")
    if mode == SEQ_LABEL_MEMBERSHIP:
        present = set(query_ids)
        return [1 if token in present else 0 for token in anchor_ids]
    matched = _longest_contiguous_anchor_prefix(anchor_ids, query_ids)
    return [1] * matched + [0] * (len(anchor_ids) - matched)
