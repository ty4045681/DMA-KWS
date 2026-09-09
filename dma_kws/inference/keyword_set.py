"""Keyword-set enrollment, IDs, and max-aggregation for Stage-II any-hit eval.

``prep.keyword_eval.mode=any`` enrolls every configured keyword and
pronunciation once on the main process. Scoring still uses the existing
verifier, calibrator, and tokenizer; this module only parses the lexicon,
assigns deterministic IDs, and aggregates per-query scores.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from dma_kws.g2p import make_g2p, text_to_phonemes
from dma_kws.tokenizer import tokenize_phoneme_string, unsupported_phones


KEYWORD_SET_SCHEMA_VERSION = 1
TEXT_NORMALIZATION_VERSION = 1
KEYWORD_EVAL_MODES = frozenset({"per_row", "any"})
DEFAULT_KEYWORD_EVAL_MODE = "per_row"
DEFAULT_QUERY_BATCH_SIZE = 64
SCORE_SEMANTICS = "max_over_keywords_and_pronunciations"
CLIP_EVAL_PROTOCOL = "stage2_clip_keyword_set"
WINDOW_EVAL_PROTOCOL = "stage2_window_keyword_set"
ANY_RESULT_SCHEMA_VERSION = 2
SKIP_REASON_TOO_SHORT = "too_short"

_WHITESPACE_RE = re.compile(r"\s+")
_ALLOWED_TARGET_KEYS = frozenset({"text", "pronunciations"})
_UNSUPPORTED_ANY_MESSAGE = (
    "does not support prep.keyword_eval.mode=any; this entry still uses the "
    "legacy per-row keyword protocol"
)

ANY_MANIFEST_FORMAT_EXAMPLES = """\
Any-mode manifests use one source-audio row and one of:

  {"audio_path":"audio/eva.wav","keyword_labels":{"hey eva":1,"ok lamp":0}}

  {"audio_path":"audio/neg.wav","label_scope":"target_set",\
"target_texts":["hey eva","ok lamp"],"label":0}

Unlabeled prediction rows may omit every label field. Top-level \
`keyword` / `keyword_phonemes` pair fields are rejected."""


class KeywordEvalConfigError(ValueError):
    """Invalid ``prep.keyword_eval`` configuration or enrollment."""


class KeywordSetManifestError(ValueError):
    """Invalid any-mode manifest row or label contract."""


class KeywordSetScoreError(ValueError):
    """Non-finite or incomplete any-mode scores."""


@dataclass(frozen=True)
class ResolvedPronunciation:
    pronunciation_id: str
    phonemes: tuple[str, ...]
    token_ids: tuple[int, ...]
    source: str
    query_index: int


@dataclass(frozen=True)
class ResolvedKeyword:
    keyword_id: str
    text: str
    display_text: str
    pronunciations: tuple[ResolvedPronunciation, ...]


@dataclass(frozen=True)
class ResolvedQuery:
    query_index: int
    token_ids: tuple[int, ...]
    phonemes: tuple[str, ...]


@dataclass(frozen=True)
class ResolvedKeywordSet:
    keyword_set_id: str
    keywords: tuple[ResolvedKeyword, ...]
    queries: tuple[ResolvedQuery, ...]
    query_batch_size: int
    tokenizer_dict_sha256: str
    merge_diagnostics: tuple[str, ...]

    @property
    def texts(self) -> tuple[str, ...]:
        return tuple(keyword.text for keyword in self.keywords)

    @property
    def num_queries(self) -> int:
        return len(self.queries)

    def query_token_ids(self) -> list[list[int]]:
        return [list(query.token_ids) for query in self.queries]


@dataclass(frozen=True)
class KeywordSetAggregation:
    qbyt_raw_logit: float | None
    qbyt_score: float
    detected: bool
    best_keyword: str | None
    best_pronunciation_id: str | None
    matched_keywords: tuple[str, ...]
    keyword_results: tuple[dict[str, Any], ...]


def canonical_json_sha256(payload: object) -> str:
    """SHA-256 of canonical JSON (sorted keys, no NaN, compact separators)."""

    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).expanduser().open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_keyword_text(value: object, *, field: str = "text") -> str:
    """Trim, collapse whitespace, and casefold. Keep punctuation and word spaces."""

    if not isinstance(value, str):
        raise KeywordEvalConfigError(f"{field} must be a non-empty string, got {value!r}")
    normalized = _WHITESPACE_RE.sub(" ", value.strip()).casefold()
    if not normalized:
        raise KeywordEvalConfigError(f"{field} must be a non-empty string, got {value!r}")
    return normalized


def require_positive_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise KeywordEvalConfigError(
            f"{field} must be a positive int (bool rejected), got {value!r}"
        )
    return value


def require_qbyt_threshold(value: object, *, field: str = "demo.qbyt_threshold") -> float:
    if isinstance(value, bool):
        raise KeywordEvalConfigError(
            f"{field} must be a finite number in [0, 1], not bool"
        )
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise KeywordEvalConfigError(
            f"{field} must be a finite number in [0, 1], got {value!r}"
        ) from exc
    if not math.isfinite(number) or number < 0.0 or number > 1.0:
        raise KeywordEvalConfigError(
            f"{field} must be a finite number in [0, 1], got {value!r}"
        )
    return number


def _plain_container(value: object) -> object:
    try:
        from omegaconf import OmegaConf

        if OmegaConf.is_config(value):
            return OmegaConf.to_container(value, resolve=True)
    except ImportError:
        pass
    return value


def _keyword_eval_section(prep: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(prep, Mapping):
        return {}
    section = _plain_container(prep.get("keyword_eval", {}))
    if section is None:
        return {}
    if not isinstance(section, dict):
        raise KeywordEvalConfigError(
            "prep.keyword_eval must be a mapping, got "
            f"{type(section).__name__}"
        )
    return section


def keyword_eval_mode(prep: Mapping[str, Any] | None) -> str:
    section = _keyword_eval_section(prep)
    raw = section.get("mode", DEFAULT_KEYWORD_EVAL_MODE)
    if raw is None or raw == "":
        return DEFAULT_KEYWORD_EVAL_MODE
    if not isinstance(raw, str):
        raise KeywordEvalConfigError(
            "prep.keyword_eval.mode must be 'per_row' or 'any', "
            f"got {raw!r}"
        )
    mode = raw.strip()
    if mode not in KEYWORD_EVAL_MODES:
        raise KeywordEvalConfigError(
            "prep.keyword_eval.mode must be 'per_row' or 'any', "
            f"got {raw!r}"
        )
    return mode


def reject_unsupported_any_mode(prep: Mapping[str, Any] | None, *, entry: str) -> None:
    """Raise when a legacy pair-protocol entry is asked for keyword-set eval."""

    if keyword_eval_mode(prep) == "any":
        raise KeywordEvalConfigError(f"{entry} {_UNSUPPORTED_ANY_MESSAGE}")


def _legacy_field_is_set(value: object) -> bool:
    value = _plain_container(value)
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return any(str(item).strip() for item in value)
    return bool(value)


def _assert_any_mode_legacy_fields_empty(prep: Mapping[str, Any]) -> None:
    conflicts = []
    if _legacy_field_is_set(prep.get("keyword")):
        conflicts.append("prep.keyword")
    if _legacy_field_is_set(prep.get("keyword_phonemes")):
        conflicts.append("prep.keyword_phonemes")
    if _legacy_field_is_set(prep.get("keywords")):
        conflicts.append("prep.keywords")
    if conflicts:
        raise KeywordEvalConfigError(
            "prep.keyword_eval.mode=any cannot be combined with non-empty "
            f"{', '.join(conflicts)}"
        )


def _target_error(index: int, text: object, message: str, *, pronunciation_index: int | None = None) -> None:
    location = f"prep.keyword_eval.targets[{index}]"
    if text is not None and text != "":
        location = f"{location} text={text!r}"
    if pronunciation_index is not None:
        location = f"{location} pronunciations[{pronunciation_index}]"
    raise KeywordEvalConfigError(f"{location}: {message}")


def _parse_explicit_pronunciations(
    raw: object,
    *,
    target_index: int,
    text: str,
) -> tuple[tuple[str, ...], ...]:
    raw = _plain_container(raw)
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        _target_error(
            target_index,
            text,
            "pronunciations must be a non-empty list of ARPAbet strings",
        )
    if len(raw) == 0:
        _target_error(
            target_index,
            text,
            "pronunciations must be a non-empty list; omit the field or set "
            "null for one G2P sequence",
        )
    parsed: list[tuple[str, ...]] = []
    from dma_kws.inference.stage2_clip import parse_phoneme_sequence

    for pronunciation_index, item in enumerate(raw):
        if not isinstance(item, str) or not item.strip():
            _target_error(
                target_index,
                text,
                "each pronunciation must be a non-empty ARPAbet string "
                f"(got {item!r})",
                pronunciation_index=pronunciation_index,
            )
        try:
            phonemes = parse_phoneme_sequence(
                item,
                field_name=(
                    f"prep.keyword_eval.targets[{target_index}] "
                    f"text={text!r} pronunciations[{pronunciation_index}]"
                ),
            )
        except ValueError as exc:
            raise KeywordEvalConfigError(str(exc)) from exc
        parsed.append(tuple(phonemes))
    return tuple(parsed)


def _g2p_pronunciation(g2p, text: str, *, target_index: int) -> tuple[str, ...]:
    phonemes = tuple(text_to_phonemes(g2p, text))
    if not phonemes:
        _target_error(target_index, text, "G2P produced an empty pronunciation")
    illegal = unsupported_phones(phonemes)
    if illegal:
        _target_error(
            target_index,
            text,
            "G2P produced unsupported phonemes: " + ", ".join(illegal),
        )
    return phonemes


def _enroll_tokens(tokenizer, phonemes: Sequence[str], *, field: str) -> tuple[int, ...]:
    token_ids = tuple(tokenize_phoneme_string(tokenizer, " ".join(phonemes)))
    if len(token_ids) != len(phonemes):
        raise KeywordEvalConfigError(
            f"{field}: phoneme/token length mismatch "
            f"phonemes={len(phonemes)}, token_ids={len(token_ids)}"
        )
    symbol_table = getattr(tokenizer, "symbol_table", None)
    if isinstance(symbol_table, Mapping):
        unk_id = symbol_table.get("<unk>")
        if unk_id is not None and unk_id in token_ids:
            raise KeywordEvalConfigError(
                f"{field}: tokenization produced <unk>; check the ARPAbet sequence"
            )
    return token_ids


def _keyword_id(text: str) -> str:
    return canonical_json_sha256(
        {
            "kind": "keyword",
            "text_normalization_version": TEXT_NORMALIZATION_VERSION,
            "text": text,
        }
    )


def _pronunciation_id(token_ids: Sequence[int]) -> str:
    return canonical_json_sha256(
        {
            "kind": "pronunciation",
            "token_ids": [int(token) for token in token_ids],
        }
    )


def keyword_set_identity_payload(
    *,
    texts_to_token_tuples: Mapping[str, Sequence[Sequence[int]]],
    tokenizer_dict_sha256: str,
) -> dict[str, Any]:
    lexicon = {
        text: sorted(
            ([int(token) for token in tokens] for tokens in token_tuples),
            key=lambda item: tuple(item),
        )
        for text, token_tuples in texts_to_token_tuples.items()
    }
    return {
        "keyword_set_schema_version": KEYWORD_SET_SCHEMA_VERSION,
        "text_normalization_version": TEXT_NORMALIZATION_VERSION,
        "lexicon": dict(sorted(lexicon.items())),
        "tokenizer_dict_sha256": str(tokenizer_dict_sha256),
    }


def keyword_eval_provenance_block(
    keyword_set: ResolvedKeywordSet | None,
    *,
    mode: str,
) -> dict[str, Any]:
    if mode == "per_row" or keyword_set is None:
        return {"mode": "per_row"}
    lexicon = {
        keyword.text: sorted(
            (list(pron.token_ids) for pron in keyword.pronunciations),
            key=lambda item: tuple(item),
        )
        for keyword in keyword_set.keywords
    }
    return {
        "mode": "any",
        "aggregation": SCORE_SEMANTICS,
        "keyword_set_id": keyword_set.keyword_set_id,
        "texts": list(keyword_set.texts),
        "token_sequences": dict(sorted(lexicon.items())),
    }


def semantic_keyword_eval(keyword_eval: object | None, *, schema_version: int) -> dict[str, Any]:
    """Normalize keyword_eval for provenance comparison.

    Schema 3 has no field and compares equal to schema-4 ``per_row``. Schema 4
    ``any`` keeps mode, aggregation, keyword_set_id, texts, and token sequences.
    """

    if schema_version < 4 or keyword_eval is None:
        return {"mode": "per_row"}
    if not isinstance(keyword_eval, Mapping):
        raise KeywordEvalConfigError("keyword_eval must be a mapping")
    mode = str(keyword_eval.get("mode", "per_row"))
    if mode == "per_row":
        return {"mode": "per_row"}
    if mode != "any":
        raise KeywordEvalConfigError(f"unsupported keyword_eval.mode {mode!r}")
    return {
        "mode": "any",
        "aggregation": str(keyword_eval.get("aggregation", SCORE_SEMANTICS)),
        "keyword_set_id": str(keyword_eval.get("keyword_set_id", "")),
        "texts": list(keyword_eval.get("texts") or []),
        "token_sequences": dict(keyword_eval.get("token_sequences") or {}),
    }


def resolve_keyword_set(
    prep: Mapping[str, Any],
    tokenizer,
    *,
    g2p=None,
    tokenizer_dict_path: str | Path | None = None,
    tokenizer_dict_sha256: str | None = None,
) -> ResolvedKeywordSet | None:
    """Parse and enroll ``prep.keyword_eval``.

    Returns ``None`` for ``mode=per_row``. ``mode=any`` always returns a
    non-empty enrolled set or raises :class:`KeywordEvalConfigError`.
    """

    mode = keyword_eval_mode(prep)
    section = _keyword_eval_section(prep)
    raw_targets = _plain_container(section.get("targets", []))
    query_batch_size = require_positive_int(
        section.get("query_batch_size", DEFAULT_QUERY_BATCH_SIZE),
        field="prep.keyword_eval.query_batch_size",
    )

    if mode == "per_row":
        if raw_targets:
            raise KeywordEvalConfigError(
                "prep.keyword_eval.targets must be empty when mode=per_row; "
                "set mode=any to enroll a keyword set"
            )
        return None

    _assert_any_mode_legacy_fields_empty(prep)
    if not isinstance(raw_targets, Sequence) or isinstance(raw_targets, (str, bytes)):
        raise KeywordEvalConfigError(
            "prep.keyword_eval.targets must be a non-empty list of objects"
        )
    if len(raw_targets) == 0:
        raise KeywordEvalConfigError(
            "prep.keyword_eval.mode=any requires a non-empty targets list"
        )

    if tokenizer_dict_sha256 is None:
        if tokenizer_dict_path is None:
            raise KeywordEvalConfigError(
                "tokenizer_dict_path or tokenizer_dict_sha256 is required to "
                "enroll a keyword set"
            )
        tokenizer_dict_sha256 = file_sha256(tokenizer_dict_path)

    grouped: dict[str, dict[str, Any]] = {}
    merge_diagnostics: list[str] = []
    for target_index, raw_target in enumerate(raw_targets):
        target = _plain_container(raw_target)
        if not isinstance(target, Mapping):
            _target_error(target_index, None, "each target must be a mapping with text")
        if "phonemes" in target:
            _target_error(
                target_index,
                target.get("text"),
                "unpublished key 'phonemes' is not supported; use 'pronunciations'",
            )
        unknown = sorted(set(target) - _ALLOWED_TARGET_KEYS)
        if unknown:
            _target_error(
                target_index,
                target.get("text"),
                f"unsupported keys {unknown}; allowed keys are text, pronunciations",
            )
        raw_text = target.get("text")
        try:
            normalized = normalize_keyword_text(
                raw_text,
                field=f"prep.keyword_eval.targets[{target_index}].text",
            )
        except KeywordEvalConfigError:
            raise
        display_text = _WHITESPACE_RE.sub(" ", str(raw_text).strip())
        pronunciations_field = target.get("pronunciations", None)
        if pronunciations_field is None:
            if g2p is None:
                g2p = make_g2p()
            phoneme_seqs = [_g2p_pronunciation(g2p, normalized, target_index=target_index)]
            sources = ["g2p"]
        else:
            phoneme_seqs = list(
                _parse_explicit_pronunciations(
                    pronunciations_field,
                    target_index=target_index,
                    text=normalized,
                )
            )
            sources = ["explicit"] * len(phoneme_seqs)

        bucket = grouped.setdefault(
            normalized,
            {"display_text": display_text, "pairs": [], "entry_count": 0},
        )
        bucket["entry_count"] += 1
        for phonemes, source in zip(phoneme_seqs, sources):
            bucket["pairs"].append((tuple(phonemes), source, target_index))

    for text, bucket in sorted(grouped.items()):
        if bucket["entry_count"] > 1:
            merge_diagnostics.append(
                f"merged {bucket['entry_count']} target entries for text {text!r}"
            )

    keywords: list[ResolvedKeyword] = []
    query_index_by_tokens: dict[tuple[int, ...], int] = {}
    queries: list[ResolvedQuery] = []
    texts_to_token_tuples: dict[str, list[tuple[int, ...]]] = {}

    for text in sorted(grouped):
        bucket = grouped[text]
        unique_by_tokens: dict[tuple[int, ...], tuple[tuple[str, ...], str]] = {}
        for phonemes, source, target_index in bucket["pairs"]:
            field = f"keyword {text!r} pronunciation {' '.join(phonemes)}"
            token_ids = _enroll_tokens(tokenizer, phonemes, field=field)
            previous = unique_by_tokens.get(token_ids)
            if previous is None:
                unique_by_tokens[token_ids] = (phonemes, source)
            elif previous[1] != source:
                unique_by_tokens[token_ids] = (previous[0], f"{previous[1]}+{source}")
            else:
                merge_diagnostics.append(
                    f"dropped duplicate token sequence for {text!r}: "
                    f"{' '.join(phonemes)}"
                )
        pronunciations: list[ResolvedPronunciation] = []
        token_tuples: list[tuple[int, ...]] = []
        for token_ids in sorted(unique_by_tokens):
            phonemes, source = unique_by_tokens[token_ids]
            if token_ids not in query_index_by_tokens:
                query_index_by_tokens[token_ids] = len(queries)
                queries.append(
                    ResolvedQuery(
                        query_index=len(queries),
                        token_ids=token_ids,
                        phonemes=phonemes,
                    )
                )
            pronunciations.append(
                ResolvedPronunciation(
                    pronunciation_id=_pronunciation_id(token_ids),
                    phonemes=phonemes,
                    token_ids=token_ids,
                    source=source.split("+", 1)[0],
                    query_index=query_index_by_tokens[token_ids],
                )
            )
            token_tuples.append(token_ids)
        texts_to_token_tuples[text] = token_tuples
        keywords.append(
            ResolvedKeyword(
                keyword_id=_keyword_id(text),
                text=text,
                display_text=bucket["display_text"],
                pronunciations=tuple(pronunciations),
            )
        )

    if not queries:
        raise KeywordEvalConfigError(
            "prep.keyword_eval.mode=any produced no valid pronunciations"
        )

    identity = keyword_set_identity_payload(
        texts_to_token_tuples=texts_to_token_tuples,
        tokenizer_dict_sha256=tokenizer_dict_sha256,
    )
    return ResolvedKeywordSet(
        keyword_set_id=canonical_json_sha256(identity),
        keywords=tuple(keywords),
        queries=tuple(queries),
        query_batch_size=query_batch_size,
        tokenizer_dict_sha256=tokenizer_dict_sha256,
        merge_diagnostics=tuple(merge_diagnostics),
    )


def _finite_or_raise(value: object, *, field: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise KeywordSetScoreError(f"{field} is not a number: {value!r}") from exc
    if not math.isfinite(number):
        raise KeywordSetScoreError(f"{field} is not finite: {value!r}")
    return number


def aggregate_query_scores(
    keyword_set: ResolvedKeywordSet,
    query_scores: Sequence[tuple[float | None, float] | Sequence[float]],
    *,
    threshold: float,
    audio_id: object = None,
    scored: bool = True,
) -> KeywordSetAggregation:
    """Calibrated-per-query max aggregation.

    ``query_scores[i]`` is ``(raw_logit, calibrated_score)`` for
    ``keyword_set.queries[i]``. Ties pick the winner by normalized text, then
    token-id lexicographic order. ``detected`` is false when ``scored`` is false.
    """

    threshold = require_qbyt_threshold(threshold, field="threshold")
    if len(query_scores) != keyword_set.num_queries:
        raise KeywordSetScoreError(
            "scorer returned a different number of query scores than enrolled "
            f"queries: expected={keyword_set.num_queries}, actual={len(query_scores)}"
            + (f" audio={audio_id!r}" if audio_id is not None else "")
        )

    parsed: list[tuple[float | None, float]] = []
    for query, raw_pair in zip(keyword_set.queries, query_scores):
        if not isinstance(raw_pair, Sequence) or len(raw_pair) < 2:
            raise KeywordSetScoreError(
                f"query {query.query_index} score pair is incomplete"
                + (f" audio={audio_id!r}" if audio_id is not None else "")
            )
        raw_logit, calibrated = raw_pair[0], raw_pair[1]
        location = (
            f"audio={audio_id!r} query_index={query.query_index} "
            f"token_ids={list(query.token_ids)}"
            if audio_id is not None
            else f"query_index={query.query_index}"
        )
        if raw_logit is not None:
            raw_logit = _finite_or_raise(raw_logit, field=f"{location} qbyt_raw_logit")
        calibrated = _finite_or_raise(calibrated, field=f"{location} qbyt_score")
        parsed.append((raw_logit, calibrated))

    keyword_results: list[dict[str, Any]] = []
    matched: list[str] = []
    best_key: tuple[float, str, tuple[int, ...]] | None = None
    best_keyword: str | None = None
    best_pronunciation_id: str | None = None
    best_raw: float | None = None
    best_score = 0.0

    for keyword in keyword_set.keywords:
        pronunciation_results: list[dict[str, Any]] = []
        keyword_best: tuple[float, tuple[int, ...], ResolvedPronunciation, float | None] | None = None
        for pronunciation in keyword.pronunciations:
            raw_logit, calibrated = parsed[pronunciation.query_index]
            detected = bool(scored) and calibrated >= threshold
            pronunciation_results.append(
                {
                    "pronunciation_id": pronunciation.pronunciation_id,
                    "phonemes": list(pronunciation.phonemes),
                    "token_ids": list(pronunciation.token_ids),
                    "qbyt_raw_logit": raw_logit,
                    "qbyt_score": calibrated,
                    "detected": detected,
                }
            )
            candidate = (calibrated, pronunciation.token_ids, pronunciation, raw_logit)
            if keyword_best is None or candidate[0] > keyword_best[0] or (
                candidate[0] == keyword_best[0] and candidate[1] < keyword_best[1]
            ):
                keyword_best = candidate
        assert keyword_best is not None
        keyword_score, _tokens, winning_pron, keyword_raw = keyword_best
        keyword_detected = bool(scored) and keyword_score >= threshold
        if keyword_detected:
            matched.append(keyword.text)
        keyword_results.append(
            {
                "keyword_id": keyword.keyword_id,
                "text": keyword.text,
                "qbyt_raw_logit": keyword_raw,
                "qbyt_score": keyword_score,
                "detected": keyword_detected,
                "best_pronunciation_id": winning_pron.pronunciation_id,
                "pronunciation_results": pronunciation_results,
            }
        )
        rank = (keyword_score, keyword.text, winning_pron.token_ids)
        if best_key is None or rank[0] > best_key[0] or (
            rank[0] == best_key[0]
            and (rank[1], rank[2]) < (best_key[1], best_key[2])
        ):
            best_key = rank
            best_keyword = keyword.text
            best_pronunciation_id = winning_pron.pronunciation_id
            best_raw = keyword_raw
            best_score = keyword_score

    detected = bool(scored) and best_score >= threshold and best_keyword is not None
    return KeywordSetAggregation(
        qbyt_raw_logit=best_raw if scored else None,
        qbyt_score=best_score if scored else 0.0,
        detected=detected,
        best_keyword=best_keyword if scored else None,
        best_pronunciation_id=best_pronunciation_id if scored else None,
        matched_keywords=tuple(matched),
        keyword_results=tuple(keyword_results),
    )


def skipped_keyword_set_fields(
    keyword_set: ResolvedKeywordSet,
    *,
    threshold: float,
) -> dict[str, Any]:
    threshold = require_qbyt_threshold(threshold, field="threshold")
    return {
        "result_schema_version": ANY_RESULT_SCHEMA_VERSION,
        "keyword_eval_mode": "any",
        "keyword_set_id": keyword_set.keyword_set_id,
        "qbyt_raw_logit": None,
        "qbyt_score": 0.0,
        "threshold": threshold,
        "detected": False,
        "skipped": True,
        "skip_reason": SKIP_REASON_TOO_SHORT,
        "best_keyword": None,
        "best_pronunciation_id": None,
        "matched_keywords": [],
        "keyword_results": [],
    }


def scored_keyword_set_fields(
    keyword_set: ResolvedKeywordSet,
    aggregation: KeywordSetAggregation,
    *,
    threshold: float,
) -> dict[str, Any]:
    threshold = require_qbyt_threshold(threshold, field="threshold")
    return {
        "result_schema_version": ANY_RESULT_SCHEMA_VERSION,
        "keyword_eval_mode": "any",
        "keyword_set_id": keyword_set.keyword_set_id,
        "qbyt_raw_logit": aggregation.qbyt_raw_logit,
        "qbyt_score": aggregation.qbyt_score,
        "threshold": threshold,
        "detected": aggregation.detected,
        "skipped": False,
        "skip_reason": None,
        "best_keyword": aggregation.best_keyword,
        "best_pronunciation_id": aggregation.best_pronunciation_id,
        "matched_keywords": list(aggregation.matched_keywords),
        "keyword_results": [dict(item) for item in aggregation.keyword_results],
    }


def keyword_set_summary_fields(keyword_set: ResolvedKeywordSet) -> dict[str, Any]:
    return {
        "keyword_eval_mode": "any",
        "keyword_set_id": keyword_set.keyword_set_id,
        "texts": list(keyword_set.texts),
        "num_keywords": len(keyword_set.keywords),
        "num_pronunciations": keyword_set.num_queries,
        "query_batch_size": keyword_set.query_batch_size,
        "score_semantics": SCORE_SEMANTICS,
        "merge_diagnostics": list(keyword_set.merge_diagnostics),
        "keywords": [
            {
                "keyword_id": keyword.keyword_id,
                "text": keyword.text,
                "display_text": keyword.display_text,
                "pronunciations": [
                    {
                        "pronunciation_id": pronunciation.pronunciation_id,
                        "phonemes": list(pronunciation.phonemes),
                        "token_ids": list(pronunciation.token_ids),
                        "source": pronunciation.source,
                        "query_index": pronunciation.query_index,
                    }
                    for pronunciation in keyword.pronunciations
                ],
            }
            for keyword in keyword_set.keywords
        ],
    }


def is_any_mode_record(record: Mapping[str, Any]) -> bool:
    mode = record.get("keyword_eval_mode")
    protocol = record.get("eval_protocol")
    return mode == "any" or protocol in {CLIP_EVAL_PROTOCOL, WINDOW_EVAL_PROTOCOL}


def any_mode_summary_identity_error(
    summary: Mapping[str, Any],
    *,
    keyword_set_id: object,
    eval_protocol: object | None = None,
) -> str | None:
    """Return an error if ``summary`` is not the matching any-mode identity.

    Scan/plot may use summary hours only after mode, ``keyword_set_id``, and
    (when the results have one protocol) ``eval_protocol`` all match.
    """

    summary_mode = summary.get("keyword_eval_mode")
    if summary_mode != "any":
        return (
            "summary keyword_eval_mode does not match results.jsonl: "
            f"{summary_mode!r} vs 'any'"
        )
    summary_id = summary.get("keyword_set_id")
    if summary_id in (None, ""):
        return "any-mode summary is missing keyword_set_id"
    if summary_id != keyword_set_id:
        return (
            "summary keyword_set_id does not match results.jsonl: "
            f"{summary_id!r} vs {keyword_set_id!r}"
        )
    if eval_protocol:
        summary_protocol = summary.get("eval_protocol")
        if summary_protocol != eval_protocol:
            return (
                "summary eval_protocol does not match results.jsonl: "
                f"{summary_protocol!r} vs {eval_protocol!r}"
            )
    return None


__all__ = [
    "ANY_MANIFEST_FORMAT_EXAMPLES",
    "ANY_RESULT_SCHEMA_VERSION",
    "CLIP_EVAL_PROTOCOL",
    "DEFAULT_KEYWORD_EVAL_MODE",
    "DEFAULT_QUERY_BATCH_SIZE",
    "KEYWORD_EVAL_MODES",
    "KEYWORD_SET_SCHEMA_VERSION",
    "KeywordEvalConfigError",
    "KeywordSetAggregation",
    "KeywordSetManifestError",
    "KeywordSetScoreError",
    "ResolvedKeyword",
    "ResolvedKeywordSet",
    "ResolvedPronunciation",
    "ResolvedQuery",
    "SCORE_SEMANTICS",
    "SKIP_REASON_TOO_SHORT",
    "TEXT_NORMALIZATION_VERSION",
    "WINDOW_EVAL_PROTOCOL",
    "aggregate_query_scores",
    "any_mode_summary_identity_error",
    "canonical_json_sha256",
    "file_sha256",
    "is_any_mode_record",
    "keyword_eval_mode",
    "keyword_eval_provenance_block",
    "keyword_set_identity_payload",
    "keyword_set_summary_fields",
    "normalize_keyword_text",
    "reject_unsupported_any_mode",
    "require_positive_int",
    "require_qbyt_threshold",
    "resolve_keyword_set",
    "scored_keyword_set_fields",
    "semantic_keyword_eval",
    "skipped_keyword_set_fields",
]
