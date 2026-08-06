"""Select auditable Hey-Eva hard negatives from Stage-II clip scores."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
import math

from dma_kws.phonemes import normalize_english_text


def _contains_phrase(text: str, phrase: str) -> bool:
    text_tokens = normalize_english_text(text).split()
    phrase_tokens = normalize_english_text(phrase).split()
    if not phrase_tokens:
        raise ValueError("keyword must be non-empty after normalization")
    width = len(phrase_tokens)
    return any(
        text_tokens[index : index + width] == phrase_tokens
        for index in range(len(text_tokens) - width + 1)
    )


def _metadata(record: Mapping[str, object]) -> dict[str, object]:
    nested = record.get("manifest_meta")
    metadata = dict(nested) if isinstance(nested, Mapping) else {}
    for key in (
        "text_variant",
        "text_variant_phonemes",
        "text",
        "keyword_phonemes",
        "source",
        "speaker_id",
        "device",
        "session",
        "split",
    ):
        if key in record and key not in metadata:
            metadata[key] = record[key]
    return metadata


def _phoneme_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return " ".join(str(phone).strip() for phone in value if str(phone).strip())
    return " ".join(str(value).strip().split())


def select_hard_negatives(
    records: Iterable[Mapping[str, object]],
    *,
    keyword: str,
    min_score: float,
    top_k: int,
    per_speaker_cap: int,
    allowed_splits: frozenset[str] = frozenset({"train"}),
) -> list[dict[str, object]]:
    """Return high-scoring false positives without crossing split boundaries.

    Input rows are the ``results.jsonl`` records written by
    :mod:`scripts.eval_stage2_clips`.  Every accepted row must be an explicitly
    labeled negative from an allowed split and carry a speaker id, so a blind
    test row can never enter LoRA merely because it scored highly.
    """

    if not 0.0 <= float(min_score) <= 1.0:
        raise ValueError("min_score must be in [0, 1]")
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if per_speaker_cap <= 0:
        raise ValueError("per_speaker_cap must be positive")
    normalized_allowed = {str(split).strip().casefold() for split in allowed_splits}
    if not normalized_allowed:
        raise ValueError("allowed_splits must not be empty")

    candidates: list[dict[str, object]] = []
    for row_number, record in enumerate(records, start=1):
        if bool(record.get("skipped", False)):
            continue
        try:
            label = int(record["label"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"Scored row {row_number} needs an integer label; hard-negative "
                "mining never assumes unlabeled audio is negative"
            ) from exc
        if label != 0:
            continue

        try:
            score = float(record["qbyt_score"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Scored row {row_number} has no valid qbyt_score") from exc
        if not math.isfinite(score):
            raise ValueError(f"Scored row {row_number} has non-finite qbyt_score")
        if score < min_score:
            continue

        metadata = _metadata(record)
        split = str(metadata.get("split", "")).strip().casefold()
        if split not in normalized_allowed:
            continue
        speaker_id = str(metadata.get("speaker_id", "")).strip()
        if not speaker_id:
            raise ValueError(
                f"Scored row {row_number} is eligible but has no speaker_id; "
                "speaker caps and leakage checks would be impossible"
            )
        text = str(
            metadata.get("text_variant", metadata.get("text", ""))
        ).strip()
        if not text:
            raise ValueError(
                f"Scored row {row_number} is eligible but has no transcript text_variant"
            )
        if _contains_phrase(text, keyword):
            # A phrase containing the complete keyword is a positive under the
            # ordered-contiguous-prefix objective, regardless of a stale label.
            continue
        audio_path = str(record.get("audio_path", "")).strip()
        if not audio_path:
            raise ValueError(f"Scored row {row_number} has no audio_path")

        candidates.append(
            {
                "audio_path": audio_path,
                "text": text,
                "keyword": keyword,
                "keyword_phonemes": _phoneme_text(
                    metadata.get("keyword_phonemes")
                ),
                "text_variant_phonemes": _phoneme_text(
                    metadata.get("text_variant_phonemes")
                ),
                "label": 0,
                "phase": "real",
                "split": "train",
                "speaker_id": speaker_id,
                "device": str(metadata.get("device", "")),
                "session": str(metadata.get("session", "")),
                "source": str(metadata.get("source", "")),
                "negative_type": "stage2_false_positive",
                "qbyt_score": score,
            }
        )

    candidates.sort(key=lambda row: (-float(row["qbyt_score"]), str(row["audio_path"])))
    selected: list[dict[str, object]] = []
    seen_audio: set[str] = set()
    speaker_counts: dict[str, int] = defaultdict(int)
    for candidate in candidates:
        audio_path = str(candidate["audio_path"])
        speaker_id = str(candidate["speaker_id"])
        if audio_path in seen_audio or speaker_counts[speaker_id] >= per_speaker_cap:
            continue
        selected.append(candidate)
        seen_audio.add(audio_path)
        speaker_counts[speaker_id] += 1
        if len(selected) >= top_k:
            break
    return selected


def merge_hard_negative_manifest_rows(
    base_records: Iterable[Mapping[str, object]],
    hard_negatives: Iterable[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Merge mined train rows into a train/eval source manifest safely."""

    rows = [dict(record) for record in base_records]
    by_audio: dict[str, dict[str, object]] = {}
    for row_number, row in enumerate(rows, start=1):
        audio_path = str(row.get("audio_path", "")).strip()
        if not audio_path:
            raise ValueError(f"Base row {row_number} has no audio_path")
        if audio_path in by_audio:
            raise ValueError(f"Duplicate audio_path in base manifest: {audio_path}")
        by_audio[audio_path] = row

    for negative in hard_negatives:
        candidate = dict(negative)
        audio_path = str(candidate.get("audio_path", "")).strip()
        existing = by_audio.get(audio_path)
        if existing is None:
            candidate["mined_hard_negative"] = 1
            rows.append(candidate)
            by_audio[audio_path] = candidate
            continue

        # The real near-negative pool is deliberately scored alongside generic
        # accent speech, but those same rows already live in real_source.csv.
        # Annotate an identical base row instead of duplicating its audio.
        compatible = (
            str(existing.get("split", "")).strip().casefold() == "train"
            and int(existing.get("label", -1)) == 0
            and str(existing.get("speaker_id", "")).strip().casefold()
            == str(candidate.get("speaker_id", "")).strip().casefold()
            and str(existing.get("text", "")).strip().casefold()
            == str(candidate.get("text", "")).strip().casefold()
        )
        if not compatible:
            raise ValueError(
                f"Hard-negative duplicate conflicts with base manifest: {audio_path}"
            )
        existing["mined_hard_negative"] = 1
        if candidate.get("qbyt_score") not in (None, ""):
            existing["qbyt_score"] = candidate["qbyt_score"]
    if not rows:
        raise ValueError("Combined adaptation manifest would be empty")

    seen_audio: set[str] = set()
    speakers_by_split: dict[str, set[str]] = defaultdict(set)
    for row_number, row in enumerate(rows, start=1):
        audio_path = str(row.get("audio_path", "")).strip()
        if not audio_path:
            raise ValueError(f"Combined row {row_number} has no audio_path")
        if audio_path in seen_audio:
            raise ValueError(f"Duplicate audio_path in combined manifest: {audio_path}")
        seen_audio.add(audio_path)

        split = str(row.get("split", "")).strip().casefold()
        if split not in {"train", "eval"}:
            raise ValueError(
                f"Combined row {row_number} has split={split!r}; expected train or eval"
            )
        speaker_id = str(row.get("speaker_id", "")).strip().casefold()
        if not speaker_id:
            raise ValueError(f"Combined row {row_number} has no speaker_id")
        speakers_by_split[split].add(speaker_id)

    if not speakers_by_split["train"] or not speakers_by_split["eval"]:
        raise ValueError("Combined adaptation manifest requires both train and eval rows")
    overlap = sorted(speakers_by_split["train"] & speakers_by_split["eval"])
    if overlap:
        raise ValueError(f"Combined adaptation manifest leaks speakers: {overlap}")
    return rows


__all__ = ["merge_hard_negative_manifest_rows", "select_hard_negatives"]
