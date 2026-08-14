#!/usr/bin/env python3
"""Build a speaker-disjoint LoRA adaptation manifest from a TTS CSV."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Sequence


REQUIRED_SOURCE_FIELDS = {
    "audio_path",
    "keyword",
    "label",
    "voice_id",
    "text_variant",
}
LEADING_OUTPUT_FIELDS = [
    "audio_path",
    "text",
    "label",
    "phase",
    "split",
    "speaker_id",
    "tts_provider",
    "source_keyword",
]
RESERVED_SOURCE_FIELDS = {
    "audio_path",
    "keyword",
    "label",
    "phase",
    "split",
    "speaker_id",
    "tts_provider",
    "source_keyword",
    "text",
}


@dataclass
class NormalizedRow:
    data: dict[str, Any]
    source_line: int
    dedupe_key: str

    @property
    def speaker_id(self) -> str:
        return str(self.data["speaker_id"])

    @property
    def label(self) -> int:
        return int(self.data["label"])


@dataclass(frozen=True)
class SpeakerGroup:
    speaker_id: str
    rows: tuple[NormalizedRow, ...]
    positive: int
    negative: int
    tie_breaker: str

    @property
    def total(self) -> int:
        return len(self.rows)


def _normalized_phrase(value: str) -> str:
    return " ".join(value.split()).casefold()


def _required_text(row: dict[str, str], field: str, *, source: Path, line: int) -> str:
    value = str(row.get(field, "")).strip()
    if not value:
        raise ValueError(f"{source}:{line}: {field} must not be empty")
    return value


def _binary_label(value: str, *, source: Path, line: int) -> int:
    try:
        numeric = float(str(value).strip())
    except ValueError as exc:
        raise ValueError(f"{source}:{line}: label must be 0 or 1, got {value!r}") from exc
    if not numeric.is_integer() or int(numeric) not in {0, 1}:
        raise ValueError(f"{source}:{line}: label must be 0 or 1, got {value!r}")
    return int(numeric)


def _provider_from_audio_path(audio_path: str) -> str:
    parts = PurePosixPath(audio_path.replace("\\", "/")).parts
    normalized_parts = [part.casefold() for part in parts]
    if "elevenlabs_output" in normalized_parts:
        return "elevenlabs"
    if "googletts_output" in normalized_parts:
        return "googletts"
    for normalized in reversed(normalized_parts):
        if normalized.endswith("_output") and len(normalized) > len("_output"):
            return normalized[: -len("_output")]
    return "tts"


def _resolved_audio_path(raw_path: str, audio_root: Path) -> Path:
    path = Path(raw_path).expanduser()
    return path.resolve() if path.is_absolute() else (audio_root / path).resolve()


def _dedupe_signature(row: NormalizedRow) -> tuple[Any, ...]:
    semantic_fields = (
        "text",
        "label",
        "speaker_id",
    )
    signature = tuple((field, row.data[field]) for field in semantic_fields)
    phoneme_signature = tuple(
        sorted(
            (field, value)
            for field, value in row.data.items()
            if field.endswith("_phonemes")
        )
    )
    return (*signature, *phoneme_signature)


def _read_source_rows(
    source: Path,
    *,
    audio_root: Path,
    keyword: str,
    verify_audio: bool,
) -> tuple[list[NormalizedRow], int]:
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or ())
        missing = REQUIRED_SOURCE_FIELDS - fields
        if missing:
            raise ValueError(f"{source} missing columns: {sorted(missing)}")

        rows: list[NormalizedRow] = []
        seen: dict[str, NormalizedRow] = {}
        duplicate_count = 0
        expected_keyword = _normalized_phrase(keyword)
        for line_number, source_row in enumerate(reader, start=2):
            raw_audio_path = _required_text(
                source_row,
                "audio_path",
                source=source,
                line=line_number,
            )
            source_keyword = _required_text(
                source_row,
                "keyword",
                source=source,
                line=line_number,
            )
            if _normalized_phrase(source_keyword) != expected_keyword:
                raise ValueError(
                    f"{source}:{line_number}: keyword {source_keyword!r} does not "
                    f"match requested keyword {keyword!r}"
                )
            text = _required_text(
                source_row,
                "text_variant",
                source=source,
                line=line_number,
            )
            voice_id = _required_text(
                source_row,
                "voice_id",
                source=source,
                line=line_number,
            )
            label = _binary_label(
                source_row.get("label", ""),
                source=source,
                line=line_number,
            )
            audio_path = _resolved_audio_path(raw_audio_path, audio_root)
            if verify_audio and not audio_path.is_file():
                raise FileNotFoundError(
                    f"{source}:{line_number}: audio file not found: {audio_path}"
                )

            provider = _provider_from_audio_path(raw_audio_path)
            # Match the case-insensitive identity used by downstream leakage checks.
            speaker_id = f"{provider.casefold()}:{voice_id.casefold()}"
            output_row: dict[str, Any] = {
                "audio_path": str(audio_path),
                "text": text,
                "label": label,
                "phase": "tts",
                "split": "",
                "speaker_id": speaker_id,
                "tts_provider": provider,
                "source_keyword": source_keyword,
            }
            for field, value in source_row.items():
                if field is None or field in RESERVED_SOURCE_FIELDS:
                    continue
                output_row[field] = "" if value is None else str(value).strip()

            sha256 = str(output_row.get("sha256", "")).strip().casefold()
            if sha256 and re.fullmatch(r"[0-9a-f]{64}", sha256) is None:
                raise ValueError(
                    f"{source}:{line_number}: sha256 must be 64 hexadecimal "
                    f"characters or empty, got {sha256!r}"
                )
            dedupe_key = f"sha256:{sha256}" if sha256 else f"path:{audio_path}"
            normalized = NormalizedRow(
                data=output_row,
                source_line=line_number,
                dedupe_key=dedupe_key,
            )
            previous = seen.get(dedupe_key)
            if previous is not None:
                if _dedupe_signature(previous) != _dedupe_signature(normalized):
                    raise ValueError(
                        f"{source}:{line_number}: duplicate audio key {dedupe_key!r} "
                        f"conflicts with line {previous.source_line}"
                    )
                duplicate_count += 1
                continue
            seen[dedupe_key] = normalized
            rows.append(normalized)

    if not rows:
        raise ValueError(f"{source} contains no usable rows")
    return rows, duplicate_count


def _speaker_groups(
    rows: Sequence[NormalizedRow],
    *,
    seed: int,
) -> list[SpeakerGroup]:
    grouped: dict[str, list[NormalizedRow]] = {}
    for row in rows:
        grouped.setdefault(row.speaker_id, []).append(row)
    if len(grouped) < 2:
        raise ValueError("Speaker-disjoint train/eval requires at least two voice_id values")

    groups = [
        SpeakerGroup(
            speaker_id=speaker_id,
            rows=tuple(speaker_rows),
            positive=sum(row.label == 1 for row in speaker_rows),
            negative=sum(row.label == 0 for row in speaker_rows),
            tie_breaker=hashlib.sha256(
                f"{seed}:{speaker_id}".encode("utf-8")
            ).hexdigest(),
        )
        for speaker_id, speaker_rows in grouped.items()
    ]
    groups.sort(key=lambda group: (-group.total, group.tie_breaker, group.speaker_id))

    for label, name in ((0, "negative"), (1, "positive")):
        speakers = sum(any(row.label == label for row in group.rows) for group in groups)
        if speakers < 2:
            raise ValueError(
                f"Both train and eval need {name} samples, but label={label} occurs "
                f"in only {speakers} speaker group(s)"
            )
    return groups


def _assignment_score(
    *,
    eval_rows: int,
    eval_positive: int,
    eval_negative: int,
    eval_speakers: int,
    total_rows: int,
    total_positive: int,
    total_negative: int,
    total_speakers: int,
    eval_fraction: float,
) -> tuple[int, int, float, float, float]:
    train_rows = total_rows - eval_rows
    train_positive = total_positive - eval_positive
    train_negative = total_negative - eval_negative
    empty_split = int(eval_speakers == 0 or eval_speakers == total_speakers)
    missing_classes = sum(
        value == 0
        for value in (
            eval_positive,
            eval_negative,
            train_positive,
            train_negative,
        )
    )
    target_eval_rows = total_rows * eval_fraction
    target_eval_positive = total_positive * eval_fraction
    target_eval_negative = total_negative * eval_fraction
    target_eval_speakers = total_speakers * eval_fraction
    return (
        empty_split,
        missing_classes,
        abs(eval_rows - target_eval_rows),
        abs(eval_positive - target_eval_positive)
        + abs(eval_negative - target_eval_negative),
        abs(eval_speakers - target_eval_speakers),
    )


def assign_speaker_splits(
    groups: Sequence[SpeakerGroup],
    *,
    eval_fraction: float = 0.5,
) -> dict[str, str]:
    """Assign whole speakers while balancing rows first and labels second."""

    if not 0.0 < eval_fraction < 1.0:
        raise ValueError("eval_fraction must be strictly between 0 and 1")
    total_rows = sum(group.total for group in groups)
    total_positive = sum(group.positive for group in groups)
    total_negative = sum(group.negative for group in groups)
    total_speakers = len(groups)

    def score(
        eval_rows: int,
        eval_positive: int,
        eval_negative: int,
        eval_speakers: int,
    ) -> tuple[int, int, float, float, float]:
        return _assignment_score(
            eval_rows=eval_rows,
            eval_positive=eval_positive,
            eval_negative=eval_negative,
            eval_speakers=eval_speakers,
            total_rows=total_rows,
            total_positive=total_positive,
            total_negative=total_negative,
            total_speakers=total_speakers,
            eval_fraction=eval_fraction,
        )

    assignment: dict[str, str] = {
        groups[0].speaker_id: "train",
        groups[1].speaker_id: "eval",
    }
    eval_rows = groups[1].total
    eval_positive = groups[1].positive
    eval_negative = groups[1].negative
    eval_speakers = 1

    for group in groups[2:]:
        train_score = score(
            eval_rows,
            eval_positive,
            eval_negative,
            eval_speakers,
        )
        eval_score = score(
            eval_rows + group.total,
            eval_positive + group.positive,
            eval_negative + group.negative,
            eval_speakers + 1,
        )
        choose_eval = eval_score < train_score or (
            eval_score == train_score and int(group.tie_breaker[-1], 16) % 2 == 1
        )
        assignment[group.speaker_id] = "eval" if choose_eval else "train"
        if choose_eval:
            eval_rows += group.total
            eval_positive += group.positive
            eval_negative += group.negative
            eval_speakers += 1

    group_by_id = {group.speaker_id: group for group in groups}
    current_score = score(eval_rows, eval_positive, eval_negative, eval_speakers)
    # Improve the greedy result with deterministic single-speaker moves and swaps.
    for _ in range(max(1, total_speakers * 2)):
        best: tuple[
            tuple[int, int, float, float, float],
            str,
            str,
            int,
            int,
            int,
            int,
        ] | None = None
        train_ids = sorted(
            speaker_id for speaker_id, split in assignment.items() if split == "train"
        )
        eval_ids = sorted(
            speaker_id for speaker_id, split in assignment.items() if split == "eval"
        )

        for speaker_id in train_ids:
            if len(train_ids) <= 1:
                break
            group = group_by_id[speaker_id]
            candidate = (
                score(
                    eval_rows + group.total,
                    eval_positive + group.positive,
                    eval_negative + group.negative,
                    eval_speakers + 1,
                ),
                "move_to_eval",
                speaker_id,
                eval_rows + group.total,
                eval_positive + group.positive,
                eval_negative + group.negative,
                eval_speakers + 1,
            )
            if best is None or candidate < best:
                best = candidate

        for speaker_id in eval_ids:
            if len(eval_ids) <= 1:
                break
            group = group_by_id[speaker_id]
            candidate = (
                score(
                    eval_rows - group.total,
                    eval_positive - group.positive,
                    eval_negative - group.negative,
                    eval_speakers - 1,
                ),
                "move_to_train",
                speaker_id,
                eval_rows - group.total,
                eval_positive - group.positive,
                eval_negative - group.negative,
                eval_speakers - 1,
            )
            if best is None or candidate < best:
                best = candidate

        for train_id in train_ids:
            train_group = group_by_id[train_id]
            for eval_id in eval_ids:
                eval_group = group_by_id[eval_id]
                candidate = (
                    score(
                        eval_rows + train_group.total - eval_group.total,
                        eval_positive + train_group.positive - eval_group.positive,
                        eval_negative + train_group.negative - eval_group.negative,
                        eval_speakers,
                    ),
                    "swap",
                    f"{train_id}\0{eval_id}",
                    eval_rows + train_group.total - eval_group.total,
                    eval_positive + train_group.positive - eval_group.positive,
                    eval_negative + train_group.negative - eval_group.negative,
                    eval_speakers,
                )
                if best is None or candidate < best:
                    best = candidate

        if best is None or best[0] >= current_score:
            break
        _, operation, operand, eval_rows, eval_positive, eval_negative, eval_speakers = best
        if operation == "move_to_eval":
            assignment[operand] = "eval"
        elif operation == "move_to_train":
            assignment[operand] = "train"
        else:
            train_id, eval_id = operand.split("\0", maxsplit=1)
            assignment[train_id] = "eval"
            assignment[eval_id] = "train"
        current_score = best[0]

    if current_score[0] or current_score[1]:
        raise ValueError(
            "Could not construct speaker-disjoint train/eval splits that both "
            "contain positive and negative samples"
        )
    return assignment


def _manifest_summary(
    rows: Sequence[NormalizedRow],
    *,
    source_rows: int,
    duplicate_rows: int,
    eval_fraction: float,
    source: Path,
    output: Path,
    audio_root: Path,
) -> dict[str, Any]:
    by_split: dict[str, dict[str, Any]] = {}
    for split in ("train", "eval"):
        split_rows = [row for row in rows if row.data["split"] == split]
        by_split[split] = {
            "rows": len(split_rows),
            "speakers": len({row.speaker_id for row in split_rows}),
            "positive": sum(row.label == 1 for row in split_rows),
            "negative": sum(row.label == 0 for row in split_rows),
        }
    train_speakers = {
        row.speaker_id for row in rows if row.data["split"] == "train"
    }
    eval_speakers = {
        row.speaker_id for row in rows if row.data["split"] == "eval"
    }
    overlap = sorted(train_speakers & eval_speakers)
    if overlap:
        raise AssertionError(f"Internal error: speaker leakage detected: {overlap}")

    train_rows = int(by_split["train"]["rows"])
    eval_rows = int(by_split["eval"]["rows"])
    return {
        "input_manifest": str(source.resolve()),
        "output_manifest": str(output.resolve()),
        "audio_root": str(audio_root),
        "source_rows": source_rows,
        "duplicate_rows_removed": duplicate_rows,
        "output_rows": len(rows),
        "unique_speakers": len(train_speakers | eval_speakers),
        "requested_eval_fraction": eval_fraction,
        "actual_eval_fraction": eval_rows / len(rows),
        "row_gap": abs(train_rows - eval_rows),
        "exact_1_to_1": train_rows == eval_rows,
        "speaker_overlap": overlap,
        "splits": by_split,
    }


def build_tts_lora_manifest(
    source: Path,
    output: Path,
    *,
    audio_root: Path | None = None,
    keyword: str = "hey eva",
    eval_fraction: float = 0.5,
    seed: int = 2025,
    verify_audio: bool = True,
) -> dict[str, Any]:
    source = source.expanduser().resolve()
    output = output.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Input manifest not found: {source}")
    if source == output:
        raise ValueError("Input and output manifest paths must be different")
    resolved_audio_root = (
        audio_root.expanduser().resolve() if audio_root is not None else source.parent
    )
    if not 0.0 < eval_fraction < 1.0:
        raise ValueError("eval_fraction must be strictly between 0 and 1")

    rows, duplicate_count = _read_source_rows(
        source,
        audio_root=resolved_audio_root,
        keyword=keyword,
        verify_audio=verify_audio,
    )
    source_row_count = len(rows) + duplicate_count
    groups = _speaker_groups(rows, seed=seed)
    assignment = assign_speaker_splits(groups, eval_fraction=eval_fraction)
    for row in rows:
        row.data["split"] = assignment[row.speaker_id]
    rows.sort(
        key=lambda row: (
            row.data["split"] != "train",
            row.speaker_id,
            str(row.data["audio_path"]),
        )
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = LEADING_OUTPUT_FIELDS + sorted(
        {field for row in rows for field in row.data} - set(LEADING_OUTPUT_FIELDS)
    )
    temporary = output.with_suffix(output.suffix + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(row.data for row in rows)
        temporary.replace(output)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass

    return _manifest_summary(
        rows,
        source_rows=source_row_count,
        duplicate_rows=duplicate_count,
        eval_fraction=eval_fraction,
        source=source,
        output=output,
        audio_root=resolved_audio_root,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument(
        "--audio-root",
        type=Path,
        help="Root for relative audio_path values (default: input CSV directory)",
    )
    parser.add_argument("--keyword", default="hey eva")
    parser.add_argument(
        "--eval-fraction",
        type=float,
        default=0.5,
        help="Target row fraction for eval; speakers remain indivisible (default: 0.5)",
    )
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument(
        "--skip-audio-existence-check",
        action="store_true",
        help="Allow manifest generation before audio is mounted at --audio-root",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = build_parser().parse_args(argv)
    summary = build_tts_lora_manifest(
        args.input_manifest,
        args.output_manifest,
        audio_root=args.audio_root,
        keyword=args.keyword,
        eval_fraction=args.eval_fraction,
        seed=args.seed,
        verify_audio=not args.skip_audio_existence_check,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return summary


if __name__ == "__main__":
    main()
