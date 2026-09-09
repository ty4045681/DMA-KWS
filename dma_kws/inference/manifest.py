"""Manifest loading and generation for batch two-stage evaluation."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

from dma_kws.inference.keyword_set import KeywordSetManifestError


_REQUIRED_COLUMNS = ("audio_path", "keyword")
_DEFAULT_AUDIO_EXTENSIONS = (".wav", ".flac", ".mp3", ".m4a")


def load_audio_file_list(path: str | Path) -> list[Path]:
    """Load a deterministic list of canonical audio file paths.

    Blank lines and lines whose first non-space character is ``#`` are ignored.
    Relative entries are resolved against the list file's directory. Missing
    files and duplicate canonical paths are rejected so train/eval allowlists
    cannot overlap accidentally through alternate path spellings or symlinks.
    """

    list_path = Path(path).expanduser()
    if not list_path.is_file():
        raise FileNotFoundError(f"Audio file list not found: {list_path}")

    base_dir = list_path.parent.resolve()
    files: list[Path] = []
    first_line_by_path: dict[Path, int] = {}
    with list_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            entry = line.strip()
            if not entry or entry.startswith("#"):
                continue
            candidate = Path(entry).expanduser()
            if not candidate.is_absolute():
                candidate = base_dir / candidate
            resolved = candidate.resolve()
            if not resolved.is_file():
                raise FileNotFoundError(
                    f"{list_path}:{line_number} audio file not found: {candidate}"
                )
            previous_line = first_line_by_path.get(resolved)
            if previous_line is not None:
                raise ValueError(
                    f"{list_path}:{line_number} duplicates canonical audio path "
                    f"from line {previous_line}: {resolved}"
                )
            first_line_by_path[resolved] = line_number
            files.append(resolved)

    return sorted(files, key=lambda item: str(item))


def _normalize_row(
    row: Mapping[str, Any],
    base_dir: Path | None,
    *,
    row_number: int,
) -> dict:
    normalized = {
        key.strip(): (value.strip() if isinstance(value, str) else value)
        for key, value in row.items()
    }
    audio_path = normalized.get("audio_path", "")
    if audio_path and base_dir is not None and not Path(audio_path).is_absolute():
        normalized["audio_path"] = str((base_dir / audio_path).resolve())

    if "label" in normalized and normalized["label"] != "":
        try:
            normalized["label"] = int(normalized["label"])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Manifest row {row_number} has invalid label {normalized['label']!r}; "
                "expected an integer"
            ) from exc
    elif "label" in normalized:
        normalized.pop("label")

    return normalized


def _validate_row(row: dict, index: int) -> None:
    missing = [column for column in _REQUIRED_COLUMNS if not row.get(column)]
    if missing:
        raise ValueError(f"Manifest row {index} is missing required columns: {', '.join(missing)}")


def load_manifest(path: str | Path) -> list[dict]:
    """Load a CSV or JSONL manifest with ``audio_path``, ``keyword``, optional ``label``."""
    manifest_path = Path(path)
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)

    base_dir = manifest_path.parent
    suffix = manifest_path.suffix.lower()
    rows: list[dict] = []

    if suffix == ".jsonl":
        with manifest_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                row = _normalize_row(
                    json.loads(line),
                    base_dir,
                    row_number=line_number,
                )
                _validate_row(row, line_number)
                rows.append(row)
        return rows

    if suffix == ".csv":
        with manifest_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for line_number, row in enumerate(reader, start=2):
                normalized = _normalize_row(
                    row,
                    base_dir,
                    row_number=line_number,
                )
                _validate_row(normalized, line_number)
                rows.append(normalized)
        return rows

    raise ValueError(f"Unsupported manifest format: {manifest_path.suffix}")


def _strip_manifest_mapping(row: Mapping[str, Any]) -> dict:
    return {
        str(key).strip(): (value.strip() if isinstance(value, str) else value)
        for key, value in row.items()
        if key is not None and str(key).strip() != ""
    }


def _resolve_audio_path_field(
    row: dict,
    base_dir: Path | None,
) -> dict:
    audio_path = row.get("audio_path", "")
    if audio_path and base_dir is not None and not Path(str(audio_path)).is_absolute():
        row = dict(row)
        row["audio_path"] = str((base_dir / str(audio_path)).resolve())
    return row


def _iter_manifest_dicts(path: str | Path) -> Iterator[tuple[int, dict]]:
    """Yield ``(row_number, stripped row)`` with resolved ``audio_path``."""

    manifest_path = Path(path)
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    base_dir = manifest_path.parent
    suffix = manifest_path.suffix.lower()
    if suffix == ".jsonl":
        with manifest_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    parsed = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Manifest row {line_number} is not valid JSON: {exc.msg}"
                    ) from exc
                if not isinstance(parsed, Mapping):
                    raise ValueError(f"Manifest row {line_number} must be a JSON object")
                yield line_number, _resolve_audio_path_field(
                    _strip_manifest_mapping(parsed),
                    base_dir,
                )
        return
    if suffix == ".csv":
        with manifest_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for line_number, row in enumerate(reader, start=2):
                yield line_number, _resolve_audio_path_field(
                    _strip_manifest_mapping(row),
                    base_dir,
                )
        return
    raise ValueError(f"Unsupported manifest format: {manifest_path.suffix}")


def _parse_json_cell(value: object, *, field: str, row_number: int) -> object:
    if isinstance(value, (dict, list)):
        return value
    if not isinstance(value, str) or not value.strip():
        raise KeywordSetManifestError(
            f"Manifest row {row_number} {field} must be JSON text"
        )
    try:
        return json.loads(value)
    except json.JSONDecodeError as exc:
        raise KeywordSetManifestError(
            f"Manifest row {row_number} {field} is not valid JSON: {exc.msg}"
        ) from exc


def _strict_binary_int(
    value: object,
    *,
    field: str,
    row_number: int,
    allow_csv_string: bool = False,
) -> int:
    if allow_csv_string and isinstance(value, str) and value in {"0", "1"}:
        return int(value)
    if isinstance(value, bool) or type(value) is not int or value not in (0, 1):
        raise KeywordSetManifestError(
            f"Manifest row {row_number} {field} must be integer 0 or 1, "
            f"got {value!r}"
        )
    return value


def _row_has_field(row: Mapping[str, Any], key: str) -> bool:
    if key not in row:
        return False
    value = row[key]
    if value is None:
        return False
    if isinstance(value, str) and value.strip() == "":
        return False
    return True


def load_keyword_set_manifest(
    path: str | Path,
    target_texts: Sequence[str],
) -> list[dict]:
    """Load an any-mode audio-level manifest.

    Accepts JSONL or CSV. One source audio per row. Labels never change the
    enrolled query set; they only populate ``label`` / ``keyword_labels``.
    """

    from dma_kws.inference.keyword_set import (
        ANY_MANIFEST_FORMAT_EXAMPLES,
        normalize_keyword_text,
    )

    try:
        configured = [
            normalize_keyword_text(text, field="target_texts")
            for text in target_texts
        ]
    except Exception as exc:
        raise KeywordSetManifestError(str(exc)) from exc
    configured_set = set(configured)
    if len(configured_set) != len(configured):
        raise KeywordSetManifestError("target_texts must not contain duplicates")
    if not configured_set:
        raise KeywordSetManifestError("target_texts must not be empty")

    rows: list[dict] = []
    seen_paths: dict[Path, int] = {}
    for row_number, raw in _iter_manifest_dicts(path):
        audio_path = raw.get("audio_path", "")
        if not audio_path:
            raise KeywordSetManifestError(
                f"Manifest row {row_number} is missing audio_path"
            )
        resolved = Path(str(audio_path)).expanduser().resolve()
        previous = seen_paths.get(resolved)
        if previous is not None:
            raise KeywordSetManifestError(
                f"Manifest row {row_number} duplicates canonical audio path "
                f"from row {previous}: {resolved}"
            )
        seen_paths[resolved] = row_number

        if _row_has_field(raw, "keyword") or _row_has_field(raw, "keyword_phonemes"):
            raise KeywordSetManifestError(
                f"Manifest row {row_number} uses legacy pair fields "
                "`keyword` / `keyword_phonemes`, which any-mode rejects.\n"
                f"{ANY_MANIFEST_FORMAT_EXAMPLES}"
            )

        has_keyword_labels = _row_has_field(raw, "keyword_labels")
        has_scope = _row_has_field(raw, "label_scope")
        has_target_texts = _row_has_field(raw, "target_texts")
        has_label = _row_has_field(raw, "label")
        labeled_bits = (has_keyword_labels, has_scope, has_target_texts, has_label)

        row = dict(raw)
        row["audio_path"] = str(resolved)
        row["_manifest_row_number"] = row_number
        for key in ("label", "keyword_labels", "label_scope", "target_texts"):
            if not _row_has_field(row, key):
                row.pop(key, None)

        if not any(labeled_bits):
            rows.append(row)
            continue

        if has_keyword_labels:
            if has_scope or has_target_texts:
                raise KeywordSetManifestError(
                    f"Manifest row {row_number} mixes keyword_labels with "
                    "label_scope/target_texts"
                )
            labels_raw = raw["keyword_labels"]
            if isinstance(labels_raw, str):
                labels_raw = _parse_json_cell(
                    labels_raw,
                    field="keyword_labels",
                    row_number=row_number,
                )
            if not isinstance(labels_raw, Mapping):
                raise KeywordSetManifestError(
                    f"Manifest row {row_number} keyword_labels must be a mapping"
                )
            normalized_labels: dict[str, int] = {}
            unused: dict[str, int] = {}
            for key, value in labels_raw.items():
                try:
                    text = normalize_keyword_text(
                        key,
                        field=f"Manifest row {row_number} keyword_labels key",
                    )
                except Exception as exc:
                    raise KeywordSetManifestError(str(exc)) from exc
                flag = _strict_binary_int(
                    value,
                    field=f"keyword_labels[{key!r}]",
                    row_number=row_number,
                )
                if text in configured_set:
                    previous_flag = normalized_labels.get(text)
                    if previous_flag is not None and previous_flag != flag:
                        raise KeywordSetManifestError(
                            f"Manifest row {row_number} has conflicting labels "
                            f"for {text!r}"
                        )
                    normalized_labels[text] = flag
                else:
                    unused[text] = flag
            missing = sorted(configured_set - set(normalized_labels))
            if missing:
                raise KeywordSetManifestError(
                    f"Manifest row {row_number} keyword_labels is missing "
                    f"configured texts: {missing}. A negative label for one "
                    "keyword does not imply the whole target set is negative."
                )
            derived = max(normalized_labels[text] for text in configured)
            if has_label:
                given = _strict_binary_int(
                    raw["label"],
                    field="label",
                    row_number=row_number,
                    allow_csv_string=True,
                )
                if given != derived:
                    raise KeywordSetManifestError(
                        f"Manifest row {row_number} label={given} does not "
                        f"match max(keyword_labels)={derived}"
                    )
            row["keyword_labels"] = {
                text: normalized_labels[text] for text in configured
            }
            if unused:
                row["unused_keyword_labels"] = unused
            row["label"] = derived
            row["label_scope"] = "target_set"
            rows.append(row)
            continue

        if not (has_scope and has_target_texts and has_label):
            raise KeywordSetManifestError(
                f"Manifest row {row_number} has incomplete label fields. "
                "Provide keyword_labels covering every configured text, or "
                "label_scope='target_set' with matching target_texts and "
                f"label 0|1, or omit every label field.\n"
                f"{ANY_MANIFEST_FORMAT_EXAMPLES}"
            )
        scope = raw["label_scope"]
        if scope != "target_set":
            raise KeywordSetManifestError(
                f"Manifest row {row_number} label_scope must be 'target_set', "
                f"got {scope!r}"
            )
        texts_raw = raw["target_texts"]
        if isinstance(texts_raw, str):
            texts_raw = _parse_json_cell(
                texts_raw,
                field="target_texts",
                row_number=row_number,
            )
        if not isinstance(texts_raw, list) or not all(
            isinstance(item, str) for item in texts_raw
        ):
            raise KeywordSetManifestError(
                f"Manifest row {row_number} target_texts must be a JSON list "
                "of strings"
            )
        try:
            provided = {
                normalize_keyword_text(
                    item,
                    field=f"Manifest row {row_number} target_texts",
                )
                for item in texts_raw
            }
        except Exception as exc:
            raise KeywordSetManifestError(str(exc)) from exc
        if provided != configured_set:
            raise KeywordSetManifestError(
                f"Manifest row {row_number} target_texts {sorted(provided)} "
                f"must equal configured texts {sorted(configured_set)}"
            )
        row["label"] = _strict_binary_int(
            raw["label"],
            field="label",
            row_number=row_number,
            allow_csv_string=True,
        )
        row["label_scope"] = "target_set"
        row["target_texts"] = list(configured)
        rows.append(row)

    return rows


def iter_audio_files(
    input_dir: str | Path,
    *,
    extensions: Iterable[str] = _DEFAULT_AUDIO_EXTENSIONS,
    recursive: bool = True,
) -> list[Path]:
    """Return sorted audio files under ``input_dir`` filtered by ``extensions``."""
    directory = Path(input_dir)
    if not directory.is_dir():
        raise NotADirectoryError(f"Audio input directory not found: {directory}")

    allowed = {ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in extensions}
    walker = directory.rglob("*") if recursive else directory.glob("*")
    return sorted(
        path for path in walker if path.is_file() and path.suffix.lower() in allowed
    )


def normalize_keyword(value: str, *, casefold: bool = True) -> str:
    """Normalize a keyword-like string for robust matching.

    Normalization removes spaces and underscores. Matching can optionally be
    case-insensitive via ``casefold``.
    """
    text = value.casefold() if casefold else value
    return "".join(char for char in text if char not in {" ", "_"})


def filename_keyword_candidate(audio_path: str | Path) -> str:
    """Extract the keyword candidate from filename stem.

    Rule: split stem by ``_``, drop the last segment, and join the remaining
    segments without separators.
    """
    stem = Path(audio_path).stem
    parts = [part for part in stem.split("_") if part != ""]
    if len(parts) <= 1:
        return ""
    return "".join(parts[:-1])


def _keyword_lookup(
    keywords: Sequence[str],
    *,
    casefold: bool,
) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for keyword in keywords:
        normalized = normalize_keyword(keyword, casefold=casefold)
        if not normalized:
            raise ValueError("keywords must contain non-empty strings")
        previous = lookup.get(normalized)
        if previous is not None and previous != keyword:
            raise ValueError(
                f"Ambiguous keywords after normalization: '{previous}' and '{keyword}'"
            )
        lookup[normalized] = keyword
    return lookup


def match_keyword_from_filename(
    audio_path: str | Path,
    keywords: Sequence[str],
    *,
    casefold: bool = True,
) -> str | None:
    """Match an audio filename to one keyword from ``keywords``.

    Filename candidate is extracted via :func:`filename_keyword_candidate`.
    Keyword comparison uses :func:`normalize_keyword`.
    """
    if not keywords:
        raise ValueError("keywords must not be empty")
    lookup = _keyword_lookup(keywords, casefold=casefold)
    candidate = filename_keyword_candidate(audio_path)
    if not candidate:
        return None
    normalized_candidate = normalize_keyword(candidate, casefold=casefold)
    return lookup.get(normalized_candidate)


def build_manifest_rows(
    audio_paths: Sequence[str | Path],
    keyword: str,
    *,
    label: int | None = None,
    manifest_dir: str | Path | None = None,
    relative: bool = True,
) -> list[dict]:
    """Build manifest rows for a single keyword shared across all audio files.

    When ``relative`` is set and ``manifest_dir`` is provided, ``audio_path`` is
    written relative to the manifest directory so it round-trips through
    :func:`load_manifest` (which resolves relative paths against the manifest's
    parent). Paths that cannot be made relative fall back to absolute.
    """
    if not keyword:
        raise ValueError("keyword is required to build manifest rows")

    base_dir = Path(manifest_dir).resolve() if manifest_dir is not None else None
    rows: list[dict] = []
    for audio_path in audio_paths:
        resolved = Path(audio_path).resolve()
        if relative and base_dir is not None:
            try:
                written_path = str(resolved.relative_to(base_dir))
            except ValueError:
                written_path = str(resolved)
        else:
            written_path = str(resolved)

        row: dict = {"audio_path": written_path, "keyword": keyword}
        if label is not None:
            row["label"] = int(label)
        rows.append(row)
    return rows


def build_manifest_rows_by_filename(
    audio_paths: Sequence[str | Path],
    keywords: Sequence[str],
    *,
    keyword_labels: Mapping[str, int] | None = None,
    manifest_dir: str | Path | None = None,
    relative: bool = True,
    skip_unmatched: bool = True,
    casefold: bool = True,
) -> tuple[list[dict], list[str]]:
    """Build manifest rows by assigning each file a keyword from its filename.

    Returns ``(rows, unmatched_paths)``. Labels are assigned by exact keyword
    text through ``keyword_labels``.
    """
    if not keywords:
        raise ValueError("keywords must not be empty")
    if keyword_labels is None:
        raise ValueError("keyword_labels is required for filename assignment mode")

    lookup = _keyword_lookup(keywords, casefold=casefold)
    missing_label_keywords = [keyword for keyword in lookup.values() if keyword not in keyword_labels]
    if missing_label_keywords:
        missing_display = ", ".join(sorted(missing_label_keywords))
        raise ValueError(f"Missing labels for keywords: {missing_display}")

    base_dir = Path(manifest_dir).resolve() if manifest_dir is not None else None
    rows: list[dict] = []
    unmatched: list[str] = []
    for audio_path in audio_paths:
        resolved = Path(audio_path).resolve()
        candidate = filename_keyword_candidate(resolved)
        matched_keyword = lookup.get(normalize_keyword(candidate, casefold=casefold)) if candidate else None
        if matched_keyword is None:
            unmatched.append(str(resolved))
            if skip_unmatched:
                continue
            raise ValueError(f"No keyword matched for file: {resolved}")

        if relative and base_dir is not None:
            try:
                written_path = str(resolved.relative_to(base_dir))
            except ValueError:
                written_path = str(resolved)
        else:
            written_path = str(resolved)

        rows.append(
            {
                "audio_path": written_path,
                "keyword": matched_keyword,
                "label": int(keyword_labels[matched_keyword]),
            }
        )

    if not rows:
        raise ValueError("No files matched any keyword")

    return rows, unmatched


def write_manifest(
    path: str | Path,
    rows: Sequence[dict],
    *,
    manifest_format: str = "auto",
) -> Path:
    """Write manifest ``rows`` to ``path`` as CSV or JSONL.

    ``manifest_format`` of ``"auto"`` infers the format from the file suffix.
    """
    manifest_path = Path(path)
    if not rows:
        raise ValueError("Cannot write an empty manifest")

    resolved_format = manifest_format.lower()
    if resolved_format == "auto":
        suffix = manifest_path.suffix.lower()
        if suffix == ".csv":
            resolved_format = "csv"
        elif suffix == ".jsonl":
            resolved_format = "jsonl"
        else:
            raise ValueError(f"Cannot infer manifest format from suffix: {manifest_path.suffix}")

    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    if resolved_format == "jsonl":
        with manifest_path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        return manifest_path

    if resolved_format == "csv":
        fieldnames = ["audio_path", "keyword"]
        if any("label" in row for row in rows):
            fieldnames.append("label")
        with manifest_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        return manifest_path

    raise ValueError(f"Unsupported manifest format: {manifest_format}")
