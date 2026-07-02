"""Manifest loading and generation for batch two-stage evaluation."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable, Sequence


_REQUIRED_COLUMNS = ("audio_path", "keyword")
_DEFAULT_AUDIO_EXTENSIONS = (".wav", ".flac", ".mp3", ".m4a")


def _normalize_row(row: dict[str, str], base_dir: Path | None) -> dict:
    normalized = {key.strip(): (value.strip() if isinstance(value, str) else value) for key, value in row.items()}
    audio_path = normalized.get("audio_path", "")
    if audio_path and base_dir is not None and not Path(audio_path).is_absolute():
        normalized["audio_path"] = str((base_dir / audio_path).resolve())

    if "label" in normalized and normalized["label"] != "":
        normalized["label"] = int(normalized["label"])
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
                row = _normalize_row(json.loads(line), base_dir)
                _validate_row(row, line_number)
                rows.append(row)
        return rows

    if suffix == ".csv":
        with manifest_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for line_number, row in enumerate(reader, start=2):
                normalized = _normalize_row(row, base_dir)
                _validate_row(normalized, line_number)
                rows.append(normalized)
        return rows

    raise ValueError(f"Unsupported manifest format: {manifest_path.suffix}")


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
