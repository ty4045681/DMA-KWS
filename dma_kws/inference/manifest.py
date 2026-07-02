"""Manifest loading for batch two-stage evaluation."""

from __future__ import annotations

import csv
import json
from pathlib import Path


_REQUIRED_COLUMNS = ("audio_path", "keyword")


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
