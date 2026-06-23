"""JSONL reading helpers shared across DMA-KWS scripts."""

from __future__ import annotations

import json
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    """Read a JSONL file into a list of dicts, skipping blank lines."""
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
