#!/usr/bin/env python3
"""Select LoRA hard negatives from ``eval_stage2_clips.py`` results."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from dma_kws.inference.score_provenance import (
    semantic_score_provenance,
    validate_score_provenance,
)
from dma_kws.stage2.hard_negative_mining import (
    merge_hard_negative_manifest_rows,
    select_hard_negatives,
)


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(value)
    return rows


def _read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _validate_comparable_results(paths: list[Path]) -> None:
    """Reject score files produced with different model/input semantics."""

    if len(paths) < 2:
        return
    canonical: str | None = None
    for path in paths:
        summary_path = path.parent / "summary.json"
        if not summary_path.is_file():
            raise ValueError(
                f"Multiple --results require each sibling summary.json for provenance: "
                f"missing {summary_path}"
            )
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if not isinstance(summary, dict):
            raise ValueError(f"{summary_path} must contain a JSON object")
        provenance = summary.get("provenance")
        if provenance is None:
            raise ValueError(f"{summary_path} has no provenance mapping")
        provenance = validate_score_provenance(
            provenance,
            source=summary_path,
        )
        value = json.dumps(
            semantic_score_provenance(provenance),
            sort_keys=True,
            separators=(",", ":"),
        )
        if canonical is None:
            canonical = value
        elif value != canonical:
            raise ValueError(
                "Cannot mix hard-negative scores from different checkpoint/readout/"
                "fbank/padding provenance"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--base-manifest",
        type=Path,
        help="Optional real_source.csv; output becomes a prepare-ready train+eval manifest",
    )
    parser.add_argument("--keyword", default="hey eva")
    parser.add_argument("--min-score", type=float, default=0.5)
    parser.add_argument("--top-k", type=int, default=1000)
    parser.add_argument("--per-speaker-cap", type=int, default=20)
    parser.add_argument("--allowed-split", action="append")
    args = parser.parse_args()

    _validate_comparable_results(args.results)
    scored_rows = [row for path in args.results for row in _read_jsonl(path)]
    selected = select_hard_negatives(
        scored_rows,
        keyword=args.keyword,
        min_score=args.min_score,
        top_k=args.top_k,
        per_speaker_cap=args.per_speaker_cap,
        allowed_splits=frozenset(args.allowed_split or ["train"]),
    )
    base_rows = _read_csv(args.base_manifest) if args.base_manifest else []
    output_rows = (
        merge_hard_negative_manifest_rows(base_rows, selected)
        if args.base_manifest
        else selected
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    preferred_fields = [
        "audio_path",
        "text",
        "keyword",
        "label",
        "phase",
        "split",
        "speaker_id",
        "device",
        "session",
        "source",
        "negative_type",
        "qbyt_score",
    ]
    fieldnames = preferred_fields + sorted(
        {key for row in output_rows for key in row} - set(preferred_fields)
    )
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)
    print(
        json.dumps(
            {
                "scored_rows": len(scored_rows),
                "selected": len(selected),
                "base_rows": len(base_rows),
                "output_rows": len(output_rows),
                "output": str(args.output.resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
