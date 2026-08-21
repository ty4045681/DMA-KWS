#!/usr/bin/env python3
"""Build a positive-only ``hey eva`` evaluation manifest from an audio folder."""

from __future__ import annotations

import argparse
import csv
import os
import sys
import tempfile
from pathlib import Path
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dma_kws.g2p import HEY_EVA_PHONEMES
from dma_kws.inference.manifest import iter_audio_files


KEYWORD = "hey eva"
LABEL = 1
KEYWORD_PHONEMES = " ".join(HEY_EVA_PHONEMES)
FIELDNAMES = ("audio_path", "keyword", "label", "keyword_phonemes")


def resolve_output_path(audio_dir: Path, output_name: str | None = None) -> Path:
    """Return a CSV path beside ``audio_dir``, never inside it."""

    name = output_name or f"{audio_dir.name}.csv"
    name_path = Path(name)
    if name_path.name != name or name_path.suffix.lower() != ".csv":
        raise ValueError("--output-name must be a CSV filename, not a path")
    return audio_dir.parent / name


def build_rows(audio_dir: Path, *, recursive: bool = True) -> list[dict[str, object]]:
    """Collect audio files and write paths relative to the future CSV directory."""

    audio_files = iter_audio_files(audio_dir, recursive=recursive)
    if not audio_files:
        raise ValueError(f"No supported audio files found under {audio_dir}")

    manifest_dir = audio_dir.parent
    return [
        {
            "audio_path": path.relative_to(manifest_dir).as_posix(),
            "keyword": KEYWORD,
            "label": LABEL,
            "keyword_phonemes": KEYWORD_PHONEMES,
        }
        for path in audio_files
    ]


def write_manifest(
    audio_dir: str | Path,
    *,
    output_name: str | None = None,
    recursive: bool = True,
    force: bool = False,
) -> tuple[Path, int]:
    """Write the sibling CSV atomically and return ``(path, row_count)``."""

    resolved_audio_dir = Path(audio_dir).expanduser().resolve()
    if not resolved_audio_dir.is_dir():
        raise NotADirectoryError(f"Audio directory not found: {resolved_audio_dir}")

    output_path = resolve_output_path(resolved_audio_dir, output_name)
    if output_path.exists() and not force:
        raise FileExistsError(
            f"Output already exists: {output_path}; pass --force to replace it"
        )

    rows = build_rows(resolved_audio_dir, recursive=recursive)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            dir=output_path.parent,
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary_path, output_path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

    return output_path, len(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a positive-only hey-eva CSV beside an audio directory. "
            "Supported formats: WAV, FLAC, MP3 and M4A."
        )
    )
    parser.add_argument("audio_dir", help="Directory containing the audio files")
    parser.add_argument(
        "--output-name",
        help="Sibling CSV filename (default: <audio-directory-name>.csv)",
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Only include audio files directly inside the directory",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace the output CSV if it already exists",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        output_path, count = write_manifest(
            args.audio_dir,
            output_name=args.output_name,
            recursive=not args.no_recursive,
            force=args.force,
        )
    except (FileExistsError, NotADirectoryError, OSError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    print(f"Wrote {count} rows to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
