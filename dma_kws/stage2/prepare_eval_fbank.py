"""Precompute LibriPhrase eval fbank ``.npy`` files next to eval ``.wav`` clips."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable

from dma_kws.stage2.dataset import _DEFAULT_EVAL_CSV
from dma_kws.stage2.prepare_paper import compute_fbank_for_clip


def iter_eval_wav_files(test_dir: Path) -> Iterable[Path]:
    """Yield all ``.wav`` files under a LibriPhrase eval ``test_dir``."""
    return sorted(test_dir.rglob("*.wav"))


def collect_eval_wav_relpaths(test_dir: Path, csv_files: list[str]) -> set[str]:
    """Collect unique ``anchor`` and ``comparison`` wav paths referenced by eval CSVs."""
    import pandas as pd

    rel_paths: set[str] = set()
    for rel_csv in csv_files:
        csv_path = test_dir / rel_csv
        if not csv_path.is_file():
            raise FileNotFoundError(f"Eval CSV not found: {csv_path}")

        df = pd.read_csv(csv_path)
        for column in ("anchor", "comparison"):
            if column not in df.columns:
                raise ValueError(f"Eval CSV {csv_path} missing required column: {column}")
            for value in df[column].dropna().astype(str):
                rel_paths.add(value)

    return rel_paths


def resolve_eval_wav_path(test_dir: Path, rel_wav: str) -> Path:
    return test_dir / rel_wav


def resolve_eval_fbank_path_from_wav(wav_path: Path) -> Path:
    return wav_path.with_suffix(".npy")


def prepare_eval_fbank(
    test_dir: Path | str,
    *,
    wav_paths: Iterable[Path] | None = None,
    skip_existing: bool = True,
    limit: int = 0,
    num_mel_bins: int = 80,
    frame_length: int = 25,
    frame_shift: int = 10,
    dither: float = 0.1,
    window_type: str = "povey",
    log_interval: int = 1000,
    compute_fn: Callable[..., str] = compute_fbank_for_clip,
) -> tuple[int, int, int]:
    """Convert eval wav clips to co-located fbank ``.npy`` files.

    Returns ``(written, skipped, failed)`` counts.
    """
    test_dir = Path(test_dir)
    if not test_dir.is_dir():
        raise FileNotFoundError(f"Eval test_dir not found: {test_dir}")

    if wav_paths is None:
        wav_paths = iter_eval_wav_files(test_dir)

    written = 0
    skipped = 0
    failed = 0

    for index, wav_path in enumerate(wav_paths, start=1):
        if limit and index > limit:
            break

        wav_path = Path(wav_path)
        npy_path = resolve_eval_fbank_path_from_wav(wav_path)

        if skip_existing and npy_path.is_file():
            skipped += 1
            continue

        if not wav_path.is_file():
            failed += 1
            continue

        try:
            npy_path.parent.mkdir(parents=True, exist_ok=True)
            compute_fn(
                wav_path,
                npy_path,
                num_mel_bins=num_mel_bins,
                frame_length=frame_length,
                frame_shift=frame_shift,
                dither=dither,
                window_type=window_type,
            )
            written += 1
        except Exception:
            failed += 1
            continue

        if log_interval and written % log_interval == 0:
            print(f"Wrote {written} eval fbank files (skipped={skipped}, failed={failed})")

    return written, skipped, failed


def prepare_eval_fbank_from_csv(
    test_dir: Path | str,
    *,
    csv_files: list[str] | None = None,
    skip_existing: bool = True,
    limit: int = 0,
    num_mel_bins: int = 80,
    frame_length: int = 25,
    frame_shift: int = 10,
    dither: float = 0.1,
    window_type: str = "povey",
    log_interval: int = 1000,
    compute_fn: Callable[..., str] = compute_fbank_for_clip,
) -> tuple[int, int, int]:
    """Convert only wav files referenced by LibriPhrase eval CSV manifests."""
    test_dir = Path(test_dir)
    csv_files = list(csv_files or _DEFAULT_EVAL_CSV)
    rel_paths = collect_eval_wav_relpaths(test_dir, csv_files)
    wav_paths = [resolve_eval_wav_path(test_dir, rel_path) for rel_path in sorted(rel_paths)]
    if limit:
        wav_paths = wav_paths[:limit]
    return prepare_eval_fbank(
        test_dir,
        wav_paths=wav_paths,
        skip_existing=skip_existing,
        limit=0,
        num_mel_bins=num_mel_bins,
        frame_length=frame_length,
        frame_shift=frame_shift,
        dither=dither,
        window_type=window_type,
        log_interval=log_interval,
        compute_fn=compute_fn,
    )
