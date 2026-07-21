"""Reclassify cached WeNet transcripts and rebuild filtered MUSAN trees."""
from __future__ import annotations

import json
import math
import shutil
import tempfile
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from rapidfuzz import fuzz

from dma_kws.inference.musan_asr_filter import normalize_match_text, validate_keywords


def _candidates(transcript: str, keyword: str) -> list[str]:
    text = normalize_match_text(transcript)
    target = normalize_match_text(keyword)
    if not text or not target:
        return []
    if len(target) < 4:
        return [target] if target in text else []
    minimum = max(4, math.ceil(len(target) * 0.7))
    maximum = math.ceil(len(target) * 1.4)
    return [
        text[start:end]
        for start in range(len(text))
        for end in range(start + minimum, min(len(text), start + maximum) + 1)
    ]


def bounded_keyword_match(transcript: str, keywords: Sequence[str]) -> tuple[str | None, str | None, float]:
    """Return the best length-bounded keyword match for one ASR transcript."""
    best_keyword: str | None = None
    best_candidate: str | None = None
    best_score = 0.0
    for keyword in keywords:
        target = normalize_match_text(keyword)
        for candidate in _candidates(transcript, keyword):
            score = float(fuzz.ratio(target, candidate))
            if score > best_score:
                best_keyword, best_candidate, best_score = keyword, candidate, score
    return best_keyword, best_candidate, best_score


def classify_score(score: float, *, remove_threshold: float, review_threshold: float) -> str:
    if not 0 <= review_threshold < remove_threshold <= 100:
        raise ValueError("require 0 <= review_threshold < remove_threshold <= 100")
    if score >= remove_threshold:
        return "remove"
    if score >= review_threshold:
        return "review"
    return "keep"


def rescore_records(
    records: Iterable[Mapping],
    keywords: Sequence[str],
    *,
    remove_threshold: float = 92.0,
    review_threshold: float = 78.0,
) -> list[dict]:
    keyword_values = validate_keywords(keywords)
    results: list[dict] = []
    for source_record in records:
        record = dict(source_record)
        windows: list[dict] = []
        best_keyword = best_candidate = None
        best_score = 0.0
        for source_window in record.get("windows", []):
            window = dict(source_window)
            keyword, candidate, score = bounded_keyword_match(str(window.get("transcript", "")), keyword_values)
            decision = classify_score(score, remove_threshold=remove_threshold, review_threshold=review_threshold)
            window.update({"best_keyword": keyword, "best_candidate": candidate, "match_score": score, "decision": decision})
            windows.append(window)
            if score > best_score:
                best_keyword, best_candidate, best_score = keyword, candidate, score
        decision = classify_score(best_score, remove_threshold=remove_threshold, review_threshold=review_threshold)
        record.update({"windows": windows, "decision": decision, "matched": decision == "remove", "matched_keyword": best_keyword if decision != "keep" else None, "best_candidate": best_candidate, "best_match_score": best_score})
        results.append(record)
    return results


def _copy_tree(source: Path, destination: Path, paths: set[Path], audio_paths: set[Path]) -> None:
    if destination.exists():
        raise FileExistsError(f"Output path already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    staged = temporary / destination.name
    def ignore(directory: str, names: list[str]) -> set[str]:
        rel = Path(directory).resolve().relative_to(source)
        return {name for name in names if rel / name in audio_paths and rel / name not in paths}
    try:
        shutil.copytree(source, staged, symlinks=True, copy_function=shutil.copy2, ignore=ignore)
        staged.replace(destination)
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


def load_results(path: str | Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    if not records:
        raise ValueError(f"No records in results file: {path}")
    if any("relative_path" not in record or "windows" not in record for record in records):
        raise ValueError("results.jsonl must contain relative_path and windows fields")
    return records


def rebuild_from_results(*, musan_root: str | Path, results_path: str | Path, output_root: str | Path, review_root: str | Path, report_dir: str | Path, keywords: Sequence[str], remove_threshold: float = 92.0, review_threshold: float = 78.0) -> dict:
    source = Path(musan_root).expanduser().resolve()
    if not source.is_dir():
        raise NotADirectoryError(source)
    output = Path(output_root).expanduser().resolve()
    review = Path(review_root).expanduser().resolve()
    report = Path(report_dir).expanduser().resolve()
    if len({output, review, report}) != 3 or any(path.exists() for path in (output, review, report)):
        raise ValueError("output_root, review_root, and report_dir must be distinct paths that do not exist")
    records = rescore_records(load_results(results_path), keywords, remove_threshold=remove_threshold, review_threshold=review_threshold)
    by_decision: dict[str, set[Path]] = defaultdict(set)
    for record in records:
        by_decision[str(record["decision"])].add(Path(str(record["relative_path"])))
    audio_paths = {Path(str(record["relative_path"])) for record in records}
    _copy_tree(source, output, by_decision["keep"], audio_paths)
    _copy_tree(source, review, by_decision["review"], audio_paths)
    report.mkdir(parents=True)
    results_out = report / "results.jsonl"
    with results_out.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    counts = Counter(record["decision"] for record in records)
    summary = {"source_results": str(Path(results_path).resolve()), "musan_root": str(source), "output_root": str(output), "review_root": str(review), "keywords": list(keywords), "remove_threshold": remove_threshold, "review_threshold": review_threshold, "counts": dict(counts), "results": str(results_out.resolve())}
    with (report / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    return summary
