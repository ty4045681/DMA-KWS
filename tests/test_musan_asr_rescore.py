from __future__ import annotations

import json
from pathlib import Path

from dma_kws.inference.musan_asr_rescore import bounded_keyword_match, rebuild_from_results, rescore_records


def test_bounded_match_rejects_single_character_false_positives():
    for transcript in ("I", "A", "THE", ""):
        keyword, candidate, score = bounded_keyword_match(transcript, ["Hey Siri", "Hey Tesla"])
        assert (keyword, candidate, score) == (None, None, 0.0)


def test_bounded_match_classifies_keep_review_remove():
    records = [
        {"relative_path": "music/a.wav", "windows": [{"transcript": "I"}]},
        {"relative_path": "music/b.wav", "windows": [{"transcript": "hey siri"}]},
        {"relative_path": "music/c.wav", "windows": [{"transcript": "hey sirry"}]},
    ]
    rescored = rescore_records(records, ["Hey Siri"], remove_threshold=95, review_threshold=70)

    assert [record["decision"] for record in rescored] == ["keep", "remove", "review"]
    assert rescored[1]["best_candidate"] == "heysiri"


def test_rebuild_from_results_uses_cached_transcripts(tmp_path):
    source = tmp_path / "musan"
    (source / "music").mkdir(parents=True)
    for name in ("keep.wav", "remove.wav", "review.wav"):
        (source / "music" / name).write_bytes(name.encode())
    results = tmp_path / "old-results.jsonl"
    rows = [
        {"relative_path": "music/keep.wav", "windows": [{"transcript": "I"}]},
        {"relative_path": "music/remove.wav", "windows": [{"transcript": "HEY SIRI"}]},
        {"relative_path": "music/review.wav", "windows": [{"transcript": "HEY SIRRY"}]},
    ]
    results.write_text("".join(json.dumps(row) + "\n" for row in rows))

    summary = rebuild_from_results(
        musan_root=source,
        results_path=results,
        output_root=tmp_path / "filtered",
        review_root=tmp_path / "review",
        report_dir=tmp_path / "report",
        keywords=["Hey Siri"],
        remove_threshold=95,
        review_threshold=70,
    )

    assert (tmp_path / "filtered" / "music" / "keep.wav").is_file()
    assert not (tmp_path / "filtered" / "music" / "remove.wav").exists()
    assert not (tmp_path / "filtered" / "music" / "review.wav").exists()
    assert (tmp_path / "review" / "music" / "review.wav").is_file()
    assert summary["counts"] == {"keep": 1, "remove": 1, "review": 1}
