from __future__ import annotations

import pytest

from dma_kws.inference.keyword_set import WINDOW_EVAL_PROTOCOL, resolve_keyword_set
from dma_kws.inference.musan_fa import (
    compute_file_metrics,
    false_accept_metrics,
    merge_musan_summaries,
    musan_keyword_set_result_record,
)
from dma_kws.inference.stage2_clip import Stage2ClipRunner
from dma_kws.tokenizer import load_char_tokenizer
from tests.test_stage2_clip import FakeMultiVerifier, _fake_g2p, _fake_phonemes


def _keyword_set(targets):
    tokenizer = load_char_tokenizer("data/dict/lang_char.txt", split_with_space=" ")
    return resolve_keyword_set(
        {
            "keyword": "",
            "keyword_phonemes": "",
            "keywords": [],
            "keyword_eval": {
                "mode": "any",
                "query_batch_size": 8,
                "targets": targets,
            },
        },
        tokenizer,
        g2p=_fake_g2p(),
        tokenizer_dict_path="data/dict/lang_char.txt",
    )


def _window_record(*, path, score, window_index, keyword_set, detected=None):
    threshold = 0.5
    detected = bool(score >= threshold) if detected is None else detected
    runner_result = {
        "result_schema_version": 2,
        "eval_protocol": WINDOW_EVAL_PROTOCOL,
        "keyword_eval_mode": "any",
        "keyword_set_id": keyword_set.keyword_set_id,
        "qbyt_raw_logit": 1.0,
        "qbyt_score": score,
        "threshold": threshold,
        "detected": detected,
        "skipped": False,
        "skip_reason": None,
        "best_keyword": keyword_set.texts[0],
        "best_pronunciation_id": keyword_set.keywords[0].pronunciations[0].pronunciation_id,
        "matched_keywords": list(keyword_set.texts) if detected else [],
        "keyword_results": [
            {
                "keyword_id": keyword.keyword_id,
                "text": keyword.text,
                "qbyt_raw_logit": 1.0,
                "qbyt_score": score,
                "detected": detected,
                "best_pronunciation_id": keyword.pronunciations[0].pronunciation_id,
                "pronunciation_results": [
                    {
                        "pronunciation_id": pron.pronunciation_id,
                        "phonemes": list(pron.phonemes),
                        "token_ids": list(pron.token_ids),
                        "qbyt_raw_logit": 1.0,
                        "qbyt_score": score,
                        "detected": detected,
                    }
                    for pron in keyword.pronunciations
                ],
            }
            for keyword in keyword_set.keywords
        ],
        "clip_span_sec": {
            "start_sec": float(window_index),
            "end_sec": float(window_index) + 3.0,
        },
        "window_index": window_index,
    }
    return musan_keyword_set_result_record(
        path,
        "noise",
        runner_result,
        window_index=window_index,
    )


def test_one_window_two_keywords_counts_one_fp():
    keyword_set = _keyword_set(
        [
            {"text": "hey eva", "pronunciations": ["HH EY1 IY1 V AH0"]},
            {"text": "ok lamp", "pronunciations": ["OW1 K EY1 L AE1 M P"]},
        ]
    )
    record = _window_record(
        path="/musan/noise/a.wav",
        score=0.9,
        window_index=0,
        keyword_set=keyword_set,
    )
    assert record["matched_keywords"] == ["hey eva", "ok lamp"]
    metrics = false_accept_metrics(
        [record],
        threshold=0.5,
        total_hours=1.0,
    )
    assert metrics["fp"] == 1.0
    assert metrics["fa_per_hour"] == pytest.approx(1.0)
    assert metrics["total_hours"] == pytest.approx(1.0)


def test_multi_window_same_file_counts_one_triggered_file():
    keyword_set = _keyword_set(
        [{"text": "hey eva", "pronunciations": ["HH EY1 IY1 V AH0"]}]
    )
    path = "/musan/noise/a.wav"
    results = [
        _window_record(path=path, score=0.9, window_index=0, keyword_set=keyword_set),
        _window_record(path=path, score=0.8, window_index=1, keyword_set=keyword_set),
    ]
    source_files = [
        {"path": path, "subset": "noise", "duration": 6.0, "scored_window_count": 2}
    ]
    file_metrics = compute_file_metrics(results, source_files, threshold=0.5)
    assert file_metrics["num_triggered_files"] == 1
    assert file_metrics["num_scored_files"] == 1
    assert file_metrics["file_trigger_rate"] == pytest.approx(1.0)
    metrics = false_accept_metrics(results, threshold=0.5, total_hours=1.0)
    assert metrics["fp"] == 2.0


def test_merge_recomputes_file_metrics_and_keeps_zero_window_files(tmp_path):
    keyword_set = _keyword_set(
        [{"text": "hey eva", "pronunciations": ["HH EY1 IY1 V AH0"]}]
    )
    identity = {
        "eval_protocol": WINDOW_EVAL_PROTOCOL,
        "keyword_eval_mode": "any",
        "keyword_set_id": keyword_set.keyword_set_id,
        "score_semantics": "max_over_keywords_and_pronunciations",
        "stage2_ckpt": "a.pt",
        "window_sec": 3.0,
        "hop_sec": 3.0,
        "musan_root": "/musan",
        "musan_catalog_sha256": "abc",
        "stream": {"mode": "test"},
        "provenance": {"schema_version": 4, "keyword_eval": {"mode": "any"}},
        "amp": "off",
        "fbank_windows": "independent",
        "batch_size": 16,
    }
    window = _window_record(
        path="/musan/noise/a.wav",
        score=0.9,
        window_index=0,
        keyword_set=keyword_set,
    )
    shard0 = {
        **identity,
        "total_hours": 1.0,
        "total_files": 2,
        "source_files": [
            {
                "path": "/musan/noise/a.wav",
                "subset": "noise",
                "duration": 3600.0,
                "scored_window_count": 1,
            },
            {
                "path": "/musan/noise/short.wav",
                "subset": "noise",
                "duration": 1.0,
                "scored_window_count": 0,
            },
        ],
    }
    shard1 = {
        **identity,
        "total_hours": 0.5,
        "total_files": 1,
        "source_files": [
            {
                "path": "/musan/music/empty.wav",
                "subset": "music",
                "duration": 1800.0,
                "scored_window_count": 0,
            }
        ],
    }
    merged, results = merge_musan_summaries(
        [shard0, shard1],
        [[window], []],
        output_dir=tmp_path / "merged",
        threshold=0.5,
    )
    assert merged["total_hours"] == pytest.approx(1.5)
    assert merged["file_metrics"]["total_files"] == 3
    assert merged["file_metrics"]["num_scored_files"] == 1
    assert merged["file_metrics"]["num_skipped_files"] == 2
    assert merged["file_metrics"]["num_triggered_files"] == 1
    assert merged["metrics"]["fp"] == 1.0
    assert len(results) == 1
    assert len(merged["source_files"]) == 3


def test_merge_rejects_mixed_keyword_set_id(tmp_path):
    first = _keyword_set(
        [{"text": "hey eva", "pronunciations": ["HH EY1 IY1 V AH0"]}]
    )
    second = _keyword_set(
        [
            {
                "text": "hey eva",
                "pronunciations": ["HH EY1 IY1 V AH0", "HH EY1 EY1 V AH0"],
            }
        ]
    )
    base = {
        "eval_protocol": WINDOW_EVAL_PROTOCOL,
        "keyword_eval_mode": "any",
        "score_semantics": "max_over_keywords_and_pronunciations",
        "stage2_ckpt": "a.pt",
        "window_sec": 3.0,
        "hop_sec": 3.0,
        "musan_root": "/musan",
        "musan_catalog_sha256": "abc",
        "stream": {"mode": "test"},
        "provenance": {"schema_version": 4},
        "amp": "off",
        "fbank_windows": "independent",
        "batch_size": 16,
        "total_hours": 1.0,
        "total_files": 1,
    }
    with pytest.raises(ValueError, match="keyword_set_id"):
        merge_musan_summaries(
            [
                {**base, "keyword_set_id": first.keyword_set_id},
                {**base, "keyword_set_id": second.keyword_set_id},
            ],
            [[], []],
            output_dir=tmp_path,
            threshold=0.5,
        )


def test_score_window_features_multi_uses_shared_multi_scorer(monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchaudio")
    monkeypatch.setattr("dma_kws.inference.stage2_clip.make_g2p", _fake_g2p)
    monkeypatch.setattr(
        "dma_kws.inference.stage2_clip.text_to_phonemes",
        lambda _g2p, text: _fake_phonemes(text),
    )
    tokenizer = load_char_tokenizer("data/dict/lang_char.txt", split_with_space=" ")
    keyword_set = _keyword_set(
        [
            {"text": "hey eva", "pronunciations": ["HH EY1 IY1 V AH0"]},
            {"text": "ok lamp", "pronunciations": ["OW1 K EY1 L AE1 M P"]},
        ]
    )
    verifier = FakeMultiVerifier(scored=[(2.0, 0.9), (0.2, 0.2)] * 8)
    runner = Stage2ClipRunner(
        verifier=verifier,
        tokenizer=tokenizer,
        demo_cfg={"qbyt_threshold": 0.5},
        sample_rate=16000,
    )
    feats = [torch.zeros(20, 80), torch.zeros(20, 80)]
    spans = [(0, 0.0, 3.0), (1, 3.0, 6.0)]
    results = runner.score_window_features_multi(
        "/musan/noise/a.wav",
        feats,
        spans,
        keyword_set,
        batch_size=8,
    )
    assert len(results) == 2
    assert results[0]["eval_protocol"] == WINDOW_EVAL_PROTOCOL
    assert results[0]["detected"] is True
    assert results[0]["window_index"] == 0
    assert "keyword" not in results[0]
    assert verifier.multi_calls == 1
