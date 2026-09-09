from __future__ import annotations

import json
import pytest

from dma_kws.inference.keyword_set import (
    keyword_eval_provenance_block,
    resolve_keyword_set,
    semantic_keyword_eval,
)
from dma_kws.inference.score_provenance import (
    LEGACY_PROVENANCE_SCHEMA_VERSION,
    PROVENANCE_SCHEMA_VERSION,
    semantic_score_provenance,
    validate_score_provenance,
)
from dma_kws.tokenizer import load_char_tokenizer
from scripts.fit_stage2_calibration import load_labeled_scores
from scripts.scan_stage2_thresholds import load_scan_input
from tests.test_keyword_set import FakeG2P, HEY_EVA_A, HEY_EVA_B


def _tokenizer():
    return load_char_tokenizer("data/dict/lang_char.txt", split_with_space=" ")


def _keyword_set(pronunciations):
    return resolve_keyword_set(
        {
            "keyword": "",
            "keyword_phonemes": "",
            "keywords": [],
            "keyword_eval": {
                "mode": "any",
                "query_batch_size": 8,
                "targets": [{"text": "hey eva", "pronunciations": list(pronunciations)}],
            },
        },
        _tokenizer(),
        g2p=FakeG2P({"hey eva": HEY_EVA_A.split()}),
        tokenizer_dict_path="data/dict/lang_char.txt",
    )


def _schema3_payload() -> dict:
    return {
        "schema_version": 3,
        "checkpoint": {"path": "/ckpt.pt", "size_bytes": 1, "sha256": "a" * 64},
        "calibration": {"type": "identity_logit_sigmoid"},
        "qbyt_alignment": {"topology": "keyword_filler_segmental_crf_v1"},
        "stream": {"mode": "test"},
        "audio_padding_ms": {"left": 0, "right": 0},
        "fbank": {"num_mel_bins": 80},
        "tokenizer": {
            "path": "/dict.txt",
            "size_bytes": 1,
            "sha256": "b" * 64,
            "split_with_space": " ",
        },
        "sequence_objective": {
            "target_mode": "ordered_contiguous_prefix",
            "progress_weight": 0.3,
            "normalization": "sample",
        },
    }


def test_validate_accepts_schema_3_and_schema_4_per_row():
    schema3 = validate_score_provenance(_schema3_payload())
    assert schema3["schema_version"] == LEGACY_PROVENANCE_SCHEMA_VERSION
    payload = _schema3_payload()
    payload["schema_version"] = 4
    payload["keyword_eval"] = {"mode": "per_row"}
    schema4 = validate_score_provenance(payload)
    assert schema4["schema_version"] == PROVENANCE_SCHEMA_VERSION
    semantic3 = semantic_score_provenance(schema3)
    semantic4 = semantic_score_provenance(schema4)
    assert semantic3["keyword_eval"] == {"mode": "per_row"}
    assert semantic4["keyword_eval"] == {"mode": "per_row"}


def test_schema_3_is_not_interpreted_as_any():
    payload = _schema3_payload()
    validated = validate_score_provenance(payload)
    assert semantic_keyword_eval(
        validated.get("keyword_eval"),
        schema_version=3,
    ) == {"mode": "per_row"}


def test_unknown_schema_is_rejected():
    payload = _schema3_payload()
    payload["schema_version"] = 5
    with pytest.raises(ValueError, match="unsupported score provenance version"):
        validate_score_provenance(payload)


def test_schema_4_any_requires_keyword_eval_fields():
    payload = _schema3_payload()
    payload["schema_version"] = 4
    payload["keyword_eval"] = {"mode": "any"}
    with pytest.raises(ValueError, match="missing"):
        validate_score_provenance(payload)


def test_keyword_set_id_stable_under_reorder_and_duplicate():
    first = _keyword_set([HEY_EVA_B, HEY_EVA_A, HEY_EVA_A])
    second = _keyword_set([HEY_EVA_A, HEY_EVA_B])
    assert first.keyword_set_id == second.keyword_set_id
    third = _keyword_set([HEY_EVA_A])
    assert first.keyword_set_id != third.keyword_set_id


def test_new_writes_use_schema_4_keyword_eval():
    assert PROVENANCE_SCHEMA_VERSION == 4
    payload = _schema3_payload()
    payload["schema_version"] = 4
    payload["keyword_eval"] = {"mode": "per_row"}
    assert validate_score_provenance(payload)["keyword_eval"] == {"mode": "per_row"}
    keyword_set = _keyword_set([HEY_EVA_A, HEY_EVA_B])
    payload["keyword_eval"] = keyword_eval_provenance_block(keyword_set, mode="any")
    validated = validate_score_provenance(payload)
    assert validated["keyword_eval"]["mode"] == "any"
    assert validated["keyword_eval"]["keyword_set_id"] == keyword_set.keyword_set_id
    assert "query_batch_size" not in validated["keyword_eval"]


def test_scan_uses_root_score_and_ignores_saved_detected(tmp_path):
    keyword_set = _keyword_set([HEY_EVA_A, HEY_EVA_B])
    results = tmp_path / "results.jsonl"
    summary = tmp_path / "summary.json"
    rows = [
        {
            "audio_path": "a.wav",
            "label": 1,
            "qbyt_score": 0.7,
            "detected": True,
            "matched_keywords": ["hey eva"],
            "skipped": False,
            "keyword_eval_mode": "any",
            "keyword_set_id": keyword_set.keyword_set_id,
            "eval_protocol": "stage2_clip_keyword_set",
            "keyword_results": [
                {"text": "hey eva", "qbyt_score": 0.7, "pronunciation_results": [{}, {}]},
            ],
        },
        {
            "audio_path": "b.wav",
            "label": 0,
            "qbyt_score": 0.6,
            "detected": True,
            "matched_keywords": ["hey eva"],
            "skipped": False,
            "keyword_eval_mode": "any",
            "keyword_set_id": keyword_set.keyword_set_id,
            "eval_protocol": "stage2_clip_keyword_set",
        },
    ]
    results.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    summary.write_text(
        json.dumps(
            {
                "keyword_eval_mode": "any",
                "keyword_set_id": keyword_set.keyword_set_id,
                "eval_protocol": "stage2_clip_keyword_set",
                "score_semantics": "max_over_keywords_and_pronunciations",
            }
        ),
        encoding="utf-8",
    )
    scan = load_scan_input(tmp_path, mode="clips")
    assert scan.scores.tolist() == [0.7, 0.6]
    assert scan.keyword_eval_mode == "any"
    assert scan.num_skipped == 0
    # Nested pronunciation_results must not add extra samples.
    assert scan.scores.size == 2
    from scripts.scan_stage2_thresholds import scan_thresholds
    import numpy as np

    arrays, _workers = scan_thresholds(scan, np.array([0.8, 0.5], dtype=np.float64))
    # Saved detected=true at 0.6 must not count as a hit when scanning 0.8.
    high = int(np.where(arrays["threshold"] == 0.8)[0][0])
    low = int(np.where(arrays["threshold"] == 0.5)[0][0])
    assert int(arrays["tp"][high]) == 0
    assert int(arrays["fp"][high]) == 0
    assert int(arrays["tp"][low]) == 1
    assert int(arrays["fp"][low]) == 1


def test_scan_rejects_mixed_keyword_sets(tmp_path):
    results = tmp_path / "results.jsonl"
    results.write_text(
        json.dumps(
            {
                "audio_path": "a.wav",
                "label": 0,
                "qbyt_score": 0.1,
                "skipped": False,
                "keyword_eval_mode": "any",
                "keyword_set_id": "aaa",
            }
        )
        + "\n"
        + json.dumps(
            {
                "audio_path": "b.wav",
                "label": 0,
                "qbyt_score": 0.2,
                "skipped": False,
                "keyword_eval_mode": "any",
                "keyword_set_id": "bbb",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "summary.json").write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit, match="keyword_set_id"):
        load_scan_input(tmp_path, mode="clips")


def test_scan_rejects_per_row_summary_hours_for_any_results(tmp_path):
    keyword_set = _keyword_set([HEY_EVA_A])
    (tmp_path / "results.jsonl").write_text(
        json.dumps(
            {
                "audio_path": "a.wav",
                "label": 0,
                "qbyt_score": 0.9,
                "skipped": False,
                "keyword_eval_mode": "any",
                "keyword_set_id": keyword_set.keyword_set_id,
                "eval_protocol": "stage2_window_keyword_set",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "summary.json").write_text(
        json.dumps(
            {
                "keyword_eval_mode": "per_row",
                "total_hours": 1000.0,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(SystemExit, match="keyword_eval_mode"):
        load_scan_input(tmp_path, mode="musan")


def test_schema3_and_schema4_per_row_semantic_provenance_are_equal():
    schema3 = validate_score_provenance(_schema3_payload())
    schema4 = _schema3_payload()
    schema4["schema_version"] = 4
    schema4["keyword_eval"] = {"mode": "per_row"}
    schema4 = validate_score_provenance(schema4)
    assert semantic_score_provenance(schema3) == semantic_score_provenance(schema4)
    any_payload = _schema3_payload()
    any_payload["schema_version"] = 4
    any_payload["keyword_eval"] = keyword_eval_provenance_block(
        _keyword_set([HEY_EVA_A]),
        mode="any",
    )
    assert semantic_score_provenance(schema3) != semantic_score_provenance(
        validate_score_provenance(any_payload)
    )


def test_fit_rejects_any_mode_rows(tmp_path):
    path = tmp_path / "scores.jsonl"
    path.write_text(
        json.dumps(
            {
                "label": 1,
                "qbyt_raw_logit": 1.2,
                "keyword_eval_mode": "any",
                "eval_protocol": "stage2_clip_keyword_set",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(SystemExit, match="any-mode"):
        load_labeled_scores(path)


def test_scan_any_unlabeled_row_still_errors(tmp_path):
    keyword_set = _keyword_set([HEY_EVA_A])
    (tmp_path / "results.jsonl").write_text(
        json.dumps(
            {
                "audio_path": "a.wav",
                "qbyt_score": 0.4,
                "skipped": False,
                "keyword_eval_mode": "any",
                "keyword_set_id": keyword_set.keyword_set_id,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "summary.json").write_text(
        json.dumps(
            {
                "keyword_eval_mode": "any",
                "keyword_set_id": keyword_set.keyword_set_id,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(SystemExit, match="no label"):
        load_scan_input(tmp_path, mode="clips")


def test_libriphrase_entry_rejects_any_mode(monkeypatch):
    import inspect

    import scripts.eval_stage2_libriphrase as libriphrase
    from omegaconf import OmegaConf

    monkeypatch.setattr(
        libriphrase,
        "resolved_config",
        lambda _cfg: {
            "paths": {},
            "stage1": {},
            "stage2": {},
            "tokenizer": {},
        },
    )
    monkeypatch.setattr(libriphrase, "require_sections", lambda *_a, **_k: None)
    cfg = OmegaConf.create(
        {
            "prep": {
                "keyword_eval": {
                    "mode": "any",
                    "targets": [{"text": "hey eva"}],
                }
            },
            "run": {"device": "cpu"},
        }
    )
    with pytest.raises(SystemExit, match="does not support"):
        inspect.unwrap(libriphrase.main)(cfg)


def test_two_stage_and_libriphrase_reject_any():
    from dma_kws.inference.keyword_set import KeywordEvalConfigError, reject_unsupported_any_mode

    prep = {"keyword_eval": {"mode": "any", "targets": [{"text": "hey eva"}]}}
    with pytest.raises(KeywordEvalConfigError, match="does not support"):
        reject_unsupported_any_mode(prep, entry="scripts/eval_two_stage_kws.py")
    with pytest.raises(KeywordEvalConfigError, match="does not support"):
        reject_unsupported_any_mode(prep, entry="scripts/eval_stage2_libriphrase.py")
