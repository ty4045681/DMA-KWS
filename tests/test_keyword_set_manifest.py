from __future__ import annotations

import json
from pathlib import Path

import pytest

from dma_kws.inference.keyword_set import KeywordSetManifestError
from dma_kws.inference.manifest import load_keyword_set_manifest, load_manifest


TARGETS = ("hey eva", "ok lamp")


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")
    return path


def test_jsonl_keyword_labels_derives_set_label(tmp_path):
    eva = _touch(tmp_path / "audio" / "eva.wav")
    lamp = _touch(tmp_path / "audio" / "lamp.wav")
    both = _touch(tmp_path / "audio" / "both.wav")
    neg = _touch(tmp_path / "audio" / "negative.wav")
    manifest = tmp_path / "eval.jsonl"
    manifest.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "audio_path": "audio/eva.wav",
                        "keyword_labels": {"hey eva": 1, "ok lamp": 0},
                    }
                ),
                json.dumps(
                    {
                        "audio_path": "audio/lamp.wav",
                        "keyword_labels": {"hey eva": 0, "ok lamp": 1},
                    }
                ),
                json.dumps(
                    {
                        "audio_path": "audio/both.wav",
                        "keyword_labels": {"hey eva": 1, "ok lamp": 1},
                    }
                ),
                json.dumps(
                    {
                        "audio_path": "audio/negative.wav",
                        "keyword_labels": {"hey eva": 0, "ok lamp": 0},
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    rows = load_keyword_set_manifest(manifest, TARGETS)
    by_path = {Path(row["audio_path"]).name: row for row in rows}
    assert by_path["eva.wav"]["label"] == 1
    assert by_path["lamp.wav"]["label"] == 1
    assert by_path["both.wav"]["label"] == 1
    assert by_path["negative.wav"]["label"] == 0
    assert by_path["eva.wav"]["keyword_labels"] == {"hey eva": 1, "ok lamp": 0}
    assert by_path["eva.wav"]["label_scope"] == "target_set"
    assert Path(by_path["eva.wav"]["audio_path"]) == eva.resolve()
    assert Path(by_path["lamp.wav"]["audio_path"]) == lamp.resolve()
    assert Path(by_path["both.wav"]["audio_path"]) == both.resolve()
    assert Path(by_path["neg.wav"]["audio_path"] if "neg.wav" in by_path else by_path["negative.wav"]["audio_path"]) == neg.resolve()


def test_csv_keyword_labels_matches_jsonl(tmp_path):
    _touch(tmp_path / "audio" / "eva.wav")
    jsonl = tmp_path / "eval.jsonl"
    csv_path = tmp_path / "eval.csv"
    jsonl.write_text(
        json.dumps(
            {
                "audio_path": "audio/eva.wav",
                "keyword_labels": {"Hey EVA": 1, "ok lamp": 0, "unused": 1},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    csv_path.write_text(
        "audio_path,keyword_labels,label\n"
        'audio/eva.wav,"{""hey eva"":1,""ok lamp"":0,""unused"":1}",1\n',
        encoding="utf-8",
    )
    json_rows = load_keyword_set_manifest(jsonl, TARGETS)
    csv_rows = load_keyword_set_manifest(csv_path, TARGETS)
    assert json_rows[0]["label"] == csv_rows[0]["label"] == 1
    assert json_rows[0]["keyword_labels"] == csv_rows[0]["keyword_labels"]
    assert json_rows[0]["unused_keyword_labels"] == {"unused": 1}


def test_target_set_labels_do_not_create_per_keyword_metrics_fields(tmp_path):
    _touch(tmp_path / "audio" / "neg.wav")
    manifest = tmp_path / "eval.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "audio_path": "audio/neg.wav",
                "label_scope": "target_set",
                "target_texts": ["ok lamp", "hey eva"],
                "label": 0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    rows = load_keyword_set_manifest(manifest, TARGETS)
    assert rows[0]["label"] == 0
    assert rows[0]["label_scope"] == "target_set"
    assert "keyword_labels" not in rows[0]


def test_csv_empty_label_cells_are_unlabeled_and_writable(tmp_path):
    from dma_kws.inference.stage2_reporting import build_keyword_set_result_record

    _touch(tmp_path / "audio" / "eva.wav")
    _touch(tmp_path / "audio" / "open.wav")
    csv_path = tmp_path / "eval.csv"
    csv_path.write_text(
        "audio_path,keyword_labels,label\n"
        'audio/eva.wav,"{""hey eva"":1,""ok lamp"":0}",1\n'
        "audio/open.wav,,\n",
        encoding="utf-8",
    )
    rows = load_keyword_set_manifest(csv_path, TARGETS)
    by_name = {Path(row["audio_path"]).name: row for row in rows}
    assert by_name["eva.wav"]["label"] == 1
    unlabeled = by_name["open.wav"]
    assert "label" not in unlabeled
    assert "keyword_labels" not in unlabeled
    record = build_keyword_set_result_record(
        unlabeled,
        {
            "result_schema_version": 2,
            "eval_protocol": "stage2_clip_keyword_set",
            "keyword_eval_mode": "any",
            "keyword_set_id": "abc",
            "qbyt_score": 0.1,
            "qbyt_raw_logit": 0.0,
            "threshold": 0.5,
            "detected": False,
            "skipped": False,
            "skip_reason": None,
            "best_keyword": "hey eva",
            "best_pronunciation_id": "p",
            "matched_keywords": [],
            "keyword_results": [],
        },
    )
    assert "label" not in record


def test_jsonl_null_label_fields_are_unlabeled(tmp_path):
    _touch(tmp_path / "audio" / "x.wav")
    manifest = tmp_path / "eval.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "audio_path": "audio/x.wav",
                "label": None,
                "keyword_labels": None,
                "label_scope": None,
                "target_texts": None,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    rows = load_keyword_set_manifest(manifest, TARGETS)
    assert "label" not in rows[0]
    assert "keyword_labels" not in rows[0]
    assert "label_scope" not in rows[0]
    assert "target_texts" not in rows[0]


def test_unlabeled_row_is_allowed(tmp_path):
    _touch(tmp_path / "audio" / "x.wav")
    manifest = tmp_path / "eval.jsonl"
    manifest.write_text(
        json.dumps({"audio_path": "audio/x.wav", "note": "predict"}) + "\n",
        encoding="utf-8",
    )
    rows = load_keyword_set_manifest(manifest, TARGETS)
    assert "label" not in rows[0]
    assert "keyword_labels" not in rows[0]
    assert rows[0]["note"] == "predict"


def test_old_loader_does_not_accept_keyword_set_rows(tmp_path):
    _touch(tmp_path / "audio" / "eva.wav")
    manifest = tmp_path / "eval.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "audio_path": "audio/eva.wav",
                "keyword_labels": {"hey eva": 1, "ok lamp": 0},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="missing required columns: keyword"):
        load_manifest(manifest)


def test_legacy_pair_fields_are_rejected_with_format_examples(tmp_path):
    _touch(tmp_path / "audio" / "eva.wav")
    manifest = tmp_path / "eval.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "audio_path": "audio/eva.wav",
                "keyword": "hey eva",
                "keyword_labels": {"hey eva": 1, "ok lamp": 0},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(KeywordSetManifestError, match="keyword_labels"):
        load_keyword_set_manifest(manifest, TARGETS)


def test_missing_keyword_label_is_not_inferred_from_other_negatives(tmp_path):
    _touch(tmp_path / "audio" / "eva.wav")
    manifest = tmp_path / "eval.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "audio_path": "audio/eva.wav",
                "keyword_labels": {"hey eva": 0},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(KeywordSetManifestError, match="ok lamp"):
        load_keyword_set_manifest(manifest, TARGETS)


def test_conflicting_top_level_label_errors(tmp_path):
    _touch(tmp_path / "audio" / "eva.wav")
    manifest = tmp_path / "eval.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "audio_path": "audio/eva.wav",
                "keyword_labels": {"hey eva": 1, "ok lamp": 0},
                "label": 0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(KeywordSetManifestError, match="does not match"):
        load_keyword_set_manifest(manifest, TARGETS)


def test_bool_and_float_labels_are_rejected(tmp_path):
    _touch(tmp_path / "audio" / "eva.wav")
    manifest = tmp_path / "eval.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "audio_path": "audio/eva.wav",
                "keyword_labels": {"hey eva": True, "ok lamp": 0},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(KeywordSetManifestError, match="integer 0 or 1"):
        load_keyword_set_manifest(manifest, TARGETS)

    manifest.write_text(
        json.dumps(
            {
                "audio_path": "audio/eva.wav",
                "keyword_labels": {"hey eva": 1.0, "ok lamp": 0},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(KeywordSetManifestError, match="integer 0 or 1"):
        load_keyword_set_manifest(manifest, TARGETS)


def test_incomplete_target_set_fields_error(tmp_path):
    _touch(tmp_path / "audio" / "eva.wav")
    manifest = tmp_path / "eval.jsonl"
    manifest.write_text(
        json.dumps({"audio_path": "audio/eva.wav", "label": 1}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(KeywordSetManifestError, match="incomplete"):
        load_keyword_set_manifest(manifest, TARGETS)


def test_target_texts_must_equal_configured_set(tmp_path):
    _touch(tmp_path / "audio" / "eva.wav")
    manifest = tmp_path / "eval.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "audio_path": "audio/eva.wav",
                "label_scope": "target_set",
                "target_texts": ["hey eva"],
                "label": 1,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(KeywordSetManifestError, match="must equal configured"):
        load_keyword_set_manifest(manifest, TARGETS)


def test_duplicate_canonical_audio_paths_are_rejected(tmp_path):
    audio = _touch(tmp_path / "audio" / "eva.wav")
    manifest = tmp_path / "eval.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "audio_path": "audio/eva.wav",
                "keyword_labels": {"hey eva": 1, "ok lamp": 0},
            }
        )
        + "\n"
        + json.dumps(
            {
                "audio_path": str(audio.resolve()),
                "keyword_labels": {"hey eva": 0, "ok lamp": 0},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(KeywordSetManifestError, match="duplicates canonical audio path"):
        load_keyword_set_manifest(manifest, TARGETS)


def test_csv_outer_label_accepts_exact_zero_one_strings(tmp_path):
    _touch(tmp_path / "audio" / "eva.wav")
    csv_path = tmp_path / "eval.csv"
    csv_path.write_text(
        "audio_path,keyword_labels,label\n"
        'audio/eva.wav,"{""hey eva"":0,""ok lamp"":0}",0\n',
        encoding="utf-8",
    )
    rows = load_keyword_set_manifest(csv_path, TARGETS)
    assert rows[0]["label"] == 0
