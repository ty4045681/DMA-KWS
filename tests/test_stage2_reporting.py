from __future__ import annotations

import pytest

from dma_kws.inference.stage2_reporting import (
    build_keyword_set_result_record,
    build_result_record,
)


def test_result_record_exports_only_the_deployed_qbyt_score():
    record = build_result_record(
        {"audio_path": "clip.wav", "keyword": "hey eva", "label": 1},
        {
            "keyword_phonemes": ["HH", "EY1", "IY1", "V", "AH0"],
            "qbyt_score": 0.75,
            "detected": True,
            "threshold": 0.5,
            "skipped": False,
        },
    )

    assert record == {
        "audio_path": "clip.wav",
        "keyword": "hey eva",
        "keyword_phonemes": ["HH", "EY1", "IY1", "V", "AH0"],
        "qbyt_score": pytest.approx(0.75),
        "detected": True,
        "threshold": pytest.approx(0.5),
        "skipped": False,
        "label": 1,
    }
    assert "qbyt_eps_position_logits" not in record


def test_result_record_exports_optional_eps_position_logits():
    phonemes = ["HH", "EY1", "IY1", "V", "AH0"]
    positions = [0.82, 0.71, 0.66, 0.59, 0.47]
    record = build_result_record(
        {"audio_path": "clip.wav", "keyword": "hey eva", "label": 1},
        {
            "keyword_phonemes": phonemes,
            "qbyt_eps_position_logits": positions,
            "qbyt_raw_logit": 0.61,
            "qbyt_score": 0.648,
            "detected": True,
            "threshold": 0.5,
            "skipped": False,
        },
    )
    assert record["qbyt_score"] == pytest.approx(0.648)
    assert record["qbyt_raw_logit"] == pytest.approx(0.61)
    assert record["qbyt_eps_position_logits"] == positions
    assert record["qbyt_eps_position_logits"] is not positions


@pytest.mark.parametrize(
    ("positions", "match"),
    [
        ([0.1, 0.2], "length"),
        ([0.1, 0.2, 0.3, 0.4, float("nan")], "finite"),
        ([0.1, 0.2, 0.3, 0.4, float("inf")], "finite"),
        ("0.1 0.2", "sequence|array"),
        ({"0": 0.1}, "sequence|array"),
        (True, "sequence|array|bool"),
    ],
)
def test_result_record_rejects_invalid_eps_position_logits(positions, match):
    with pytest.raises(ValueError, match=match):
        build_result_record(
            {"audio_path": "clip.wav", "keyword": "hey eva"},
            {
                "keyword_phonemes": ["HH", "EY1", "IY1", "V", "AH0"],
                "qbyt_eps_position_logits": positions,
                "qbyt_score": 0.75,
                "detected": True,
                "threshold": 0.5,
                "skipped": False,
            },
        )


def test_result_record_preserves_augmented_model_input_duration():
    record = build_result_record(
        {"audio_path": "clip.wav", "keyword": "hello"},
        {
            "keyword_phonemes": ["HH", "AH0", "L", "OW1"],
            "qbyt_score": 0.75,
            "detected": True,
            "threshold": 0.5,
            "skipped": False,
            "clip_span_sec": {"start_sec": 0.0, "end_sec": 1.0},
            "augmented_duration_sec": 0.8,
        },
    )

    assert record["clip_span_sec"] == {"start_sec": 0.0, "end_sec": 1.0}
    assert record["augmented_duration_sec"] == pytest.approx(0.8)


@pytest.mark.parametrize(
    ("runner_fields", "match"),
    [
        ({"clip_span_sec": {"start_sec": -0.1, "end_sec": 1.0}}, "start_sec"),
        ({"clip_span_sec": {"start_sec": 1.0, "end_sec": 0.5}}, "end_sec"),
        ({"augmented_duration_sec": 0.0}, "augmented_duration_sec"),
    ],
)
def test_result_record_rejects_invalid_audio_durations(runner_fields, match):
    runner_result = {
        "keyword_phonemes": ["HH"],
        "qbyt_score": 0.75,
        "detected": True,
        "threshold": 0.5,
        "skipped": False,
        **runner_fields,
    }

    with pytest.raises(ValueError, match=match):
        build_result_record(
            {"audio_path": "clip.wav", "keyword": "hello"},
            runner_result,
        )


def test_keyword_set_result_record_is_nested_and_rejects_forged_keyword():
    runner_result = {
        "result_schema_version": 2,
        "eval_protocol": "stage2_clip_keyword_set",
        "keyword_eval_mode": "any",
        "keyword_set_id": "abc",
        "qbyt_score": 0.9,
        "qbyt_raw_logit": 1.5,
        "threshold": 0.5,
        "detected": True,
        "skipped": False,
        "skip_reason": None,
        "best_keyword": "hey eva",
        "best_pronunciation_id": "p1",
        "matched_keywords": ["hey eva"],
        "keyword_results": [
            {
                "keyword_id": "k1",
                "text": "hey eva",
                "qbyt_raw_logit": 1.5,
                "qbyt_score": 0.9,
                "detected": True,
                "best_pronunciation_id": "p1",
                "pronunciation_results": [
                    {
                        "pronunciation_id": "p1",
                        "phonemes": ["HH", "EY1"],
                        "token_ids": [1, 2],
                        "qbyt_raw_logit": 1.5,
                        "qbyt_score": 0.9,
                        "detected": True,
                    }
                ],
            }
        ],
        "clip_span_sec": {"start_sec": 0.0, "end_sec": 1.2},
    }
    record = build_keyword_set_result_record(
        {
            "audio_path": "clip.wav",
            "label": 1,
            "label_scope": "target_set",
            "keyword_labels": {"hey eva": 1},
        },
        runner_result,
    )
    assert "keyword" not in record
    assert record["eval_protocol"] == "stage2_clip_keyword_set"
    assert record["keyword_eval_mode"] == "any"
    assert record["qbyt_score"] == pytest.approx(0.9)
    assert record["label"] == 1
    assert "qbyt_eps_position_logits" not in record["keyword_results"][0][
        "pronunciation_results"
    ][0]
    with pytest.raises(ValueError, match="must not forge"):
        build_keyword_set_result_record(
            {"audio_path": "clip.wav"},
            {**runner_result, "keyword": "hey eva"},
        )


def test_keyword_set_result_record_validates_pronunciation_position_logits():
    runner_result = {
        "result_schema_version": 3,
        "eval_protocol": "stage2_clip_keyword_set",
        "keyword_eval_mode": "any",
        "keyword_set_id": "abc",
        "qbyt_score": 0.9,
        "qbyt_raw_logit": 1.5,
        "threshold": 0.5,
        "detected": True,
        "skipped": False,
        "skip_reason": None,
        "best_keyword": "hey eva",
        "best_pronunciation_id": "p1",
        "matched_keywords": ["hey eva"],
        "keyword_results": [
            {
                "keyword_id": "k1",
                "text": "hey eva",
                "qbyt_raw_logit": 1.5,
                "qbyt_score": 0.9,
                "detected": True,
                "best_pronunciation_id": "p1",
                "pronunciation_results": [
                    {
                        "pronunciation_id": "p1",
                        "phonemes": ["HH", "EY1"],
                        "token_ids": [1, 2],
                        "qbyt_eps_position_logits": [0.82, 0.71],
                        "qbyt_raw_logit": 1.5,
                        "qbyt_score": 0.9,
                        "detected": True,
                    }
                ],
            }
        ],
    }
    record = build_keyword_set_result_record(
        {"audio_path": "clip.wav"},
        runner_result,
    )
    positions = record["keyword_results"][0]["pronunciation_results"][0][
        "qbyt_eps_position_logits"
    ]
    assert positions == [0.82, 0.71]
    assert "qbyt_eps_position_logits" not in record
    bad = {
        **runner_result,
        "keyword_results": [
            {
                **runner_result["keyword_results"][0],
                "pronunciation_results": [
                    {
                        **runner_result["keyword_results"][0]["pronunciation_results"][0],
                        "qbyt_eps_position_logits": [0.82],
                    }
                ],
            }
        ],
    }
    with pytest.raises(ValueError, match="length"):
        build_keyword_set_result_record({"audio_path": "clip.wav"}, bad)
