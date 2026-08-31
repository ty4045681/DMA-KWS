from __future__ import annotations

import pytest

from dma_kws.inference.stage2_reporting import build_result_record


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
