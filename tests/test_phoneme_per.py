from __future__ import annotations

import pytest

from dma_kws.inference.phoneme_per import (
    build_per_record,
    select_per_rows,
    summarize_per_results,
)


def _rows():
    return [
        {
            "audio_path": "/audio/positive.wav",
            "keyword": "hey snips",
            "label": 1,
            "text_variant": "hey snips",
        },
        {
            "audio_path": "/audio/negative.wav",
            "keyword": "hey snips",
            "label": 0,
            "text_variant": "turn on the lights",
        },
    ]


def test_keyword_reference_requires_positive_filter():
    with pytest.raises(ValueError, match="positive clips"):
        select_per_rows(_rows(), reference_column="keyword")

    with pytest.raises(ValueError, match="positive clips"):
        select_per_rows(_rows(), reference_column="keyword", label_filter=0)


def test_keyword_reference_selects_only_positive_rows():
    selected = select_per_rows(
        _rows(),
        reference_column="keyword",
        label_filter=1,
    )

    assert [row["audio_path"] for row in selected] == ["/audio/positive.wav"]


def test_text_variant_reference_supports_positive_and_negative_rows():
    selected = select_per_rows(_rows(), reference_column="text_variant")

    assert len(selected) == 2


def test_selected_row_must_have_reference_column():
    rows = [{"audio_path": "/audio/a.wav", "keyword": "hey snips", "label": 1}]

    with pytest.raises(ValueError, match="empty 'text_variant'"):
        select_per_rows(rows, reference_column="text_variant")


def test_csv_row_numbering_starts_after_header():
    rows = [
        {
            "audio_path": "/audio/a.wav",
            "keyword": "hey snips",
            "label": 1,
            "text_variant": "",
        }
    ]

    with pytest.raises(ValueError, match="Manifest row 2 has an empty 'text_variant'"):
        select_per_rows(
            rows,
            reference_column="text_variant",
            first_row_number=2,
        )


def test_direct_rows_with_invalid_label_report_row_number():
    rows = [
        {
            "audio_path": "/audio/a.wav",
            "keyword": "hey snips",
            "label": "pos",
            "text_variant": "hey snips",
        }
    ]

    with pytest.raises(ValueError, match="Manifest row 2 has invalid label 'pos'"):
        select_per_rows(
            rows,
            reference_column="text_variant",
            first_row_number=2,
        )


def test_build_per_record_invalid_label_reports_audio_context():
    row = {
        "audio_path": "/audio/a.wav",
        "keyword": "hey snips",
        "label": "pos",
        "text_variant": "hey snips",
    }

    with pytest.raises(
        ValueError,
        match="PER result for audio '/audio/a.wav' has invalid label 'pos'",
    ):
        build_per_record(
            row,
            reference_column="text_variant",
            reference_phonemes=["HH"],
            reference_ids=[2],
            hypothesis_phonemes=["HH"],
            hypothesis_ids=[2],
            skipped=False,
        )


def test_limit_is_applied_after_label_filter():
    rows = [
        {"audio_path": f"/{index}.wav", "keyword": "hey snips", "label": label}
        for index, label in enumerate([0, 1, 1])
    ]

    selected = select_per_rows(
        rows,
        reference_column="keyword",
        label_filter=1,
        limit=1,
    )

    assert selected[0]["audio_path"] == "/1.wav"


def test_per_summary_is_corpus_weighted_and_counts_skips():
    first = build_per_record(
        _rows()[0],
        reference_column="text_variant",
        reference_phonemes=["HH", "EY1"],
        reference_ids=[2, 3],
        hypothesis_phonemes=["HH"],
        hypothesis_ids=[2],
        skipped=False,
    )
    second = build_per_record(
        _rows()[1],
        reference_column="text_variant",
        reference_phonemes=["T", "ER1", "N"],
        reference_ids=[4, 5, 6],
        hypothesis_phonemes=[],
        hypothesis_ids=[],
        skipped=True,
    )

    summary = summarize_per_results([first, second])

    assert first["edit_distance"] == 1
    assert second["edit_distance"] == 3
    assert summary["total_edit_distance"] == 4
    assert summary["total_reference_phonemes"] == 5
    assert summary["per"] == pytest.approx(0.8)
    assert summary["num_skipped"] == 1
    assert summary["num_empty_hypotheses"] == 1
