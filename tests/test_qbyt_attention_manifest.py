from pathlib import Path

import pytest

from dma_kws.inference.manifest import load_manifest
from dma_kws.inference.qbyt_attention_manifest import validate_attention_manifest_rows


FIXTURES = Path(__file__).resolve().parent / "fixtures"
MINIMAL_CSV = FIXTURES / "qbyt_sink_manifest_minimal.csv"
EXTENDED_CSV = FIXTURES / "qbyt_sink_manifest_extended.csv"


def test_minimal_fixture_empty_label_stays_unlabeled_and_auto_ids_use_record_numbers():
    rows = load_manifest(MINIMAL_CSV)
    validated = validate_attention_manifest_rows(rows)

    assert [row.keyword for row in validated] == ["hey eva", "hey eva", "hey eva"]
    assert [row.label for row in validated] == [1, 0, None]
    assert [row.sample_id for row in validated] == [
        "row_000002",
        "row_000003",
        "row_000004",
    ]
    assert [row.record_number for row in validated] == [2, 3, 4]
    assert [row.condition for row in validated] == ["unknown", "unknown", "unknown"]
    for row in validated:
        assert row.pair_id is None
        assert row.pair_status == "ok"
        assert row.keyword_spans is None
        assert row.noise_spans is None
        assert row.keyword_phonemes is None
        assert Path(row.audio_path).is_absolute()
        assert row.internal_sample_id
        assert row.internal_sample_id.replace("_", "").isalnum()


def test_minimal_fixture_resolves_relative_audio_path_like_eval_stage2_clips():
    rows = load_manifest(MINIMAL_CSV)
    validated = validate_attention_manifest_rows(rows)

    expected = (FIXTURES / "audio" / "clean_001.wav").resolve()
    assert Path(validated[0].audio_path) == expected
    assert rows[0]["audio_path"] == str(expected)


def test_extended_fixture_quoting_spans_and_pair_baseline():
    rows = load_manifest(EXTENDED_CSV)
    validated = validate_attention_manifest_rows(rows)

    assert [row.sample_id for row in validated] == [
        "clean_001",
        "noisy_001",
        "noise_001",
        "speech_001",
    ]
    assert validated[0].keyword_spans == ((0.8, 1.6),)
    assert validated[0].noise_spans == ()
    assert validated[1].noise_spans == ((0.5, 2.0),)
    assert validated[2].keyword_spans is None
    assert validated[2].noise_spans == ((0.0, 2.0),)
    assert validated[3].keyword_spans is None
    assert validated[3].noise_spans is None
    assert validated[0].pair_id == "p001"
    assert validated[1].pair_id == "p001"
    assert validated[2].pair_id is None
    assert validated[3].pair_id is None
    assert validated[0].pair_status == "ok"
    assert validated[1].pair_status == "ok"
    assert validated[0].condition == "clean"
    assert validated[1].condition == "noisy"


def test_extra_columns_survive_load_manifest_then_validation(tmp_path):
    manifest_path = tmp_path / "extra.csv"
    manifest_path.write_text(
        "audio_path,keyword,label,text_variant,note\n"
        "clip.wav,hey eva,1,hey eever,keep-me\n",
        encoding="utf-8",
    )

    rows = load_manifest(manifest_path)
    assert rows[0]["text_variant"] == "hey eever"
    assert rows[0]["note"] == "keep-me"

    validated = validate_attention_manifest_rows(rows)
    assert validated[0].extra_fields["text_variant"] == "hey eever"
    assert validated[0].extra_fields["note"] == "keep-me"


def test_empty_keyword_phonemes_cell_is_distinct_from_missing_column(tmp_path):
    missing = tmp_path / "missing.csv"
    missing.write_text("audio_path,keyword\na.wav,hey eva\n", encoding="utf-8")
    empty = tmp_path / "empty.csv"
    empty.write_text(
        "audio_path,keyword,keyword_phonemes\na.wav,hey eva,\n",
        encoding="utf-8",
    )

    missing_row = validate_attention_manifest_rows(load_manifest(missing))[0]
    empty_row = validate_attention_manifest_rows(load_manifest(empty))[0]

    assert "keyword_phonemes" not in load_manifest(missing)[0]
    assert load_manifest(empty)[0]["keyword_phonemes"] == ""
    assert missing_row.keyword_phonemes is None
    assert empty_row.keyword_phonemes == ""
    assert missing_row.phoneme_override is None
    assert empty_row.phoneme_override is None


def test_illegal_label_is_rejected_after_shared_loader(tmp_path):
    manifest_path = tmp_path / "bad_label.csv"
    manifest_path.write_text(
        "audio_path,keyword,label\na.wav,hey eva,2\n",
        encoding="utf-8",
    )
    rows = load_manifest(manifest_path)
    assert rows[0]["label"] == 2

    with pytest.raises(ValueError, match="label"):
        validate_attention_manifest_rows(rows)


def test_duplicate_sample_id_is_rejected(tmp_path):
    manifest_path = tmp_path / "dup_id.csv"
    manifest_path.write_text(
        "audio_path,keyword,sample_id\n"
        "a.wav,hey eva,same\n"
        "b.wav,hey eva,same\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="sample_id"):
        validate_attention_manifest_rows(load_manifest(manifest_path))


def test_user_sample_id_does_not_become_a_raw_path_and_collisions_are_distinct(tmp_path):
    manifest_path = tmp_path / "ids.csv"
    manifest_path.write_text(
        "audio_path,keyword,sample_id\n"
        "a.wav,hey eva,foo-bar\n"
        "b.wav,hey eva,foo_bar\n"
        'c.wav,hey eva,"id with spaces/and\\\\slashes"\n',
        encoding="utf-8",
    )
    validated = validate_attention_manifest_rows(load_manifest(manifest_path))

    internals = [row.internal_sample_id for row in validated]
    assert len(set(internals)) == 3
    for internal in internals:
        assert internal.replace("_", "").isalnum()
        assert "/" not in internal
        assert "\\" not in internal
        assert " " not in internal
        assert "-" not in internal
    assert validated[0].sample_id == "foo-bar"
    assert validated[0].internal_sample_id != "foo-bar"


def test_nan_and_infinity_spans_are_rejected(tmp_path):
    manifest_path = tmp_path / "nan_spans.csv"
    manifest_path.write_text(
        "audio_path,keyword,keyword_spans\n"
        'a.wav,hey eva,"[[NaN, 1.0]]"\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="keyword_spans"):
        validate_attention_manifest_rows(load_manifest(manifest_path))

    inf_path = tmp_path / "inf_spans.csv"
    inf_path.write_text(
        "audio_path,keyword,noise_spans\n"
        'a.wav,hey eva,"[[0.0, Infinity]]"\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="noise_spans"):
        validate_attention_manifest_rows(load_manifest(inf_path))

    overflow = tmp_path / "overflow_spans.csv"
    overflow.write_text(
        "audio_path,keyword,keyword_spans\n"
        'a.wav,hey eva,"[[0.0, 1e999]]"\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="keyword_spans"):
        validate_attention_manifest_rows(load_manifest(overflow))


def test_span_duration_bound_applies_only_when_supplied(tmp_path):
    manifest_path = tmp_path / "spans.csv"
    manifest_path.write_text(
        "audio_path,keyword,keyword_spans\n"
        'a.wav,hey eva,"[[0.0, 1.5]]"\n'
        'b.wav,hey eva,"[[0.0, 3.0]]"\n',
        encoding="utf-8",
    )
    rows = load_manifest(manifest_path)

    unbounded = validate_attention_manifest_rows(rows)
    assert unbounded[1].keyword_spans == ((0.0, 3.0),)

    ok = validate_attention_manifest_rows(
        rows,
        source_durations_sec=[2.0, None],
    )
    assert ok[0].keyword_spans == ((0.0, 1.5),)

    with pytest.raises(ValueError, match="duration"):
        validate_attention_manifest_rows(rows, source_durations_sec=[1.0, None])


def test_overlapping_spans_are_kept(tmp_path):
    manifest_path = tmp_path / "overlap.csv"
    manifest_path.write_text(
        "audio_path,keyword,noise_spans\n"
        'a.wav,hey eva,"[[0.0, 1.0], [0.5, 1.5]]"\n',
        encoding="utf-8",
    )
    row = validate_attention_manifest_rows(load_manifest(manifest_path))[0]
    assert row.noise_spans == ((0.0, 1.0), (0.5, 1.5))


def test_pair_missing_and_multiple_clean_and_inconsistent_override(tmp_path):
    missing = tmp_path / "missing_clean.csv"
    missing.write_text(
        "audio_path,keyword,condition,pair_id\n"
        "a.wav,hey eva,noisy,p1\n"
        "b.wav,hey eva,noise_only,p1\n",
        encoding="utf-8",
    )
    missing_rows = validate_attention_manifest_rows(load_manifest(missing))
    assert {row.pair_status for row in missing_rows} == {"missing_clean_baseline"}
    assert all(row.pair_reason for row in missing_rows)

    multiple = tmp_path / "multiple_clean.csv"
    multiple.write_text(
        "audio_path,keyword,condition,pair_id\n"
        "a.wav,hey eva,clean,p1\n"
        "b.wav,hey eva,clean,p1\n",
        encoding="utf-8",
    )
    multiple_rows = validate_attention_manifest_rows(load_manifest(multiple))
    assert {row.pair_status for row in multiple_rows} == {"multiple_clean_baselines"}

    inconsistent = tmp_path / "inconsistent.csv"
    inconsistent.write_text(
        "audio_path,keyword,keyword_phonemes,condition,pair_id\n"
        "a.wav,hey eva,HH EY1 IY1 V AH0,clean,p1\n"
        "b.wav,hey eva,HH EY1 AH0 V AH0,noisy,p1\n",
        encoding="utf-8",
    )
    inconsistent_rows = validate_attention_manifest_rows(load_manifest(inconsistent))
    assert {row.pair_status for row in inconsistent_rows} == {
        "inconsistent_keyword_or_phonemes"
    }


def test_pair_groups_by_keyword_and_phoneme_override_not_csv_order(tmp_path):
    manifest_path = tmp_path / "pairs.csv"
    manifest_path.write_text(
        "audio_path,keyword,keyword_phonemes,condition,pair_id\n"
        "noisy.wav,hey eva,HH EY1 IY1 V AH0,noisy,p1\n"
        "other.wav,ok google,,clean,p1\n"
        "clean.wav,hey eva,HH EY1 IY1 V AH0,clean,p1\n",
        encoding="utf-8",
    )
    rows = validate_attention_manifest_rows(load_manifest(manifest_path))

    assert rows[1].pair_status == "inconsistent_keyword_or_phonemes"
    assert rows[0].pair_status == "inconsistent_keyword_or_phonemes"
    assert rows[2].pair_status == "inconsistent_keyword_or_phonemes"


def test_auto_ids_respect_first_record_number():
    rows = load_manifest(MINIMAL_CSV)
    validated = validate_attention_manifest_rows(rows, first_record_number=10)
    assert [row.sample_id for row in validated] == [
        "row_000010",
        "row_000011",
        "row_000012",
    ]
    assert [row.record_number for row in validated] == [10, 11, 12]
