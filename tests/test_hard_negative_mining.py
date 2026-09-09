from __future__ import annotations

from copy import deepcopy
import json

import pytest

from dma_kws.stage2.hard_negative_mining import (
    merge_hard_negative_manifest_rows,
    select_hard_negatives,
)
from scripts.mine_hey_eva_hard_negatives import _validate_comparable_results


def _score_provenance(
    *,
    checkpoint_path: str,
    tokenizer_path: str,
    checkpoint_sha256: str = "a" * 64,
) -> dict:
    return {
        "schema_version": 3,
        "checkpoint": {
            "path": checkpoint_path,
            "size_bytes": 123,
            "sha256": checkpoint_sha256,
        },
        "calibration": {"type": "identity_logit_sigmoid"},
        "qbyt_alignment": {
            "topology": "keyword_filler_segmental_crf_v1",
            "min_phone_duration_frames": 1,
            "max_phone_duration_frames": 8,
            "max_inter_phone_gap_frames": 1,
            "max_keyword_span_frames": 30,
            "weakest_phone_temperature": 0.2,
            "weakest_phone_weight": 1.0,
            "local_context_kernel": 5,
        },
        "stream": "backend=zipformer chunking=off",
        "audio_padding_ms": {"left": 160, "right": 160},
        "fbank": {"num_mel_bins": 80, "dither": 0.0},
        "tokenizer": {
            "path": tokenizer_path,
            "size_bytes": 456,
            "sha256": "b" * 64,
            "split_with_space": " ",
        },
        "sequence_objective": {
            "target_mode": "ordered_contiguous_prefix",
            "progress_weight": 0.3,
            "normalization": "sample",
        },
    }


def _results_with_summary(tmp_path, name: str, provenance: dict):
    output_dir = tmp_path / name
    output_dir.mkdir()
    results_path = output_dir / "results.jsonl"
    results_path.write_text("{}\n", encoding="utf-8")
    (output_dir / "summary.json").write_text(
        json.dumps({"provenance": provenance}),
        encoding="utf-8",
    )
    return results_path


def test_comparable_results_ignore_location_only_provenance_paths(tmp_path):
    mac = _results_with_summary(
        tmp_path,
        "mac",
        _score_provenance(
            checkpoint_path="/Users/local/stage2.pt",
            tokenizer_path="/Users/local/lang_char.txt",
        ),
    )
    linux = _results_with_summary(
        tmp_path,
        "linux",
        _score_provenance(
            checkpoint_path="/home/remote/stage2.pt",
            tokenizer_path="/home/remote/lang_char.txt",
        ),
    )

    _validate_comparable_results([mac, linux])


def test_comparable_results_accept_schema3_and_schema4_per_row(tmp_path):
    schema3 = _score_provenance(
        checkpoint_path="/models/stage2.pt",
        tokenizer_path="/dict/lang_char.txt",
    )
    schema4 = deepcopy(schema3)
    schema4["schema_version"] = 4
    schema4["keyword_eval"] = {"mode": "per_row"}
    first = _results_with_summary(tmp_path, "schema3", schema3)
    second = _results_with_summary(tmp_path, "schema4", schema4)
    _validate_comparable_results([first, second])


def test_comparable_results_reject_content_or_score_semantics_mismatch(tmp_path):
    baseline = _score_provenance(
        checkpoint_path="/models/stage2.pt",
        tokenizer_path="/dict/lang_char.txt",
    )
    different_hash = _score_provenance(
        checkpoint_path="/copied/stage2.pt",
        tokenizer_path="/copied/lang_char.txt",
        checkpoint_sha256="c" * 64,
    )
    different_fbank = deepcopy(baseline)
    different_fbank["fbank"]["dither"] = 1.0
    different_tokenizer = deepcopy(baseline)
    different_tokenizer["tokenizer"]["sha256"] = "d" * 64
    different_alignment = deepcopy(baseline)
    different_alignment["qbyt_alignment"]["max_inter_phone_gap_frames"] = 0
    different_calibration = deepcopy(baseline)
    different_calibration["calibration"] = {
        "path": "/models/calibration.json",
        "size_bytes": 128,
        "sha256": "e" * 64,
    }

    first = _results_with_summary(tmp_path, "first", baseline)
    second = _results_with_summary(tmp_path, "second", different_hash)
    third = _results_with_summary(tmp_path, "third", different_fbank)
    fourth = _results_with_summary(tmp_path, "fourth", different_tokenizer)
    fifth = _results_with_summary(tmp_path, "fifth", different_alignment)
    sixth = _results_with_summary(tmp_path, "sixth", different_calibration)

    with pytest.raises(ValueError, match="Cannot mix hard-negative scores"):
        _validate_comparable_results([first, second])
    with pytest.raises(ValueError, match="Cannot mix hard-negative scores"):
        _validate_comparable_results([first, third])
    with pytest.raises(ValueError, match="Cannot mix hard-negative scores"):
        _validate_comparable_results([first, fourth])
    with pytest.raises(ValueError, match="Cannot mix hard-negative scores"):
        _validate_comparable_results([first, fifth])
    with pytest.raises(ValueError, match="Cannot mix hard-negative scores"):
        _validate_comparable_results([first, sixth])


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value.update(schema_version=2), "unsupported score provenance"),
        (lambda value: value.pop("schema_version"), "missing fields"),
        (lambda value: value["checkpoint"].pop("sha256"), "checkpoint.*missing"),
    ],
)
def test_comparable_results_validate_provenance_schema(
    tmp_path,
    mutation,
    message,
):
    valid = _score_provenance(
        checkpoint_path="/models/stage2.pt",
        tokenizer_path="/dict/lang_char.txt",
    )
    invalid = deepcopy(valid)
    mutation(invalid)
    first = _results_with_summary(tmp_path, "valid", valid)
    second = _results_with_summary(tmp_path, "invalid", invalid)

    with pytest.raises(ValueError, match=message):
        _validate_comparable_results([first, second])


def _row(
    audio: str,
    *,
    score: float,
    speaker: str,
    text: str = "hey ava",
    split: str = "train",
    label: int = 0,
) -> dict:
    return {
        "audio_path": audio,
        "qbyt_score": score,
        "label": label,
        "skipped": False,
        "manifest_meta": {
            "text_variant": text,
            "speaker_id": speaker,
            "split": split,
            "source": "speechocean762",
        },
        "keyword_phonemes": ["HH", "EY1", "IY1", "V", "AH0"],
        "text_variant_phonemes": ["HH", "EY1", "EY1", "V", "AH0"],
    }


def test_selects_high_scores_with_per_speaker_cap_and_deterministic_order():
    selected = select_hard_negatives(
        [
            _row("/audio/b.wav", score=0.9, speaker="s1"),
            _row("/audio/a.wav", score=0.9, speaker="s1"),
            _row("/audio/c.wav", score=0.8, speaker="s2"),
            _row("/audio/d.wav", score=0.4, speaker="s3"),
        ],
        keyword="hey eva",
        min_score=0.5,
        top_k=10,
        per_speaker_cap=1,
    )

    assert [row["audio_path"] for row in selected] == ["/audio/a.wav", "/audio/c.wav"]
    assert all(row["label"] == 0 for row in selected)
    assert all(row["negative_type"] == "stage2_false_positive" for row in selected)
    assert selected[0]["keyword_phonemes"] == "HH EY1 IY1 V AH0"
    assert selected[0]["text_variant_phonemes"] == "HH EY1 EY1 V AH0"


def test_excludes_blind_rows_positives_and_text_containing_complete_keyword():
    selected = select_hard_negatives(
        [
            _row("/audio/blind.wav", score=0.99, speaker="s1", split="test"),
            _row("/audio/positive.wav", score=0.99, speaker="s2", label=1),
            _row(
                "/audio/stale.wav",
                score=0.99,
                speaker="s3",
                text="please, Hey Eva now",
            ),
            _row("/audio/valid.wav", score=0.7, speaker="s4", text="hey eve"),
        ],
        keyword="hey eva",
        min_score=0.5,
        top_k=10,
        per_speaker_cap=5,
    )

    assert [row["audio_path"] for row in selected] == ["/audio/valid.wav"]


def test_rejects_eligible_row_without_speaker_id():
    row = _row("/audio/a.wav", score=0.9, speaker="")
    with pytest.raises(ValueError, match="speaker_id"):
        select_hard_negatives(
            [row],
            keyword="hey eva",
            min_score=0.5,
            top_k=10,
            per_speaker_cap=5,
        )


def test_merge_hard_negatives_keeps_eval_and_rejects_speaker_leakage():
    base = [
        {
            "audio_path": "/audio/train.wav",
            "text": "hey eva",
            "label": 1,
            "phase": "real",
            "split": "train",
            "speaker_id": "real-train",
        },
        {
            "audio_path": "/audio/eval.wav",
            "text": "hey eva",
            "label": 1,
            "phase": "real",
            "split": "eval",
            "speaker_id": "real-eval",
        },
    ]
    negatives = [
        {
            "audio_path": "/audio/hard.wav",
            "text": "hey eve",
            "label": 0,
            "phase": "real",
            "split": "train",
            "speaker_id": "accent-train",
        }
    ]

    assert len(merge_hard_negative_manifest_rows(base, negatives)) == 3
    negatives[0]["speaker_id"] = "real-eval"
    with pytest.raises(ValueError, match="leaks speakers"):
        merge_hard_negative_manifest_rows(base, negatives)


def test_merge_hard_negatives_deduplicates_existing_real_negative():
    base = [
        {
            "audio_path": "/audio/negative.wav",
            "text": "hey ava",
            "label": 0,
            "phase": "real",
            "split": "train",
            "speaker_id": "real-train",
        },
        {
            "audio_path": "/audio/eval.wav",
            "text": "hey eva",
            "label": 1,
            "phase": "real",
            "split": "eval",
            "speaker_id": "real-eval",
        },
    ]
    selected = [
        {
            **base[0],
            "negative_type": "stage2_false_positive",
            "qbyt_score": 0.91,
        }
    ]

    merged = merge_hard_negative_manifest_rows(base, selected)

    assert len(merged) == 2
    assert merged[0]["mined_hard_negative"] == 1
    assert merged[0]["qbyt_score"] == pytest.approx(0.91)
