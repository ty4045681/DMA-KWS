import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from omegaconf import OmegaConf

from dma_kws.inference.manifest import load_manifest
from dma_kws.inference.metrics import binary_eer, summarize_labeled_results
import scripts.eval_stage2_clips as eval_stage2_clips
from scripts.eval_stage2_clips import (
    _result_record as stage2_clip_result_record,
    _resolve_audio_padding_ms,
    _score_provenance,
)
from scripts.eval_two_stage_kws import _result_record as two_stage_result_record


FIXTURES = Path(__file__).resolve().parent / "fixtures"


def test_stage2_clip_audio_padding_defaults_to_160ms_per_side():
    assert _resolve_audio_padding_ms({}) == (160, 160)


def test_stage2_clip_audio_padding_can_be_overridden_per_side():
    assert _resolve_audio_padding_ms(
        {"left_padding_ms": 0, "right_padding_ms": 240}
    ) == (0, 240)


@pytest.mark.parametrize(
    "prep",
    [
        {"left_padding_ms": -1},
        {"right_padding_ms": -1},
    ],
)
def test_stage2_clip_audio_padding_rejects_negative_values(prep):
    with pytest.raises(SystemExit, match="must be >= 0"):
        _resolve_audio_padding_ms(prep)


def test_stage2_clip_eval_applies_and_records_default_padding(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    checkpoint_path = tmp_path / "stage2.pt"
    torch.save(
        {
            "config": {
                "stage2": {
                    "sequence_loss": {
                        "target_mode": "ordered_contiguous_prefix",
                        "progress_weight": 0.5,
                        "completion_weight": 0.5,
                        "normalization": "sample",
                    }
                }
            }
        },
        checkpoint_path,
    )
    tokenizer_path = tmp_path / "lang_char.txt"
    tokenizer_path.write_text("<blank> 0\nHH 1\n", encoding="utf-8")
    rows = [
        {
            "audio_path": "clip.wav",
            "keyword": "hello",
            "keyword_phonemes": "HH",
            "text_variant": "hullo",
        }
    ]
    captured = {}

    class FakeStreamPolicy:
        @staticmethod
        def describe():
            return {"mode": "test"}

    class FakeRunner:
        _demo_cfg = {"qbyt_threshold": 0.5}
        stream_policy = FakeStreamPolicy()

        def run_batch(self, batch_rows, **kwargs):
            captured["rows"] = batch_rows
            captured["kwargs"] = kwargs
            return [
                {
                    "qbyt_score": 0.75,
                    "qbyt_logit": 1.0986122886681098,
                    "keyword_phonemes": ["HH"],
                    "text_variant_phonemes": ["HH", "AH0", "L", "OW1"],
                    "eps_position_logits": [1.0986122886681098],
                    "detected": True,
                    "threshold": 0.5,
                    "skipped": False,
                }
            ]

    runner = FakeRunner()

    class FakeRunnerFactory:
        @staticmethod
        def from_config(_config, _prep, _device):
            return runner

    monkeypatch.setattr(
        eval_stage2_clips,
        "resolved_config",
        lambda _cfg: {
            "paths": {},
            "stage1": {},
            "stage2": {
                "qbyt_readout": {"mode": "eps_mean"},
                "sequence_loss": {
                    "target_mode": "ordered_contiguous_prefix",
                    "completion_weight": 0.5,
                },
                "validation": {
                    "ece_num_bins": 9,
                    "seq_diagnostic_threshold": 0.4,
                },
            },
            "demo": {},
            "tokenizer": {
                "dict_path": str(tokenizer_path),
                "split_with_space": " ",
            },
        },
    )
    monkeypatch.setattr(eval_stage2_clips, "load_manifest", lambda _path: rows)
    monkeypatch.setattr(
        eval_stage2_clips, "resolve_accelerator", lambda _device: ("cpu", 1)
    )
    monkeypatch.setattr(eval_stage2_clips, "Stage2ClipRunner", FakeRunnerFactory)

    cfg = OmegaConf.create(
        {
            "prep": {
                "manifest": "manifest.csv",
                "stage2_ckpt": str(checkpoint_path),
                "output_dir": str(tmp_path),
                "num_workers": 1,
            },
            "run": {"device": "cpu"},
        }
    )

    summary = eval_stage2_clips.run_eval(cfg)

    assert captured["rows"] == rows
    assert captured["kwargs"]["left_padding_ms"] == 160
    assert captured["kwargs"]["right_padding_ms"] == 160
    assert captured["kwargs"]["include_score_details"] is True
    assert captured["kwargs"]["include_eps_positions"] is True
    assert summary["audio_padding_ms"] == {"left": 160, "right": 160}
    assert summary["provenance"]["checkpoint"]["path"] == str(
        checkpoint_path.resolve()
    )
    assert summary["provenance"]["checkpoint"]["sha256"] == hashlib.sha256(
        checkpoint_path.read_bytes()
    ).hexdigest()
    assert summary["provenance"]["sequence_objective"]["target_mode"] == (
        "ordered_contiguous_prefix"
    )
    assert summary["score_diagnostic_config"] == {
        "utterance_threshold": 0.5,
        "completion_threshold": 0.4,
        "ece_num_bins": 9,
    }
    saved_summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert saved_summary["audio_padding_ms"] == {"left": 160, "right": 160}
    saved_result = json.loads(
        (tmp_path / "results.jsonl").read_text(encoding="utf-8").strip()
    )
    assert saved_result["keyword_phonemes"] == ["HH"]
    assert saved_result["text_variant_phonemes"] == ["HH", "AH0", "L", "OW1"]
    assert saved_result["manifest_meta"] == {"text_variant": "hullo"}
    assert saved_result["eps_position_logits"] == pytest.approx(
        [1.0986122886681098]
    )


def test_score_provenance_classifies_missing_checkpoint_config_as_legacy(tmp_path):
    torch = pytest.importorskip("torch")
    checkpoint_path = tmp_path / "legacy.pt"
    torch.save({"state_dict": {}}, checkpoint_path)
    tokenizer_path = tmp_path / "lang_char.txt"
    tokenizer_path.write_text("<blank> 0\nHH 1\n", encoding="utf-8")
    config = {
        "stage1": {},
        "stage2": {"qbyt_readout": {"mode": "gru_last"}},
        "fbank": {},
        "tokenizer": {
            "dict_path": str(tokenizer_path),
            "split_with_space": " ",
        },
    }

    provenance = _score_provenance(
        config,
        checkpoint_path=checkpoint_path,
        stream="full-context",
        left_padding_ms=160,
        right_padding_ms=160,
    )

    assert provenance["sequence_objective"] == {
        "target_mode": "membership",
        "progress_weight": 1.0,
        "completion_weight": 0.0,
        "normalization": "token",
    }
    assert provenance["tokenizer"]["sha256"] == hashlib.sha256(
        tokenizer_path.read_bytes()
    ).hexdigest()


def test_load_manifest_csv_smoke():
    rows = load_manifest(FIXTURES / "manifest_smoke.csv")

    assert len(rows) == 3
    assert rows[0]["keyword"] == "hello world"
    assert rows[0]["label"] == 1
    assert Path(rows[0]["audio_path"]).name == "audio_a.wav"


def test_load_manifest_requires_audio_and_keyword(tmp_path):
    manifest = tmp_path / "bad.csv"
    manifest.write_text("audio_path,keyword\n,hello\n", encoding="utf-8")

    with pytest.raises(ValueError, match="missing required columns"):
        load_manifest(manifest)


def test_summarize_labeled_results_metrics():
    results = [
        {"label": 1, "best_qbyt_score": 0.9, "detected": True},
        {"label": 0, "best_qbyt_score": 0.1, "detected": False},
        {"label": 1, "best_qbyt_score": 0.4, "detected": False},
    ]

    summary = summarize_labeled_results(results, threshold=0.5)

    assert summary["num_samples"] == 3.0
    assert summary["tp"] == 1.0
    assert summary["tn"] == 1.0
    assert summary["fp"] == 0.0
    assert summary["fn"] == 1.0
    assert summary["accuracy"] == pytest.approx(2 / 3)
    assert summary["precision"] == pytest.approx(1.0)
    assert summary["recall"] == pytest.approx(0.5)
    assert summary["fpr"] == pytest.approx(0.0)
    assert summary["fnr"] == pytest.approx(0.5)
    assert 0.0 <= summary["auc"] <= 1.0
    assert 0.0 <= summary["eer"] <= 1.0
    assert 0.0 <= summary["eer_threshold"] <= 1.0


def test_summarize_labeled_results_reports_eer_threshold_away_from_operating_point():
    # Scores are perfectly ranked but shifted far above the 0.5 operating
    # point, so every sample is accepted there while the EER sits near 0.8.
    results = [
        {"label": 1, "best_qbyt_score": score}
        for score in (0.99, 0.98, 0.97, 0.96)
    ] + [
        {"label": 0, "best_qbyt_score": score}
        for score in (0.95, 0.94, 0.93, 0.92)
    ]

    summary = summarize_labeled_results(results, threshold=0.5)

    assert summary["fpr"] == pytest.approx(1.0)
    assert summary["auc"] == pytest.approx(1.0)
    assert summary["eer"] == pytest.approx(0.0)
    assert summary["eer_threshold"] == pytest.approx(0.96)


def _eer(labels, scores):
    return binary_eer(np.array(labels, dtype=np.int64), np.array(scores, dtype=np.float64))


def test_binary_eer_perfectly_separable():
    assert _eer([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == (pytest.approx(0.0), pytest.approx(0.8))


def test_binary_eer_perfectly_inverted():
    eer, _ = _eer([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1])
    assert eer == pytest.approx(1.0)


def test_binary_eer_all_scores_tied():
    eer, _ = _eer([0, 1], [0.5, 0.5])
    assert eer == pytest.approx(0.5)


def test_binary_eer_random_scores_approach_one_half():
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 2, 20_000)
    scores = rng.random(20_000)

    eer, _ = _eer(labels, scores)

    assert eer == pytest.approx(0.5, abs=0.02)


def test_binary_eer_is_invariant_to_monotone_rescaling():
    labels = [0, 0, 1, 0, 1, 1, 0, 1]
    scores = np.array([0.1, 0.4, 0.35, 0.8, 0.7, 0.9, 0.2, 0.6])

    eer, threshold = _eer(labels, scores)
    rescaled_eer, rescaled_threshold = _eer(labels, scores**2)

    assert rescaled_eer == pytest.approx(eer)
    assert rescaled_threshold == pytest.approx(threshold**2)


def test_binary_eer_is_zero_only_when_auc_is_one():
    labels = [0, 0, 1, 1]
    summary = summarize_labeled_results(
        [
            {"label": label, "best_qbyt_score": score}
            for label, score in zip(labels, [0.1, 0.6, 0.4, 0.9])
        ],
        threshold=0.5,
    )

    assert summary["auc"] < 1.0
    assert summary["eer"] > 0.0


def test_binary_eer_single_class_returns_zero():
    assert _eer([0, 0, 0], [0.1, 0.2, 0.3]) == (0.0, 0.0)


def test_summarize_labeled_results_empty_without_labels():
    results = [{"best_qbyt_score": 0.5, "detected": True}]

    assert summarize_labeled_results(results, threshold=0.5) == {}


def test_two_stage_result_record_includes_manifest_meta_for_extra_columns():
    manifest_row = {
        "audio_path": "/tmp/audio.wav",
        "keyword": "hello",
        "label": "1",
        "speaker_id": "spk-001",
        "source_split": "dev",
    }
    pipeline_result = {
        "detected": True,
        "best_qbyt_score": 0.91,
        "threshold": 0.5,
        "stage1_candidates": [],
        "stage2_scores": [],
    }

    record = two_stage_result_record(manifest_row, pipeline_result)

    assert record["audio_path"] == "/tmp/audio.wav"
    assert record["keyword"] == "hello"
    assert record["label"] == 1
    assert record["manifest_meta"] == {
        "speaker_id": "spk-001",
        "source_split": "dev",
    }


def test_stage2_clip_result_record_includes_manifest_meta_for_extra_columns():
    manifest_row = {
        "audio_path": "/tmp/audio.wav",
        "keyword": "hello",
        "label": "0",
        "keyword_phonemes": "HH AH0 L OW1",
        "text_variant": "hullo",
        "speaker_id": "spk-002",
        "utterance_id": "utt-77",
    }
    runner_result = {
        "qbyt_score": 0.12,
        "detected": False,
        "threshold": 0.5,
        "skipped": False,
        "keyword_phonemes": ["HH", "AH0", "L", "OW1"],
        "text_variant_phonemes": ["HH", "AH0", "L", "OW1"],
    }

    record = stage2_clip_result_record(manifest_row, runner_result)

    assert record["audio_path"] == "/tmp/audio.wav"
    assert record["keyword"] == "hello"
    assert record["label"] == 0
    assert record["keyword_phonemes"] == ["HH", "AH0", "L", "OW1"]
    assert record["text_variant_phonemes"] == ["HH", "AH0", "L", "OW1"]
    assert record["manifest_meta"] == {
        "text_variant": "hullo",
        "speaker_id": "spk-002",
        "utterance_id": "utt-77",
    }


def test_stage2_clip_result_record_preserves_raw_head_logits():
    record = stage2_clip_result_record(
        {"audio_path": "/tmp/audio.wav", "keyword": "hello", "label": 0},
        {
            "qbyt_score": 0.9,
            "qbyt_logit": 2.1972246,
            "keyword_phonemes": ["HH", "AH0"],
            "eps_position_logits": [1.1972246, 3.1972246],
            "completion_score": 0.25,
            "completion_logit": -1.0986123,
            "detected": True,
            "threshold": 0.5,
            "skipped": False,
        },
    )

    assert record["qbyt_logit"] == pytest.approx(2.1972246)
    assert record["keyword_phonemes"] == ["HH", "AH0"]
    assert record["eps_position_logits"] == pytest.approx(
        [1.1972246, 3.1972246]
    )
    assert record["completion_score"] == pytest.approx(0.25)
    assert record["completion_logit"] == pytest.approx(-1.0986123)


def test_stage2_clip_result_record_rejects_non_finite_scores():
    with pytest.raises(ValueError, match="qbyt_logit must be finite"):
        stage2_clip_result_record(
            {"audio_path": "/tmp/audio.wav", "keyword": "hello", "label": 0},
            {
                "qbyt_score": 0.9,
                "qbyt_logit": float("nan"),
                "completion_score": 0.25,
                "completion_logit": -1.0,
                "detected": True,
                "threshold": 0.5,
                "skipped": False,
            },
        )


def test_stage2_clip_result_record_rejects_invalid_eps_position_details():
    base_result = {
        "qbyt_score": 0.5,
        "qbyt_logit": 0.0,
        "keyword_phonemes": ["HH", "EY1"],
        "eps_position_logits": [-1.0, 1.0],
        "detected": True,
        "threshold": 0.5,
        "skipped": False,
    }
    manifest_row = {
        "audio_path": "/tmp/audio.wav",
        "keyword": "hey",
        "label": 0,
    }

    bad_length = dict(base_result, eps_position_logits=[0.0])
    with pytest.raises(ValueError, match="one value per keyword phoneme"):
        stage2_clip_result_record(manifest_row, bad_length)

    bad_mean = dict(base_result, eps_position_logits=[1.0, 1.0])
    with pytest.raises(ValueError, match=r"mean\(eps_position_logits\)"):
        stage2_clip_result_record(manifest_row, bad_mean)

    bad_finite = dict(base_result, eps_position_logits=[float("nan"), 0.0])
    with pytest.raises(ValueError, match=r"eps_position_logits\[0\] must be finite"):
        stage2_clip_result_record(manifest_row, bad_finite)
