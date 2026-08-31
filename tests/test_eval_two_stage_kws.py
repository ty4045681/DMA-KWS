import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from omegaconf import OmegaConf

import dma_kws.inference.detection_plots as detection_plots
from dma_kws.inference.detection_plots import select_roc_constraint_point
from dma_kws.inference.manifest import load_manifest
from dma_kws.inference.metrics import binary_eer, summarize_labeled_results
import scripts.eval_stage2_clips as eval_stage2_clips
from scripts.eval_stage2_clips import (
    _binary_roc_points,
    _result_record as stage2_clip_result_record,
    _resolve_audio_padding_ms,
    _score_provenance,
    _write_detection_plots,
)
from scripts.eval_two_stage_kws import _result_record as two_stage_result_record


FIXTURES = Path(__file__).resolve().parent / "fixtures"


def _qbyt_alignment() -> dict:
    return {
        "topology": "bounded_segmental_v1",
        "min_phone_duration_frames": 1,
        "max_phone_duration_frames": 8,
        "max_inter_phone_gap_frames": 2,
        "max_keyword_span_frames": 30,
        "temperature": 0.2,
        "local_context_kernel": 5,
    }


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


def test_binary_roc_points_keep_tied_scores_at_one_operating_point():
    records = [
        {"label": 1, "qbyt_score": 0.9},
        {"label": 0, "qbyt_score": 0.8},
        {"label": 1, "qbyt_score": 0.8},
        {"label": 0, "qbyt_score": 0.1},
        {"label": 0, "qbyt_score": 1.0, "skipped": True},
    ]

    curve = _binary_roc_points(records, score_field="qbyt_score")

    assert curve is not None
    assert curve["num_samples"] == 4
    assert curve["num_positive"] == 2
    assert curve["num_negative"] == 2
    assert curve["thresholds"] == pytest.approx([np.inf, 0.9, 0.8, 0.1])
    assert curve["fpr"] == pytest.approx([0.0, 0.0, 0.5, 1.0])
    assert curve["tpr"] == pytest.approx([0.0, 0.5, 1.0, 1.0])


@pytest.mark.parametrize(
    ("constraint", "expected"),
    [
        (
            {"min_recall": 0.75},
            {
                "kind": "min_recall",
                "metric": "recall",
                "requested": 0.75,
                "exact": False,
                "actual_recall": 0.8,
                "actual_fpr": 0.1,
                "guide_recall": 0.75,
                "guide_fpr": 0.1,
                "threshold": 0.8,
            },
        ),
        (
            {"min_recall": 0.8},
            {
                "kind": "min_recall",
                "metric": "recall",
                "requested": 0.8,
                "exact": True,
                "actual_recall": 0.8,
                "actual_fpr": 0.1,
                "guide_recall": 0.8,
                "guide_fpr": 0.1,
                "threshold": 0.8,
            },
        ),
        (
            {"max_fpr": 0.25},
            {
                "kind": "max_fpr",
                "metric": "fpr",
                "requested": 0.25,
                "exact": False,
                "actual_recall": 0.8,
                "actual_fpr": 0.1,
                "guide_recall": 0.8,
                "guide_fpr": 0.25,
                "threshold": 0.8,
            },
        ),
        (
            {"max_fpr": 0.1},
            {
                "kind": "max_fpr",
                "metric": "fpr",
                "requested": 0.1,
                "exact": True,
                "actual_recall": 0.8,
                "actual_fpr": 0.1,
                "guide_recall": 0.8,
                "guide_fpr": 0.1,
                "threshold": 0.8,
            },
        ),
    ],
)
def test_select_roc_constraint_point_uses_scan_selector_tie_breaks(
    constraint, expected
):
    curve = {
        "fpr": np.asarray([0.0, 0.05, 0.1, 0.1, 0.2, 0.3]),
        "tpr": np.asarray([0.0, 0.5, 0.8, 0.8, 0.8, 0.9]),
        "thresholds": np.asarray([np.inf, 0.9, 0.8, 0.7, 0.6, 0.5]),
    }

    point = select_roc_constraint_point(curve, **constraint)

    assert point == expected


def test_select_roc_constraint_point_returns_none_without_constraint():
    curve = {
        "fpr": np.asarray([0.0, 1.0]),
        "tpr": np.asarray([0.0, 1.0]),
        "thresholds": np.asarray([np.inf, 0.5]),
    }

    assert select_roc_constraint_point(curve) is None


@pytest.mark.parametrize(
    ("constraint", "expected_actual", "expected_guide"),
    [
        ({"min_recall": 0.0}, (0.5, 0.0), (0.0, 0.0)),
        ({"max_fpr": 1.0}, (1.0, 0.5), (1.0, 1.0)),
    ],
)
def test_select_roc_constraint_point_marks_empirical_endpoints_exact(
    constraint,
    expected_actual,
    expected_guide,
):
    curve = {
        "fpr": np.asarray([0.0, 0.0, 0.5, 1.0]),
        "tpr": np.asarray([0.0, 0.5, 1.0, 1.0]),
        "thresholds": np.asarray([np.inf, 0.9, 0.8, 0.1]),
    }

    point = select_roc_constraint_point(curve, **constraint)

    assert point is not None
    assert point["exact"] is True
    assert (point["actual_recall"], point["actual_fpr"]) == expected_actual
    assert (point["guide_recall"], point["guide_fpr"]) == expected_guide


def test_select_roc_constraint_point_rejects_multiple_constraints():
    curve = {
        "fpr": np.asarray([0.0, 1.0]),
        "tpr": np.asarray([0.0, 1.0]),
        "thresholds": np.asarray([np.inf, 0.5]),
    }

    with pytest.raises(ValueError):
        select_roc_constraint_point(curve, min_recall=0.5, max_fpr=0.5)


@pytest.mark.parametrize(
    "constraint",
    [
        {"min_recall": -0.01},
        {"min_recall": 1.01},
        {"min_recall": np.nan},
        {"min_recall": np.inf},
        {"max_fpr": -0.01},
        {"max_fpr": 1.01},
        {"max_fpr": np.nan},
        {"max_fpr": np.inf},
    ],
)
def test_select_roc_constraint_point_rejects_invalid_rates(constraint):
    curve = {
        "fpr": np.asarray([0.0, 1.0]),
        "tpr": np.asarray([0.0, 1.0]),
        "thresholds": np.asarray([np.inf, 0.5]),
    }

    with pytest.raises(ValueError):
        select_roc_constraint_point(curve, **constraint)


def test_detection_plots_skip_single_class_without_creating_files(tmp_path):
    plot_summary = _write_detection_plots(
        [
            {"label": 0, "qbyt_score": 0.2},
            {"label": 0, "qbyt_score": 0.1},
        ],
        output_dir=tmp_path,
        threshold=0.5,
        metrics={"auc": 0.0, "eer": 0.0, "eer_threshold": 0.0},
    )

    assert plot_summary["status"] == "skipped"
    assert "positive and negative" in plot_summary["reason"]
    assert not (tmp_path / "roc_curve.png").exists()
    assert not (tmp_path / "det_curve.png").exists()
    assert not (tmp_path / "roc_curve.csv").exists()


def test_detection_plots_write_roc_and_det_pngs(tmp_path):
    pytest.importorskip("matplotlib")
    records = [
        {"label": 1, "qbyt_score": 0.95},
        {"label": 0, "qbyt_score": 0.75},
        {"label": 1, "qbyt_score": 0.65},
        {"label": 0, "qbyt_score": 0.10},
    ]
    metrics = summarize_labeled_results(
        [
            {"label": record["label"], "best_qbyt_score": record["qbyt_score"]}
            for record in records
        ],
        threshold=0.5,
    )

    plot_summary = _write_detection_plots(
        records,
        output_dir=tmp_path,
        threshold=0.5,
        metrics=metrics,
        dpi=72,
    )

    roc_path = tmp_path / "roc_curve.png"
    det_path = tmp_path / "det_curve.png"
    csv_path = tmp_path / "roc_curve.csv"
    assert plot_summary == {
        "status": "generated",
        "score_field": "qbyt_score",
        "num_samples": 4,
        "num_positive": 2,
        "num_negative": 2,
        "roc": str(roc_path.resolve()),
        "det": str(det_path.resolve()),
        "roc_curve_csv": str(csv_path.resolve()),
    }
    assert roc_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert det_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    curve = _binary_roc_points(records, score_field="qbyt_score")
    assert curve is not None
    csv_rows = csv_path.read_text(encoding="utf-8").strip().splitlines()
    assert csv_rows[0] == "threshold,tpr,fpr"
    parsed = [tuple(row.split(",")) for row in csv_rows[1:]]
    assert [row[0] for row in parsed] == ["inf", "0.95", "0.75", "0.65", "0.1"]
    assert [float(row[1]) for row in parsed] == pytest.approx(curve["tpr"].tolist())
    assert [float(row[2]) for row in parsed] == pytest.approx(curve["fpr"].tolist())


def test_detection_plots_write_constraint_marker_and_summary(tmp_path, monkeypatch):
    pytest.importorskip("matplotlib")
    guide_calls = []
    real_draw_constraint_guides = detection_plots._draw_constraint_guides

    def capture_constraint_guides(axis, **kwargs):
        guide_calls.append(kwargs)
        return real_draw_constraint_guides(axis, **kwargs)

    monkeypatch.setattr(
        detection_plots,
        "_draw_constraint_guides",
        capture_constraint_guides,
    )
    records = [
        {"label": 1, "qbyt_score": 0.95},
        {"label": 0, "qbyt_score": 0.75},
        {"label": 1, "qbyt_score": 0.65},
        {"label": 0, "qbyt_score": 0.10},
    ]
    metrics = summarize_labeled_results(
        [
            {"label": record["label"], "best_qbyt_score": record["qbyt_score"]}
            for record in records
        ],
        threshold=0.5,
    )

    plot_summary = _write_detection_plots(
        records,
        output_dir=tmp_path,
        threshold=0.5,
        metrics=metrics,
        dpi=72,
        min_recall=0.75,
    )

    assert plot_summary["constraint"] == {
        "kind": "min_recall",
        "metric": "recall",
        "requested": 0.75,
        "exact": False,
        "actual_recall": 1.0,
        "actual_fpr": 0.5,
        "guide_recall": 0.75,
        "guide_fpr": 0.5,
        "threshold": 0.65,
    }
    assert len(guide_calls) == 2
    assert guide_calls[0]["x"] == pytest.approx(0.5)
    assert guide_calls[0]["y"] == pytest.approx(0.75)
    assert guide_calls[0]["x_label"] == "FPR=0.5000"
    assert guide_calls[0]["y_label"] == "Recall=0.7500"
    assert guide_calls[1]["x"] == pytest.approx(0.0)
    assert guide_calls[1]["y"] < 0.0
    assert guide_calls[1]["x_label"] == "FPR=50.00%"
    assert guide_calls[1]["y_label"] == "FNR=25.00%\nRecall=75.00%"
    assert (tmp_path / "roc_curve.png").read_bytes().startswith(
        b"\x89PNG\r\n\x1a\n"
    )
    assert (tmp_path / "det_curve.png").read_bytes().startswith(
        b"\x89PNG\r\n\x1a\n"
    )


def test_stage2_clip_eval_applies_and_records_default_padding(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    checkpoint_path = tmp_path / "stage2.pt"
    torch.save(
        {
            "config": {
                "stage2": {
                    "qbyt_alignment": _qbyt_alignment(),
                    "sequence_loss": {
                        "target_mode": "ordered_contiguous_prefix",
                        "progress_weight": 0.5,
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
            "label": 1,
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
                    "keyword_phonemes": ["HH"],
                    "detected": True,
                    "threshold": 0.5,
                    "skipped": False,
                    "augmented_duration_sec": 1.25,
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
                "qbyt_alignment": _qbyt_alignment(),
                "sequence_loss": {
                    "target_mode": "ordered_contiguous_prefix",
                    "progress_weight": 0.5,
                    "normalization": "sample",
                },
                "validation": {
                    "ece_num_bins": 9,
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
    real_write_detection_plots = eval_stage2_clips._write_detection_plots

    def capture_plot_options(*args, **kwargs):
        captured["plot_options"] = kwargs
        return real_write_detection_plots(*args, **kwargs)

    monkeypatch.setattr(
        eval_stage2_clips,
        "_write_detection_plots",
        capture_plot_options,
    )

    cfg = OmegaConf.create(
        {
            "prep": {
                "manifest": "manifest.csv",
                "stage2_ckpt": str(checkpoint_path),
                "output_dir": str(tmp_path),
                "num_workers": 1,
                "plot_min_recall": 0.8,
                "plot_max_fpr": None,
            },
            "run": {"device": "cpu"},
        }
    )

    summary = eval_stage2_clips.run_eval(cfg)

    assert captured["rows"] == rows
    assert captured["kwargs"]["left_padding_ms"] == 160
    assert captured["kwargs"]["right_padding_ms"] == 160
    assert captured["kwargs"]["waveform_transform"] is None
    assert captured["plot_options"]["min_recall"] == pytest.approx(0.8)
    assert captured["plot_options"]["max_fpr"] is None
    assert summary["audio_padding_ms"] == {"left": 160, "right": 160}
    assert summary["audio_aug"]["enabled"] is False
    assert summary["musan_mix"]["enabled"] is False
    assert summary["provenance"]["checkpoint"]["path"] == str(
        checkpoint_path.resolve()
    )
    assert summary["provenance"]["checkpoint"]["sha256"] == hashlib.sha256(
        checkpoint_path.read_bytes()
    ).hexdigest()
    assert summary["provenance"]["sequence_objective"]["target_mode"] == (
        "ordered_contiguous_prefix"
    )
    assert summary["provenance"]["qbyt_alignment"] == _qbyt_alignment()
    assert summary["score_diagnostic_config"] == {
        "threshold": 0.5,
        "ece_num_bins": 9,
    }
    assert summary["plots"]["status"] == "skipped"
    assert "positive and negative" in summary["plots"]["reason"]
    saved_summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert saved_summary["audio_padding_ms"] == {"left": 160, "right": 160}
    assert saved_summary["audio_aug"]["enabled"] is False
    assert saved_summary["musan_mix"]["enabled"] is False
    assert saved_summary["plots"] == summary["plots"]
    saved_result = json.loads(
        (tmp_path / "results.jsonl").read_text(encoding="utf-8").strip()
    )
    assert saved_result["keyword_phonemes"] == ["HH"]
    assert saved_result["manifest_meta"] == {"text_variant": "hullo"}
    assert saved_result["qbyt_score"] == pytest.approx(0.75)
    assert saved_result["augmented_duration_sec"] == pytest.approx(1.25)


def test_stage2_clip_eval_records_enabled_waveform_augmentations(
    tmp_path, monkeypatch
):
    pytest.importorskip("torch")
    captured = {}
    rows = [{"audio_path": "clip.wav", "keyword": "hello"}]

    class FakeMixer:
        enabled = True

        @classmethod
        def from_prep(cls, prep, *, audio_paths):
            captured["mixer_prep"] = prep
            captured["mixer_audio_paths"] = audio_paths
            return cls()

        def __call__(self, index, waveform, sample_rate):
            del index, sample_rate
            return waveform

        @staticmethod
        def summary():
            return {
                "enabled": True,
                "seed": 7,
                "stationary_noise": {
                    "enabled": True,
                    "kind": "white_gaussian",
                    "snr_db": 18.0,
                },
                "burst_noise": {
                    "enabled": True,
                    "snr_db": 8.0,
                    "snr_scope": "active_event",
                    "event_count_min": 1,
                    "event_count_max": 1,
                    "duration_ms_min": 100.0,
                    "duration_ms_max": 200.0,
                    "fade_ms": 10.0,
                    "allow_overlap": False,
                    "min_gap_ms": 50.0,
                    "num_files": 3,
                },
                "volume_variation": {
                    "enabled": True,
                    "low_gain_db": -9.0,
                    "high_gain_db": 3.0,
                    "segment_ms_min": 200.0,
                    "segment_ms_max": 600.0,
                    "transition_ms": 40.0,
                },
                "processing_order": (
                    "volume_variation_then_scale_all_additive_components_against_"
                    "the_same_varied_clean_then_sum_without_clipping"
                ),
            }

        @staticmethod
        def recipe_metadata(index):
            return {
                "row_index": index,
                "noise": {"source": "noise/sample.wav", "snr_db": 10.0},
                "music": {"source": "music/sample.wav", "snr_db": 12.0},
                "stationary_noise": {
                    "kind": "white_gaussian",
                    "snr_db": 18.0,
                    "recipe_seed": 101,
                },
                "burst_noise": {
                    "snr_db": 8.0,
                    "snr_scope": "active_event",
                    "recipe_seed": 102,
                    "event_count": 1,
                    "events": [
                        {
                            "event_index": 0,
                            "source": "noise/burst.wav",
                            "duration_ms": 120.0,
                            "start_fraction": 0.25,
                            "source_offset_fraction": 0.5,
                            "recipe_seed": 104,
                        }
                    ],
                },
                "volume_variation": {
                    "low_gain_db": -9.0,
                    "high_gain_db": 3.0,
                    "segment_ms_min": 200.0,
                    "segment_ms_max": 600.0,
                    "transition_ms": 40.0,
                    "recipe_seed": 103,
                },
            }

    class FakeAudioAug:
        enabled = True

        @classmethod
        def from_prep(cls, prep, *, audio_paths):
            captured["audio_aug_prep"] = prep
            captured["audio_aug_audio_paths"] = audio_paths
            return cls()

        @staticmethod
        def summary():
            return {"enabled": True, "seed": 11}

        @staticmethod
        def recipe_metadata(index):
            return {
                "row_index": index,
                "applied_order": ["volume_gain"],
                "transforms": {"volume_gain": {"gain_db": 3.0}},
            }

    class FakePipeline:
        def __init__(self, audio_aug, musan_mixer):
            captured["pipeline_audio_aug"] = audio_aug
            captured["pipeline_musan_mixer"] = musan_mixer
            self.audio_aug = audio_aug
            self.musan_mixer = musan_mixer
            self.enabled = audio_aug.enabled or musan_mixer.enabled

        def __call__(self, index, waveform, sample_rate):
            del index, sample_rate
            return waveform

        def summary(self):
            return {
                "audio_aug": self.audio_aug.summary(),
                "musan_mix": self.musan_mixer.summary(),
            }

        def recipe_metadata(self, index):
            return {
                "audio_aug": self.audio_aug.recipe_metadata(index),
                "musan_mix": self.musan_mixer.recipe_metadata(index),
            }

    class FakeAudioExporter:
        enabled = True

        @classmethod
        def from_prep(cls, prep, *, output_dir, audio_paths):
            captured["audio_export_prep"] = prep
            captured["audio_export_dir"] = output_dir
            captured["audio_export_paths"] = audio_paths
            return cls()

        def prepare(self):
            captured["audio_export_prepared"] = True

        @staticmethod
        def finalize():
            return {
                "status": "generated",
                "mode": "random",
                "requested_count": 1,
                "seed": 13,
                "num_selected": 1,
                "num_exported": 1,
                "stage": "post_augmentation_pre_padding",
                "format": "WAV",
                "subtype": "FLOAT",
                "directory": "/exports",
                "manifest": "/exports/index.jsonl",
                "row_indices": [0],
            }

        @staticmethod
        def result_path(index):
            return "/exports/row_00000000.wav" if index == 0 else None

    class FakeStreamPolicy:
        @staticmethod
        def describe():
            return {"mode": "test"}

    class FakeRunner:
        _demo_cfg = {"qbyt_threshold": 0.5}
        stream_policy = FakeStreamPolicy()

        def run_batch(self, batch_rows, **kwargs):
            captured["rows"] = batch_rows
            captured["run_kwargs"] = kwargs
            return [
                {
                    "qbyt_score": 0.25,
                    "keyword_phonemes": ["HH"],
                    "detected": False,
                    "threshold": 0.5,
                    "skipped": False,
                }
            ]

    class FakeRunnerFactory:
        @staticmethod
        def from_config(_config, _prep, _device):
            return FakeRunner()

    monkeypatch.setattr(
        eval_stage2_clips,
        "resolved_config",
        lambda _cfg: {
            "paths": {},
            "stage1": {},
            "stage2": {"validation": {}},
            "demo": {},
            "tokenizer": {},
        },
    )
    monkeypatch.setattr(eval_stage2_clips, "load_manifest", lambda _path: rows)
    monkeypatch.setattr(
        eval_stage2_clips, "AudioAugWaveformTransform", FakeAudioAug
    )
    monkeypatch.setattr(eval_stage2_clips, "MusanWaveformMixer", FakeMixer)
    monkeypatch.setattr(
        eval_stage2_clips, "WaveformAugmentationPipeline", FakePipeline
    )
    monkeypatch.setattr(
        eval_stage2_clips, "SelectedWaveformExporter", FakeAudioExporter
    )
    monkeypatch.setattr(eval_stage2_clips, "Stage2ClipRunner", FakeRunnerFactory)
    monkeypatch.setattr(
        eval_stage2_clips, "resolve_accelerator", lambda _device: ("cpu", 1)
    )
    monkeypatch.setattr(
        eval_stage2_clips,
        "_score_provenance",
        lambda *_args, **_kwargs: {
            "qbyt_alignment": _qbyt_alignment(),
            "sequence_objective": {
                "target_mode": "ordered_contiguous_prefix",
                "progress_weight": 0.5,
                "normalization": "sample",
            },
        },
    )

    cfg = OmegaConf.create(
        {
            "prep": {
                "manifest": "manifest.csv",
                "stage2_ckpt": "stage2.pt",
                "output_dir": str(tmp_path),
                "audio_export": {"mode": "random", "count": 1, "seed": 13},
                "musan_root": "/musan",
                "musan_mix": {
                    "seed": 7,
                    "noise": {"enabled": True, "snr_db": 10.0},
                    "music": {"enabled": True, "snr_db": 12.0},
                    "speech": {"enabled": False, "relative_db": 0.0},
                    "stationary_noise": {
                        "enabled": True,
                        "kind": "white_gaussian",
                        "snr_db": 18.0,
                    },
                    "burst_noise": {
                        "enabled": True,
                        "snr_db": 8.0,
                        "snr_scope": "active_event",
                        "event_count_min": 1,
                        "event_count_max": 1,
                        "duration_ms_min": 100.0,
                        "duration_ms_max": 200.0,
                        "fade_ms": 10.0,
                        "allow_overlap": False,
                        "min_gap_ms": 50.0,
                    },
                    "volume_variation": {
                        "enabled": True,
                        "low_gain_db": -9.0,
                        "high_gain_db": 3.0,
                        "segment_ms_min": 200.0,
                        "segment_ms_max": 600.0,
                        "transition_ms": 40.0,
                    },
                },
                "audio_aug": {
                    "seed": 11,
                    "transforms": {
                        "volume_gain": {"enabled": True, "gain_db": 3.0}
                    },
                },
            },
            "run": {"device": "cpu"},
        }
    )

    summary = eval_stage2_clips.run_eval(cfg)

    assert captured["audio_aug_audio_paths"] == ["clip.wav"]
    assert captured["mixer_audio_paths"] == ["clip.wav"]
    assert captured["mixer_prep"]["musan_mix"]["stationary_noise"] == {
        "enabled": True,
        "kind": "white_gaussian",
        "snr_db": 18.0,
    }
    assert captured["mixer_prep"]["musan_mix"]["volume_variation"] == {
        "enabled": True,
        "low_gain_db": -9.0,
        "high_gain_db": 3.0,
        "segment_ms_min": 200.0,
        "segment_ms_max": 600.0,
        "transition_ms": 40.0,
    }
    assert captured["rows"] == rows
    assert isinstance(captured["run_kwargs"]["waveform_transform"], FakePipeline)
    assert isinstance(
        captured["run_kwargs"]["waveform_observer"], FakeAudioExporter
    )
    assert captured["audio_export_prepared"] is True
    assert captured["audio_export_paths"] == ["clip.wav"]
    assert summary["audio_aug"] == {"enabled": True, "seed": 11}
    assert summary["musan_mix"] == FakeMixer.summary()
    assert summary["audio_exports"]["num_exported"] == 1
    saved_result = json.loads(
        (tmp_path / "results.jsonl").read_text(encoding="utf-8").strip()
    )
    assert saved_result["musan_mix"] == {
        "row_index": 0,
        "noise": {"source": "noise/sample.wav", "snr_db": 10.0},
        "music": {"source": "music/sample.wav", "snr_db": 12.0},
        "stationary_noise": {
            "kind": "white_gaussian",
            "snr_db": 18.0,
            "recipe_seed": 101,
        },
        "burst_noise": {
            "snr_db": 8.0,
            "snr_scope": "active_event",
            "recipe_seed": 102,
            "event_count": 1,
            "events": [
                {
                    "event_index": 0,
                    "source": "noise/burst.wav",
                    "duration_ms": 120.0,
                    "start_fraction": 0.25,
                    "source_offset_fraction": 0.5,
                    "recipe_seed": 104,
                }
            ],
        },
        "volume_variation": {
            "low_gain_db": -9.0,
            "high_gain_db": 3.0,
            "segment_ms_min": 200.0,
            "segment_ms_max": 600.0,
            "transition_ms": 40.0,
            "recipe_seed": 103,
        },
    }
    assert saved_result["audio_aug"] == {
        "row_index": 0,
        "applied_order": ["volume_gain"],
        "transforms": {"volume_gain": {"gain_db": 3.0}},
    }
    assert saved_result["exported_audio_path"] == "/exports/row_00000000.wav"


def test_legacy_musan_mix_schema_remains_valid_without_new_sections(tmp_path):
    noise_dir = tmp_path / "noise"
    noise_dir.mkdir()
    (noise_dir / "legacy.wav").write_bytes(b"legacy source placeholder")
    mixer = eval_stage2_clips.MusanWaveformMixer.from_prep(
        {
            "musan_root": str(tmp_path),
            "musan_mix": {
                "seed": 7,
                "noise": {"enabled": True, "snr_db": 10.0},
                "music": {"enabled": False, "snr_db": 12.0},
                "speech": {"enabled": False, "relative_db": 3.0},
            }
        },
        audio_paths=["clip.wav"],
    )

    summary = mixer.summary()
    assert mixer.enabled is True
    assert summary["musan_root"] == str(tmp_path.resolve())
    assert summary["noise"]["num_files"] == 1
    assert summary["stationary_noise"]["enabled"] is False
    assert summary["burst_noise"]["enabled"] is False
    assert summary["volume_variation"]["enabled"] is False


@pytest.mark.parametrize(
    ("section", "settings"),
    [
        (
            "stationary_noise",
            {
                "enabled": True,
                "kind": "white_gaussian",
                "snr_db": 18.0,
            },
        ),
        (
            "volume_variation",
            {
                "enabled": True,
                "low_gain_db": -9.0,
                "high_gain_db": 3.0,
                "segment_ms_min": 200.0,
                "segment_ms_max": 600.0,
                "transition_ms": 40.0,
            },
        ),
    ],
)
def test_synthetic_or_volume_only_musan_pipeline_does_not_require_root(
    section,
    settings,
):
    torch = pytest.importorskip("torch")
    mixer = eval_stage2_clips.MusanWaveformMixer.from_prep(
        {"musan_mix": {"seed": 19, section: settings}},
        audio_paths=["clip.wav"],
    )
    pipeline = eval_stage2_clips.WaveformAugmentationPipeline(
        musan_mixer=mixer,
    )
    waveform = torch.linspace(-0.4, 0.4, 16000).unsqueeze(0)

    output = pipeline(0, waveform, 16000)
    summary = pipeline.summary()["musan_mix"]
    metadata = pipeline.recipe_metadata(0)["musan_mix"]

    assert pipeline.enabled is True
    assert summary["musan_root"] is None
    assert summary[section]["enabled"] is True
    assert section in metadata
    assert output.shape == waveform.shape
    assert torch.isfinite(output).all()
    assert not torch.equal(output, waveform)


def test_stage2_clip_eval_reports_invalid_audio_aug_config(tmp_path, monkeypatch):
    pytest.importorskip("torch")

    class InvalidAudioAug:
        @classmethod
        def from_prep(cls, _prep, *, audio_paths):
            assert audio_paths == ["clip.wav"]
            raise ValueError(
                "signal_mimic overlaps enabled atomic transforms; "
                "set allow_signal_mimic_overlap=true to allow it"
            )

    monkeypatch.setattr(
        eval_stage2_clips,
        "resolved_config",
        lambda _cfg: {
            "paths": {},
            "stage1": {},
            "stage2": {},
            "demo": {},
            "tokenizer": {},
        },
    )
    monkeypatch.setattr(
        eval_stage2_clips,
        "load_manifest",
        lambda _path: [{"audio_path": "clip.wav", "keyword": "hello"}],
    )
    monkeypatch.setattr(
        eval_stage2_clips, "AudioAugWaveformTransform", InvalidAudioAug
    )
    cfg = OmegaConf.create(
        {
            "prep": {
                "manifest": "manifest.csv",
                "stage2_ckpt": "stage2.pt",
                "output_dir": str(tmp_path),
                "audio_aug": {
                    "allow_signal_mimic_overlap": False,
                    "transforms": {
                        "subband_eq": {"enabled": True},
                        "signal_mimic": {"enabled": True},
                    },
                },
            },
            "run": {"device": "cpu"},
        }
    )

    with pytest.raises(
        SystemExit,
        match="Invalid prep.audio_aug configuration: signal_mimic overlaps",
    ):
        eval_stage2_clips.run_eval(cfg)


def test_stage2_clip_eval_reports_invalid_musan_mix_context(tmp_path, monkeypatch):
    pytest.importorskip("torch")

    class InvalidMusanMixer:
        @classmethod
        def from_prep(cls, prep, *, audio_paths):
            assert audio_paths == ["clip.wav"]
            assert prep["musan_mix"]["stationary_noise"]["enabled"] is True
            raise ValueError(
                "prep.musan_mix.stationary_noise.kind must be white_gaussian"
            )

    monkeypatch.setattr(
        eval_stage2_clips,
        "resolved_config",
        lambda _cfg: {
            "paths": {},
            "stage1": {},
            "stage2": {},
            "demo": {},
            "tokenizer": {},
        },
    )
    monkeypatch.setattr(
        eval_stage2_clips,
        "load_manifest",
        lambda _path: [{"audio_path": "clip.wav", "keyword": "hello"}],
    )
    monkeypatch.setattr(
        eval_stage2_clips,
        "MusanWaveformMixer",
        InvalidMusanMixer,
    )
    cfg = OmegaConf.create(
        {
            "prep": {
                "manifest": "manifest.csv",
                "stage2_ckpt": "stage2.pt",
                "output_dir": str(tmp_path),
                "musan_mix": {
                    "stationary_noise": {
                        "enabled": True,
                        "kind": "pink_gaussian",
                        "snr_db": 20.0,
                    }
                },
            },
            "run": {"device": "cpu"},
        }
    )

    with pytest.raises(
        SystemExit,
        match="Invalid prep.musan_mix configuration:",
    ):
        eval_stage2_clips.run_eval(cfg)


def test_score_provenance_records_alignment_and_checkpoint_objective(tmp_path):
    torch = pytest.importorskip("torch")
    checkpoint_path = tmp_path / "stage2.pt"
    torch.save(
        {
            "config": {
                "stage2": {
                    "sequence_loss": {
                        "target_mode": "ordered_contiguous_prefix",
                        "progress_weight": 0.5,
                        "normalization": "sample",
                    }
                }
            }
        },
        checkpoint_path,
    )
    tokenizer_path = tmp_path / "lang_char.txt"
    tokenizer_path.write_text("<blank> 0\nHH 1\n", encoding="utf-8")
    config = {
        "stage1": {},
        "stage2": {"qbyt_alignment": _qbyt_alignment()},
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
        "target_mode": "ordered_contiguous_prefix",
        "progress_weight": 0.5,
        "normalization": "sample",
    }
    assert provenance["qbyt_alignment"] == _qbyt_alignment()
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
    }

    record = stage2_clip_result_record(manifest_row, runner_result)

    assert record["audio_path"] == "/tmp/audio.wav"
    assert record["keyword"] == "hello"
    assert record["label"] == 0
    assert record["keyword_phonemes"] == ["HH", "AH0", "L", "OW1"]
    assert record["manifest_meta"] == {
        "text_variant": "hullo",
        "speaker_id": "spk-002",
        "utterance_id": "utt-77",
    }


def test_stage2_clip_result_record_exports_only_the_deployed_score():
    record = stage2_clip_result_record(
        {"audio_path": "/tmp/audio.wav", "keyword": "hello", "label": 0},
        {
            "qbyt_score": 0.9,
            "keyword_phonemes": ["HH", "AH0"],
            "detected": True,
            "threshold": 0.5,
            "skipped": False,
        },
    )

    assert record == {
        "audio_path": "/tmp/audio.wav",
        "keyword": "hello",
        "keyword_phonemes": ["HH", "AH0"],
        "qbyt_score": pytest.approx(0.9),
        "detected": True,
        "threshold": pytest.approx(0.5),
        "skipped": False,
        "label": 0,
    }


@pytest.mark.parametrize("field", ["qbyt_score", "threshold"])
def test_stage2_clip_result_record_rejects_non_finite_deployed_score(field):
    runner_result = {
        "qbyt_score": 0.9,
        "keyword_phonemes": ["HH"],
        "detected": True,
        "threshold": 0.5,
        "skipped": False,
    }
    runner_result[field] = float("nan")

    with pytest.raises(ValueError, match=rf"{field} must be finite"):
        stage2_clip_result_record(
            {"audio_path": "/tmp/audio.wav", "keyword": "hello", "label": 0},
            runner_result,
        )
