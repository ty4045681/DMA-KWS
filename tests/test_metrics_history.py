"""Dense metrics-history CSV helpers and run records (no torch required)."""

import csv
import sys
from pathlib import Path
from types import SimpleNamespace

from dma_kws.training.metrics_history import (
    append_wide_row,
    build_metrics_history_callback,
    build_run_record,
    collect_hparams,
    metric_direction,
    numeric_callback_metrics,
    tracks_best_metric,
    update_best_metrics,
)

CONFIG = {
    "paths": {"exp_root": "/tmp/exp"},
    "training": {"seed": 7, "recipe": "adapt"},
    "stage2": {
        "learning_rate": 5e-4,
        "warmup_steps": 2500,
        "max_steps": 50000,
        "batch_size_per_gpu": 64,
        "accumulate_grad_batches": 2,
        "qbyt_alignment": {
            "topology": "keyword_filler_segmental_crf_v1",
            "min_phone_duration_frames": 1,
            "max_phone_duration_frames": 8,
            "max_inter_phone_gap_frames": 1,
            "max_keyword_span_frames": 30,
            "local_context_kernel": 5,
            "weakest_phone_temperature": 0.2,
            "weakest_phone_weight": 1.0,
        },
        "sequence_loss": {
            "target_mode": "ordered_contiguous_prefix",
            "progress_weight": 0.3,
            "normalization": "sample",
        },
        "validation": {"val_check_interval": 1000},
    },
    "adapt": {
        "lr": None,
        "learning_rate": 4e-4,
        "optimizer": "adamw",
        "warmup_steps": 100,
        "max_steps": 3000,
        "batch_size_per_gpu": 32,
        "rank": 8,
        "alpha": 16,
        "mix_ratio": 0.6,
        "keyword": "hey eva",
        "phase": "tts",
    },
}


def _read_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_append_wide_row_creates_file_with_header(tmp_path: Path):
    path = tmp_path / "eval_history.csv"
    append_wide_row(path, {"run": "r1", "step": 100, "val/auc": 0.912345678})

    rows = _read_rows(path)
    assert len(rows) == 1
    assert rows[0]["run"] == "r1"
    assert rows[0]["step"] == "100"
    assert rows[0]["val/auc"] == "0.912346"


def test_append_wide_row_appends_matching_columns(tmp_path: Path):
    path = tmp_path / "history.csv"
    append_wide_row(path, {"run": "r1", "step": 100, "val/auc": 0.9})
    append_wide_row(path, {"run": "r1", "step": 200, "val/auc": 0.95})

    rows = _read_rows(path)
    assert [row["step"] for row in rows] == ["100", "200"]


def test_append_wide_row_expands_header_for_new_columns(tmp_path: Path):
    path = tmp_path / "history.csv"
    append_wide_row(path, {"run": "r1", "step": 100, "val/auc": 0.9})
    append_wide_row(path, {"run": "r1", "step": 200, "val/auc": 0.95, "val/eer": 0.08})

    rows = _read_rows(path)
    assert rows[0]["val/eer"] == ""
    assert rows[1]["val/eer"] == "0.08"
    with path.open(newline="", encoding="utf-8") as handle:
        header = next(csv.reader(handle))
    assert header == ["run", "step", "val/auc", "val/eer"]


def test_append_wide_row_none_becomes_empty(tmp_path: Path):
    path = tmp_path / "history.csv"
    append_wide_row(path, {"run": "r1", "note": None})
    assert _read_rows(path)[0]["note"] == ""


def test_metric_direction():
    assert metric_direction("val/auc") == "max"
    assert metric_direction("val/target_auc") == "max"
    assert metric_direction("val/eer") == "min"
    assert metric_direction("val/utt_loss") == "min"


def test_best_tracking_excludes_thresholds_counts_and_score_quantiles():
    assert tracks_best_metric("val/auc")
    assert tracks_best_metric("val/tpr_at_fpr_1e-3")
    assert tracks_best_metric("val/deploy_fpr")
    assert not tracks_best_metric("val/eer_threshold")
    assert not tracks_best_metric("val/score_neg_p95")
    assert not tracks_best_metric("val/num_positive")

    best: dict[str, float] = {}
    updated = update_best_metrics(
        best,
        {
            "val/eer_threshold": 0.97,
            "val/score_neg_p95": 0.99,
            "val/auc": 0.94,
        },
    )
    assert updated == {"val/auc"}
    assert best == {"val/auc": 0.94}


def test_update_best_metrics_honors_direction():
    best: dict[str, float] = {}
    first_updated = update_best_metrics(best, {"val/auc": 0.9, "val/eer": 0.1})
    second_updated = update_best_metrics(best, {"val/auc": 0.85, "val/eer": 0.05})

    assert first_updated == {"val/auc", "val/eer"}
    assert second_updated == {"val/eer"}
    assert best == {"val/auc": 0.9, "val/eer": 0.05}


def test_update_best_metrics_ignores_nonfinite_values():
    best = {"val/auc": 0.9, "val/eer": 0.1}

    updated = update_best_metrics(
        best,
        {"val/auc": float("nan"), "val/eer": float("inf")},
    )

    assert updated == set()
    assert best == {"val/auc": 0.9, "val/eer": 0.1}


def test_history_callback_tracks_validation_and_best_metric_steps(monkeypatch, tmp_path: Path):
    monkeypatch.setitem(sys.modules, "pytorch_lightning", SimpleNamespace(Callback=object))
    callback = build_metrics_history_callback(
        run_name="stage2_qbyt",
        run_id="stage2_qbyt/version_2",
        default_dir=tmp_path,
    )
    trainer = SimpleNamespace(
        sanity_checking=False,
        is_global_zero=True,
        callback_metrics={"val/auc": 0.9, "val/eer": 0.1},
        global_step=100,
        current_epoch=0,
    )

    callback.on_validation_end(trainer, None)
    trainer.callback_metrics = {"val/auc": 0.85, "val/eer": 0.05}
    trainer.global_step = 200
    trainer.current_epoch = 1
    callback.on_validation_end(trainer, None)

    assert callback.last_validation_step == 200
    assert callback.best == {"val/auc": 0.9, "val/eer": 0.05}
    assert callback.best_steps == {"val/auc": 100, "val/eer": 200}


def test_history_callback_uses_canonical_run_dir_not_first_logger(monkeypatch, tmp_path: Path):
    monkeypatch.setitem(sys.modules, "pytorch_lightning", SimpleNamespace(Callback=object))
    run_dir = tmp_path / "logs" / "run" / "version_4"
    callback = build_metrics_history_callback(
        run_name="run",
        default_dir=run_dir,
    )
    trainer = SimpleNamespace(
        loggers=[SimpleNamespace(log_dir=tmp_path / "wandb-backend")],
        global_step=0,
    )

    callback.on_fit_start(trainer, None)

    assert callback.csv_path == run_dir / "eval_history.csv"


def test_history_callback_state_roundtrip_preserves_resume_best_and_steps(
    monkeypatch, tmp_path: Path
):
    monkeypatch.setitem(sys.modules, "pytorch_lightning", SimpleNamespace(Callback=object))
    callback = build_metrics_history_callback(
        run_name="run",
        default_dir=tmp_path,
    )
    callback.best = {"val/auc": 0.94, "val/eer": 0.08}
    callback.best_steps = {"val/auc": 1200, "val/eer": 1500}
    callback.last_validation_step = 1500
    callback._duration_before_resume = 42.5

    restored = build_metrics_history_callback(
        run_name="run",
        default_dir=tmp_path,
    )
    restored.load_state_dict(callback.state_dict())

    assert restored.best == callback.best
    assert restored.best_steps == callback.best_steps
    assert restored.last_validation_step == 1500
    assert restored.duration_seconds == 42.5


def test_history_duration_freezes_when_fit_ends(monkeypatch, tmp_path: Path):
    monkeypatch.setitem(sys.modules, "pytorch_lightning", SimpleNamespace(Callback=object))
    times = iter([10.0, 15.5, 99.0])
    monkeypatch.setattr("dma_kws.training.metrics_history.time.monotonic", lambda: next(times))
    callback = build_metrics_history_callback(run_name="run", default_dir=tmp_path)
    trainer = SimpleNamespace(global_step=0)

    callback.on_fit_start(trainer, None)
    callback.on_fit_end(trainer, None)

    assert callback.duration_seconds == 5.5
    # Accessing it later must not include artifact export/post-processing time.
    assert callback.duration_seconds == 5.5


def test_numeric_callback_metrics_drops_alias_and_non_numeric():
    metrics = numeric_callback_metrics(
        {
            "val/auc": 0.9,
            "val_auc": 0.9,
            "val_target_auc": 0.91,
            "val_per": 0.2,
            "train/microbatch/loss_seq_weighted": 0.3,
            "note": "text",
            "val/eer": "0.1",
        }
    )
    assert metrics == {
        "train/microbatch/loss_seq_weighted": 0.3,
        "val/auc": 0.9,
        "val/eer": 0.1,
    }


def test_collect_hparams_stage2_section():
    hparams = collect_hparams(CONFIG)
    assert hparams["learning_rate"] == 5e-4
    assert hparams["max_steps"] == 50000
    assert hparams["batch_size_per_gpu"] == 64
    assert hparams["accumulate_grad_batches"] == 2
    assert hparams["val_check_interval_train_batches"] == 1000
    assert hparams["val_check_interval_optimizer_steps_approx"] == 500
    assert hparams["seed"] == 7
    assert hparams["seq_target_mode"] == "ordered_contiguous_prefix"
    assert hparams["seq_progress_weight"] == 0.3
    assert hparams["seq_normalization"] == "sample"
    assert hparams["qbyt_alignment_topology"] == "keyword_filler_segmental_crf_v1"
    assert hparams["qbyt_weakest_phone_temperature"] == 0.2
    assert hparams["qbyt_weakest_phone_weight"] == 1.0
    assert hparams["qbyt_max_phone_duration_frames"] == 8
    assert hparams["qbyt_max_inter_phone_gap_frames"] == 1
    assert hparams["qbyt_max_keyword_span_frames"] == 30
    assert hparams["qbyt_deployment_threshold"] == 0.5
    assert hparams["score_ece_num_bins"] == 15
    assert "rank" not in hparams


def test_collect_hparams_records_alignment_overrides():
    config = dict(CONFIG)
    config["stage2"] = dict(
        CONFIG["stage2"],
        qbyt_alignment={
            "topology": "keyword_filler_segmental_crf_v1",
            "min_phone_duration_frames": 2,
            "max_phone_duration_frames": 6,
            "max_inter_phone_gap_frames": 1,
            "max_keyword_span_frames": 24,
            "local_context_kernel": 3,
            "weakest_phone_temperature": 0.5,
            "weakest_phone_weight": 0.75,
        },
    )

    hparams = collect_hparams(config)

    assert hparams["qbyt_alignment_topology"] == "keyword_filler_segmental_crf_v1"
    assert hparams["qbyt_weakest_phone_temperature"] == 0.5
    assert hparams["qbyt_weakest_phone_weight"] == 0.75
    assert hparams["qbyt_max_phone_duration_frames"] == 6
    assert hparams["qbyt_max_inter_phone_gap_frames"] == 1
    assert hparams["qbyt_max_keyword_span_frames"] == 24


def test_collect_hparams_adapt_section():
    hparams = collect_hparams(CONFIG, section="adapt", extra={"slug": "hey-eva"})
    assert hparams["learning_rate"] == 4e-4
    assert hparams["optimizer"] == "adamw"
    assert hparams["max_steps"] == 3000
    assert hparams["batch_size_per_gpu"] == 32
    assert hparams["rank"] == 8
    assert hparams["alpha"] == 16.0
    assert hparams["mix_ratio"] == 0.6
    assert hparams["keyword"] == "hey eva"
    assert hparams["phase"] == "tts"
    assert hparams["slug"] == "hey-eva"


def test_collect_hparams_adapt_lr_alias_wins():
    config = {**CONFIG, "adapt": {**CONFIG["adapt"], "lr": 1e-3}}
    assert collect_hparams(config, section="adapt")["learning_rate"] == 1e-3


def test_collect_hparams_uses_effective_runtime_max_steps():
    hparams = collect_hparams(CONFIG, effective_max_steps=12)

    assert hparams["max_steps"] == 12


def test_collect_hparams_phoneme_adapter_uses_its_own_section():
    config = {
        **CONFIG,
        "phoneme_adapter": {
            "learning_rate": 2e-3,
            "optimizer": "adamw",
            "weight_decay": 0.02,
            "warmup_steps": 50,
            "max_steps": 600,
            "batch_size_per_gpu": 8,
            "accumulate_grad_batches": 3,
        },
    }

    hparams = collect_hparams(config, section="phoneme_adapter")

    assert hparams["learning_rate"] == 2e-3
    assert hparams["max_steps"] == 600
    assert hparams["batch_size_per_gpu"] == 8
    assert hparams["accumulate_grad_batches"] == 3
    assert "seq_target_mode" not in hparams


def test_collect_hparams_stage1_uses_max_train_steps():
    config = {
        **CONFIG,
        "stage1": {
            "learning_rate": 3e-3,
            "warmup_steps": 20,
            "max_train_steps": 100,
            "batch_size_per_gpu": 4,
        },
    }

    hparams = collect_hparams(config, section="stage1")

    assert hparams["learning_rate"] == 3e-3
    assert hparams["max_steps"] == 100
    assert hparams["max_epochs"] == 1
    assert hparams["batch_size_per_gpu"] == 4
    assert "seq_target_mode" not in hparams


def test_build_run_record_layout():
    record = build_run_record(
        run_name="adapt_hey-eva_tts",
        hparams={"learning_rate": 4e-4, "rank": 8},
        final_metrics={"val/auc": 0.91, "train/window/loss_total": 0.2},
        best_metrics={"val/auc": 0.93},
        global_step=3000,
        duration_seconds=125.67,
        metric_step=2800,
        best_steps={"val/auc": 2400},
        identity={"phase": "tts"},
    )
    assert record["run"] == "adapt_hey-eva_tts"
    assert record["phase"] == "tts"
    assert record["global_step"] == 3000
    assert record["metric_step"] == 2800
    assert record["duration_s"] == 125.7
    assert record["final/val/auc"] == 0.91
    assert record["best/val/auc"] == 0.93
    assert record["best_step/val/auc"] == 2400
    assert "timestamp" in record


def test_build_run_record_marks_missing_validation_step():
    record = build_run_record(
        run_name="stage2_qbyt",
        hparams={},
        final_metrics={},
        best_metrics={},
        global_step=3,
        duration_seconds=1.0,
    )

    assert record["metric_step"] == ""
    assert not any(name.startswith("best_step/") for name in record)


def test_run_records_accumulate_across_runs(tmp_path: Path):
    runs_csv = tmp_path / "runs.csv"
    for step, auc in ((1000, 0.9), (2000, 0.92)):
        append_wide_row(
            runs_csv,
            build_run_record(
                run_name="stage2_qbyt",
                hparams={"learning_rate": 5e-4},
                final_metrics={"val/auc": auc},
                best_metrics={"val/auc": auc},
                global_step=step,
                duration_seconds=10.0,
            ),
        )
    rows = _read_rows(runs_csv)
    assert len(rows) == 2
    assert [row["global_step"] for row in rows] == ["1000", "2000"]
