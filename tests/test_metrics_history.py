"""Dense metrics-history CSV helpers and run records (no torch required)."""

import csv
from pathlib import Path

from dma_kws.training.metrics_history import (
    append_wide_row,
    build_run_record,
    collect_hparams,
    metric_direction,
    numeric_callback_metrics,
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


def test_update_best_metrics_honors_direction():
    best: dict[str, float] = {}
    update_best_metrics(best, {"val/auc": 0.9, "val/eer": 0.1})
    update_best_metrics(best, {"val/auc": 0.85, "val/eer": 0.05})
    assert best == {"val/auc": 0.9, "val/eer": 0.05}


def test_numeric_callback_metrics_drops_alias_and_non_numeric():
    metrics = numeric_callback_metrics(
        {"val/auc": 0.9, "val_auc": 0.9, "note": "text", "val/eer": "0.1"}
    )
    assert metrics == {"val/auc": 0.9, "val/eer": 0.1}


def test_collect_hparams_stage2_section():
    hparams = collect_hparams(CONFIG)
    assert hparams["learning_rate"] == 5e-4
    assert hparams["max_steps"] == 50000
    assert hparams["batch_size_per_gpu"] == 64
    assert hparams["accumulate_grad_batches"] == 2
    assert hparams["seed"] == 7
    assert "rank" not in hparams


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


def test_build_run_record_layout():
    record = build_run_record(
        run_name="adapt_hey-eva_tts",
        hparams={"learning_rate": 4e-4, "rank": 8},
        final_metrics={"val/auc": 0.91, "train/loss": 0.2},
        best_metrics={"val/auc": 0.93},
        global_step=3000,
        duration_seconds=125.67,
        identity={"phase": "tts"},
    )
    assert record["run"] == "adapt_hey-eva_tts"
    assert record["phase"] == "tts"
    assert record["global_step"] == 3000
    assert record["duration_s"] == 125.7
    assert record["final/val/auc"] == 0.91
    assert record["best/val/auc"] == 0.93
    assert "timestamp" in record


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
