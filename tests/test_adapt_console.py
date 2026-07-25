"""Console table builders for Stage II LoRA adaptation."""

import pytest

pytest.importorskip("rich")

from dma_kws.stage2 import adapt_console
from dma_kws.stage2.prep_console import Stage2PrepReporter


PREP_STATS = {
    "keyword": "hey eva",
    "slug": "hey_eva",
    "unique_texts": 3,
    "fbank_written": 12,
    "fbank_skipped": 4,
    "phases": {
        "tts": {
            "total": 8,
            "train": 6,
            "eval": 2,
            "train_positive": 3,
            "train_negative": 3,
            "eval_positive": 1,
            "eval_negative": 1,
            "train_manifest": "/data/manifests/tts_train.csv",
            "eval_manifest": "/data/manifests/tts_eval.csv",
        },
        "real": {
            "total": 8,
            "train": 6,
            "eval": 2,
            "train_positive": 3,
            "train_negative": 3,
            "eval_positive": 1,
            "eval_negative": 1,
            "train_manifest": "/data/manifests/real_train.csv",
            "eval_manifest": "/data/manifests/real_eval.csv",
        },
    },
}


def test_prep_phase_table_lists_each_phase_once():
    columns, rows = adapt_console.prep_phase_table(PREP_STATS)
    assert columns[0] == "phase"
    assert [row[0] for row in rows] == ["real", "tts"]
    assert rows[1][1].startswith("6 (3/3)")


def test_prep_stats_rows_totals_samples():
    rows = dict(adapt_console.prep_stats_rows(PREP_STATS))
    assert rows["samples"] == "16"
    assert rows["unique_texts"] == "3"
    assert rows["fbank_written"] == "12"
    assert rows["fbank_skipped"] == "4"


def test_lora_rows_report_scaling_and_share():
    rows = dict(
        adapt_console.lora_rows(
            rank=8,
            alpha=16,
            targets=("in_proj_weight", "out_proj.weight"),
            injected=["a", "b", "c"],
            param_counts={"lora_trainable": 100, "trainable": 100, "total": 10000},
        )
    )
    assert rows["scaling (alpha/rank)"] == "2.000"
    assert rows["injected_matrices"] == "3"
    assert "1.000% of model" in rows["trainable_params"]


def test_eval_comparison_table_computes_deltas():
    report = {
        "target_base": {"auc": 0.80, "eer": 0.20},
        "target_adapted": {"auc": 0.95, "eer": 0.05},
        "lph_base": {"auc": 0.90, "eer": 0.10},
        "lph_adapted": {"auc": 0.88, "eer": 0.12},
    }
    columns, rows = adapt_console.eval_comparison_table(report)
    assert columns == ["metric", "base", "adapted", "delta"]
    table = {row[0]: row for row in rows}
    assert table["target/auc"][3] == "+0.1500"
    assert table["lph/auc"][3] == "-0.0200"


def test_eval_comparison_table_skips_missing_groups():
    _columns, rows = adapt_console.eval_comparison_table(
        {"lph_base": {"auc": 0.9}, "lph_adapted": {"auc": 0.9}}
    )
    assert [row[0] for row in rows] == ["lph/auc"]


def test_sweep_results_table_orders_by_score():
    columns, rows = adapt_console.sweep_results_table(
        [
            {"number": 0, "score": 0.5, "target_auc": 0.7, "lph_auc": 0.8, "params": {"rank": 4}},
            {"number": 1, "score": 0.9, "target_auc": 0.95, "lph_auc": 0.89, "params": {"rank": 16}},
        ]
    )
    assert columns[0] == "trial"
    assert [row[0] for row in rows] == ["1", "0"]


def test_reporter_falls_back_to_plain_text(capsys):
    reporter = Stage2PrepReporter(use_rich=False)
    reporter.section("Plan")
    reporter.print_table(["a", "b"], [["1", "2"]], title="Demo")
    reporter.done("finished")
    output = capsys.readouterr().out
    assert "=== Plan ===" in output
    assert "=== Demo ===" in output
    assert "a | b" in output
    assert "1 | 2" in output
    assert "finished" in output
