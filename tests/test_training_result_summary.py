"""Final training-result summary contract."""

from __future__ import annotations

from io import StringIO

import pytest

from dma_kws.training.callbacks import (
    build_training_result_rows,
    print_training_result_summary,
)
from dma_kws.training.run_context import RunContext


def _context(tmp_path) -> RunContext:
    run_dir = tmp_path / "logs" / "paper" / "version_3"
    return RunContext(
        section="stage2",
        run_name="paper",
        version=3,
        log_root=str(tmp_path / "logs"),
        run_dir=str(run_dir),
        effective_max_steps=100,
        log_interval=10,
    )


def test_result_rows_expose_identity_progress_freshness_and_outputs(tmp_path):
    rows = dict(
        build_training_result_rows(
            run_context=_context(tmp_path),
            global_step=97,
            last_validation_step=90,
            best_checkpoint_monitor="val/auc",
            best_checkpoint_path=tmp_path / "best.ckpt",
            best_checkpoint_score=0.987654321,
            final_metrics={
                "train/microbatch/loss_total": 0.2,
                "val/auc": 0.94,
                "val/eer_threshold": 0.97,
                "val/score_neg_mean": 0.4,
                "val/score_neg_p95": 0.99,
                "val/eer": 0.09,
            },
            artifact_paths={
                "final_checkpoint": tmp_path / "final.pt",
                "optional_export": None,
            },
            artifact_sources={"final_checkpoint": "final_weights@step=97"},
        )
    )

    assert rows["run_id"] == "paper/version_3"
    assert rows["run_dir"].endswith("logs/paper/version_3")
    assert rows["global_step"] == "97"
    assert rows["effective_max_steps"] == "100"
    assert rows["last_validation_step"] == "90"
    assert rows["validation_staleness_steps"] == "7"
    assert rows["metrics_source"] == "last_trainer_state"
    assert rows["best_checkpoint_monitor"] == "val/auc"
    assert rows["best_checkpoint_path"].endswith("best.ckpt")
    assert rows["best_checkpoint_score"] == "0.987654"
    assert rows["metric/val/auc"] == "0.94"
    assert rows["metric/val/eer_threshold"] == "0.97"
    assert rows["metric/val/score_neg_p95"] == "0.99"
    assert rows["metric/val/eer"] == "0.09"
    assert "metric/val/score_neg_mean" not in rows
    assert "metric/train/microbatch/loss_total" not in rows
    assert rows["artifact/final_checkpoint"].endswith("final.pt")
    assert rows["artifact_source/final_checkpoint"] == "final_weights@step=97"
    assert rows["artifact/optional_export"] == "(not produced)"


def test_result_rows_make_absent_validation_and_checkpoint_explicit(tmp_path):
    rows = dict(
        build_training_result_rows(
            run_context=_context(tmp_path),
            global_step=0,
            last_validation_step=None,
        )
    )

    assert rows["last_validation_step"] == "(not validated)"
    assert rows["validation_staleness_steps"] == "(not available)"
    assert rows["best_checkpoint_path"] == "(not available)"
    assert rows["best_checkpoint_score"] == "(not available)"


def test_plain_result_summary_is_deterministic_on_rank_zero(tmp_path, monkeypatch):
    monkeypatch.setattr("dma_kws.training.ddp.process_rank", lambda: 0)
    output = StringIO()

    print_training_result_summary(
        run_context=_context(tmp_path),
        global_step=97,
        last_validation_step=90,
        best_checkpoint_path="best.ckpt",
        best_checkpoint_score=0.9,
        artifact_paths={"final": "final.pt"},
        title="Stage II Result",
        rich=False,
        stream=output,
    )

    rendered = output.getvalue()
    assert rendered.startswith("=== Stage II Result ===\n")
    assert "  run_id: paper/version_3\n" in rendered
    assert "  validation_staleness_steps: 7\n" in rendered
    assert "  artifact/final: final.pt\n" in rendered


def test_result_summary_is_silent_off_rank_zero(tmp_path, monkeypatch):
    monkeypatch.setattr("dma_kws.training.ddp.process_rank", lambda: 1)
    output = StringIO()

    print_training_result_summary(
        run_context=_context(tmp_path),
        global_step=10,
        last_validation_step=10,
        stream=output,
    )

    assert output.getvalue() == ""


def test_rich_result_summary_renders_for_tty(tmp_path, monkeypatch):
    pytest.importorskip("rich")
    monkeypatch.setattr("dma_kws.training.ddp.process_rank", lambda: 0)

    class TTYBuffer(StringIO):
        def isatty(self) -> bool:
            return True

    output = TTYBuffer()
    print_training_result_summary(
        run_context=_context(tmp_path),
        global_step=100,
        last_validation_step=100,
        title="Training Result",
        rich=True,
        stream=output,
    )

    rendered = output.getvalue()
    assert "Training Result" in rendered
    assert "paper/version_3" in rendered
    assert "validation_staleness_steps" in rendered
