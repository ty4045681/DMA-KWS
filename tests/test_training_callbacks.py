"""Stage II console and diagnostics callback configuration."""

from __future__ import annotations

import pytest

pytest.importorskip("pytorch_lightning")

from dma_kws.training.callbacks import build_stage2_callbacks
from dma_kws.training.progress import _curate


def _config(tmp_path, *, console):
    return {
        "paths": {"exp_root": str(tmp_path)},
        "stage2": {
            "checkpoint_dir": str(tmp_path / "checkpoints"),
            "checkpoint": {"monitor": ""},
            "console": console,
        }
    }


def test_rich_false_explicitly_selects_plain_console_callbacks(tmp_path):
    callbacks = build_stage2_callbacks(
        _config(tmp_path, console={"rich": False}),
        "paper",
    )
    callback_names = [type(callback).__name__ for callback in callbacks]

    assert callback_names.count("CuratedTQDMProgressBar") == 1
    assert callback_names.count("ModelSummary") == 1
    assert "CuratedRichProgressBar" not in callback_names
    assert "RichModelSummary" not in callback_names

    model_summary = next(
        callback for callback in callbacks if type(callback).__name__ == "ModelSummary"
    )
    assert model_summary._max_depth == 2

    progress = next(
        callback
        for callback in callbacks
        if type(callback).__name__ == "CuratedTQDMProgressBar"
    )
    assert progress.refresh_rate == 10


def test_throughput_monitor_receives_stage2_batch_size_function(tmp_path):
    callbacks = build_stage2_callbacks(
        _config(tmp_path, console={"rich": False, "throughput": True}),
        "paper",
    )
    throughput = next(
        callback for callback in callbacks if type(callback).__name__ == "ThroughputMonitor"
    )

    assert throughput.batch_size_fn({"label": [1, 0, 1]}) == 3
    assert throughput.batch_size_fn({"feat": [[0.0], [1.0]]}) == 2
    assert throughput.batch_size_fn({"feats": [[0.0], [1.0], [2.0]]}) == 3


def test_throughput_batch_size_error_names_expected_fields(tmp_path):
    callbacks = build_stage2_callbacks(
        _config(tmp_path, console={"rich": False, "throughput": True}),
        "paper",
    )
    throughput = next(
        callback for callback in callbacks if type(callback).__name__ == "ThroughputMonitor"
    )

    with pytest.raises(ValueError, match="label.*feat.*feats.*anchor.*targets"):
        throughput.batch_size_fn({"metadata": ["sample"]})


def test_progress_metrics_are_curated_and_renamed():
    metrics = _curate(
        {
            "v_num": 3,
            "train/microbatch/loss_total": 0.4,
            "train/microbatch/loss_seq_progress_raw": 0.9,
            "train/microbatch/ctc_skip_rate": 0.25,
            "train/optimizer/lr": 1e-4,
            "val/auc": 0.95,
        }
    )

    assert metrics == {
        "v_num": 3,
        "loss": 0.4,
        "skip": 0.25,
        "lr": 1e-4,
        "v_auc": 0.95,
    }


def test_plain_progress_honors_leave_setting(tmp_path):
    callbacks = build_stage2_callbacks(
        _config(tmp_path, console={"rich": False, "leave": True}),
        "paper",
    )
    progress = next(
        callback
        for callback in callbacks
        if type(callback).__name__ == "CuratedTQDMProgressBar"
    )

    assert progress._leave is True


def test_rich_constructor_dependency_failure_falls_back_to_tqdm(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "dma_kws.training.progress.CuratedRichProgressBar",
        lambda **kwargs: (_ for _ in ()).throw(ModuleNotFoundError("rich")),
    )

    callbacks = build_stage2_callbacks(
        _config(tmp_path, console={"rich": True}),
        "paper",
    )

    names = [type(callback).__name__ for callback in callbacks]
    assert "CuratedTQDMProgressBar" in names
    assert "RichModelSummary" not in names
