"""Adaptation hyperparameter resolution and run-summary rows (no torch required)."""

import yaml

from dma_kws.training.adapt_params import (
    DEFAULT_ADAPT_LR,
    load_adapt_params_file,
    merge_adapt_params,
    normalize_adapt_params,
    resolve_adapt_lr,
)
from dma_kws.training.callbacks import build_run_summary_rows

ADAPT_SECTION = {
    "lr": None,
    "learning_rate": 4e-4,
    "optimizer": "adam",
    "weight_decay": 0.0,
    "warmup_steps": 100,
    "max_steps": 3000,
    "batch_size_per_gpu": 32,
    "num_workers": 2,
    "rank": 16,
    "alpha": 32,
}

STAGE2_SECTION = {
    "learning_rate": 5e-4,
    "warmup_steps": 2500,
    "total_scheduler_steps": 50000,
    "max_steps": 50000,
    "batch_size_per_gpu": 64,
    "num_workers": 8,
    "accumulate_grad_batches": 1,
    "precision": "bf16-mixed",
}


def _rows_to_dict(rows):
    return {key: value for key, value in rows}


def test_learning_rate_defaults_to_canonical_key():
    assert resolve_adapt_lr(dict(ADAPT_SECTION)) == (4e-4, "learning_rate")
    assert resolve_adapt_lr({}) == (DEFAULT_ADAPT_LR, "default")


def test_lr_alias_wins_when_explicitly_set():
    adapt = dict(ADAPT_SECTION, lr=1e-3)
    assert resolve_adapt_lr(adapt) == (1e-3, "lr")


def test_swept_learning_rate_is_not_shadowed_by_lr_default():
    """Regression: adapt.lr used to shadow the swept learning_rate."""
    adapt = dict(ADAPT_SECTION)
    adapt.update(normalize_adapt_params({"learning_rate": 9e-4, "rank": 8, "alpha_ratio": 2.0}))
    assert resolve_adapt_lr(adapt) == (9e-4, "learning_rate")


def test_normalize_derives_alpha_from_alpha_ratio():
    normalized = normalize_adapt_params(
        {"rank": 8, "alpha_ratio": 2.0, "learning_rate": 1e-3, "max_steps": 1000}
    )
    assert normalized == {"rank": 8, "alpha": 16, "learning_rate": 1e-3, "max_steps": 1000}


def test_normalize_drops_bookkeeping_keys_and_keeps_explicit_alpha():
    normalized = normalize_adapt_params(
        {"score": 0.91, "_trial_number": 3, "alpha_ratio": 1.0, "rank": 4, "alpha": 99}
    )
    assert normalized == {"rank": 4, "alpha": 99}


def test_normalize_uses_config_rank_when_params_omit_it():
    normalized = normalize_adapt_params({"alpha_ratio": 2.0}, rank=16)
    assert normalized["alpha"] == 32
    assert "rank" not in normalized


def test_merge_params_file_reaches_training_config(tmp_path):
    """Sweep best params must survive params_file → adapt config → optimizer."""
    best_path = tmp_path / "best_params.yaml"
    # Optuna persists the searched space: alpha_ratio, not the absolute alpha.
    best_path.write_text(
        yaml.safe_dump({"score": 0.91, "rank": 8, "alpha_ratio": 2.0, "learning_rate": 9e-4, "max_steps": 1000}),
        encoding="utf-8",
    )

    adapt = dict(ADAPT_SECTION)
    merge_adapt_params(adapt, load_adapt_params_file(best_path))

    assert adapt["rank"] == 8
    assert adapt["alpha"] == 16
    assert adapt["max_steps"] == 1000
    assert resolve_adapt_lr(adapt) == (9e-4, "learning_rate")
    assert "score" not in adapt and "alpha_ratio" not in adapt


def test_merge_params_file_overrides_stale_lr_alias(tmp_path):
    """A params file learning_rate must not be shadowed by a configured adapt.lr."""
    params_file = tmp_path / "best_params.yaml"
    params_file.write_text(yaml.safe_dump({"learning_rate": 2e-3, "rank": 4}), encoding="utf-8")

    adapt = dict(ADAPT_SECTION, lr=4e-4)
    merge_adapt_params(adapt, load_adapt_params_file(params_file))

    assert resolve_adapt_lr(adapt) == (2e-3, "learning_rate")


def test_merge_params_keeps_explicit_lr_from_params_file(tmp_path):
    params_file = tmp_path / "best_params.yaml"
    params_file.write_text(yaml.safe_dump({"lr": 1e-3}), encoding="utf-8")

    adapt = dict(ADAPT_SECTION)
    merge_adapt_params(adapt, load_adapt_params_file(params_file))

    assert resolve_adapt_lr(adapt) == (1e-3, "lr")


def test_adapt_summary_rows_use_adapt_section_not_stage2():
    config = {
        "adapt": dict(ADAPT_SECTION, learning_rate=9e-4),
        "stage2": dict(STAGE2_SECTION),
        "training": {"recipe": "adapt"},
    }
    rows = _rows_to_dict(
        build_run_summary_rows(
            config=config,
            devices=1,
            accelerator="gpu",
            train_samples=3000,
            val_samples=120,
            section="adapt",
        )
    )

    assert rows["learning_rate"] == "0.0009"
    assert rows["max_steps"] == "3000"
    assert rows["warmup_steps"] == "100"
    assert rows["batch_size_per_gpu"] == "32"
    assert rows["num_workers"] == "2"
    # trainer-level settings still come from stage2
    assert rows["precision"] == "bf16-mixed"
    assert rows["accumulate_grad_batches"] == "1"


def test_adapt_summary_rows_flag_lr_alias_source():
    config = {
        "adapt": dict(ADAPT_SECTION, lr=2e-3),
        "stage2": dict(STAGE2_SECTION),
        "training": {},
    }
    rows = _rows_to_dict(
        build_run_summary_rows(
            config=config,
            devices=2,
            accelerator="gpu",
            train_samples=1,
            val_samples=1,
            section="adapt",
            extra_rows=[("phase", "tts")],
        )
    )

    assert rows["learning_rate (lr)"] == "0.002"
    assert rows["effective_batch"] == "64"
    assert rows["phase"] == "tts"


def test_stage2_summary_rows_unchanged():
    config = {"stage2": dict(STAGE2_SECTION), "training": {"recipe": "paper"}}
    rows = _rows_to_dict(
        build_run_summary_rows(
            config=config,
            devices=1,
            accelerator="cpu",
            train_samples=10,
            val_samples=5,
        )
    )

    assert rows["learning_rate"] == "0.0005"
    assert rows["warmup_steps"] == "2500"
    assert rows["max_steps"] == "50000"
    assert rows["batch_size_per_gpu"] == "64"
    assert rows["qbyt_deployment_threshold"] == "0.5"
    assert rows["score_ece_num_bins"] == "15"
    assert rows["seq_diagnostic_threshold"] == "0.5"


def test_summary_can_report_effective_logging_backends():
    rows = _rows_to_dict(
        build_run_summary_rows(
            config={
                "stage2": {
                    **STAGE2_SECTION,
                    "logging": {"backends": ["csv", "tensorboard", "wandb"]},
                },
                "training": {},
            },
            devices=1,
            accelerator="cpu",
            train_samples=1,
            val_samples=1,
            effective_logging_backends=["csv"],
        )
    )

    assert rows["logging_backends"] == "csv"


def test_stage1_summary_uses_stage1_runtime_contract():
    config = {
        "stage1": {
            "learning_rate": 0.003,
            "warmup_steps": 0,
            "max_train_steps": 0,
            "batch_size_per_gpu": 8,
            "accumulate_grad_batches": 3,
            "precision": "32-true",
            "num_workers": 2,
            "logging": {"backends": ["csv"]},
            "validation": {"check_val_every_n_epoch": 1},
        },
        # Deliberately conflicting values: Stage I must not report these.
        "stage2": {
            "accumulate_grad_batches": 9,
            "precision": "bf16-mixed",
        },
        "training": {},
    }
    rows = _rows_to_dict(
        build_run_summary_rows(
            config=config,
            devices=2,
            accelerator="gpu",
            train_samples=10,
            val_samples=5,
            section="stage1",
            effective_max_steps=-1,
        )
    )

    assert rows["effective_batch"] == "48"
    assert rows["accumulate_grad_batches"] == "3"
    assert rows["precision"] == "32-true"
    assert rows["optimizer"] == "adam"
    assert rows["max_epochs"] == "1"
    assert rows["stop_condition"] == "max_epochs=1"
    assert rows["checkpoint_monitor"] == "val/per"
    assert "ema" not in rows
    assert "scheduler_total_steps" not in rows


def test_phoneme_adapter_summary_uses_its_own_section():
    config = {
        "phoneme_adapter": {
            "learning_rate": 0.002,
            "optimizer": "adamw",
            "weight_decay": 0.01,
            "warmup_steps": 50,
            "max_steps": 600,
            "batch_size_per_gpu": 4,
            "accumulate_grad_batches": 2,
            "precision": "16-mixed",
            "num_workers": 1,
            "logging": {"backends": ["csv", "tensorboard"]},
        },
        "stage2": {"accumulate_grad_batches": 7},
        "training": {},
    }
    rows = _rows_to_dict(
        build_run_summary_rows(
            config=config,
            devices=2,
            accelerator="gpu",
            train_samples=10,
            val_samples=5,
            section="phoneme_adapter",
        )
    )

    assert rows["effective_batch"] == "16"
    assert rows["precision"] == "16-mixed"
    assert rows["optimizer"] == "adamw"
    assert rows["checkpoint_monitor"] == "val/per"
    assert rows["logging_backends"] == "csv, tensorboard"
    assert "ema" not in rows
