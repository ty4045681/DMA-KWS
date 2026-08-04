from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import pickle
from types import SimpleNamespace
import time
from unittest.mock import MagicMock, patch

import pytest

from dma_kws.runlog import (
    build_loggers,
    _build_trackio_logger,
    _logger_warning,
    logger_backend_names,
    reserve_logger_version,
    resolve_logger_job_key,
    resolve_logger_version,
)
from dma_kws.training.loaders import build_loader_kwargs
from dma_kws.training.ddp import build_trainer_kwargs, resolve_precision
from dma_kws.training.scheduler import build_optimizer_config


def test_build_loader_kwargs_with_workers():
    kwargs = build_loader_kwargs(
        4,
        {"pin_memory": True, "persistent_workers": True, "prefetch_factor": 8},
    )

    assert kwargs == {
        "pin_memory": True,
        "persistent_workers": True,
        "prefetch_factor": 8,
    }


def test_build_loader_kwargs_zero_workers_omits_persistent_and_prefetch():
    kwargs = build_loader_kwargs(0, {"pin_memory": False, "persistent_workers": True, "prefetch_factor": 8})

    assert kwargs == {"pin_memory": False}


def test_build_optimizer_config_defaults():
    cfg = build_optimizer_config({"learning_rate": 0.0005, "warmup_steps": 100, "max_steps": 1000})

    assert cfg == {
        "optimizer": "adam",
        "lr": 0.0005,
        "weight_decay": 0.0,
        "warmup_steps": 100,
        "total_steps": 1000,
    }


def test_build_optimizer_config_adamw():
    cfg = build_optimizer_config(
        {
            "optimizer": "adamw",
            "learning_rate": 0.001,
            "weight_decay": 0.01,
            "total_scheduler_steps": 50000,
        }
    )

    assert cfg["optimizer"] == "adamw"
    assert cfg["weight_decay"] == 0.01
    assert cfg["total_steps"] == 50000


def test_resolve_precision_gpu_default():
    assert resolve_precision({}, "gpu") == "bf16-mixed"


def test_resolve_precision_cpu_default():
    assert resolve_precision({}, "cpu") == "32-true"


def test_resolve_precision_explicit_override():
    assert resolve_precision({"precision": "16-mixed"}, "gpu") == "16-mixed"


@patch("dma_kws.runlog.CSVLogger", create=True)
@patch("dma_kws.runlog.TensorBoardLogger", create=True)
def test_build_loggers_default_backends(mock_tb, mock_csv, tmp_path: Path):
    mock_csv.return_value = MagicMock(name="csv")
    mock_tb.return_value = MagicMock(name="tb")

    with patch.dict(
        "sys.modules",
        {
            "pytorch_lightning": MagicMock(),
            "pytorch_lightning.loggers": MagicMock(CSVLogger=mock_csv, TensorBoardLogger=mock_tb),
        },
    ):
        loggers = build_loggers(
            tmp_path,
            "run-a",
            config={"stage2": {"logging": {"backends": ["csv", "tensorboard"]}}},
        )

    assert len(loggers) == 2
    mock_csv.assert_called_once_with(
        save_dir=str(tmp_path), name="run-a", version=0
    )
    mock_tb.assert_called_once_with(
        save_dir=str(tmp_path),
        name="run-a",
        version=0,
        default_hp_metric=False,
    )


@patch("dma_kws.runlog.CSVLogger", create=True)
def test_build_loggers_legacy_without_config(mock_csv, tmp_path: Path):
    mock_csv.return_value = MagicMock(name="csv")

    with patch.dict(
        "sys.modules",
        {
            "pytorch_lightning": MagicMock(),
            "pytorch_lightning.loggers": MagicMock(CSVLogger=mock_csv, TensorBoardLogger=MagicMock()),
        },
    ):
        loggers = build_loggers(tmp_path, "run-b")

    assert len(loggers) == 2
    mock_csv.assert_called_once()


def test_resolve_logger_version_skips_non_version_directories(tmp_path):
    run_root = tmp_path / "run-a"
    (run_root / "version_0").mkdir(parents=True)
    (run_root / "version_3").mkdir()
    (run_root / "version_bad").mkdir()
    (run_root / "artifacts").mkdir()

    assert resolve_logger_version(tmp_path, "run-a") == 4


def test_concurrent_single_rank_jobs_reserve_distinct_versions(tmp_path: Path):
    jobs = 12

    def reserve(index: int) -> int:
        return reserve_logger_version(
            tmp_path,
            "run-a",
            job_key=f"single-rank-{index}",
            rank=0,
        )

    with ThreadPoolExecutor(max_workers=jobs) as executor:
        versions = list(executor.map(reserve, range(jobs)))

    assert sorted(versions) == list(range(jobs))
    assert len(set(versions)) == jobs
    for version in versions:
        assert (tmp_path / "run-a" / f"version_{version}").is_dir()


def test_rank_one_can_start_first_and_receives_rank_zero_version(tmp_path: Path):
    with ThreadPoolExecutor(max_workers=1) as executor:
        rank_one = executor.submit(
            reserve_logger_version,
            tmp_path,
            "run-a",
            job_key="shared-ddp-job",
            rank=1,
            timeout_seconds=2.0,
            poll_seconds=0.005,
        )
        claim_root = tmp_path / "run-a" / ".dma_kws_run_claims"
        deadline = time.monotonic() + 1.0
        while not claim_root.is_dir() and time.monotonic() < deadline:
            time.sleep(0.005)
        assert claim_root.is_dir(), "rank 1 did not enter the claim wait in time"

        rank_zero_version = reserve_logger_version(
            tmp_path,
            "run-a",
            job_key="shared-ddp-job",
            rank=0,
        )

        assert rank_one.result(timeout=1.0) == rank_zero_version


def test_nonzero_rank_times_out_without_guessing_a_version(tmp_path: Path):
    with pytest.raises(TimeoutError, match="Refusing to choose an independent"):
        reserve_logger_version(
            tmp_path,
            "run-a",
            job_key="rank-zero-never-started",
            rank=1,
            timeout_seconds=0.02,
            poll_seconds=0.005,
        )

    assert not list((tmp_path / "run-a").glob("version_*"))


def test_logger_job_key_precedence():
    launch_env = {
        "DMA_KWS_RUN_JOB_KEY": "from-env",
        "TORCHELASTIC_RUN_ID": "elastic-1",
        "TORCHELASTIC_RESTART_COUNT": "3",
        "SLURM_JOB_ID": "slurm-1",
        "SLURM_STEP_ID": "batch",
    }

    assert (
        resolve_logger_job_key("from-argument", environ=launch_env, rank=0)
        == "explicit:from-argument"
    )
    assert (
        resolve_logger_job_key(environ=launch_env, rank=0)
        == "explicit:from-env"
    )
    del launch_env["DMA_KWS_RUN_JOB_KEY"]
    assert (
        resolve_logger_job_key(environ=launch_env, rank=0)
        == "torchrun:elastic-1:restart=3"
    )
    del launch_env["TORCHELASTIC_RUN_ID"]
    assert (
        resolve_logger_job_key(environ=launch_env, rank=0)
        == "slurm:slurm-1:slurm_step_id=batch"
    )


@pytest.mark.parametrize("placeholder", ["", "   ", "none", "NONE", " None "])
def test_torchelastic_placeholder_falls_back_to_slurm(placeholder: str):
    launch_env = {
        "TORCHELASTIC_RUN_ID": placeholder,
        "SLURM_JOB_ID": "slurm-42",
        "SLURM_PROCID": "1",
    }

    assert (
        resolve_logger_job_key(environ=launch_env)
        == "slurm:slurm-42"
    )


def test_default_torchrun_id_does_not_merge_distinct_slurm_jobs(tmp_path: Path):
    def reserve(slurm_job_id: str) -> int:
        return reserve_logger_version(
            tmp_path,
            "run-a",
            environ={
                "TORCHELASTIC_RUN_ID": "none",
                "SLURM_JOB_ID": slurm_job_id,
                "SLURM_PROCID": "0",
            },
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        versions = list(executor.map(reserve, ("job-1", "job-2")))

    assert sorted(versions) == [0, 1]


def test_same_job_key_reuses_one_claim_even_with_two_rank_zero_callers(tmp_path: Path):
    def reserve(_: int) -> int:
        return reserve_logger_version(
            tmp_path,
            "run-a",
            job_key="one-logical-job",
            rank=0,
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        versions = list(executor.map(reserve, range(2)))

    assert versions == [0, 0]
    assert list((tmp_path / "run-a").glob("version_*")) == [
        tmp_path / "run-a" / "version_0"
    ]


def test_optional_backend_warning_is_silent_off_rank_zero(monkeypatch, capsys):
    monkeypatch.setenv("RANK", "1")

    _logger_warning("missing backend")

    assert capsys.readouterr().out == ""


def test_effective_logger_backend_names_are_stable():
    classes = [
        type("CSVLogger", (), {}),
        type("TensorBoardLogger", (), {}),
        type("WandbLogger", (), {}),
        type("TrackioLogger", (), {}),
    ]

    assert logger_backend_names([cls() for cls in classes]) == [
        "csv",
        "tensorboard",
        "wandb",
        "trackio",
    ]


def test_trackio_logger_is_spawn_pickle_safe(monkeypatch, tmp_path: Path):
    fake_trackio = SimpleNamespace(
        init=lambda **kwargs: object(),
        config=SimpleNamespace(update=lambda params: None),
        log=lambda payload: None,
        finish=lambda: None,
    )
    monkeypatch.setitem(__import__("sys").modules, "trackio", fake_trackio)
    logger = _build_trackio_logger(tmp_path, "run", 3, {"project": "test"})

    logger.log_hyperparams({"lr": 0.1})
    restored = pickle.loads(pickle.dumps(logger))

    assert restored.name == "run"
    assert restored.version == "3"
    assert restored._run is None


def test_build_loggers_reads_the_selected_training_section():
    mock_csv = MagicMock()
    mock_csv.return_value = MagicMock(name="csv")

    with patch.dict(
        "sys.modules",
        {
            "pytorch_lightning": MagicMock(),
            "pytorch_lightning.loggers": MagicMock(CSVLogger=mock_csv),
        },
    ):
        loggers = build_loggers(
            "/tmp/logs",
            "adapter",
            config={
                "stage2": {"logging": {"backends": ["tensorboard"]}},
                "phoneme_adapter": {"logging": {"backends": ["csv"]}},
            },
            section="phoneme_adapter",
            version="resume-a",
        )

    assert len(loggers) == 1
    mock_csv.assert_called_once_with(
        save_dir="/tmp/logs",
        name="adapter",
        version="resume-a",
    )


def test_build_loggers_rejects_unknown_backend():
    with patch.dict(
        "sys.modules",
        {
            "pytorch_lightning": MagicMock(),
            "pytorch_lightning.loggers": MagicMock(CSVLogger=MagicMock()),
        },
    ):
        with pytest.raises(ValueError, match="Unsupported stage2.logging backend"):
            build_loggers(
                "/tmp/logs",
                "bad",
                config={"stage2": {"logging": {"backends": ["csvv"]}}},
            )
