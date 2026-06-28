from unittest.mock import MagicMock, patch

from dma_kws.runlog import build_loggers
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
def test_build_loggers_default_backends(mock_tb, mock_csv):
    mock_csv.return_value = MagicMock(name="csv")
    mock_tb.return_value = MagicMock(name="tb")

    with patch.dict(
        "sys.modules",
        {
            "pytorch_lightning": MagicMock(),
            "pytorch_lightning.loggers": MagicMock(CSVLogger=mock_csv, TensorBoardLogger=mock_tb),
        },
    ):
        loggers = build_loggers("/tmp/logs", "run-a", config={"stage2": {"logging": {"backends": ["csv", "tensorboard"]}}})

    assert len(loggers) == 2
    mock_csv.assert_called_once()
    mock_tb.assert_called_once()


@patch("dma_kws.runlog.CSVLogger", create=True)
def test_build_loggers_legacy_without_config(mock_csv):
    mock_csv.return_value = MagicMock(name="csv")

    with patch.dict(
        "sys.modules",
        {
            "pytorch_lightning": MagicMock(),
            "pytorch_lightning.loggers": MagicMock(CSVLogger=mock_csv, TensorBoardLogger=MagicMock()),
        },
    ):
        loggers = build_loggers("/tmp/logs", "run-b")

    assert len(loggers) == 2
    mock_csv.assert_called_once()
