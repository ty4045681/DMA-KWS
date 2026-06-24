from pathlib import Path

import pytest
import torch

from dma_kws.training.checkpoint_avg import average_lightning_checkpoints


def _write_lightning_ckpt(path: Path, weight: float) -> None:
    torch.save(
        {
            "state_dict": {"layer.weight": torch.tensor([weight, weight + 1.0])},
            "epoch": 1,
            "global_step": 100,
        },
        path,
    )


def test_average_lightning_checkpoints_computes_mean(tmp_path: Path) -> None:
    ckpt_a = tmp_path / "step_step=000001.ckpt"
    ckpt_b = tmp_path / "step_step=000002.ckpt"
    _write_lightning_ckpt(ckpt_a, 1.0)
    _write_lightning_ckpt(ckpt_b, 3.0)

    output_path = tmp_path / "avg_2.ckpt"
    result = average_lightning_checkpoints([ckpt_a, ckpt_b], output_path)

    assert result == output_path
    averaged = torch.load(output_path, map_location="cpu")
    expected = torch.tensor([2.0, 3.0])
    assert torch.allclose(averaged["state_dict"]["layer.weight"], expected)
    assert averaged["epoch"] == 1
    assert averaged["global_step"] == 100


def test_average_lightning_checkpoints_supports_legacy_model_state_dict(tmp_path: Path) -> None:
    ckpt_a = tmp_path / "a.pt"
    ckpt_b = tmp_path / "b.pt"
    torch.save({"model_state_dict": {"bias": torch.tensor([0.0])}}, ckpt_a)
    torch.save({"model_state_dict": {"bias": torch.tensor([2.0])}}, ckpt_b)

    output_path = tmp_path / "avg.pt"
    average_lightning_checkpoints([ckpt_a, ckpt_b], output_path)

    averaged = torch.load(output_path, map_location="cpu")
    assert "model_state_dict" in averaged
    assert torch.allclose(averaged["model_state_dict"]["bias"], torch.tensor([1.0]))


def test_average_lightning_checkpoints_requires_paths(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="At least one checkpoint"):
        average_lightning_checkpoints([], tmp_path / "avg.ckpt")
