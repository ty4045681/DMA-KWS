from pathlib import Path

import pytest
import torch

from dma_kws.training.checkpoint_avg import average_lightning_checkpoints
from dma_kws.training.checkpoint_io import (
    QBYT_READOUT_VERSION,
    QBYT_READOUT_VERSION_KEY,
    STAGE2_BASE_FINGERPRINT_KEY,
    fingerprint_stage2_base,
)


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


def _write_lora_ckpt(
    path: Path,
    *,
    base: dict[str, torch.Tensor],
    adapter_value: float,
    alpha: float = 4.0,
    keyword: str = "hey eva",
    lora_targets: list[str] | tuple[str, ...] | str = ("in_proj_weight",),
) -> None:
    state = {
        key: value.clone()
        for key, value in base.items()
    }
    state.update(
        {
            "qbyt.layer.parametrizations.weight.0.lora_A": torch.full(
                (2, 3), adapter_value
            ),
            "qbyt.layer.parametrizations.weight.0.lora_B": torch.full(
                (3, 2), adapter_value
            ),
        }
    )
    torch.save(
        {
            "checkpoint_kind": "stage2_lora",
            "state_dict": state,
            "keyword": keyword,
            "phase": "tts",
            "rank": 2,
            "alpha": alpha,
            "lora_targets": lora_targets,
            "config": {
                "stage1": {
                    "encoder_type": "conformer",
                    "use_dynamic_chunk": False,
                },
                "adapt": {
                    "keyword": keyword,
                    "phase": "tts",
                    "rank": 2,
                    "alpha": alpha,
                    "lora_targets": lora_targets,
                }
            },
            QBYT_READOUT_VERSION_KEY: QBYT_READOUT_VERSION,
            STAGE2_BASE_FINGERPRINT_KEY: fingerprint_stage2_base(state),
        },
        path,
    )


def test_lora_average_only_averages_adapters_and_preserves_base_bits(tmp_path: Path):
    base = {
        "encoder.weight": torch.randn(4, 3),
        "qbyt.layer.parametrizations.weight.original": torch.randn(3, 3),
        "qbyt.bias": torch.randn(3),
    }
    first = tmp_path / "lora_a.ckpt"
    second = tmp_path / "lora_b.ckpt"
    output = tmp_path / "lora_avg.ckpt"
    _write_lora_ckpt(first, base=base, adapter_value=1.0)
    _write_lora_ckpt(second, base=base, adapter_value=3.0)

    average_lightning_checkpoints([first, second], output)

    averaged = torch.load(output, map_location="cpu")
    state = averaged["state_dict"]
    assert torch.equal(state["encoder.weight"], base["encoder.weight"])
    assert torch.equal(
        state["qbyt.layer.parametrizations.weight.original"],
        base["qbyt.layer.parametrizations.weight.original"],
    )
    assert torch.equal(
        state["qbyt.layer.parametrizations.weight.0.lora_A"],
        torch.full((2, 3), 2.0),
    )
    assert averaged[STAGE2_BASE_FINGERPRINT_KEY] == fingerprint_stage2_base(state)


def test_lora_average_rejects_different_bases_or_scaling(tmp_path: Path):
    base = {
        "encoder.weight": torch.randn(4, 3),
        "qbyt.layer.parametrizations.weight.original": torch.randn(3, 3),
    }
    first = tmp_path / "lora_a.ckpt"
    different_base = tmp_path / "lora_other_base.ckpt"
    different_alpha = tmp_path / "lora_other_alpha.ckpt"
    _write_lora_ckpt(first, base=base, adapter_value=1.0)
    changed = {key: value.clone() for key, value in base.items()}
    changed["encoder.weight"][0, 0] += 1
    _write_lora_ckpt(different_base, base=changed, adapter_value=2.0)
    _write_lora_ckpt(
        different_alpha,
        base=base,
        adapter_value=2.0,
        alpha=8.0,
    )

    with pytest.raises(ValueError, match="same frozen Stage II base"):
        average_lightning_checkpoints(
            [first, different_base],
            tmp_path / "bad_base.ckpt",
        )
    with pytest.raises(ValueError, match="different LoRA alpha"):
        average_lightning_checkpoints(
            [first, different_alpha],
            tmp_path / "bad_alpha.ckpt",
        )


def test_lora_average_normalizes_keyword_and_target_aliases(tmp_path: Path):
    base = {
        "encoder.weight": torch.randn(4, 3),
        "qbyt.layer.parametrizations.weight.original": torch.randn(3, 3),
    }
    first = tmp_path / "canonical.ckpt"
    second = tmp_path / "aliases.ckpt"
    output = tmp_path / "average.ckpt"
    _write_lora_ckpt(
        first,
        base=base,
        adapter_value=1.0,
        keyword="hey eva",
        lora_targets=["in_proj_weight", "out_proj.weight"],
    )
    _write_lora_ckpt(
        second,
        base=base,
        adapter_value=3.0,
        alpha=4.000000000001,
        keyword="Hey-Eva!!!",
        lora_targets=["out_proj", "in_proj_weight", "out_proj.weight"],
    )
    # Exercise normalization across both metadata locations within one file.
    aliased = torch.load(second, map_location="cpu")
    aliased["config"]["adapt"]["keyword"] = "HEY, EVA"
    aliased["config"]["adapt"]["lora_targets"] = [
        "in_proj_weight",
        "out_proj.weight",
    ]
    torch.save(aliased, second)

    average_lightning_checkpoints([first, second], output)

    averaged = torch.load(output, map_location="cpu")
    assert torch.equal(
        averaged["state_dict"]["qbyt.layer.parametrizations.weight.0.lora_A"],
        torch.full((2, 3), 2.0),
    )


def test_lora_average_rejects_unknown_targets(tmp_path: Path):
    base = {
        "encoder.weight": torch.randn(4, 3),
        "qbyt.layer.parametrizations.weight.original": torch.randn(3, 3),
    }
    first = tmp_path / "unknown_a.ckpt"
    second = tmp_path / "unknown_b.ckpt"
    _write_lora_ckpt(
        first,
        base=base,
        adapter_value=1.0,
        lora_targets=["in_proj_weigth"],
    )
    _write_lora_ckpt(
        second,
        base=base,
        adapter_value=3.0,
        lora_targets=["in_proj_weigth"],
    )

    with pytest.raises(ValueError, match="Unsupported LoRA targets"):
        average_lightning_checkpoints(
            [first, second],
            tmp_path / "invalid.ckpt",
        )
