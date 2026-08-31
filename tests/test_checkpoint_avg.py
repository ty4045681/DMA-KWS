from pathlib import Path

import pytest
import torch

from dma_kws.stage2.readout import QbyTAlignmentSpec
from dma_kws.training.checkpoint_avg import average_lightning_checkpoints
from dma_kws.training.checkpoint_io import (
    QBYT_ALIGNMENT_SPEC_KEY,
    QBYT_READOUT_VERSION,
    QBYT_READOUT_VERSION_KEY,
    STAGE2_BASE_FINGERPRINT_KEY,
    fingerprint_stage2_base,
)


def _alignment(**overrides) -> dict:
    spec = QbyTAlignmentSpec().as_dict()
    spec.update(overrides)
    return QbyTAlignmentSpec(**spec).as_dict()


def _objective(*, progress_weight: float = 0.3) -> dict:
    return {
        "target_mode": "ordered_contiguous_prefix",
        "progress_weight": progress_weight,
        "normalization": "sample",
    }


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


def test_average_rejects_different_stage2_sequence_objectives(tmp_path: Path) -> None:
    first = tmp_path / "first.ckpt"
    current = tmp_path / "current.ckpt"
    state = {"qbyt.weight": torch.tensor([1.0])}
    torch.save(
        {
            "state_dict": state,
            "config": {
                "stage2": {
                    "qbyt_alignment": _alignment(),
                    "sequence_loss": _objective(progress_weight=0.3),
                }
            },
            QBYT_READOUT_VERSION_KEY: QBYT_READOUT_VERSION,
            QBYT_ALIGNMENT_SPEC_KEY: _alignment(),
        },
        first,
    )
    torch.save(
        {
            "state_dict": state,
            "config": {
                "stage2": {
                    "qbyt_alignment": _alignment(),
                    "sequence_loss": _objective(progress_weight=0.5),
                }
            },
            QBYT_READOUT_VERSION_KEY: QBYT_READOUT_VERSION,
            QBYT_ALIGNMENT_SPEC_KEY: _alignment(),
        },
        current,
    )

    with pytest.raises(ValueError, match="different Stage II sequence objectives"):
        average_lightning_checkpoints([first, current], tmp_path / "bad.ckpt")


def _write_qbyt_alignment_ckpt(
    path: Path,
    *,
    alignment: dict | None = None,
    version: int = QBYT_READOUT_VERSION,
    weight: float = 1.0,
) -> None:
    alignment = _alignment() if alignment is None else alignment
    torch.save(
        {
            "state_dict": {"qbyt.weight": torch.tensor([weight])},
            "config": {
                "stage2": {
                    "qbyt_alignment": alignment,
                    "sequence_loss": _objective(),
                }
            },
            QBYT_READOUT_VERSION_KEY: version,
            QBYT_ALIGNMENT_SPEC_KEY: alignment,
        },
        path,
    )


@pytest.mark.parametrize(
    "changed",
    [
        {"min_phone_duration_frames": 2},
        {"max_phone_duration_frames": 9},
        {"max_inter_phone_gap_frames": 1},
        {"max_keyword_span_frames": 40},
        {"temperature": 0.35},
        {"local_context_kernel": 7},
    ],
)
def test_average_rejects_any_qbyt_alignment_difference(
    tmp_path: Path,
    changed: dict,
) -> None:
    baseline = tmp_path / "baseline.ckpt"
    different = tmp_path / "different.ckpt"
    _write_qbyt_alignment_ckpt(baseline)
    _write_qbyt_alignment_ckpt(different, alignment=_alignment(**changed))

    with pytest.raises(ValueError, match="different QbyT alignments"):
        average_lightning_checkpoints(
            [baseline, different],
            tmp_path / "bad_alignment.ckpt",
        )


@pytest.mark.parametrize("version", [1, 2, 3, 4])
def test_average_rejects_pre_v5_qbyt_checkpoint(
    tmp_path: Path,
    version: int,
) -> None:
    legacy = tmp_path / f"v{version}.ckpt"
    current = tmp_path / "v5.ckpt"
    _write_qbyt_alignment_ckpt(legacy, version=version)
    _write_qbyt_alignment_ckpt(current)

    with pytest.raises(ValueError, match=f"unsupported QbyT readout version {version}"):
        average_lightning_checkpoints([legacy, current], tmp_path / "bad.ckpt")


def test_average_accepts_only_same_v5_alignment(tmp_path: Path) -> None:
    first = tmp_path / "first.ckpt"
    second = tmp_path / "second.ckpt"
    output = tmp_path / "mean.ckpt"
    alignment = _alignment(max_keyword_span_frames=40, temperature=0.35)
    _write_qbyt_alignment_ckpt(first, alignment=alignment, weight=1.0)
    _write_qbyt_alignment_ckpt(second, alignment=alignment, weight=3.0)

    average_lightning_checkpoints([first, second], output)

    averaged = torch.load(output, map_location="cpu")
    assert averaged[QBYT_READOUT_VERSION_KEY] == QBYT_READOUT_VERSION
    assert averaged[QBYT_ALIGNMENT_SPEC_KEY] == alignment
    torch.testing.assert_close(
        averaged["state_dict"]["qbyt.weight"],
        torch.tensor([2.0]),
    )


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
    lora_targets: list[str] | tuple[str, ...] | str = ("audio_key.weight",),
    alignment: dict | None = None,
) -> None:
    alignment = _alignment() if alignment is None else alignment
    state = {
        key: value.clone()
        for key, value in base.items()
    }
    for projection in ("audio_projection", "audio_key", "text_query"):
        root = f"qbyt.{projection}.parametrizations.weight"
        original = state.get(f"{root}.original")
        if original is None:
            continue
        state[f"{root}.0.lora_A"] = torch.full(
            (2, int(original.shape[1])), adapter_value
        )
        state[f"{root}.0.lora_B"] = torch.full(
            (int(original.shape[0]), 2), adapter_value
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
                "stage2": {
                    "qbyt_alignment": alignment,
                    "sequence_loss": _objective(),
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
            QBYT_ALIGNMENT_SPEC_KEY: alignment,
            STAGE2_BASE_FINGERPRINT_KEY: fingerprint_stage2_base(state),
        },
        path,
    )


def test_lora_average_only_averages_adapters_and_preserves_base_bits(tmp_path: Path):
    base = {
        "encoder.weight": torch.randn(4, 3),
        "qbyt.audio_key.parametrizations.weight.original": torch.randn(3, 3),
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
        state["qbyt.audio_key.parametrizations.weight.original"],
        base["qbyt.audio_key.parametrizations.weight.original"],
    )
    assert torch.equal(
        state["qbyt.audio_key.parametrizations.weight.0.lora_A"],
        torch.full((2, 3), 2.0),
    )
    assert averaged[STAGE2_BASE_FINGERPRINT_KEY] == fingerprint_stage2_base(state)


def test_lora_average_rejects_different_bases_or_scaling(tmp_path: Path):
    base = {
        "encoder.weight": torch.randn(4, 3),
        "qbyt.audio_key.parametrizations.weight.original": torch.randn(3, 3),
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


def test_lora_average_rejects_different_alignment_specs(tmp_path: Path):
    base = {
        "encoder.weight": torch.randn(4, 3),
        "qbyt.audio_key.parametrizations.weight.original": torch.randn(3, 3),
    }
    first = tmp_path / "softmin_a.ckpt"
    second = tmp_path / "softmin_b.ckpt"
    _write_lora_ckpt(
        first,
        base=base,
        adapter_value=1.0,
        alignment=_alignment(temperature=0.3),
    )
    _write_lora_ckpt(
        second,
        base=base,
        adapter_value=3.0,
        alignment=_alignment(temperature=0.4),
    )

    with pytest.raises(ValueError, match="different QbyT alignments"):
        average_lightning_checkpoints(
            [first, second],
            tmp_path / "bad_readout.ckpt",
        )


def test_lora_average_normalizes_keyword_and_target_order(tmp_path: Path):
    base = {
        "encoder.weight": torch.randn(4, 3),
        "qbyt.audio_key.parametrizations.weight.original": torch.randn(3, 3),
        "qbyt.text_query.parametrizations.weight.original": torch.randn(3, 3),
    }
    first = tmp_path / "canonical.ckpt"
    second = tmp_path / "aliases.ckpt"
    output = tmp_path / "average.ckpt"
    _write_lora_ckpt(
        first,
        base=base,
        adapter_value=1.0,
        keyword="hey eva",
        lora_targets=["audio_key.weight", "text_query.weight"],
    )
    _write_lora_ckpt(
        second,
        base=base,
        adapter_value=3.0,
        alpha=4.000000000001,
        keyword="Hey-Eva!!!",
        lora_targets=["text_query.weight", "audio_key.weight"],
    )
    # Exercise normalization across both metadata locations within one file.
    aliased = torch.load(second, map_location="cpu")
    aliased["config"]["adapt"]["keyword"] = "HEY, EVA"
    aliased["config"]["adapt"]["lora_targets"] = [
        "audio_key.weight",
        "text_query.weight",
    ]
    torch.save(aliased, second)

    average_lightning_checkpoints([first, second], output)

    averaged = torch.load(output, map_location="cpu")
    assert torch.equal(
        averaged["state_dict"]["qbyt.audio_key.parametrizations.weight.0.lora_A"],
        torch.full((2, 3), 2.0),
    )


def test_lora_average_rejects_unknown_targets(tmp_path: Path):
    base = {
        "encoder.weight": torch.randn(4, 3),
        "qbyt.audio_key.parametrizations.weight.original": torch.randn(3, 3),
    }
    first = tmp_path / "unknown_a.ckpt"
    second = tmp_path / "unknown_b.ckpt"
    _write_lora_ckpt(
        first,
        base=base,
        adapter_value=1.0,
        lora_targets=["audio_key.weigth"],
    )
    _write_lora_ckpt(
        second,
        base=base,
        adapter_value=3.0,
        lora_targets=["audio_key.weigth"],
    )

    with pytest.raises(ValueError, match="Unsupported LoRA targets"):
        average_lightning_checkpoints(
            [first, second],
            tmp_path / "invalid.ckpt",
        )
