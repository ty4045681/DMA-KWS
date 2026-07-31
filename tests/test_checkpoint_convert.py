from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from dma_kws.training.checkpoint_convert import (
    CheckpointConversionError,
    convert_checkpoint,
    merge_lora_checkpoint_state,
)
from dma_kws.training.checkpoint_io import (
    QBYT_READOUT_VERSION,
    QBYT_READOUT_VERSION_KEY,
    assert_qbyt_readout_version,
    extract_state_dict,
)
from dma_kws.training.lora import inject_qbyt_lora, merge_lora
from dma_kws.pathing import load_qbyt_class
from scripts.convert_stage2_checkpoints import (
    build_parser,
    discover_checkpoints,
    main,
    run,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DICT_PATH = REPO_ROOT / "data" / "dict" / "lang_char.txt"
VOCAB_SIZE = 71


@pytest.fixture(autouse=True)
def _lightweight_encoder_validation(monkeypatch):
    """Exercise strict encoder loading without optional Wenet/icefall deps."""
    monkeypatch.setattr(
        "dma_kws.nn.build_encoder",
        lambda *_args, **_kwargs: nn.Linear(3, 2),
    )


def _config(
    *,
    rank: int = 2,
    alpha: float | None = 4.0,
    targets: list[str] | None = None,
) -> dict:
    adapt = {
        "keyword": "hey eva",
        "slug": "",
        "phase": "tts",
        "rank": rank,
        "lora_targets": targets or ["in_proj_weight", "out_proj.weight"],
    }
    if alpha is not None:
        adapt["alpha"] = alpha
    return {
        "tokenizer": {
            "dict_path": str(DICT_PATH),
            "split_with_space": " ",
        },
        "stage1": {
            "encoder_type": "conformer",
            "encoder_output_dim": 3,
        },
        "stage2": {"qbyt_embed_dim": 4, "qbyt_layers": 1},
        "adapt": adapt,
    }


def _stage2_state() -> dict[str, torch.Tensor]:
    QbyT = load_qbyt_class()
    qbyt = QbyT(
        encoder_output_size=3,
        num_embeds=VOCAB_SIZE,
        embed_dim=4,
        post_num_layers=1,
    )
    state = {
        "encoder.weight": torch.arange(6, dtype=torch.float32).reshape(2, 3),
        "encoder.bias": torch.arange(2, dtype=torch.float32),
    }
    state.update(
        {
            f"qbyt.{key}": value
            for key, value in qbyt.state_dict().items()
        }
    )
    return state


def _checkpoint(
    state: dict[str, torch.Tensor],
    *,
    config: dict | None = None,
    global_step: int = 123,
    vocab_size: int = VOCAB_SIZE,
    readout_version: int | None = QBYT_READOUT_VERSION,
) -> dict:
    checkpoint = {
        "state_dict": state,
        "global_step": global_step,
        "step": 999,
        "hyper_parameters": {"vocab_size": vocab_size},
        "optimizer_states": [{"should_not": "survive"}],
        "callbacks": {"should_not": "survive"},
    }
    if config is not None:
        checkpoint["config"] = config
    if readout_version is not None:
        checkpoint[QBYT_READOUT_VERSION_KEY] = readout_version
    return checkpoint


def _add_lora_group(
    state: dict[str, torch.Tensor],
    target: str,
    *,
    original: torch.Tensor,
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
) -> None:
    state.pop(target, None)
    module, parameter = target.rsplit(".", 1)
    root = f"{module}.parametrizations.{parameter}"
    state[f"{root}.original"] = original
    state[f"{root}.0.lora_A"] = lora_a
    state[f"{root}.0.lora_B"] = lora_b


def _lora_state() -> tuple[
    dict[str, torch.Tensor],
    dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
]:
    state = _stage2_state()
    groups = {
        "qbyt.phone_matchor.layers.0.self_attn.in_proj_weight": (
            torch.arange(48, dtype=torch.float32).reshape(12, 4),
            torch.tensor(
                [[1.0, 2.0, 3.0, 4.0], [0.5, 1.0, 1.5, 2.0]]
            ),
            torch.arange(24, dtype=torch.float32).reshape(12, 2) / 10,
        ),
        "qbyt.phone_matchor.layers.0.self_attn.out_proj.weight": (
            torch.arange(16, dtype=torch.float32).reshape(4, 4),
            torch.tensor(
                [[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]]
            ),
            torch.tensor(
                [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]
            ),
        ),
    }
    for target, (original, lora_a, lora_b) in groups.items():
        _add_lora_group(
            state,
            target,
            original=original,
            lora_a=lora_a,
            lora_b=lora_b,
        )
    return state, groups


def test_convert_stage2_checkpoint_writes_only_inference_payload(tmp_path: Path) -> None:
    source = tmp_path / "step.ckpt"
    output = tmp_path / "step.pt"
    config = _config()
    original_state = _stage2_state()
    torch.save(_checkpoint(original_state, config=config), source)

    result = convert_checkpoint(source, output)

    assert result.kind == "stage2"
    assert result.outputs == (output,)
    payload = torch.load(output, map_location="cpu")
    assert set(payload) == {
        "model_state_dict",
        "config",
        "step",
        "tokenizer_dict_path",
        "vocab_size",
        QBYT_READOUT_VERSION_KEY,
    }
    assert payload["step"] == 123
    assert payload["vocab_size"] == VOCAB_SIZE
    assert payload["tokenizer_dict_path"] == str(DICT_PATH)
    assert payload["config"] == config
    for key, value in original_state.items():
        assert torch.equal(payload["model_state_dict"][key], value)
    assert_qbyt_readout_version(payload, source=output)
    assert extract_state_dict(payload) is payload["model_state_dict"]


def test_historical_checkpoint_uses_explicit_fallback_config(tmp_path: Path) -> None:
    source = tmp_path / "old.ckpt"
    output = tmp_path / "old.pt"
    config = _config()
    torch.save(_checkpoint(_stage2_state()), source)

    convert_checkpoint(source, output, fallback_config=config)

    assert torch.load(output, map_location="cpu")["config"] == config


def test_stage2_conversion_rejects_explicit_lora_options(tmp_path: Path) -> None:
    source = tmp_path / "stage2.ckpt"
    torch.save(_checkpoint(_stage2_state(), config=_config()), source)

    with pytest.raises(CheckpointConversionError, match="adapter output"):
        convert_checkpoint(
            source,
            tmp_path / "stage2.pt",
            adapter_output_path=tmp_path / "unused.adapter.pt",
        )
    with pytest.raises(CheckpointConversionError, match="LoRA alpha"):
        convert_checkpoint(
            source,
            tmp_path / "stage2.pt",
            lora_alpha=4.0,
        )


def test_conversion_refuses_missing_config_or_readout_version(tmp_path: Path) -> None:
    source = tmp_path / "old.ckpt"
    torch.save(_checkpoint(_stage2_state()), source)

    with pytest.raises(CheckpointConversionError, match="resolved training config"):
        convert_checkpoint(source, tmp_path / "old.pt")

    torch.save(
        _checkpoint(_stage2_state(), config=_config(), readout_version=None),
        source,
    )
    with pytest.raises(CheckpointConversionError, match="Refusing to stamp"):
        convert_checkpoint(source, tmp_path / "old.pt")


def test_conversion_rejects_fractional_integer_metadata(tmp_path: Path) -> None:
    source = tmp_path / "bad_step.ckpt"
    checkpoint = _checkpoint(_stage2_state(), config=_config())
    checkpoint["global_step"] = 3.5
    torch.save(checkpoint, source)

    with pytest.raises(CheckpointConversionError, match="must be an integer"):
        convert_checkpoint(source, tmp_path / "bad_step.pt")


def test_conversion_rejects_checkpoint_kind_mismatch(tmp_path: Path) -> None:
    source = tmp_path / "wrong_kind.ckpt"
    checkpoint = _checkpoint(_stage2_state(), config=_config())
    checkpoint["checkpoint_kind"] = "stage2_lora"
    torch.save(checkpoint, source)

    with pytest.raises(CheckpointConversionError, match="checkpoint_kind"):
        convert_checkpoint(source, tmp_path / "wrong_kind.pt")


def test_embedded_config_wins_over_redundant_fallback(tmp_path: Path) -> None:
    source = tmp_path / "step.ckpt"
    output = tmp_path / "config_mismatch.pt"
    embedded = _config()
    supplied = _config()
    supplied["stage2"]["qbyt_layers"] = 2
    torch.save(_checkpoint(_stage2_state(), config=embedded), source)

    with pytest.warns(UserWarning, match="is ignored"):
        convert_checkpoint(
            source,
            output,
            fallback_config=supplied,
        )

    assert torch.load(output, map_location="cpu")["config"] == embedded


def test_conversion_refuses_vocab_mismatch(tmp_path: Path) -> None:
    source = tmp_path / "step.ckpt"
    embedded = _config()
    torch.save(
        _checkpoint(_stage2_state(), config=embedded, vocab_size=VOCAB_SIZE + 1),
        source,
    )
    with pytest.raises(CheckpointConversionError, match="Conflicting vocab_size"):
        convert_checkpoint(source, tmp_path / "vocab_mismatch.pt")


def test_conversion_refuses_truncated_or_architecture_mismatched_state(
    tmp_path: Path,
) -> None:
    source = tmp_path / "broken.ckpt"
    truncated = _stage2_state()
    truncated.pop("qbyt.fc.bias")
    torch.save(_checkpoint(truncated, config=_config()), source)
    with pytest.raises(CheckpointConversionError, match="QbyT weights do not match"):
        convert_checkpoint(source, tmp_path / "truncated.pt")

    truncated_encoder = _stage2_state()
    truncated_encoder.pop("encoder.bias")
    torch.save(_checkpoint(truncated_encoder, config=_config()), source)
    with pytest.raises(CheckpointConversionError, match="Encoder weights do not match"):
        convert_checkpoint(source, tmp_path / "truncated_encoder.pt")

    config = _config()
    config["stage2"]["phoneme_adapter"] = {"enabled": True}
    torch.save(_checkpoint(_stage2_state(), config=config), source)
    with pytest.raises(CheckpointConversionError, match=r"adapter\.\* weights are absent"):
        convert_checkpoint(source, tmp_path / "adapter_mismatch.pt")


def test_conversion_validates_and_preserves_enabled_phoneme_adapter(
    tmp_path: Path,
) -> None:
    from dma_kws.phoneme_adapter.module import build_phoneme_adapter

    config = _config()
    adapter_cfg = {
        "enabled": True,
        "trunk": {
            "type": "conv",
            "output_dim": 5,
            "num_layers": 1,
            "kernel_size": 3,
            "dropout": 0.0,
        },
    }
    config["stage2"]["phoneme_adapter"] = adapter_cfg
    adapter = build_phoneme_adapter(
        adapter_cfg,
        input_dim=3,
        vocab_size=VOCAB_SIZE,
    )
    QbyT = load_qbyt_class()
    qbyt = QbyT(
        encoder_output_size=adapter.output_dim,
        num_embeds=VOCAB_SIZE,
        embed_dim=4,
        post_num_layers=1,
    )
    state = {
        "encoder.weight": torch.ones(2, 3),
        "encoder.bias": torch.ones(2),
    }
    state.update({f"adapter.{key}": value for key, value in adapter.state_dict().items()})
    state.update({f"qbyt.{key}": value for key, value in qbyt.state_dict().items()})
    source = tmp_path / "adapter.ckpt"
    output = tmp_path / "adapter.pt"
    torch.save(_checkpoint(state, config=config), source)

    convert_checkpoint(source, output)

    converted = torch.load(output, map_location="cpu")["model_state_dict"]
    assert any(key.startswith("adapter.") for key in converted)


def test_convert_lora_checkpoint_writes_merged_and_adapter_payloads(
    tmp_path: Path,
) -> None:
    source = tmp_path / "lora.ckpt"
    merged_output = tmp_path / "lora.pt"
    adapter_output = tmp_path / "lora.adapter.pt"
    state, groups = _lora_state()
    config = _config(alpha=4.0)
    torch.save(_checkpoint(state, config=config), source)

    result = convert_checkpoint(source, merged_output)

    assert result.kind == "lora"
    assert result.outputs == (merged_output, adapter_output)
    merged = torch.load(merged_output, map_location="cpu")
    adapter = torch.load(adapter_output, map_location="cpu")

    assert merged["keyword"] == "hey eva"
    assert merged["slug"] == "hey_eva"
    assert merged["phase"] == "tts"
    assert merged["step"] == 123
    assert not any(
        ".parametrizations." in key or key.endswith((".lora_A", ".lora_B"))
        for key in merged["model_state_dict"]
    )
    for target, (original, lora_a, lora_b) in groups.items():
        expected = original + 2.0 * (lora_b @ lora_a)
        assert torch.allclose(merged["model_state_dict"][target], expected)

    assert adapter["rank"] == 2
    assert adapter["alpha"] == 4.0
    assert adapter["lora_targets"] == ["in_proj_weight", "out_proj.weight"]
    assert adapter["keyword"] == "hey eva"
    assert adapter["slug"] == "hey_eva"
    assert adapter["phase"] == "tts"
    assert adapter["lora_state_dict"]
    assert all(
        not key.startswith("qbyt.") for key in adapter["lora_state_dict"]
    )
    assert all(
        key.endswith((".lora_A", ".lora_B"))
        for key in adapter["lora_state_dict"]
    )
    assert_qbyt_readout_version(adapter, source=adapter_output)


@pytest.mark.parametrize(
    ("mode", "payload_key"),
    [("merged", "model_state_dict"), ("adapter", "lora_state_dict")],
)
def test_convert_lora_checkpoint_supports_single_artifact_modes(
    tmp_path: Path,
    mode: str,
    payload_key: str,
) -> None:
    source = tmp_path / f"{mode}.ckpt"
    output = tmp_path / f"{mode}.pt"
    state, _ = _lora_state()
    torch.save(_checkpoint(state, config=_config()), source)

    result = convert_checkpoint(source, output, lora_output=mode)

    assert result.outputs == (output,)
    assert payload_key in torch.load(output, map_location="cpu")
    assert not output.with_name(f"{output.stem}.adapter.pt").exists()


def test_adapter_only_skips_irrelevant_full_model_validation(tmp_path: Path) -> None:
    source = tmp_path / "adapter_only.ckpt"
    adapter_output = tmp_path / "adapter_only.pt"
    merged_output = tmp_path / "merged.pt"
    state, _ = _lora_state()
    adapter_relevant_state = {
        key: value
        for key, value in state.items()
        if key == "encoder.weight" or ".parametrizations." in key
    }
    config = _config()
    config["tokenizer"]["dict_path"] = "/nonexistent/lang_char.txt"
    torch.save(_checkpoint(adapter_relevant_state, config=config), source)

    result = convert_checkpoint(
        source,
        adapter_output,
        lora_output="adapter",
    )

    assert result.outputs == (adapter_output,)
    payload = torch.load(adapter_output, map_location="cpu")
    assert "lora_state_dict" in payload
    assert "model_state_dict" not in payload
    assert payload["config"] == config

    with pytest.raises(CheckpointConversionError, match="not a complete Stage II"):
        convert_checkpoint(source, merged_output, lora_output="merged")


def test_conversion_can_select_current_model_state(tmp_path: Path) -> None:
    source = tmp_path / "ema.ckpt"
    averaged = _stage2_state()
    current = {
        key: value + 10
        for key, value in averaged.items()
    }
    checkpoint = _checkpoint(averaged, config=_config())
    checkpoint["current_model_state"] = current
    checkpoint["averaging_state"] = {"n_averaged": torch.tensor(0)}
    torch.save(checkpoint, source)

    with pytest.raises(CheckpointConversionError, match="n_averaged=0"):
        convert_checkpoint(source, tmp_path / "invalid_ema.pt")

    output = tmp_path / "current.pt"
    convert_checkpoint(source, output, weights_key="current_model_state")

    converted = torch.load(output, map_location="cpu")["model_state_dict"]
    for key, value in current.items():
        assert torch.equal(converted[key], value)


def test_raw_lora_merge_matches_formula_and_rejects_incomplete_group() -> None:
    state, groups = _lora_state()
    merged = merge_lora_checkpoint_state(state, alpha=4.0)
    for target, (original, lora_a, lora_b) in groups.items():
        assert torch.allclose(merged[target], original + 2.0 * (lora_b @ lora_a))

    broken = dict(state)
    broken.pop(
        "qbyt.phone_matchor.layers.0.self_attn.parametrizations."
        "in_proj_weight.0.lora_B"
    )
    with pytest.raises(CheckpointConversionError, match="Incomplete LoRA"):
        merge_lora_checkpoint_state(broken, alpha=4.0)

    unsupported = dict(state)
    _add_lora_group(
        unsupported,
        "encoder.block.in_proj_weight",
        original=torch.ones(4, 4),
        lora_a=torch.ones(2, 4),
        lora_b=torch.ones(4, 2),
    )
    with pytest.raises(CheckpointConversionError, match="outside QbyT"):
        merge_lora_checkpoint_state(unsupported, alpha=4.0)


def test_raw_lora_merge_matches_project_parametrization() -> None:
    layer = nn.TransformerEncoderLayer(
        d_model=4,
        nhead=2,
        dim_feedforward=8,
        batch_first=True,
    )
    qbyt = nn.Module()
    qbyt.phone_matchor = nn.TransformerEncoder(layer, num_layers=1)
    inject_qbyt_lora(qbyt, rank=2, alpha=4.0)
    with torch.no_grad():
        for name, parameter in qbyt.named_parameters():
            if name.endswith(".lora_A"):
                parameter.fill_(0.25)
            elif name.endswith(".lora_B"):
                parameter.fill_(0.5)

    wrapper = nn.Module()
    wrapper.qbyt = qbyt
    expected = copy.deepcopy(wrapper)
    merge_lora(expected.qbyt)

    converted = merge_lora_checkpoint_state(wrapper.state_dict(), alpha=4.0)
    expected_state = expected.state_dict()
    assert converted.keys() == expected_state.keys()
    for key, value in expected_state.items():
        assert torch.allclose(converted[key], value)


def test_lora_conversion_requires_exact_alpha_and_rank(tmp_path: Path) -> None:
    source = tmp_path / "lora.ckpt"
    state, _ = _lora_state()

    torch.save(_checkpoint(state, config=_config(alpha=None)), source)
    with pytest.raises(CheckpointConversionError, match="cannot be inferred"):
        convert_checkpoint(source, tmp_path / "missing_alpha.pt")

    torch.save(_checkpoint(state, config=_config(rank=4)), source)
    with pytest.raises(CheckpointConversionError, match="Conflicting LoRA rank"):
        convert_checkpoint(source, tmp_path / "rank_mismatch.pt")


def test_both_outputs_are_preflighted_before_writing(tmp_path: Path) -> None:
    source = tmp_path / "lora.ckpt"
    output = tmp_path / "lora.pt"
    adapter_output = tmp_path / "lora.adapter.pt"
    state, _ = _lora_state()
    torch.save(_checkpoint(state, config=_config()), source)
    adapter_output.write_bytes(b"existing")

    with pytest.raises(FileExistsError, match="already exists"):
        convert_checkpoint(source, output)

    assert not output.exists()
    assert adapter_output.read_bytes() == b"existing"


def test_discover_checkpoints_supports_sorted_recursive_batch(tmp_path: Path) -> None:
    nested = tmp_path / "nested"
    nested.mkdir()
    second = nested / "b.ckpt"
    first = tmp_path / "a.ckpt"
    ignored = nested / "ignored.pt"
    for path in (second, first, ignored):
        path.write_bytes(b"x")

    found = discover_checkpoints(
        checkpoints=None,
        input_dir=tmp_path,
        pattern="*.ckpt",
        recursive=True,
    )

    assert found == [first, second]


def test_cli_batch_conversion_preserves_relative_layout(tmp_path: Path) -> None:
    input_dir = tmp_path / "checkpoints"
    nested = input_dir / "nested"
    nested.mkdir(parents=True)
    first = input_dir / "a.ckpt"
    second = nested / "b.ckpt"
    for step, path in enumerate((first, second), start=1):
        torch.save(
            _checkpoint(_stage2_state(), config=_config(), global_step=step),
            path,
        )
    output_dir = tmp_path / "pt"
    args = build_parser().parse_args(
        [
            "--input-dir",
            str(input_dir),
            "--recursive",
            "--output-dir",
            str(output_dir),
        ]
    )

    results = run(args)

    assert len(results) == 2
    assert (output_dir / "a.pt").is_file()
    assert (output_dir / "nested" / "b.pt").is_file()
    assert torch.load(output_dir / "a.pt", map_location="cpu")["step"] == 1
    assert torch.load(output_dir / "nested" / "b.pt", map_location="cpu")["step"] == 2


def test_cli_batch_rejects_cross_checkpoint_adapter_collision(
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "checkpoints"
    input_dir.mkdir()
    state, _ = _lora_state()
    for name in ("a.ckpt", "a.adapter.ckpt"):
        torch.save(
            _checkpoint(state, config=_config()),
            input_dir / name,
        )
    output_dir = tmp_path / "pt"
    args = build_parser().parse_args(
        [
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(output_dir),
            "--overwrite",
        ]
    )

    with pytest.raises(CheckpointConversionError, match="collides"):
        run(args)

    assert not output_dir.exists()


def test_cli_wraps_invalid_hydra_override(tmp_path: Path) -> None:
    source = tmp_path / "stage2.ckpt"
    torch.save(_checkpoint(_stage2_state(), config=_config()), source)
    args = build_parser().parse_args(
        [
            "--checkpoint",
            str(source),
            "--default-config",
            "--override",
            "definitely_missing=1",
        ]
    )

    with pytest.raises(CheckpointConversionError, match="Failed to resolve"):
        run(args)


def test_cli_reports_corrupt_checkpoint_without_traceback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = tmp_path / "corrupt.ckpt"
    source.write_bytes(b"not a torch checkpoint")
    monkeypatch.setattr(
        sys,
        "argv",
        ["convert_stage2_checkpoints.py", "--checkpoint", str(source)],
    )

    with pytest.raises(SystemExit) as caught:
        main()

    assert caught.value.code == 2
    captured = capsys.readouterr()
    assert "error:" in captured.err
    assert "Traceback" not in captured.err
