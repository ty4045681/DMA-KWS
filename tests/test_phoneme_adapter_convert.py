from __future__ import annotations

from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from dma_kws.config import resolve_stream_policy
from dma_kws.phoneme_adapter.convert import (
    PhonemeAdapterConversionError,
    convert_phoneme_adapter_checkpoint,
)
from dma_kws.phoneme_adapter.module import build_phoneme_adapter
from dma_kws.training.checkpoint_io import assert_stream_policy_matches
from dma_kws.training.run_context import RUN_CONTEXT_KEY
from scripts.convert_phoneme_adapter_checkpoints import build_parser, run

REPO_ROOT = Path(__file__).resolve().parents[1]
DICT_PATH = REPO_ROOT / "data" / "dict" / "lang_char.txt"
VOCAB_SIZE = 11
BLANK_ID = 0


def _config(**adapter_overrides) -> dict:
    adapter_cfg: dict = {"trunk": {"type": "linear", "output_dim": 5}}
    adapter_cfg.update(adapter_overrides)
    return {
        "tokenizer": {"dict_path": str(DICT_PATH), "split_with_space": " "},
        "stage1": {"encoder_type": "conformer", "encoder_output_dim": 3},
        "phoneme_adapter": adapter_cfg,
    }


def _adapter(config: dict | None = None, *, vocab_size: int = VOCAB_SIZE):
    config = config or _config()
    return build_phoneme_adapter(
        config["phoneme_adapter"],
        input_dim=int(config["stage1"]["encoder_output_dim"]),
        vocab_size=vocab_size,
        causal=bool(config["stage1"].get("causal", False)),
        blank_id=BLANK_ID,
    )


def _lightning_checkpoint(
    adapter,
    *,
    global_step: int | None = 123,
    with_hparams: bool = True,
    with_run_context: bool = True,
) -> dict:
    state = {f"adapter.{key}": value for key, value in adapter.state_dict().items()}
    # The frozen encoder shares the Lightning state dict but must not leak into
    # the adapter-only export.
    state["encoder.weight"] = torch.zeros(2, 3)
    checkpoint: dict = {
        "state_dict": state,
        "optimizer_states": [{"should_not": "survive"}],
        "callbacks": {"should_not": "survive"},
    }
    if global_step is not None:
        checkpoint["global_step"] = global_step
    if with_hparams:
        checkpoint["hyper_parameters"] = {
            "vocab_size": VOCAB_SIZE,
            "blank_id": BLANK_ID,
            "init_checkpoint": None,
            "adapter_init_checkpoint": None,
        }
    if with_run_context:
        checkpoint[RUN_CONTEXT_KEY] = {
            "run_id": "20260806-abc",
            "run_section": "phoneme_adapter",
        }
    return checkpoint


def _save_checkpoint(tmp_path: Path, name: str = "adapter_0000123_0.1234.ckpt") -> tuple[Path, object]:
    adapter = _adapter()
    source = tmp_path / name
    torch.save(_lightning_checkpoint(adapter), source)
    return source, adapter


def test_convert_produces_loadable_adapter_export(tmp_path: Path) -> None:
    source, adapter = _save_checkpoint(tmp_path)
    output = tmp_path / "adapter.pt"

    result = convert_phoneme_adapter_checkpoint(source, output, config=_config())

    assert result == output
    payload = torch.load(output, map_location="cpu")
    assert payload["checkpoint_kind"] == "phoneme_adapter"
    assert payload["step"] == 123
    assert payload["vocab_size"] == VOCAB_SIZE
    assert payload["blank_id"] == BLANK_ID
    assert payload["tokenizer_dict_path"] == str(DICT_PATH)
    assert payload["config"]["phoneme_adapter"]["trunk"]["output_dim"] == 5
    assert payload[RUN_CONTEXT_KEY]["run_id"] == "20260806-abc"
    assert "optimizer_states" not in payload

    state = payload["model_state_dict"]
    assert state, "export must carry the adapter weights"
    assert not any(key.startswith(("adapter.", "encoder.")) for key in state)
    for key, value in adapter.state_dict().items():
        assert torch.equal(state[key], value), key

    # The two real consumers: a fresh adapter strict-loads the export, and the
    # stream-policy guard accepts the embedded config.
    fresh = _adapter()
    fresh.load_state_dict(state, strict=True)
    assert_stream_policy_matches(
        payload, resolve_stream_policy(_config()), source=output
    )


def test_convert_refuses_to_overwrite_without_flag(tmp_path: Path) -> None:
    source, _ = _save_checkpoint(tmp_path)
    output = tmp_path / "adapter.pt"
    output.write_bytes(b"existing")

    with pytest.raises(PhonemeAdapterConversionError, match="--overwrite"):
        convert_phoneme_adapter_checkpoint(source, output, config=_config())

    convert_phoneme_adapter_checkpoint(
        source, output, config=_config(), overwrite=True
    )
    assert torch.load(output, map_location="cpu")["step"] == 123


def test_convert_rejects_checkpoint_without_adapter_weights(tmp_path: Path) -> None:
    source = tmp_path / "stage1.ckpt"
    torch.save(
        {"state_dict": {"encoder.weight": torch.zeros(2, 3)}, "global_step": 5},
        source,
    )

    with pytest.raises(PhonemeAdapterConversionError, match=r"no 'adapter\.' weights"):
        convert_phoneme_adapter_checkpoint(source, tmp_path / "out.pt", config=_config())


def test_convert_rejects_already_exported_pt(tmp_path: Path) -> None:
    source = tmp_path / "adapter_best_step000123.pt"
    torch.save({"model_state_dict": {"trunk.proj.weight": torch.zeros(5, 3)}}, source)

    with pytest.raises(PhonemeAdapterConversionError, match="not a Lightning checkpoint"):
        convert_phoneme_adapter_checkpoint(source, tmp_path / "out.pt", config=_config())


def test_convert_rejects_output_suffix(tmp_path: Path) -> None:
    source, _ = _save_checkpoint(tmp_path)

    with pytest.raises(PhonemeAdapterConversionError, match="end in .pt"):
        convert_phoneme_adapter_checkpoint(source, tmp_path / "out.bin", config=_config())


def test_convert_step_falls_back_to_filename(tmp_path: Path) -> None:
    adapter = _adapter()
    checkpoint = _lightning_checkpoint(adapter, global_step=None)
    source = tmp_path / "adapter_0000456_0.4321.ckpt"
    torch.save(checkpoint, source)

    output = convert_phoneme_adapter_checkpoint(
        source, tmp_path / "out.pt", config=_config()
    )

    assert torch.load(output, map_location="cpu")["step"] == 456


def test_convert_step_requires_global_step_filename_or_override(tmp_path: Path) -> None:
    adapter = _adapter()
    checkpoint = _lightning_checkpoint(adapter, global_step=None)
    source = tmp_path / "last.ckpt"
    torch.save(checkpoint, source)

    with pytest.raises(PhonemeAdapterConversionError, match="--step"):
        convert_phoneme_adapter_checkpoint(source, tmp_path / "out.pt", config=_config())

    output = convert_phoneme_adapter_checkpoint(
        source, tmp_path / "out.pt", config=_config(), step=789
    )
    assert torch.load(output, map_location="cpu")["step"] == 789


def test_convert_rejects_config_mismatch(tmp_path: Path) -> None:
    source, _ = _save_checkpoint(tmp_path)
    mismatched = _config()
    mismatched["phoneme_adapter"]["trunk"]["output_dim"] = 6

    with pytest.raises(PhonemeAdapterConversionError, match="does not match"):
        convert_phoneme_adapter_checkpoint(
            source, tmp_path / "out.pt", config=mismatched
        )


def test_convert_recovers_vocab_and_blank_from_tokenizer(tmp_path: Path) -> None:
    adapter = _adapter(vocab_size=71)
    checkpoint = _lightning_checkpoint(adapter, with_hparams=False)
    # hparams are absent, so the recorded vocab must not be the small test value.
    checkpoint["hyper_parameters"] = {"init_checkpoint": None}
    source = tmp_path / "adapter_0000123_0.1234.ckpt"
    torch.save(checkpoint, source)

    output = convert_phoneme_adapter_checkpoint(
        source, tmp_path / "out.pt", config=_config()
    )

    payload = torch.load(output, map_location="cpu")
    assert payload["vocab_size"] == 71
    assert payload["blank_id"] == 0


def test_cli_converts_input_dir_preserving_layout(tmp_path: Path) -> None:
    input_dir = tmp_path / "checkpoints"
    nested = input_dir / "nested"
    nested.mkdir(parents=True)
    for name, directory in (("a.ckpt", input_dir), ("b.ckpt", nested)):
        torch.save(_lightning_checkpoint(_adapter()), directory / name)
    config_path = tmp_path / "config.yaml"
    OmegaConf.save(OmegaConf.create(_config()), config_path)
    output_dir = tmp_path / "pt"

    args = build_parser().parse_args(
        [
            "--input-dir",
            str(input_dir),
            "--recursive",
            "--output-dir",
            str(output_dir),
            "--config",
            str(config_path),
        ]
    )
    results = run(args)

    assert len(results) == 2
    assert (output_dir / "a.pt").is_file()
    assert (output_dir / "nested" / "b.pt").is_file()


def test_cli_rejects_output_and_step_with_multiple_checkpoints(tmp_path: Path) -> None:
    for name in ("a.ckpt", "b.ckpt"):
        torch.save(_lightning_checkpoint(_adapter()), tmp_path / name)
    config_path = tmp_path / "config.yaml"
    OmegaConf.save(OmegaConf.create(_config()), config_path)

    args = build_parser().parse_args(
        [
            "--checkpoint",
            str(tmp_path / "a.ckpt"),
            "--checkpoint",
            str(tmp_path / "b.ckpt"),
            "--output",
            str(tmp_path / "out.pt"),
            "--config",
            str(config_path),
        ]
    )
    with pytest.raises(PhonemeAdapterConversionError, match="--output"):
        run(args)

    args = build_parser().parse_args(
        [
            "--checkpoint",
            str(tmp_path / "a.ckpt"),
            "--checkpoint",
            str(tmp_path / "b.ckpt"),
            "--step",
            "10",
            "--config",
            str(config_path),
        ]
    )
    with pytest.raises(PhonemeAdapterConversionError, match="--step"):
        run(args)


def test_cli_requires_a_config_source(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--checkpoint", str(tmp_path / "a.ckpt")])
