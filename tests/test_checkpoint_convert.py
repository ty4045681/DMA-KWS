from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from dma_kws.stage2.model_factory import build_qbyt
from dma_kws.stage2.readout import QbyTAlignmentSpec, resolve_qbyt_alignment
from dma_kws.training.checkpoint_convert import (
    CheckpointConversionError,
    convert_checkpoint,
    merge_lora_checkpoint_state,
)
from dma_kws.training.checkpoint_io import (
    QBYT_ALIGNMENT_SPEC_KEY,
    QBYT_READOUT_VERSION,
    QBYT_READOUT_VERSION_KEY,
    STAGE2_BASE_FINGERPRINT_KEY,
    assert_qbyt_readout_version,
    extract_state_dict,
    fingerprint_stage2_base,
)
from dma_kws.training.lora import inject_qbyt_lora, load_lora_state_dict, merge_lora
from scripts.convert_stage2_checkpoints import (
    build_parser,
    discover_checkpoints,
    main,
    run,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DICT_PATH = REPO_ROOT / "data" / "dict" / "lang_char.txt"
VOCAB_SIZE = 71


def _alignment(**overrides) -> dict:
    spec = QbyTAlignmentSpec().as_dict()
    spec.update(overrides)
    return QbyTAlignmentSpec(**spec).as_dict()


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
    alignment: dict | None = None,
) -> dict:
    adapt = {
        "keyword": "hey eva",
        "slug": "",
        "phase": "tts",
        "rank": rank,
        "lora_targets": targets or ["audio_key.weight", "text_query.weight"],
    }
    if alpha is not None:
        adapt["alpha"] = alpha
    stage2 = {
        "qbyt_embed_dim": 4,
        "qbyt_layers": 1,
        "qbyt_alignment": _alignment() if alignment is None else alignment,
    }
    return {
        "tokenizer": {
            "dict_path": str(DICT_PATH),
            "split_with_space": " ",
        },
        "stage1": {
            "encoder_type": "conformer",
            "encoder_output_dim": 3,
        },
        "stage2": stage2,
        "adapt": adapt,
    }


def _stage2_state(
    *,
    alignment: dict | None = None,
    input_dim: int = 3,
) -> dict[str, torch.Tensor]:
    stage2 = _config(alignment=alignment)["stage2"]
    qbyt = build_qbyt(
        stage2,
        input_dim=input_dim,
        vocab_size=VOCAB_SIZE,
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
    alignment_spec: dict | None = None,
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
        if readout_version >= 5:
            if alignment_spec is None and config is not None:
                stage2 = config.get("stage2")
                if isinstance(stage2, dict):
                    alignment_spec = resolve_qbyt_alignment(stage2).as_dict()
            checkpoint[QBYT_ALIGNMENT_SPEC_KEY] = (
                _alignment() if alignment_spec is None else copy.deepcopy(alignment_spec)
            )
    return checkpoint


_POOLING_V41_READOUT = {
    "mode": "eps_softmin",
    "temperature": 1.0,
    "sink_token": True,
    "text_position": "learned",
    "audio_position": "relative_bias",
    "relative_num_buckets": 32,
    "relative_max_distance": 64,
}


def _pooling_config(
    *,
    rank: int = 2,
    alpha: float | None = 4.0,
    phase: str = "joint",
    targets: list[str] | None = None,
) -> dict:
    config = _config(
        rank=rank,
        alpha=alpha,
        targets=targets or ["in_proj_weight", "out_proj.weight"],
    )
    config["adapt"]["phase"] = phase
    config["adapt"]["slug"] = ""
    config["stage2"] = {
        "encoder_output_dim": 3,
        "qbyt_embed_dim": 8,
        "qbyt_layers": 1,
        "qbyt_readout_version": 4,
        "qbyt_readout": dict(_POOLING_V41_READOUT),
    }
    return config


def _pooling_lora_state() -> dict[str, torch.Tensor]:
    qbyt = build_qbyt(
        _pooling_config()["stage2"],
        input_dim=3,
        vocab_size=VOCAB_SIZE,
    )
    inject_qbyt_lora(qbyt, rank=2, alpha=4.0)
    with torch.no_grad():
        for name, parameter in qbyt.named_parameters():
            if name.endswith(".lora_A"):
                parameter.fill_(0.25)
            elif name.endswith(".lora_B"):
                parameter.fill_(0.5)
    state = {
        "encoder.weight": torch.arange(6, dtype=torch.float32).reshape(2, 3),
        "encoder.bias": torch.arange(2, dtype=torch.float32),
    }
    state.update({f"qbyt.{key}": value for key, value in qbyt.state_dict().items()})
    return state


def _bounded_alignment() -> dict:
    from dma_kws.stage2.readout_bounded import QbyTAlignmentSpec

    return QbyTAlignmentSpec().as_dict()


def _bounded_config(
    *,
    rank: int = 2,
    alpha: float | None = 4.0,
    phase: str = "tts",
) -> dict:
    config = _config(rank=rank, alpha=alpha)
    config["adapt"]["phase"] = phase
    config["stage2"] = {
        "encoder_output_dim": 3,
        "qbyt_embed_dim": 8,
        "qbyt_layers": 1,
        "qbyt_readout_version": 5,
        "qbyt_alignment": _bounded_alignment(),
    }
    return config


def _bounded_lora_state() -> dict[str, torch.Tensor]:
    qbyt = build_qbyt(
        _bounded_config()["stage2"],
        input_dim=3,
        vocab_size=VOCAB_SIZE,
    )
    inject_qbyt_lora(qbyt, rank=2, alpha=4.0)
    with torch.no_grad():
        for name, parameter in qbyt.named_parameters():
            if name.endswith(".lora_A"):
                parameter.fill_(0.25)
            elif name.endswith(".lora_B"):
                parameter.fill_(0.5)
    state = {
        "encoder.weight": torch.arange(6, dtype=torch.float32).reshape(2, 3),
        "encoder.bias": torch.arange(2, dtype=torch.float32),
    }
    state.update({f"qbyt.{key}": value for key, value in qbyt.state_dict().items()})
    return state


def _assert_converted_adapter_loads(
    adapter: dict,
    *,
    stage2: dict,
    rank: int,
    alpha: float,
    targets: tuple[str, ...],
) -> None:
    from dma_kws.stage2.adapt import _validate_adapter_checkpoint

    state = _validate_adapter_checkpoint(
        adapter,
        source="converted.adapter.pt",
        keyword=str(adapter["keyword"]),
        rank=rank,
        alpha=alpha,
        targets=targets,
        base_model_sha256=adapter.get(STAGE2_BASE_FINGERPRINT_KEY),
    )
    qbyt = build_qbyt(stage2, input_dim=3, vocab_size=VOCAB_SIZE)
    inject_qbyt_lora(qbyt, rank=rank, alpha=alpha, targets=list(targets))
    load_lora_state_dict(qbyt, state)


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
        "qbyt.audio_key.weight": (
            state["qbyt.audio_key.weight"].clone(),
            torch.tensor(
                [[1.0, 2.0, 3.0, 4.0], [0.5, 1.0, 1.5, 2.0]]
            ),
            torch.arange(8, dtype=torch.float32).reshape(4, 2) / 10,
        ),
        "qbyt.text_query.weight": (
            state["qbyt.text_query.weight"].clone(),
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


def test_stage2_base_fingerprint_is_stable_across_lora_parametrization() -> None:
    lora_state, groups = _lora_state()
    plain_state = {
        key: value
        for key, value in lora_state.items()
        if ".parametrizations." not in key
    }
    for target, (original, _lora_a, _lora_b) in groups.items():
        plain_state[target] = original

    assert fingerprint_stage2_base(lora_state) == fingerprint_stage2_base(plain_state)

    changed = dict(plain_state)
    changed["encoder.bias"] = changed["encoder.bias"] + 1
    assert fingerprint_stage2_base(changed) != fingerprint_stage2_base(plain_state)


def test_convert_stage2_checkpoint_writes_only_inference_payload(tmp_path: Path) -> None:
    source = tmp_path / "step.ckpt"
    output = tmp_path / "step.pt"
    alignment = _alignment(
        max_inter_phone_gap_frames=1,
        max_keyword_span_frames=40,
        weakest_phone_temperature=0.35,
        weakest_phone_weight=0.75,
        local_context_kernel=7,
    )
    config = _config(alignment=alignment)
    original_state = _stage2_state(alignment=alignment)
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
        QBYT_ALIGNMENT_SPEC_KEY,
    }
    assert payload["step"] == 123
    assert payload["vocab_size"] == VOCAB_SIZE
    assert payload["tokenizer_dict_path"] == str(DICT_PATH)
    assert payload["config"] == config
    assert payload[QBYT_ALIGNMENT_SPEC_KEY] == alignment
    for key, value in original_state.items():
        assert torch.equal(payload["model_state_dict"][key], value)
    assert_qbyt_readout_version(
        payload,
        source=output,
        expected_alignment=resolve_qbyt_alignment(config["stage2"]),
    )
    assert extract_state_dict(payload) is payload["model_state_dict"]


@pytest.mark.parametrize("method", ["qbyt_full", "encoder_qbyt_full"])
def test_full_adaptation_conversion_preserves_trained_encoder_and_qbyt(
    tmp_path: Path, method: str,
) -> None:
    config = _config()
    config["adapt"]["method"] = method
    state = _stage2_state()
    # A fine-tuned encoder must survive conversion; rebuilding the initial
    # encoder or exporting QbyT alone would silently lose these updates.
    state["encoder.weight"] = state["encoder.weight"] + 0.125
    checkpoint = _checkpoint(state, config=config)
    checkpoint["checkpoint_kind"] = f"stage2_{method}"
    checkpoint["method"] = method
    source, output = tmp_path / "full.ckpt", tmp_path / "full.pt"
    torch.save(checkpoint, source)

    result = convert_checkpoint(source, output)

    assert result.kind == "stage2"
    assert result.outputs == (output,)
    payload = torch.load(output, map_location="cpu")
    assert payload["config"]["adapt"]["method"] == method
    assert set(payload["model_state_dict"]) == set(state)
    for key, expected in state.items():
        torch.testing.assert_close(payload["model_state_dict"][key], expected, rtol=0, atol=0)
    assert "optimizer_states" not in payload
    with pytest.raises(CheckpointConversionError, match="non-LoRA"):
        convert_checkpoint(source, tmp_path / "adapter.pt", lora_output="adapter")


@pytest.mark.parametrize("version", [1, 2, 3, 4, 5])
def test_convert_rejects_pre_v6_full_qbyt_checkpoint(
    tmp_path: Path,
    version: int,
) -> None:
    source = tmp_path / f"v{version}.ckpt"
    config = _config()
    torch.save(
        _checkpoint(
            _stage2_state(),
            config=config,
            readout_version=version,
        ),
        source,
    )

    with pytest.raises(
        CheckpointConversionError,
        match=rf"readout version {version!r}",
    ):
        convert_checkpoint(source, tmp_path / f"v{version}.pt")


def test_convert_rejects_missing_partial_or_mismatched_v6_alignment_spec(
    tmp_path: Path,
) -> None:
    source = tmp_path / "invalid.ckpt"
    config = _config()
    checkpoint = _checkpoint(_stage2_state(), config=config)

    missing = copy.deepcopy(checkpoint)
    missing.pop(QBYT_ALIGNMENT_SPEC_KEY)
    torch.save(missing, source)
    with pytest.raises(CheckpointConversionError, match=QBYT_ALIGNMENT_SPEC_KEY):
        convert_checkpoint(source, tmp_path / "missing.pt")

    partial = copy.deepcopy(checkpoint)
    partial[QBYT_ALIGNMENT_SPEC_KEY] = {
        "topology": "keyword_filler_segmental_crf_v1"
    }
    torch.save(partial, source)
    with pytest.raises(CheckpointConversionError, match="missing="):
        convert_checkpoint(source, tmp_path / "partial.pt")

    mismatched = copy.deepcopy(checkpoint)
    mismatched[QBYT_ALIGNMENT_SPEC_KEY] = _alignment(
        weakest_phone_temperature=0.4
    )
    torch.save(mismatched, source)
    with pytest.raises(CheckpointConversionError, match="disagrees with config"):
        convert_checkpoint(source, tmp_path / "mismatched.pt")


def test_v6_checkpoint_without_embedded_config_uses_explicit_fallback(
    tmp_path: Path,
) -> None:
    source = tmp_path / "v6.ckpt"
    output = tmp_path / "v6.pt"
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
    with pytest.raises(CheckpointConversionError, match="unversioned"):
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
    truncated.pop("qbyt.phone_bias")
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
    qbyt = build_qbyt(
        config["stage2"],
        input_dim=adapter.output_dim,
        vocab_size=VOCAB_SIZE,
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
    assert adapter["lora_targets"] == ["audio_key.weight", "text_query.weight"]
    assert adapter["keyword"] == "hey eva"
    assert adapter["slug"] == "hey_eva"
    assert adapter["phase"] == "tts"
    assert adapter["checkpoint_kind"] == "stage2_lora_adapter"
    assert adapter[QBYT_ALIGNMENT_SPEC_KEY] == _alignment()
    assert merged[QBYT_ALIGNMENT_SPEC_KEY] == _alignment()
    assert adapter[STAGE2_BASE_FINGERPRINT_KEY] == fingerprint_stage2_base(state)
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


def test_adapter_only_skips_tokenizer_validation_but_requires_a_complete_base(
    tmp_path: Path,
) -> None:
    source = tmp_path / "adapter_only.ckpt"
    adapter_output = tmp_path / "adapter_only.pt"
    state, _ = _lora_state()
    config = _config()
    config["tokenizer"]["dict_path"] = "/nonexistent/lang_char.txt"
    torch.save(_checkpoint(state, config=config), source)

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

    truncated = dict(state)
    truncated.pop("qbyt.phone_bias")
    torch.save(_checkpoint(truncated, config=config), source)
    with pytest.raises(CheckpointConversionError, match="QbyT weights do not match"):
        convert_checkpoint(
            source,
            tmp_path / "truncated_adapter.pt",
            lora_output="adapter",
        )


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
        "qbyt.audio_key.parametrizations.weight.0.lora_B"
    )
    with pytest.raises(CheckpointConversionError, match="Incomplete LoRA"):
        merge_lora_checkpoint_state(broken, alpha=4.0)

    unsupported = dict(state)
    _add_lora_group(
        unsupported,
        "encoder.block.weight",
        original=torch.ones(4, 4),
        lora_a=torch.ones(2, 4),
        lora_b=torch.ones(4, 2),
    )
    with pytest.raises(
        CheckpointConversionError,
        match="Unsupported parametrization",
    ):
        merge_lora_checkpoint_state(unsupported, alpha=4.0)


def test_raw_lora_merge_matches_project_parametrization() -> None:
    qbyt = build_qbyt(
        _config()["stage2"],
        input_dim=3,
        vocab_size=VOCAB_SIZE,
    )
    inject_qbyt_lora(
        qbyt,
        rank=2,
        alpha=4.0,
        targets=[
            "audio_projection.weight",
            "audio_key.weight",
            "text_query.weight",
        ],
    )
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


def test_pooling_lora_merge_matches_phone_matchor_parametrization() -> None:
    qbyt = build_qbyt(
        _pooling_config()["stage2"],
        input_dim=3,
        vocab_size=VOCAB_SIZE,
    )
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


def test_convert_pooling_lora_checkpoint_writes_joint_merged_and_adapter(
    tmp_path: Path,
) -> None:
    source = tmp_path / "pooling_lora.ckpt"
    merged_output = tmp_path / "pooling_lora.pt"
    adapter_output = tmp_path / "pooling_lora.adapter.pt"
    state = _pooling_lora_state()
    config = _pooling_config()
    torch.save(
        _checkpoint(state, config=config, readout_version=4),
        source,
    )

    result = convert_checkpoint(source, merged_output)

    assert result.kind == "lora"
    assert result.outputs == (merged_output, adapter_output)
    merged = torch.load(merged_output, map_location="cpu")
    adapter = torch.load(adapter_output, map_location="cpu")

    assert merged["phase"] == "joint"
    assert adapter["phase"] == "joint"
    assert adapter["lora_targets"] == ["in_proj_weight", "out_proj.weight"]
    assert merged[QBYT_READOUT_VERSION_KEY] == 4
    assert adapter[QBYT_READOUT_VERSION_KEY] == 4
    assert QBYT_ALIGNMENT_SPEC_KEY not in merged or merged[QBYT_ALIGNMENT_SPEC_KEY] is None
    assert QBYT_ALIGNMENT_SPEC_KEY not in adapter
    assert not any(
        ".parametrizations." in key or key.endswith((".lora_A", ".lora_B"))
        for key in merged["model_state_dict"]
    )
    qbyt = build_qbyt(config["stage2"], input_dim=3, vocab_size=VOCAB_SIZE)
    qbyt.load_state_dict(
        {
            key[len("qbyt.") :]: value
            for key, value in merged["model_state_dict"].items()
            if key.startswith("qbyt.")
        },
        strict=True,
    )
    assert adapter["lora_state_dict"]
    assert all(
        not key.startswith("qbyt.") for key in adapter["lora_state_dict"]
    )
    assert all("phone_matchor" in key for key in adapter["lora_state_dict"])
    assert all(
        key.endswith((".lora_A", ".lora_B"))
        for key in adapter["lora_state_dict"]
    )
    assert adapter[STAGE2_BASE_FINGERPRINT_KEY] == fingerprint_stage2_base(state)
    assert_qbyt_readout_version(adapter, source=adapter_output)


def test_convert_pooling_lora_rewrites_default_keyword_filler_targets(
    tmp_path: Path,
) -> None:
    source = tmp_path / "pooling_fallback.ckpt"
    output = tmp_path / "pooling_fallback.pt"
    state = _pooling_lora_state()
    config = _pooling_config(targets=["audio_key.weight", "text_query.weight"])
    torch.save(_checkpoint(state, config=config, readout_version=4), source)

    convert_checkpoint(source, output)

    adapter = torch.load(output.with_name("pooling_fallback.adapter.pt"), map_location="cpu")
    assert adapter["lora_targets"] == ["in_proj_weight", "out_proj.weight"]
    assert adapter["config"]["adapt"]["lora_targets"] == [
        "in_proj_weight",
        "out_proj.weight",
    ]
    _assert_converted_adapter_loads(
        adapter,
        stage2=config["stage2"],
        rank=2,
        alpha=4.0,
        targets=("in_proj_weight", "out_proj.weight"),
    )


def test_convert_v5_bounded_lora_checkpoint_roundtrips_adapter(
    tmp_path: Path,
) -> None:
    source = tmp_path / "bounded_lora.ckpt"
    merged_output = tmp_path / "bounded_lora.pt"
    state = _bounded_lora_state()
    config = _bounded_config()
    torch.save(
        _checkpoint(
            state,
            config=config,
            readout_version=5,
            alignment_spec=_bounded_alignment(),
        ),
        source,
    )

    result = convert_checkpoint(source, merged_output)
    adapter = torch.load(result.outputs[1], map_location="cpu")
    assert adapter["lora_targets"] == ["audio_key.weight", "text_query.weight"]
    assert adapter[QBYT_READOUT_VERSION_KEY] == 5
    qbyt = build_qbyt(config["stage2"], input_dim=3, vocab_size=VOCAB_SIZE)
    qbyt.load_state_dict(
        {
            key[len("qbyt.") :]: value
            for key, value in torch.load(merged_output, map_location="cpu")[
                "model_state_dict"
            ].items()
            if key.startswith("qbyt.")
        },
        strict=True,
    )
    _assert_converted_adapter_loads(
        adapter,
        stage2=config["stage2"],
        rank=2,
        alpha=4.0,
        targets=("audio_key.weight", "text_query.weight"),
    )


def test_lora_merge_rejects_mixed_readout_families() -> None:
    state, _ = _lora_state()
    _add_lora_group(
        state,
        "qbyt.phone_matchor.layers.0.self_attn.in_proj_weight",
        original=torch.ones(12, 4),
        lora_a=torch.ones(2, 4),
        lora_b=torch.ones(12, 2),
    )
    with pytest.raises(CheckpointConversionError, match="mixes LoRA layouts"):
        merge_lora_checkpoint_state(state, alpha=4.0)


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
