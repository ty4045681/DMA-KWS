"""Convert Stage II Lightning checkpoints to inference ``.pt`` payloads.

The converter is deliberately strict about metadata.  In particular, it never
stamps an unversioned checkpoint with the current QbyT readout version, and it
never guesses the LoRA scaling factor (``alpha``), which is not stored in a
parametrized model state dict.
"""

from __future__ import annotations

import copy
import math
import os
import re
import uuid
import warnings
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Literal, Mapping

import torch

from dma_kws.pathing import resolve_dict_path
from dma_kws.stage2.adapt_paths import slugify
from dma_kws.tokenizer import validate_lang_char_dict
from dma_kws.training.checkpoint_io import (
    QBYT_ALIGNMENT_SPEC_KEY,
    QBYT_READOUT_VERSION,
    QBYT_READOUT_VERSION_KEY,
    STAGE2_BASE_FINGERPRINT_KEY,
    canonical_stage2_base_state,
    fingerprint_stage2_base,
)

CheckpointKind = Literal["stage2", "lora"]
LoraOutput = Literal["merged", "adapter", "both"]

_PARAMETRIZED_ORIGINAL_RE = re.compile(
    r"^(?P<module>.+)\.parametrizations\.(?P<parameter>[^.]+)\.original$"
)
_PROJECT_LORA_TARGET_RE = re.compile(
    r"^qbyt\.(?:audio_projection|audio_key|text_query)\.weight$"
)
_LORA_PARAMETER_SUFFIXES = (".lora_A", ".lora_B")


class CheckpointConversionError(ValueError):
    """Raised when a checkpoint cannot be converted without guessing."""


@dataclass(frozen=True)
class ConversionResult:
    """Outputs written for one source checkpoint."""

    source: Path
    kind: CheckpointKind
    outputs: tuple[Path, ...]


def _as_mapping(value: Any, *, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise CheckpointConversionError(f"{name} must be a mapping")
    return dict(value)


def _state_dict_from_checkpoint(
    checkpoint: Mapping[str, Any],
    *,
    weights_key: str,
) -> dict[str, torch.Tensor]:
    value = checkpoint.get(weights_key)
    if not isinstance(value, Mapping):
        raise CheckpointConversionError(
            f"Checkpoint has no mapping at {weights_key!r}; "
            f"available keys: {sorted(str(key) for key in checkpoint)[:20]}"
        )
    state = dict(value)
    if not state:
        raise CheckpointConversionError(f"Checkpoint {weights_key!r} is empty")
    if not all(isinstance(key, str) for key in state):
        raise CheckpointConversionError("Model state keys must all be strings")
    if not all(isinstance(value, torch.Tensor) for value in state.values()):
        bad = next(key for key, value in state.items() if not isinstance(value, torch.Tensor))
        raise CheckpointConversionError(f"Model state entry {bad!r} is not a tensor")
    if not any(key.startswith("encoder.") for key in state):
        raise CheckpointConversionError(
            "Checkpoint is not a full Stage II model: no encoder.* weights were found"
        )
    if not any(key.startswith("qbyt.") for key in state):
        raise CheckpointConversionError(
            "Checkpoint is not a Stage II model: no qbyt.* weights were found"
        )
    return state


def _validate_weight_selection(
    checkpoint: Mapping[str, Any],
    *,
    weights_key: str,
) -> None:
    """Reject an uninitialized EMA state instead of exporting initial weights."""
    if weights_key != "state_dict" or "current_model_state" not in checkpoint:
        return
    averaging_state = checkpoint.get("averaging_state")
    if not isinstance(averaging_state, Mapping):
        return
    n_averaged = averaging_state.get("n_averaged")
    if isinstance(n_averaged, torch.Tensor):
        if n_averaged.numel() != 1:
            raise CheckpointConversionError(
                "averaging_state.n_averaged must be a scalar"
            )
        n_averaged = n_averaged.item()
    if n_averaged is None:
        return
    count = _equal_int_metadata(
        "EMA update count",
        [("checkpoint.averaging_state.n_averaged", n_averaged)],
    )
    if count == 0:
        raise CheckpointConversionError(
            "Checkpoint's EMA/averaged state has n_averaged=0 and still contains "
            "the model initialization. Use --weights current_model_state for this "
            "checkpoint."
        )


def detect_checkpoint_kind(state: Mapping[str, torch.Tensor]) -> CheckpointKind:
    """Identify a normal Stage II state or a parametrized LoRA state."""
    has_lora = any(
        ".parametrizations." in key or key.endswith(_LORA_PARAMETER_SUFFIXES)
        for key in state
    )
    return "lora" if has_lora else "stage2"


def _validate_checkpoint_kind_metadata(
    checkpoint: Mapping[str, Any],
    detected: CheckpointKind,
) -> None:
    saved = checkpoint.get("checkpoint_kind")
    if saved is None:
        return
    expected = "stage2_lora" if detected == "lora" else "stage2"
    if str(saved) != expected:
        raise CheckpointConversionError(
            f"checkpoint_kind={saved!r} disagrees with detected {detected!r} weights"
        )


def inspect_checkpoint_kind(
    source: Path | str,
    *,
    weights_key: str = "state_dict",
) -> CheckpointKind:
    """Read only enough checkpoint structure to identify its Stage II kind."""
    source = Path(source)
    if not source.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {source}")
    raw_checkpoint = torch.load(source, map_location="cpu")
    checkpoint = _as_mapping(raw_checkpoint, name=f"checkpoint {source}")
    _validate_weight_selection(checkpoint, weights_key=weights_key)
    state = _state_dict_from_checkpoint(checkpoint, weights_key=weights_key)
    kind = detect_checkpoint_kind(state)
    _validate_checkpoint_kind_metadata(checkpoint, kind)
    return kind


def _resolved_config(
    checkpoint: Mapping[str, Any],
    fallback_config: Mapping[str, Any] | None,
) -> dict[str, Any]:
    embedded = checkpoint.get("config")
    embedded_config = None
    if embedded is not None:
        embedded_config = _as_mapping(embedded, name="checkpoint config")

    supplied_config = dict(fallback_config) if fallback_config is not None else None
    if embedded_config is not None and supplied_config is not None:
        if embedded_config != supplied_config:
            warnings.warn(
                "The checkpoint already embeds its resolved training config, so the "
                "supplied config fallback (--config/--experiment/--adapt-params) is "
                "ignored. Config arguments are only needed for early v6 checkpoints "
                "without embedded config.",
                UserWarning,
                stacklevel=3,
            )
        return copy.deepcopy(embedded_config)
    if embedded_config is not None:
        return copy.deepcopy(embedded_config)
    if supplied_config is not None:
        return copy.deepcopy(supplied_config)
    raise CheckpointConversionError(
        "Checkpoint does not embed its resolved training config. Supply the exact "
        "v6 training config with --experiment/--override or --config. Pre-v6 "
        "QbyT checkpoints are intentionally not convertible."
    )


def _require_compatible_readout(
    checkpoint: Mapping[str, Any],
    config: Mapping[str, Any],
) -> int:
    """Require exact v6 alignment metadata; older score heads are not convertible."""

    from dma_kws.stage2.readout import resolve_qbyt_score_spec
    from dma_kws.training.checkpoint_io import (
        QBYT_READOUT_VERSION_KEY,
        assert_qbyt_readout_version,
    )

    stage2 = config.get("stage2")
    if not isinstance(stage2, Mapping):
        raise CheckpointConversionError("Resolved config has no stage2 mapping")
    try:
        expected = resolve_qbyt_score_spec(stage2)
        assert_qbyt_readout_version(
            checkpoint,
            source="checkpoint conversion input",
            expected_alignment=expected,
        )
    except (SystemExit, ValueError) as exc:
        raise CheckpointConversionError(
            f"Checkpoint QbyT readout is not compatible with the resolved config: {exc}"
        ) from exc
    saved = checkpoint.get(QBYT_READOUT_VERSION_KEY)
    return int(saved) if saved is not None else expected.version


def _equal_int_metadata(
    name: str,
    candidates: list[tuple[str, Any]],
) -> int:
    present: list[tuple[str, int]] = []
    for source, value in candidates:
        if value is not None:
            if isinstance(value, bool):
                raise CheckpointConversionError(
                    f"{name} from {source} must be an integer, got {value!r}"
                )
            if isinstance(value, Integral):
                parsed = int(value)
            elif isinstance(value, Real):
                numeric = float(value)
                if not math.isfinite(numeric) or not numeric.is_integer():
                    raise CheckpointConversionError(
                        f"{name} from {source} must be an integer, got {value!r}"
                    )
                parsed = int(numeric)
            elif isinstance(value, str) and re.fullmatch(r"[+-]?\d+", value.strip()):
                parsed = int(value.strip())
            else:
                raise CheckpointConversionError(
                    f"{name} from {source} must be an integer, got {value!r}"
                )
            present.append((source, parsed))
    if not present:
        raise CheckpointConversionError(f"Checkpoint does not provide {name}")
    expected = present[0][1]
    disagreements = [(source, value) for source, value in present if value != expected]
    if disagreements:
        details = ", ".join(f"{source}={value}" for source, value in present)
        raise CheckpointConversionError(f"Conflicting {name} metadata: {details}")
    return expected


def _checkpoint_step(checkpoint: Mapping[str, Any]) -> int:
    # Lightning's global_step is authoritative.  The ``step`` fallback supports
    # custom payloads without deriving anything from a filename.
    global_step = checkpoint.get("global_step")
    source = "checkpoint.global_step"
    value = global_step
    if global_step is None:
        source = "checkpoint.step"
        value = checkpoint.get("step")
    step = _equal_int_metadata(
        "training step",
        [(source, value)],
    )
    if step < 0:
        raise CheckpointConversionError(f"Training step must be non-negative, got {step}")
    return step


def _state_vocab_size(state: Mapping[str, torch.Tensor]) -> int:
    embedding = state.get("qbyt.text_projection.weight")
    if embedding is None:
        raise CheckpointConversionError(
            "Checkpoint has no qbyt.text_projection.weight; it is not a complete "
            "Stage II QbyT state"
        )
    if embedding.ndim != 2:
        raise CheckpointConversionError(
            "qbyt.text_projection.weight must be a 2-D embedding matrix"
        )
    return int(embedding.shape[0])


def _tokenizer_metadata(
    checkpoint: Mapping[str, Any],
    config: Mapping[str, Any],
    state: Mapping[str, torch.Tensor],
    *,
    validate_tokenizer: bool,
) -> tuple[str, int]:
    hyper_parameters = checkpoint.get("hyper_parameters")
    hparams = dict(hyper_parameters) if isinstance(hyper_parameters, Mapping) else {}
    state_vocab_size = _state_vocab_size(state)
    vocab_size = _equal_int_metadata(
        "vocab_size",
        [
            ("checkpoint.vocab_size", checkpoint.get("vocab_size")),
            ("checkpoint.hyper_parameters.vocab_size", hparams.get("vocab_size")),
            ("qbyt.text_projection.weight", state_vocab_size),
        ],
    )
    if vocab_size <= 0:
        raise CheckpointConversionError(f"vocab_size must be positive, got {vocab_size}")

    try:
        dict_path = resolve_dict_path(config)
    except (KeyError, TypeError, ValueError) as exc:
        raise CheckpointConversionError(
            "Resolved config does not contain a usable tokenizer.dict_path"
        ) from exc

    if validate_tokenizer:
        try:
            validate_lang_char_dict(dict_path)
        except ValueError as exc:
            raise CheckpointConversionError(str(exc)) from exc
        file_vocab_size = len(dict_path.read_text(encoding="utf-8").splitlines())
        if file_vocab_size != vocab_size:
            raise CheckpointConversionError(
                f"Tokenizer {dict_path} has {file_vocab_size} entries but the checkpoint "
                f"embedding has vocab_size={vocab_size}"
            )

    return str(dict_path), vocab_size


def _submodule_state(
    state: Mapping[str, torch.Tensor],
    prefix: str,
) -> dict[str, torch.Tensor]:
    dotted = f"{prefix}."
    return {
        key[len(dotted) :]: value
        for key, value in state.items()
        if key.startswith(dotted)
    }


def _validate_deployable_model_state(
    state: Mapping[str, torch.Tensor],
    config: Mapping[str, Any],
    *,
    vocab_size: int,
    validate_encoder: bool,
) -> None:
    """Strictly validate the model portions against the resolved architecture."""
    stage1 = config.get("stage1")
    stage2 = config.get("stage2")
    if not isinstance(stage1, Mapping) or not isinstance(stage2, Mapping):
        raise CheckpointConversionError(
            "Resolved config must contain stage1 and stage2 mappings"
        )
    unexpected_roots = sorted(
        key
        for key in state
        if not key.startswith(("encoder.", "adapter.", "qbyt."))
    )
    if unexpected_roots:
        raise CheckpointConversionError(
            "Model state contains keys that the inference Stage2Model cannot load: "
            f"{unexpected_roots[:5]}"
        )

    encoder_dim = int(
        stage2.get(
            "encoder_output_dim",
            stage1.get("encoder_output_dim", 144),
        )
    )
    if validate_encoder:
        try:
            from dma_kws.nn import build_encoder

            encoder = build_encoder(dict(stage1), output_dim=encoder_dim)
            encoder.load_state_dict(_submodule_state(state, "encoder"), strict=True)
        except (ImportError, RuntimeError, SystemExit, TypeError, ValueError) as exc:
            raise CheckpointConversionError(
                "Encoder weights do not match the resolved config (or its optional "
                f"backend dependencies are unavailable): {exc}. "
                "Use --skip-encoder-validation only when dependency installation "
                "is impossible and the config is independently verified."
            ) from exc

    adapter_cfg = stage2.get("phoneme_adapter", {}) or {}
    if not isinstance(adapter_cfg, Mapping):
        raise CheckpointConversionError("config.stage2.phoneme_adapter must be a mapping")
    adapter_enabled = bool(adapter_cfg.get("enabled", False))
    adapter_state = _submodule_state(state, "adapter")
    if adapter_enabled != bool(adapter_state):
        described = "enabled" if adapter_enabled else "disabled"
        found = "present" if adapter_state else "absent"
        raise CheckpointConversionError(
            f"Config says phoneme adapter is {described}, but adapter.* weights are {found}"
        )

    qbyt_input_dim = encoder_dim
    if adapter_enabled:
        try:
            from dma_kws.phoneme_adapter.module import build_phoneme_adapter

            adapter = build_phoneme_adapter(
                adapter_cfg,
                input_dim=encoder_dim,
                vocab_size=vocab_size,
                causal=bool(stage1.get("causal", False)),
            )
            adapter.load_state_dict(adapter_state, strict=True)
            qbyt_input_dim = int(adapter.output_dim)
        except (ImportError, RuntimeError, SystemExit, ValueError) as exc:
            raise CheckpointConversionError(
                f"Adapter weights do not match the resolved config: {exc}"
            ) from exc

    try:
        from dma_kws.stage2.model_factory import build_qbyt

        qbyt = build_qbyt(
            stage2,
            input_dim=qbyt_input_dim,
            vocab_size=vocab_size,
        )
        qbyt.load_state_dict(_submodule_state(state, "qbyt"), strict=True)
    except (ImportError, RuntimeError, SystemExit, TypeError, ValueError) as exc:
        raise CheckpointConversionError(
            f"QbyT weights do not match the resolved config: {exc}"
        ) from exc


def _base_model_metadata(
    checkpoint: Mapping[str, Any],
    config: dict[str, Any],
    state: Mapping[str, torch.Tensor],
    *,
    validate_tokenizer: bool,
    validate_encoder: bool,
) -> dict[str, Any]:
    tokenizer_dict_path, vocab_size = _tokenizer_metadata(
        checkpoint,
        config,
        state,
        validate_tokenizer=validate_tokenizer,
    )
    _validate_deployable_model_state(
        state,
        config,
        vocab_size=vocab_size,
        validate_encoder=validate_encoder,
    )
    return {
        "config": config,
        "step": _checkpoint_step(checkpoint),
        "tokenizer_dict_path": tokenizer_dict_path,
        "vocab_size": vocab_size,
        QBYT_READOUT_VERSION_KEY: _require_compatible_readout(checkpoint, config),
        QBYT_ALIGNMENT_SPEC_KEY: copy.deepcopy(
            checkpoint.get(QBYT_ALIGNMENT_SPEC_KEY)
        ),
    }


def _matching_value(
    name: str,
    candidates: list[tuple[str, Any]],
    *,
    required: bool = True,
) -> Any:
    present = [(source, value) for source, value in candidates if value not in (None, "")]
    if not present:
        if required:
            raise CheckpointConversionError(f"LoRA checkpoint does not provide {name}")
        return None
    expected = present[0][1]
    if any(value != expected for _, value in present[1:]):
        details = ", ".join(f"{source}={value!r}" for source, value in present)
        raise CheckpointConversionError(f"Conflicting LoRA {name} metadata: {details}")
    return expected


def _lora_identity(
    checkpoint: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, str]:
    adapt = config.get("adapt")
    if not isinstance(adapt, Mapping):
        raise CheckpointConversionError(
            "Resolved config has no adapt mapping for LoRA metadata"
        )

    keyword = str(
        _matching_value(
            "keyword",
            [
                ("checkpoint.keyword", checkpoint.get("keyword")),
                ("config.adapt.keyword", adapt.get("keyword")),
            ],
        )
    )
    configured_slug = adapt.get("slug") or slugify(keyword)
    slug = str(
        _matching_value(
            "slug",
            [
                ("checkpoint.slug", checkpoint.get("slug")),
                ("config.adapt.slug", configured_slug),
            ],
        )
    )
    phase = str(
        _matching_value(
            "phase",
            [
                ("checkpoint.phase", checkpoint.get("phase")),
                ("config.adapt.phase", adapt.get("phase")),
            ],
        )
    )
    return {"keyword": keyword, "slug": slug, "phase": phase}


def _lora_groups(
    state: Mapping[str, torch.Tensor],
) -> list[tuple[str, str, str, str]]:
    """Return ``(target, original, A, B)`` keys for each LoRA parametrization."""
    groups: list[tuple[str, str, str, str]] = []
    for original_key in state:
        match = _PARAMETRIZED_ORIGINAL_RE.match(original_key)
        if match is None:
            continue
        parameter_root = original_key[: -len(".original")]
        a_key = f"{parameter_root}.0.lora_A"
        b_key = f"{parameter_root}.0.lora_B"
        if a_key not in state or b_key not in state:
            missing = [key for key in (a_key, b_key) if key not in state]
            raise CheckpointConversionError(
                f"Incomplete LoRA parametrization for {original_key}: missing {missing}"
            )
        target_key = f"{match.group('module')}.{match.group('parameter')}"
        if _PROJECT_LORA_TARGET_RE.fullmatch(target_key) is None:
            raise CheckpointConversionError(
                "Unsupported parametrization outside the QbyT v6 projection set "
                f"(audio_projection/audio_key/text_query weights): {target_key}"
            )
        groups.append((target_key, original_key, a_key, b_key))

    lora_keys = {
        key
        for key in state
        if key.endswith(_LORA_PARAMETER_SUFFIXES)
    }
    grouped_lora_keys = {key for group in groups for key in group[2:]}
    orphaned = sorted(lora_keys - grouped_lora_keys)
    if orphaned:
        raise CheckpointConversionError(
            f"LoRA state contains adapter tensors without matching originals: {orphaned[:5]}"
        )
    if not groups:
        raise CheckpointConversionError(
            "Checkpoint was identified as LoRA but no complete parametrizations were found"
        )
    return groups


def _inferred_lora_rank(
    state: Mapping[str, torch.Tensor],
    groups: list[tuple[str, str, str, str]],
) -> int:
    ranks: set[int] = set()
    for target_key, original_key, a_key, b_key in groups:
        original = state[original_key]
        lora_a = state[a_key]
        lora_b = state[b_key]
        if original.ndim != 2 or lora_a.ndim != 2 or lora_b.ndim != 2:
            raise CheckpointConversionError(
                f"LoRA tensors for {target_key} must all be 2-D"
            )
        rank = int(lora_a.shape[0])
        if rank <= 0:
            raise CheckpointConversionError(
                f"LoRA rank for {target_key} must be positive, got {rank}"
            )
        expected_a = (rank, int(original.shape[1]))
        expected_b = (int(original.shape[0]), rank)
        if tuple(lora_a.shape) != expected_a or tuple(lora_b.shape) != expected_b:
            raise CheckpointConversionError(
                f"LoRA shapes for {target_key} are inconsistent: "
                f"original={tuple(original.shape)} A={tuple(lora_a.shape)} "
                f"B={tuple(lora_b.shape)}"
            )
        ranks.add(rank)
    if len(ranks) != 1:
        raise CheckpointConversionError(
            f"LoRA checkpoint contains multiple ranks: {sorted(ranks)}"
        )
    return ranks.pop()


def _lora_hyperparameters(
    checkpoint: Mapping[str, Any],
    config: Mapping[str, Any],
    state: Mapping[str, torch.Tensor],
    groups: list[tuple[str, str, str, str]],
    *,
    alpha: float | None,
) -> tuple[int, float, list[str]]:
    adapt = config.get("adapt")
    if not isinstance(adapt, Mapping):
        raise CheckpointConversionError(
            "Resolved config has no adapt mapping for LoRA hyperparameters"
        )

    inferred_rank = _inferred_lora_rank(state, groups)
    rank = _equal_int_metadata(
        "LoRA rank",
        [
            ("inferred adapter tensors", inferred_rank),
            ("checkpoint.rank", checkpoint.get("rank")),
            ("checkpoint.lora_rank", checkpoint.get("lora_rank")),
            ("config.adapt.rank", adapt.get("rank")),
        ],
    )

    alpha_candidates = [
        ("--lora-alpha", alpha),
        ("checkpoint.alpha", checkpoint.get("alpha")),
        ("checkpoint.lora_alpha", checkpoint.get("lora_alpha")),
        ("config.adapt.alpha", adapt.get("alpha")),
    ]
    present_alpha: list[tuple[str, float]] = []
    for source, value in alpha_candidates:
        if value is None:
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise CheckpointConversionError(
                f"LoRA alpha from {source} must be numeric, got {value!r}"
            ) from exc
        if not math.isfinite(numeric):
            raise CheckpointConversionError(
                f"LoRA alpha from {source} must be finite, got {numeric}"
            )
        present_alpha.append((source, numeric))
    if not present_alpha:
        raise CheckpointConversionError(
            "LoRA alpha is absent. It cannot be inferred from A/B tensors; supply the "
            "exact training config or --lora-alpha."
        )
    resolved_alpha = present_alpha[0][1]
    if any(value != resolved_alpha for _, value in present_alpha[1:]):
        details = ", ".join(f"{source}={value}" for source, value in present_alpha)
        raise CheckpointConversionError(f"Conflicting LoRA alpha metadata: {details}")

    inferred_targets: set[str] = set()
    for target_key, *_ in groups:
        target = target_key.removeprefix("qbyt.")
        if target not in {
            "audio_projection.weight",
            "audio_key.weight",
            "text_query.weight",
        }:
            raise CheckpointConversionError(
                f"Unsupported LoRA target in checkpoint: {target_key}"
            )
        inferred_targets.add(target)

    for source, configured_targets in (
        ("checkpoint.lora_targets", checkpoint.get("lora_targets")),
        ("config.adapt.lora_targets", adapt.get("lora_targets")),
    ):
        if configured_targets is None:
            continue
        if isinstance(configured_targets, str):
            configured = {configured_targets}
        else:
            try:
                configured = {str(value) for value in configured_targets}
            except TypeError as exc:
                raise CheckpointConversionError(
                    "LoRA targets metadata must be a sequence of target names"
                ) from exc
        if configured != inferred_targets:
            raise CheckpointConversionError(
                f"LoRA targets disagree: {source}={sorted(configured)} "
                f"weights={sorted(inferred_targets)}"
            )

    return rank, resolved_alpha, sorted(inferred_targets)


def merge_lora_checkpoint_state(
    state: Mapping[str, torch.Tensor],
    *,
    alpha: float,
) -> dict[str, torch.Tensor]:
    """Merge all project LoRA parametrizations into an ordinary model state."""
    groups = _lora_groups(state)
    rank = _inferred_lora_rank(state, groups)
    merged = {
        key: value
        for key, value in state.items()
        if ".parametrizations." not in key
    }
    for target_key, original_key, a_key, b_key in groups:
        original = state[original_key]
        delta = state[b_key] @ state[a_key]
        merged[target_key] = original + (float(alpha) / rank) * delta

    residual = [
        key
        for key in merged
        if ".parametrizations." in key or key.endswith(_LORA_PARAMETER_SUFFIXES)
    ]
    if residual:
        raise CheckpointConversionError(
            f"LoRA merge left parametrization keys behind: {residual[:5]}"
        )
    return merged


def _adapter_state(
    state: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    adapter = {
        key[len("qbyt.") :]: value.detach().cpu().clone()
        for key, value in state.items()
        if key.startswith("qbyt.") and key.endswith(_LORA_PARAMETER_SUFFIXES)
    }
    if not adapter:
        raise CheckpointConversionError("No qbyt.* LoRA adapter tensors were found")
    return adapter


def _write_payload(
    payload: Mapping[str, Any],
    output_path: Path,
    *,
    overwrite: bool,
) -> Path:
    output_path = Path(output_path)
    if output_path.suffix != ".pt":
        raise CheckpointConversionError(f"Output must end in .pt: {output_path}")
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(
        f".{output_path.name}.{uuid.uuid4().hex}.tmp"
    )
    try:
        torch.save(dict(payload), temporary)
        os.replace(temporary, output_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return output_path


def _validate_output_paths(
    paths: list[Path],
    *,
    overwrite: bool,
) -> None:
    normalized = [Path(path) for path in paths]
    if len({path.resolve() for path in normalized}) != len(normalized):
        raise CheckpointConversionError("Two requested artifacts resolve to the same path")
    for path in normalized:
        if path.suffix != ".pt":
            raise CheckpointConversionError(f"Output must end in .pt: {path}")
        if path.exists() and not overwrite:
            raise FileExistsError(f"Output already exists: {path}")


def convert_checkpoint(
    source: Path | str,
    output_path: Path | str,
    *,
    fallback_config: Mapping[str, Any] | None = None,
    lora_output: LoraOutput = "both",
    adapter_output_path: Path | str | None = None,
    lora_alpha: float | None = None,
    weights_key: str = "state_dict",
    overwrite: bool = False,
    validate_tokenizer: bool = True,
    validate_encoder: bool = True,
) -> ConversionResult:
    """Convert one Stage II or LoRA Lightning checkpoint.

    ``output_path`` is the full-model path for Stage II and for LoRA
    ``merged``/``both`` conversion.  In adapter-only mode it is the adapter
    path.  ``adapter_output_path`` defaults to ``<stem>.adapter.pt`` in ``both``
    mode.
    """
    source = Path(source)
    output_path = Path(output_path)
    if not source.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {source}")
    if source.suffix != ".ckpt":
        raise CheckpointConversionError(f"Input must end in .ckpt: {source}")
    if lora_output not in {"merged", "adapter", "both"}:
        raise CheckpointConversionError(f"Unsupported LoRA output mode: {lora_output}")

    raw_checkpoint = torch.load(source, map_location="cpu")
    checkpoint = _as_mapping(raw_checkpoint, name=f"checkpoint {source}")
    _validate_weight_selection(checkpoint, weights_key=weights_key)
    state = _state_dict_from_checkpoint(checkpoint, weights_key=weights_key)
    kind = detect_checkpoint_kind(state)
    _validate_checkpoint_kind_metadata(checkpoint, kind)
    config = _resolved_config(checkpoint, fallback_config)
    readout_version = _require_compatible_readout(checkpoint, config)
    alignment_spec = copy.deepcopy(checkpoint[QBYT_ALIGNMENT_SPEC_KEY])

    if kind == "stage2":
        if lora_output == "adapter":
            raise CheckpointConversionError(
                "Adapter-only output was requested for a non-LoRA Stage II checkpoint"
            )
        if adapter_output_path is not None:
            raise CheckpointConversionError(
                "An explicit adapter output was requested for a non-LoRA "
                "Stage II checkpoint"
            )
        if lora_alpha is not None:
            raise CheckpointConversionError(
                "LoRA alpha was supplied for a non-LoRA Stage II checkpoint"
            )
        metadata = _base_model_metadata(
            checkpoint,
            config,
            state,
            validate_tokenizer=validate_tokenizer,
            validate_encoder=validate_encoder,
        )
        payload = {"model_state_dict": state, **metadata}
        _validate_output_paths([output_path], overwrite=overwrite)
        written = _write_payload(payload, output_path, overwrite=overwrite)
        return ConversionResult(source=source, kind=kind, outputs=(written,))

    groups = _lora_groups(state)
    rank, alpha, targets = _lora_hyperparameters(
        checkpoint,
        config,
        state,
        groups,
        alpha=lora_alpha,
    )
    identity = _lora_identity(checkpoint, config)
    try:
        base_state = canonical_stage2_base_state(state)
        base_model_sha256 = fingerprint_stage2_base(base_state)
    except (TypeError, ValueError) as exc:
        raise CheckpointConversionError(str(exc)) from exc
    if lora_output == "adapter":
        _validate_deployable_model_state(
            base_state,
            config,
            vocab_size=_state_vocab_size(base_state),
            validate_encoder=validate_encoder,
        )
    saved_base_model_sha256 = checkpoint.get(STAGE2_BASE_FINGERPRINT_KEY)
    if (
        saved_base_model_sha256 is not None
        and str(saved_base_model_sha256) != base_model_sha256
    ):
        raise CheckpointConversionError(
            f"checkpoint.{STAGE2_BASE_FINGERPRINT_KEY} does not match the "
            "frozen Stage II weights in the checkpoint"
        )
    step = _checkpoint_step(checkpoint)
    outputs: list[Path] = []
    merged_state: dict[str, torch.Tensor] | None = None
    model_metadata: dict[str, Any] | None = None
    if lora_output in {"merged", "both"}:
        merged_state = merge_lora_checkpoint_state(state, alpha=alpha)
        model_metadata = _base_model_metadata(
            checkpoint,
            config,
            merged_state,
            validate_tokenizer=validate_tokenizer,
            validate_encoder=validate_encoder,
        )

    resolved_adapter_path: Path | None = None
    if lora_output == "adapter":
        resolved_adapter_path = output_path
    elif lora_output == "both":
        if adapter_output_path is not None:
            resolved_adapter_path = Path(adapter_output_path)
        else:
            resolved_adapter_path = output_path.with_name(
                f"{output_path.stem}.adapter.pt"
            )
    planned_paths = (
        [output_path, resolved_adapter_path]
        if lora_output == "both"
        else [output_path]
    )
    _validate_output_paths(
        [path for path in planned_paths if path is not None],
        overwrite=overwrite,
    )

    if lora_output in {"merged", "both"}:
        assert merged_state is not None
        assert model_metadata is not None
        merged_payload = {
            "model_state_dict": merged_state,
            **model_metadata,
            **identity,
        }
        outputs.append(
            _write_payload(merged_payload, output_path, overwrite=overwrite)
        )

    if lora_output in {"adapter", "both"}:
        assert resolved_adapter_path is not None
        adapter_payload = {
            "checkpoint_kind": "stage2_lora_adapter",
            "lora_state_dict": _adapter_state(state),
            "config": config,
            "step": step,
            **identity,
            "rank": rank,
            "alpha": alpha,
            "lora_targets": targets,
            STAGE2_BASE_FINGERPRINT_KEY: base_model_sha256,
            QBYT_READOUT_VERSION_KEY: readout_version,
            QBYT_ALIGNMENT_SPEC_KEY: alignment_spec,
        }
        outputs.append(
            _write_payload(
                adapter_payload,
                resolved_adapter_path,
                overwrite=overwrite,
            )
        )

    return ConversionResult(source=source, kind=kind, outputs=tuple(outputs))
