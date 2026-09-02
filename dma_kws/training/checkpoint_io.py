"""Checkpoint I/O helpers shared across training stages."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any, Mapping

import torch

from dma_kws.training.checkpoint_avg import average_lightning_checkpoints

#: Bumped whenever QbyT's deployed score changes what it means.
#:
#: Version 1 (unversioned) indexed the readout with ``text_lengths +
#: speech_lengths - 1`` while the sequence was laid out as ``[text padded to the
#: batch maximum][audio]``. That index is short by the batch's text padding, so a
#: pair scored alone and the same pair scored next to a longer keyword disagreed,
#: and for short keywords it landed in the text padding entirely -- the pooled GRU
#: state had then consumed no audio at all. Version 2 re-packs each sample as
#: ``[valid text][valid audio][padding]`` and reads the final valid audio state
#: with GRU+FC. Version 3 makes the readout mode checkpoint-configured and adds
#: EPS mean pooling over a shared scorer at valid anchor positions. Version 4
#: adds temperature-configured EPS soft-min pooling. Version 5 replaces every
#: historical GRU/EPS mode with a target-only bounded segmental path average.
#: Version 6 replaces that one-sided score with a normalized phone/filler
#: competition, keyword-vs-near-miss segmental log-likelihood ratio, and an
#: aligned weakest-phone veto. Version 7 keeps the v6 keyword/filler segmental
#: graph but scores each phone's frames by its one-vs-rest log-odds instead of
#: against a query-relative filler; checkpoints trained under 6 score
#: differently on the same weights and must be retrained.
QBYT_READOUT_VERSION = 7

QBYT_READOUT_VERSION_KEY = "qbyt_readout_version"
QBYT_ALIGNMENT_SPEC_KEY = "qbyt_alignment_spec"

STAGE2_BASE_FINGERPRINT_KEY = "base_model_sha256"

_LORA_PARAMETER_SUFFIXES = (".lora_A", ".lora_B")
_PARAMETRIZED_ORIGINAL_RE = re.compile(
    r"^(?P<module>.+)\.parametrizations\.(?P<parameter>[^.]+)\.original$"
)


def restore_best_checkpoint_weights(
    model: torch.nn.Module,
    checkpoint_callback: Any,
    *,
    final_step: int,
) -> tuple[int, str]:
    """Restore the callback-selected Lightning weights before a ``.pt`` export.

    Training still completes at ``final_step`` and its final metrics remain useful,
    but the deployable artifact should match the checkpoint selected by the
    configured validation monitor.  Runs without a selected/available checkpoint
    retain the historical final-weight behavior.
    """

    best_path_value = str(getattr(checkpoint_callback, "best_model_path", "") or "")
    if not best_path_value:
        return int(final_step), f"final_weights@step={int(final_step)}"

    best_path = Path(best_path_value)
    if not best_path.is_file():
        return int(final_step), f"final_weights@step={int(final_step)}"

    checkpoint = torch.load(best_path, map_location="cpu")
    state = checkpoint.get("state_dict") if isinstance(checkpoint, Mapping) else None
    if not isinstance(state, Mapping) or not state:
        raise ValueError(
            f"Best checkpoint {best_path} has no non-empty Lightning state_dict"
        )
    model.load_state_dict(dict(state), strict=True)
    selected_step = int(checkpoint.get("global_step", final_step))
    return selected_step, f"best_checkpoint@step={selected_step}:{best_path}"


def canonical_stage2_base_state(
    state: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Return frozen Stage II tensors under their pre-LoRA parameter names."""
    canonical: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        if not isinstance(key, str) or not key.startswith(("encoder.", "adapter.", "qbyt.")):
            continue
        if key.endswith(_LORA_PARAMETER_SUFFIXES):
            continue

        normalized_key = key
        match = _PARAMETRIZED_ORIGINAL_RE.fullmatch(key)
        if match is not None:
            normalized_key = f"{match.group('module')}.{match.group('parameter')}"
        elif ".parametrizations." in key:
            raise ValueError(f"Unsupported parametrized Stage II state key: {key}")

        if normalized_key in canonical:
            raise ValueError(
                f"Duplicate Stage II base key after LoRA normalization: {normalized_key}"
            )
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"Stage II state entry {key!r} is not a tensor")
        canonical[normalized_key] = value

    for prefix in ("encoder.", "qbyt."):
        if not any(key.startswith(prefix) for key in canonical):
            raise ValueError(
                f"Cannot identify an incomplete Stage II base: no {prefix} weights"
            )
    return canonical


def fingerprint_stage2_base(state: Mapping[str, torch.Tensor]) -> str:
    """Hash the frozen Stage II base represented by a plain or LoRA state dict.

    LoRA checkpoints store targeted weights under PyTorch parametrization keys.
    Canonicalize those ``original`` tensors back to their ordinary names and
    exclude A/B tensors so a base loaded directly and the same base embedded in
    a LoRA Lightning checkpoint produce the same identity.
    """
    canonical = canonical_stage2_base_state(state)

    digest = hashlib.sha256()
    for key in sorted(canonical):
        tensor = canonical[key].detach().cpu().contiguous()
        metadata = (
            f"{key}\0{tensor.dtype}\0{tuple(tensor.shape)}\0{tensor.layout}\0"
        ).encode("utf-8")
        raw = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
        digest.update(len(metadata).to_bytes(8, "big"))
        digest.update(metadata)
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
    return digest.hexdigest()


def extract_state_dict(checkpoint: dict[str, Any]) -> dict[str, torch.Tensor]:
    """Extract a model state dict from a Lightning or custom checkpoint."""
    if "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    if "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    if "model" in checkpoint and isinstance(checkpoint["model"], dict):
        return checkpoint["model"]
    return checkpoint


def assert_stream_policy_matches(
    checkpoint: dict[str, Any],
    policy: Any,
    *,
    source: Any,
) -> None:
    """Fail when a checkpoint was produced at a different streaming operating point.

    ``.pt`` payloads written by :func:`export_model_pt` and the LoRA adaptation
    runner embed the full config, so the operating point they were trained at is
    recoverable. Lightning ``.ckpt`` files do not carry it and are skipped.
    """
    import warnings

    from dma_kws.config import resolve_stream_policy

    if not isinstance(checkpoint, dict):
        return
    config = checkpoint.get("config")
    if not isinstance(config, dict) or "stage1" not in config:
        return
    try:
        saved = resolve_stream_policy(config)
    except ValueError as exc:
        # Pre-migration checkpoints carry the multi-value `stage1.chunk_size` list,
        # which means they were trained (and validated) on a random draw per batch.
        # There is no operating point to compare against, but staying silent would
        # hide exactly the mismatch this check exists for.
        warnings.warn(
            f"Cannot verify the streaming operating point of {source}: "
            f"{str(exc).splitlines()[0]} "
            "It predates stage1.stream, so it was trained under a randomized chunk "
            "config and its metrics are not comparable with the current fixed point.",
            UserWarning,
            stacklevel=2,
        )
        return

    if not saved.enabled and not policy.enabled:
        return
    mismatch = (
        saved.enabled != policy.enabled
        or saved.chunk_size != policy.chunk_size
        or saved.left_context_frames != policy.left_context_frames
    )
    if mismatch:
        raise ValueError(
            f"Streaming operating point mismatch for {source}: checkpoint was trained at "
            f"[{saved.describe()}] but the current config resolves to [{policy.describe()}]. "
            "Scores are not comparable across operating points; set "
            "stage1.stream.chunk_size / stage1.stream.left_context_frames to match, "
            "or re-train at the new point."
        )


def _resolve_qbyt_alignment_value(value: Any) -> Any:
    """Resolve a spec object, direct spec mapping, or Stage-II config mapping."""

    from dma_kws.stage2.readout import QbyTAlignmentSpec, resolve_qbyt_alignment

    if isinstance(value, QbyTAlignmentSpec):
        return value
    if value is None:
        return resolve_qbyt_alignment({})
    if not isinstance(value, Mapping):
        raise ValueError("QbyT alignment spec must be a mapping")
    if "stage2" in value:
        stage2 = value.get("stage2")
        if not isinstance(stage2, Mapping):
            raise ValueError("config.stage2 must be a mapping")
        return resolve_qbyt_alignment(stage2)
    if "qbyt_alignment" in value or "qbyt_readout" in value:
        return resolve_qbyt_alignment(value)
    return resolve_qbyt_alignment({"qbyt_alignment": value})


def stamp_qbyt_readout_version(
    payload: dict[str, Any],
    *,
    alignment: Any,
) -> dict[str, Any]:
    """Record the complete QbyT v6 score semantics in ``payload``, in place.

    ``alignment`` is mandatory so a writer cannot silently stamp default
    semantics that differ from the model which produced the weights.
    """

    config = payload.get("config")
    stage2 = config.get("stage2") if isinstance(config, Mapping) else None
    spec = _resolve_qbyt_alignment_value(alignment)

    if isinstance(stage2, Mapping):
        configured = _resolve_qbyt_alignment_value(stage2)
        if not qbyt_readout_specs_equal(configured, spec):
            raise ValueError(
                "Explicit QbyT alignment spec disagrees with config.stage2.qbyt_alignment: "
                f"explicit={_describe_readout_spec(spec)}, "
                f"config={_describe_readout_spec(configured)}"
            )

    payload[QBYT_READOUT_VERSION_KEY] = QBYT_READOUT_VERSION
    payload[QBYT_ALIGNMENT_SPEC_KEY] = spec.as_dict()
    return payload


def qbyt_readout_specs_equal(left: Any, right: Any) -> bool:
    """Whether two resolved alignment specs produce the same deployed score."""

    return left == right


def _describe_readout_spec(spec: Any) -> str:
    if spec is None:
        return "unknown"
    values = spec.as_dict()
    fields = ", ".join(f"{key}={value!r}" for key, value in values.items())
    return f"{spec.topology}({fields})"


def checkpoint_qbyt_readout_spec(
    checkpoint: Mapping[str, Any],
    *,
    saved_version: Any | None = None,
) -> Any:
    """Resolve the complete segmental-CRF score carried by a v6 checkpoint.

    Earlier versions are intentionally not inferred or upgraded. Their
    weights were optimized for a different score, so tensor shapes cannot
    establish semantic compatibility.
    """

    from dma_kws.stage2.readout import QbyTAlignmentSpec

    saved = (
        checkpoint.get(QBYT_READOUT_VERSION_KEY)
        if saved_version is None
        else saved_version
    )
    if saved != QBYT_READOUT_VERSION:
        raise ValueError(f"unsupported QbyT readout version {saved!r}")

    raw = checkpoint.get(QBYT_ALIGNMENT_SPEC_KEY)
    if not isinstance(raw, Mapping):
        raise ValueError(
            f"QbyT v6 checkpoint must explicitly record {QBYT_ALIGNMENT_SPEC_KEY}"
        )
    required = set(QbyTAlignmentSpec.__dataclass_fields__)
    missing = sorted(required - set(raw))
    unknown = sorted(set(raw) - required)
    if missing or unknown:
        details = []
        if missing:
            details.append(f"missing={missing}")
        if unknown:
            details.append(f"unknown={unknown}")
        raise ValueError(
            f"invalid {QBYT_ALIGNMENT_SPEC_KEY}: " + ", ".join(details)
        )
    spec = _resolve_qbyt_alignment_value(raw)

    config = checkpoint.get("config")
    if isinstance(config, Mapping):
        stage2 = config.get("stage2")
        if isinstance(stage2, Mapping):
            configured = _resolve_qbyt_alignment_value(stage2)
            if not qbyt_readout_specs_equal(spec, configured):
                raise ValueError(
                    f"{QBYT_ALIGNMENT_SPEC_KEY} disagrees with "
                    "config.stage2.qbyt_alignment"
                )
    return spec


def _carries_qbyt_weights(checkpoint: Any) -> bool:
    """Whether a payload holds QbyT weights whose readout version matters.

    Deliberately narrow: Stage I exports, icefall encoder checkpoints and Step A
    adapter exports all flow through the same loaders, and none of them encodes a
    readout convention.
    """
    if not isinstance(checkpoint, dict):
        return False
    if isinstance(checkpoint.get("lora_state_dict"), dict):
        return True
    state = extract_state_dict(checkpoint)
    if not isinstance(state, dict):
        return False
    return any(isinstance(key, str) and key.startswith("qbyt.") for key in state)


def assert_qbyt_readout_version(
    checkpoint: Any,
    *,
    source: Any,
    expected_alignment: Any | None = None,
) -> None:
    """Fail unless QbyT weights carry the exact v6 segmental score semantics.

    Encoder-only checkpoints return before version validation and remain valid
    warm starts. Pre-v6 QbyT weights are never loadable.
    """

    if not _carries_qbyt_weights(checkpoint):
        return

    saved = checkpoint.get(QBYT_READOUT_VERSION_KEY)
    if expected_alignment is not None:
        expected = _resolve_qbyt_alignment_value(expected_alignment)
    else:
        expected = None

    saved_spec = None
    saved_error = None
    try:
        saved_spec = checkpoint_qbyt_readout_spec(
            checkpoint,
            saved_version=saved,
        )
    except ValueError as exc:
        saved_error = str(exc)

    compatible = saved == QBYT_READOUT_VERSION and saved_spec is not None and (
        expected is None or qbyt_readout_specs_equal(saved_spec, expected)
    )
    if compatible:
        return

    described = "unversioned (pre-fix)" if saved is None else f"version {saved!r}"
    mode_detail = (
        f" ({_describe_readout_spec(saved_spec)})"
        if saved_spec is not None
        else (f" ({saved_error})" if saved_error else "")
    )
    expected_detail = (
        f"; current config expects {_describe_readout_spec(expected)}"
        if expected
        else ""
    )
    raise SystemExit(
        f"{source} carries QbyT weights at readout {described}{mode_detail}, but this build "
        f"uses version {QBYT_READOUT_VERSION}{expected_detail}. Readout semantics differ, "
        "so loading these weights would silently change the meaning of the deployed score."
        f" Re-train Stage II at readout version {QBYT_READOUT_VERSION}."
    )


def extract_icefall_encoder_state(
    checkpoint_path: Path | str,
) -> dict[str, dict[str, torch.Tensor]]:
    """Extract encoder weights from an icefall Zipformer KWS checkpoint.
    
    Icefall checkpoints are structured as:
    {
        "model": {
            "encoder_embed.0.weight": ...,
            "encoder_embed.0.bias": ...,
            "encoder.0.self_attn.weight": ...,
            ...
        },
        "optimizer": ...,
        "scheduler": ...,
    }
    
    This function extracts the encoder_embed and encoder submodule states.
    
    Args:
        checkpoint_path: Path to icefall .pt checkpoint
    
    Returns:
        Dict with keys "encoder_embed" and "encoder", each containing
        state dict for those submodules (with prefixes stripped).
    
    Raises:
        FileNotFoundError: If checkpoint_path doesn't exist
        ValueError: If checkpoint doesn't contain expected structure
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Icefall checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    # Extract model state dict
    if "model" in checkpoint and isinstance(checkpoint["model"], dict):
        model_state = checkpoint["model"]
    else:
        raise ValueError(
            f"Icefall checkpoint format not recognized. "
            f"Expected 'model' key, got keys: {list(checkpoint.keys())}"
        )
    
    # Split encoder_embed and encoder states
    encoder_embed_state: dict[str, torch.Tensor] = {}
    encoder_state: dict[str, torch.Tensor] = {}
    
    for key, value in model_state.items():
        if key.startswith("encoder_embed."):
            # Remove "encoder_embed." prefix
            new_key = key[len("encoder_embed."):]
            encoder_embed_state[new_key] = value
        elif key.startswith("encoder."):
            # Remove "encoder." prefix
            new_key = key[len("encoder."):]
            encoder_state[new_key] = value
    
    if not encoder_embed_state and not encoder_state:
        raise ValueError(
            f"No encoder_embed or encoder weights found in checkpoint. "
            f"Available keys: {list(model_state.keys())[:10]}..."
        )
    
    return {
        "encoder_embed": encoder_embed_state,
        "encoder": encoder_state,
    }


def export_model_pt(
    model: torch.nn.Module,
    output_path: Path,
    *,
    config: dict[str, Any],
    dict_path: Path,
    vocab_size: int,
    step: int,
    blank_id: int | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Save model weights in the Stage I/II ``.pt`` checkpoint format."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "config": config,
        "step": step,
        "tokenizer_dict_path": str(dict_path),
        "vocab_size": vocab_size,
    }
    if blank_id is not None:
        payload["blank_id"] = blank_id
    if extra:
        payload.update(extra)
    torch.save(payload, output_path)
    return output_path


def select_and_average_checkpoints(
    checkpoint_dir: Path,
    *,
    last_k: int,
    pattern: str = "*.ckpt",
    output_name: str = "avg_10.ckpt",
) -> Path | None:
    """Average the last ``last_k`` checkpoints matching ``pattern``."""
    candidates = sorted(checkpoint_dir.glob(pattern))
    if not candidates:
        from dma_kws.training.ddp import rank_zero_print

        rank_zero_print(
            f"No checkpoints matched {pattern!r} in {checkpoint_dir}; skipping average."
        )
        return None
    selected = candidates[-last_k:]
    output_path = checkpoint_dir / output_name
    average_lightning_checkpoints(selected, output_path)
    from dma_kws.training.ddp import rank_zero_print

    rank_zero_print(f"Averaged {len(selected)} checkpoints -> {output_path}")
    return output_path
