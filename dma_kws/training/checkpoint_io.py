"""Checkpoint I/O helpers shared across training stages."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from dma_kws.training.checkpoint_avg import average_lightning_checkpoints

#: Bumped whenever QbyT's pooled readout changes what it means.
#:
#: Version 1 (unversioned) indexed the readout with ``text_lengths +
#: speech_lengths - 1`` while the sequence was laid out as ``[text padded to the
#: batch maximum][audio]``. That index is short by the batch's text padding, so a
#: pair scored alone and the same pair scored next to a longer keyword disagreed,
#: and for short keywords it landed in the text padding entirely -- the pooled GRU
#: state had then consumed no audio at all. Version 2 re-packs each sample as
#: ``[valid text][valid audio][padding]`` and reads the last valid frame.
QBYT_READOUT_VERSION = 2

QBYT_READOUT_VERSION_KEY = "qbyt_readout_version"


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


def stamp_qbyt_readout_version(payload: dict[str, Any]) -> dict[str, Any]:
    """Record the QbyT readout version in a checkpoint payload, in place."""
    payload[QBYT_READOUT_VERSION_KEY] = QBYT_READOUT_VERSION
    return payload


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
    allow_legacy: bool = False,
) -> None:
    """Fail when a checkpoint's QbyT weights predate the current pooled readout.

    Parameter shapes did not change across the readout fix, so a stale checkpoint
    loads cleanly and then scores from a frame it was never trained to read. That
    is the exact failure mode the fix removes, so it must not degrade to a
    warning.
    """
    import warnings

    if not _carries_qbyt_weights(checkpoint):
        return

    saved = checkpoint.get(QBYT_READOUT_VERSION_KEY)
    if saved == QBYT_READOUT_VERSION:
        return

    described = "unversioned (pre-fix)" if saved is None else f"version {saved!r}"
    if allow_legacy:
        warnings.warn(
            f"{source} carries QbyT weights at readout {described}, but this build reads "
            f"version {QBYT_READOUT_VERSION}. Proceeding because "
            "stage2.allow_legacy_qbyt_readout is set: only the encoder/adapter weights are "
            "meaningful, and any score or metric produced from this checkpoint is not "
            "comparable with the ones it was trained against.",
            UserWarning,
            stacklevel=2,
        )
        return

    raise SystemExit(
        f"{source} carries QbyT weights at readout {described} but this build reads version "
        f"{QBYT_READOUT_VERSION}. The pooled readout used to be indexed with "
        "text_lengths + speech_lengths - 1 against a batch-padded layout, so these weights "
        "were trained to read a different frame than the one they will now be scored from. "
        "The parameter shapes still match, which is why this cannot be left to a warning. "
        "Re-train Stage II, or set stage2.allow_legacy_qbyt_readout=true to reuse only the "
        "encoder/adapter weights as a warm start."
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
        print(f"No checkpoints matched {pattern!r} in {checkpoint_dir}; skipping average.")
        return None
    selected = candidates[-last_k:]
    output_path = checkpoint_dir / output_name
    average_lightning_checkpoints(selected, output_path)
    print(f"Averaged {len(selected)} checkpoints -> {output_path}")
    return output_path
