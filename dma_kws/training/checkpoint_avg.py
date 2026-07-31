"""Checkpoint averaging utilities aligned with ``qbyt/test.py``."""

from __future__ import annotations

import math
from collections import OrderedDict
from pathlib import Path

import torch

from dma_kws.phonemes import normalize_english_text
from dma_kws.training.lora import normalize_lora_targets

_LORA_PARAMETER_SUFFIXES = (".lora_A", ".lora_B")


def _state_dict_key(checkpoint: dict) -> str:
    if "state_dict" in checkpoint:
        return "state_dict"
    if "model_state_dict" in checkpoint:
        return "model_state_dict"
    raise ValueError("Checkpoint must contain 'state_dict' or 'model_state_dict'")


def _extract_state_dict(checkpoint: dict) -> dict[str, torch.Tensor]:
    return checkpoint[_state_dict_key(checkpoint)]


def _is_lora_state(state: dict[str, torch.Tensor]) -> bool:
    return any(
        ".parametrizations." in key or key.endswith(_LORA_PARAMETER_SUFFIXES)
        for key in state
    )


def _lora_metadata(checkpoint: dict, key: str):
    values = []
    if checkpoint.get(key) is not None:
        values.append(checkpoint[key])
    config = checkpoint.get("config")
    adapt = config.get("adapt") if isinstance(config, dict) else None
    if isinstance(adapt, dict) and adapt.get(key) is not None:
        values.append(adapt[key])
    if not values:
        raise ValueError(f"LoRA checkpoint does not provide {key}")

    def normalize(value):
        if key == "keyword":
            return normalize_english_text(str(value))
        if key == "lora_targets":
            source = [value] if isinstance(value, str) else value
            return frozenset(normalize_lora_targets(source))
        if key == "alpha":
            parsed = float(value)
            if not math.isfinite(parsed) or parsed <= 0.0:
                raise ValueError("LoRA alpha must be finite and positive")
            return parsed
        if key == "rank":
            if isinstance(value, bool):
                raise ValueError("LoRA rank cannot be boolean")
            parsed = int(value)
            if isinstance(value, float) and not value.is_integer():
                raise ValueError("LoRA rank must be an integer")
            if isinstance(value, str) and str(parsed) != value.strip().lstrip("+"):
                raise ValueError("LoRA rank must be an integer")
            if parsed <= 0:
                raise ValueError("LoRA rank must be positive")
            return parsed
        return str(value)

    normalized = [normalize(value) for value in values]
    if any(
        not _lora_metadata_equal(key, value, normalized[0])
        for value in normalized[1:]
    ):
        raise ValueError(f"Conflicting LoRA {key} metadata within one checkpoint")
    return normalized[0]


def _lora_metadata_equal(key: str, left, right) -> bool:
    if key == "alpha":
        return math.isclose(left, right, rel_tol=1e-12, abs_tol=0.0)
    return left == right


def average_lightning_checkpoints(paths: list[Path], output_path: Path) -> Path:
    """Average weights from Lightning or legacy ``.pt`` checkpoints."""
    if not paths:
        raise ValueError("At least one checkpoint path is required")

    template = torch.load(paths[0], map_location="cpu")
    state_key = _state_dict_key(template)

    checkpoints = [torch.load(path, map_location="cpu") for path in paths]
    state_dicts = [_extract_state_dict(checkpoint) for checkpoint in checkpoints]
    reference_keys = set(state_dicts[0])
    for path, state in zip(paths[1:], state_dicts[1:]):
        if set(state) != reference_keys:
            raise ValueError(f"Checkpoint state keys do not match: {path}")

    avg_state_dict = OrderedDict()
    if _is_lora_state(state_dicts[0]):
        if not all(_is_lora_state(state) for state in state_dicts):
            raise ValueError("Cannot average LoRA and non-LoRA checkpoints together")
        if not any(
            key.endswith(_LORA_PARAMETER_SUFFIXES)
            for key in state_dicts[0]
        ):
            raise ValueError("LoRA checkpoint contains no adapter A/B tensors")

        from dma_kws.training.checkpoint_io import (
            QBYT_READOUT_VERSION,
            QBYT_READOUT_VERSION_KEY,
            STAGE2_BASE_FINGERPRINT_KEY,
            fingerprint_stage2_base,
        )
        from dma_kws.config import resolve_stream_policy

        base_fingerprints = [
            fingerprint_stage2_base(state)
            for state in state_dicts
        ]
        if len(set(base_fingerprints)) != 1:
            raise ValueError(
                "LoRA checkpoints do not share the same frozen Stage II base"
            )
        for checkpoint, actual in zip(checkpoints, base_fingerprints):
            saved = checkpoint.get(STAGE2_BASE_FINGERPRINT_KEY)
            if saved is not None and str(saved) != actual:
                raise ValueError(
                    f"LoRA checkpoint {STAGE2_BASE_FINGERPRINT_KEY} does not "
                    "match its frozen weights"
                )

        for key in ("keyword", "phase", "rank", "alpha", "lora_targets"):
            values = [_lora_metadata(checkpoint, key) for checkpoint in checkpoints]
            if any(
                not _lora_metadata_equal(key, value, values[0])
                for value in values[1:]
            ):
                raise ValueError(f"Cannot average checkpoints with different LoRA {key}")

        readout_versions = [
            checkpoint.get(QBYT_READOUT_VERSION_KEY)
            for checkpoint in checkpoints
        ]
        if any(version != QBYT_READOUT_VERSION for version in readout_versions):
            raise ValueError(
                "All LoRA checkpoints must carry the current QbyT readout version "
                f"{QBYT_READOUT_VERSION}; got {readout_versions}"
            )
        try:
            stream_policies = [
                resolve_stream_policy(checkpoint["config"])
                for checkpoint in checkpoints
            ]
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "Every LoRA checkpoint must embed a valid streaming config"
            ) from exc
        if any(policy != stream_policies[0] for policy in stream_policies[1:]):
            raise ValueError(
                "Cannot average LoRA checkpoints trained at different streaming "
                "operating points"
            )

        for key, first in state_dicts[0].items():
            if key.endswith(_LORA_PARAMETER_SUFFIXES):
                averaged = sum(state[key].float() for state in state_dicts) / len(
                    state_dicts
                )
                avg_state_dict[key] = averaged.to(dtype=first.dtype)
            else:
                if any(not torch.equal(first, state[key]) for state in state_dicts[1:]):
                    raise ValueError(
                        f"Frozen LoRA base tensor changed across checkpoints: {key}"
                    )
                avg_state_dict[key] = first.clone()
        template[STAGE2_BASE_FINGERPRINT_KEY] = base_fingerprints[0]
    else:
        for key in state_dicts[0].keys():
            avg_state_dict[key] = (
                sum(state[key].float() for state in state_dicts) / len(state_dicts)
            )

    template[state_key] = avg_state_dict
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(template, output_path)
    return output_path
