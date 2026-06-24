"""Checkpoint averaging utilities aligned with ``qbyt/test.py``."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import torch


def _state_dict_key(checkpoint: dict) -> str:
    if "state_dict" in checkpoint:
        return "state_dict"
    if "model_state_dict" in checkpoint:
        return "model_state_dict"
    raise ValueError("Checkpoint must contain 'state_dict' or 'model_state_dict'")


def _extract_state_dict(checkpoint: dict) -> dict[str, torch.Tensor]:
    return checkpoint[_state_dict_key(checkpoint)]


def average_lightning_checkpoints(paths: list[Path], output_path: Path) -> Path:
    """Average weights from Lightning or legacy ``.pt`` checkpoints."""
    if not paths:
        raise ValueError("At least one checkpoint path is required")

    template = torch.load(paths[0], map_location="cpu")
    state_key = _state_dict_key(template)

    state_dicts = [_extract_state_dict(torch.load(path, map_location="cpu")) for path in paths]

    avg_state_dict = OrderedDict()
    for key in state_dicts[0].keys():
        avg_state_dict[key] = sum(state[key].float() for state in state_dicts) / len(state_dicts)

    template[state_key] = avg_state_dict
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(template, output_path)
    return output_path
