"""Wenet-compatible global CMVN loading and application."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


def load_json_cmvn_stats(cmvn_file: str | Path) -> tuple[torch.Tensor, torch.Tensor]:
    """Load Wenet JSON global CMVN stats and return ``(mean, istd)`` tensors."""
    path = Path(cmvn_file)
    with path.open("r", encoding="utf-8") as handle:
        obj: dict[str, Any] = json.load(handle)

    mean_stat = torch.tensor(obj["mean_stat"], dtype=torch.float32)
    var_stat = torch.tensor(obj["var_stat"], dtype=torch.float32)
    frame_num = float(obj["frame_num"])
    if frame_num <= 0:
        raise ValueError(f"Invalid frame_num in CMVN file {path}: {frame_num}")

    mean = mean_stat / frame_num
    var = var_stat / frame_num - mean * mean
    istd = 1.0 / torch.sqrt(torch.clamp(var, min=1.0e-20))
    return mean, istd


class GlobalCMVN(nn.Module):
    """Apply global mean/variance normalization to fbank features."""

    def __init__(self, mean: torch.Tensor, istd: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("mean", mean)
        self.register_buffer("istd", istd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) * self.istd


def build_global_cmvn(stage1_cfg: dict[str, Any]) -> GlobalCMVN | None:
    """Build a GlobalCMVN module from ``stage1`` config, or return None if disabled."""
    if stage1_cfg.get("cmvn") != "global_cmvn":
        return None

    cmvn_conf = stage1_cfg.get("cmvn_conf") or {}
    cmvn_file = cmvn_conf.get("cmvn_file")
    if not cmvn_file:
        raise ValueError("stage1.cmvn_conf.cmvn_file is required when stage1.cmvn is global_cmvn")

    is_json_cmvn = cmvn_conf.get("is_json_cmvn", True)
    if not is_json_cmvn:
        raise ValueError("Only JSON global CMVN is supported (set cmvn_conf.is_json_cmvn: true)")

    mean, istd = load_json_cmvn_stats(cmvn_file)
    return GlobalCMVN(mean, istd)
