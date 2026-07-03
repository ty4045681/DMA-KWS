"""LoRA parametrization for QbyT phoneme matcher attention layers."""

from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn
from torch.nn.utils import parametrize


class LoRAParametrization(nn.Module):
    """Low-rank delta on a weight matrix: W + (alpha/rank) * (B @ A)."""

    def __init__(
        self,
        features_in: int,
        features_out: int,
        *,
        rank: int,
        alpha: float,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError(f"LoRA rank must be positive, got {rank}")
        self.rank = rank
        self.scaling = alpha / rank
        self.lora_A = nn.Parameter(torch.empty(rank, features_in))
        self.lora_B = nn.Parameter(torch.zeros(features_out, rank))
        nn.init.normal_(self.lora_A, mean=0.0, std=0.02)
        nn.init.zeros_(self.lora_B)

    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        delta = self.lora_B @ self.lora_A
        return weight + self.scaling * delta


_DEFAULT_LORA_TARGETS = ("in_proj_weight", "out_proj.weight")


def _normalize_lora_targets(targets: Iterable[str] | None) -> set[str]:
    normalized: set[str] = set()
    for target in targets or _DEFAULT_LORA_TARGETS:
        if target == "out_proj":
            normalized.add("out_proj.weight")
        else:
            normalized.add(target)
    return normalized


def _iter_phone_matchor_attn_layers(qbyt: nn.Module) -> Iterable[nn.Module]:
    phone_matchor = qbyt.phone_matchor
    for layer in phone_matchor.layers:
        yield layer.self_attn


def inject_qbyt_lora(
    qbyt: nn.Module,
    *,
    rank: int,
    alpha: float,
    targets: Iterable[str] | None = None,
) -> list[str]:
    """Freeze QbyT base weights and inject LoRA on matcher self-attention matrices."""
    target_set = _normalize_lora_targets(targets)
    qbyt.requires_grad_(False)
    injected: list[str] = []

    for layer_idx, attn in enumerate(_iter_phone_matchor_attn_layers(qbyt)):
        embed_dim = int(attn.embed_dim)
        prefix = f"phone_matchor.layers.{layer_idx}.self_attn"

        if "in_proj_weight" in target_set:
            parametrize.register_parametrization(
                attn,
                "in_proj_weight",
                LoRAParametrization(
                    embed_dim,
                    3 * embed_dim,
                    rank=rank,
                    alpha=alpha,
                ),
            )
            injected.append(f"{prefix}.in_proj_weight")

        if "out_proj.weight" in target_set:
            parametrize.register_parametrization(
                attn.out_proj,
                "weight",
                LoRAParametrization(
                    embed_dim,
                    embed_dim,
                    rank=rank,
                    alpha=alpha,
                ),
            )
            injected.append(f"{prefix}.out_proj.weight")

    for name, param in qbyt.named_parameters():
        param.requires_grad = ".parametrizations." in name and ("lora_A" in name or "lora_B" in name)

    return injected


def _iter_parametrized_modules(model: nn.Module) -> Iterable[tuple[nn.Module, str]]:
    for module in model.modules():
        if not hasattr(module, "parametrizations"):
            continue
        for param_name in list(module.parametrizations.keys()):
            yield module, param_name


def merge_lora(model: nn.Module) -> None:
    """Merge LoRA deltas into base weights in-place."""
    for module, param_name in _iter_parametrized_modules(model):
        parametrize.remove_parametrizations(module, param_name, leave_parametrized=True)


def lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Export only LoRA adapter parameters (A/B tensors)."""
    state: dict[str, torch.Tensor] = {}
    for name, param in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            state[name] = param.detach().cpu().clone()
    return state


def load_lora_state_dict(model: nn.Module, state: dict[str, torch.Tensor], *, strict: bool = True) -> None:
    """Load LoRA adapter weights into a model that already has LoRA injected."""
    current = lora_state_dict(model)
    if strict:
        missing = set(current) - set(state)
        unexpected = set(state) - set(current)
        if missing or unexpected:
            raise KeyError(f"LoRA state mismatch: missing={sorted(missing)} unexpected={sorted(unexpected)}")

    model_state = model.state_dict()
    for key, value in state.items():
        if key not in model_state:
            if strict:
                raise KeyError(f"Missing LoRA parameter in model: {key}")
            continue
        model_state[key] = value
    model.load_state_dict(model_state, strict=False)


def count_lora_params(model: nn.Module) -> dict[str, int]:
    """Count trainable LoRA vs total model parameters."""
    trainable = 0
    lora = 0
    total = 0
    for name, param in model.named_parameters():
        n = param.numel()
        total += n
        if param.requires_grad:
            trainable += n
            if "lora_A" in name or "lora_B" in name:
                lora += n
    return {
        "lora_trainable": lora,
        "trainable": trainable,
        "total": total,
    }


def print_lora_param_counts(model: nn.Module, *, prefix: str = "") -> dict[str, int]:
    """Print and return LoRA parameter statistics."""
    counts = count_lora_params(model)
    label = f"{prefix} " if prefix else ""
    print(
        f"{label}LoRA params: trainable={counts['trainable']:,} "
        f"(lora={counts['lora_trainable']:,}) / total={counts['total']:,}"
    )
    return counts
