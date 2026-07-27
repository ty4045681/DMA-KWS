"""Trainable trunks that sit between a frozen encoder and the phoneme CTC head.

The trunk is the only module the CTC loss and QbyT share, so its capacity and
its receptive field decide how much of the "BPE/transducer acoustic space ->
phoneme space" mapping can actually be learned.

``linear`` and ``mlp`` are pointwise: they can re-mix channels at a frame but
cannot move evidence between frames. They exist as controls for how linearly
separable the frozen representation already is. ``conv`` and ``conformer`` have
a receptive field, which is what an RNN-T-trained encoder needs, because
transducer emission is systematically delayed relative to the acoustics.
"""

from __future__ import annotations

from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

TRUNK_TYPES = ("linear", "mlp", "conv", "conformer")


def _lengths_from_mask(mask: torch.Tensor | None, batch: int, frames: int, device) -> torch.Tensor:
    if mask is None:
        return torch.full((batch,), frames, dtype=torch.long, device=device)
    return mask.squeeze(1).sum(dim=1).to(dtype=torch.long)


def _apply_mask(x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    """Zero padded frames so they cannot leak into convolutions or statistics."""
    if mask is None:
        return x
    return x * mask.squeeze(1).unsqueeze(-1).to(dtype=x.dtype)


class LinearTrunk(nn.Module):
    """Single projection. The weakest control: pointwise, no nonlinearity."""

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.proj = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        return _apply_mask(self.proj(x), mask)


class MlpTrunk(nn.Module):
    """Pointwise MLP. Adds capacity but still no temporal context."""

    def __init__(self, input_dim: int, output_dim: int, *, dropout: float) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        return _apply_mask(self.net(x), mask)


class _ConvBlock(nn.Module):
    """Pre-norm depthwise-separable conv block with a residual connection."""

    def __init__(self, dim: int, *, kernel_size: int, dropout: float, causal: bool) -> None:
        super().__init__()
        if kernel_size < 1:
            raise ValueError(f"trunk.kernel_size must be >= 1, got {kernel_size}")
        self.causal = causal
        self.kernel_size = kernel_size
        self.norm = nn.LayerNorm(dim)
        # Padding is applied explicitly in forward so the causal case can pad on
        # the left only; nn.Conv1d's own padding is always symmetric.
        self.depthwise = nn.Conv1d(dim, dim, kernel_size, groups=dim, padding=0)
        self.pointwise = nn.Conv1d(dim, dim, 1)
        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def _pad(self, x: torch.Tensor) -> torch.Tensor:
        total = self.kernel_size - 1
        if self.causal:
            return F.pad(x, (total, 0))
        left = total // 2
        return F.pad(x, (left, total - left))

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        # Zero the padded frames again: LayerNorm re-introduces a bias there and
        # the convolution would smear it into valid frames.
        x = _apply_mask(x, mask).transpose(1, 2)
        x = self.pointwise(self.depthwise(self._pad(x)))
        x = x.transpose(1, 2)
        return residual + self.dropout(self.activation(x))


class ConvTrunk(nn.Module):
    """Depthwise-separable conv stack with a receptive field over time."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        num_layers: int,
        kernel_size: int,
        dropout: float,
        causal: bool,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError(f"trunk.num_layers must be >= 1, got {num_layers}")
        self.output_dim = output_dim
        self.causal = causal
        self.proj = nn.Linear(input_dim, output_dim)
        self.blocks = nn.ModuleList(
            _ConvBlock(output_dim, kernel_size=kernel_size, dropout=dropout, causal=causal)
            for _ in range(num_layers)
        )
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        x = _apply_mask(self.proj(x), mask)
        for block in self.blocks:
            x = block(x, mask)
        return _apply_mask(self.norm(x), mask)


class ConformerTrunk(nn.Module):
    """Wenet Conformer blocks. Expressiveness upper bound for the controls."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        num_layers: int,
        kernel_size: int,
        dropout: float,
        causal: bool,
        chunk_size: int,
        left_context_chunks: int,
        attention_heads: int,
        linear_units: int,
    ) -> None:
        super().__init__()
        if causal and chunk_size <= 0:
            raise ValueError(
                "trunk.chunk_size must be > 0 for a conformer trunk on a causal encoder: "
                "unrestricted attention would give the trunk lookahead the deployed "
                "streaming pipeline cannot provide. Units are trunk-input frames."
            )
        from dma_kws.pathing import ensure_qbyt_on_path

        ensure_qbyt_on_path()
        from models.encoder import ConformerEncoder

        self.output_dim = output_dim
        self.left_context_chunks = left_context_chunks
        self.encoder = ConformerEncoder(
            input_size=input_dim,
            output_size=output_dim,
            attention_heads=attention_heads,
            linear_units=linear_units,
            num_blocks=num_layers,
            dropout_rate=dropout,
            positional_dropout_rate=dropout,
            attention_dropout_rate=0.0,
            input_layer="linear",
            pos_enc_layer_type="rel_pos",
            selfattention_layer_type="rel_selfattn",
            use_cnn_module=True,
            cnn_module_kernel=kernel_size,
            causal=causal,
            cnn_module_norm="layer_norm",
            static_chunk_size=chunk_size,
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        lengths = _lengths_from_mask(mask, x.size(0), x.size(1), x.device)
        out, out_mask = self.encoder(
            x,
            lengths,
            decoding_chunk_size=0,
            num_decoding_left_chunks=self.left_context_chunks,
        )
        return _apply_mask(out, out_mask)


def build_trunk(
    trunk_cfg: Mapping[str, Any],
    *,
    input_dim: int,
    causal: bool,
) -> nn.Module:
    """Build the trunk selected by ``trunk_cfg["type"]``.

    Every trunk exposes ``forward(x, mask) -> (N, T, output_dim)`` and an
    ``output_dim`` attribute, so callers never need to know which one is active.
    """
    trunk_type = str(trunk_cfg.get("type", "conv")).lower()
    output_dim = int(trunk_cfg.get("output_dim", 192))
    dropout = float(trunk_cfg.get("dropout", 0.1))
    num_layers = int(trunk_cfg.get("num_layers", 2))
    kernel_size = int(trunk_cfg.get("kernel_size", 7))

    if trunk_type == "linear":
        return LinearTrunk(input_dim, output_dim)
    if trunk_type == "mlp":
        return MlpTrunk(input_dim, output_dim, dropout=dropout)
    if trunk_type == "conv":
        return ConvTrunk(
            input_dim,
            output_dim,
            num_layers=num_layers,
            kernel_size=kernel_size,
            dropout=dropout,
            causal=causal,
        )
    if trunk_type == "conformer":
        return ConformerTrunk(
            input_dim,
            output_dim,
            num_layers=num_layers,
            kernel_size=kernel_size,
            dropout=dropout,
            causal=causal,
            chunk_size=int(trunk_cfg.get("chunk_size", 0)),
            left_context_chunks=int(trunk_cfg.get("left_context_chunks", -1)),
            attention_heads=int(trunk_cfg.get("attention_heads", 4)),
            linear_units=int(trunk_cfg.get("linear_units", 512)),
        )
    raise ValueError(f"Unknown trunk.type {trunk_type!r}, expected one of {TRUNK_TYPES}")
