"""Icefall Zipformer2 encoder adapter for Stage II training compatibility.

Wraps icefall's Zipformer2 encoder + Conv2dSubsampling to expose a Wenet-compatible
interface: forward(feat, feat_lengths) → (encoder_out, encoder_mask).

The adapter ensures shape consistency:
- Wenet: (N, T, C) → encoder → (N, T', C_out) + mask (N, 1, T')
- Icefall: (N, T, C) → embed → (N, T_e, D) → encoder → (N, T', max_dim)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from dma_kws.pathing import ensure_icefall_on_path


def _load_icefall_modules() -> tuple[type, type]:
    """Lazily import icefall Zipformer2 and Conv2dSubsampling.
    
    Returns:
        (Conv2dSubsampling, Zipformer2) classes
    """
    ensure_icefall_on_path()
    try:
        from subsampling import Conv2dSubsampling
        from zipformer import Zipformer2
    except ImportError as exc:
        raise SystemExit(
            "Failed to import icefall Zipformer2/Conv2dSubsampling. "
            "Ensure ICEFALL_ROOT env var is set to icefall repo root "
            "and ICEFALL_ROOT/egs/gigaspeech/KWS/zipformer is in sys.path."
        ) from exc
    return Conv2dSubsampling, Zipformer2


class IcefallZipformerEncoder(nn.Module):
    """Wenet-compatible wrapper for icefall Zipformer2 encoder.
    
    Exposes forward(feat, feat_lengths) → (encoder_out, encoder_mask)
    to match Wenet ConformerEncoder interface for seamless swapping in Stage II.
    """

    def __init__(
        self,
        encoder_embed: nn.Module,
        encoder: nn.Module,
        output_dim: int | None = None,
    ) -> None:
        """Initialize wrapper.
        
        Args:
            encoder_embed: Conv2dSubsampling module (input projection + downsampling)
            encoder: Zipformer2 module (core encoder)
            output_dim: Output dimension (inferred from encoder if None)
        """
        super().__init__()
        self.encoder_embed = encoder_embed
        self.encoder = encoder
        
        # Infer output dim from Zipformer2 (max of encoder_dim across stacks)
        if output_dim is None:
            # Zipformer2 output is max of its encoder_dim list
            if hasattr(encoder, "encoder_dim"):
                if isinstance(encoder.encoder_dim, (list, tuple)):
                    output_dim = max(encoder.encoder_dim)
                else:
                    output_dim = encoder.encoder_dim
            else:
                output_dim = 128  # Default fallback
        
        self.output_dim = output_dim

    def forward(
        self,
        feat: torch.Tensor,
        feat_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract encoder embeddings and compute output mask.
        
        Args:
            feat: Input fbank features, shape (N, T, input_dim)
            feat_lengths: Length of each sequence, shape (N,)
        
        Returns:
            encoder_out: Encoded features, shape (N, T', output_dim)
            encoder_mask: Padding mask, shape (N, 1, T')
        """
        # Stage 1: Subsampling + embedding projection
        # Conv2dSubsampling expects (N, T, C), returns (N, T_subsample, D)
        x, x_lens = self.encoder_embed(feat)
        
        # Stage 2: Zipformer2 encoder
        # Returns (N, T', max_encoder_dim) and lens (N,)
        encoder_out, encoder_out_lens = self.encoder(x, x_lens)
        
        # Stage 3: Compute padding mask to match Wenet format (N, 1, T')
        # encoder_out_lens is (N,) containing actual sequence lengths after encoding
        batch_size = encoder_out.size(0)
        max_len = encoder_out.size(1)
        device = encoder_out.device
        
        # Create mask: False for valid positions, True for padding
        positions = torch.arange(max_len, device=device).unsqueeze(0)  # (1, T')
        encoder_mask = positions >= encoder_out_lens.unsqueeze(1)  # (N, T')
        encoder_mask = encoder_mask.unsqueeze(1)  # (N, 1, T') to match Wenet
        
        return encoder_out, encoder_mask

    @staticmethod
    def build_from_params(
        stage1_cfg: dict[str, Any],
        *,
        output_dim: int = 128,
    ) -> IcefallZipformerEncoder:
        """Build encoder from stage1 config.
        
        Args:
            stage1_cfg: Stage1 config dict containing zipformer hyperparams
            output_dim: Desired output dimension (default 128 for Zipformer2)
        
        Returns:
            Initialized IcefallZipformerEncoder
            
        Example:
            stage1_cfg = {
                "encoder_dim": "128,128,128,128,128,128",
                "num_encoder_layers": "2,4,3,2,4,3",
                "downsampling_factor": "1,2,4,8,4,2",
                # ... other params
            }
            encoder = IcefallZipformerEncoder.build_from_params(stage1_cfg)
        """
        Conv2dSubsampling, Zipformer2 = _load_icefall_modules()
        
        # Parse comma-separated config strings into tuples
        def _parse_tuple(s: str | tuple) -> tuple[int, ...]:
            if isinstance(s, tuple):
                return s
            return tuple(map(int, str(s).split(",")))
        
        encoder_dim = _parse_tuple(stage1_cfg.get("encoder_dim", "128,128,128,128,128,128"))
        num_encoder_layers = _parse_tuple(stage1_cfg.get("num_encoder_layers", "2,4,3,2,4,3"))
        downsampling_factor = _parse_tuple(stage1_cfg.get("downsampling_factor", "1,2,4,8,4,2"))
        feedforward_dim = _parse_tuple(stage1_cfg.get("feedforward_dim", "512,768,1024,1536,1024,768"))
        num_heads = _parse_tuple(stage1_cfg.get("num_heads", "4,4,4,8,4,4"))
        query_head_dim = _parse_tuple(stage1_cfg.get("query_head_dim", "32"))
        value_head_dim = _parse_tuple(stage1_cfg.get("value_head_dim", "12"))
        pos_head_dim = _parse_tuple(stage1_cfg.get("pos_head_dim", "4"))
        pos_dim = int(stage1_cfg.get("pos_dim", 48))
        encoder_unmasked_dim = _parse_tuple(stage1_cfg.get("encoder_unmasked_dim", "128,128,128,128,128,128"))
        cnn_module_kernel = _parse_tuple(stage1_cfg.get("cnn_module_kernel", "31,31,15,15,15,31"))
        causal = bool(stage1_cfg.get("causal", False))
        chunk_size = _parse_tuple(stage1_cfg.get("chunk_size", "-1"))
        left_context_frames = _parse_tuple(stage1_cfg.get("left_context_frames", "-1"))
        
        # Input feature dimension (always 80 for fbank)
        input_dim = int(stage1_cfg.get("input_dim", 80))
        
        # Build Conv2dSubsampling: (N, T, 80) → (N, T//4, encoder_dim[0])
        embed_module = Conv2dSubsampling(
            in_channels=input_dim,
            out_channels=encoder_dim[0],
            dropout=float(stage1_cfg.get("dropout_rate", 0.1)),
        )
        
        # Build Zipformer2 encoder
        encoder_module = Zipformer2(
            output_downsampling_factor=2,
            downsampling_factor=downsampling_factor,
            num_encoder_layers=num_encoder_layers,
            encoder_dim=encoder_dim,
            encoder_unmasked_dim=encoder_unmasked_dim,
            query_head_dim=query_head_dim,
            pos_head_dim=pos_head_dim,
            value_head_dim=value_head_dim,
            pos_dim=pos_dim,
            num_heads=num_heads,
            feedforward_dim=feedforward_dim,
            cnn_module_kernel=cnn_module_kernel,
            dropout=float(stage1_cfg.get("dropout_rate", 0.1)),
            warmup_batches=float(stage1_cfg.get("warmup_batches", 4000.0)),
            causal=causal,
            chunk_size=chunk_size,
            left_context_frames=left_context_frames,
        )
        
        return IcefallZipformerEncoder(embed_module, encoder_module, output_dim=output_dim)
