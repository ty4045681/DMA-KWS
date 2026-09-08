"""Icefall Zipformer2 encoder adapter for Stage II training compatibility.

Wraps icefall's Zipformer2 encoder + Conv2dSubsampling to expose a Wenet-compatible
interface: forward(feat, feat_lengths) → (encoder_out, encoder_mask).

The adapter ensures shape consistency:
- Wenet: (N, T, C) → encoder → (N, T', C_out) + mask (N, 1, T')
- Icefall: (N, T, C) → embed → (N, T_e, D) → encoder → (N, T', max_dim)
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from dma_kws.pathing import ensure_icefall_on_path

#: ``Zipformer2`` halves its own output on top of ``Conv2dSubsampling``. Shared by
#: the constructor call and :meth:`IcefallZipformerEncoder.output_frames` so the
#: two can never disagree about the encoder's frame rate.
OUTPUT_DOWNSAMPLING_FACTOR = 2


def embed_output_frames(num_frames: int) -> int:
    """Frames ``Conv2dSubsampling`` emits for ``num_frames`` input frames.

    Transcribes the upstream formula: its class docstring states
    ``T' = (T - 3) // 2 - 2 == (T - 7) // 2`` and its ``forward`` computes the
    same thing as ``x_lens = (x_lens - 7) // 2``. Short inputs give a
    non-positive result, which is what makes them unusable rather than merely
    short.
    """
    return (num_frames - 3) // 2 - 2


def _load_icefall_modules() -> tuple[type, type, type]:
    """Lazily import icefall Zipformer2 and Conv2dSubsampling.
    
    Returns:
        (Conv2dSubsampling, Zipformer2, ScheduledFloat) classes
    """
    ensure_icefall_on_path()
    try:
        from subsampling import Conv2dSubsampling  # pyright: ignore[reportMissingImports]
        from scaling import ScheduledFloat  # pyright: ignore[reportMissingImports]
        from zipformer import Zipformer2  # pyright: ignore[reportMissingImports]
    except ImportError as exc:
        raise SystemExit(
            "Failed to import icefall Zipformer2/Conv2dSubsampling. "
            "Ensure ICEFALL_ROOT env var is set to icefall repo root "
            "and ICEFALL_ROOT/egs/gigaspeech/KWS/zipformer is in sys.path."
        ) from exc
    return Conv2dSubsampling, Zipformer2, ScheduledFloat


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

    def set_batch_count(self, batch_count: float) -> None:
        """Set all Icefall schedules, including the input subsampling modules.

        Mirror the recipe's ``train.set_batch_count`` without importing its
        command-line training entrypoint and unrelated training dependencies.
        ``ScheduledFloat`` is an nn.Module, so named_modules reaches schedules
        as well as modules that carry their own progress-dependent behavior.
        """
        if not math.isfinite(batch_count) or batch_count < 0:
            raise ValueError("Icefall batch_count must be finite and non-negative")
        for name, module in self.named_modules():
            if hasattr(module, "batch_count"):
                module.batch_count = float(batch_count)
            if hasattr(module, "name"):
                module.name = name

    def apply_stream_config(
        self,
        chunk_sizes: tuple[int, ...],
        left_context_frames: tuple[int, ...],
    ) -> None:
        """Set the chunked-attention settings used by the next forward pass.

        ``Zipformer2.get_chunk_info`` draws from these tuples with
        ``random.choice`` and is *not* gated on ``self.training``, so a
        multi-value tuple randomizes every forward pass, eval included. Callers
        go through :func:`dma_kws.nn.run_encoder`, which passes a single-element
        tuple for every phase except opt-in multi-latency training.
        """
        self.encoder.chunk_size = tuple(chunk_sizes)
        self.encoder.left_context_frames = tuple(left_context_frames)

    def output_frames(self, num_input_frames: int) -> int:
        """Encoder frames produced for ``num_input_frames`` fbank frames.

        ``Conv2dSubsampling`` reduces the length first, then ``Zipformer2``
        applies ``output_downsampling_factor`` as ``(t + 1) // 2``. Callers use
        this to reject inputs that would subsample away to nothing before the
        convolutions raise a shape error.
        """
        subsampled = embed_output_frames(num_input_frames)
        if subsampled <= 0:
            return 0
        return (subsampled + 1) // OUTPUT_DOWNSAMPLING_FACTOR

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
            encoder_mask: Valid-frame mask, shape (N, 1, T')
        """
        # Stage 1: Subsampling + embedding projection
        # Conv2dSubsampling expects (N, T, C), returns (N, T_subsample, D)
        x, x_lens = self.encoder_embed(feat, feat_lengths)

        # icefall Zipformer2 expects (T, N, C), so follow the same layout as
        # icefall's own AsrModel.forward_encoder before and after the encoder.
        src_key_padding_mask = _make_pad_mask(x_lens)
        x = x.permute(1, 0, 2)
        
        # Stage 2: Zipformer2 encoder
        # Returns (T', N, max_encoder_dim) and lens (N,)
        encoder_out, encoder_out_lens = self.encoder(x, x_lens, src_key_padding_mask)
        encoder_out = encoder_out.permute(1, 0, 2).contiguous()
        
        # Stage 3: Compute valid-frame mask to match the existing Wenet contract.
        # encoder_out_lens is (N,) containing actual sequence lengths after encoding
        max_len = encoder_out.size(1)
        device = encoder_out.device
        
        # Create mask: True for valid positions, False for padding.
        positions = torch.arange(max_len, device=device).unsqueeze(0)  # (1, T')
        encoder_mask = positions < encoder_out_lens.unsqueeze(1)  # (N, T')
        encoder_mask = encoder_mask.unsqueeze(1)  # (N, 1, T') to match Wenet
        
        return encoder_out, encoder_mask

    @staticmethod
    def build_from_params(
        stage1_cfg: dict[str, Any],
        *,
        output_dim: int = 128,
        policy: Any = None,
    ) -> IcefallZipformerEncoder:
        """Build encoder from stage1 config.
        
        Args:
            stage1_cfg: Stage1 config dict containing zipformer hyperparams
            output_dim: Desired output dimension (default 128 for Zipformer2)
            policy: Resolved :class:`~dma_kws.configs.schema.StreamPolicy`; when
                omitted it is resolved from ``stage1_cfg``. The encoder is left
                at the deployment operating point so that any caller bypassing
                :func:`dma_kws.nn.run_encoder` still gets deterministic output.
        
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
        if policy is None:
            from dma_kws.config import resolve_stream_policy

            policy = resolve_stream_policy(stage1_cfg)
        Conv2dSubsampling, Zipformer2, ScheduledFloat = _load_icefall_modules()
        
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
        # Built at the deployment operating point; dma_kws.nn.run_encoder swaps in
        # the training tuples per forward pass when multi-latency training is on.
        chunk_size = (policy.chunk_size,) if policy.enabled else (-1,)
        left_context_frames = (policy.left_context_frames,) if policy.enabled else (-1,)
        
        # Input feature dimension (always 80 for fbank)
        input_dim = int(stage1_cfg.get("input_dim", 80))
        
        # Match icefall's dropout behavior used in train/finetune by default.
        use_dropout_schedule = bool(stage1_cfg.get("use_icefall_dropout_schedule", True))
        if use_dropout_schedule:
            dropout = ScheduledFloat((0.0, 0.3), (20000.0, 0.1))
        else:
            dropout = float(stage1_cfg.get("dropout_rate", 0.1))

        # Build Conv2dSubsampling: (N, T, 80) → (N, T//4, encoder_dim[0])
        embed_module = Conv2dSubsampling(
            in_channels=input_dim,
            out_channels=encoder_dim[0],
            dropout=dropout,
        )
        
        # Build Zipformer2 encoder
        encoder_module = Zipformer2(
            output_downsampling_factor=OUTPUT_DOWNSAMPLING_FACTOR,
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
            dropout=dropout,
            warmup_batches=float(stage1_cfg.get("warmup_batches", 4000.0)),
            causal=causal,
            chunk_size=chunk_size,
            left_context_frames=left_context_frames,
        )
        
        return IcefallZipformerEncoder(embed_module, encoder_module, output_dim=output_dim)


def _make_pad_mask(lengths: torch.Tensor) -> torch.Tensor:
    """Create a padding mask of shape (N, T), True means masked position."""
    max_len = int(lengths.max().item()) if lengths.numel() > 0 else 0
    positions = torch.arange(max_len, device=lengths.device).unsqueeze(0)
    return positions >= lengths.unsqueeze(1)
