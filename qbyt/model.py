"""Query-by-text keyword verifier with structural temporal alignment.

The scorer deliberately has no text/audio global-attention path. It first
builds a local phone/frame emission lattice and then permits the utterance score
to see that lattice only through a bounded monotonic segmental aligner.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from qbyt.monotonic_alignment import BoundedSegmentalAligner


class _LocalContextBlock(nn.Module):
    """Residual temporal block whose receptive field is explicitly bounded."""

    def __init__(self, dim: int, *, kernel_size: int, dropout: float) -> None:
        super().__init__()
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("local context kernel must be a positive odd integer")
        self.norm = nn.LayerNorm(dim)
        self.depthwise = nn.Conv1d(
            dim,
            dim,
            kernel_size,
            padding=kernel_size // 2,
            groups=dim,
        )
        self.expand = nn.Conv1d(dim, dim * 2, 1)
        self.project = nn.Conv1d(dim, dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, states: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        residual = states
        values = self.norm(states)
        values = values * mask.unsqueeze(-1).to(values.dtype)
        values = values.transpose(1, 2)
        values = self.depthwise(values)
        values = F.glu(self.expand(values), dim=1)
        values = self.project(self.dropout(values)).transpose(1, 2)
        states = residual + self.dropout(values)
        return states * mask.unsqueeze(-1).to(states.dtype)


class QbyT(nn.Module):
    """Score a phoneme query against speech using one compact legal path.

    ``seq_logits[:, i]`` is the calibrated score for completing query phones
    ``0..i``. The utterance logit is exactly the final valid prefix logit; there
    is no independent classifier capable of bypassing the alignment topology.
    """

    def __init__(
        self,
        encoder_output_size: int = 144,
        num_embeds: int = 73,
        embed_dim: int = 128,
        post_num_layers: int = 2,
        *,
        local_context_kernel: int = 5,
        min_phone_duration_frames: int = 1,
        max_phone_duration_frames: int = 8,
        max_inter_phone_gap_frames: int = 2,
        max_keyword_span_frames: int = 30,
        alignment_temperature: float = 0.2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if post_num_layers < 0:
            raise ValueError("post_num_layers must be non-negative")
        if embed_dim <= 0:
            raise ValueError("embed_dim must be positive")

        self.audio_projection = nn.Linear(encoder_output_size, embed_dim)
        self.text_projection = nn.Embedding(
            num_embeddings=num_embeds,
            embedding_dim=embed_dim,
            padding_idx=0,
        )
        self.audio_context = nn.ModuleList(
            _LocalContextBlock(
                embed_dim,
                kernel_size=local_context_kernel,
                dropout=dropout,
            )
            for _ in range(post_num_layers)
        )
        # Query context never sees the audio sequence.
        self.text_context = nn.ModuleList(
            _LocalContextBlock(embed_dim, kernel_size=3, dropout=dropout)
            for _ in range(post_num_layers)
        )
        self.audio_key = nn.Linear(embed_dim, embed_dim, bias=False)
        self.text_query = nn.Linear(embed_dim, embed_dim, bias=False)
        self.match_log_scale = nn.Parameter(torch.tensor(math.log(10.0)))
        self.match_bias = nn.Parameter(torch.zeros(()))

        self.aligner = BoundedSegmentalAligner(
            min_phone_duration_frames=min_phone_duration_frames,
            max_phone_duration_frames=max_phone_duration_frames,
            max_inter_phone_gap_frames=max_inter_phone_gap_frames,
            max_keyword_span_frames=max_keyword_span_frames,
            temperature=alignment_temperature,
        )

        # The aligner emits a normalized log probability (<= 0). A shared
        # monotone calibration cannot create an alternate route around the DP.
        initial_scale = 4.0
        self.raw_score_scale = nn.Parameter(
            torch.tensor(math.log(math.expm1(initial_scale)))
        )
        self.score_bias = nn.Parameter(torch.tensor(3.0))

    @staticmethod
    def _resolve_lengths(
        values: torch.Tensor | None,
        *,
        fallback: torch.Tensor,
        batch_size: int,
        maximum: int,
        device: torch.device,
        name: str,
    ) -> torch.Tensor:
        if values is None:
            values = fallback
        values = torch.as_tensor(values, device=device)
        if values.ndim != 1 or values.numel() != batch_size:
            raise ValueError(
                f"{name} must have shape [{batch_size}], got {tuple(values.shape)}"
            )
        if values.dtype == torch.bool or torch.is_floating_point(values):
            raise ValueError(f"{name} must contain integer lengths")
        values = values.to(dtype=torch.long)
        if torch.any(values < 0) or torch.any(values > maximum):
            raise ValueError(f"{name} entries must lie in [0, {maximum}]")
        return values

    @staticmethod
    def _length_mask(lengths: torch.Tensor, width: int) -> torch.Tensor:
        return torch.arange(width, device=lengths.device).unsqueeze(0) < lengths.unsqueeze(1)

    def _encode_lattice(
        self,
        speech: torch.Tensor,
        text: torch.Tensor,
        speech_lengths: torch.Tensor | None,
        text_lengths: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if speech.ndim != 3:
            raise ValueError("speech must have shape [B, T, C]")
        if text.ndim != 2 or text.size(0) != speech.size(0):
            raise ValueError("text must have shape [B, U] with the same batch size as speech")

        batch_size, frame_width, _ = speech.shape
        phone_width = text.size(1)
        speech_lengths = self._resolve_lengths(
            speech_lengths,
            fallback=torch.full(
                (batch_size,), frame_width, device=speech.device, dtype=torch.long
            ),
            batch_size=batch_size,
            maximum=frame_width,
            device=speech.device,
            name="speech_lengths",
        )
        text_lengths = self._resolve_lengths(
            text_lengths,
            fallback=text.ne(0).sum(dim=1),
            batch_size=batch_size,
            maximum=phone_width,
            device=speech.device,
            name="text_lengths",
        )
        frame_mask = self._length_mask(speech_lengths, frame_width)
        phone_mask = self._length_mask(text_lengths, phone_width)

        audio_states = self.audio_projection(speech)
        audio_states = audio_states * frame_mask.unsqueeze(-1).to(audio_states.dtype)
        for block in self.audio_context:
            audio_states = block(audio_states, frame_mask)

        text_states = self.text_projection(text)
        text_states = text_states * phone_mask.unsqueeze(-1).to(text_states.dtype)
        for block in self.text_context:
            text_states = block(text_states, phone_mask)

        audio_keys = F.normalize(self.audio_key(audio_states), dim=-1, eps=1e-6)
        text_queries = F.normalize(self.text_query(text_states), dim=-1, eps=1e-6)
        match_scale = self.match_log_scale.clamp(
            min=math.log(1.0e-2), max=math.log(100.0)
        ).exp()
        emissions = match_scale * torch.einsum(
            "bud,btd->but", text_queries, audio_keys
        ) + self.match_bias
        valid_lattice = phone_mask.unsqueeze(2) & frame_mask.unsqueeze(1)
        emissions = torch.where(valid_lattice, emissions, torch.zeros_like(emissions))
        return emissions, text_lengths, speech_lengths, phone_mask, frame_mask

    def forward(
        self,
        speech: torch.Tensor,
        text: torch.Tensor,
        speech_lengths: torch.Tensor | None = None,
        text_lengths: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        (
            emissions,
            text_lengths,
            speech_lengths,
            _,
            _,
        ) = self._encode_lattice(speech, text, speech_lengths, text_lengths)
        prefix_evidence = self.aligner(emissions, text_lengths, speech_lengths)
        score_scale = F.softplus(self.raw_score_scale) + 1.0e-4
        seq_logits = score_scale * prefix_evidence + self.score_bias

        if text.size(1) == 0:
            utterance_logits = seq_logits.new_full(
                (text.size(0),), self.aligner.invalid_score
            )
        else:
            last = (text_lengths - 1).clamp_min(0)
            utterance_logits = seq_logits.gather(1, last.unsqueeze(1)).squeeze(1)
            invalid = score_scale * self.aligner.invalid_score + self.score_bias
            utterance_logits = torch.where(
                text_lengths.gt(0), utterance_logits, invalid.expand_as(utterance_logits)
            )
        return utterance_logits, seq_logits


__all__ = ["QbyT"]
