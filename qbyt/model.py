"""Query-by-text verifier with a keyword-vs-filler segmental readout.

The adapter-free v6 scorer learns one normalized inventory of phone, blank and
noise evidence directly on top of the encoder.  A bounded segmental graph is the
only route from that frame lattice to the deployed score; global audio/text
pooling cannot bypass the ordered keyword path.
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
    """Score a phoneme query against explicit filler and near-miss hypotheses.

    ``seq_logits[:, i]`` is the raw structural logit for completing query phones
    ``0..i``. The utterance logit is exactly the final valid prefix logit.
    Probability calibration is intentionally external to this module.
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
        max_inter_phone_gap_frames: int = 1,
        max_keyword_span_frames: int = 30,
        weakest_phone_temperature: float = 0.2,
        weakest_phone_weight: float = 1.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if post_num_layers < 0:
            raise ValueError("post_num_layers must be non-negative")
        if embed_dim <= 0:
            raise ValueError("embed_dim must be positive")
        if num_embeds <= 1:
            raise ValueError("num_embeds must contain blank and at least one phone")
        if (
            not math.isfinite(weakest_phone_temperature)
            or weakest_phone_temperature <= 0
        ):
            raise ValueError("weakest_phone_temperature must be finite and positive")
        if not math.isfinite(weakest_phone_weight) or weakest_phone_weight < 0:
            raise ValueError("weakest_phone_weight must be finite and non-negative")

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
        self.audio_key = nn.Linear(embed_dim, embed_dim, bias=False)
        self.text_query = nn.Linear(embed_dim, embed_dim, bias=False)
        self.match_log_scale = nn.Parameter(torch.tensor(math.log(10.0)))
        # Token 0 is the CTC blank/padding id and is deliberately not represented
        # by the padding-fixed embedding row. Phones 1..V-1 share the metric
        # inventory; blank and non-speech have independent audio-conditioned
        # logits and therefore remain meaningful without a phoneme adapter.
        self.phone_bias = nn.Parameter(torch.zeros(num_embeds - 1))
        self.blank_head = nn.Linear(embed_dim, 1)
        self.noise_head = nn.Linear(embed_dim, 1)

        duration_count = max_phone_duration_frames - min_phone_duration_frames + 1
        if duration_count <= 0:
            raise ValueError(
                "max_phone_duration_frames must be >= min_phone_duration_frames"
            )
        self.duration_logits = nn.Embedding(
            num_embeddings=num_embeds,
            embedding_dim=duration_count,
            padding_idx=0,
        )
        nn.init.zeros_(self.duration_logits.weight)

        self.aligner = BoundedSegmentalAligner(
            min_phone_duration_frames=min_phone_duration_frames,
            max_phone_duration_frames=max_phone_duration_frames,
            max_inter_phone_gap_frames=max_inter_phone_gap_frames,
            max_keyword_span_frames=max_keyword_span_frames,
            temperature=weakest_phone_temperature,
        )
        self.weakest_phone_weight = float(weakest_phone_weight)

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
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
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

        audio_keys = F.normalize(self.audio_key(audio_states), dim=-1, eps=1e-6)
        phone_prototypes = F.normalize(
            self.text_query(self.text_projection.weight[1:]),
            dim=-1,
            eps=1e-6,
        )
        match_scale = self.match_log_scale.clamp(
            min=math.log(1.0e-2), max=math.log(100.0)
        ).exp()
        phone_logits = match_scale * torch.einsum(
            "btd,vd->btv", audio_keys, phone_prototypes
        ) + self.phone_bias
        class_logits = torch.cat(
            (
                phone_logits,
                self.blank_head(audio_states),
                self.noise_head(audio_states),
            ),
            dim=-1,
        )
        # The graph DP always runs in float32, and its probabilistic contract
        # begins here with one normalized phone/blank/noise competition.
        class_log_probs = F.log_softmax(class_logits.float(), dim=-1)

        phone_class_count = self.text_projection.num_embeddings - 1
        safe_phone_indices = (text - 1).clamp(min=0, max=phone_class_count - 1)
        target_log_probs = class_log_probs[:, :, :phone_class_count].transpose(1, 2)
        target_log_probs = target_log_probs.gather(
            1,
            safe_phone_indices.unsqueeze(-1).expand(-1, -1, frame_width),
        )

        # Filler is query-relative: blank, noise, and every inventory phone not
        # used by this query. A one-phone deletion graph can consequently explain
        # a substitution as filler while the exact graph must explain every phone.
        query_phone_counts = torch.zeros(
            batch_size,
            phone_class_count,
            device=text.device,
            dtype=torch.long,
        )
        query_phone_counts.scatter_add_(
            1,
            safe_phone_indices,
            phone_mask.to(dtype=torch.long),
        )
        filler_class_mask = torch.cat(
            (
                query_phone_counts.eq(0),
                torch.ones(batch_size, 2, device=text.device, dtype=torch.bool),
            ),
            dim=1,
        )
        filler_log_probs = torch.logsumexp(
            class_log_probs.masked_fill(
                ~filler_class_mask.unsqueeze(1),
                -torch.inf,
            ),
            dim=-1,
        )
        target_llr = target_log_probs - filler_log_probs.unsqueeze(1)

        valid_lattice = phone_mask.unsqueeze(2) & frame_mask.unsqueeze(1)
        target_llr = torch.where(
            valid_lattice,
            target_llr,
            torch.zeros_like(target_llr),
        )
        filler_log_probs = torch.where(
            frame_mask,
            filler_log_probs,
            torch.zeros_like(filler_log_probs),
        )

        duration_count = self.duration_logits.embedding_dim
        duration_potentials = F.log_softmax(
            self.duration_logits(text).float(),
            dim=-1,
        ) + math.log(float(duration_count))
        duration_potentials = torch.where(
            phone_mask.unsqueeze(-1),
            duration_potentials,
            torch.zeros_like(duration_potentials),
        )
        return (
            target_llr,
            filler_log_probs,
            duration_potentials,
            text_lengths,
            speech_lengths,
            phone_mask,
            frame_mask,
        )

    def forward(
        self,
        speech: torch.Tensor,
        text: torch.Tensor,
        speech_lengths: torch.Tensor | None = None,
        text_lengths: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        (
            target_llr,
            filler_log_probs,
            duration_potentials,
            text_lengths,
            speech_lengths,
            phone_mask,
            _,
        ) = self._encode_lattice(speech, text, speech_lengths, text_lengths)
        alignment = self.aligner(
            target_llr,
            filler_log_probs,
            text_lengths,
            speech_lengths,
            duration_log_probs=duration_potentials,
        )
        # log-sigmoid is always <= 0, so completeness can veto an incomplete
        # keyword but can never create a positive route around the graph LLR.
        legal_prefix = phone_mask & alignment.has_legal_path
        completeness_veto = torch.where(
            legal_prefix,
            self.weakest_phone_weight
            * F.logsigmoid(alignment.weakest_phone_evidence),
            torch.zeros_like(alignment.weakest_phone_evidence),
        )
        seq_logits = alignment.prefix_llr + completeness_veto
        seq_logits = torch.where(
            legal_prefix,
            seq_logits,
            torch.full_like(seq_logits, self.aligner.invalid_score),
        )

        if text.size(1) == 0:
            utterance_logits = seq_logits.new_full(
                (text.size(0),), self.aligner.invalid_score
            )
        else:
            last = (text_lengths - 1).clamp_min(0)
            utterance_logits = seq_logits.gather(1, last.unsqueeze(1)).squeeze(1)
            utterance_logits = torch.where(
                text_lengths.gt(0),
                utterance_logits,
                torch.full_like(utterance_logits, self.aligner.invalid_score),
            )
        return utterance_logits, seq_logits


__all__ = ["QbyT"]
