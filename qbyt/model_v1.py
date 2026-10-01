"""Paper-original (readout version 1) speaker-independent QbyT.

Behaviour-faithful port of the author's v1 implementation
(train/two_stage/model.py in aizhiqi-work/DMA-KWS, HEAD 207d056; the same file
as decode/stage2/model.py). The deployed score is the GRU state at the last
position of the concatenated [text][audio] sequence INCLUDING batch padding.
The paper's training and its decode/stage2/test.py evaluation ran that exact
function, so the padding dependence is part of the checkpoint's semantics and is
preserved here on purpose. Do not "fix" it to a masked or last-valid readout:
the released weights were trained against this function.

Parameter names match the qbyt.* keys of the released 155k-v2-ft.ckpt
one-to-one, so a strict load_state_dict is the compatibility check.

Only the text+audio (SI) scorer is implemented. The multimodal enrolment
variants (SD: 155k-mm-f1/f2) add modality_enc.enr_audio_emb and cross_attn.* and
are intentionally out of scope.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

QBYT_V1_READOUT = "gru_last_padded"
DEFAULT_QBYT_V1_EMBED_DIM = 128
DEFAULT_QBYT_V1_POST_NUM_LAYERS = 2
_QBYT_V1_NHEAD = 4


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding, verbatim from the author's v1 code."""

    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class ModalityEmbedding(nn.Module):
    """Modality-type embedding distinguishing text and audio tokens."""

    def __init__(self, d_model):
        super().__init__()
        self.text_emb = nn.Parameter(torch.randn(1, 1, d_model))
        self.audio_emb = nn.Parameter(torch.randn(1, 1, d_model))

    def forward(self, x, modality_type):
        batch_size, seq_len = x.size(0), x.size(1)
        if modality_type == "text":
            return x + self.text_emb.expand(batch_size, seq_len, -1)
        if modality_type == "audio":
            return x + self.audio_emb.expand(batch_size, seq_len, -1)
        raise ValueError(f"unsupported modality type: {modality_type}")


class QbyT(nn.Module):
    """Text+audio QbyT with the v1 padded-sequence GRU readout."""

    def __init__(
        self,
        encoder_output_size: int = 144,
        num_embeds: int = 73,
        embed_dim: int = DEFAULT_QBYT_V1_EMBED_DIM,
        post_num_layers: int = DEFAULT_QBYT_V1_POST_NUM_LAYERS,
    ) -> None:
        super().__init__()
        self.audio_projection = nn.Linear(encoder_output_size, embed_dim)
        self.text_projection = nn.Embedding(
            num_embeddings=num_embeds, embedding_dim=embed_dim
        )
        self.pos_enc = PositionalEncoding(embed_dim)
        self.modality_enc = ModalityEmbedding(embed_dim)

        self.phone_matchor = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=_QBYT_V1_NHEAD,
                dim_feedforward=embed_dim * 4,
                dropout=0.1,
                batch_first=True,
            ),
            num_layers=post_num_layers,
        )
        self.gru = nn.GRU(embed_dim, embed_dim, batch_first=True)
        self.fc = nn.Linear(embed_dim, 1)
        self.seq_fc = nn.Linear(embed_dim, 1)

    def forward(self, speech, text):
        """Score (speech, text) exactly as the v1 implementation does.

        speech is the encoder output for the whole padded batch and text the
        padded keyword ids. No lengths or masks are accepted because the
        original function has none: the utterance logit is the final position of
        the padded concatenation, and seq_fc reads the text block.
        """
        text_emb = self.text_projection(text)
        text_emb = self.pos_enc(text_emb)
        text_emb = self.modality_enc(text_emb, "text")

        audio_emb = self.audio_projection(speech)
        audio_emb = self.pos_enc(audio_emb)
        audio_emb = self.modality_enc(audio_emb, "audio")

        combined_feat = torch.cat([text_emb, audio_emb], dim=1)
        combined_feat = self.phone_matchor(combined_feat)
        gru_out, _ = self.gru(combined_feat)
        logits = self.fc(gru_out[:, -1, :]).squeeze(-1)

        text_logits = self.seq_fc(combined_feat[:, : text_emb.shape[1], :]).squeeze(-1)
        return logits, text_logits
