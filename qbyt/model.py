import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import math
from typing import NamedTuple

# Keep this vendored model importable on its own (for example from ``qbyt/``).
# These public mode spellings intentionally mirror ``dma_kws.stage2.readout``;
# the application layer validates/configures them before constructing QbyT, and
# this local guard prevents the vendored model from depending back on dma_kws.
GRU_LAST_READOUT = "gru_last"
EPS_MEAN_READOUT = "eps_mean"
EPS_SOFTMIN_READOUT = "eps_softmin"
_QBYT_READOUT_MODES = frozenset(
    {GRU_LAST_READOUT, EPS_MEAN_READOUT, EPS_SOFTMIN_READOUT}
)


class QbyTReadoutDetails(NamedTuple):
    """Analysis-only tensors produced by the configured utterance readout."""

    position_logits: torch.Tensor | None
    position_mask: torch.Tensor


def _normalize_qbyt_readout_mode(value):
    mode = str(value).strip().lower()
    if mode not in _QBYT_READOUT_MODES:
        choices = ", ".join(sorted(_QBYT_READOUT_MODES))
        raise ValueError(
            f"Unsupported QbyT readout mode {value!r}; expected one of: {choices}"
        )
    return mode


def _normalize_readout_temperature(value):
    if isinstance(value, bool):
        raise ValueError(
            f"QbyT readout_temperature must be a finite positive number, got {value!r}"
        )
    try:
        temperature = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"QbyT readout_temperature must be a finite positive number, got {value!r}"
        ) from exc
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError(
            f"QbyT readout_temperature must be a finite positive number, got {value!r}"
        )
    return temperature


def _masked_normalized_softmin(position_logits, position_mask, temperature):
    """Pool valid raw logits with a stable, length-normalized soft minimum.

    The minimum shift avoids forming ``-position_logits / temperature``
    directly, which can overflow for a small temperature even though the
    mathematical result is finite. The reduction deliberately runs in float32
    under mixed precision. Normalizing by the valid-position count makes equal
    position logits pool back to that same logit rather than adding a keyword-
    length-dependent offset.
    """

    logits = position_logits.float()
    mask = position_mask.to(device=logits.device, dtype=torch.bool)
    batch_size, width = logits.shape
    if width == 0:
        return logits.new_zeros((batch_size,))

    counts = mask.sum(dim=1)
    empty = counts.eq(0)
    first_position = torch.arange(width, device=logits.device).eq(0).unsqueeze(0)
    empty_sentinel = empty.unsqueeze(1) & first_position
    safe_mask = mask | empty_sentinel

    # An empty anchor receives one synthetic zero solely inside the reduction.
    # It therefore has the same finite neutral logit as the existing mean
    # readout, without evaluating logsumexp over an all--inf row.
    safe_logits = torch.where(mask, logits, torch.zeros_like(logits))
    minimum = safe_logits.masked_fill(~safe_mask, torch.inf).min(dim=1).values
    shifted = torch.where(
        mask,
        -(logits - minimum.unsqueeze(1)) / temperature,
        torch.full_like(logits, -torch.inf),
    )
    shifted = torch.where(empty_sentinel, torch.zeros_like(shifted), shifted)
    log_mean_exp = torch.logsumexp(shifted, dim=1) - counts.clamp_min(1).log()
    pooled = minimum - temperature * log_mean_exp
    return torch.where(empty, torch.zeros_like(pooled), pooled)


class PositionalEncoding(nn.Module):
    """位置编码模块"""
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class ModalityEmbedding(nn.Module):
    """模态类型嵌入，用于区分不同模态"""
    def __init__(self, d_model):
        super(ModalityEmbedding, self).__init__()
        self.text_emb = nn.Parameter(torch.randn(1, 1, d_model))
        self.audio_emb = nn.Parameter(torch.randn(1, 1, d_model))

    def forward(self, x, modality_type):
        batch_size, seq_len = x.size(0), x.size(1)
        if modality_type == 'text':
            return x + self.text_emb.expand(batch_size, seq_len, -1)
        elif modality_type == 'audio':
            return x + self.audio_emb.expand(batch_size, seq_len, -1)
        else:
            raise ValueError(f"不支持的模态类型: {modality_type}")


class GRUFCModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GRUFCModel, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        gru_out, _ = self.gru(x)
        gru_last_output = gru_out[:, -1, :]
        fc_out = self.fc(gru_last_output)
        return fc_out


class QbyT(nn.Module):
    def __init__(
        self,
        encoder_output_size=144,
        num_embeds=73, # 其中0,1,2均不会使用 <blank> <unk> <sos/eos>
        embed_dim=128,
        post_num_layers=2,
        readout_mode=GRU_LAST_READOUT,
        readout_temperature=1.0,
    ):
        super().__init__()
        self.audio_projection = nn.Linear(encoder_output_size, embed_dim)
        self.text_projection = nn.Embedding(num_embeddings=num_embeds, embedding_dim=embed_dim)
        self.pos_enc = PositionalEncoding(embed_dim)
        self.modality_enc = ModalityEmbedding(embed_dim)

        self.phone_matchor = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=4,
                dim_feedforward=embed_dim*4,
                dropout=0.1,
                batch_first=True
            ),
            num_layers=post_num_layers
        )
        self.readout_mode = _normalize_qbyt_readout_mode(readout_mode)
        self.readout_temperature = _normalize_readout_temperature(
            readout_temperature
        )
        if self.readout_mode == GRU_LAST_READOUT:
            self.gru = nn.GRU(embed_dim, embed_dim, batch_first=True)
            self.fc = nn.Linear(embed_dim, 1)
            self.final_pos_fc = None
        else:
            self.gru = None
            self.fc = None
            self.final_pos_fc = nn.Linear(embed_dim, 1)
        self.seq_fc = nn.Linear(embed_dim, 1)


    def forward(self, speech, text, speech_lengths=None, text_lengths=None):
        """Score each (text, speech) pair in the batch.

        Pass ``speech_lengths`` and ``text_lengths`` whenever the batch is
        padded. Omitting ``speech_lengths`` treats every audio frame -- including
        padding -- as valid, which shifts the readout into the padded tail; the
        ``text.ne(0)`` fallback for ``text_lengths`` assumes id 0 is only ever
        padding.
        """
        logits, text_logits, _ = self._forward_impl(
            speech,
            text,
            speech_lengths=speech_lengths,
            text_lengths=text_lengths,
            include_readout_details=False,
        )
        return logits, text_logits

    def forward_with_readout_details(
        self,
        speech,
        text,
        speech_lengths=None,
        text_lengths=None,
    ):
        """Score pairs and expose valid EPS position logits for diagnostics.

        The ordinary :meth:`forward` contract remains a two-tuple so training
        and deployment callers are unaffected. For either EPS readout, position
        logits are zero outside ``position_mask``; for ``gru_last`` they are
        ``None`` because that readout has no shared per-position scorer.
        """
        logits, text_logits, details = self._forward_impl(
            speech,
            text,
            speech_lengths=speech_lengths,
            text_lengths=text_lengths,
            include_readout_details=True,
        )
        return logits, text_logits, details

    def _forward_impl(
        self,
        speech,
        text,
        *,
        speech_lengths=None,
        text_lengths=None,
        include_readout_details,
    ):
        # 文本处理
        text_emb = self.text_projection(text)
        text_emb = self.pos_enc(text_emb)
        text_emb = self.modality_enc(text_emb, 'text')

        # 音频处理
        audio_emb = self.audio_projection(speech)
        audio_emb = self.pos_enc(audio_emb)
        audio_emb = self.modality_enc(audio_emb, 'audio')

        combined_feat = torch.cat([text_emb, audio_emb], dim=1)
        batch_size = combined_feat.size(0)
        if speech_lengths is None:
            speech_lengths = torch.full(
                (batch_size,),
                speech.size(1),
                device=speech.device,
                dtype=torch.long,
            )
        else:
            speech_lengths = speech_lengths.to(device=speech.device, dtype=torch.long)
        speech_lengths = speech_lengths.clamp(min=0, max=speech.size(1))

        if text_lengths is None:
            text_lengths = text.ne(0).sum(dim=1)
        else:
            text_lengths = text_lengths.to(device=speech.device, dtype=torch.long)
        text_lengths = text_lengths.clamp(min=0, max=text.size(1))

        # Both blocks are padded to their batch maximum, so the naive concat is
        # [valid text][text padding][valid audio][audio padding]. Re-pack each
        # sample as [valid text][valid audio][padding] instead. Leaving the text
        # padding in the middle would make the readout batch-dependent twice
        # over: the last valid audio frame sits at a batch-dependent index, and
        # the GRU -- which src_key_padding_mask does not reach, it only masks
        # attention keys -- would walk that padding before reaching it.
        text_width = text_emb.size(1)
        total_width = combined_feat.size(1)
        positions = torch.arange(total_width, device=speech.device).unsqueeze(0)
        text_lengths_col = text_lengths.unsqueeze(1)
        # clamp(min=1) keeps position 0 valid: an all-masked row makes the
        # attention softmax produce NaN rather than fail.
        valid_lengths = (text_lengths + speech_lengths).clamp(min=1, max=total_width)
        valid = positions < valid_lengths.unsqueeze(1)
        source_indices = torch.where(
            positions < text_lengths_col,
            positions.expand(batch_size, -1),
            text_width + (positions - text_lengths_col).clamp(min=0),
        ).clamp(max=total_width - 1)
        combined_feat = combined_feat.gather(
            1, source_indices.unsqueeze(-1).expand(-1, -1, combined_feat.size(-1))
        ) * valid.unsqueeze(-1).to(combined_feat.dtype)

        combined_feat = self.phone_matchor(combined_feat, src_key_padding_mask=~valid)
        # After re-packing, positions [0, text_lengths) hold this sample's text
        # and [text_lengths, text_width) hold audio frames. That is safe because
        # build_seq_label emits one label per anchor token, so the Stage II
        # sequence loss masks everything past text_lengths -- but it does mean
        # this slice must not be read as "the text block".
        text_states = combined_feat[:, :text_width, :]
        text_logits = self.seq_fc(text_states).squeeze(-1)

        if self.readout_mode == GRU_LAST_READOUT:
            # No pack_padded_sequence needed: the readout below sits at the last
            # valid frame, so padding walked afterwards cannot reach it.
            gru_out, _ = self.gru(combined_feat)
            last_valid_indices = (valid_lengths - 1).view(batch_size, 1, 1).expand(
                -1, 1, gru_out.size(-1)
            )
            gru_out = gru_out.gather(1, last_valid_indices).squeeze(1)
            logits = self.fc(gru_out).squeeze(-1)
            if include_readout_details:
                text_positions = torch.arange(text_width, device=text.device).unsqueeze(0)
                text_mask = text_positions < text_lengths.unsqueeze(1)
                readout_details = QbyTReadoutDetails(
                    position_logits=None,
                    position_mask=text_mask,
                )
            else:
                readout_details = None
        elif self.readout_mode in (EPS_MEAN_READOUT, EPS_SOFTMIN_READOUT):
            position_logits = self.final_pos_fc(text_states).squeeze(-1)
            text_positions = torch.arange(text_width, device=text.device).unsqueeze(0)
            text_mask = text_positions < text_lengths.unsqueeze(1)
            # For a short anchor, [text_length, text_width) contains re-packed
            # audio frames rather than text padding. Masking is part of the
            # readout definition, not only a numerical optimization.
            valid_position_logits = torch.where(
                text_mask,
                position_logits,
                torch.zeros_like(position_logits),
            )
            if self.readout_mode == EPS_MEAN_READOUT:
                logits = valid_position_logits.sum(dim=1) / text_mask.sum(
                    dim=1
                ).clamp_min(1)
            else:
                logits = _masked_normalized_softmin(
                    position_logits,
                    text_mask,
                    self.readout_temperature,
                )
            readout_details = (
                QbyTReadoutDetails(
                    position_logits=valid_position_logits,
                    position_mask=text_mask,
                )
                if include_readout_details
                else None
            )
        else:  # normalize_qbyt_readout_mode makes this unreachable.
            raise RuntimeError(f"Unhandled QbyT readout mode: {self.readout_mode}")
        return logits, text_logits, readout_details


if __name__ == "__main__":
    # Imported here rather than at module scope: the only use is this demo, and
    # models.encoder pulls in whisper, which would make importing QbyT depend on
    # a heavy optional dependency.
    from models.encoder import ConformerEncoder

    encoder = ConformerEncoder(
        input_size=80,
        output_size=144,
        attention_heads=4,
        linear_units=576,
        num_blocks=6,
        dropout_rate=0.1,
        positional_dropout_rate=0.1,
        attention_dropout_rate=0.0,
        use_cnn_module=True,
        input_layer="conv2d",
        pos_enc_layer_type="rel_pos",
        selfattention_layer_type="rel_selfattn",
        cnn_module_kernel=3,
    )

    model = QbyT()

    import torch
    feat = torch.randn(32, 204, 80)
    feat_lengths = torch.tensor([feat.shape[1]], device=feat.device)
    text = torch.randint(0, 71, (32, 12))
    print(text.shape)

    encoder_out, _ = encoder(feat, feat_lengths)
    logits, text_logits = model(encoder_out, text)


    print(logits.shape)
    print(text_logits.shape)
