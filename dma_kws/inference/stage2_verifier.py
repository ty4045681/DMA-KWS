"""Stage II QbyT verifier.

Loads a trained Stage II checkpoint and scores Stage I candidate regions with
the QbyT query-by-text model, returning per-candidate detection scores.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from dma_kws.config import FbankConfig, fbank_kwargs, get_eval_fbank_config
from dma_kws.inference.audio_utils import has_min_fbank_frames
from dma_kws.nn import build_encoder
from dma_kws.pathing import load_qbyt_class
from dma_kws.stage1.candidates import KeywordCandidate
from dma_kws.stage2.fbank import FbankExtractor
from dma_kws.stage2.features import waveform_to_fbank


def _load_model_state(model, ckpt_path: str, load_fn):
    ckpt = load_fn(ckpt_path, map_location="cpu")
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=True)
    return model


class Stage2Verifier:
    """Verify Stage I candidates with the QbyT Stage II model."""

    def __init__(
        self,
        *,
        stage1_cfg: Mapping[str, Any],
        stage2_cfg: Mapping[str, Any],
        demo_cfg: Mapping[str, Any],
        fbank_cfg: FbankConfig,
        stage2_ckpt: str,
        vocab_size: int,
        device,
    ) -> None:
        try:
            import torch
        except ImportError as exc:
            raise SystemExit(
                "Missing torch/torchaudio. Install CUDA PyTorch on the remote training machine first."
            ) from exc

        QbyT = load_qbyt_class()

        self._torch = torch
        self._demo_cfg = dict(demo_cfg)
        self._device = device
        self._fbank_kwargs = fbank_kwargs(fbank_cfg)
        self._fbank_extractor = FbankExtractor(**self._fbank_kwargs)
        stage2_encoder_dim = int(stage2_cfg.get("encoder_output_dim", 144))

        class Stage2Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = build_encoder(stage1_cfg, output_dim=stage2_encoder_dim)
                self.qbyt = QbyT(
                    encoder_output_size=stage2_encoder_dim,
                    num_embeds=vocab_size,
                    embed_dim=int(stage2_cfg.get("qbyt_embed_dim", 128)),
                    post_num_layers=int(stage2_cfg.get("qbyt_layers", 2)),
                )

            def forward(self, feats, feat_lengths, anchors, anchor_lengths):
                encoder_out, encoder_mask = self.encoder(feats, feat_lengths)
                encoder_lens = encoder_mask.squeeze(1).sum(1)
                logits, _ = self.qbyt(
                    encoder_out,
                    anchors,
                    speech_lengths=encoder_lens,
                    text_lengths=anchor_lengths,
                )
                return torch.sigmoid(logits)

        model = Stage2Model()
        try:
            _load_model_state(model, stage2_ckpt, torch.load)
        except RuntimeError as exc:
            raise SystemExit(
                f"Checkpoint {stage2_ckpt} is incompatible with the model architecture: {exc}"
            ) from exc
        self._model = model.to(device).eval()

    @property
    def fbank_extractor(self) -> FbankExtractor:
        return self._fbank_extractor

    @property
    def fbank_kwargs(self) -> dict:
        return dict(self._fbank_kwargs)

    @classmethod
    def from_config(cls, config: Mapping[str, Any], prep: Mapping[str, Any], device) -> "Stage2Verifier":
        from dma_kws.pathing import resolve_dict_path
        from dma_kws.tokenizer import load_char_tokenizer

        stage1_cfg = config.get("stage1")
        if not isinstance(stage1_cfg, Mapping):
            raise ValueError("Config section 'stage1' must be a mapping")
        stage2_cfg = config.get("stage2")
        if not isinstance(stage2_cfg, Mapping):
            raise ValueError("Config section 'stage2' must be a mapping")
        demo_cfg = config.get("demo")
        if not isinstance(demo_cfg, Mapping):
            demo_cfg = {}

        stage2_ckpt = str(prep.get("stage2_ckpt", ""))
        if not stage2_ckpt:
            raise SystemExit("prep.stage2_ckpt is required for Stage II verification")

        fbank_cfg = get_eval_fbank_config(dict(config))
        # The text embedding table is sized by the phoneme vocabulary, so it has
        # to come from the same dict the checkpoint was trained with.
        tokenizer = load_char_tokenizer(resolve_dict_path(config))

        return cls(
            stage1_cfg=stage1_cfg,
            stage2_cfg=stage2_cfg,
            demo_cfg=demo_cfg,
            fbank_cfg=fbank_cfg,
            stage2_ckpt=stage2_ckpt,
            vocab_size=len(tokenizer.symbol_table),
            device=device,
        )

    def score_clip_feats(
        self,
        feats: Sequence,
        keyword_ids_batch: Sequence[Sequence[int]],
    ) -> list[float]:
        """Score a batch of full-clip fbank features against per-clip keyword ids."""
        torch = self._torch
        if not feats:
            return []
        from torch.nn.utils.rnn import pad_sequence

        padded_feats = pad_sequence(list(feats), batch_first=True, padding_value=0)
        feat_lengths = torch.tensor([f.size(0) for f in feats], dtype=torch.long)
        anchors = pad_sequence(
            [torch.tensor(list(ids), dtype=torch.long) for ids in keyword_ids_batch],
            batch_first=True,
            padding_value=0,
        )
        anchor_lengths = torch.tensor(
            [len(ids) for ids in keyword_ids_batch], dtype=torch.long
        )
        with torch.no_grad():
            scores = self._model(
                padded_feats.to(self._device),
                feat_lengths.to(self._device),
                anchors.to(self._device),
                anchor_lengths.to(self._device),
            )
        return [float(value) for value in scores.reshape(-1).cpu()]

    def verify_candidates(
        self,
        waveform,
        sample_rate: int,
        keyword_ids: Sequence[int],
        candidates: Sequence[KeywordCandidate],
    ) -> list[dict]:
        """Score each candidate region and return per-candidate results."""
        torch = self._torch
        anchor = torch.tensor([list(keyword_ids)], dtype=torch.long).to(self._device)
        anchor_lengths = torch.tensor([len(keyword_ids)], dtype=torch.long).to(self._device)
        min_stage2_fbank_frames = int(self._demo_cfg.get("min_stage2_fbank_frames", 7))
        waveform, sample_rate = self._fbank_extractor.prepare_waveform(
            waveform,
            sample_rate,
        )

        scores: list[dict] = []
        for candidate in candidates:
            start = max(0, int(candidate.start_sec * sample_rate))
            end = min(waveform.size(1), int(candidate.end_sec * sample_rate))
            if end <= start:
                continue
            if not has_min_fbank_frames(
                end - start,
                min_frames=min_stage2_fbank_frames,
                sample_rate=sample_rate,
                frame_length_ms=float(self._fbank_kwargs["frame_length"]),
                frame_shift_ms=float(self._fbank_kwargs["frame_shift"]),
                snip_edges=bool(self._fbank_kwargs["snip_edges"]),
            ):
                continue
            candidate_wave = waveform[:, start:end]
            candidate_feat = waveform_to_fbank(
                candidate_wave,
                sample_rate=sample_rate,
                extractor=self._fbank_extractor,
                **self._fbank_kwargs,
            ).unsqueeze(0)
            candidate_lens = torch.tensor([candidate_feat.size(1)], dtype=torch.long)
            with torch.no_grad():
                score = float(
                    self._model(
                        candidate_feat.to(self._device),
                        candidate_lens.to(self._device),
                        anchor,
                        anchor_lengths,
                    )
                    .cpu()
                    .item()
                )
            scores.append(
                {
                    "start_sec": candidate.start_sec,
                    "end_sec": candidate.end_sec,
                    "stage1_score": candidate.stage1_score,
                    "qbyt_score": score,
                }
            )
        return scores
