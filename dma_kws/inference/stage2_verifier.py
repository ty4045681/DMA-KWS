"""Stage II QbyT verifier.

Loads a trained Stage II checkpoint and scores Stage I candidate regions with
the QbyT query-by-text model, returning per-candidate detection scores.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator, Mapping, Sequence

_AMP_DTYPES = {
    "fp16": "float16",
    "float16": "float16",
    "bf16": "bfloat16",
    "bfloat16": "bfloat16",
}


def resolve_inference_amp(value: object) -> str | None:
    """Return ``fp16`` / ``bf16`` or ``None`` when mixed precision is off."""

    if value is None:
        return None
    text = str(value).strip().lower()
    if not text or text in {"off", "false", "0", "none", "32", "fp32", "float32"}:
        return None
    if text in _AMP_DTYPES:
        return "fp16" if _AMP_DTYPES[text] == "float16" else "bf16"
    raise ValueError(
        "prep.amp must be one of off, fp16, bf16 "
        f"(got {value!r})"
    )

from dma_kws.config import (
    FbankConfig,
    fbank_kwargs,
    get_eval_fbank_config,
    resolve_min_encoder_frames,
    resolve_stream_policy,
)
from dma_kws.inference.audio_utils import has_min_fbank_frames
from dma_kws.nn import (
    build_encoder,
    min_input_frames_for_encoder,
    run_encoder,
)
from dma_kws.stage1.candidates import KeywordCandidate
from dma_kws.stage2.fbank import FbankExtractor
from dma_kws.stage2.features import waveform_to_fbank


def _load_model_state(
    model,
    ckpt_path: str,
    load_fn,
    *,
    stream_policy=None,
    expected_qbyt_alignment=None,
):
    ckpt = load_fn(ckpt_path, map_location="cpu")
    if stream_policy is not None:
        from dma_kws.training.checkpoint_io import assert_stream_policy_matches

        assert_stream_policy_matches(ckpt, stream_policy, source=ckpt_path)
    from dma_kws.training.checkpoint_io import assert_qbyt_readout_version

    assert_qbyt_readout_version(
        ckpt,
        source=ckpt_path,
        expected_alignment=expected_qbyt_alignment,
    )
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
        amp: str | None = None,
    ) -> None:
        try:
            import torch
        except ImportError as exc:
            raise SystemExit(
                "Missing torch/torchaudio. Install CUDA PyTorch on the remote training machine first."
            ) from exc

        self._torch = torch
        self._demo_cfg = dict(demo_cfg)
        self._device = device
        self._amp = resolve_inference_amp(amp)
        self._fbank_kwargs = fbank_kwargs(fbank_cfg)
        self._fbank_extractor = FbankExtractor(**self._fbank_kwargs)
        stage2_encoder_dim = int(stage2_cfg.get("encoder_output_dim", 144))
        stream_policy = resolve_stream_policy(stage1_cfg)
        self._stream_policy = stream_policy
        adapter_cfg = stage2_cfg.get("phoneme_adapter", {}) or {}
        adapter_enabled = bool(adapter_cfg.get("enabled", False))
        from dma_kws.stage2.readout import resolve_qbyt_alignment

        qbyt_alignment = resolve_qbyt_alignment(stage2_cfg)
        self.qbyt_alignment = qbyt_alignment

        class Stage2Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = build_encoder(stage1_cfg, output_dim=stage2_encoder_dim)
                # Mirrors Stage2LightningModule: the checkpoint is loaded with
                # strict=True below, so an architecture that disagrees with the
                # training-time one fails here instead of scoring silently wrong.
                if adapter_enabled:
                    from dma_kws.phoneme_adapter.module import build_phoneme_adapter

                    self.adapter = build_phoneme_adapter(
                        adapter_cfg,
                        input_dim=stage2_encoder_dim,
                        vocab_size=vocab_size,
                        causal=bool(stage1_cfg.get("causal", False)),
                    )
                    qbyt_input_dim = self.adapter.output_dim
                else:
                    self.adapter = None
                    qbyt_input_dim = stage2_encoder_dim
                from dma_kws.stage2.model_factory import build_qbyt

                self.qbyt = build_qbyt(
                    stage2_cfg,
                    input_dim=qbyt_input_dim,
                    vocab_size=vocab_size,
                )

            def _encode_and_score(
                self,
                feats,
                feat_lengths,
                anchors,
                anchor_lengths,
            ):
                # Inference always runs at the deployment operating point.
                encoder_out, encoder_mask = run_encoder(
                    self.encoder,
                    feats,
                    feat_lengths,
                    policy=stream_policy,
                    mode="eval",
                )
                encoder_lens = encoder_mask.squeeze(1).sum(1)
                speech = encoder_out
                if self.adapter is not None:
                    # Inference never scores the CTC posteriors; ctc_lo stays in
                    # the state dict only so training checkpoints load strictly.
                    speech, _ = self.adapter(
                        encoder_out, encoder_mask, with_log_probs=False
                    )
                return self.qbyt(
                    speech,
                    anchors,
                    speech_lengths=encoder_lens,
                    text_lengths=anchor_lengths,
                )

            def forward(self, feats, feat_lengths, anchors, anchor_lengths):
                logits, _ = self._encode_and_score(
                    feats,
                    feat_lengths,
                    anchors,
                    anchor_lengths,
                )
                return torch.sigmoid(logits)

        model = Stage2Model()
        try:
            _load_model_state(
                model,
                stage2_ckpt,
                torch.load,
                stream_policy=stream_policy,
                # Inference must never score a semantically stale alignment.
                expected_qbyt_alignment=qbyt_alignment,
            )
        except RuntimeError as exc:
            raise SystemExit(
                f"Checkpoint {stage2_ckpt} is incompatible with the model architecture: {exc}"
            ) from exc
        self._model = model.to(device).eval()
        # A span can be long enough to produce fbank frames and still subsample
        # away to nothing, so the guard is expressed in encoder frames and
        # converted using the encoder that will actually run.
        self._min_fbank_frames = min_input_frames_for_encoder(
            model.encoder, resolve_min_encoder_frames(self._demo_cfg)
        )

    @property
    def amp(self) -> str | None:
        """Requested mixed-precision mode, or ``None`` for fp32."""
        return getattr(self, "_amp", None)

    @contextmanager
    def _inference_amp(self) -> Iterator[None]:
        torch = self._torch
        amp = getattr(self, "_amp", None)
        if amp is None or getattr(self._device, "type", None) != "cuda":
            yield
            return
        dtype = torch.float16 if amp == "fp16" else torch.bfloat16
        with torch.autocast(device_type="cuda", dtype=dtype):
            yield

    @property
    def fbank_extractor(self) -> FbankExtractor:
        return self._fbank_extractor

    @property
    def fbank_kwargs(self) -> dict:
        return dict(self._fbank_kwargs)

    @property
    def stream_policy(self):
        """Resolved streaming operating point used for every score."""
        return self._stream_policy

    @property
    def min_fbank_frames(self) -> int:
        """Shortest fbank input this encoder can score, in frames."""
        return self._min_fbank_frames

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
        try:
            amp = resolve_inference_amp(prep.get("amp"))
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc

        return cls(
            stage1_cfg=stage1_cfg,
            stage2_cfg=stage2_cfg,
            demo_cfg=demo_cfg,
            fbank_cfg=fbank_cfg,
            stage2_ckpt=stage2_ckpt,
            vocab_size=len(tokenizer.symbol_table),
            device=device,
            amp=amp,
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
        if len(feats) != len(keyword_ids_batch):
            raise ValueError("feats and keyword_ids_batch must have the same length")
        if any(not ids for ids in keyword_ids_batch):
            raise ValueError("keyword ids must be non-empty")
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
        with torch.no_grad(), self._inference_amp():
            scores = self._model(
                padded_feats.to(self._device),
                feat_lengths.to(self._device),
                anchors.to(self._device),
                anchor_lengths.to(self._device),
            )
        return [float(value) for value in scores.reshape(-1).cpu()]

    def decode_phoneme_feats(self, feats: Sequence) -> list[list[int]]:
        """Greedily decode phoneme ids from full-clip fbank features.

        This uses the encoder and phoneme adapter already loaded for Stage II
        scoring, so checkpoint weights, fbank settings, and streaming policy are
        identical to :meth:`score_clip_feats`.
        """
        torch = self._torch
        if not feats:
            return []

        adapter = getattr(self._model, "adapter", None)
        if adapter is None:
            raise RuntimeError(
                "Phoneme PER requires stage2.phoneme_adapter.enabled=true and a "
                "Stage II checkpoint containing adapter weights"
            )

        from torch.nn.utils.rnn import pad_sequence

        from dma_kws.metrics import collapse_ctc

        padded_feats = pad_sequence(list(feats), batch_first=True, padding_value=0)
        feat_lengths = torch.tensor(
            [feat.size(0) for feat in feats],
            dtype=torch.long,
        )
        with torch.no_grad(), self._inference_amp():
            encoder_out, encoder_mask = run_encoder(
                self._model.encoder,
                padded_feats.to(self._device),
                feat_lengths.to(self._device),
                policy=self._stream_policy,
                mode="eval",
            )
            _hidden, log_probs = adapter(
                encoder_out,
                encoder_mask,
                with_log_probs=True,
            )

        if log_probs is None:
            raise RuntimeError("Phoneme adapter did not return CTC log-probabilities")

        frame_lengths = (
            encoder_mask.squeeze(1)
            .sum(dim=1)
            .to(dtype=torch.long)
            .detach()
            .cpu()
            .tolist()
        )
        frame_ids = log_probs.argmax(dim=-1).detach().cpu().tolist()
        blank_id = int(adapter.blank_id)
        return [
            collapse_ctc(ids[: int(length)], blank_id=blank_id)
            for ids, length in zip(frame_ids, frame_lengths)
        ]

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
        min_stage2_fbank_frames = self._min_fbank_frames
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
            with torch.no_grad(), self._inference_amp():
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
