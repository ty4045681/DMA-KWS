"""Stage II QbyT verifier.

Loads a trained Stage II checkpoint and scores Stage I candidate regions with
the QbyT query-by-text model, returning per-candidate detection scores.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from dma_kws.config import (
    FbankConfig,
    fbank_kwargs,
    get_eval_fbank_config,
    resolve_min_encoder_frames,
    resolve_stream_policy,
)
from dma_kws.inference.audio_utils import has_min_fbank_frames
from dma_kws.nn import build_encoder, min_input_frames_for_encoder, run_encoder
from dma_kws.pathing import load_qbyt_class
from dma_kws.stage1.candidates import KeywordCandidate
from dma_kws.stage2.fbank import FbankExtractor
from dma_kws.stage2.features import waveform_to_fbank


def _load_model_state(
    model,
    ckpt_path: str,
    load_fn,
    *,
    stream_policy=None,
    allow_legacy_qbyt_readout: bool = False,
    expected_qbyt_readout_mode: str | None = None,
    expected_qbyt_readout_temperature: float | None = None,
):
    ckpt = load_fn(ckpt_path, map_location="cpu")
    if stream_policy is not None:
        from dma_kws.training.checkpoint_io import assert_stream_policy_matches

        assert_stream_policy_matches(ckpt, stream_policy, source=ckpt_path)
    from dma_kws.training.checkpoint_io import assert_qbyt_readout_version

    assert_qbyt_readout_version(
        ckpt,
        source=ckpt_path,
        allow_legacy=allow_legacy_qbyt_readout,
        expected_mode=expected_qbyt_readout_mode,
        expected_temperature=expected_qbyt_readout_temperature,
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
        stream_policy = resolve_stream_policy(stage1_cfg)
        self._stream_policy = stream_policy
        adapter_cfg = stage2_cfg.get("phoneme_adapter", {}) or {}
        adapter_enabled = bool(adapter_cfg.get("enabled", False))
        from dma_kws.stage2.readout import resolve_qbyt_readout

        qbyt_readout = resolve_qbyt_readout(stage2_cfg)
        qbyt_readout_mode = qbyt_readout.mode
        qbyt_readout_temperature = qbyt_readout.temperature
        self.qbyt_readout_mode = qbyt_readout_mode
        self.qbyt_readout_temperature = qbyt_readout_temperature

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
                self.qbyt = QbyT(
                    encoder_output_size=qbyt_input_dim,
                    num_embeds=vocab_size,
                    embed_dim=int(stage2_cfg.get("qbyt_embed_dim", 128)),
                    post_num_layers=int(stage2_cfg.get("qbyt_layers", 2)),
                    readout_mode=qbyt_readout_mode,
                    readout_temperature=qbyt_readout_temperature,
                )

            def _encode_and_score(
                self,
                feats,
                feat_lengths,
                anchors,
                anchor_lengths,
                *,
                include_readout_details=False,
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
                score_fn = (
                    self.qbyt.forward_with_readout_details
                    if include_readout_details
                    else self.qbyt
                )
                return score_fn(
                    speech,
                    anchors,
                    speech_lengths=encoder_lens,
                    text_lengths=anchor_lengths,
                )

            def forward_logits(self, feats, feat_lengths, anchors, anchor_lengths):
                """Return unsquashed utterance/completion logits for analysis."""
                from dma_kws.stage2.scoring import gather_completion_logits

                logits, seq_logits = self._encode_and_score(
                    feats,
                    feat_lengths,
                    anchors,
                    anchor_lengths,
                )
                completion_logits, completion_valid = gather_completion_logits(
                    seq_logits,
                    anchor_lengths,
                )
                return logits, completion_logits, completion_valid

            def forward_logits_with_readout_details(
                self, feats, feat_lengths, anchors, anchor_lengths
            ):
                """Return scalar heads plus analysis-only QbyT readout tensors."""
                (
                    logits,
                    completion_logits,
                    completion_valid,
                    _seq_logits,
                    readout_details,
                ) = self.forward_logits_with_position_details(
                    feats,
                    feat_lengths,
                    anchors,
                    anchor_lengths,
                )
                return logits, completion_logits, completion_valid, readout_details

            def forward_logits_with_position_details(
                self, feats, feat_lengths, anchors, anchor_lengths
            ):
                """Return scalar heads plus both analysis-only position tensors."""
                from dma_kws.stage2.scoring import gather_completion_logits

                logits, seq_logits, readout_details = self._encode_and_score(
                    feats,
                    feat_lengths,
                    anchors,
                    anchor_lengths,
                    include_readout_details=True,
                )
                completion_logits, completion_valid = gather_completion_logits(
                    seq_logits,
                    anchor_lengths,
                )
                return (
                    logits,
                    completion_logits,
                    completion_valid,
                    seq_logits,
                    readout_details,
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
                # Inference must never score a random or semantically stale
                # readout. The legacy flag is reserved for training warm starts.
                allow_legacy_qbyt_readout=False,
                expected_qbyt_readout_mode=qbyt_readout_mode,
                expected_qbyt_readout_temperature=qbyt_readout_temperature,
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

    def score_clip_feats_detailed(
        self,
        feats: Sequence,
        keyword_ids_batch: Sequence[Sequence[int]],
        *,
        include_eps_positions: bool = False,
        include_seq_positions: bool = False,
    ) -> list[dict[str, Any]]:
        """Return raw and probability scores for both Stage II heads.

        This analysis-only API deliberately leaves :meth:`score_clip_feats`
        unchanged, so enabling diagnostics cannot alter deployed decisions.
        Empty anchors have no completion score and are represented by ``None``.
        When ``include_eps_positions`` is true, ``eps_position_logits`` contains
        one raw shared-scorer logit per valid anchor token. It is ``None`` for a
        ``gru_last`` readout and an empty list for an empty EPS anchor.
        When ``include_seq_positions`` is true, ``seq_position_logits`` contains
        the progress/completion head output for every valid anchor token. The
        final value is required to agree with the separately exported completion
        scalar. Probability- and target-based diagnostics are derived by the
        evaluation script rather than mixed into this inference API.
        """

        torch = self._torch
        qbyt_readout_mode = getattr(self, "qbyt_readout_mode", "eps_mean")
        qbyt_readout_temperature = float(
            getattr(self, "qbyt_readout_temperature", 1.0)
        )
        if not feats:
            return []
        if len(feats) != len(keyword_ids_batch):
            raise ValueError("feats and keyword_ids_batch must have the same length")

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
            model_args = (
                padded_feats.to(self._device),
                feat_lengths.to(self._device),
                anchors.to(self._device),
                anchor_lengths.to(self._device),
            )
            if include_seq_positions:
                (
                    utt_logits,
                    completion_logits,
                    completion_valid,
                    seq_logits,
                    readout_details,
                ) = self._model.forward_logits_with_position_details(*model_args)
            elif include_eps_positions:
                (
                    utt_logits,
                    completion_logits,
                    completion_valid,
                    readout_details,
                ) = self._model.forward_logits_with_readout_details(*model_args)
                seq_logits = None
            else:
                utt_logits, completion_logits, completion_valid = (
                    self._model.forward_logits(*model_args)
                )
                seq_logits = None
                readout_details = None
            utt_scores = torch.sigmoid(utt_logits)
            completion_scores = torch.sigmoid(completion_logits)

        utt_logits = utt_logits.reshape(-1).detach().cpu()
        utt_scores = utt_scores.reshape(-1).detach().cpu()
        completion_logits = completion_logits.reshape(-1).detach().cpu()
        completion_scores = completion_scores.reshape(-1).detach().cpu()
        completion_valid = completion_valid.reshape(-1).detach().cpu()
        if seq_logits is not None:
            seq_logits = seq_logits.detach().cpu()

        position_logits = None
        position_mask = None
        if readout_details is not None:
            position_mask = readout_details.position_mask.detach().cpu()
            if readout_details.position_logits is not None:
                position_logits = readout_details.position_logits.detach().cpu()

        records = []
        for index, (
            utt_logit,
            utt_score,
            completion_logit,
            completion_score,
            is_valid,
        ) in enumerate(
            zip(
                utt_logits,
                utt_scores,
                completion_logits,
                completion_scores,
                completion_valid,
            )
        ):
            record = {
                "qbyt_logit": float(utt_logit),
                "qbyt_score": float(utt_score),
                "completion_logit": (
                    float(completion_logit) if bool(is_valid) else None
                ),
                "completion_score": (
                    float(completion_score) if bool(is_valid) else None
                ),
            }
            if include_seq_positions:
                expected_length = int(anchor_lengths[index])
                sample_seq_logits = seq_logits[index, :expected_length]
                if sample_seq_logits.numel() != expected_length:
                    raise RuntimeError(
                        "Sequence-position width does not match the anchor: "
                        f"sample={index}, logits={sample_seq_logits.numel()}, "
                        f"anchor={expected_length}"
                    )
                if not bool(torch.isfinite(sample_seq_logits).all()):
                    raise RuntimeError(
                        f"Sequence position logits contain non-finite values for sample {index}"
                    )
                if bool(is_valid) != bool(expected_length):
                    raise RuntimeError(
                        "Completion validity disagrees with the anchor length: "
                        f"sample={index}, valid={bool(is_valid)}, "
                        f"anchor={expected_length}"
                    )
                if bool(is_valid) and not bool(
                    torch.isclose(
                        sample_seq_logits[-1],
                        completion_logit,
                        rtol=1e-5,
                        atol=1e-6,
                    )
                ):
                    raise RuntimeError(
                        "Final sequence-position logit disagrees with completion: "
                        f"sample={index}, final={float(sample_seq_logits[-1])}, "
                        f"completion={float(completion_logit)}"
                    )
                record["seq_position_logits"] = [
                    float(value) for value in sample_seq_logits
                ]
            if include_eps_positions:
                if position_logits is None:
                    record["eps_position_logits"] = None
                else:
                    sample_mask = position_mask[index]
                    sample_logits = position_logits[index][sample_mask]
                    expected_length = int(anchor_lengths[index])
                    if sample_logits.numel() != expected_length:
                        raise RuntimeError(
                            "EPS position mask length does not match the anchor: "
                            f"sample={index}, mask={sample_logits.numel()}, "
                            f"anchor={expected_length}"
                        )
                    if not bool(torch.isfinite(sample_logits).all()):
                        raise RuntimeError(
                            f"EPS position logits contain non-finite values for sample {index}"
                        )
                    if not sample_logits.numel():
                        expected_logit = torch.zeros_like(utt_logit)
                    elif qbyt_readout_mode == "eps_mean":
                        expected_logit = sample_logits.mean()
                    elif qbyt_readout_mode == "eps_softmin":
                        values = sample_logits.float()
                        expected_logit = -qbyt_readout_temperature * (
                            torch.logsumexp(
                                -values / qbyt_readout_temperature,
                                dim=0,
                            )
                            - math.log(values.numel())
                        )
                    else:
                        raise RuntimeError(
                            "EPS position logits were exported for a non-EPS "
                            f"readout: {qbyt_readout_mode!r}"
                        )
                    if not bool(
                        torch.isclose(expected_logit, utt_logit, rtol=1e-5, atol=1e-6)
                    ):
                        raise RuntimeError(
                            "EPS position-logit aggregation disagrees with the utterance "
                            f"logit: sample={index}, mode={qbyt_readout_mode}, "
                            f"expected={float(expected_logit)}, "
                            f"utterance={float(utt_logit)}"
                        )
                    record["eps_position_logits"] = [
                        float(value) for value in sample_logits
                    ]
            records.append(record)
        return records

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
        with torch.no_grad():
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
