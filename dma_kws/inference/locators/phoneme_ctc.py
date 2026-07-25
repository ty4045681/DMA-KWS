"""Phoneme CTC keyword locator.

Wraps the Stage I phoneme-CTC encoder and the ContextGraph prefix beam search
in :mod:`dma_kws.stage1.streaming_search`. Given an audio file and a keyword,
it produces coarse :class:`KeywordCandidate` regions for Stage II verification.
"""

from __future__ import annotations

from typing import Any, Mapping

from dma_kws.audio import extract_fbank, load_audio
from dma_kws.config import get_tokenizer_config
from dma_kws.g2p import make_g2p, text_to_phonemes
from dma_kws.nn import build_encoder
from dma_kws.pathing import ensure_qbyt_on_path, resolve_dict_path
from dma_kws.stage1.candidates import KeywordCandidate
from dma_kws.stage1.streaming_search import (
    DEFAULT_FRAME_SHIFT_SEC,
    build_keyword_context_graph,
    decode_keyword_candidates,
)
from dma_kws.tokenizer import load_char_tokenizer, tokenize_phoneme_string


def _load_model_state(model, ckpt_path: str, load_fn):
    ckpt = load_fn(ckpt_path, map_location="cpu")
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=True)
    return model


class PhonemeCtcLocator:
    """Locate keyword regions with a Stage I phoneme-CTC encoder."""

    def __init__(
        self,
        *,
        config: Mapping[str, Any],
        prep: Mapping[str, Any] | None = None,
        device: Any = None,
    ) -> None:
        try:
            import torch
        except ImportError as exc:
            raise SystemExit(
                "Missing torch/torchaudio. Install CUDA PyTorch on the remote training machine first."
            ) from exc

        prep = prep or {}
        stage1_cfg = config.get("stage1")
        if not isinstance(stage1_cfg, Mapping):
            raise ValueError("Config section 'stage1' must be a mapping")
        demo_cfg = config.get("demo")
        if not isinstance(demo_cfg, Mapping):
            demo_cfg = {}

        stage1_ckpt = str(prep.get("stage1_ckpt", ""))
        if not stage1_ckpt:
            raise SystemExit("prep.stage1_ckpt is required for the phoneme_ctc locator")

        tokenizer_cfg = get_tokenizer_config(dict(config))
        dict_path = resolve_dict_path(config)
        split_with_space = tokenizer_cfg.get("split_with_space", " ")
        tokenizer = load_char_tokenizer(dict_path, split_with_space=split_with_space)

        ensure_qbyt_on_path()
        from models.ctc import CTC

        self._torch = torch
        self._stage1_cfg = dict(stage1_cfg)
        self._demo_cfg = dict(demo_cfg)
        self._tokenizer = tokenizer
        self._device = device
        self._g2p = make_g2p()

        self._sample_rate = int(stage1_cfg.get("sample_rate", 16000))
        self._num_mel_bins = int(stage1_cfg.get("input_dim", 80))
        encoder_dim = int(stage1_cfg.get("encoder_output_dim", 144))
        # The CTC head is sized by the phoneme vocabulary, so it has to come from
        # the same dict the checkpoint was trained with.
        vocab_size = len(tokenizer.symbol_table)

        class Stage1Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = build_encoder(stage1_cfg, output_dim=encoder_dim)
                self.ctc = CTC(vocab_size, encoder_dim, blank_id=0)

            def forward(self, feats, feat_lengths):
                encoder_out, encoder_mask = self.encoder(feats, feat_lengths)
                return self.ctc.log_softmax(encoder_out), encoder_mask

        model = Stage1Model()
        try:
            _load_model_state(model, stage1_ckpt, torch.load)
        except RuntimeError as exc:
            raise SystemExit(
                f"Checkpoint {stage1_ckpt} is incompatible with the model architecture: {exc}"
            ) from exc
        self._model = model.to(device).eval()

        # Populated by ``locate`` so the pipeline can surface diagnostics.
        self.last_keyword_phonemes: list[str] = []
        self.last_decoded_phonemes: list[str] = []

    @classmethod
    def from_config(cls, config: Mapping[str, Any], prep: Mapping[str, Any], device) -> "PhonemeCtcLocator":
        """Build from a resolved config dict (compatibility shim)."""
        return cls(config=config, prep=prep, device=device)

    def locate(self, audio_path: str, keyword: str) -> list[KeywordCandidate]:
        torch = self._torch
        tokenizer = self._tokenizer

        keyword_phonemes = text_to_phonemes(self._g2p, keyword)
        keyword_g2p_text = " ".join(keyword_phonemes)
        keyword_ids = tokenize_phoneme_string(tokenizer, keyword_g2p_text)
        self.last_keyword_phonemes = keyword_phonemes

        waveform, sample_rate = load_audio(audio_path, sample_rate=self._sample_rate)
        feat = extract_fbank(
            waveform,
            num_mel_bins=self._num_mel_bins,
            sample_rate=sample_rate,
            dither=0.0,
        ).unsqueeze(0)
        feat_lengths = torch.tensor([feat.size(1)], dtype=torch.long)

        with torch.no_grad():
            log_probs, mask = self._model(feat.to(self._device), feat_lengths.to(self._device))
        encoder_lens = mask.squeeze(1).sum(1).to(torch.long)

        context_graph = build_keyword_context_graph(keyword_ids, tokenizer.symbol_table)
        candidates = decode_keyword_candidates(
            log_probs,
            encoder_lens,
            context_graph,
            frame_shift_sec=DEFAULT_FRAME_SHIFT_SEC,
            margin_sec=float(self._demo_cfg.get("stage1_candidate_margin_sec", 0.15)),
            symbol_table=tokenizer.symbol_table,
        )

        from models.search import ctc_greedy_search

        greedy_results = ctc_greedy_search(log_probs, encoder_lens, blank_id=0)
        self.last_decoded_phonemes = tokenizer.ids2tokens(greedy_results[0].tokens)

        return candidates
