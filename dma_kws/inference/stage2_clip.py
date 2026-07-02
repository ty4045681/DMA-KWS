"""Stage II-only keyword clip inference.

Treats the full input audio as a single keyword candidate and scores it with
the Stage II QbyT verifier, bypassing Stage I locator models entirely.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Mapping

from dma_kws.audio import load_audio
from dma_kws.g2p import make_g2p, text_to_phonemes
from dma_kws.stage1.candidates import KeywordCandidate
from dma_kws.tokenizer import load_char_tokenizer, tokenize_phoneme_string

if TYPE_CHECKING:
    from dma_kws.inference.stage2_verifier import Stage2Verifier


class Stage2ClipRunner:
    """Score a pre-cropped keyword clip with Stage II only."""

    def __init__(
        self,
        *,
        verifier: "Stage2Verifier",
        tokenizer,
        demo_cfg: Mapping[str, Any],
        sample_rate: int,
    ) -> None:
        self._verifier = verifier
        self._tokenizer = tokenizer
        self._demo_cfg = dict(demo_cfg)
        self._sample_rate = int(sample_rate)
        self._g2p = make_g2p()

    @classmethod
    def from_config(cls, config: Mapping[str, Any], prep: Mapping[str, Any], device) -> "Stage2ClipRunner":
        from dma_kws.config import get_tokenizer_config
        from dma_kws.inference.stage2_verifier import Stage2Verifier
        from dma_kws.pathing import resolve_dict_path

        stage1_cfg = config.get("stage1")
        if not isinstance(stage1_cfg, Mapping):
            raise ValueError("Config section 'stage1' must be a mapping")
        demo_cfg = config.get("demo")
        if not isinstance(demo_cfg, Mapping):
            demo_cfg = {}

        tokenizer_cfg = get_tokenizer_config(dict(config))
        dict_path = resolve_dict_path(config)
        split_with_space = tokenizer_cfg.get("split_with_space", " ")
        tokenizer = load_char_tokenizer(dict_path, split_with_space=split_with_space)
        verifier = Stage2Verifier.from_config(config, prep, device)

        return cls(
            verifier=verifier,
            tokenizer=tokenizer,
            demo_cfg=demo_cfg,
            sample_rate=int(stage1_cfg.get("sample_rate", 16000)),
        )

    def run(self, audio_path: str, keyword: str) -> dict:
        """Run Stage II verification on the full audio clip."""
        keyword_phonemes = text_to_phonemes(self._g2p, keyword)
        keyword_g2p_text = " ".join(keyword_phonemes)
        keyword_ids = tokenize_phoneme_string(self._tokenizer, keyword_g2p_text)

        waveform, sample_rate = load_audio(audio_path, sample_rate=self._sample_rate)
        end_sec = waveform.size(1) / sample_rate
        candidates = [
            KeywordCandidate(
                start_sec=0.0,
                end_sec=end_sec,
                stage1_score=0.0,
                phonemes=[],
            )
        ]
        stage2_scores = self._verifier.verify_candidates(
            waveform,
            sample_rate,
            keyword_ids,
            candidates,
        )

        threshold = float(self._demo_cfg.get("qbyt_threshold", 0.5))
        qbyt_score = max((item["qbyt_score"] for item in stage2_scores), default=0.0)
        return {
            "audio": audio_path,
            "keyword": keyword,
            "keyword_phonemes": keyword_phonemes,
            "clip_span_sec": {"start_sec": 0.0, "end_sec": end_sec},
            "qbyt_score": qbyt_score,
            "threshold": threshold,
            "detected": qbyt_score >= threshold,
            "skipped": not stage2_scores,
        }
