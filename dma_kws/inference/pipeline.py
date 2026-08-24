"""Two-stage keyword spotting pipeline.

Wires a Stage I :class:`KeywordLocator` to the Stage II
:class:`Stage2Verifier`, exposing a single ``run(audio_path, keyword)`` entry
point that returns the same result dict as the original two-stage demo.
"""

from __future__ import annotations

from typing import Any, Mapping

from dma_kws.audio import load_audio
from dma_kws.config import get_tokenizer_config
from dma_kws.g2p import make_g2p, text_to_phonemes
from dma_kws.inference.locator import KeywordLocator, build_locator
from dma_kws.inference.stage2_clip import parse_phoneme_sequence
from dma_kws.inference.stage2_verifier import Stage2Verifier
from dma_kws.pathing import resolve_dict_path
from dma_kws.tokenizer import load_char_tokenizer, tokenize_phoneme_string


class TwoStageKWSPipeline:
    """Locate keyword candidates and verify them with the Stage II model."""

    def __init__(
        self,
        *,
        locator: KeywordLocator,
        verifier: Stage2Verifier,
        tokenizer,
        demo_cfg: Mapping[str, Any],
        sample_rate: int,
    ) -> None:
        self._locator = locator
        self._verifier = verifier
        self._tokenizer = tokenizer
        self._demo_cfg = dict(demo_cfg)
        self._sample_rate = int(sample_rate)
        self._g2p = make_g2p()

    @property
    def stream_policy(self):
        """Resolved streaming operating point shared by Stage I and Stage II."""
        return self._verifier.stream_policy

    @classmethod
    def from_config(cls, config: Mapping[str, Any], prep: Mapping[str, Any], device) -> "TwoStageKWSPipeline":
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

        locator = build_locator(config, prep, device)
        verifier = Stage2Verifier.from_config(config, prep, device)

        return cls(
            locator=locator,
            verifier=verifier,
            tokenizer=tokenizer,
            demo_cfg=demo_cfg,
            sample_rate=int(stage1_cfg.get("sample_rate", 16000)),
        )

    def resolve_keyword_phonemes(
        self,
        keyword: str,
        keyword_phonemes: object | None = None,
        *,
        field_name: str = "keyword_phonemes",
    ) -> list[str]:
        """Return the effective enrollment sequence for one keyword.

        ``None`` retains automatic G2P. Any explicit value uses the same strict
        ARPAbet parser and vocabulary validation as Stage-II clip evaluation.
        """
        if keyword_phonemes is None:
            return text_to_phonemes(self._g2p, keyword)
        return parse_phoneme_sequence(keyword_phonemes, field_name=field_name)

    def run(
        self,
        audio_path: str,
        keyword: str,
        keyword_phonemes: object | None = None,
    ) -> dict:
        """Run the two-stage pipeline and return the demo-style result dict."""
        keyword_phonemes = self.resolve_keyword_phonemes(
            keyword,
            keyword_phonemes,
            field_name="keyword_phonemes",
        )
        keyword_g2p_text = " ".join(keyword_phonemes)
        keyword_ids = tokenize_phoneme_string(self._tokenizer, keyword_g2p_text)

        candidates = self._locator.locate(
            audio_path,
            keyword,
            keyword_phonemes=keyword_phonemes,
        )

        if candidates:
            waveform, sample_rate = load_audio(audio_path, sample_rate=self._sample_rate)
            stage2_scores = self._verifier.verify_candidates(
                waveform,
                sample_rate,
                keyword_ids,
                candidates,
            )
        else:
            stage2_scores = []

        threshold = float(self._demo_cfg.get("qbyt_threshold", 0.5))
        decoded_phonemes = list(getattr(self._locator, "last_decoded_phonemes", []))
        best_qbyt_score = max((item["qbyt_score"] for item in stage2_scores), default=0.0)
        return {
            "audio": audio_path,
            "keyword": keyword,
            "keyword_phonemes": keyword_phonemes,
            "decoded_phonemes": decoded_phonemes,
            "stage1_candidates": [candidate.__dict__ for candidate in candidates],
            "stage2_scores": stage2_scores,
            "threshold": threshold,
            "detected": any(item["qbyt_score"] >= threshold for item in stage2_scores),
            "best_qbyt_score": best_qbyt_score,
        }
