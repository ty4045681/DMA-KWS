"""Stage II-only keyword clip inference.

Treats the full input audio as a single keyword candidate and scores it with
the Stage II QbyT verifier, bypassing Stage I locator models entirely.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Mapping, Sequence

from dma_kws.audio import load_audio
from dma_kws.g2p import make_g2p, text_to_phonemes
from dma_kws.stage1.candidates import KeywordCandidate
from dma_kws.tokenizer import (
    load_char_tokenizer,
    tokenize_phoneme_string,
    unsupported_phones,
)

if TYPE_CHECKING:
    from dma_kws.inference.stage2_verifier import Stage2Verifier

__all__ = ["ClipFeatureDataset", "Stage2ClipRunner", "collate_clip_feature_batch"]


def _parse_manifest_phonemes(
    value: object,
    *,
    field_name: str,
) -> list[str]:
    """Parse an explicitly supplied manifest phoneme override.

    CSV manifests normally use a space-separated ARPAbet string, while JSONL
    manifests may use either that string or a JSON array of strings. Invalid or
    empty values fail loudly instead of silently falling back to G2P or being
    tokenized as ``<unk>``.
    """
    parsed: object = value
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            raise ValueError(f"{field_name} must not be empty")
        if stripped.startswith("["):
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{field_name} must be a space-separated ARPAbet string or "
                    "a JSON array of strings"
                ) from exc
        else:
            parsed = stripped.split()

    if not isinstance(parsed, (list, tuple)) or not parsed:
        raise ValueError(
            f"{field_name} must be a non-empty ARPAbet string or sequence of strings"
        )
    if not all(isinstance(phone, str) and phone.strip() for phone in parsed):
        raise ValueError(f"{field_name} must contain only non-empty strings")

    phonemes = [phone.strip() for phone in parsed]
    unsupported = unsupported_phones(phonemes)
    if unsupported:
        raise ValueError(
            f"{field_name} contains unsupported phonemes: {', '.join(unsupported)}"
        )
    return phonemes


class ClipFeatureDataset:
    """Load clips and compute full-clip fbank features.

    Waveform padding is part of the encoder input, so the encoder-length guard
    is evaluated after padding. It is not a source-duration quality filter: an
    otherwise too-short source clip can become scoreable after padding.
    """

    def __init__(
        self,
        *,
        audio_paths: Sequence[str],
        sample_rate: int,
        fbank_extractor,
        fbank_kwargs: Mapping[str, Any],
        min_fbank_frames: int,
        left_padding_ms: int = 0,
        right_padding_ms: int = 0,
    ) -> None:
        left_padding_ms = int(left_padding_ms)
        right_padding_ms = int(right_padding_ms)
        if left_padding_ms < 0 or right_padding_ms < 0:
            raise ValueError("left_padding_ms and right_padding_ms must be >= 0")
        self._audio_paths = list(audio_paths)
        self._sample_rate = int(sample_rate)
        self._extractor = fbank_extractor
        self._fbank_kwargs = dict(fbank_kwargs)
        self._min_fbank_frames = int(min_fbank_frames)
        self._left_padding_ms = left_padding_ms
        self._right_padding_ms = right_padding_ms

    def __len__(self) -> int:
        return len(self._audio_paths)

    def __getitem__(self, index: int):
        from dma_kws.inference.audio_utils import has_min_fbank_frames
        from dma_kws.stage2.features import waveform_to_fbank

        waveform, sample_rate = load_audio(
            self._audio_paths[index], sample_rate=self._sample_rate
        )
        # Keep result spans in source-audio coordinates; padding is synthetic
        # context used only by the model input.
        end_sec = waveform.size(1) / sample_rate
        waveform, sample_rate = self._extractor.prepare_waveform(waveform, sample_rate)
        left_samples = round(sample_rate * self._left_padding_ms / 1000)
        right_samples = round(sample_rate * self._right_padding_ms / 1000)
        if left_samples or right_samples:
            from torch.nn.functional import pad

            waveform = pad(waveform, (left_samples, right_samples))
        # This guard prevents an input from subsampling to zero encoder frames;
        # it intentionally validates the transformed model input.
        if not has_min_fbank_frames(
            waveform.size(1),
            min_frames=self._min_fbank_frames,
            sample_rate=sample_rate,
            frame_length_ms=float(self._fbank_kwargs["frame_length"]),
            frame_shift_ms=float(self._fbank_kwargs["frame_shift"]),
            snip_edges=bool(self._fbank_kwargs["snip_edges"]),
        ):
            return index, None, end_sec
        feat = waveform_to_fbank(
            waveform,
            sample_rate=sample_rate,
            extractor=self._extractor,
            **self._fbank_kwargs,
        )
        return index, feat, end_sec


def collate_clip_feature_batch(batch):
    """Keep variable-length clip feature records as a list for runner batching."""
    return batch


# Compatibility aliases for any out-of-tree callers that imported the old
# private names before the feature-loading path became shared by PER evaluation.
_ClipFeatureDataset = ClipFeatureDataset
_list_collate = collate_clip_feature_batch


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

    @property
    def stream_policy(self):
        """Resolved streaming operating point used for every score."""
        return self._verifier.stream_policy

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
        return self._clip_result(
            audio_path,
            keyword,
            keyword_phonemes,
            end_sec,
            qbyt_score,
            threshold,
            skipped=not stage2_scores,
        )

    def run_batch(
        self,
        rows: Sequence[Mapping[str, Any]],
        *,
        batch_size: int = 64,
        num_workers: int = 0,
        left_padding_ms: int = 0,
        right_padding_ms: int = 0,
        include_score_details: bool = False,
        include_eps_positions: bool = False,
        include_seq_positions: bool = False,
    ) -> list[dict]:
        """Run Stage II verification on many clips with batched GPU scoring.

        ``rows`` are mappings with ``audio_path`` and ``keyword`` keys. An
        optional ``keyword_phonemes`` value overrides G2P for that row; it may
        be a space-separated ARPAbet string or a sequence of strings. A
        ``text_variant_phonemes`` value similarly overrides diagnostic query
        G2P; otherwise a non-empty ``text_variant`` is converted automatically.
        Results are returned in the same order as ``rows`` and use the ``run``
        schema.
        Optional zero-valued waveform padding is applied in memory before fbank
        extraction; source audio files are not modified. The padding counts
        toward the minimum encoder-input length and can make a short clip
        scoreable. ``include_eps_positions`` requires score details and exposes
        one EPS readout logit per enrollment phoneme when that readout is active.
        ``include_seq_positions`` has the same requirement and exposes the
        progress/completion head logit at every enrollment position.
        """
        try:
            from torch.utils.data import DataLoader
        except ImportError as exc:
            raise SystemExit(
                "Missing torch/torchaudio. Install CUDA PyTorch on the remote training machine first."
            ) from exc

        if not include_score_details and (
            include_eps_positions or include_seq_positions
        ):
            raise ValueError(
                "Position-logit export requires include_score_details=true"
            )

        rows = list(rows)
        threshold = float(self._demo_cfg.get("qbyt_threshold", 0.5))
        auto_phoneme_cache: dict[str, list[str]] = {}

        def auto_phonemes(text: str) -> list[str]:
            if text not in auto_phoneme_cache:
                auto_phoneme_cache[text] = text_to_phonemes(self._g2p, text)
            return auto_phoneme_cache[text]

        keyword_cache: dict[
            tuple[str, tuple[str, ...] | None],
            tuple[list[str], list[int]],
        ] = {}
        row_keyword_keys: list[tuple[str, tuple[str, ...] | None]] = []
        row_text_variant_phonemes: list[list[str] | None] = []
        for row_index, row in enumerate(rows, start=1):
            keyword = str(row["keyword"])
            override = (
                _parse_manifest_phonemes(
                    row["keyword_phonemes"],
                    field_name=f"Manifest row {row_index} keyword_phonemes",
                )
                if "keyword_phonemes" in row
                else None
            )
            keyword_key = (
                keyword,
                tuple(override) if override is not None else None,
            )
            row_keyword_keys.append(keyword_key)
            if keyword_key not in keyword_cache:
                phonemes = (
                    list(override)
                    if override is not None
                    else auto_phonemes(keyword)
                )
                keyword_ids = tokenize_phoneme_string(
                    self._tokenizer, " ".join(phonemes)
                )
                if len(keyword_ids) != len(phonemes):
                    raise RuntimeError(
                        "Enrollment phoneme/token length mismatch: "
                        f"keyword={keyword!r}, phonemes={len(phonemes)}, "
                        f"token_ids={len(keyword_ids)}"
                    )
                keyword_cache[keyword_key] = (phonemes, keyword_ids)

            text_variant_override_raw = row.get("text_variant_phonemes")
            has_text_variant_override = text_variant_override_raw is not None and not (
                isinstance(text_variant_override_raw, str)
                and not text_variant_override_raw.strip()
            )
            text_variant_override = (
                _parse_manifest_phonemes(
                    text_variant_override_raw,
                    field_name=f"Manifest row {row_index} text_variant_phonemes",
                )
                if has_text_variant_override
                else None
            )
            text_variant_raw = row.get("text_variant")
            text_variant = (
                "" if text_variant_raw is None else str(text_variant_raw).strip()
            )
            row_text_variant_phonemes.append(
                list(text_variant_override)
                if text_variant_override is not None
                else (list(auto_phonemes(text_variant)) if text_variant else None)
            )

        dataset = ClipFeatureDataset(
            audio_paths=[row["audio_path"] for row in rows],
            sample_rate=self._sample_rate,
            fbank_extractor=self._verifier.fbank_extractor,
            fbank_kwargs=self._verifier.fbank_kwargs,
            min_fbank_frames=self._verifier.min_fbank_frames,
            left_padding_ms=left_padding_ms,
            right_padding_ms=right_padding_ms,
        )
        loader = DataLoader(
            dataset,
            batch_size=max(1, int(batch_size)),
            shuffle=False,
            num_workers=max(0, int(num_workers)),
            collate_fn=collate_clip_feature_batch,
        )

        results: list[dict | None] = [None] * len(rows)
        for batch in loader:
            feats = []
            keyword_ids_batch = []
            pending: list[tuple[int, float]] = []
            for index, feat, end_sec in batch:
                keyword = rows[index]["keyword"]
                phonemes, keyword_ids = keyword_cache[row_keyword_keys[index]]
                text_variant_phonemes = row_text_variant_phonemes[index]
                if feat is None:
                    results[index] = self._clip_result(
                        rows[index]["audio_path"],
                        keyword,
                        phonemes,
                        end_sec,
                        0.0,
                        threshold,
                        skipped=True,
                        text_variant_phonemes=text_variant_phonemes,
                        score_details=(
                            {
                                "qbyt_logit": None,
                                "completion_logit": None,
                                "completion_score": None,
                                **(
                                    {"eps_position_logits": None}
                                    if include_eps_positions
                                    else {}
                                ),
                                **(
                                    {"seq_position_logits": None}
                                    if include_seq_positions
                                    else {}
                                ),
                            }
                            if include_score_details
                            else None
                        ),
                    )
                    continue
                feats.append(feat)
                keyword_ids_batch.append(keyword_ids)
                pending.append((index, end_sec))
            if not feats:
                continue
            if include_score_details:
                position_kwargs = {}
                if include_eps_positions:
                    position_kwargs["include_eps_positions"] = True
                if include_seq_positions:
                    position_kwargs["include_seq_positions"] = True
                detailed_scores = self._verifier.score_clip_feats_detailed(
                    feats,
                    keyword_ids_batch,
                    **position_kwargs,
                )
            else:
                detailed_scores = [
                    {"qbyt_score": score}
                    for score in self._verifier.score_clip_feats(feats, keyword_ids_batch)
                ]
            for (index, end_sec), score_details in zip(pending, detailed_scores):
                keyword = rows[index]["keyword"]
                phonemes, _ = keyword_cache[row_keyword_keys[index]]
                results[index] = self._clip_result(
                    rows[index]["audio_path"],
                    keyword,
                    phonemes,
                    end_sec,
                    float(score_details["qbyt_score"]),
                    threshold,
                    skipped=False,
                    text_variant_phonemes=row_text_variant_phonemes[index],
                    score_details=score_details if include_score_details else None,
                )
        return results

    @staticmethod
    def _clip_result(
        audio_path: str,
        keyword: str,
        keyword_phonemes: list[str],
        end_sec: float,
        qbyt_score: float,
        threshold: float,
        *,
        skipped: bool,
        start_sec: float = 0.0,
        text_variant_phonemes: Sequence[str] | None = None,
        score_details: Mapping[str, Any] | None = None,
    ) -> dict:
        result = {
            "audio": audio_path,
            "keyword": keyword,
            "keyword_phonemes": keyword_phonemes,
            "clip_span_sec": {"start_sec": start_sec, "end_sec": end_sec},
            "qbyt_score": qbyt_score,
            "threshold": threshold,
            "detected": qbyt_score >= threshold,
            "skipped": skipped,
        }
        if text_variant_phonemes is not None:
            result["text_variant_phonemes"] = list(text_variant_phonemes)
        if score_details is not None:
            result.update(
                {
                    "qbyt_logit": score_details.get("qbyt_logit"),
                    "completion_logit": score_details.get("completion_logit"),
                    "completion_score": score_details.get("completion_score"),
                }
            )
            if "eps_position_logits" in score_details:
                result["eps_position_logits"] = score_details["eps_position_logits"]
            if "seq_position_logits" in score_details:
                result["seq_position_logits"] = score_details["seq_position_logits"]
        return result

    def run_file_windows(
        self,
        audio_path: str,
        keyword: str,
        *,
        window_sec: float,
        hop_sec: float,
    ) -> list[dict]:
        """Run Stage-II verification on sliding windows of a long audio file.

        The file is divided into overlapping windows of length ``window_sec``
        advanced by ``hop_sec``. Each window is scored independently and a result
        dict in the same format as :meth:`run` is returned.
        """
        from dma_kws.inference.audio_utils import has_min_fbank_frames
        from dma_kws.stage2.features import waveform_to_fbank

        keyword_phonemes = text_to_phonemes(self._g2p, keyword)
        keyword_ids = tokenize_phoneme_string(
            self._tokenizer, " ".join(keyword_phonemes)
        )
        threshold = float(self._demo_cfg.get("qbyt_threshold", 0.5))
        min_stage2_fbank_frames = self._verifier.min_fbank_frames

        waveform, sample_rate = load_audio(audio_path, sample_rate=self._sample_rate)
        waveform, sample_rate = self._verifier.fbank_extractor.prepare_waveform(
            waveform, sample_rate
        )
        total_samples = waveform.size(1)
        window_samples = int(window_sec * sample_rate)
        hop_samples = int(hop_sec * sample_rate)

        if window_samples <= 0 or hop_samples <= 0 or total_samples < window_samples:
            return []

        fbank_kwargs = self._verifier.fbank_kwargs
        feats = []
        spans: list[tuple[float, float]] = []
        for start in range(0, total_samples - window_samples + 1, hop_samples):
            end = start + window_samples
            if not has_min_fbank_frames(
                end - start,
                min_frames=min_stage2_fbank_frames,
                sample_rate=sample_rate,
                frame_length_ms=float(fbank_kwargs["frame_length"]),
                frame_shift_ms=float(fbank_kwargs["frame_shift"]),
                snip_edges=bool(fbank_kwargs.get("snip_edges", True)),
            ):
                continue
            window_wave = waveform[:, start:end]
            feat = waveform_to_fbank(
                window_wave,
                sample_rate=sample_rate,
                extractor=self._verifier.fbank_extractor,
                **fbank_kwargs,
            )
            feats.append(feat)
            spans.append((start / sample_rate, end / sample_rate))

        if not feats:
            return []

        scores = self._verifier.score_clip_feats(feats, [keyword_ids] * len(feats))
        results = []
        for (start_sec, end_sec), score in zip(spans, scores):
            results.append(
                self._clip_result(
                    audio_path,
                    keyword,
                    keyword_phonemes,
                    end_sec=end_sec,
                    qbyt_score=float(score),
                    threshold=threshold,
                    skipped=False,
                    start_sec=start_sec,
                )
            )
        return results
