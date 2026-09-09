"""Stage II-only keyword clip inference.

Treats the full input audio as a single keyword candidate and scores it with
the Stage II QbyT verifier, bypassing Stage I locator models entirely.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Mapping, NamedTuple, Sequence

_THREAD_EXTRACTORS = threading.local()

from dma_kws.audio import load_audio
from dma_kws.g2p import make_g2p, text_to_phonemes
from dma_kws.inference.waveform_augmentation import WaveformTransform
from dma_kws.stage1.candidates import KeywordCandidate
from dma_kws.tokenizer import (
    load_char_tokenizer,
    tokenize_phoneme_string,
    unsupported_phones,
)

if TYPE_CHECKING:
    from dma_kws.inference.stage2_verifier import Stage2Verifier

__all__ = [
    "ClipFeatureDataset",
    "PreparedFileWindows",
    "Stage2ClipRunner",
    "collate_clip_feature_batch",
    "parse_phoneme_sequence",
]


def parse_phoneme_sequence(
    value: object,
    *,
    field_name: str,
) -> list[str]:
    """Parse an explicitly supplied phoneme override.

    CLI/config values and CSV manifests normally use a space-separated ARPAbet
    string, while JSONL manifests may use either that string or a JSON array of
    strings. Invalid or empty values fail loudly instead of silently falling
    back to G2P or being tokenized as ``<unk>``.
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
        waveform_transform: WaveformTransform | None = None,
        waveform_observer: Callable[[int, Any, int], None] | None = None,
        include_augmented_duration: bool = False,
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
        self._waveform_transform = waveform_transform
        self._waveform_observer = waveform_observer
        self._include_augmented_duration = bool(include_augmented_duration)

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
        if self._waveform_transform is not None:
            waveform = self._waveform_transform(index, waveform, sample_rate)
            transformed_shape = None if waveform is None else tuple(waveform.shape)
            if (
                waveform is None
                or len(transformed_shape) != 2
                or transformed_shape[0] != 1
                or transformed_shape[1] <= 0
            ):
                raise ValueError(
                    "waveform_transform must return a non-empty mono 2-D waveform "
                    f"with shape (1, samples), got {transformed_shape}"
                )
        if self._waveform_observer is not None:
            self._waveform_observer(index, waveform, sample_rate)
        augmented_duration_sec = waveform.size(1) / sample_rate
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
            result = (index, None, end_sec)
            if self._include_augmented_duration:
                return (*result, augmented_duration_sec)
            return result
        feat = waveform_to_fbank(
            waveform,
            sample_rate=sample_rate,
            extractor=self._extractor,
            **self._fbank_kwargs,
        )
        result = (index, feat, end_sec)
        if self._include_augmented_duration:
            return (*result, augmented_duration_sec)
        return result


def collate_clip_feature_batch(batch):
    """Keep variable-length clip feature records as a list for runner batching."""
    return batch


class PreparedFileWindows(NamedTuple):
    """Window features extracted from one audio file, ready to score."""

    audio_path: str
    keyword: str
    keyword_phonemes: list[str]
    keyword_ids: list[int]
    feats: list[Any]
    spans: list[tuple[int, float, float]]


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

    @property
    def verifier(self) -> "Stage2Verifier":
        """Loaded Stage II verifier backing every score this runner produces."""
        return self._verifier

    @property
    def sample_rate(self) -> int:
        """Sample rate every clip is loaded and resampled to."""
        return self._sample_rate

    def _score_feats_with_logits(
        self,
        feats: Sequence,
        keyword_ids_batch: Sequence[Sequence[int]],
    ) -> list[tuple[float | None, float]]:
        """Score clips while retaining raw logits when the verifier exposes them."""

        scorer = getattr(self._verifier, "score_clip_feats_with_logits", None)
        if scorer is not None:
            return list(scorer(feats, keyword_ids_batch))
        # Compatibility for small test doubles and third-party wrappers using
        # the pre-v6 probability-only verifier interface.
        return [
            (None, float(score))
            for score in self._verifier.score_clip_feats(feats, keyword_ids_batch)
        ]

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
        best = max(stage2_scores, key=lambda item: item["qbyt_score"], default=None)
        qbyt_score = float(best["qbyt_score"]) if best is not None else 0.0
        qbyt_raw_logit = (
            float(best["qbyt_raw_logit"])
            if best is not None and "qbyt_raw_logit" in best
            else None
        )
        return self._clip_result(
            audio_path,
            keyword,
            keyword_phonemes,
            end_sec,
            qbyt_score,
            threshold,
            skipped=not stage2_scores,
            qbyt_raw_logit=qbyt_raw_logit,
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
        ARPAbet parser and vocabulary validation as manifest overrides.
        """
        if keyword_phonemes is None:
            return text_to_phonemes(self._g2p, keyword)
        return parse_phoneme_sequence(keyword_phonemes, field_name=field_name)

    def enroll_phonemes(self, phonemes: Sequence[str]) -> list[int]:
        """Tokenize one enrollment sequence into QbyT anchor ids.

        The single enrollment path: batch scoring and offline diagnostics must
        not tokenize a keyword through two code paths, or a diagnostic could
        report on an anchor the deployed score never saw.
        """
        keyword_ids = tokenize_phoneme_string(self._tokenizer, " ".join(phonemes))
        if len(keyword_ids) != len(phonemes):
            raise RuntimeError(
                "Enrollment phoneme/token length mismatch: "
                f"phonemes={len(phonemes)}, token_ids={len(keyword_ids)}"
            )
        return keyword_ids

    def run_batch(
        self,
        rows: Sequence[Mapping[str, Any]],
        *,
        batch_size: int = 64,
        num_workers: int = 0,
        left_padding_ms: int = 0,
        right_padding_ms: int = 0,
        waveform_transform: WaveformTransform | None = None,
        waveform_observer: Callable[[int, Any, int], None] | None = None,
    ) -> list[dict]:
        """Run Stage II verification on many clips with batched GPU scoring.

        ``rows`` are mappings with ``audio_path`` and ``keyword`` keys. An
        optional ``keyword_phonemes`` value overrides G2P for that row; it may
        be a space-separated ARPAbet string or a sequence of strings.
        Results are returned in the same order as ``rows`` and use the ``run``
        schema.
        ``waveform_transform``, when supplied, is a pickle-friendly callable that
        receives ``(row_index, waveform, sample_rate)`` after fbank sample-rate
        preparation and must preserve a mono two-dimensional waveform; its
        sample count may change. A phased
        ``WaveformAugmentationPipeline`` satisfies the same callable contract. It
        runs before optional zero-valued padding. Source audio files are not
        modified. ``waveform_observer``, when supplied, receives the prepared and
        transformed waveform immediately before padding and cannot replace it.
        It is also called when ``waveform_transform`` is absent. The padding counts
        toward the minimum encoder-input length and can make a short clip scoreable.
        """
        try:
            from torch.utils.data import DataLoader
        except ImportError as exc:
            raise SystemExit(
                "Missing torch/torchaudio. Install CUDA PyTorch on the remote training machine first."
            ) from exc

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
        for row_index, row in enumerate(rows, start=1):
            keyword = str(row["keyword"])
            override = (
                parse_phoneme_sequence(
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
                keyword_cache[keyword_key] = (
                    phonemes,
                    self.enroll_phonemes(phonemes),
                )

        transform_enabled = waveform_transform is not None and bool(
            getattr(waveform_transform, "enabled", True)
        )
        reports_augmented_duration = transform_enabled and bool(
            getattr(waveform_transform, "changes_duration", True)
        )
        dataset = ClipFeatureDataset(
            audio_paths=[row["audio_path"] for row in rows],
            sample_rate=self._sample_rate,
            fbank_extractor=self._verifier.fbank_extractor,
            fbank_kwargs=self._verifier.fbank_kwargs,
            min_fbank_frames=self._verifier.min_fbank_frames,
            left_padding_ms=left_padding_ms,
            right_padding_ms=right_padding_ms,
            waveform_transform=waveform_transform,
            waveform_observer=waveform_observer,
            include_augmented_duration=True,
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
            pending: list[tuple[int, float, float]] = []
            for index, feat, end_sec, augmented_duration_sec in batch:
                keyword = rows[index]["keyword"]
                phonemes, keyword_ids = keyword_cache[row_keyword_keys[index]]
                if feat is None:
                    results[index] = self._clip_result(
                        rows[index]["audio_path"],
                        keyword,
                        phonemes,
                        end_sec,
                        0.0,
                        threshold,
                        skipped=True,
                        augmented_duration_sec=(
                            augmented_duration_sec
                            if reports_augmented_duration
                            else None
                        ),
                    )
                    continue
                feats.append(feat)
                keyword_ids_batch.append(keyword_ids)
                pending.append((index, end_sec, augmented_duration_sec))
            if not feats:
                continue
            scores = self._score_feats_with_logits(feats, keyword_ids_batch)
            for (
                index,
                end_sec,
                augmented_duration_sec,
            ), (raw_logit, score) in zip(pending, scores):
                keyword = rows[index]["keyword"]
                phonemes, _ = keyword_cache[row_keyword_keys[index]]
                results[index] = self._clip_result(
                    rows[index]["audio_path"],
                    keyword,
                    phonemes,
                    end_sec,
                    float(score),
                    threshold,
                    skipped=False,
                    qbyt_raw_logit=raw_logit,
                    augmented_duration_sec=(
                        augmented_duration_sec
                        if reports_augmented_duration
                        else None
                    ),
                )
        return results

    def enroll_keyword_set(
        self,
        prep: Mapping[str, Any],
        *,
        tokenizer_dict_path: str | Path,
    ):
        """Resolve ``prep.keyword_eval`` with this runner's tokenizer and G2P."""

        from dma_kws.inference.keyword_set import resolve_keyword_set

        return resolve_keyword_set(
            prep,
            self._tokenizer,
            g2p=self._g2p,
            tokenizer_dict_path=tokenizer_dict_path,
        )

    def _score_feats_multi_with_logits(
        self,
        feats: Sequence,
        query_ids: Sequence[Sequence[int]],
        *,
        query_batch_size: int,
    ) -> list[list[tuple[float | None, float]]]:
        scorer = getattr(self._verifier, "score_clip_feats_multi_with_logits", None)
        if scorer is None:
            raise RuntimeError(
                "verifier does not implement score_clip_feats_multi_with_logits"
            )
        scored = scorer(
            feats,
            query_ids,
            query_batch_size=query_batch_size,
        )
        if len(scored) != len(feats):
            raise RuntimeError(
                "Stage-II multi-query scorer returned a different number of "
                f"clips than inputs: expected={len(feats)}, actual={len(scored)}"
            )
        expected_queries = len(query_ids)
        for row in scored:
            if len(row) != expected_queries:
                raise RuntimeError(
                    "Stage-II multi-query scorer returned a different number of "
                    f"queries than enrolled: expected={expected_queries}, "
                    f"actual={len(row)}"
                )
        return scored

    def run_batch_multi(
        self,
        rows: Sequence[Mapping[str, Any]],
        keyword_set,
        *,
        batch_size: int = 64,
        num_workers: int = 0,
        left_padding_ms: int = 0,
        right_padding_ms: int = 0,
        waveform_transform: WaveformTransform | None = None,
        waveform_observer: Callable[[int, Any, int], None] | None = None,
        eval_protocol: str | None = None,
    ) -> list[dict]:
        """Score each source-audio row once against an enrolled keyword set."""

        from dma_kws.inference.keyword_set import (
            CLIP_EVAL_PROTOCOL,
            aggregate_query_scores,
            require_qbyt_threshold,
            scored_keyword_set_fields,
            skipped_keyword_set_fields,
        )

        try:
            from torch.utils.data import DataLoader
        except ImportError as exc:
            raise SystemExit(
                "Missing torch/torchaudio. Install CUDA PyTorch on the remote training machine first."
            ) from exc

        rows = list(rows)
        threshold = require_qbyt_threshold(
            self._demo_cfg.get("qbyt_threshold", 0.5),
            field="demo.qbyt_threshold",
        )
        protocol = str(eval_protocol or CLIP_EVAL_PROTOCOL)
        query_ids = keyword_set.query_token_ids()
        transform_enabled = waveform_transform is not None and bool(
            getattr(waveform_transform, "enabled", True)
        )
        reports_augmented_duration = transform_enabled and bool(
            getattr(waveform_transform, "changes_duration", True)
        )
        dataset = ClipFeatureDataset(
            audio_paths=[row["audio_path"] for row in rows],
            sample_rate=self._sample_rate,
            fbank_extractor=self._verifier.fbank_extractor,
            fbank_kwargs=self._verifier.fbank_kwargs,
            min_fbank_frames=self._verifier.min_fbank_frames,
            left_padding_ms=left_padding_ms,
            right_padding_ms=right_padding_ms,
            waveform_transform=waveform_transform,
            waveform_observer=waveform_observer,
            include_augmented_duration=True,
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
            pending: list[tuple[int, float, float]] = []
            for index, feat, end_sec, augmented_duration_sec in batch:
                if feat is None:
                    result = skipped_keyword_set_fields(
                        keyword_set,
                        threshold=threshold,
                    )
                    result["eval_protocol"] = protocol
                    result["audio_path"] = rows[index]["audio_path"]
                    result["clip_span_sec"] = {"start_sec": 0.0, "end_sec": end_sec}
                    if reports_augmented_duration:
                        result["augmented_duration_sec"] = float(augmented_duration_sec)
                    results[index] = result
                    continue
                feats.append(feat)
                pending.append((index, end_sec, augmented_duration_sec))
            if not feats:
                continue
            scored = self._score_feats_multi_with_logits(
                feats,
                query_ids,
                query_batch_size=keyword_set.query_batch_size,
            )
            if len(scored) != len(pending):
                raise RuntimeError(
                    "Stage-II multi-query scorer returned a different number of "
                    f"clips than inputs: expected={len(pending)}, actual={len(scored)}"
                )
            for (index, end_sec, augmented_duration_sec), query_scores in zip(
                pending,
                scored,
                strict=True,
            ):
                aggregation = aggregate_query_scores(
                    keyword_set,
                    query_scores,
                    threshold=threshold,
                    audio_id=rows[index]["audio_path"],
                    scored=True,
                )
                result = scored_keyword_set_fields(
                    keyword_set,
                    aggregation,
                    threshold=threshold,
                )
                result["eval_protocol"] = protocol
                result["audio_path"] = rows[index]["audio_path"]
                result["clip_span_sec"] = {"start_sec": 0.0, "end_sec": end_sec}
                if reports_augmented_duration:
                    result["augmented_duration_sec"] = float(augmented_duration_sec)
                results[index] = result
        if any(result is None for result in results):
            raise RuntimeError("Stage-II multi-query runner dropped one or more clips")
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
        qbyt_raw_logit: float | None = None,
        start_sec: float = 0.0,
        augmented_duration_sec: float | None = None,
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
        if qbyt_raw_logit is not None:
            result["qbyt_raw_logit"] = float(qbyt_raw_logit)
        if augmented_duration_sec is not None:
            result["augmented_duration_sec"] = float(augmented_duration_sec)
        return result

    def prepare_file_windows(
        self,
        audio_path: str,
        keyword: str,
        *,
        window_sec: float,
        hop_sec: float,
        keyword_phonemes: object | None = None,
        fbank_windows: str = "independent",
    ) -> PreparedFileWindows:
        """Load one file and extract hop-grid window features on CPU."""

        keyword_phonemes = self.resolve_keyword_phonemes(
            keyword,
            keyword_phonemes,
            field_name="keyword_phonemes",
        )
        keyword_ids = tokenize_phoneme_string(
            self._tokenizer, " ".join(keyword_phonemes)
        )
        if len(keyword_ids) != len(keyword_phonemes):
            raise RuntimeError(
                "Enrollment phoneme/token length mismatch: "
                f"keyword={keyword!r}, phonemes={len(keyword_phonemes)}, "
                f"token_ids={len(keyword_ids)}"
            )
        feats, spans = self.extract_file_window_features(
            audio_path,
            window_sec=window_sec,
            hop_sec=hop_sec,
            fbank_windows=fbank_windows,
        )
        return PreparedFileWindows(
            audio_path=audio_path,
            keyword=keyword,
            keyword_phonemes=keyword_phonemes,
            keyword_ids=keyword_ids,
            feats=feats,
            spans=spans,
        )

    def extract_file_window_features(
        self,
        audio_path: str,
        *,
        window_sec: float,
        hop_sec: float,
        fbank_windows: str = "independent",
    ) -> tuple[list[Any], list[tuple[int, float, float]]]:
        """Extract hop-grid window features without scoring or enrollment."""

        from dma_kws.inference.audio_utils import (
            has_min_fbank_frames,
            window_fbank_frame_span,
        )
        from dma_kws.stage2.features import waveform_to_fbank

        mode = str(fbank_windows or "independent").strip().lower()
        if mode not in {"independent", "file"}:
            raise ValueError(
                "fbank_windows must be 'independent' or 'file', "
                f"got {fbank_windows!r}"
            )
        min_stage2_fbank_frames = self._verifier.min_fbank_frames
        extractor = getattr(_THREAD_EXTRACTORS, "extractor", None)
        if extractor is None:
            from dma_kws.stage2.fbank import FbankExtractor

            extractor = FbankExtractor(**self._verifier.fbank_kwargs)
            _THREAD_EXTRACTORS.extractor = extractor

        waveform, sample_rate = load_audio(audio_path, sample_rate=self._sample_rate)
        waveform, sample_rate = extractor.prepare_waveform(waveform, sample_rate)
        total_samples = waveform.size(1)
        window_samples = int(window_sec * sample_rate)
        hop_samples = int(hop_sec * sample_rate)
        if window_samples <= 0 or hop_samples <= 0 or total_samples < window_samples:
            return [], []

        fbank_kwargs = self._verifier.fbank_kwargs
        frame_kwargs = {
            "sample_rate": sample_rate,
            "frame_length_ms": float(fbank_kwargs["frame_length"]),
            "frame_shift_ms": float(fbank_kwargs["frame_shift"]),
            "snip_edges": bool(fbank_kwargs.get("snip_edges", True)),
        }
        file_feat = None
        if mode == "file":
            file_feat = waveform_to_fbank(
                waveform,
                sample_rate=sample_rate,
                extractor=extractor,
                **fbank_kwargs,
            )

        feats = []
        spans: list[tuple[int, float, float]] = []
        for window_index, start in enumerate(
            range(0, total_samples - window_samples + 1, hop_samples)
        ):
            end = start + window_samples
            if not has_min_fbank_frames(
                end - start,
                min_frames=min_stage2_fbank_frames,
                **frame_kwargs,
            ):
                continue
            if mode == "file":
                start_frame, end_frame = window_fbank_frame_span(
                    start,
                    end,
                    **frame_kwargs,
                )
                if end_frame > file_feat.size(0):
                    raise RuntimeError(
                        "Full-file fbank is shorter than the sliced window: "
                        f"file_frames={file_feat.size(0)}, "
                        f"window=[{start_frame}, {end_frame})"
                    )
                feat = file_feat[start_frame:end_frame]
            else:
                feat = waveform_to_fbank(
                    waveform[:, start:end],
                    sample_rate=sample_rate,
                    extractor=extractor,
                    **fbank_kwargs,
                )
            feats.append(feat)
            spans.append((window_index, start / sample_rate, end / sample_rate))
        return feats, spans

    def score_prepared_windows(
        self,
        prepared: PreparedFileWindows,
        *,
        batch_size: int = 64,
    ) -> list[dict]:
        """Score previously extracted window features."""

        if not prepared.feats:
            return []
        threshold = float(self._demo_cfg.get("qbyt_threshold", 0.5))
        scores = self._score_window_feats(
            prepared.feats,
            prepared.keyword_ids,
            batch_size=batch_size,
        )
        if len(scores) != len(prepared.spans):
            raise RuntimeError(
                "Stage-II window scorer returned an unexpected result count: "
                f"expected={len(prepared.spans)}, actual={len(scores)}"
            )
        results = []
        for (window_index, start_sec, end_sec), (raw_logit, score) in zip(
            prepared.spans, scores
        ):
            result = self._clip_result(
                prepared.audio_path,
                prepared.keyword,
                prepared.keyword_phonemes,
                end_sec=end_sec,
                qbyt_score=float(score),
                threshold=threshold,
                skipped=False,
                qbyt_raw_logit=raw_logit,
                start_sec=start_sec,
            )
            result["window_index"] = window_index
            results.append(result)
        return results

    def score_window_features_multi(
        self,
        audio_path: str,
        feats: Sequence,
        spans: Sequence[tuple[int, float, float]],
        keyword_set,
        *,
        batch_size: int = 64,
    ) -> list[dict]:
        """Score extracted windows against an enrolled keyword set."""

        from dma_kws.inference.keyword_set import (
            WINDOW_EVAL_PROTOCOL,
            aggregate_query_scores,
            require_qbyt_threshold,
            scored_keyword_set_fields,
        )

        if not feats:
            return []
        if len(feats) != len(spans):
            raise RuntimeError("window feats and spans must have the same length")
        threshold = require_qbyt_threshold(
            self._demo_cfg.get("qbyt_threshold", 0.5),
            field="demo.qbyt_threshold",
        )
        query_ids = keyword_set.query_token_ids()
        chunk = max(1, int(batch_size))
        results: list[dict] = []
        for start in range(0, len(feats), chunk):
            batch_feats = list(feats[start : start + chunk])
            batch_spans = list(spans[start : start + chunk])
            scored = self._score_feats_multi_with_logits(
                batch_feats,
                query_ids,
                query_batch_size=keyword_set.query_batch_size,
            )
            if len(scored) != len(batch_spans):
                raise RuntimeError(
                    "Stage-II multi-query window scorer returned an unexpected "
                    f"result count: expected={len(batch_spans)}, actual={len(scored)}"
                )
            for (window_index, start_sec, end_sec), query_scores in zip(
                batch_spans,
                scored,
                strict=True,
            ):
                aggregation = aggregate_query_scores(
                    keyword_set,
                    query_scores,
                    threshold=threshold,
                    audio_id=f"{audio_path}#window={window_index}",
                )
                result = scored_keyword_set_fields(
                    keyword_set,
                    aggregation,
                    threshold=threshold,
                )
                result["eval_protocol"] = WINDOW_EVAL_PROTOCOL
                result["audio_path"] = audio_path
                result["clip_span_sec"] = {
                    "start_sec": start_sec,
                    "end_sec": end_sec,
                }
                result["window_index"] = window_index
                results.append(result)
        return results

    def _score_window_feats(
        self,
        feats: Sequence,
        keyword_ids: Sequence[int],
        *,
        batch_size: int,
    ) -> list[tuple[float | None, float]]:
        """Score window features in bounded GPU batches, preserving order."""

        if not feats:
            return []
        chunk = max(1, int(batch_size))
        keyword_ids_list = list(keyword_ids)
        scores: list[tuple[float | None, float]] = []
        for start in range(0, len(feats), chunk):
            batch_feats = list(feats[start : start + chunk])
            batch_ids = [keyword_ids_list] * len(batch_feats)
            scores.extend(self._score_feats_with_logits(batch_feats, batch_ids))
        return scores

    def run_file_windows(
        self,
        audio_path: str,
        keyword: str,
        *,
        window_sec: float,
        hop_sec: float,
        keyword_phonemes: object | None = None,
        batch_size: int = 64,
        fbank_windows: str = "independent",
    ) -> list[dict]:
        """Run Stage-II verification on sliding windows of a long audio file.

        The file is divided into windows of length ``window_sec`` advanced by
        ``hop_sec``. ``window_sec == hop_sec`` is a non-overlapping grid. Each
        window is scored independently and a result dict in the same format as
        :meth:`run` is returned. ``keyword_phonemes`` overrides automatic G2P
        when supplied.

        ``fbank_windows="independent"`` extracts fbank on each window waveform
        (bit-identical to the original path). ``fbank_windows="file"`` extracts
        fbank once on the full file and slices frames; that is faster but not
        bit-identical when ``snip_edges`` is false.
        """
        prepared = self.prepare_file_windows(
            audio_path,
            keyword,
            window_sec=window_sec,
            hop_sec=hop_sec,
            keyword_phonemes=keyword_phonemes,
            fbank_windows=fbank_windows,
        )
        return self.score_prepared_windows(
            prepared,
            batch_size=batch_size,
        )
