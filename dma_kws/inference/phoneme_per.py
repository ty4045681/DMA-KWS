"""Phoneme-adapter PER evaluation on generic clip manifests."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Mapping, Sequence

from dma_kws.g2p import make_g2p, text_to_phonemes
from dma_kws.metrics import edit_distance
from dma_kws.tokenizer import tokenize_phoneme_string

if TYPE_CHECKING:
    from dma_kws.inference.stage2_verifier import Stage2Verifier


def select_per_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    reference_column: str,
    label_filter: int | None = None,
    limit: int = 0,
    first_row_number: int = 1,
) -> list[dict[str, Any]]:
    """Validate and select manifest rows whose reference text is meaningful.

    ``keyword`` is a query, not generally an audio transcript. It is accepted as
    the PER reference only for rows explicitly filtered to positive examples.
    ``first_row_number`` lets file-backed callers report CSV rows after the
    header using the same numbering as
    :func:`dma_kws.inference.manifest.load_manifest`.
    """
    reference_column = str(reference_column).strip()
    if not reference_column:
        raise ValueError("prep.per_reference_column must be non-empty")
    if limit < 0:
        raise ValueError("prep.limit must be >= 0")
    if first_row_number < 1:
        raise ValueError("first_row_number must be >= 1")
    if reference_column == "keyword" and label_filter != 1:
        raise ValueError(
            "Using keyword as the PER reference is valid only for positive clips; "
            "set prep.per_label_filter=1, or provide text_variant for every row and use "
            "prep.per_reference_column=text_variant"
        )

    selected: list[dict[str, Any]] = []
    for row_number, raw_row in enumerate(rows, start=first_row_number):
        row = dict(raw_row)
        if "label" in row:
            try:
                row["label"] = int(row["label"])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Manifest row {row_number} has invalid label {row['label']!r}; "
                    "expected an integer"
                ) from exc
        if label_filter is not None:
            if "label" not in row:
                raise ValueError(
                    f"Manifest row {row_number} has no label required by "
                    f"prep.per_label_filter={label_filter}"
                )
            if row["label"] != label_filter:
                continue
        value = row.get(reference_column)
        if value is None or not str(value).strip():
            raise ValueError(
                f"Manifest row {row_number} has an empty {reference_column!r} reference"
            )
        selected.append(row)

    if not selected:
        suffix = (
            f" after filtering label={label_filter}" if label_filter is not None else ""
        )
        raise ValueError(f"Manifest contains no PER evaluation rows{suffix}")
    if limit:
        selected = selected[:limit]
    return selected


def build_per_record(
    row: Mapping[str, Any],
    *,
    reference_column: str,
    reference_phonemes: Sequence[str],
    reference_ids: Sequence[int],
    hypothesis_phonemes: Sequence[str],
    hypothesis_ids: Sequence[int],
    skipped: bool,
) -> dict[str, Any]:
    """Build one inspectable utterance-level PER result."""
    distance = edit_distance(list(reference_ids), list(hypothesis_ids))
    ref_length = len(reference_ids)
    record: dict[str, Any] = {
        "audio_path": str(row["audio_path"]),
        "keyword": str(row["keyword"]),
        "reference_column": reference_column,
        "reference_text": str(row[reference_column]),
        "reference_phonemes": list(reference_phonemes),
        "hypothesis_phonemes": list(hypothesis_phonemes),
        "edit_distance": distance,
        "reference_length": ref_length,
        "per": distance / ref_length if ref_length else None,
        "skipped": bool(skipped),
    }
    if "label" in row:
        try:
            record["label"] = int(row["label"])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"PER result for audio {row.get('audio_path')!r} has invalid label "
                f"{row['label']!r}; expected an integer"
            ) from exc
    manifest_meta = {
        key: value
        for key, value in row.items()
        if key not in {"audio_path", "keyword", "label", reference_column}
    }
    if manifest_meta:
        record["manifest_meta"] = manifest_meta
    return record


def summarize_per_results(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Return corpus-weighted PER and diagnostic counts."""
    total_distance = sum(int(row["edit_distance"]) for row in results)
    total_reference = sum(int(row["reference_length"]) for row in results)
    utterance_pers = [
        float(row["per"]) for row in results if row.get("per") is not None
    ]
    return {
        "num_samples": len(results),
        "num_skipped": sum(bool(row.get("skipped", False)) for row in results),
        "num_empty_hypotheses": sum(
            not row.get("hypothesis_phonemes") for row in results
        ),
        "total_edit_distance": total_distance,
        "total_reference_phonemes": total_reference,
        "per": total_distance / total_reference if total_reference else None,
        "mean_utterance_per": (
            sum(utterance_pers) / len(utterance_pers) if utterance_pers else None
        ),
    }


class PhonemePerRunner:
    """Decode clip-manifest audio through a loaded Stage II phoneme adapter."""

    def __init__(
        self,
        *,
        verifier: "Stage2Verifier",
        tokenizer,
        sample_rate: int,
        g2p=None,
    ) -> None:
        self._verifier = verifier
        self._tokenizer = tokenizer
        self._sample_rate = int(sample_rate)
        self._g2p = g2p if g2p is not None else make_g2p()

    @property
    def stream_policy(self):
        return self._verifier.stream_policy

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
        prep: Mapping[str, Any],
        device,
    ) -> "PhonemePerRunner":
        from dma_kws.config import get_tokenizer_config
        from dma_kws.inference.stage2_verifier import Stage2Verifier
        from dma_kws.pathing import resolve_dict_path
        from dma_kws.tokenizer import load_char_tokenizer

        stage1_cfg = config.get("stage1")
        if not isinstance(stage1_cfg, Mapping):
            raise ValueError("Config section 'stage1' must be a mapping")
        tokenizer_cfg = get_tokenizer_config(dict(config))
        tokenizer = load_char_tokenizer(
            resolve_dict_path(config),
            split_with_space=tokenizer_cfg.get("split_with_space", " "),
        )
        return cls(
            verifier=Stage2Verifier.from_config(config, prep, device),
            tokenizer=tokenizer,
            sample_rate=int(stage1_cfg.get("sample_rate", 16000)),
        )

    def run_batch(
        self,
        rows: Sequence[Mapping[str, Any]],
        *,
        reference_column: str,
        batch_size: int = 64,
        num_workers: int = 0,
    ) -> list[dict[str, Any]]:
        """Compute full-clip greedy CTC hypotheses and utterance PER records."""
        try:
            from torch.utils.data import DataLoader
        except ImportError as exc:
            raise SystemExit(
                "Missing torch/torchaudio. Install CUDA PyTorch on the remote training machine first."
            ) from exc

        from dma_kws.inference.stage2_clip import (
            ClipFeatureDataset,
            collate_clip_feature_batch,
        )

        rows = list(rows)
        reference_cache: dict[str, tuple[list[str], list[int]]] = {}
        for row in rows:
            reference_text = str(row[reference_column]).strip()
            if reference_text not in reference_cache:
                phonemes = text_to_phonemes(self._g2p, reference_text)
                token_ids = tokenize_phoneme_string(
                    self._tokenizer,
                    " ".join(phonemes),
                )
                reference_cache[reference_text] = (phonemes, token_ids)

        dataset = ClipFeatureDataset(
            audio_paths=[str(row["audio_path"]) for row in rows],
            sample_rate=self._sample_rate,
            fbank_extractor=self._verifier.fbank_extractor,
            fbank_kwargs=self._verifier.fbank_kwargs,
            min_fbank_frames=self._verifier.min_fbank_frames,
        )
        loader = DataLoader(
            dataset,
            batch_size=max(1, int(batch_size)),
            shuffle=False,
            num_workers=max(0, int(num_workers)),
            collate_fn=collate_clip_feature_batch,
        )

        results: list[dict[str, Any] | None] = [None] * len(rows)

        def store(index: int, hypothesis_ids: Sequence[int], *, skipped: bool) -> None:
            row = rows[index]
            reference_text = str(row[reference_column]).strip()
            reference_phonemes, reference_ids = reference_cache[reference_text]
            hypothesis_ids = list(hypothesis_ids)
            results[index] = build_per_record(
                row,
                reference_column=reference_column,
                reference_phonemes=reference_phonemes,
                reference_ids=reference_ids,
                hypothesis_phonemes=self._tokenizer.ids2tokens(hypothesis_ids),
                hypothesis_ids=hypothesis_ids,
                skipped=skipped,
            )

        for batch in loader:
            feats = []
            pending_indices: list[int] = []
            for index, feat, _end_sec in batch:
                if feat is None:
                    # A too-short clip is a real recognition failure. Keep it in
                    # corpus PER as an empty hypothesis instead of lowering PER by
                    # silently dropping it.
                    store(index, [], skipped=True)
                    continue
                feats.append(feat)
                pending_indices.append(index)
            if not feats:
                continue
            hypotheses = self._verifier.decode_phoneme_feats(feats)
            if len(hypotheses) != len(pending_indices):
                raise RuntimeError(
                    "Phoneme decoder returned a different number of hypotheses than inputs"
                )
            for index, hypothesis_ids in zip(pending_indices, hypotheses):
                store(index, hypothesis_ids, skipped=False)

        if any(result is None for result in results):
            raise RuntimeError(
                "PER evaluation did not produce a result for every manifest row"
            )
        return [result for result in results if result is not None]
