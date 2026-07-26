"""Stage I dataset and collation helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from dma_kws.audio import extract_fbank, load_audio
from dma_kws.jsonl import read_jsonl
from dma_kws.stage1.prepare_fbank import resolve_record_fbank_path
from dma_kws.tokenizer import tokenize_phoneme_string, unsupported_phones

_MANIFEST_VALIDATION_SAMPLE = 200


def phonemes_to_g2p_string(phonemes: list[str] | str) -> str:
    """Convert a phoneme list or space-separated string to a G2P target string."""
    if isinstance(phonemes, str):
        return phonemes.strip()
    return " ".join(phonemes)


def manifest_phoneme_tokens(record: dict[str, Any]) -> list[str]:
    """Return the phoneme tokens a manifest record will be tokenized from."""
    if "phonemes_g2p" in record:
        return str(record["phonemes_g2p"]).split()
    if "phonemes" in record:
        return phonemes_to_g2p_string(record["phonemes"]).split()
    raise KeyError("Manifest record must contain 'phonemes_g2p' or 'phonemes'")


def encode_manifest_target(record: dict[str, Any], tokenizer) -> list[int]:
    """Encode a manifest record using Wenet ``CharTokenizer``."""
    return tokenize_phoneme_string(tokenizer, " ".join(manifest_phoneme_tokens(record)))


def validate_manifest_phonemes(
    records: list[dict[str, Any]],
    *,
    manifest_path: Path | str,
    sample_size: int = _MANIFEST_VALIDATION_SAMPLE,
) -> None:
    """Reject manifests whose phonemes are outside the current vocabulary.

    Manifests written before the stress-marked vocabulary carry stress-stripped
    phonemes (``AH`` instead of ``AH1``), and ``CharTokenizer`` maps every one of
    them to ``<unk>`` — training would see no vowels at all. Checking a bounded
    sample is enough because a manifest is generated in one pass.
    """
    for record in records[:sample_size]:
        unsupported = unsupported_phones(manifest_phoneme_tokens(record))
        if unsupported:
            raise ValueError(
                f"{manifest_path}: phonemes outside the vocabulary: {', '.join(unsupported)}. "
                "Manifests predating the stress-marked vocabulary tokenize entirely to <unk>; "
                "regenerate them with scripts/prepare_stage1_librispeech.py."
            )


class Stage1Dataset(Dataset):
    """JSONL manifest dataset for Stage I CTC training."""

    def __init__(
        self,
        manifest_path: Path,
        *,
        tokenizer,
        sample_rate: int,
        num_mel_bins: int,
        fbank_root: Path | str | None = None,
        audio_root: Path | str | None = None,
    ) -> None:
        self.records = read_jsonl(manifest_path)
        validate_manifest_phonemes(self.records, manifest_path=manifest_path)
        self.tokenizer = tokenizer
        self.sample_rate = sample_rate
        self.num_mel_bins = num_mel_bins
        self.fbank_root = Path(fbank_root) if fbank_root else None
        self.audio_root = Path(audio_root) if audio_root else None

    def __len__(self) -> int:
        return len(self.records)

    def _load_features(self, record: dict[str, Any]) -> torch.Tensor:
        fbank_path = resolve_record_fbank_path(
            record,
            fbank_root=self.fbank_root,
            audio_root=self.audio_root,
        )
        if fbank_path is not None and fbank_path.exists():
            import numpy as np

            return torch.from_numpy(np.load(fbank_path))

        waveform, sr = load_audio(record["wav_path"], sample_rate=self.sample_rate)
        return extract_fbank(
            waveform,
            num_mel_bins=self.num_mel_bins,
            sample_rate=sr,
            dither=0.1,
        )

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        record = self.records[index]
        feat = self._load_features(record)
        target = torch.tensor(
            encode_manifest_target(record, self.tokenizer),
            dtype=torch.long,
        )
        return {"feat": feat, "target": target}


def stage1_collate_fn(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    feats = [item["feat"] for item in batch]
    targets = [item["target"] for item in batch]
    return {
        "feats": pad_sequence(feats, batch_first=True, padding_value=0.0),
        "feat_lengths": torch.tensor([feat.size(0) for feat in feats], dtype=torch.long),
        "targets": pad_sequence(targets, batch_first=True, padding_value=0),
        "target_lengths": torch.tensor([target.size(0) for target in targets], dtype=torch.long),
    }
