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
from dma_kws.tokenizer import tokenize_phoneme_string


def phonemes_to_g2p_string(phonemes: list[str] | str) -> str:
    """Convert a phoneme list or space-separated string to a G2P target string."""
    if isinstance(phonemes, str):
        return phonemes.strip()
    return " ".join(phonemes)


def encode_manifest_target(record: dict[str, Any], tokenizer) -> list[int]:
    """Encode a manifest record using Wenet ``CharTokenizer``."""
    if "phonemes_g2p" in record:
        return tokenize_phoneme_string(tokenizer, str(record["phonemes_g2p"]))
    if "phonemes" in record:
        return tokenize_phoneme_string(tokenizer, phonemes_to_g2p_string(record["phonemes"]))
    raise KeyError("Manifest record must contain 'phonemes_g2p' or 'phonemes'")


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
