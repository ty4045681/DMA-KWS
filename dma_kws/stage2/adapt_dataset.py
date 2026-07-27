"""Keyword continual-adaptation datasets mixed with LibriPhrase for anti-forgetting."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from dma_kws.g2p import make_g2p, text_to_phonemes
from dma_kws.stage2.adapt_paths import wav_to_fbank_mirror
from dma_kws.tokenizer import build_seq_label, tokenize_phoneme_string


def _resolve_adapt_fbank_path(fbank_root: Path, audio_path: str, *, manifest_root: Path | None = None) -> Path:
    path = Path(audio_path)
    if path.is_absolute():
        return wav_to_fbank_mirror(fbank_root, path)
    base = manifest_root or fbank_root
    return wav_to_fbank_mirror(fbank_root, (base / path).resolve())


def _text_to_g2p(g2p: Any, text: str) -> str:
    phones = text_to_phonemes(g2p, text)
    return " ".join(phones)


class KeywordAdaptationDataset(Dataset):
    """Manifest-driven keyword adaptation samples aligned with LibriPhraseTrainDataset."""

    def __init__(
        self,
        *,
        manifest_path: str | Path,
        keyword: str,
        fbank_root: str | Path,
        tokenizer: Any,
        g2p: Any | None = None,
        manifest_root: str | Path | None = None,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.keyword = keyword
        self.fbank_root = Path(fbank_root)
        self.manifest_root = Path(manifest_root) if manifest_root else self.manifest_path.parent
        self.tokenizer = tokenizer
        self.g2p = g2p if g2p is not None else make_g2p()

        df = pd.read_csv(self.manifest_path)
        required = {"audio_path", "text", "label"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{self.manifest_path} missing columns: {sorted(missing)}")
        self.df = df.reset_index(drop=True)

        self.anchor_g2p = _text_to_g2p(self.g2p, keyword)
        self._anchor_seq = tokenize_phoneme_string(self.tokenizer, self.anchor_g2p)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> dict:
        row = self.df.iloc[int(index)]
        audio_path = str(row["audio_path"])
        text = str(row["text"])
        label = int(row["label"])

        fbank_path = _resolve_adapt_fbank_path(
            self.fbank_root,
            audio_path,
            manifest_root=self.manifest_root,
        )
        feats = torch.from_numpy(np.load(fbank_path))

        query_g2p = _text_to_g2p(self.g2p, text)
        query_seq = tokenize_phoneme_string(self.tokenizer, query_g2p)
        seq_label = build_seq_label(self._anchor_seq, query_seq)

        return {
            "anchor_seq": torch.tensor(self._anchor_seq, dtype=torch.long),
            # Kept so the adaptation batch schema matches Stage II training even
            # though the auxiliary CTC loss is off during LoRA (the trunk is
            # frozen there).
            "query_seq": torch.tensor(query_seq, dtype=torch.long),
            "feat": feats,
            "label": torch.tensor(label, dtype=torch.long),
            "seq_label": torch.tensor(seq_label, dtype=torch.long),
        }


class TargetKeywordValDataset(Dataset):
    """Held-out keyword clips for adaptation validation (utterance-level AUC)."""

    def __init__(
        self,
        *,
        manifest_path: str | Path,
        keyword: str,
        fbank_root: str | Path,
        tokenizer: Any,
        g2p: Any | None = None,
        manifest_root: str | Path | None = None,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.keyword = keyword
        self.fbank_root = Path(fbank_root)
        self.manifest_root = Path(manifest_root) if manifest_root else self.manifest_path.parent
        self.tokenizer = tokenizer
        self.g2p = g2p if g2p is not None else make_g2p()

        df = pd.read_csv(self.manifest_path)
        if "audio_path" not in df.columns or "label" not in df.columns:
            raise ValueError(f"{self.manifest_path} must contain audio_path and label")
        self.df = df.reset_index(drop=True)

        self.anchor_g2p = _text_to_g2p(self.g2p, keyword)
        self._anchor_seq = tokenize_phoneme_string(self.tokenizer, self.anchor_g2p)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> dict:
        row = self.df.iloc[int(index)]
        audio_path = str(row["audio_path"])
        label = int(row["label"])

        fbank_path = _resolve_adapt_fbank_path(
            self.fbank_root,
            audio_path,
            manifest_root=self.manifest_root,
        )
        feats = torch.from_numpy(np.load(fbank_path))

        return {
            "anchor_seq": torch.tensor(self._anchor_seq, dtype=torch.long),
            "feat": feats,
            "label": torch.tensor(label, dtype=torch.long),
        }


class MixedAdaptationDataset(Dataset):
    """Mix keyword adaptation samples with LibriPhrase at ``mix_ratio``."""

    def __init__(
        self,
        *,
        keyword_dataset: KeywordAdaptationDataset,
        libri_dataset: Any,
        mix_ratio: float = 0.5,
        sample_lens: int = 5000,
        seed: int | None = None,
    ) -> None:
        if not 0.0 <= mix_ratio <= 1.0:
            raise ValueError(f"mix_ratio must be in [0, 1], got {mix_ratio}")
        self.keyword_dataset = keyword_dataset
        self.libri_dataset = libri_dataset
        self.mix_ratio = mix_ratio
        self.sample_lens = sample_lens
        self._rng = random.Random(seed)

    def __len__(self) -> int:
        return self.sample_lens

    #: ``source`` values attached to mixed samples for per-source loss logging.
    SOURCE_LIBRIPHRASE = 0
    SOURCE_KEYWORD = 1

    def __getitem__(self, index: int) -> dict:
        del index
        if self._rng.random() < self.mix_ratio:
            kw_index = self._rng.randrange(len(self.keyword_dataset))
            item = dict(self.keyword_dataset[kw_index])
            item["source"] = self.SOURCE_KEYWORD
        else:
            lp_index = self._rng.randrange(len(self.libri_dataset))
            item = dict(self.libri_dataset[lp_index])
            item["source"] = self.SOURCE_LIBRIPHRASE
        return item


TargetKeywordEvalDataset = TargetKeywordValDataset


def load_adapt_manifest(path: str | Path) -> list[dict[str, Any]]:
    df = pd.read_csv(path)
    required = {"audio_path", "text", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    return df[list(required)].to_dict(orient="records")


def clips_eval_manifest_from_adapt(eval_manifest: Path, keyword: str, output_path: Path) -> Path:
    df = pd.read_csv(eval_manifest)
    if "keyword" not in df.columns:
        df["keyword"] = keyword
    cols = [col for col in ("audio_path", "keyword", "label") if col in df.columns]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df[cols].to_csv(output_path, index=False)
    return output_path
