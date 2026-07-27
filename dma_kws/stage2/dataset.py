"""Stage II LibriPhrase training and evaluation datasets aligned with main qbyt datasets."""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from dma_kws.g2p import make_g2p, text_to_phonemes
from dma_kws.tokenizer import build_seq_label, load_char_tokenizer, tokenize_phoneme_string

_PARQUET_COLUMNS = ["ngram", "ngram_g2p", "clips_file", "distances_file"]

_EVAL_COLUMNS = [
    "anchor_text",
    "anchor",
    "anchor_dur",
    "comparison_text",
    "comparison",
    "comparison_dur",
    "target",
    "type",
]

_DEFAULT_EVAL_CSV = [
    "evaluation_set/libriphrase_diffspk_all_1word.csv",
    "evaluation_set/libriphrase_diffspk_all_2word.csv",
    "evaluation_set/libriphrase_diffspk_all_3word.csv",
    "evaluation_set/libriphrase_diffspk_all_4word.csv",
]

_EASY_TYPES = {"diffspk_easyneg", "diffspk_positive"}
_HARD_TYPES = {"diffspk_hardneg", "diffspk_positive"}


def _resolve_fbank_path(wav_dir: str | Path, query_wav: str) -> str:
    path = os.path.join(wav_dir, query_wav)
    for prefix in ("LP-460", "GP-1000", "LP-100"):
        path = path.replace(prefix, f"{prefix}-fbank")
    return path.replace(".wav", ".npy")


class LibriPhraseTrainDataset(Dataset):
    """LibriPhrase Stage II training dataset with random and hard negatives."""

    def __init__(
        self,
        *,
        wav_dir: str | Path,
        parquet_file: str | Path | None = None,
        tokenizer: Any | None = None,
        dict_path: str | Path | None = None,
        negative_ratio: int = 1,
        hard_negative_ratio: int = 1,
        sample_lens: int = 5000,
        seed: int | None = None,
        df: pd.DataFrame | None = None,
        augment: bool = False,
        noise_list_path: str | Path | None = None,
    ) -> None:
        if tokenizer is None:
            if dict_path is None:
                raise ValueError("Either tokenizer or dict_path must be provided")
            tokenizer = load_char_tokenizer(Path(dict_path))
        self.tokenizer = tokenizer

        if df is not None:
            self.df = df.reset_index(drop=True)
        else:
            if parquet_file is None:
                raise ValueError("Either df or parquet_file must be provided")
            self.df = pd.read_parquet(parquet_file, columns=_PARQUET_COLUMNS)

        self.anchor_lists = self.df["ngram"].tolist()
        self.anchor2idx = {anchor: idx for idx, anchor in enumerate(self.anchor_lists)}
        self.negative_ratio = negative_ratio
        self.hard_negative_ratio = hard_negative_ratio
        self.sample_lens = sample_lens
        self.wav_dir = str(wav_dir)
        self._rng = random.Random(seed)
        self._feature_extractor = None
        if augment:
            from dma_kws.stage2.features import FeatureExtractor

            self._feature_extractor = FeatureExtractor(
                augment=True,
                wav_dir=wav_dir,
                noise_list_path=noise_list_path,
            )

    def __len__(self) -> int:
        return self.sample_lens

    def get_random_clips(self, clips_file: str) -> dict:
        number = int(os.path.basename(clips_file).split("-")[-2])
        random_number = self._rng.randint(0, number - 1)
        clips_data = np.load(clips_file, allow_pickle=True)
        return clips_data[random_number]

    def get_random_distances(self, distances_file: str) -> dict | None:
        number = int(os.path.basename(distances_file).split("-")[-2])
        if number == 0:
            return None
        random_number = self._rng.randint(0, number - 1)
        distances_data = np.load(distances_file, allow_pickle=True)
        return distances_data[random_number]

    def get_negative(self, index: int) -> tuple[dict, str, str]:
        remaining_indices = [i for i in range(len(self.anchor_lists)) if i != index]
        random_index = self._rng.choice(remaining_indices)

        negative = self.anchor_lists[random_index]
        negative_inform = self.df.iloc[random_index]
        negative_clips = negative_inform["clips_file"]
        negative_wav = self.get_random_clips(negative_clips)
        negative_g2p = negative_inform["ngram_g2p"]
        return negative_wav, negative_g2p, negative

    def get_hard_negative(self, hard_negative: dict) -> tuple[dict, str, str]:
        hard_ngram = hard_negative["ngram"]

        idx = self.anchor2idx[hard_ngram]
        hard_ngram_inform = self.df.iloc[idx]

        hard_ngram_wav = self.get_random_clips(hard_ngram_inform["clips_file"])
        hard_ngram_g2p = hard_ngram_inform["ngram_g2p"]
        return hard_ngram_wav, hard_ngram_g2p, hard_ngram

    def _load_fbank(self, query_wav: str) -> torch.Tensor:
        if self._feature_extractor is not None:
            return self._feature_extractor.process(query_wav)["feat"]
        fbank_path = _resolve_fbank_path(self.wav_dir, query_wav)
        feats = torch.from_numpy(np.load(fbank_path))
        return feats

    def __getitem__(self, index: int) -> dict:
        index = index % len(self.anchor_lists)
        anchor = self.anchor_lists[index]
        anchor_inform = self.df.iloc[index]
        anchor_g2p = anchor_inform["ngram_g2p"]
        label = 1

        if self._rng.random() < 0.5:
            label = 1
            anchor_clips = anchor_inform["clips_file"]
            query_g2p = anchor_g2p
            query_wav = self.get_random_clips(anchor_clips)["audio_path"]
        else:
            label = 0
            hard_neg = self.get_random_distances(anchor_inform["distances_file"])
            negative_type = self._rng.choices(
                [1, 2],
                weights=[self.negative_ratio, self.hard_negative_ratio],
                k=1,
            )[0]
            if negative_type == 1 or hard_neg is None:
                negative_wav, negative_g2p, _negative = self.get_negative(index)
            else:
                negative_wav, negative_g2p, _negative = self.get_hard_negative(hard_neg)

            query_wav = negative_wav["audio_path"]
            query_g2p = negative_g2p

        feats = self._load_fbank(query_wav)

        anchor_seq = tokenize_phoneme_string(self.tokenizer, anchor_g2p)
        query_seq = tokenize_phoneme_string(self.tokenizer, query_g2p)
        seq_label = build_seq_label(anchor_seq, query_seq)

        return {
            "anchor_seq": torch.tensor(anchor_seq, dtype=torch.long),
            # The clip's own phoneme sequence, kept so the auxiliary CTC loss can
            # supervise the adapter trunk. For negatives this differs from the
            # anchor, which is precisely why it has to be carried separately.
            "query_seq": torch.tensor(query_seq, dtype=torch.long),
            "feat": feats,
            "label": torch.tensor(label, dtype=torch.long),
            "seq_label": torch.tensor(seq_label, dtype=torch.long),
        }


def _resolve_eval_fbank_path(
    test_dir: str | Path,
    query_wav: str,
    *,
    fbank_dir: str | Path | None = None,
) -> str:
    path = os.path.join(str(fbank_dir or test_dir), query_wav)
    return path.replace(".wav", ".npy")


def _build_eval_dataframe(
    test_dir: str | Path,
    csv_files: list[str],
    aggregate_csv: str | Path,
) -> pd.DataFrame:
    aggregate_path = Path(aggregate_csv)
    if not aggregate_path.is_absolute():
        aggregate_path = Path(test_dir) / aggregate_path

    if aggregate_path.exists():
        return pd.read_csv(aggregate_path)

    frames: list[pd.DataFrame] = []
    for rel_path in csv_files:
        csv_path = Path(test_dir) / rel_path
        df = pd.read_csv(csv_path)
        anc = df[
            [
                "anchor_text",
                "anchor",
                "anchor_dur",
                "comparison_text",
                "comparison",
                "comparison_dur",
                "target",
                "type",
            ]
        ]
        com = df[
            [
                "comparison_text",
                "comparison",
                "comparison_dur",
                "anchor_text",
                "anchor",
                "anchor_dur",
                "target",
                "type",
            ]
        ].rename(
            columns={
                "comparison_text": "anchor_text",
                "comparison": "anchor",
                "comparison_dur": "anchor_dur",
                "anchor_text": "comparison_text",
                "anchor": "comparison",
                "anchor_dur": "comparison_dur",
            }
        )
        frames.extend([anc, com])

    data = pd.concat(frames, ignore_index=True)
    aggregate_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(aggregate_path, index=False)
    return data


def _filter_eval_split(df: pd.DataFrame, split: str) -> pd.DataFrame:
    if split == "easy":
        return df.loc[df["type"].isin(_EASY_TYPES)].reset_index(drop=True)
    if split == "hard":
        return df.loc[df["type"].isin(_HARD_TYPES)].reset_index(drop=True)
    if split == "all":
        return df.reset_index(drop=True)
    raise ValueError(f"Unsupported eval split: {split!r} (expected easy, hard, or all)")


def resolve_stage2_eval_paths(config: dict[str, Any]) -> dict[str, Any]:
    """Resolve LibriPhrase eval dataset paths from a loaded YAML config."""
    stage2 = config["stage2"]
    paths = config.get("paths", {})
    eval_cfg = stage2.get("eval", {}) or {}

    test_dir = eval_cfg.get("test_dir", "")
    if not test_dir:
        for key in ("libriphrase460_root", "libriphrase100_root", "libriphrase_root"):
            root = paths.get(key, "")
            if root:
                test_dir = str(Path(root) / "eval")
                break
    if not test_dir:
        raise ValueError(
            "Missing stage2.eval.test_dir and no libriphrase*_root in paths for eval fallback"
        )

    csv_files = list(eval_cfg.get("csv_files", _DEFAULT_EVAL_CSV))
    aggregate_csv = eval_cfg.get(
        "aggregate_csv",
        "evaluation_set/test_all_phrase.csv",
    )
    batch_size = int(eval_cfg.get("batch_size", stage2.get("validation", {}).get("batch_size", 256)))
    num_workers = int(eval_cfg.get("num_workers", stage2.get("num_workers", 4)))
    fbank_dir = eval_cfg.get("fbank_dir", "") or test_dir

    return {
        "test_dir": Path(test_dir),
        "fbank_dir": Path(fbank_dir),
        "csv_files": csv_files,
        "aggregate_csv": aggregate_csv,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "split": str(eval_cfg.get("split", "hard")),
    }


class LibriPhraseEvalDataset(Dataset):
    """LibriPhrase hard/easy evaluation dataset aligned with main ``LibriPhrasetTEST``."""

    def __init__(
        self,
        *,
        test_dir: str | Path,
        fbank_dir: str | Path | None = None,
        split: str = "hard",
        csv_files: list[str] | None = None,
        aggregate_csv: str | Path | None = None,
        tokenizer: Any | None = None,
        dict_path: str | Path | None = None,
        split_with_space: str = " ",
        df: pd.DataFrame | None = None,
        g2p: Any | None = None,
    ) -> None:
        if tokenizer is None:
            if dict_path is None:
                raise ValueError("Either tokenizer or dict_path must be provided")
            tokenizer = load_char_tokenizer(Path(dict_path), split_with_space=split_with_space)
        self.tokenizer = tokenizer
        self.test_dir = str(test_dir)
        self.fbank_dir = str(fbank_dir or test_dir)
        self.split = split
        self.g2p = g2p if g2p is not None else make_g2p()

        if df is not None:
            data = df.copy()
        else:
            data = _build_eval_dataframe(
                test_dir,
                csv_files or _DEFAULT_EVAL_CSV,
                aggregate_csv or "evaluation_set/test_all_phrase.csv",
            )

        missing = [column for column in _EVAL_COLUMNS if column not in data.columns]
        if missing:
            raise ValueError(f"Eval dataframe missing columns: {missing}")

        self.data = _filter_eval_split(data, split).values.tolist()

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> dict:
        (
            anchor_text,
            _anchor,
            _anchor_dur,
            _comparison_text,
            comparison_wav,
            _comparison_dur,
            target,
            _sample_type,
        ) = self.data[index]

        # Must go through the same helper as training/inference: tokenizing raw
        # g2p_en output here would leave stress digits attached in one place and
        # not the others, and every mismatching symbol silently becomes <unk>.
        anchor_g2p = " ".join(text_to_phonemes(self.g2p, anchor_text))

        fbank_path = _resolve_eval_fbank_path(
            self.test_dir,
            comparison_wav,
            fbank_dir=self.fbank_dir,
        )
        feats = torch.from_numpy(np.load(fbank_path))

        anchor_seq = tokenize_phoneme_string(self.tokenizer, anchor_g2p)

        return {
            "anchor_seq": torch.tensor(anchor_seq, dtype=torch.long),
            "feat": feats,
            "label": torch.tensor(target, dtype=torch.long),
        }
