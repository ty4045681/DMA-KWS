"""Stage II LibriPhrase training and evaluation datasets aligned with main qbyt datasets."""

from __future__ import annotations

import os
import random
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.utils.data
from torch.utils.data import Dataset

from dma_kws.g2p import make_g2p, text_to_phonemes
from dma_kws.tokenizer import (
    DEFAULT_SEQ_LABEL_MODE,
    SEQ_LABEL_ORDERED_CONTIGUOUS_PREFIX,
    build_seq_label,
    load_char_tokenizer,
    normalize_seq_label_mode,
    tokenize_phoneme_string,
)
from dma_kws.training.ddp import process_rank

_PARQUET_COLUMNS = ["ngram", "ngram_g2p", "clips_file", "distances_file"]
_MAX_CONTAINING_NEGATIVE_DRAWS = 16

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


#: Attributes on a dataset that may hold another dataset with its own ``_rng``.
#: ``MixedAdaptDataset`` wraps a ``LibriPhraseTrainDataset``, so reseeding only the
#: outer object would leave the inner negative sampling replicated across workers.
_NESTED_DATASET_ATTRS = ("keyword_dataset", "libri_dataset", "dataset")


def _reseed_rng_holders(obj: Any, seed: int, seen: set[int]) -> None:
    if obj is None or id(obj) in seen:
        return
    seen.add(id(obj))
    if isinstance(getattr(obj, "_rng", None), random.Random):
        # Offset per holder so a wrapper and the dataset it wraps do not draw the
        # identical stream, which would correlate the mix decision with the
        # negative it selects.
        obj._rng = random.Random(seed + 7919 * len(seen))
    for attr in _NESTED_DATASET_ATTRS:
        _reseed_rng_holders(getattr(obj, attr, None), seed, seen)


def stage2_worker_init_fn(worker_id: int) -> None:
    """Give every DataLoader worker its own random stream.

    ``LibriPhraseTrainDataset`` builds ``random.Random(seed)`` in the parent
    process, so each worker inherits a copy whose state is already identical.
    Without this hook all ``num_workers`` replicas replay the same
    positive/negative decisions, hard-negative picks and clip choices, and the
    only randomness left across workers is which anchor index they are handed.
    Nothing in the loss curves reveals it: a batch is produced by a single
    worker, so each batch still looks varied.

    ``get_worker_info().seed`` is derived by PyTorch as ``base_seed + worker_id``
    and the base changes per epoch, so the streams stay distinct, reproducible,
    and non-repeating across epochs. Explicit ``seed_everything(seed)`` gives each
    DDP rank the same base seed, however, so fold the rank in ourselves; otherwise
    ``MixedAdaptationDataset`` ignores the sampler index and two GPUs replay the
    exact same random samples.
    """
    info = torch.utils.data.get_worker_info()
    if info is None:
        # num_workers=0: the dataset runs in the main process and the seed passed
        # to the constructor is already the intended one.
        return
    rank_seed = int(info.seed) + 1_000_003 * process_rank()
    # PyTorch seeds these module-global RNGs before invoking worker_init_fn, but
    # its base seed can be identical across externally launched DDP ranks. Fold
    # the rank in here as well so fbank dither and any future transforms do not
    # replay the same random stream on every GPU.
    random.seed(rank_seed)
    np.random.seed(rank_seed % (2**32))
    torch.manual_seed(rank_seed)
    _reseed_rng_holders(info.dataset, rank_seed, set())


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
        noise_augmentation: Mapping[str, Any] | None = None,
        fbank_kwargs: Mapping[str, Any] | None = None,
        seq_label_mode: str = DEFAULT_SEQ_LABEL_MODE,
    ) -> None:
        if tokenizer is None:
            if dict_path is None:
                raise ValueError("Either tokenizer or dict_path must be provided")
            tokenizer = load_char_tokenizer(Path(dict_path))
        self.tokenizer = tokenizer
        self.seq_label_mode = normalize_seq_label_mode(seq_label_mode)

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
        self._noise_augmenter = None
        self._noise_probability = 0.0
        if augment:
            from dma_kws.stage2.features import FeatureExtractor

            self._feature_extractor = FeatureExtractor(
                augment=True,
                wav_dir=wav_dir,
                noise_list_path=noise_list_path,
            )

        if noise_augmentation is not None and not isinstance(
            noise_augmentation, Mapping
        ):
            raise ValueError("stage2.noise_augmentation must be a mapping")
        noise_cfg = dict(noise_augmentation or {})
        allowed_noise_keys = {
            "enabled",
            "probability",
            "waveform_dir",
            "noise_list_path",
            "snr_db_min",
            "snr_db_max",
        }
        unknown_noise_keys = sorted(set(noise_cfg) - allowed_noise_keys)
        if unknown_noise_keys:
            raise ValueError(
                "Unknown stage2.noise_augmentation fields: "
                + ", ".join(unknown_noise_keys)
            )
        if bool(noise_cfg.get("enabled", False)):
            if augment:
                raise ValueError(
                    "Legacy Stage II augment and stage2.noise_augmentation cannot "
                    "be enabled together"
                )
            probability = float(noise_cfg.get("probability", 0.3))
            if not 0.0 <= probability <= 1.0:
                raise ValueError(
                    "Stage II noise augmentation probability must be between 0 and 1"
                )
            waveform_dir = str(noise_cfg.get("waveform_dir", "")).strip()
            noise_path = str(noise_cfg.get("noise_list_path", "")).strip()
            if not waveform_dir:
                raise ValueError(
                    "stage2.noise_augmentation.waveform_dir is required when enabled"
                )
            if not noise_path:
                raise ValueError(
                    "stage2.noise_augmentation.noise_list_path is required when enabled"
                )

            from dma_kws.stage2.features import TrainingNoiseAugmenter

            self._noise_augmenter = TrainingNoiseAugmenter(
                waveform_dir=waveform_dir,
                noise_list_path=noise_path,
                snr_db_min=float(noise_cfg.get("snr_db_min", 10.0)),
                snr_db_max=float(noise_cfg.get("snr_db_max", 20.0)),
                fbank_kwargs=fbank_kwargs,
            )
            self._noise_probability = probability

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
        should_add_noise = self._noise_augmenter is not None and (
            self._noise_probability >= 1.0
            or (
                self._noise_probability > 0.0
                and self._rng.random() < self._noise_probability
            )
        )
        if should_add_noise:
            return self._noise_augmenter.extract(query_wav, rng=self._rng)
        if self._feature_extractor is not None:
            return self._feature_extractor.process(query_wav)["feat"]
        fbank_path = _resolve_fbank_path(self.wav_dir, query_wav)
        feats = torch.from_numpy(np.load(fbank_path))
        return feats

    def _draw_negative(self, index: int, anchor_inform) -> tuple[str, str]:
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
        return negative_wav["audio_path"], negative_g2p

    def __getitem__(self, index: int) -> dict:
        index = index % len(self.anchor_lists)
        anchor_inform = self.df.iloc[index]
        anchor_g2p = anchor_inform["ngram_g2p"]
        anchor_seq = tokenize_phoneme_string(self.tokenizer, anchor_g2p)
        label = 1

        if self._rng.random() < 0.5:
            anchor_clips = anchor_inform["clips_file"]
            query_wav = self.get_random_clips(anchor_clips)["audio_path"]
            query_seq = list(anchor_seq)
            seq_label = build_seq_label(
                anchor_seq,
                query_seq,
                mode=self.seq_label_mode,
            )
        else:
            label = 0
            for _ in range(_MAX_CONTAINING_NEGATIVE_DRAWS):
                query_wav, query_g2p = self._draw_negative(index, anchor_inform)
                query_seq = tokenize_phoneme_string(self.tokenizer, query_g2p)
                seq_label = build_seq_label(
                    anchor_seq,
                    query_seq,
                    mode=self.seq_label_mode,
                )
                # Stage II is keyword-occurrence detection: an n-gram that
                # contains the complete anchor is a positive, not a hard
                # negative. Re-draw so the intended 50/50 sampling ratio is not
                # distorted by nested LibriPhrase n-grams.
                if (
                    self.seq_label_mode != SEQ_LABEL_ORDERED_CONTIGUOUS_PREFIX
                    or not seq_label[-1]
                ):
                    break
            else:
                # A tiny/adversarial phrase pool may contain no valid negative.
                # The selected audio still contains the keyword, so relabeling
                # is the only supervision-consistent fallback.
                label = 1

        feats = self._load_fbank(query_wav)
        if (
            self.seq_label_mode == SEQ_LABEL_ORDERED_CONTIGUOUS_PREFIX
            and bool(seq_label[-1]) != bool(label)
        ):
            raise RuntimeError(
                "Internal Stage II target mismatch: utterance and completion "
                "labels must agree under keyword-occurrence semantics"
            )

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
            "sample_id": torch.tensor(index, dtype=torch.long),
            "anchor_seq": torch.tensor(anchor_seq, dtype=torch.long),
            "feat": feats,
            "label": torch.tensor(target, dtype=torch.long),
        }
