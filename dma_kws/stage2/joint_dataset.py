"""Deterministic, balanced batches for one-pass keyword LoRA adaptation."""

from __future__ import annotations

import heapq
import math
import os
import random
from collections import defaultdict
from collections.abc import Iterator
from fractions import Fraction
from typing import Any

import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler

from dma_kws.training.ddp import process_rank


def _unit_interval(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1], got {value}")
    return value


def _positive_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive int, got {value!r}")
    return value


def _ticket_seed(value: int) -> int:
    """SplitMix64 bijection; unlike hash(), stable across spawned workers."""
    mask = (1 << 64) - 1
    value = (value + 0x9E3779B97F4A7C15) & mask
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & mask
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & mask
    return value ^ (value >> 31)


def _weighted_kinds(weights: dict[str, float], offset: int) -> Iterator[str]:
    """Seek into one continuous weighted sequence without replaying history.

    A stratum of weight w emits at virtual times (n + 1/2) / w. Merging these
    streams preserves fractional quotas across epoch boundaries. Rational
    arithmetic prevents floating-point ties drifting after a long run.
    """
    rational = {kind: Fraction(str(weight)) for kind, weight in weights.items() if weight > 0}
    total = sum(rational.values())
    rational = {kind: weight / total for kind, weight in rational.items()}
    half = Fraction(1, 2)
    # Before virtual time offset - K there cannot be more than offset events:
    # each of the K strata contributes at most time*w + 1/2 events. This puts
    # the seek within O(K) heap operations of the requested absolute offset.
    before = max(0, offset - len(rational))
    counts = {kind: int(before * weight + half) for kind, weight in rational.items()}
    heap = [
        ((counts[kind] + half) / weight, order, kind)
        for order, (kind, weight) in enumerate(rational.items())
    ]
    heapq.heapify(heap)

    def next_kind() -> str:
        _, order, kind = heapq.heappop(heap)
        counts[kind] += 1
        heapq.heappush(heap, ((counts[kind] + half) / rational[kind], order, kind))
        return kind

    for _ in range(offset - sum(counts.values())):
        next_kind()
    while True:
        yield next_kind()


class _KeywordPool:
    def __init__(self, dataset: Any, *, domain: str) -> None:
        self.dataset = dataset
        frame = dataset.df
        if frame.empty:
            raise ValueError(f"Joint {domain} manifest is empty")
        if "label" not in frame or not frame["label"].isin([0, 1]).all():
            raise ValueError(f"Joint {domain} manifest must have binary 0/1 labels")
        if set(frame["label"].astype(int)) != {0, 1}:
            raise ValueError(f"Joint {domain} manifest must contain positives and negatives")
        identity_columns = ("speaker_id",) if domain == "real" else ("voice_id", "speaker_id")
        # identity -> negative text -> row indices. Select each level uniformly,
        # so repeated recordings and large TTS synthesis groups gain no weight.
        grouped: dict[int, dict[str, dict[str, list[int]]]] = {
            0: defaultdict(lambda: defaultdict(list)),
            1: defaultdict(lambda: defaultdict(list)),
        }
        for index, (_, row) in enumerate(frame.iterrows()):
            identity = f"row:{index}"
            for column in identity_columns:
                value = row.get(column)
                if value is not None and not pd.isna(value) and str(value).strip():
                    identity = f"{column}:{str(value).strip()}"
                    break
            label = int(row["label"])
            text = str(row.get("text", "")) if domain == "tts" and label == 0 else ""
            grouped[label][identity][text].append(index)
        # Materialize plain tuples: lambdas/defaultdicts cannot be pickled by
        # DataLoader's spawn context (the default on macOS).
        self.groups = {
            label: tuple(tuple(tuple(rows) for rows in texts.values()) for texts in identities.values())
            for label, identities in grouped.items()
        }

    def sample(self, label: int, rng: random.Random) -> dict:
        texts = rng.choice(self.groups[label])
        rows = rng.choice(texts)
        return dict(self.dataset[rng.choice(rows)])


class JointAdaptationDataset(Dataset):
    """Eight explicit strata spanning real, TTS, LibriPhrase and backgrounds.

    ``mix_ratio`` is the combined keyword fraction, ``real_fraction`` splits
    that fraction between real and TTS. Background probability is conditional
    on the negative half of LibriPhrase replay, matching the legacy dataset.
    With 0.5 / 0.6 and background probability 0.4, domain shares are
    real=30%, TTS=20%, LibriPhrase=40%, background=10%.

    Use :class:`JointBatchSampler` for training. Its ``(kind, seed)`` tickets
    control both source selection and all sample draws, independently of worker
    assignment. Integer indices are deterministic convenience draws for probes.
    """

    SOURCE_LIBRIPHRASE = 0
    SOURCE_KEYWORD = 1
    DOMAIN_LIBRIPHRASE = 0
    DOMAIN_REAL = 1
    DOMAIN_TTS = 2
    DOMAIN_BACKGROUND = 3

    def __init__(
        self,
        real_dataset: Any,
        tts_dataset: Any,
        libri_dataset: Any,
        mix_ratio: float = 0.5,
        real_fraction: float = 0.6,
        background_keyword_fraction: float = 0.5,
        sample_lens: int = 5000,
        seed: int = 2025,
    ) -> None:
        self.mix_ratio = _unit_interval("mix_ratio", mix_ratio)
        self.real_fraction = _unit_interval("real_fraction", real_fraction)
        background_keyword_fraction = _unit_interval(
            "background_keyword_fraction", background_keyword_fraction
        )
        self.background_keyword_fraction = background_keyword_fraction
        self.sample_lens = _positive_int("sample_lens", sample_lens)
        self.seed = int(seed)
        self.real_dataset = real_dataset
        self.tts_dataset = tts_dataset
        self.libri_dataset = libri_dataset
        self._real_pool = _KeywordPool(real_dataset, domain="real")
        self._tts_pool = _KeywordPool(tts_dataset, domain="tts")
        self._anchor_seq = list(real_dataset._anchor_seq)
        if not self._anchor_seq or self._anchor_seq != list(tts_dataset._anchor_seq):
            raise ValueError("Joint real and TTS datasets must have the same non-empty keyword anchor")
        if libri_dataset.num_anchors < 1:
            raise ValueError("Joint LibriPhrase anchor pool is empty")
        background_probability = _unit_interval(
            "background_negative.probability", libri_dataset._background_probability
        )
        if background_probability > 0 and not libri_dataset.background_enabled:
            raise ValueError("Joint background probability requires an enabled background sampler")
        half_replay = (1.0 - self.mix_ratio) * 0.5
        background = half_replay * background_probability
        half_real = self.mix_ratio * self.real_fraction * 0.5
        half_tts = self.mix_ratio * (1.0 - self.real_fraction) * 0.5
        self.weights = {
            "real_positive": half_real,
            "real_negative": half_real,
            "tts_positive": half_tts,
            "tts_negative": half_tts,
            "libri_positive": half_replay,
            "libri_negative": half_replay - background,
            "background_target": background * background_keyword_fraction,
            "background_generic": background * (1.0 - background_keyword_fraction),
        }
        if self.weights["libri_negative"] > 0 and libri_dataset.num_anchors < 2:
            raise ValueError("Joint LibriPhrase speech negatives require at least two anchors")

    def __len__(self) -> int:
        return self.sample_lens

    def __getitem__(self, ticket: tuple[str, int] | int) -> dict:
        if isinstance(ticket, int):
            seed = _ticket_seed(self.seed + ticket)
            rng = random.Random(seed)
            kind = rng.choices(tuple(self.weights), weights=tuple(self.weights.values()), k=1)[0]
        else:
            kind, seed = ticket
            rng = random.Random(seed)
        if kind not in self.weights or self.weights[kind] <= 0:
            raise ValueError(f"Unknown or disabled joint sample kind: {kind!r}")
        if kind.startswith("real_"):
            item = self._real_pool.sample(int(kind == "real_positive"), rng)
            domain, source = self.DOMAIN_REAL, self.SOURCE_KEYWORD
        elif kind.startswith("tts_"):
            item = self._tts_pool.sample(int(kind == "tts_positive"), rng)
            domain, source = self.DOMAIN_TTS, self.SOURCE_KEYWORD
        else:
            index = rng.randrange(self.libri_dataset.num_anchors)
            is_background = kind.startswith("background_")
            pair_kind = "background" if is_background else kind.removeprefix("libri_")
            item = dict(self.libri_dataset.sample_pair(
                index,
                pair_kind,
                rng=rng,
                anchor_seq=self._anchor_seq if kind == "background_target" else None,
            ))
            domain = self.DOMAIN_BACKGROUND if is_background else self.DOMAIN_LIBRIPHRASE
            source = self.SOURCE_LIBRIPHRASE
        item["source"] = source
        item["domain_source"] = domain
        return item


class JointBatchSampler(Sampler[list[tuple[str, int]]]):
    """Carry fractional source quotas across batches/epochs and issue tickets.

    Every rank has the same batch composition and a disjoint ticket seed stream.
    Distributed rank/world size are resolved at iteration time because Lightning
    may initialize its process group after constructing the DataLoader.

    ``start_batch`` is an epoch-local, consumed batch count. ``set_epoch`` and
    that offset reconstruct the exact suffix without accessing skipped samples,
    so worker prefetch cannot advance checkpoint state. A caller tracking an
    absolute consumed count can use ``divmod(count, full_num_batches)``.
    """

    def __init__(
        self,
        dataset: JointAdaptationDataset,
        batch_size: int,
        seed: int = 2025,
        *,
        start_batch: int = 0,
    ) -> None:
        self.dataset = dataset
        self.batch_size = _positive_int("batch_size", batch_size)
        self.seed = int(seed)
        self.drop_last = True
        self.set_epoch(0, start_batch=start_batch)

    @staticmethod
    def _rank_world() -> tuple[int, int]:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            rank, world_size = torch.distributed.get_rank(), torch.distributed.get_world_size()
        else:
            rank, world_size = process_rank(), int(os.environ.get("WORLD_SIZE", "1"))
        if world_size < 1 or not 0 <= rank < world_size:
            raise ValueError(f"Invalid joint distributed rank/world size: {rank}/{world_size}")
        return rank, world_size

    @property
    def full_num_batches(self) -> int:
        _, world_size = self._rank_world()
        batches = len(self.dataset) // (world_size * self.batch_size)
        if batches < 1:
            raise ValueError("joint sample_lens must cover at least one full batch per rank")
        return batches

    def set_epoch(self, epoch: int, start_batch: int = 0) -> None:
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise ValueError("epoch must be a non-negative int")
        if isinstance(start_batch, bool) or not isinstance(start_batch, int) or start_batch < 0:
            raise ValueError("start_batch must be a non-negative int")
        self.epoch = epoch
        self.start_batch = start_batch

    def __len__(self) -> int:
        batches = self.full_num_batches
        if not 0 <= self.start_batch <= batches:
            raise ValueError("start_batch must be an epoch-local batch offset")
        return batches - self.start_batch

    def __iter__(self) -> Iterator[list[tuple[str, int]]]:
        rank, world_size = self._rank_world()
        batches = self.full_num_batches
        # Validate the cursor even when DataLoader does not ask for len().
        len(self)
        absolute_start = (self.epoch * batches + self.start_batch) * self.batch_size
        kinds = _weighted_kinds(self.dataset.weights, absolute_start)
        batch: list[tuple[str, int]] = []
        for slot in range((batches - self.start_batch) * self.batch_size):
            kind = next(kinds)
            absolute_slot = absolute_start + slot
            seed = _ticket_seed(self.seed + absolute_slot * world_size + rank)
            batch.append((kind, seed))
            if len(batch) == self.batch_size:
                random.Random(_ticket_seed(seed)).shuffle(batch)
                yield batch
                batch = []
