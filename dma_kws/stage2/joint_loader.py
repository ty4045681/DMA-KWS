"""Consumed-batch checkpointing for deterministic joint adaptation tickets."""

from __future__ import annotations

import math
import os

import torch.distributed as dist
from torch.utils.data import DataLoader, Sampler

from dma_kws.training.ddp import process_rank


def world_size() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return int(os.environ.get("WORLD_SIZE", "1"))


class JointDataLoader(DataLoader):
    """Save consumed tickets, never the prefetched iterator position.

    Lightning recognizes the state_dict/load_state_dict protocol and restores it
    before creating workers. The module acknowledges each completed train step.
    Epoch lengths stay full-sized: Lightning already restores its batch counter.
    """

    def __init__(self, *args, data_signature: str, accumulation_steps: int = 1, **kwargs):
        super().__init__(*args, **kwargs)
        if isinstance(accumulation_steps, bool) or not isinstance(accumulation_steps, int) or accumulation_steps < 1:
            raise ValueError("Joint accumulation_steps must be a positive integer")
        self.accumulation_steps = accumulation_steps
        self.data_signature = data_signature
        self.consumed_batches = 0

    def __len__(self) -> int:
        batches = self.batch_sampler.full_num_batches
        if batches % self.accumulation_steps:
            raise ValueError(
                "Joint batches per rank per epoch must be divisible by stage2.accumulate_grad_batches; "
                "adjust adapt.sample_lens or batch_size_per_gpu"
            )
        return batches

    def __iter__(self):
        epoch, offset = divmod(self.consumed_batches, len(self))
        self.batch_sampler.set_epoch(epoch, start_batch=offset)
        return super().__iter__()

    def mark_consumed(self) -> None:
        self.consumed_batches += 1

    def state_dict(self) -> dict:
        if self.consumed_batches % self.accumulation_steps:
            raise ValueError("Joint checkpoints must be saved at optimizer boundaries, after gradient accumulation")
        return {
            "version": 1,
            "consumed_batches": self.consumed_batches,
            "data_signature": self.data_signature,
            "world_size": world_size(),
            "batch_size": self.batch_sampler.batch_size,
            "batches_per_epoch": len(self),
            "accumulation_steps": self.accumulation_steps,
        }

    def load_state_dict(self, state: dict) -> None:
        expected = self.state_dict()
        for key in expected.keys() - {"consumed_batches"}:
            if state.get(key) != expected[key]:
                raise ValueError(
                    f"Cannot resume joint sampling: {key} changed "
                    f"({state.get(key)!r} -> {expected[key]!r})"
                )
        consumed = state.get("consumed_batches")
        if isinstance(consumed, bool) or not isinstance(consumed, int) or consumed < 0:
            raise ValueError("Invalid joint sampling consumed_batches")
        if consumed % self.accumulation_steps:
            raise ValueError("Cannot resume joint checkpoint saved between optimizer boundaries")
        self.consumed_batches = consumed


class JointEvalSampler(Sampler[int]):
    """Shard validation at iterator time, including Lightning-launched DDP.

    Equal-length ranks avoid unmatched DDP forwards. Metric sample IDs remove
    the padding duplicates before computing exact diagnostics.
    """

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self) -> int:
        return math.ceil(len(self.dataset) / world_size())

    def __iter__(self):
        size = len(self.dataset)
        rank, replicas = process_rank(), world_size()
        return iter((rank + i * replicas) % size for i in range(len(self)))
