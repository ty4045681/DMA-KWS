"""Small distributed reductions for metrics that are ratios of global sums."""

from __future__ import annotations

import torch
import torch.distributed as dist


def sum_across_processes(values: torch.Tensor) -> torch.Tensor:
    """Return a detached sum of ``values`` over all initialized processes.

    This helper is intentionally lower level than Lightning's ``sync_dist``.
    Averaging an already-computed ratio on every rank is generally wrong when
    its denominator differs by rank; callers instead pack all numerators and
    denominators into one tensor, sum them once, and divide afterwards.
    """
    totals = values.detach().clone()
    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
    return totals


def ddp_global_mean_loss(
    local_mean: torch.Tensor,
    *,
    local_count: int,
    global_sum: torch.Tensor,
    global_count: int,
    world_size: int | None = None,
) -> torch.Tensor:
    """Return a global-mean scalar with the correct DDP gradient scale.

    DDP averages gradients from ranks, so backpropagating each rank's local
    mean produces a mean-of-rank-means when valid counts differ.  Scale the
    differentiable local numerator by ``world_size/global_count`` and replace
    only its forward value with the detached global mean.  The returned tensor
    therefore logs identically on every rank while its DDP-averaged gradient is
    exactly the gradient of ``sum(all valid losses) / global_count``.
    """
    if local_count < 0 or global_count < 0:
        raise ValueError("loss counts must be non-negative")
    if global_count == 0:
        return local_mean * 0.0
    if world_size is None:
        world_size = (
            dist.get_world_size()
            if dist.is_available() and dist.is_initialized()
            else 1
        )
    if world_size <= 0:
        raise ValueError("world_size must be positive")

    local_sum = local_mean * int(local_count)
    gradient_value = local_sum * (float(world_size) / float(global_count))
    global_mean = global_sum.to(local_mean) / float(global_count)
    return gradient_value + (global_mean - gradient_value.detach())


def gather_variable_rows(rows: torch.Tensor) -> torch.Tensor:
    """Gather a variable number of 2-D rows from every initialized process.

    ``DistributedSampler`` may pad validation shards by repeating samples.  A
    simple all-reduce cannot distinguish those repeats, so callers first gather
    per-sample rows and then de-duplicate them by a stable sample id.
    """
    if rows.ndim != 2:
        raise ValueError(f"Expected a 2-D row tensor, got shape {tuple(rows.shape)}")

    rows = rows.detach()
    if not (
        dist.is_available()
        and dist.is_initialized()
        and dist.get_world_size() > 1
    ):
        return rows.clone()

    world_size = dist.get_world_size()
    local_size = torch.tensor([rows.size(0)], device=rows.device, dtype=torch.long)
    gathered_sizes = [torch.zeros_like(local_size) for _ in range(world_size)]
    dist.all_gather(gathered_sizes, local_size)
    sizes = [int(size.item()) for size in gathered_sizes]
    max_size = max(sizes, default=0)
    if max_size == 0:
        return rows.new_empty((0, rows.size(1)))

    padded = rows.new_zeros((max_size, rows.size(1)))
    if rows.size(0):
        padded[: rows.size(0)] = rows
    gathered = [torch.empty_like(padded) for _ in range(world_size)]
    dist.all_gather(gathered, padded)
    return torch.cat(
        [rank_rows[:rank_size] for rank_rows, rank_size in zip(gathered, sizes)],
        dim=0,
    )


def gather_unique_sample_values(
    sample_ids: torch.Tensor,
    values: torch.Tensor,
) -> torch.Tensor:
    """Gather values and retain one row for each non-negative sample id.

    Dataset-provided ids are non-negative and therefore identify padding
    duplicates exactly.  Negative ids are reserved for legacy/manual batches
    that do not carry a stable id; every such row is retained so this helper
    remains backward compatible instead of guessing that two samples match.
    """
    if sample_ids.ndim != 1:
        raise ValueError(
            f"Expected 1-D sample_ids, got shape {tuple(sample_ids.shape)}"
        )
    if values.ndim != 2 or values.size(0) != sample_ids.numel():
        raise ValueError(
            "values must be a 2-D tensor with one row per sample id; "
            f"got ids {tuple(sample_ids.shape)} and values {tuple(values.shape)}"
        )

    # float64 represents integer ids exactly for all realistic dataset sizes
    # and lets ids and metric columns travel in a single collective call.
    packed = torch.cat(
        [
            sample_ids.detach().to(device=values.device, dtype=torch.float64).unsqueeze(1),
            values.detach().to(dtype=torch.float64),
        ],
        dim=1,
    )
    gathered = gather_variable_rows(packed)
    if gathered.size(0) == 0:
        return gathered[:, 1:]

    retained: list[int] = []
    seen: set[int] = set()
    for row_index, raw_id in enumerate(gathered[:, 0].tolist()):
        sample_id = int(raw_id)
        if sample_id >= 0:
            if sample_id in seen:
                continue
            seen.add(sample_id)
        retained.append(row_index)

    indices = torch.tensor(retained, device=gathered.device, dtype=torch.long)
    return gathered.index_select(0, indices)[:, 1:]
