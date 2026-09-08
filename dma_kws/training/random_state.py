"""Per-rank stochastic state for resumable full encoder adaptation."""

from __future__ import annotations

import random
from typing import Any

import numpy as np
import torch
import torch.distributed as dist


def _world_size() -> int:
    return dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1


def capture_rank_random_states() -> dict[str, Any]:
    """Collect on all ranks; the resulting payload can be persisted by rank zero."""
    numpy_state = np.random.get_state()
    local = {
        "python": random.getstate(),
        "numpy": (numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None,
    }
    states: list[Any] = [None] * _world_size()
    if len(states) > 1:
        dist.all_gather_object(states, local)
    else:
        states[0] = local
    return {"version": 1, "states": states}


def validate_rank_random_states(payload: Any) -> None:
    if not isinstance(payload, dict) or payload.get("version") != 1:
        raise ValueError("Checkpoint is missing supported adaptation random state")
    states = payload.get("states")
    if not isinstance(states, list) or len(states) != _world_size():
        raise ValueError("Cannot resume encoder adaptation with changed world size; use weights-only initialization")
    for state in states:
        if not isinstance(state, dict) or set(state) != {"python", "numpy", "torch", "cuda"}:
            raise ValueError("Checkpoint adaptation random state is invalid")


def restore_rank_random_states(payload: dict[str, Any]) -> None:
    """Restore after model construction, trainer setup and sanity validation."""
    validate_rank_random_states(payload)
    rank = dist.get_rank() if _world_size() > 1 else 0
    state = payload["states"][rank]
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] is not None:
        if not torch.cuda.is_available() or len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError("Cannot restore CUDA random state on a different CUDA device topology")
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])
