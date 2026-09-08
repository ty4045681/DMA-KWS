"""Duration schedules and serializable stochastic state for encoder adaptation."""

from __future__ import annotations

import copy
import random
import socket
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

from dma_kws.stage2.encoder_schedule import EncoderFinetuneSchedule
from dma_kws.stage2.icefall_encoder import IcefallZipformerEncoder
from dma_kws.training.random_state import (
    capture_rank_random_states,
    restore_rank_random_states,
    validate_rank_random_states,
)


def test_duration_clock_counts_unpadded_frames_and_restores_exactly():
    config = {"fbank": {"frame_shift": 10}, "adapt": {
        "encoder_schedule": {"start_batch_count": 100000.0, "reference_duration": 600.0},
    }}
    clock = EncoderFinetuneSchedule.from_config(config)
    assert clock.batch_count == 100000.0
    # Different per-rank and per-microbatch lengths contribute their sum once.
    clock.advance(101 + 79)
    clock.advance(103 + 87)
    assert clock.consumed_frames == 370
    assert clock.batch_count == 100000.0 + 3.7 / 600
    continued = EncoderFinetuneSchedule.from_config(config)
    continued.load_state_dict(clock.state_dict())
    clock.advance(137)
    continued.advance(137)
    assert continued.state_dict() == clock.state_dict()


@pytest.mark.parametrize("key,value", [
    ("start_batch_count", -1), ("start_batch_count", float("nan")),
    ("reference_duration", 0), ("reference_duration", float("inf")),
])
def test_encoder_schedule_rejects_invalid_configuration(key, value):
    with pytest.raises(ValueError, match="finite"):
        EncoderFinetuneSchedule.from_config({"adapt": {"encoder_schedule": {key: value}}})


@pytest.mark.parametrize("value", [[], False, "oops"])
def test_encoder_schedule_rejects_non_mapping(value):
    with pytest.raises(ValueError, match="mapping"):
        EncoderFinetuneSchedule.from_config({"adapt": {"encoder_schedule": value}})


@pytest.mark.parametrize("change", ["frame_shift", "reference_duration", "start_batch_count", "frames"])
def test_schedule_rejects_incompatible_or_invalid_checkpoint(change):
    clock = EncoderFinetuneSchedule.from_config({})
    state = copy.deepcopy(clock.state_dict())
    if change == "frames":
        state["consumed_frames"] = 1.5
    elif change == "frame_shift":
        state["config"]["frame_shift_seconds"] *= 2
    else:
        state["config"][change] += 1
    with pytest.raises(ValueError, match="configuration|consumed_frames"):
        clock.load_state_dict(state)


def test_wrapper_sets_nested_subsampling_and_encoder_schedule_counts():
    class Schedule(nn.Module):
        def __init__(self):
            super().__init__()
            self.batch_count = None
            self.name = None

    embed, encoder = nn.Sequential(Schedule()), nn.Sequential(nn.Sequential(Schedule()))
    wrapper = IcefallZipformerEncoder(embed, encoder, output_dim=8)
    wrapper.set_batch_count(100001.25)
    assert embed[0].batch_count == encoder[0][0].batch_count == 100001.25
    assert embed[0].name == "encoder_embed.0"
    assert encoder[0][0].name == "encoder.0.0"
    wrapper.eval()
    assert embed[0].batch_count == 100001.25  # eval never advances stored progress


def test_rank_random_state_roundtrip_is_weights_only_compatible(tmp_path):
    before = capture_rank_random_states()
    try:
        random.seed(91)
        np.random.seed(72)
        torch.manual_seed(81)
        payload = capture_rank_random_states()
        expected = (random.random(), np.random.rand(3), torch.rand(3))
        path = tmp_path / "rng.pt"
        torch.save(payload, path)
        loaded = torch.load(path, weights_only=True)
        # Consume extra draws to model reconstruction and sanity validation.
        random.random()
        np.random.rand(10)
        torch.rand(10)
        restore_rank_random_states(loaded)
        assert random.random() == expected[0]
        np.testing.assert_array_equal(np.random.rand(3), expected[1])
        torch.testing.assert_close(torch.rand(3), expected[2], rtol=0, atol=0)
    finally:
        restore_rank_random_states(before)


def _distributed_random_state_worker(rank, rendezvous_path, checkpoint_path):
    dist.init_process_group(
        "gloo", init_method=Path(rendezvous_path).as_uri(), rank=rank,
        world_size=2, timeout=timedelta(seconds=30),
    )
    try:
        random.seed(100 + rank)
        np.random.seed(200 + rank)
        torch.manual_seed(300 + rank)
        payload = capture_rank_random_states()
        expected = (random.random(), np.random.rand(3), torch.rand(3))
        assert not torch.equal(payload["states"][0]["torch"], payload["states"][1]["torch"])
        if rank == 0:
            torch.save(payload, checkpoint_path)
        dist.barrier()
        loaded = torch.load(checkpoint_path, weights_only=True)
        random.seed(901)
        np.random.seed(902)
        torch.manual_seed(903)
        restore_rank_random_states(loaded)
        assert random.random() == expected[0]
        np.testing.assert_array_equal(np.random.rand(3), expected[1])
        torch.testing.assert_close(torch.rand(3), expected[2], rtol=0, atol=0)
        with pytest.raises(ValueError, match="world size"):
            validate_rank_random_states({"version": 1, "states": loaded["states"][:1]})
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_available() or not dist.is_gloo_available(), reason="Gloo is unavailable")
def test_random_state_restores_each_rank_from_a_shared_checkpoint(tmp_path, monkeypatch):
    # These workers share one host. Avoid hostname resolution selecting a VPN or
    # otherwise non-local interface on developer laptops.
    loopback = next((name for _, name in socket.if_nameindex() if name in {"lo", "lo0"}), None)
    if loopback is not None:
        monkeypatch.setenv("GLOO_SOCKET_IFNAME", loopback)
    context = mp.spawn(
        _distributed_random_state_worker,
        args=(str(tmp_path / "rendezvous"), str(tmp_path / "random_state.pt")),
        nprocs=2, join=False,
    )
    deadline = time.monotonic() + 45
    try:
        while not context.join(timeout=1):
            if time.monotonic() > deadline:
                pytest.fail("Two-rank random-state checkpoint test exceeded 45 seconds")
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
