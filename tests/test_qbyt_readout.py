from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from dma_kws.stage2.readout import resolve_qbyt_readout_mode
from dma_kws.stage2.readout import assert_qbyt_readout_state_loaded
from dma_kws.training.checkpoint_io import (
    QBYT_READOUT_VERSION,
    QBYT_READOUT_VERSION_KEY,
    assert_qbyt_readout_version,
    stamp_qbyt_readout_version,
)


def _payload(*, version: int, mode: str | None = None) -> dict:
    payload = {
        "model_state_dict": {"qbyt.dummy": torch.zeros(1)},
        QBYT_READOUT_VERSION_KEY: version,
    }
    if mode is not None:
        payload["config"] = {"stage2": {"qbyt_readout": {"mode": mode}}}
    return payload


def test_v2_checkpoint_remains_compatible_with_explicit_gru_readout() -> None:
    assert_qbyt_readout_version(
        _payload(version=2),
        source="v2.pt",
        expected_mode="gru_last",
    )


def test_v2_checkpoint_cannot_be_scored_as_eps() -> None:
    with pytest.raises(SystemExit, match="current config expects eps_mean"):
        assert_qbyt_readout_version(
            _payload(version=2),
            source="v2.pt",
            expected_mode="eps_mean",
        )


def test_readout_mode_mismatch_is_rejected_at_current_version() -> None:
    payload = _payload(version=QBYT_READOUT_VERSION, mode="eps_mean")
    with pytest.raises(SystemExit, match="current config expects gru_last"):
        assert_qbyt_readout_version(
            payload,
            source="eps.pt",
            expected_mode="gru_last",
        )


def test_current_eps_checkpoint_is_accepted() -> None:
    payload = stamp_qbyt_readout_version(
        {
            "model_state_dict": {"qbyt.final_pos_fc.weight": torch.zeros(1, 2)},
            "config": {"stage2": {"qbyt_readout": {"mode": "eps_mean"}}},
        }
    )
    assert payload[QBYT_READOUT_VERSION_KEY] == QBYT_READOUT_VERSION
    assert_qbyt_readout_version(
        payload,
        source="eps.pt",
        expected_mode="eps_mean",
    )


def test_readout_config_defaults_to_gru_and_rejects_unknown_mode() -> None:
    assert resolve_qbyt_readout_mode({}) == "gru_last"
    with pytest.raises(ValueError, match="Unsupported QbyT readout mode"):
        resolve_qbyt_readout_mode({"qbyt_readout": {"mode": "bad"}})


def test_non_strict_eval_load_rejects_readout_head_mismatch() -> None:
    with pytest.raises(SystemExit, match="does not carry.*eps_mean"):
        assert_qbyt_readout_state_loaded(
            ["qbyt.final_pos_fc.weight", "qbyt.final_pos_fc.bias"],
            ["qbyt.gru.weight_ih_l0", "qbyt.fc.weight"],
            source="wrong.pt",
            expected_mode="eps_mean",
        )

    assert_qbyt_readout_state_loaded(
        ["encoder.optional"],
        ["unrelated.optional"],
        source="ok.pt",
        expected_mode="eps_mean",
    )
