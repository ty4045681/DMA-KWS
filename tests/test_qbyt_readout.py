from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from dma_kws.stage2.readout import assert_qbyt_readout_state_loaded
from dma_kws.stage2.readout import resolve_qbyt_readout, resolve_qbyt_readout_mode
from dma_kws.training.checkpoint_io import (
    QBYT_READOUT_VERSION,
    QBYT_READOUT_VERSION_KEY,
    assert_qbyt_readout_version,
    stamp_qbyt_readout_version,
)


def _payload(
    *,
    version: int,
    mode: str | None = None,
    temperature: float | None = None,
) -> dict:
    payload = {
        "model_state_dict": {"qbyt.dummy": torch.zeros(1)},
        QBYT_READOUT_VERSION_KEY: version,
    }
    if mode is not None:
        readout = {"mode": mode}
        if temperature is not None:
            readout["temperature"] = temperature
        payload["config"] = {"stage2": {"qbyt_readout": readout}}
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


def test_v3_gru_and_eps_mean_checkpoints_remain_compatible() -> None:
    assert_qbyt_readout_version(
        _payload(version=3, mode="gru_last"),
        source="v3-gru.pt",
        expected_mode="gru_last",
    )
    assert_qbyt_readout_version(
        _payload(version=3, mode="eps_mean"),
        source="v3-mean.pt",
        expected_mode="eps_mean",
    )


def test_v3_checkpoint_cannot_claim_softmin_semantics() -> None:
    with pytest.raises(SystemExit, match="version 3 cannot carry mode"):
        assert_qbyt_readout_version(
            _payload(version=3, mode="eps_softmin", temperature=0.5),
            source="v3-softmin.pt",
            expected_mode="eps_softmin",
            expected_temperature=0.5,
        )


def test_readout_mode_mismatch_is_rejected_at_current_version() -> None:
    payload = _payload(version=QBYT_READOUT_VERSION, mode="eps_mean")
    with pytest.raises(SystemExit, match="current config expects gru_last"):
        assert_qbyt_readout_version(
            payload,
            source="eps.pt",
            expected_mode="gru_last",
        )


def test_readout_temperature_mismatch_is_rejected_at_current_version() -> None:
    payload = _payload(
        version=QBYT_READOUT_VERSION,
        mode="eps_mean",
        temperature=0.5,
    )
    with pytest.raises(SystemExit, match=r"expects eps_mean\(temperature=1\)"):
        assert_qbyt_readout_version(
            payload,
            source="eps.pt",
            expected_mode="eps_mean",
            expected_temperature=1.0,
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


def test_current_softmin_checkpoint_requires_and_matches_temperature() -> None:
    payload = _payload(
        version=QBYT_READOUT_VERSION,
        mode="eps_softmin",
        temperature=0.5,
    )
    assert_qbyt_readout_version(
        payload,
        source="softmin.pt",
        expected_mode="eps_softmin",
        expected_temperature=0.5,
    )

    with pytest.raises(SystemExit, match=r"temperature=0\.75"):
        assert_qbyt_readout_version(
            payload,
            source="softmin.pt",
            expected_mode="eps_softmin",
            expected_temperature=0.75,
        )


def test_current_softmin_checkpoint_without_explicit_temperature_is_rejected() -> None:
    with pytest.raises(SystemExit, match="explicitly record.*temperature"):
        assert_qbyt_readout_version(
            _payload(version=QBYT_READOUT_VERSION, mode="eps_softmin"),
            source="ambiguous-softmin.pt",
            expected_mode="eps_softmin",
            expected_temperature=1.0,
        )


def test_current_eps_head_without_config_is_ambiguous() -> None:
    payload = {
        "model_state_dict": {
            "qbyt.final_pos_fc.weight": torch.zeros(1, 2),
        },
        QBYT_READOUT_VERSION_KEY: QBYT_READOUT_VERSION,
    }
    with pytest.raises(SystemExit, match="identify EPS mean versus soft-min"):
        assert_qbyt_readout_version(
            payload,
            source="ambiguous.pt",
            expected_mode="eps_mean",
        )


def test_readout_config_defaults_to_gru_and_rejects_unknown_mode() -> None:
    assert QBYT_READOUT_VERSION == 4
    assert resolve_qbyt_readout_mode({}) == "gru_last"
    with pytest.raises(ValueError, match="Unsupported QbyT readout mode"):
        resolve_qbyt_readout_mode({"qbyt_readout": {"mode": "bad"}})


def test_softmin_readout_config_resolves_temperature() -> None:
    readout = resolve_qbyt_readout(
        {"qbyt_readout": {"mode": "eps_softmin", "temperature": "0.5"}}
    )

    assert readout.mode == "eps_softmin"
    assert readout.temperature == 0.5
    assert readout.as_dict() == {
        "mode": "eps_softmin",
        "temperature": 0.5,
    }


@pytest.mark.parametrize("temperature", [True, 0.0, -1.0, float("nan"), float("inf")])
def test_readout_config_rejects_invalid_temperature(temperature) -> None:
    with pytest.raises(ValueError, match="temperature must be a finite number"):
        resolve_qbyt_readout(
            {
                "qbyt_readout": {
                    "mode": "eps_softmin",
                    "temperature": temperature,
                }
            }
        )


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
