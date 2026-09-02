from __future__ import annotations

import copy

import pytest

torch = pytest.importorskip("torch")

from dma_kws.stage2.readout import (
    QBYT_ALIGNMENT_TOPOLOGY,
    QbyTAlignmentSpec,
    assert_qbyt_alignment_state_loaded,
    normalize_qbyt_alignment_topology,
    resolve_qbyt_alignment,
)
from dma_kws.training.checkpoint_io import (
    QBYT_ALIGNMENT_SPEC_KEY,
    QBYT_READOUT_VERSION,
    QBYT_READOUT_VERSION_KEY,
    assert_qbyt_readout_version,
    checkpoint_qbyt_readout_spec,
    stamp_qbyt_readout_version,
)


def _spec(**overrides) -> dict:
    values = QbyTAlignmentSpec().as_dict()
    values.update(overrides)
    return values


def _payload(
    *,
    version: int | None = QBYT_READOUT_VERSION,
    stamped_spec: dict | None = None,
    config_spec: dict | None = None,
    carries_qbyt: bool = True,
) -> dict:
    payload = {
        "model_state_dict": {
            ("qbyt.alignment_emission.weight" if carries_qbyt else "encoder.weight"):
            torch.zeros(1)
        },
    }
    if version is not None:
        payload[QBYT_READOUT_VERSION_KEY] = version
    if stamped_spec is not None:
        payload[QBYT_ALIGNMENT_SPEC_KEY] = stamped_spec
    if config_spec is not None:
        payload["config"] = {"stage2": {"qbyt_alignment": config_spec}}
    return payload


def test_v6_alignment_defaults_are_the_only_topology() -> None:
    assert QBYT_READOUT_VERSION == 7
    assert normalize_qbyt_alignment_topology(None) == QBYT_ALIGNMENT_TOPOLOGY
    assert resolve_qbyt_alignment({}).as_dict() == {
        "topology": "keyword_filler_segmental_crf_v1",
        "min_phone_duration_frames": 1,
        "max_phone_duration_frames": 8,
        "max_inter_phone_gap_frames": 3,
        "max_keyword_span_frames": 30,
        "local_context_kernel": 5,
        "weakest_phone_temperature": 0.2,
        "weakest_phone_weight": 1.0,
    }


def test_alignment_config_canonicalizes_numeric_strings() -> None:
    alignment = resolve_qbyt_alignment(
        {
            "qbyt_alignment": {
                "topology": "keyword_filler_segmental_crf_v1",
                "min_phone_duration_frames": "2",
                "max_phone_duration_frames": "9",
                "max_inter_phone_gap_frames": "3",
                "max_keyword_span_frames": "40",
                "local_context_kernel": "7",
                "weakest_phone_temperature": "0.35",
                "weakest_phone_weight": "0.75",
            }
        }
    )
    assert alignment.as_dict() == {
        "topology": "keyword_filler_segmental_crf_v1",
        "min_phone_duration_frames": 2,
        "max_phone_duration_frames": 9,
        "max_inter_phone_gap_frames": 3,
        "max_keyword_span_frames": 40,
        "local_context_kernel": 7,
        "weakest_phone_temperature": 0.35,
        "weakest_phone_weight": 0.75,
    }


@pytest.mark.parametrize("mode", ["gru_last", "eps_mean", "eps_softmin"])
def test_historical_readout_modes_are_not_configuration_options(mode: str) -> None:
    with pytest.raises(ValueError, match="legacy GRU/EPS switch"):
        resolve_qbyt_alignment({"qbyt_readout": {"mode": mode}})
    with pytest.raises(ValueError, match="Historical GRU/EPS"):
        resolve_qbyt_alignment({"qbyt_alignment": {"topology": mode}})


def test_target_only_bounded_segmental_topology_is_rejected() -> None:
    with pytest.raises(ValueError, match="target-only bounded-segmental"):
        resolve_qbyt_alignment(
            {"qbyt_alignment": {"topology": "bounded_segmental_v1"}}
        )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"min_phone_duration_frames": True}, "must be an integer"),
        ({"min_phone_duration_frames": 0}, "must be an integer"),
        ({"max_phone_duration_frames": 1.5}, "must be an integer"),
        (
            {"min_phone_duration_frames": 4, "max_phone_duration_frames": 3},
            "max_phone_duration_frames must be >=",
        ),
        (
            {"max_phone_duration_frames": 8, "max_keyword_span_frames": 7},
            "max_keyword_span_frames must be >=",
        ),
        (
            {"weakest_phone_temperature": True},
            "temperature must be a finite number",
        ),
        (
            {"weakest_phone_temperature": 0.0},
            "temperature must be a finite number",
        ),
        (
            {"weakest_phone_temperature": float("nan")},
            "temperature must be a finite number",
        ),
        ({"weakest_phone_weight": True}, "weakest_phone_weight must be a finite"),
        ({"weakest_phone_weight": -0.1}, "weakest_phone_weight must be a finite"),
        ({"local_context_kernel": 4}, "local_context_kernel must be odd"),
    ],
)
def test_alignment_config_rejects_invalid_values(overrides: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        resolve_qbyt_alignment({"qbyt_alignment": _spec(**overrides)})


def test_alignment_config_rejects_unknown_fields() -> None:
    with pytest.raises(ValueError, match="unknown fields.*gap_penalty"):
        resolve_qbyt_alignment(
            {"qbyt_alignment": {**_spec(), "gap_penalty": 0.5}}
        )


def test_stamp_records_version_and_complete_actual_alignment() -> None:
    configured = _spec(
        max_inter_phone_gap_frames=1,
        max_keyword_span_frames=24,
        weakest_phone_temperature=0.3,
        weakest_phone_weight=0.7,
    )
    payload = stamp_qbyt_readout_version(
        {
            "model_state_dict": {"qbyt.alignment_emission.weight": torch.zeros(1)},
            "config": {"stage2": {"qbyt_alignment": configured}},
        },
        alignment=configured,
    )
    assert payload[QBYT_READOUT_VERSION_KEY] == QBYT_READOUT_VERSION
    assert payload[QBYT_ALIGNMENT_SPEC_KEY] == configured
    assert checkpoint_qbyt_readout_spec(payload).as_dict() == configured


def test_stamp_requires_and_accepts_explicit_spec() -> None:
    explicit = QbyTAlignmentSpec(local_context_kernel=7)
    payload = stamp_qbyt_readout_version(
        {"model_state_dict": {"qbyt.alignment_emission.weight": torch.zeros(1)}},
        alignment=explicit,
    )
    assert payload[QBYT_ALIGNMENT_SPEC_KEY] == explicit.as_dict()

def test_stamp_rejects_explicit_spec_that_disagrees_with_embedded_config() -> None:
    with pytest.raises(ValueError, match="disagrees with config"):
        stamp_qbyt_readout_version(
            {
                "config": {
                    "stage2": {
                        "qbyt_alignment": _spec(max_keyword_span_frames=20)
                    }
                }
            },
            alignment=QbyTAlignmentSpec(max_keyword_span_frames=30),
        )


def test_current_checkpoint_requires_exact_stamped_and_configured_spec() -> None:
    actual = _spec(max_inter_phone_gap_frames=0, weakest_phone_temperature=0.4)
    payload = _payload(stamped_spec=actual, config_spec=actual)
    assert_qbyt_readout_version(
        payload,
        source="v6.pt",
        expected_alignment=QbyTAlignmentSpec(**actual),
    )

    expected = QbyTAlignmentSpec(**{**actual, "max_inter_phone_gap_frames": 1})
    with pytest.raises(SystemExit, match="current config expects.*max_inter_phone_gap_frames"):
        assert_qbyt_readout_version(
            payload,
            source="v6.pt",
            expected_alignment=expected,
        )


@pytest.mark.parametrize("version", [None, 1, 2, 3, 4, 5])
def test_pre_v6_qbyt_weights_are_always_rejected(version: int | None) -> None:
    payload = _payload(
        version=version,
        stamped_spec=None,
        config_spec=None,
    )
    with pytest.raises(SystemExit, match="Readout semantics differ"):
        assert_qbyt_readout_version(
            payload,
            source="legacy.pt",
        )


def test_encoder_only_checkpoint_remains_a_valid_training_warm_start() -> None:
    # The shared checkpoint loader also sees Stage-I/Icefall and adapter exports.
    # With no QbyT weights, readout metadata is irrelevant and must be skipped.
    assert_qbyt_readout_version(
        _payload(
            version=4,
            stamped_spec=None,
            config_spec=None,
            carries_qbyt=False,
        ),
        source="encoder-only.pt",
    )
    assert_qbyt_readout_version(
        {"model": {"encoder.layer.weight": torch.zeros(1)}},
        source="icefall.pt",
    )


def test_v6_checkpoint_rejects_missing_partial_or_conflicting_spec() -> None:
    missing = _payload(stamped_spec=None, config_spec=_spec())
    with pytest.raises(SystemExit, match=QBYT_ALIGNMENT_SPEC_KEY):
        assert_qbyt_readout_version(missing, source="missing.pt")

    partial = _payload(
        stamped_spec={"topology": QBYT_ALIGNMENT_TOPOLOGY},
        config_spec=None,
    )
    with pytest.raises(SystemExit, match="missing="):
        assert_qbyt_readout_version(partial, source="partial.pt")

    conflicting = _payload(
        stamped_spec=_spec(max_keyword_span_frames=30),
        config_spec=_spec(max_keyword_span_frames=40),
    )
    with pytest.raises(SystemExit, match="disagrees with config"):
        assert_qbyt_readout_version(conflicting, source="conflicting.pt")


def test_lora_adapter_payload_is_also_version_and_spec_checked() -> None:
    payload = {
        "lora_state_dict": {"phone_matchor.lora_A": torch.zeros(1)},
        QBYT_READOUT_VERSION_KEY: 4,
    }
    with pytest.raises(SystemExit, match="version 4"):
        assert_qbyt_readout_version(payload, source="legacy-adapter.pt")


def test_non_strict_load_rejects_any_qbyt_state_mismatch() -> None:
    with pytest.raises(SystemExit, match="complete.*keyword_filler_segmental_crf_v1"):
        assert_qbyt_alignment_state_loaded(
            ["qbyt.alignment_emission.weight"],
            ["qbyt.final_pos_fc.weight"],
            source="wrong.pt",
        )

    assert_qbyt_alignment_state_loaded(
        ["encoder.optional"],
        ["unrelated.optional"],
        source="ok.pt",
    )


def test_stamped_spec_is_a_copy_of_mutable_input() -> None:
    raw = _spec()
    payload = stamp_qbyt_readout_version({}, alignment=raw)
    saved = copy.deepcopy(payload[QBYT_ALIGNMENT_SPEC_KEY])
    raw["max_keyword_span_frames"] = 999
    assert payload[QBYT_ALIGNMENT_SPEC_KEY] == saved
