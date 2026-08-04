import pytest

from dma_kws.stage2.objective import (
    CURRENT_SEQUENCE_OBJECTIVE,
    LEGACY_SEQUENCE_OBJECTIVE,
    assert_sequence_objective_matches,
    checkpoint_sequence_objective,
    resolve_sequence_objective,
)


def test_missing_runtime_config_uses_current_objective():
    assert resolve_sequence_objective({}) == CURRENT_SEQUENCE_OBJECTIVE


def test_missing_checkpoint_metadata_is_identified_as_legacy():
    assert checkpoint_sequence_objective({"state_dict": {}}) == LEGACY_SEQUENCE_OBJECTIVE
    assert checkpoint_sequence_objective(
        {"config": {"stage2": {}}}
    ) == LEGACY_SEQUENCE_OBJECTIVE


def test_full_resume_accepts_identical_sequence_objective():
    stage2 = {"sequence_loss": CURRENT_SEQUENCE_OBJECTIVE.as_dict()}
    checkpoint = {"config": {"stage2": stage2}}

    assert_sequence_objective_matches(checkpoint, stage2, source="same.ckpt")


def test_full_resume_rejects_legacy_checkpoint_under_current_objective():
    with pytest.raises(ValueError, match="Do not resume optimizer state"):
        assert_sequence_objective_matches(
            {"config": {"stage2": {}}},
            {"sequence_loss": CURRENT_SEQUENCE_OBJECTIVE.as_dict()},
            source="legacy.ckpt",
        )


def test_membership_completion_combination_is_rejected():
    with pytest.raises(ValueError, match="completion_weight requires"):
        resolve_sequence_objective(
            {
                "sequence_loss": {
                    "target_mode": "membership",
                    "progress_weight": 1.0,
                    "completion_weight": 0.5,
                    "normalization": "token",
                }
            }
        )
