import pytest

from dma_kws.stage2.objective import (
    CURRENT_SEQUENCE_OBJECTIVE,
    assert_sequence_objective_matches,
    checkpoint_sequence_objective,
    resolve_sequence_objective,
)


def test_missing_runtime_config_uses_current_objective():
    assert resolve_sequence_objective({}) == CURRENT_SEQUENCE_OBJECTIVE


def test_v5_checkpoint_requires_sequence_objective_metadata():
    with pytest.raises(ValueError, match="must record config.stage2.sequence_loss"):
        checkpoint_sequence_objective({"state_dict": {}})
    with pytest.raises(ValueError, match="must record config.stage2.sequence_loss"):
        checkpoint_sequence_objective({"config": {"stage2": {}}})


def test_full_resume_accepts_identical_sequence_objective():
    stage2 = {"sequence_loss": CURRENT_SEQUENCE_OBJECTIVE.as_dict()}
    checkpoint = {"config": {"stage2": stage2}}

    assert_sequence_objective_matches(checkpoint, stage2, source="same.ckpt")


def test_full_resume_rejects_checkpoint_without_objective():
    with pytest.raises(ValueError, match="must record config.stage2.sequence_loss"):
        assert_sequence_objective_matches(
            {"config": {"stage2": {}}},
            {"sequence_loss": CURRENT_SEQUENCE_OBJECTIVE.as_dict()},
            source="legacy.ckpt",
        )


def test_unknown_sequence_loss_field_is_rejected():
    with pytest.raises(ValueError, match="unknown fields.*extra_weight"):
        resolve_sequence_objective(
            {
                "sequence_loss": {
                    "target_mode": "ordered_contiguous_prefix",
                    "progress_weight": 0.3,
                    "extra_weight": 0.5,
                    "normalization": "sample",
                }
            }
        )
