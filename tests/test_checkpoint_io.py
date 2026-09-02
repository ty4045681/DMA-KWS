from types import SimpleNamespace

import pytest
import torch

from dma_kws.stage2.readout import QbyTAlignmentSpec
from dma_kws.training.checkpoint_io import (
    QBYT_ALIGNMENT_SPEC_KEY,
    QBYT_READOUT_VERSION,
    QBYT_READOUT_VERSION_KEY,
    assert_qbyt_readout_version,
    restore_best_checkpoint_weights,
)


def test_restore_best_checkpoint_weights_loads_selected_state(tmp_path):
    model = torch.nn.Linear(2, 1)
    with torch.no_grad():
        model.weight.zero_()
        model.bias.zero_()

    selected = torch.nn.Linear(2, 1)
    with torch.no_grad():
        selected.weight.fill_(3.0)
        selected.bias.fill_(4.0)
    best_path = tmp_path / "best.ckpt"
    torch.save(
        {"state_dict": selected.state_dict(), "global_step": 17},
        best_path,
    )

    step, source = restore_best_checkpoint_weights(
        model,
        SimpleNamespace(best_model_path=str(best_path)),
        final_step=20,
    )

    assert step == 17
    assert source == f"best_checkpoint@step=17:{best_path}"
    assert torch.equal(model.weight, selected.weight)
    assert torch.equal(model.bias, selected.bias)


def test_restore_best_checkpoint_weights_falls_back_to_final_state(tmp_path):
    model = torch.nn.Linear(2, 1)
    original = {key: value.detach().clone() for key, value in model.state_dict().items()}

    step, source = restore_best_checkpoint_weights(
        model,
        SimpleNamespace(best_model_path=str(tmp_path / "missing.ckpt")),
        final_step=20,
    )

    assert step == 20
    assert source == "final_weights@step=20"
    assert all(torch.equal(model.state_dict()[key], value) for key, value in original.items())


def test_restore_best_checkpoint_weights_strictly_rejects_incompatible_state(tmp_path):
    model = torch.nn.Linear(2, 1)
    best_path = tmp_path / "incompatible.ckpt"
    torch.save(
        {"state_dict": {"unexpected": torch.ones(1)}, "global_step": 17},
        best_path,
    )

    with pytest.raises(RuntimeError, match="state_dict"):
        restore_best_checkpoint_weights(
            model,
            SimpleNamespace(best_model_path=str(best_path)),
            final_step=20,
        )


def test_readout_version_6_qbyt_checkpoint_is_refused_despite_a_valid_spec():
    # Version 6 used the query-relative filler; its weights load shape-for-shape
    # into the one-vs-rest model, so only the version stamp can catch them.
    assert QBYT_READOUT_VERSION == 7
    checkpoint = {
        "model_state_dict": {"qbyt.phone_bias": torch.zeros(3)},
        QBYT_READOUT_VERSION_KEY: 6,
        QBYT_ALIGNMENT_SPEC_KEY: QbyTAlignmentSpec().as_dict(),
    }

    with pytest.raises(SystemExit, match="readout version 6.*uses version 7"):
        assert_qbyt_readout_version(checkpoint, source="v6.pt")
