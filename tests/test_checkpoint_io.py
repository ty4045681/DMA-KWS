from types import SimpleNamespace

import pytest
import torch

from dma_kws.training.checkpoint_io import restore_best_checkpoint_weights


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
