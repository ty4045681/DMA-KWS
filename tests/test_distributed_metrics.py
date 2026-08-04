import torch

from dma_kws.training.distributed_metrics import (
    ddp_global_mean_loss,
    gather_unique_sample_values,
)


def test_gather_unique_sample_values_drops_only_stable_id_duplicates(monkeypatch):
    def fake_gather(local_rows):
        peer_rows = torch.tensor(
            [
                [1.0, 5.0, 6.0],
                [0.0, 99.0, 99.0],
                [-1.0, 7.0, 8.0],
            ],
            dtype=local_rows.dtype,
            device=local_rows.device,
        )
        return torch.cat([local_rows, peer_rows], dim=0)

    monkeypatch.setattr(
        "dma_kws.training.distributed_metrics.gather_variable_rows",
        fake_gather,
    )
    values = gather_unique_sample_values(
        torch.tensor([0, 2, -1]),
        torch.tensor([[1.0, 2.0], [3.0, 4.0], [9.0, 10.0]]),
    )

    # Stable id 0 is retained once; negative compatibility ids are all retained.
    torch.testing.assert_close(
        values,
        torch.tensor(
            [[1.0, 2.0], [3.0, 4.0], [9.0, 10.0], [5.0, 6.0], [7.0, 8.0]],
            dtype=torch.float64,
        ),
    )


def test_ddp_global_mean_loss_has_global_value_and_correct_local_gradient_scale():
    # Rank 0 has one valid sample with loss 10; rank 1 has three with total 0.
    # The global displayed loss is 10/4=2.5. DDP will average the two rank
    # gradients, so rank 0's local numerator must be scaled by world_size/count.
    local_mean = torch.tensor(10.0, requires_grad=True)
    corrected = ddp_global_mean_loss(
        local_mean,
        local_count=1,
        global_sum=torch.tensor(10.0),
        global_count=4,
        world_size=2,
    )

    assert float(corrected.detach()) == 2.5
    corrected.backward()
    assert float(local_mean.grad) == 0.5
