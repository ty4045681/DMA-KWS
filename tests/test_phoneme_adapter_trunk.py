import pytest

torch = pytest.importorskip("torch")

from dma_kws.phoneme_adapter.trunk import TRUNK_TYPES, build_trunk


def _mask(lengths: list[int], frames: int) -> torch.Tensor:
    lens = torch.tensor(lengths, dtype=torch.long)
    return (torch.arange(frames).unsqueeze(0) < lens.unsqueeze(1)).unsqueeze(1)


@pytest.mark.parametrize("trunk_type", ["linear", "mlp", "conv"])
def test_trunk_shapes(trunk_type):
    trunk = build_trunk({"type": trunk_type, "output_dim": 32}, input_dim=16, causal=False)
    x = torch.randn(2, 7, 16)
    out = trunk(x, _mask([7, 4], 7))

    assert out.shape == (2, 7, 32)
    assert trunk.output_dim == 32


@pytest.mark.parametrize("trunk_type", ["linear", "mlp", "conv"])
def test_trunk_zeros_padded_frames(trunk_type):
    trunk = build_trunk({"type": trunk_type, "output_dim": 8}, input_dim=4, causal=False)
    trunk.eval()
    out = trunk(torch.randn(1, 6, 4), _mask([3], 6))

    assert torch.all(out[0, 3:] == 0)


def test_conv_trunk_is_causal_when_requested():
    """A causal trunk must not let future frames change the current output.

    Left-padding is the only thing standing between "streaming-safe" and a trunk
    that trains with lookahead the deployed pipeline cannot provide, and that
    failure is invisible in the loss curve.
    """
    trunk = build_trunk(
        {"type": "conv", "output_dim": 8, "num_layers": 2, "kernel_size": 5},
        input_dim=4,
        causal=True,
    )
    trunk.eval()

    x = torch.randn(1, 12, 4)
    perturbed = x.clone()
    perturbed[:, 8:] += 10.0

    with torch.no_grad():
        base = trunk(x, None)
        changed = trunk(perturbed, None)

    torch.testing.assert_close(base[:, :8], changed[:, :8])
    assert not torch.allclose(base[:, 8:], changed[:, 8:])


def test_conv_trunk_non_causal_sees_future():
    trunk = build_trunk(
        {"type": "conv", "output_dim": 8, "num_layers": 1, "kernel_size": 5},
        input_dim=4,
        causal=False,
    )
    trunk.eval()

    x = torch.randn(1, 12, 4)
    perturbed = x.clone()
    perturbed[:, 8:] += 10.0

    with torch.no_grad():
        base = trunk(x, None)
        changed = trunk(perturbed, None)

    assert not torch.allclose(base[:, :8], changed[:, :8])


def test_conformer_trunk_requires_chunk_size_when_causal():
    with pytest.raises(ValueError, match="chunk_size"):
        build_trunk({"type": "conformer", "output_dim": 8}, input_dim=4, causal=True)


def test_unknown_trunk_type_is_rejected():
    with pytest.raises(ValueError, match="Unknown trunk.type"):
        build_trunk({"type": "transformer"}, input_dim=4, causal=False)

    assert set(TRUNK_TYPES) == {"linear", "mlp", "conv", "conformer"}
