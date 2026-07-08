from __future__ import annotations

import pytest

pytest.importorskip("torch")

import torch
import torch.nn as nn

from dma_kws.stage2.icefall_encoder import IcefallZipformerEncoder


class _FakeEmbed(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[tuple[torch.Tensor, torch.Tensor]] = []

    def forward(self, feat: torch.Tensor, feat_lengths: torch.Tensor):
        self.calls.append((feat, feat_lengths))
        x = feat + 1.0
        x_lens = feat_lengths - 1
        return x, x_lens


class _FakeEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[tuple[torch.Tensor, torch.Tensor]] = []
        self.encoder_dim = (16, 32, 48)

    def forward(self, x: torch.Tensor, x_lens: torch.Tensor, src_key_padding_mask=None):
        self.calls.append((x, x_lens))
        del src_key_padding_mask
        # Return a non-square tensor to ensure the adapter permutes it back.
        out = x + 2.0
        return out, x_lens - 1


@pytest.fixture
def icefall_encoder() -> tuple[IcefallZipformerEncoder, _FakeEmbed, _FakeEncoder]:
    embed = _FakeEmbed()
    encoder = _FakeEncoder()
    adapter = IcefallZipformerEncoder(embed, encoder, output_dim=48)
    return adapter, embed, encoder


def test_forward_passes_lengths_and_returns_wenet_layout(icefall_encoder):
    adapter, embed, encoder = icefall_encoder

    feat = torch.randn(2, 5, 80)
    feat_lengths = torch.tensor([5, 3], dtype=torch.long)

    encoder_out, encoder_mask = adapter(feat, feat_lengths)

    assert len(embed.calls) == 1
    embed_feat, embed_lens = embed.calls[0]
    assert torch.equal(embed_feat, feat)
    assert torch.equal(embed_lens, feat_lengths)

    assert len(encoder.calls) == 1
    encoder_x, encoder_x_lens = encoder.calls[0]
    assert encoder_x.shape == (5, 2, 80)
    assert torch.equal(encoder_x_lens, feat_lengths - 1)

    assert encoder_out.shape == (2, 5, 80)
    assert encoder_mask.shape == (2, 1, 5)
    assert encoder_mask.dtype == torch.bool
    assert torch.equal(encoder_mask.squeeze(1).sum(dim=1), feat_lengths - 2)
    assert encoder_mask[0, 0].tolist() == [True, True, True, True, False]
    assert encoder_mask[1, 0].tolist() == [True, True, False, False, False]


def test_build_from_params_uses_loaded_modules(monkeypatch):
    fake_embed_cls = type("FakeEmbedCls", (nn.Module,), {"__init__": lambda self, **kwargs: nn.Module.__init__(self)})
    fake_encoder_cls = type(
        "FakeEncoderCls",
        (nn.Module,),
        {
            "__init__": lambda self, **kwargs: nn.Module.__init__(self),
            "encoder_dim": (64, 128),
        },
    )

    monkeypatch.setattr(
        "dma_kws.stage2.icefall_encoder._load_icefall_modules",
        lambda: (fake_embed_cls, fake_encoder_cls),
    )
    monkeypatch.setattr(
        fake_embed_cls,
        "forward",
        lambda self, feat, feat_lengths: (feat, feat_lengths),
        raising=False,
    )
    monkeypatch.setattr(
        fake_encoder_cls,
        "forward",
        lambda self, x, x_lens, src_key_padding_mask=None: (x, x_lens),
        raising=False,
    )

    adapter = IcefallZipformerEncoder.build_from_params({"input_dim": 80}, output_dim=128)

    assert adapter.output_dim == 128
    assert isinstance(adapter.encoder_embed, fake_embed_cls)
    assert isinstance(adapter.encoder, fake_encoder_cls)
