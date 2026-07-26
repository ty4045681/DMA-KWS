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
        self.chunk_size = (-1,)
        self.left_context_frames = (-1,)

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
    # _FakeEmbed drops one frame and _FakeEncoder drops another, so the valid
    # lengths are feat_lengths - 2 == [3, 1].
    assert torch.equal(encoder_mask.squeeze(1).sum(dim=1), feat_lengths - 2)
    assert encoder_mask[0, 0].tolist() == [True, True, True, False, False]
    assert encoder_mask[1, 0].tolist() == [True, False, False, False, False]


def test_apply_stream_config_sets_single_value_tuples(icefall_encoder):
    adapter, _embed, encoder = icefall_encoder

    adapter.apply_stream_config((16,), (64,))

    assert encoder.chunk_size == (16,)
    assert encoder.left_context_frames == (64,)


def test_apply_stream_config_accepts_multi_latency_tuples(icefall_encoder):
    adapter, _embed, encoder = icefall_encoder

    adapter.apply_stream_config((16, 32, 64, -1), (64, 128, 256, -1))

    assert encoder.chunk_size == (16, 32, 64, -1)
    assert encoder.left_context_frames == (64, 128, 256, -1)


def _install_fake_icefall_modules(monkeypatch):
    """Stub the three classes ``_load_icefall_modules`` returns."""
    fake_embed_cls = type("FakeEmbedCls", (nn.Module,), {"__init__": lambda self, **kwargs: nn.Module.__init__(self)})

    def _encoder_init(self, **kwargs):
        nn.Module.__init__(self)
        self.init_kwargs = kwargs

    fake_encoder_cls = type(
        "FakeEncoderCls",
        (nn.Module,),
        {"__init__": _encoder_init, "encoder_dim": (64, 128)},
    )
    fake_scheduled_float_cls = type(
        "FakeScheduledFloat", (), {"__init__": lambda self, *args, **kwargs: None}
    )

    monkeypatch.setattr(
        "dma_kws.stage2.icefall_encoder._load_icefall_modules",
        lambda: (fake_embed_cls, fake_encoder_cls, fake_scheduled_float_cls),
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
    return fake_embed_cls, fake_encoder_cls


def test_build_from_params_uses_loaded_modules(monkeypatch):
    fake_embed_cls, fake_encoder_cls = _install_fake_icefall_modules(monkeypatch)

    adapter = IcefallZipformerEncoder.build_from_params(
        {"input_dim": 80, "encoder_type": "icefall_zipformer"}, output_dim=128
    )

    assert adapter.output_dim == 128
    assert isinstance(adapter.encoder_embed, fake_embed_cls)
    assert isinstance(adapter.encoder, fake_encoder_cls)


@pytest.mark.parametrize(
    ("chunk_size", "left_context_frames"),
    [(16, 64), (32, 128), (64, 256), (-1, -1)],
)
def test_build_from_params_starts_at_the_deployment_point(
    monkeypatch, chunk_size, left_context_frames
):
    """A caller that bypasses run_encoder must still get deterministic output."""
    _install_fake_icefall_modules(monkeypatch)

    adapter = IcefallZipformerEncoder.build_from_params(
        {
            "input_dim": 80,
            "encoder_type": "icefall_zipformer",
            "causal": True,
            "stream": {
                "chunk_size": chunk_size,
                "left_context_frames": left_context_frames,
                "train_policy": "multi",
                "train_chunk_size": "16,32,64,-1",
                "train_left_context_frames": "64,128,256,-1",
            },
        },
        output_dim=128,
    )

    assert adapter.encoder.init_kwargs["chunk_size"] == (chunk_size,)
    assert adapter.encoder.init_kwargs["left_context_frames"] == (left_context_frames,)
