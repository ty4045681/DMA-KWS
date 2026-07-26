"""``run_encoder`` must pin every phase except opt-in multi-latency training."""

from __future__ import annotations

import pytest

from dma_kws.config import resolve_stream_policy
from dma_kws.configs.schema import StreamPolicy
from dma_kws.nn import (
    encoder_output_frames,
    min_input_frames_for_encoder,
    run_encoder,
    stream_chunk_tuples,
)


class _FakeIcefallEncoder:
    """Mimics IcefallZipformerEncoder's stream-config setter."""

    def __init__(self) -> None:
        self.applied: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
        self.calls = 0

    def apply_stream_config(self, chunk_sizes, left_context_frames) -> None:
        self.applied.append((tuple(chunk_sizes), tuple(left_context_frames)))

    def __call__(self, feat, feat_lengths):
        self.calls += 1
        return feat, feat_lengths


class _FakeWenetEncoder:
    """Mimics wenet's BaseEncoder.forward keyword arguments."""

    def __init__(self) -> None:
        self.kwargs: list[dict] = []

    def __call__(self, feat, feat_lengths, **kwargs):
        self.kwargs.append(kwargs)
        return feat, feat_lengths


def _icefall_policy(**stream):
    return resolve_stream_policy(
        {
            "stage1": {
                "encoder_type": "icefall_zipformer",
                "causal": True,
                "downsampling_factor": "1,2,4,8,4,2",
                "cnn_module_kernel": "31,31,15,15,15,31",
                "stream": stream,
            }
        }
    )


def _wenet_policy(**stream):
    return resolve_stream_policy(
        {"stage1": {"encoder_type": "conformer", "use_dynamic_chunk": True, "stream": stream}}
    )


@pytest.mark.parametrize(
    ("chunk_size", "left_context_frames"),
    [(16, 64), (32, 128), (64, 256), (-1, -1)],
)
@pytest.mark.parametrize("mode", ["train", "eval"])
def test_match_policy_pins_both_phases(chunk_size, left_context_frames, mode):
    policy = _icefall_policy(chunk_size=chunk_size, left_context_frames=left_context_frames)
    encoder = _FakeIcefallEncoder()

    run_encoder(encoder, "feat", "lens", policy=policy, mode=mode)

    assert encoder.applied == [((chunk_size,), (left_context_frames,))]
    assert encoder.calls == 1


def test_multi_policy_only_widens_the_training_phase():
    policy = _icefall_policy(
        chunk_size=16,
        left_context_frames=64,
        train_policy="multi",
        train_chunk_size="16,32,64,-1",
        train_left_context_frames="64,128,256,-1",
    )
    encoder = _FakeIcefallEncoder()

    run_encoder(encoder, "feat", "lens", policy=policy, mode="train")
    run_encoder(encoder, "feat", "lens", policy=policy, mode="eval")

    assert encoder.applied == [
        ((16, 32, 64, -1), (64, 128, 256, -1)),
        ((16,), (64,)),
    ]


def test_default_mode_is_eval():
    """Any path that forgets to declare a mode must land on the deployment point."""
    policy = _icefall_policy(
        chunk_size=16,
        left_context_frames=64,
        train_policy="multi",
        train_chunk_size="16,32,64,-1",
        train_left_context_frames="64,128,256,-1",
    )
    encoder = _FakeIcefallEncoder()

    run_encoder(encoder, "feat", "lens", policy=policy)

    assert encoder.applied == [((16,), (64,))]


def test_disabled_policy_calls_the_encoder_untouched():
    policy = StreamPolicy(backend="conformer", enabled=False)
    encoder = _FakeWenetEncoder()

    run_encoder(encoder, "feat", "lens", policy=policy, mode="train")

    assert encoder.kwargs == [{}]


def test_wenet_eval_uses_fixed_chunk_and_left_chunk_count():
    policy = _wenet_policy(chunk_size=8, left_context_frames=32)
    encoder = _FakeWenetEncoder()

    run_encoder(encoder, "feat", "lens", policy=policy, mode="eval")

    assert encoder.kwargs == [{"decoding_chunk_size": 8, "num_decoding_left_chunks": 4}]


def test_wenet_multi_training_delegates_to_the_builtin_sampler():
    policy = _wenet_policy(chunk_size=8, left_context_frames=32, train_policy="multi")
    encoder = _FakeWenetEncoder()

    run_encoder(encoder, "feat", "lens", policy=policy, mode="train")
    run_encoder(encoder, "feat", "lens", policy=policy, mode="eval")

    # decoding_chunk_size 0 is wenet's "sample a random chunk" sentinel; it must
    # never leak into eval.
    assert encoder.kwargs == [
        {"decoding_chunk_size": 0, "num_decoding_left_chunks": -1},
        {"decoding_chunk_size": 8, "num_decoding_left_chunks": 4},
    ]


def test_wenet_full_context_point_maps_to_negative_chunk():
    policy = _wenet_policy(chunk_size=-1, left_context_frames=-1)
    encoder = _FakeWenetEncoder()

    run_encoder(encoder, "feat", "lens", policy=policy, mode="eval")

    assert encoder.kwargs == [{"decoding_chunk_size": -1, "num_decoding_left_chunks": -1}]


def test_unknown_mode_is_rejected():
    policy = _icefall_policy(chunk_size=16, left_context_frames=64)

    with pytest.raises(ValueError, match="mode must be one of"):
        run_encoder(_FakeIcefallEncoder(), "feat", "lens", policy=policy, mode="validate")


def test_icefall_backend_requires_the_adapter():
    policy = _icefall_policy(chunk_size=16, left_context_frames=64)

    with pytest.raises(TypeError, match="apply_stream_config"):
        run_encoder(_FakeWenetEncoder(), "feat", "lens", policy=policy, mode="eval")


def test_stream_chunk_tuples_rejects_unknown_mode():
    policy = _icefall_policy(chunk_size=16, left_context_frames=64)

    with pytest.raises(ValueError, match="mode must be one of"):
        stream_chunk_tuples(policy, "test")


class _FakeWenetEmbed:
    """Wenet subsampling modules publish their own rate and right context."""

    def __init__(self, subsampling_rate: int, right_context: int) -> None:
        self.subsampling_rate = subsampling_rate
        self.right_context = right_context


class _WenetShapedEncoder:
    def __init__(self, subsampling_rate: int, right_context: int) -> None:
        self.embed = _FakeWenetEmbed(subsampling_rate, right_context)


class _IcefallShapedEncoder:
    """Mirrors IcefallZipformerEncoder.output_frames without importing icefall."""

    def output_frames(self, num_input_frames):
        from dma_kws.stage2.icefall_encoder import (
            OUTPUT_DOWNSAMPLING_FACTOR,
            embed_output_frames,
        )

        subsampled = embed_output_frames(num_input_frames)
        if subsampled <= 0:
            return 0
        return (subsampled + 1) // OUTPUT_DOWNSAMPLING_FACTOR


@pytest.mark.parametrize(
    ("num_frames", "expected"),
    # wenet Conv2dSubsampling4 computes ((T - 1) // 2 - 1) // 2.
    [(6, 0), (7, 1), (10, 1), (11, 2), (15, 3)],
)
def test_wenet_output_frames_match_conv2dsubsampling4(num_frames, expected):
    encoder = _WenetShapedEncoder(subsampling_rate=4, right_context=6)

    assert encoder_output_frames(encoder, num_frames) == expected


@pytest.mark.parametrize(
    ("num_frames", "expected"),
    # icefall Conv2dSubsampling gives (T - 7) // 2, then Zipformer2 halves again.
    [(7, 0), (8, 0), (9, 1), (10, 1), (11, 1), (13, 2), (17, 3)],
)
def test_icefall_output_frames_match_the_documented_formula(num_frames, expected):
    assert encoder_output_frames(_IcefallShapedEncoder(), num_frames) == expected


def test_icefall_needs_more_input_frames_than_wenet():
    """The regression this replaces: 7 fbank frames is fine for wenet, empty for icefall."""
    wenet = _WenetShapedEncoder(subsampling_rate=4, right_context=6)
    icefall = _IcefallShapedEncoder()

    assert min_input_frames_for_encoder(wenet, 1) == 7
    assert min_input_frames_for_encoder(icefall, 1) == 9
    assert encoder_output_frames(icefall, 7) == 0


@pytest.mark.parametrize("min_output_frames", [1, 2, 5, 25])
def test_min_input_frames_is_the_tightest_bound(min_output_frames):
    for encoder in (_WenetShapedEncoder(subsampling_rate=4, right_context=6), _IcefallShapedEncoder()):
        minimum = min_input_frames_for_encoder(encoder, min_output_frames)
        assert encoder_output_frames(encoder, minimum) >= min_output_frames
        assert encoder_output_frames(encoder, minimum - 1) < min_output_frames


def test_encoder_without_declared_subsampling_is_rejected():
    with pytest.raises(TypeError, match="subsampled length is unknown"):
        encoder_output_frames(object(), 100)


def _real_wenet_encoder(output_dim: int, cfg: dict):
    """Build the vendored wenet ConformerEncoder, or skip if it cannot import."""
    from dma_kws.nn import build_encoder

    try:
        return build_encoder(cfg, output_dim=output_dim)
    except (SystemExit, ImportError, ModuleNotFoundError) as exc:  # optional deps
        pytest.skip(f"wenet ConformerEncoder unavailable: {exc}")


def test_wenet_eval_output_is_bit_identical_across_calls():
    """Regression guard: wenet's dynamic chunk sampler is not gated on training mode.

    Calling ``encoder(feat, lens)`` without ``decoding_chunk_size`` leaves
    ``add_optional_chunk_mask`` on its ``torch.randint`` branch, so eval scores
    were not reproducible. ``run_encoder`` must pin them.
    """
    torch = pytest.importorskip("torch")

    cfg = {
        "input_dim": 80,
        "num_blocks": 2,
        "causal": True,
        "cnn_module_norm": "layer_norm",
        "use_dynamic_chunk": True,
        "use_dynamic_left_chunk": True,
        "stream": {"chunk_size": 8, "left_context_frames": 32, "train_policy": "multi"},
    }
    policy = resolve_stream_policy(cfg)
    torch.manual_seed(0)
    encoder = _real_wenet_encoder(144, cfg).eval()

    feat = torch.randn(2, 200, 80)
    lens = torch.tensor([200, 150])
    with torch.no_grad():
        outputs = [
            run_encoder(encoder, feat, lens, policy=policy, mode="eval")[0] for _ in range(5)
        ]

    assert all(torch.equal(outputs[0], other) for other in outputs[1:])
