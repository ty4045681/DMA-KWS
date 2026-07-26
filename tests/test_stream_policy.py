"""Resolution and validation of the encoder's chunked-attention operating point."""

from __future__ import annotations

import pytest

from dma_kws.config import resolve_stream_policy


def _icefall(**stream):
    return {
        "stage1": {
            "encoder_type": "icefall_zipformer",
            "causal": True,
            "downsampling_factor": "1,2,4,8,4,2",
            "cnn_module_kernel": "31,31,15,15,15,31",
            "stream": stream,
        }
    }


def _conformer(**overrides):
    stage1 = {"encoder_type": "conformer", "use_dynamic_chunk": True}
    stage1.update(overrides)
    return {"stage1": stage1}


@pytest.mark.parametrize(
    ("chunk_size", "left_context_frames", "expected_chunks"),
    [(16, 64, 4), (16, 128, 8), (32, 128, 4), (64, 256, 4), (64, 64, 1), (-1, -1, -1)],
)
def test_resolves_operating_point(chunk_size, left_context_frames, expected_chunks):
    policy = resolve_stream_policy(
        _icefall(chunk_size=chunk_size, left_context_frames=left_context_frames)
    )

    assert policy.enabled is True
    assert policy.backend == "icefall_zipformer"
    assert policy.chunk_size == chunk_size
    assert policy.left_context_frames == left_context_frames
    assert policy.left_context_chunks == expected_chunks


def test_match_policy_reuses_the_deployment_point_for_training():
    policy = resolve_stream_policy(_icefall(chunk_size=32, left_context_frames=128))

    assert policy.train_policy == "match"
    assert policy.train_chunk_sizes == (32,)
    assert policy.train_left_context_frames == (128,)


def test_multi_policy_keeps_the_deployment_point_separate():
    policy = resolve_stream_policy(
        _icefall(
            chunk_size=16,
            left_context_frames=64,
            train_policy="multi",
            train_chunk_size="16,32,64,-1",
            train_left_context_frames="64,128,256,-1",
        )
    )

    assert policy.chunk_size == 16
    assert policy.train_chunk_sizes == (16, 32, 64, -1)
    assert policy.train_left_context_frames == (64, 128, 256, -1)


def test_legacy_keys_raise_with_a_migration_hint():
    config = {"stage1": {"encoder_type": "icefall_zipformer", "causal": True, "chunk_size": "16,32,64,-1"}}

    with pytest.raises(ValueError) as excinfo:
        resolve_stream_policy(config)

    message = str(excinfo.value)
    assert "stage1.chunk_size" in message
    assert "stage1.stream" in message
    assert "train_policy" in message


def test_multi_value_operating_point_is_rejected():
    with pytest.raises(ValueError, match="must be a single value"):
        resolve_stream_policy(_icefall(chunk_size="16,32,64,-1", left_context_frames=64))


def test_missing_operating_point_is_rejected():
    with pytest.raises(ValueError, match="stage1.stream.chunk_size is required"):
        resolve_stream_policy(_icefall())


@pytest.mark.parametrize("chunk_size", [12, 20, 36])
def test_chunk_size_must_divide_every_downsampling_factor(chunk_size):
    with pytest.raises(ValueError, match="not divisible by downsampling factors"):
        resolve_stream_policy(_icefall(chunk_size=chunk_size, left_context_frames=256))


def test_left_context_below_the_convolution_requirement_is_rejected():
    # cnn_module_kernel 31 with downsampling_factor 8 needs 7*8 = 56 frames.
    with pytest.raises(ValueError, match="frames of left context"):
        resolve_stream_policy(_icefall(chunk_size=16, left_context_frames=16))


def test_full_context_requires_both_values_to_be_disabled():
    with pytest.raises(ValueError, match="left_context_frames must also be -1"):
        resolve_stream_policy(_icefall(chunk_size=-1, left_context_frames=64))


def test_training_lists_are_validated_too():
    with pytest.raises(ValueError, match="not divisible by downsampling factors"):
        resolve_stream_policy(
            _icefall(
                chunk_size=16,
                left_context_frames=64,
                train_policy="multi",
                train_chunk_size="16,12",
                train_left_context_frames="64",
            )
        )


def test_non_causal_zipformer_must_not_declare_an_operating_point():
    config = {
        "stage1": {
            "encoder_type": "icefall_zipformer",
            "causal": False,
            "stream": {"chunk_size": 16, "left_context_frames": 64},
        }
    }

    with pytest.raises(ValueError, match="causal=false"):
        resolve_stream_policy(config)


def test_non_causal_zipformer_resolves_to_full_context():
    policy = resolve_stream_policy({"stage1": {"encoder_type": "icefall_zipformer", "causal": False}})

    assert policy.enabled is False
    assert policy.left_context_chunks == -1


def test_wenet_dynamic_chunk_requires_an_operating_point():
    with pytest.raises(ValueError, match="stage1.stream.chunk_size is required"):
        resolve_stream_policy(_conformer())


def test_wenet_multi_policy_does_not_require_training_lists():
    policy = resolve_stream_policy(
        _conformer(stream={"chunk_size": 8, "left_context_frames": 32, "train_policy": "multi"})
    )

    assert policy.backend == "conformer"
    assert policy.left_context_chunks == 4
    assert policy.train_chunk_sizes == ()


def test_wenet_without_dynamic_chunk_must_not_declare_an_operating_point():
    config = {
        "stage1": {
            "encoder_type": "conformer",
            "use_dynamic_chunk": False,
            "stream": {"chunk_size": 8, "left_context_frames": 32},
        }
    }

    with pytest.raises(ValueError, match="use_dynamic_chunk=false"):
        resolve_stream_policy(config)


def test_unknown_train_policy_is_rejected():
    with pytest.raises(ValueError, match="train_policy must be"):
        resolve_stream_policy(
            _icefall(chunk_size=16, left_context_frames=64, train_policy="random")
        )


def _locator_config(decode_args=None, **stream):
    config = _icefall(**stream)
    config["locator"] = {
        "type": "icefall_pt",
        "root": ".",
        "decode_script": "decode.py",
        "checkpoint": "ckpt.pt",
        "decode_args": decode_args or [],
    }
    return config


@pytest.mark.parametrize(
    ("chunk_size", "left_context_frames"),
    [(16, 64), (32, 128), (-1, -1)],
)
def test_locator_decode_args_follow_the_operating_point(chunk_size, left_context_frames):
    from dma_kws.inference.locators.icefall_pt import _stream_decode_args

    args = _stream_decode_args(
        _locator_config(chunk_size=chunk_size, left_context_frames=left_context_frames)
    )

    assert args == [
        "--causal",
        "1",
        "--chunk-size",
        str(chunk_size),
        "--left-context-frames",
        str(left_context_frames),
    ]


def test_locator_decode_args_for_a_non_causal_checkpoint():
    from dma_kws.inference.locators.icefall_pt import _stream_decode_args

    assert _stream_decode_args({"stage1": {"encoder_type": "icefall_zipformer"}}) == [
        "--causal",
        "0",
    ]


@pytest.mark.parametrize(
    "decode_args",
    [["--chunk-size", "32"], ["--chunk-size=32"], ["--causal", "0"], ["--left-context-frames=8"]],
)
def test_locator_rejects_hand_written_streaming_flags(decode_args):
    from dma_kws.inference.locators.icefall_pt import IcefallPtKwsLocator

    config = _locator_config(decode_args=decode_args, chunk_size=16, left_context_frames=64)

    with pytest.raises(ValueError, match="must not set"):
        IcefallPtKwsLocator(config=config)


def test_locator_keeps_unrelated_decode_args():
    from dma_kws.inference.locators.icefall_pt import IcefallPtKwsLocator

    config = _locator_config(
        decode_args=["--keywords-score", "1.5"], chunk_size=16, left_context_frames=64
    )
    locator = IcefallPtKwsLocator(config=config)

    command = locator._build_command("a.wav", "hey eva")

    assert command[-2:] == ["--keywords-score", "1.5"]
    assert "--chunk-size" in command
    assert command[command.index("--chunk-size") + 1] == "16"


def test_removed_min_fbank_frames_key_raises_with_a_migration_hint():
    from dma_kws.config import resolve_min_encoder_frames

    with pytest.raises(ValueError) as excinfo:
        resolve_min_encoder_frames({"min_stage2_fbank_frames": 7})

    message = str(excinfo.value)
    assert "min_stage2_encoder_frames" in message
    assert "subsampled away to zero frames" in message


def test_min_encoder_frames_defaults_to_one():
    from dma_kws.config import resolve_min_encoder_frames

    assert resolve_min_encoder_frames({}) == 1
    assert resolve_min_encoder_frames({"min_stage2_encoder_frames": 4}) == 4


def test_min_encoder_frames_must_be_positive():
    from dma_kws.config import resolve_min_encoder_frames

    with pytest.raises(ValueError, match="at least 1"):
        resolve_min_encoder_frames({"min_stage2_encoder_frames": 0})


def test_describe_reports_both_phases():
    policy = resolve_stream_policy(
        _icefall(
            chunk_size=16,
            left_context_frames=64,
            train_policy="multi",
            train_chunk_size="16,32,64,-1",
            train_left_context_frames="64,128,256,-1",
        )
    )

    described = policy.describe()
    assert "eval=16/64" in described
    assert "multi chunk=16,32,64,-1" in described
