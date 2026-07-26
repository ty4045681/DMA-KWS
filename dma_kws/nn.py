"""Neural network construction helpers shared across DMA-KWS scripts.

Consolidates the identical ConformerEncoder kwargs block duplicated in
train_stage1_ctc.py, train_stage2_qbyt.py, and run_two_stage_demo.py. qbyt/ is
added to sys.path and ConformerEncoder is imported lazily so the friendly
SystemExit on a missing torch install is preserved.

Also supports icefall Zipformer2 encoder via lazy import of egs/gigaspeech/KWS/zipformer.
"""

from __future__ import annotations

from typing import Any

from dma_kws.config import resolve_stream_policy
from dma_kws.configs.schema import StreamPolicy
from dma_kws.pathing import ensure_qbyt_on_path

STREAM_MODES = ("train", "eval")


def stream_chunk_tuples(policy: StreamPolicy, mode: str) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Return the ``(chunk_sizes, left_context_frames)`` tuples for one phase.

    ``eval`` always resolves to the single deployment operating point. ``train``
    only widens to the multi-latency lists when ``train_policy`` opts in.
    """
    if mode not in STREAM_MODES:
        raise ValueError(f"mode must be one of {STREAM_MODES}, got {mode!r}")
    if mode == "train" and policy.train_policy == "multi":
        return policy.train_chunk_sizes, policy.train_left_context_frames
    return (policy.chunk_size,), (policy.left_context_frames,)


def run_encoder(
    encoder,
    feat,
    feat_lengths,
    *,
    policy: StreamPolicy,
    mode: str = "eval",
):
    """Call ``encoder`` with the chunked-attention settings for ``mode``.

    ``mode`` defaults to ``"eval"`` on purpose: every path that does not
    explicitly opt into randomized training gets the deployment operating point.
    It cannot be derived from ``encoder.training`` because Stage II forces the
    encoder into ``eval()`` while it is frozen, including during training steps.
    """
    if mode not in STREAM_MODES:
        raise ValueError(f"mode must be one of {STREAM_MODES}, got {mode!r}")
    if not policy.enabled:
        return encoder(feat, feat_lengths)

    if policy.backend == "icefall_zipformer":
        apply = getattr(encoder, "apply_stream_config", None)
        if apply is None:
            raise TypeError(
                f"{type(encoder).__name__} does not expose apply_stream_config(); "
                "stage1.encoder_type=icefall_zipformer expects IcefallZipformerEncoder"
            )
        apply(*stream_chunk_tuples(policy, mode))
        return encoder(feat, feat_lengths)

    # Wenet ConformerEncoder takes the chunk settings as forward arguments.
    # decoding_chunk_size: 0 -> wenet's own random sampler, <0 -> full context, >0 -> fixed.
    # num_decoding_left_chunks: <0 -> all left chunks, >=0 -> that many chunks.
    if mode == "train" and policy.train_policy == "multi":
        decoding_chunk_size, num_left_chunks = 0, -1
    else:
        decoding_chunk_size = policy.chunk_size
        num_left_chunks = policy.left_context_chunks
    return encoder(
        feat,
        feat_lengths,
        decoding_chunk_size=decoding_chunk_size,
        num_decoding_left_chunks=num_left_chunks,
    )


def encoder_output_frames(encoder, num_input_frames: int) -> int:
    """Frames ``encoder`` emits for ``num_input_frames`` fbank frames.

    Every encoder subsamples, and by a different amount: an input long enough to
    produce fbank frames is not necessarily long enough to survive the encoder.
    Rather than hardcoding a per-backend minimum, ask the architecture:
    :class:`~dma_kws.stage2.icefall_encoder.IcefallZipformerEncoder` implements
    ``output_frames`` directly, and wenet's subsampling modules publish
    ``subsampling_rate``/``right_context``.
    """
    if num_input_frames <= 0:
        return 0

    own = getattr(encoder, "output_frames", None)
    if callable(own):
        return int(own(num_input_frames))

    embed = getattr(encoder, "embed", None)
    rate = getattr(embed, "subsampling_rate", None)
    right_context = getattr(embed, "right_context", None)
    if rate is None or right_context is None:
        raise TypeError(
            f"{type(encoder).__name__} exposes neither output_frames() nor "
            "embed.subsampling_rate/right_context, so its subsampled length is unknown"
        )
    # Standard strided-convolution output length for a receptive field of
    # right_context + 1 input frames.
    span = num_input_frames - int(right_context) - 1
    if span < 0:
        return 0
    return span // int(rate) + 1


def min_input_frames_for_encoder(encoder, min_output_frames: int) -> int:
    """Smallest fbank length that yields at least ``min_output_frames``.

    Solved by search rather than by inverting each backend's formula, which is
    where off-by-one errors live. The loop is bounded because every supported
    subsampling is monotonically non-decreasing in its input length.
    """
    if min_output_frames <= 0:
        return 0
    num_frames = 1
    while encoder_output_frames(encoder, num_frames) < min_output_frames:
        num_frames += 1
        if num_frames > _MAX_MIN_FRAME_SEARCH:
            raise ValueError(
                f"{type(encoder).__name__} does not reach {min_output_frames} output frames "
                f"within {_MAX_MIN_FRAME_SEARCH} input frames"
            )
    return num_frames


#: Upper bound for the search above; 100 fbank frames is 1 s at a 10 ms shift, and
#: every supported subsampling reaches one output frame in far less than that.
_MAX_MIN_FRAME_SEARCH = 10_000


def build_encoder(stage1_cfg: dict[str, Any], *, output_dim: int):
    """Build an encoder from the ``stage1`` config section.
    
    Selects encoder type based on ``stage1_cfg["encoder_type"]``:
    - "conformer" (default): Wenet ConformerEncoder
    - "icefall_zipformer": Icefall Zipformer2 encoder (requires ICEFALL_ROOT env var)
    
    Args:
        stage1_cfg: Stage1 configuration dict
        output_dim: Target output dimension for encoder
    
    Returns:
        Initialized encoder module with forward(feat, feat_lengths) → (out, mask) interface
    """
    encoder_type = stage1_cfg.get("encoder_type", "conformer").lower()
    policy = resolve_stream_policy(stage1_cfg)
    print(f"Encoder stream policy: {policy.describe()}")

    if encoder_type == "icefall_zipformer":
        from dma_kws.stage2.icefall_encoder import IcefallZipformerEncoder
        return IcefallZipformerEncoder.build_from_params(
            stage1_cfg, output_dim=output_dim, policy=policy
        )
    
    # Default: Wenet ConformerEncoder
    if encoder_type != "conformer":
        import warnings
        warnings.warn(
            f"Unknown encoder_type {encoder_type!r}, falling back to 'conformer'",
            UserWarning,
        )
    
    ensure_qbyt_on_path()
    try:
        from models.encoder import ConformerEncoder
    except ImportError as exc:
        raise SystemExit(
            "Missing torch/torchaudio. Install CUDA PyTorch on the remote training machine first."
        ) from exc

    from dma_kws.cmvn import build_global_cmvn

    return ConformerEncoder(
        input_size=int(stage1_cfg.get("input_dim", 80)),
        output_size=output_dim,
        attention_heads=int(stage1_cfg.get("attention_heads", 4)),
        linear_units=int(stage1_cfg.get("linear_units", 576)),
        num_blocks=int(stage1_cfg.get("num_blocks", 6)),
        dropout_rate=float(stage1_cfg.get("dropout_rate", 0.1)),
        positional_dropout_rate=float(stage1_cfg.get("positional_dropout_rate", 0.1)),
        attention_dropout_rate=float(stage1_cfg.get("attention_dropout_rate", 0.0)),
        use_cnn_module=True,
        input_layer="conv2d",
        pos_enc_layer_type="rel_pos",
        selfattention_layer_type="rel_selfattn",
        cnn_module_kernel=int(stage1_cfg.get("cnn_module_kernel", 3)),
        causal=bool(stage1_cfg.get("causal", False)),
        cnn_module_norm=str(stage1_cfg.get("cnn_module_norm", "batch_norm")),
        use_dynamic_chunk=bool(stage1_cfg.get("use_dynamic_chunk", False)),
        use_dynamic_left_chunk=bool(stage1_cfg.get("use_dynamic_left_chunk", False)),
        gradient_checkpointing=bool(stage1_cfg.get("gradient_checkpointing", False)),
        global_cmvn=build_global_cmvn(stage1_cfg),
    )
