"""Neural network construction helpers shared across DMA-KWS scripts.

Consolidates the identical ConformerEncoder kwargs block duplicated in
train_stage1_ctc.py, train_stage2_qbyt.py, and run_two_stage_demo.py. qbyt/ is
added to sys.path and ConformerEncoder is imported lazily so the friendly
SystemExit on a missing torch install is preserved.

Also supports icefall Zipformer2 encoder via lazy import of egs/gigaspeech/KWS/zipformer.
"""

from __future__ import annotations

from typing import Any

from dma_kws.pathing import ensure_qbyt_on_path


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
    
    if encoder_type == "icefall_zipformer":
        from dma_kws.stage2.icefall_encoder import IcefallZipformerEncoder
        return IcefallZipformerEncoder.build_from_params(stage1_cfg, output_dim=output_dim)
    
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
