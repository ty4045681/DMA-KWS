"""Configuration helpers for DMA-KWS scripts."""

from __future__ import annotations

import os
from dataclasses import fields, replace
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf

from dma_kws.configs.schema import DMAKWSConfig, FbankConfig, StreamPolicy

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "configs"

#: ``stage1`` keys that used to hold icefall's multi-value chunk lists directly.
#: They now live under ``stage1.stream`` with a different meaning, so their
#: presence must fail loudly instead of silently changing behaviour.
_LEGACY_STREAM_KEYS = ("chunk_size", "left_context_frames")

_MIGRATION_HINT = """
`stage1.chunk_size` / `stage1.left_context_frames` were removed.

The chunked-attention operating point now lives under `stage1.stream`, where
`chunk_size` / `left_context_frames` are the *deployment* point and must be
single values (they are also used for validation, offline eval and inference).
Randomized multi-latency training is opt-in.

Migrate:

  stage1:
    stream:
      chunk_size: 16              # was the first entry of the old list
      left_context_frames: 64
      train_policy: match         # or: multi
      train_chunk_size: "16,32,64,-1"
      train_left_context_frames: "64,128,256,-1"
""".strip()


def _expand_value(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expanduser(os.path.expandvars(value))
    if isinstance(value, list):
        return [_expand_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand_value(item) for key, item in value.items()}
    return value


def _schema_defaults() -> DictConfig:
    return OmegaConf.structured(DMAKWSConfig)


def compose_config(
    experiment: str | None = None,
    overrides: Iterable[str] = (),
) -> DictConfig:
    """Compose a validated config via Hydra (groups + optional experiment overlay)."""
    GlobalHydra.instance().clear()
    override_list = list(overrides)
    if experiment:
        override_list.insert(0, f"+experiment={experiment}")
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        return compose(config_name="config", overrides=override_list)


def _merge_with_schema(cfg: DictConfig) -> DictConfig:
    base = OmegaConf.create(OmegaConf.to_container(_schema_defaults(), resolve=False))
    OmegaConf.set_struct(base, False)
    return OmegaConf.merge(base, cfg)


def config_to_dict(cfg: DictConfig) -> dict[str, Any]:
    """Resolve a composed DictConfig to a plain dict with env/user expansion."""
    merged = _merge_with_schema(cfg)
    container = OmegaConf.to_container(merged, resolve=True)
    if not isinstance(container, dict):
        raise ValueError("Config must resolve to a mapping")
    return _expand_value(container)


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file merged onto schema defaults (backward compatible)."""
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping at top level: {config_path}")

    base = OmegaConf.create(OmegaConf.to_container(_schema_defaults(), resolve=False))
    OmegaConf.set_struct(base, False)
    merged = OmegaConf.merge(base, OmegaConf.create(data))
    container = OmegaConf.to_container(merged, resolve=False)
    if not isinstance(container, dict):
        raise ValueError(f"Config must resolve to a mapping: {config_path}")
    return _expand_value(container)


def require_sections(config: dict[str, Any], sections: Iterable[str]) -> None:
    """Raise a readable error when required top-level config sections are absent."""
    missing = [section for section in sections if section not in config]
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"Missing required config sections: {joined}")


def get_tokenizer_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return the tokenizer section from a loaded config."""
    require_sections(config, ["tokenizer"])
    tokenizer = config["tokenizer"]
    if not isinstance(tokenizer, dict):
        raise ValueError("Config section 'tokenizer' must be a mapping")
    return tokenizer


def _fbank_field_names() -> tuple[str, ...]:
    return tuple(field.name for field in fields(FbankConfig))


def _resolve_num_mel_bins(config: dict[str, Any], fbank_section: dict[str, Any]) -> int:
    if "num_mel_bins" in fbank_section:
        return int(fbank_section["num_mel_bins"])
    stage1 = config.get("stage1")
    if isinstance(stage1, dict) and "input_dim" in stage1:
        return int(stage1["input_dim"])
    return FbankConfig.num_mel_bins


def get_fbank_config(config: dict[str, Any]) -> FbankConfig:
    """Return shared fbank settings from optional top-level ``fbank`` config."""
    fbank_section = config.get("fbank", {})
    if not isinstance(fbank_section, dict):
        raise ValueError("Config section 'fbank' must be a mapping")

    kwargs: dict[str, Any] = {
        "num_mel_bins": _resolve_num_mel_bins(config, fbank_section),
    }
    for name in _fbank_field_names():
        if name == "num_mel_bins":
            continue
        if name in fbank_section:
            kwargs[name] = fbank_section[name]
    return FbankConfig(**kwargs)


def get_eval_fbank_config(config: dict[str, Any]) -> FbankConfig:
    """Return fbank settings for eval prep, with optional ``stage2.eval.fbank`` overrides."""
    base = get_fbank_config(config)
    stage2 = config.get("stage2")
    if not isinstance(stage2, dict):
        return base

    eval_section = stage2.get("eval")
    if not isinstance(eval_section, dict):
        return base

    eval_fbank = eval_section.get("fbank")
    if eval_fbank is None:
        return base
    if not isinstance(eval_fbank, dict):
        raise ValueError("Config section 'stage2.eval.fbank' must be a mapping")

    overrides = {name: eval_fbank[name] for name in _fbank_field_names() if name in eval_fbank}
    return replace(base, **overrides)


def _parse_int_tuple(raw: Any, *, key: str) -> tuple[int, ...]:
    """Parse a comma-separated int list, as used by icefall's CLI arguments."""
    if isinstance(raw, (list, tuple)):
        values = list(raw)
    else:
        values = [item for item in str(raw).split(",") if item.strip()]
    if not values:
        raise ValueError(f"{key} must list at least one value")
    try:
        return tuple(int(str(item).strip()) for item in values)
    except ValueError as exc:
        raise ValueError(f"{key} must be a comma-separated list of ints, got {raw!r}") from exc


def _parse_operating_point(raw: Any, *, key: str) -> int:
    """Parse a single-valued operating-point entry, rejecting icefall-style lists."""
    if raw is None:
        raise ValueError(f"{key} is required but was not declared")
    if isinstance(raw, (list, tuple)) or "," in str(raw):
        raise ValueError(
            f"{key} must be a single value, got {raw!r}. The deployment operating point "
            "cannot be randomized; put multi-latency lists in "
            "stage1.stream.train_chunk_size / train_left_context_frames and set "
            "stage1.stream.train_policy=multi."
        )
    try:
        return int(str(raw).strip())
    except ValueError as exc:
        raise ValueError(f"{key} must be an int, got {raw!r}") from exc


def _validate_icefall_point(
    chunk_size: int,
    left_context_frames: int,
    stage1: Mapping[str, Any],
    *,
    origin: str,
) -> None:
    """Reproduce icefall's ``_get_attn_mask`` asserts at config-resolution time."""
    if chunk_size <= 0:
        return
    downsampling_factor = _parse_int_tuple(
        stage1.get("downsampling_factor", "1,2,4,8,4,2"), key="stage1.downsampling_factor"
    )
    offenders = [factor for factor in downsampling_factor if chunk_size % factor != 0]
    if offenders:
        raise ValueError(
            f"{origin}={chunk_size} is not divisible by downsampling factors {offenders}; "
            f"icefall requires chunk_size % d == 0 for every d in {list(downsampling_factor)}"
        )

    left_context_chunks = -1 if left_context_frames < 0 else max(1, left_context_frames // chunk_size)
    if left_context_chunks < 0:
        return
    cnn_module_kernel = _parse_int_tuple(
        stage1.get("cnn_module_kernel", "31,31,15,15,15,31"), key="stage1.cnn_module_kernel"
    )
    if len(cnn_module_kernel) == 1:
        cnn_module_kernel = cnn_module_kernel * len(downsampling_factor)
    required = max(
        (kernel // 2) * factor for kernel, factor in zip(cnn_module_kernel, downsampling_factor)
    )
    available = chunk_size * left_context_chunks
    if available < required:
        raise ValueError(
            f"{origin}={chunk_size} with left_context_frames={left_context_frames} gives only "
            f"{available} frames of left context, but icefall's convolution modules need at least "
            f"{required} (max of cnn_module_kernel[i]//2 * downsampling_factor[i]). "
            "Increase left_context_frames or chunk_size."
        )


def _stream_section(stage1: Mapping[str, Any]) -> Mapping[str, Any]:
    stream = stage1.get("stream")
    if stream is None:
        return {}
    if not isinstance(stream, Mapping):
        raise ValueError("Config section 'stage1.stream' must be a mapping")
    return stream


def resolve_stream_policy(config: Mapping[str, Any]) -> StreamPolicy:
    """Resolve and validate the encoder's chunked-attention policy.

    The returned policy is the single source of truth for every phase: training
    (``mode="train"``), validation, offline eval and inference (``mode="eval"``).
    See :func:`dma_kws.nn.run_encoder` for how it is applied.
    """
    stage1 = config.get("stage1", config)
    if not isinstance(stage1, Mapping):
        raise ValueError("Config section 'stage1' must be a mapping")

    legacy = [key for key in _LEGACY_STREAM_KEYS if stage1.get(key) is not None]
    if legacy:
        joined = ", ".join(f"stage1.{key}" for key in legacy)
        raise ValueError(f"Removed config keys are still set ({joined}).\n\n{_MIGRATION_HINT}")

    backend = str(stage1.get("encoder_type", "conformer")).lower()
    stream = _stream_section(stage1)
    declared = stream.get("chunk_size") is not None or stream.get("left_context_frames") is not None

    if backend == "icefall_zipformer":
        chunking_supported = bool(stage1.get("causal", False))
        disabled_reason = (
            "stage1.causal=false, so icefall's Zipformer2 ignores chunk_size entirely"
        )
    else:
        chunking_supported = bool(stage1.get("use_dynamic_chunk", False))
        disabled_reason = (
            "stage1.use_dynamic_chunk=false, so the Wenet encoder ignores decoding_chunk_size"
        )

    if not chunking_supported:
        if declared:
            raise ValueError(
                f"stage1.stream declares an operating point but {disabled_reason}. "
                "Remove stage1.stream.chunk_size / left_context_frames, or enable chunking."
            )
        return StreamPolicy(backend=backend, enabled=False)

    chunk_size = _parse_operating_point(
        stream.get("chunk_size"), key="stage1.stream.chunk_size"
    )
    left_context_frames = _parse_operating_point(
        stream.get("left_context_frames"), key="stage1.stream.left_context_frames"
    )
    if chunk_size == 0 or chunk_size < -1:
        raise ValueError(
            f"stage1.stream.chunk_size must be -1 (full context) or a positive int, got {chunk_size}"
        )
    if chunk_size == -1 and left_context_frames != -1:
        raise ValueError(
            "stage1.stream.chunk_size=-1 means full context, so "
            f"left_context_frames must also be -1, got {left_context_frames}"
        )
    if left_context_frames == 0 or left_context_frames < -1:
        raise ValueError(
            "stage1.stream.left_context_frames must be -1 (unlimited) or a positive int, "
            f"got {left_context_frames}"
        )

    train_policy = str(stream.get("train_policy", "match")).lower()
    if train_policy not in {"match", "multi"}:
        raise ValueError(
            f"stage1.stream.train_policy must be 'match' or 'multi', got {train_policy!r}"
        )

    if train_policy == "multi" and backend == "icefall_zipformer":
        # icefall randomizes over explicit lists; wenet has its own dynamic-chunk
        # sampler and ignores these, so they are only required here.
        train_chunk_sizes = _parse_int_tuple(
            stream.get("train_chunk_size", ""), key="stage1.stream.train_chunk_size"
        )
        train_left_context_frames = _parse_int_tuple(
            stream.get("train_left_context_frames", ""),
            key="stage1.stream.train_left_context_frames",
        )
    elif train_policy == "multi":
        train_chunk_sizes = ()
        train_left_context_frames = ()
    else:
        train_chunk_sizes = (chunk_size,)
        train_left_context_frames = (left_context_frames,)

    if backend == "icefall_zipformer":
        _validate_icefall_point(
            chunk_size, left_context_frames, stage1, origin="stage1.stream.chunk_size"
        )
        for train_chunk in train_chunk_sizes:
            for train_left in train_left_context_frames:
                _validate_icefall_point(
                    train_chunk, train_left, stage1, origin="stage1.stream.train_chunk_size entry"
                )

    return StreamPolicy(
        backend=backend,
        enabled=True,
        chunk_size=chunk_size,
        left_context_frames=left_context_frames,
        train_policy=train_policy,
        train_chunk_sizes=train_chunk_sizes,
        train_left_context_frames=train_left_context_frames,
    )


def resolve_min_encoder_frames(demo_cfg: Mapping[str, Any]) -> int:
    """Read ``demo.min_stage2_encoder_frames``, rejecting the removed fbank-frame key."""
    if demo_cfg.get("min_stage2_fbank_frames") is not None:
        raise ValueError(
            "`demo.min_stage2_fbank_frames` was removed. It hardcoded a length that only "
            "held for wenet's subsampling, so an icefall Zipformer candidate short enough "
            "to pass it subsampled away to zero frames. Declare the requirement in encoder "
            "frames instead and let each backend derive its own fbank length:\n\n"
            "  demo:\n"
            "    min_stage2_encoder_frames: 1\n"
        )
    value = int(demo_cfg.get("min_stage2_encoder_frames", 1))
    if value < 1:
        raise ValueError(
            f"demo.min_stage2_encoder_frames must be at least 1, got {value}"
        )
    return value


def fbank_kwargs(cfg: FbankConfig) -> dict[str, Any]:
    """Return keyword arguments suitable for ``compute_fbank_for_clip``."""
    return {
        "num_mel_bins": cfg.num_mel_bins,
        "frame_length": cfg.frame_length,
        "frame_shift": cfg.frame_shift,
        "dither": cfg.dither,
        "window_type": cfg.window_type,
        "backend": cfg.backend,
        "target_sample_rate": cfg.target_sample_rate,
        "snip_edges": cfg.snip_edges,
        "low_freq": cfg.low_freq,
        "high_freq": cfg.high_freq,
    }
