"""Temporary-hook capture of pooling QbyT ``nn.MultiheadAttention`` weights.

This module does not copy the Transformer forward and does not change QbyT
parameters, buffers, or ``state_dict`` keys. Capture is eval float32 eager on
the live module device (CPU or CUDA) with autocast disabled.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Sequence
from weakref import WeakSet

import torch
import torch.nn as nn


__all__ = [
    "AttentionCaptureSpec",
    "AttentionParityError",
    "SampleAttentionDiagnostics",
    "SampleAttentionTrace",
    "SinkAblationResult",
    "SinkAblationSpec",
    "assert_attention_capture_parity",
    "assert_pooling_sink_attention_compatible",
    "capture_pooling_attention",
]


_MODELS_IN_CAPTURE: WeakSet[nn.Module] = WeakSet()
_BYTES_PER_FLOAT32 = 4


@dataclass(frozen=True)
class AttentionCaptureSpec:
    layers: tuple[int, ...]
    heads: tuple[int, ...]
    save_full_attention: bool = False
    max_combined_tokens: int = 1024
    max_attention_bytes: int = 268435456


@dataclass(frozen=True)
class SinkAblationSpec:
    name: str
    blocked_layers: tuple[int, ...] = ()


class AttentionParityError(RuntimeError):
    """Normal capture disagreed with the unhooked details forward."""

    def __init__(
        self,
        *,
        sample_index: int,
        field: str,
        max_error: float,
        atol: float,
        rtol: float,
    ) -> None:
        self.sample_index = sample_index
        self.field = field
        self.max_error = max_error
        self.atol = atol
        self.rtol = rtol
        super().__init__(
            f"attention capture parity failed for sample {sample_index} "
            f"{field}: max_error={max_error} (atol={atol}, rtol={rtol})"
        )


@dataclass(frozen=True)
class SinkAblationResult:
    """One capture condition, including the deployed ``normal`` pass.

    ``qbyt_score`` for a non-empty ``blocked_layers`` spec is the original
    calibrator applied to the intervened raw logit (原校准变换后的干预分数).
    It is not a claim that the intervened score remains calibrated.
    """

    spec: SinkAblationSpec
    trace: SampleAttentionTrace
    raw_logit: float
    qbyt_score: float
    threshold: float
    detected: bool
    delta_raw_logit: float
    delta_qbyt_score: float
    delta_position_logits: torch.Tensor


@dataclass(frozen=True)
class SampleAttentionDiagnostics:
    """Per-clip normal scores plus every requested sink-key ablation.

    ``normal_raw_logit``, ``normal_qbyt_score``, and ``threshold`` are the
    deployed (unintervened) values and are stored independently of the
    per-condition ``normal`` result.
    """

    normal_raw_logit: float
    normal_qbyt_score: float
    threshold: float
    normal: SinkAblationResult
    ablations: tuple[SinkAblationResult, ...]


@dataclass(frozen=True)
class SampleAttentionTrace:
    layer_ids: tuple[int, ...]
    head_ids: tuple[int, ...]
    audio_to_sink: torch.Tensor
    text_to_sink: torch.Tensor
    text_to_audio: torch.Tensor
    position_logits: torch.Tensor
    raw_logit: float
    text_length: int
    audio_length: int
    sink_index: int
    row_sum_max_error: float
    padding_mass_max: float
    full_attention: torch.Tensor | None = None


@dataclass(frozen=True)
class _LayerSampleSlice:
    audio_to_sink: torch.Tensor
    text_to_sink: torch.Tensor
    text_to_audio: torch.Tensor
    full_attention: torch.Tensor | None
    row_sum_max_error: float
    padding_mass_max: float


def capture_pooling_attention(
    qbyt,
    speech,
    anchors,
    speech_lengths,
    anchor_lengths,
    *,
    capture_spec: AttentionCaptureSpec,
    ablation_spec: SinkAblationSpec,
) -> list[SampleAttentionTrace]:
    """Run one pooling QbyT forward and slice per-sample attention traces."""

    _require_eval_fp32(qbyt, speech, anchors)
    if qbyt.sink_token is None:
        raise ValueError(
            "capture_pooling_attention requires a pooling QbyT with sink_token=True"
        )

    n_layers, n_heads = _matcher_geometry(qbyt)
    layer_ids = _validate_selected_indices(
        capture_spec.layers, bound=n_layers, name="layers"
    )
    head_ids = _validate_selected_indices(
        capture_spec.heads, bound=n_heads, name="heads"
    )
    blocked_layers = _validate_selected_indices(
        ablation_spec.blocked_layers, bound=n_layers, name="blocked_layers"
    )

    batch_size = int(speech.size(0))
    if anchors.size(0) != batch_size:
        raise ValueError(
            "speech and anchors batch sizes differ: "
            f"{speech.size(0)} vs {anchors.size(0)}"
        )
    speech_lengths = _as_cpu_lengths(
        speech_lengths, batch_size=batch_size, width=int(speech.size(1)), name="speech_lengths"
    )
    anchor_lengths = _as_cpu_lengths(
        anchor_lengths, batch_size=batch_size, width=int(anchors.size(1)), name="anchor_lengths"
    )

    packed_l = int(speech.size(1)) + int(anchors.size(1)) + 1
    if packed_l > capture_spec.max_combined_tokens:
        raise ValueError(
            f"packed sequence length {packed_l} exceeds max_combined_tokens="
            f"{capture_spec.max_combined_tokens}"
        )
    if capture_spec.save_full_attention:
        nbytes = (
            batch_size
            * len(layer_ids)
            * len(head_ids)
            * packed_l
            * packed_l
            * _BYTES_PER_FLOAT32
        )
        if nbytes > capture_spec.max_attention_bytes:
            raise ValueError(
                f"full attention storage estimate {nbytes} bytes exceeds "
                f"max_attention_bytes={capture_spec.max_attention_bytes}"
            )

    if qbyt in _MODELS_IN_CAPTURE:
        raise RuntimeError(
            "nested or concurrent capture on the same model is not supported"
        )

    captured: dict[int, list[_LayerSampleSlice] | None] = {
        layer_id: None for layer_id in layer_ids
    }
    text_lens = [int(length) for length in anchor_lengths.tolist()]
    audio_lens = [int(length) for length in speech_lengths.tolist()]
    if any(length < 1 for length in text_lens):
        raise ValueError("empty keywords are not supported for attention capture")
    if any(length < 1 for length in audio_lens):
        raise ValueError(
            "encoder valid frames T == 0 is unscorable for sink-key attention "
            "diagnostics"
        )
    head_index = torch.tensor(head_ids, dtype=torch.long)
    handles: list[Any] = []
    original_fastpath = torch.backends.mha.get_fastpath_enabled()
    layer_id_set = set(layer_ids)
    blocked_set = set(blocked_layers)
    _MODELS_IN_CAPTURE.add(qbyt)
    try:
        torch.backends.mha.set_fastpath_enabled(False)
        for layer_id in sorted(layer_id_set | blocked_set):
            self_attn = qbyt.phone_matchor.layers[layer_id].self_attn
            handles.append(
                self_attn.register_forward_pre_hook(
                    _make_self_attn_pre_hook(
                        force_weights=layer_id in layer_id_set,
                        block_sink=layer_id in blocked_set,
                        text_lens=text_lens,
                        audio_lens=audio_lens,
                    ),
                    with_kwargs=True,
                )
            )
            if layer_id not in layer_id_set:
                continue
            handles.append(
                self_attn.register_forward_hook(
                    _make_attention_forward_hook(
                        layer_id=layer_id,
                        captured=captured,
                        head_index=head_index,
                        text_lens=text_lens,
                        audio_lens=audio_lens,
                        save_full_attention=capture_spec.save_full_attention,
                    )
                )
            )
        with torch.no_grad():
            logits, _, details = qbyt.forward_with_readout_details(
                speech,
                anchors,
                speech_lengths=speech_lengths,
                text_lengths=anchor_lengths,
            )
        for layer_id in layer_ids:
            if captured[layer_id] is None:
                raise RuntimeError(
                    f"attention hook for layer {layer_id} did not fire; fused "
                    "MHA/encoder fastpath may have skipped the Python module"
                )
        if details.position_logits is None:
            raise ValueError(
                "capture_pooling_attention requires an EPS readout with "
                "final_pos_fc position logits"
            )
        return _assemble_traces(
            captured=captured,
            layer_ids=layer_ids,
            head_ids=head_ids,
            logits=logits,
            position_logits=details.position_logits,
            text_lens=text_lens,
            audio_lens=audio_lens,
            save_full_attention=capture_spec.save_full_attention,
        )
    finally:
        try:
            for handle in handles:
                handle.remove()
        finally:
            torch.backends.mha.set_fastpath_enabled(original_fastpath)
            _MODELS_IN_CAPTURE.discard(qbyt)


def _require_eval_fp32(qbyt, speech, anchors) -> None:
    if qbyt.training:
        raise RuntimeError(
            "capture_pooling_attention requires the model to already be in eval(); "
            "refusing to call eval() on a training-mode QbyT"
        )
    if speech.dtype != torch.float32:
        raise ValueError(
            f"capture_pooling_attention requires float32 speech, got {speech.dtype}"
        )
    try:
        param = next(qbyt.parameters())
    except StopIteration as exc:
        raise ValueError("capture_pooling_attention requires a parameterized QbyT") from exc
    if param.dtype != torch.float32:
        raise ValueError(
            "capture_pooling_attention requires a float32 QbyT, got "
            f"dtype={param.dtype}"
        )
    if speech.device != param.device or anchors.device != param.device:
        raise ValueError(
            "capture_pooling_attention requires speech and anchors on the QbyT "
            f"device, got module={param.device} speech={speech.device} "
            f"anchors={anchors.device}"
        )
    if _autocast_enabled():
        raise RuntimeError(
            "capture_pooling_attention does not support autocast; use FP32 eager"
        )


def _autocast_enabled() -> bool:
    if torch.is_autocast_enabled():
        return True
    try:
        return bool(torch.is_autocast_enabled("cpu"))
    except TypeError:
        cpu_enabled = getattr(torch, "is_autocast_cpu_enabled", None)
        return bool(callable(cpu_enabled) and cpu_enabled())


def _matcher_geometry(qbyt) -> tuple[int, int]:
    encoder = getattr(qbyt, "phone_matchor", None)
    layers = getattr(encoder, "layers", None)
    if layers is None:
        raise ValueError("QbyT.phone_matchor.layers is required for attention capture")
    n_layers = len(layers)
    if n_layers == 0:
        n_heads = int(getattr(qbyt, "nhead", 0))
        return n_layers, n_heads
    n_heads = int(layers[0].self_attn.num_heads)
    return n_layers, n_heads


def _validate_selected_indices(
    values: Sequence[int],
    *,
    bound: int,
    name: str,
) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)) or isinstance(values, bool):
        raise TypeError(f"{name} must be a sequence of 0-based integers, got {values!r}")
    selected: list[int] = []
    seen: set[int] = set()
    for raw in values:
        if isinstance(raw, bool) or not isinstance(raw, int):
            raise TypeError(f"{name} must be 0-based integers, got {raw!r}")
        if raw < 0 or raw >= bound:
            raise ValueError(f"{name} index {raw} is out of range [0, {bound})")
        if raw in seen:
            raise ValueError(f"{name} contains duplicate index {raw}")
        seen.add(raw)
        selected.append(raw)
    return tuple(selected)


def _as_cpu_lengths(
    lengths,
    *,
    batch_size: int,
    width: int,
    name: str,
) -> torch.Tensor:
    tensor = torch.as_tensor(lengths).detach().to(device="cpu", dtype=torch.long)
    if tensor.ndim != 1 or tensor.numel() != batch_size:
        raise ValueError(
            f"{name} must have shape ({batch_size},), got {tuple(tensor.shape)}"
        )
    if bool((tensor < 0).any()) or bool((tensor > width).any()):
        raise ValueError(
            f"{name} must satisfy 0 <= length <= {width}, got {tensor.tolist()}"
        )
    return tensor


def assert_pooling_sink_attention_compatible(qbyt, qbyt_score) -> None:
    """Require version=4 pooling ``eps_softmin`` with a live sink token."""

    if qbyt_score is None:
        raise ValueError(
            "attention diagnostics require a QbyT score spec with version=4 "
            "pooling eps_softmin and sink_token=true"
        )
    version = getattr(qbyt_score, "version", None)
    family = getattr(qbyt_score, "family", None)
    value = getattr(qbyt_score, "value", None)
    spec_mode = getattr(value, "mode", None)
    spec_sink = getattr(value, "sink_token", None)
    if version != 4:
        raise ValueError(
            "attention diagnostics require QbyT readout version 4 pooling "
            f"eps_softmin with sink_token=true, got version={version!r}"
        )
    if family != "pooling":
        raise ValueError(
            "attention diagnostics require family='pooling', got "
            f"{family!r}"
        )
    if spec_mode != "eps_softmin":
        raise ValueError(
            "attention diagnostics require qbyt_score mode='eps_softmin', "
            f"got {spec_mode!r}"
        )
    if spec_sink is not True:
        raise ValueError(
            "attention diagnostics require qbyt_score sink_token=true, "
            f"got {spec_sink!r}"
        )
    live_sink = getattr(qbyt, "sink_token", None)
    live_mode = getattr(qbyt, "readout_mode", None)
    if live_sink is None:
        raise ValueError(
            "spec mismatch: qbyt_score.sink_token=true but the loaded QbyT "
            "has no sink_token parameter"
        )
    if live_mode != "eps_softmin":
        raise ValueError(
            "spec mismatch: qbyt_score mode is eps_softmin but the loaded "
            f"QbyT readout_mode is {live_mode!r}"
        )


def assert_attention_capture_parity(
    traces: Sequence[SampleAttentionTrace],
    *,
    logits: torch.Tensor,
    position_logits: torch.Tensor,
    atol: float,
    rtol: float,
) -> None:
    """Compare normal capture logits to an unhooked details forward."""

    if position_logits is None:
        raise ValueError("parity comparison requires EPS position logits")
    expected_logits = logits.detach().to(device="cpu", dtype=torch.float32).reshape(-1)
    if int(expected_logits.numel()) != len(traces):
        raise RuntimeError(
            "parity comparison batch size mismatch: "
            f"{int(expected_logits.numel())} logits vs {len(traces)} traces"
        )
    worst_index = 0
    worst_field = "raw_logit"
    worst_error = 0.0
    failed = False
    for index, trace in enumerate(traces):
        raw_error = _parity_abs_error(
            torch.as_tensor(trace.raw_logit, dtype=torch.float32),
            expected_logits[index],
        )
        raw_excess = _parity_excess(
            torch.as_tensor(trace.raw_logit, dtype=torch.float32),
            expected_logits[index],
            atol=atol,
            rtol=rtol,
        )
        if raw_excess > 0.0 and (not failed or raw_error > worst_error):
            worst_index = index
            worst_field = "raw_logit"
            worst_error = raw_error
            failed = True
        expected_positions = position_logits[index, : trace.text_length].detach().to(
            device="cpu", dtype=torch.float32
        )
        pos_error = _parity_abs_error(trace.position_logits, expected_positions)
        pos_excess = _parity_excess(
            trace.position_logits, expected_positions, atol=atol, rtol=rtol
        )
        if pos_excess > 0.0 and (not failed or pos_error > worst_error):
            worst_index = index
            worst_field = "position_logits"
            worst_error = pos_error
            failed = True
    if failed:
        raise AttentionParityError(
            sample_index=worst_index,
            field=worst_field,
            max_error=worst_error,
            atol=atol,
            rtol=rtol,
        )


def _parity_abs_error(actual, expected) -> float:
    actual_tensor = torch.as_tensor(actual, dtype=torch.float32).reshape(-1)
    expected_tensor = torch.as_tensor(expected, dtype=torch.float32).reshape(-1)
    if actual_tensor.numel() == 0:
        return 0.0
    return float((actual_tensor - expected_tensor).abs().max().item())


def _parity_excess(actual, expected, *, atol: float, rtol: float) -> float:
    actual_tensor = torch.as_tensor(actual, dtype=torch.float32).reshape(-1)
    expected_tensor = torch.as_tensor(expected, dtype=torch.float32).reshape(-1)
    if actual_tensor.numel() == 0:
        return 0.0
    allowed = atol + rtol * expected_tensor.abs()
    return float((actual_tensor - expected_tensor).abs().sub(allowed).max().item())


def _force_attention_weights_pre_hook(module, args, kwargs):
    # MHA.forward is (query, key, value, key_padding_mask, need_weights, ...).
    # Rewrite in-place on a BoundArguments map so a value is never supplied both
    # positionally and as a keyword.
    signature = inspect.signature(module.forward)
    bound = signature.bind_partial(*args, **kwargs)
    bound.arguments["need_weights"] = True
    bound.arguments["average_attn_weights"] = False
    return bound.args, dict(bound.kwargs)


def _make_self_attn_pre_hook(
    *,
    force_weights: bool,
    block_sink: bool,
    text_lens: Sequence[int],
    audio_lens: Sequence[int],
):
    def _hook(module, args, kwargs):
        if force_weights:
            args, kwargs = _force_attention_weights_pre_hook(module, args, kwargs)
        if not block_sink:
            return args, kwargs
        signature = inspect.signature(module.forward)
        bound = signature.bind_partial(*args, **kwargs)
        query = bound.arguments["query"]
        attn_mask = bound.arguments.get("attn_mask")
        key_padding_mask = bound.arguments.get("key_padding_mask")
        new_attn_mask, new_key_padding_mask = _clone_and_block_sink_key(
            query=query,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            text_lens=text_lens,
            audio_lens=audio_lens,
            num_heads=int(module.num_heads),
            batch_first=bool(getattr(module, "batch_first", False)),
        )
        bound.arguments["attn_mask"] = new_attn_mask
        bound.arguments["key_padding_mask"] = new_key_padding_mask
        return bound.args, dict(bound.kwargs)

    return _hook


def _blocked_key_fill(mask: torch.Tensor):
    if mask.dtype == torch.bool:
        return True
    if torch.is_floating_point(mask):
        return float("-inf")
    raise RuntimeError(
        "sink-key ablation supports bool or floating attention masks, got "
        f"{mask.dtype}"
    )


def _clone_and_block_sink_key(
    *,
    query: torch.Tensor,
    attn_mask: torch.Tensor | None,
    key_padding_mask: torch.Tensor | None,
    text_lens: Sequence[int],
    audio_lens: Sequence[int],
    num_heads: int,
    batch_first: bool,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if not batch_first:
        raise RuntimeError(
            "sink-key ablation requires MultiheadAttention(batch_first=True)"
        )
    if query.dim() != 3:
        raise RuntimeError(
            f"expected batched query [B, L, E], got {tuple(query.shape)}"
        )
    batch_size, seq_len, _ = query.shape
    if len(text_lens) != batch_size or len(audio_lens) != batch_size:
        raise RuntimeError(
            "text/audio lengths do not match the attention batch size "
            f"{batch_size}"
        )
    for batch_index, (text_len, audio_len) in enumerate(zip(text_lens, audio_lens)):
        if int(text_len) + int(audio_len) <= 0:
            raise ValueError(
                "blocking the sink key would mask every remaining key for "
                f"sample {batch_index} (text_length={text_len}, "
                f"audio_length={audio_len})"
            )
        sink_index = int(text_len)
        if sink_index < 0 or sink_index >= seq_len:
            raise RuntimeError(
                f"sink key index {sink_index} is outside sequence length "
                f"{seq_len} for sample {batch_index}"
            )

    if attn_mask is not None:
        return (
            _block_sink_column_on_attn_mask(
                attn_mask,
                text_lens=text_lens,
                batch_size=batch_size,
                num_heads=num_heads,
                seq_len=seq_len,
            ),
            key_padding_mask,
        )
    if key_padding_mask is not None:
        return (
            attn_mask,
            _block_sink_on_key_padding_mask(
                key_padding_mask,
                text_lens=text_lens,
                batch_size=batch_size,
            ),
        )
    constructed = query.new_zeros(batch_size * num_heads, seq_len, seq_len)
    return (
        _fill_bh_attn_sink_columns(
            constructed,
            text_lens=text_lens,
            num_heads=num_heads,
        ),
        key_padding_mask,
    )


def _block_sink_column_on_attn_mask(
    attn_mask: torch.Tensor,
    *,
    text_lens: Sequence[int],
    batch_size: int,
    num_heads: int,
    seq_len: int,
) -> torch.Tensor:
    mask = attn_mask.detach().clone()
    if mask.size(-1) < seq_len:
        raise RuntimeError(
            f"attn_mask key width {mask.size(-1)} is smaller than sequence "
            f"length {seq_len}"
        )
    if mask.dim() == 2:
        mask = mask.unsqueeze(0).repeat(batch_size * num_heads, 1, 1)
        return _fill_bh_attn_sink_columns(
            mask, text_lens=text_lens, num_heads=num_heads
        )
    if mask.dim() == 3:
        if mask.size(0) == batch_size * num_heads:
            return _fill_bh_attn_sink_columns(
                mask, text_lens=text_lens, num_heads=num_heads
            )
        if mask.size(0) == batch_size:
            fill = _blocked_key_fill(mask)
            for batch_index, text_len in enumerate(text_lens):
                mask[batch_index, :, int(text_len)] = fill
            return mask
        raise RuntimeError(
            "expected attn_mask leading dim batch*heads or batch, got "
            f"{tuple(mask.shape)}"
        )
    if mask.dim() == 4:
        fill = _blocked_key_fill(mask)
        for batch_index, text_len in enumerate(text_lens):
            mask[batch_index, :, :, int(text_len)] = fill
        return mask
    raise RuntimeError(
        f"unsupported attn_mask rank {mask.dim()} with shape {tuple(mask.shape)}"
    )


def _fill_bh_attn_sink_columns(
    mask: torch.Tensor,
    *,
    text_lens: Sequence[int],
    num_heads: int,
) -> torch.Tensor:
    fill = _blocked_key_fill(mask)
    for batch_index, text_len in enumerate(text_lens):
        start = batch_index * num_heads
        mask[start : start + num_heads, :, int(text_len)] = fill
    return mask


def _block_sink_on_key_padding_mask(
    key_padding_mask: torch.Tensor,
    *,
    text_lens: Sequence[int],
    batch_size: int,
) -> torch.Tensor:
    mask = key_padding_mask.detach().clone()
    fill = _blocked_key_fill(mask)
    if mask.dim() == 1:
        if batch_size != 1:
            raise RuntimeError(
                "1-D key_padding_mask requires batch size 1, got "
                f"{batch_size}"
            )
        mask[int(text_lens[0])] = fill
        return mask
    if mask.dim() != 2 or mask.size(0) != batch_size:
        raise RuntimeError(
            "expected key_padding_mask shape [batch, keys], got "
            f"{tuple(mask.shape)}"
        )
    for batch_index, text_len in enumerate(text_lens):
        mask[batch_index, int(text_len)] = fill
    return mask


def _make_attention_forward_hook(
    *,
    layer_id: int,
    captured: dict[int, list[_LayerSampleSlice] | None],
    head_index: torch.Tensor,
    text_lens: Sequence[int],
    audio_lens: Sequence[int],
    save_full_attention: bool,
):
    def _hook(_module, _inputs, output):
        if captured[layer_id] is not None:
            raise RuntimeError(
                f"attention hook for layer {layer_id} fired more than once in one forward"
            )
        if not isinstance(output, tuple) or len(output) < 2:
            raise RuntimeError(
                "phone_matchor self_attn did not return (attn_output, attn_weights); "
                f"got {type(output).__name__}"
            )
        weights = output[1]
        if weights is None:
            raise RuntimeError(
                "attention weights are None; need_weights=True was ignored or the "
                "fused fastpath skipped the Python MultiheadAttention"
            )
        if weights.dim() != 4:
            raise RuntimeError(
                "expected unaveraged attention weights of shape [batch, heads, L, L], "
                f"got {tuple(weights.shape)}"
            )
        selected_heads = weights.index_select(1, head_index.to(device=weights.device))
        captured[layer_id] = [
            _slice_sample(
                selected_heads[batch_index],
                text_len=text_lens[batch_index],
                audio_len=audio_lens[batch_index],
                save_full_attention=save_full_attention,
            )
            for batch_index in range(selected_heads.size(0))
        ]
        return None

    return _hook


def _slice_sample(
    weights: torch.Tensor,
    *,
    text_len: int,
    audio_len: int,
    save_full_attention: bool,
) -> _LayerSampleSlice:
    # Packed layout after pooling re-pack: [valid text][sink][valid audio][pad].
    # U/T are caller lengths, not pooling's internal speech_lengths + 1.
    sink_index = text_len
    audio_start = text_len + 1
    audio_end = audio_start + audio_len
    valid_l = text_len + 1 + audio_len
    seq_len = int(weights.size(-1))
    if valid_l > seq_len:
        raise RuntimeError(
            f"valid packed length {valid_l} exceeds attention key width {seq_len}"
        )
    row_sum_max_error, padding_mass_max = _row_diagnostics(
        weights, valid_l=valid_l, seq_len=seq_len
    )
    return _LayerSampleSlice(
        audio_to_sink=_to_cpu(weights[:, audio_start:audio_end, sink_index]),
        text_to_sink=_to_cpu(weights[:, :text_len, sink_index]),
        text_to_audio=_to_cpu(weights[:, :text_len, audio_start:audio_end]),
        full_attention=(
            _to_cpu(weights[:, :valid_l, :valid_l]) if save_full_attention else None
        ),
        row_sum_max_error=row_sum_max_error,
        padding_mass_max=padding_mass_max,
    )


def _row_diagnostics(weights: torch.Tensor, *, valid_l: int, seq_len: int) -> tuple[float, float]:
    if weights.size(0) == 0 or valid_l == 0:
        return 0.0, 0.0
    valid_rows = weights[:, :valid_l, :]
    row_sum_max_error = float((valid_rows.sum(dim=-1) - 1.0).abs().max().item())
    if valid_l < seq_len:
        padding_mass_max = float(valid_rows[:, :, valid_l:].sum(dim=-1).max().item())
    else:
        padding_mass_max = 0.0
    return row_sum_max_error, padding_mass_max


def _to_cpu(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().to(device="cpu", dtype=torch.float32).contiguous().clone()


def _assemble_traces(
    *,
    captured: dict[int, list[_LayerSampleSlice] | None],
    layer_ids: tuple[int, ...],
    head_ids: tuple[int, ...],
    logits: torch.Tensor,
    position_logits: torch.Tensor,
    text_lens: Sequence[int],
    audio_lens: Sequence[int],
    save_full_attention: bool,
) -> list[SampleAttentionTrace]:
    batch_size = len(text_lens)
    traces: list[SampleAttentionTrace] = []
    n_heads = len(head_ids)
    for batch_index in range(batch_size):
        text_len = text_lens[batch_index]
        audio_len = audio_lens[batch_index]
        if layer_ids:
            layer_slices = []
            for layer_id in layer_ids:
                layer_batch = captured[layer_id]
                if layer_batch is None:
                    raise RuntimeError(
                        f"attention hook for layer {layer_id} did not fire; fused "
                        "MHA/encoder fastpath may have skipped the Python module"
                    )
                layer_slices.append(layer_batch[batch_index])
            audio_to_sink = torch.stack(
                [item.audio_to_sink for item in layer_slices], dim=0
            )
            text_to_sink = torch.stack(
                [item.text_to_sink for item in layer_slices], dim=0
            )
            text_to_audio = torch.stack(
                [item.text_to_audio for item in layer_slices], dim=0
            )
            row_sum_max_error = max(item.row_sum_max_error for item in layer_slices)
            padding_mass_max = max(item.padding_mass_max for item in layer_slices)
            if save_full_attention:
                full_attention = torch.stack(
                    [item.full_attention for item in layer_slices], dim=0
                )
            else:
                full_attention = None
        else:
            audio_to_sink = torch.zeros(0, n_heads, audio_len, dtype=torch.float32)
            text_to_sink = torch.zeros(0, n_heads, text_len, dtype=torch.float32)
            text_to_audio = torch.zeros(
                0, n_heads, text_len, audio_len, dtype=torch.float32
            )
            row_sum_max_error = 0.0
            padding_mass_max = 0.0
            full_attention = None
        traces.append(
            SampleAttentionTrace(
                layer_ids=layer_ids,
                head_ids=head_ids,
                audio_to_sink=audio_to_sink,
                text_to_sink=text_to_sink,
                text_to_audio=text_to_audio,
                position_logits=_to_cpu(position_logits[batch_index, :text_len]),
                raw_logit=float(logits[batch_index].detach().cpu()),
                text_length=text_len,
                audio_length=audio_len,
                sink_index=text_len,
                row_sum_max_error=row_sum_max_error,
                padding_mass_max=padding_mass_max,
                full_attention=full_attention,
            )
        )
    return traces
