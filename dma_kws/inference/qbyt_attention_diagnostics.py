"""Temporary-hook capture of pooling QbyT ``nn.MultiheadAttention`` weights.

This module does not copy the Transformer forward and does not change QbyT
parameters, buffers, or ``state_dict`` keys. Capture is CPU float32 eager only.
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
    "SampleAttentionTrace",
    "SinkAblationSpec",
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

    _require_eval_cpu_fp32(qbyt, speech, anchors)
    if ablation_spec.blocked_layers:
        raise ValueError(
            "SinkAblationSpec.blocked_layers is not supported until sink-key "
            f"ablation is implemented; got {tuple(ablation_spec.blocked_layers)!r}"
        )
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
    head_index = torch.tensor(head_ids, dtype=torch.long)
    handles: list[Any] = []
    original_fastpath = torch.backends.mha.get_fastpath_enabled()
    _MODELS_IN_CAPTURE.add(qbyt)
    try:
        torch.backends.mha.set_fastpath_enabled(False)
        for layer_id in layer_ids:
            self_attn = qbyt.phone_matchor.layers[layer_id].self_attn
            handles.append(
                self_attn.register_forward_pre_hook(
                    _force_attention_weights_pre_hook, with_kwargs=True
                )
            )
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


def _require_eval_cpu_fp32(qbyt, speech, anchors) -> None:
    if qbyt.training:
        raise RuntimeError(
            "capture_pooling_attention requires the model to already be in eval(); "
            "refusing to call eval() on a training-mode QbyT"
        )
    if speech.device.type != "cpu" or anchors.device.type != "cpu":
        raise ValueError("capture_pooling_attention supports CPU tensors only")
    if speech.dtype != torch.float32:
        raise ValueError(
            f"capture_pooling_attention requires float32 speech, got {speech.dtype}"
        )
    try:
        param = next(qbyt.parameters())
    except StopIteration as exc:
        raise ValueError("capture_pooling_attention requires a parameterized QbyT") from exc
    if param.device.type != "cpu" or param.dtype != torch.float32:
        raise ValueError(
            "capture_pooling_attention requires a CPU float32 QbyT, got "
            f"device={param.device} dtype={param.dtype}"
        )
    if torch.is_autocast_enabled():
        raise RuntimeError(
            "capture_pooling_attention does not support autocast; use FP32 eager"
        )


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
    tensor = torch.as_tensor(lengths, dtype=torch.long, device="cpu")
    if tensor.ndim != 1 or tensor.numel() != batch_size:
        raise ValueError(
            f"{name} must have shape ({batch_size},), got {tuple(tensor.shape)}"
        )
    if bool((tensor < 0).any()) or bool((tensor > width).any()):
        raise ValueError(
            f"{name} must satisfy 0 <= length <= {width}, got {tensor.tolist()}"
        )
    return tensor


def _force_attention_weights_pre_hook(module, args, kwargs):
    # MHA.forward is (query, key, value, key_padding_mask, need_weights, ...).
    # Rewrite in-place on a BoundArguments map so a value is never supplied both
    # positionally and as a keyword.
    signature = inspect.signature(module.forward)
    bound = signature.bind_partial(*args, **kwargs)
    bound.arguments["need_weights"] = True
    bound.arguments["average_attn_weights"] = False
    return bound.args, dict(bound.kwargs)


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
