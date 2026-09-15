"""Time-axis mapping, Matplotlib PNGs, and an offline-local HTML report.

Rebuild figures from a diagnose_qbyt_sink run directory (``run.json`` / CSV /
NPZ) without the original WAV. Spectrogram data come from ``prepared_waveform``
in traces, not a second file load.

Matplotlib is imported lazily when figures are drawn. If it is missing::

    pip install matplotlib>=3.8
"""

from __future__ import annotations

import csv
import json
import math
import re
from dataclasses import dataclass
from html import escape as html_escape
from pathlib import Path
from typing import Any, Mapping, Sequence


__all__ = [
    "EncoderTimeMap",
    "FbankTimeSpec",
    "SYNTHETIC_FIXTURE_BANNER",
    "SampleTimeAxis",
    "assemble_text_key_heatmap",
    "build_sample_time_axis",
    "extra_region_metric_rows",
    "fbank_frame_center_sec",
    "inspect_encoder_time_map",
    "length_bin_label",
    "noise_region_masks",
    "pair_time_grids_comparable",
    "region_stats",
    "render_sink_attention_report",
    "serialize_encoder_time_map",
]

SYNTHETIC_FIXTURE_BANNER = "合成 fixture / 非训练模型结论"


_MAX_FRAME_SEARCH = 10_000
_LOG_FLOOR = 1e-6
TIME_AXIS_UNAVAILABLE = "unavailable"
_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")


@dataclass(frozen=True)
class FbankTimeSpec:
    frame_length_ms: float
    frame_shift_ms: float
    snip_edges: bool
    model_sample_rate: int
    backend: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "frame_length_ms": float(self.frame_length_ms),
            "frame_shift_ms": float(self.frame_shift_ms),
            "snip_edges": bool(self.snip_edges),
            "model_sample_rate": int(self.model_sample_rate),
            "backend": str(self.backend),
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any] | None) -> "FbankTimeSpec | None":
        if not payload:
            return None
        try:
            return cls(
                frame_length_ms=float(payload["frame_length_ms"]),
                frame_shift_ms=float(payload["frame_shift_ms"]),
                snip_edges=bool(payload["snip_edges"]),
                model_sample_rate=int(
                    payload.get("model_sample_rate")
                    or payload.get("sample_rate")
                    or 0
                ),
                backend=str(payload.get("backend", "")),
            )
        except (KeyError, TypeError, ValueError):
            return None


@dataclass(frozen=True)
class EncoderTimeMap:
    method: str
    kind: str
    subsampling_rate: int = 1
    right_context: int = 0
    offset: int = 0
    stride: int = 1
    receptive_field: int = 1
    lookup: tuple[int, ...] | None = None

    def output_frames(self, num_input_frames: int) -> int:
        n = int(num_input_frames)
        if n <= 0:
            return 0
        if self.stride >= 1 and self.receptive_field >= 1:
            needed = int(self.offset) + int(self.receptive_field)
            if n < needed:
                return 0
            return (n - needed) // int(self.stride) + 1
        if self.kind == "identity":
            return n
        if self.kind == "wenet":
            span = n - int(self.right_context) - 1
            if span < 0:
                return 0
            return span // int(self.subsampling_rate) + 1
        if self.kind == "icefall":
            from dma_kws.stage2.icefall_encoder import (
                OUTPUT_DOWNSAMPLING_FACTOR,
                embed_output_frames,
            )

            subsampled = embed_output_frames(n)
            if subsampled <= 0:
                return 0
            return (subsampled + 1) // OUTPUT_DOWNSAMPLING_FACTOR
        if self.lookup is not None:
            if n >= len(self.lookup):
                raise ValueError(
                    f"encoder output_frames lookup ends at {len(self.lookup) - 1}, got {n}"
                )
            return int(self.lookup[n])
        raise ValueError(f"encoder time map kind {self.kind!r} cannot evaluate output_frames")

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "method": self.method,
            "kind": self.kind,
            "subsampling_rate": int(self.subsampling_rate),
            "right_context": int(self.right_context),
            "offset": int(self.offset),
            "stride": int(self.stride),
            "receptive_field": int(self.receptive_field),
        }
        if self.lookup is not None:
            payload["output_frames_lookup"] = [int(v) for v in self.lookup]
        return payload


@dataclass(frozen=True)
class SampleTimeAxis:
    status: str
    method: str
    centers_source_sec: Any
    is_padding: Any
    is_left_padding: Any
    is_right_padding: Any
    fbank_support: Any
    encoder_frame_index: Any
    left_padding_sec: float = 0.0
    right_padding_sec: float = 0.0
    source_duration_sec: float = 0.0

    @classmethod
    def unavailable(cls, audio_length: int, *, method: str = TIME_AXIS_UNAVAILABLE) -> "SampleTimeAxis":
        import numpy as np

        n = max(0, int(audio_length))
        empty = np.zeros((0,), dtype=np.float64)
        return cls(
            status=TIME_AXIS_UNAVAILABLE,
            method=method,
            centers_source_sec=empty,
            is_padding=np.zeros((0,), dtype=bool),
            is_left_padding=np.zeros((0,), dtype=bool),
            is_right_padding=np.zeros((0,), dtype=bool),
            fbank_support=np.zeros((0, 2), dtype=np.int64),
            encoder_frame_index=np.arange(n, dtype=np.int64),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "method": self.method,
            "centers_source_sec": _json_floats(self.centers_source_sec),
            "is_padding": _json_bools(self.is_padding),
            "is_left_padding": _json_bools(self.is_left_padding),
            "is_right_padding": _json_bools(self.is_right_padding),
            "fbank_support": _json_int_pairs(self.fbank_support),
            "encoder_frame_index": _json_ints(self.encoder_frame_index),
            "left_padding_sec": float(self.left_padding_sec),
            "right_padding_sec": float(self.right_padding_sec),
            "source_duration_sec": float(self.source_duration_sec),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any] | None, *, audio_length: int = 0) -> "SampleTimeAxis":
        import numpy as np

        if not payload:
            return cls.unavailable(int(audio_length))
        axis_status = str(
            payload.get("time_axis_status") or payload.get("status") or ""
        )
        if axis_status != "ok":
            return cls.unavailable(
                int(audio_length),
                method=str(payload.get("time_axis_method") or payload.get("method") or TIME_AXIS_UNAVAILABLE),
            )
        centers = np.asarray(payload.get("centers_source_sec") or [], dtype=np.float64)
        if centers.size == 0:
            return cls.unavailable(
                int(audio_length),
                method=str(payload.get("time_axis_method") or payload.get("method") or TIME_AXIS_UNAVAILABLE),
            )
        padding = np.asarray(payload.get("is_padding") or [False] * int(centers.size), dtype=bool)
        support = np.asarray(payload.get("fbank_support") or np.zeros((centers.size, 2)), dtype=np.int64)
        index = np.asarray(
            payload.get("encoder_frame_index") or list(range(int(centers.size))),
            dtype=np.int64,
        )
        return cls(
            status="ok",
            method=str(payload.get("method") or ""),
            centers_source_sec=centers,
            is_padding=padding,
            is_left_padding=np.asarray(
                payload.get("is_left_padding") or padding, dtype=bool
            ),
            is_right_padding=np.asarray(
                payload.get("is_right_padding") or np.zeros_like(padding), dtype=bool
            ),
            fbank_support=support if support.size else np.zeros((centers.size, 2), dtype=np.int64),
            encoder_frame_index=index,
            left_padding_sec=float(payload.get("left_padding_sec") or 0.0),
            right_padding_sec=float(payload.get("right_padding_sec") or 0.0),
            source_duration_sec=float(payload.get("source_duration_sec") or 0.0),
        )


def fbank_frame_center_sec(
    frame_index: int,
    *,
    frame_length_ms: float,
    frame_shift_ms: float,
    snip_edges: bool,
) -> float:
    """Nominal Kaldi fbank frame center in model-input seconds."""

    index = int(frame_index)
    length = float(frame_length_ms)
    shift = float(frame_shift_ms)
    if snip_edges:
        return (index * shift + length / 2.0) / 1000.0
    return (index * shift) / 1000.0


def inspect_encoder_time_map(encoder) -> EncoderTimeMap | None:
    """Return a verified fbank→encoder mapping, or ``None`` if unverifiable."""

    if encoder is None:
        return None
    from dma_kws.nn import encoder_output_frames

    probes = (1, 2, 3, 4, 7, 8, 9, 11, 15, 16, 32)
    try:
        values = [int(encoder_output_frames(encoder, n)) for n in probes]
    except (TypeError, ValueError):
        return None
    if any(value < 0 for value in values):
        return None

    def _output_fn(num_input_frames: int) -> int:
        return int(encoder_output_frames(encoder, num_input_frames))

    geometry = _fit_regular_geometry(_output_fn)
    if geometry is None:
        return None
    offset, stride, receptive_field = geometry

    if all(int(encoder_output_frames(encoder, n)) == n for n in probes):
        return EncoderTimeMap(
            method="kaldi_fbank_center+identity_output_frames",
            kind="identity",
            offset=offset,
            stride=stride,
            receptive_field=receptive_field,
        )

    embed = getattr(encoder, "embed", None)
    rate = getattr(embed, "subsampling_rate", None)
    right_context = getattr(embed, "right_context", None)
    if rate is not None and right_context is not None:
        expected = []
        for n in probes:
            span = n - int(right_context) - 1
            expected.append(0 if span < 0 else span // int(rate) + 1)
        if (
            values == expected
            and stride == int(rate)
            and receptive_field == int(right_context) + 1
        ):
            return EncoderTimeMap(
                method="kaldi_fbank_center+wenet_embed",
                kind="wenet",
                subsampling_rate=int(rate),
                right_context=int(right_context),
                offset=offset,
                stride=stride,
                receptive_field=receptive_field,
            )

    try:
        from dma_kws.stage2.icefall_encoder import (
            OUTPUT_DOWNSAMPLING_FACTOR,
            embed_output_frames,
        )

        icefall = []
        for n in probes:
            subsampled = embed_output_frames(n)
            icefall.append(
                0 if subsampled <= 0 else (subsampled + 1) // OUTPUT_DOWNSAMPLING_FACTOR
            )
        if values == icefall:
            return EncoderTimeMap(
                method="kaldi_fbank_center+icefall_zipformer",
                kind="icefall",
                offset=offset,
                stride=stride,
                receptive_field=receptive_field,
            )
    except Exception:
        pass

    return EncoderTimeMap(
        method="kaldi_fbank_center+regular_subsampling",
        kind="regular",
        subsampling_rate=stride,
        right_context=receptive_field - 1,
        offset=offset,
        stride=stride,
        receptive_field=receptive_field,
    )


def serialize_encoder_time_map(
    time_map: EncoderTimeMap | None, *, max_input_frames: int = 0
) -> dict[str, Any]:
    if time_map is None:
        return {"method": TIME_AXIS_UNAVAILABLE, "kind": TIME_AXIS_UNAVAILABLE}
    payload = time_map.as_dict()
    if time_map.kind == "lookup" or max_input_frames > 0 and time_map.lookup is None:
        cap = max(int(max_input_frames), 0)
        if time_map.lookup is not None:
            cap = max(cap, len(time_map.lookup) - 1)
        if cap > 0 and time_map.kind == "lookup":
            lookup = [int(time_map.output_frames(n)) for n in range(cap + 1)]
            payload["output_frames_lookup"] = lookup
    return payload


def encoder_time_map_from_json(payload: Mapping[str, Any] | None) -> EncoderTimeMap | None:
    if not payload:
        return None
    kind = str(payload.get("kind") or "")
    method = str(payload.get("method") or TIME_AXIS_UNAVAILABLE)
    if kind in {"", TIME_AXIS_UNAVAILABLE, "none"}:
        return None
    lookup = payload.get("output_frames_lookup")
    lookup_t = tuple(int(v) for v in lookup) if isinstance(lookup, list) else None
    return EncoderTimeMap(
        method=method,
        kind=kind,
        subsampling_rate=int(payload.get("subsampling_rate") or 1),
        right_context=int(payload.get("right_context") or 0),
        offset=int(payload.get("offset") or 0),
        stride=int(payload.get("stride") or 1),
        receptive_field=int(payload.get("receptive_field") or 1),
        lookup=lookup_t,
    )


def _min_input_reaching(output_fn, min_out: int) -> int:
    if min_out <= 0:
        return 0
    n = 1
    while int(output_fn(n)) < min_out:
        n += 1
        if n > _MAX_FRAME_SEARCH:
            raise ValueError(
                f"encoder output_frames does not reach {min_out} within {_MAX_FRAME_SEARCH}"
            )
    return n


def _min_input_for_output(time_map: EncoderTimeMap, min_out: int) -> int:
    return _min_input_reaching(time_map.output_frames, min_out)


def _fit_regular_geometry(output_fn) -> tuple[int, int, int] | None:
    """Fit offset=0, constant stride, and overlapping receptive field.

    ``last_i`` is the last fbank index needed to emit output ``i``. For a
    regular strided convolution that starts at input 0, ``last_i = i * stride
    + rf - 1``. Using newly-added frames between outputs as the support would
    drop the overlapping receptive field.
    """

    lasts: list[int] = []
    try:
        for index in range(6):
            lasts.append(_min_input_reaching(output_fn, index + 1) - 1)
    except ValueError:
        pass
    if len(lasts) < 2:
        return None
    stride = lasts[1] - lasts[0]
    if stride < 1:
        return None
    offset = 0
    receptive_field = lasts[0] - offset + 1
    if receptive_field < 1:
        return None
    for index, last in enumerate(lasts):
        expected = offset + index * stride + receptive_field - 1
        if last != expected:
            return None
        if int(output_fn(last + 1)) != index + 1:
            return None
        if int(output_fn(last)) != index:
            return None
    return offset, stride, receptive_field


def _fbank_support(time_map: EncoderTimeMap, encoder_index: int) -> tuple[int, int]:
    stride = int(time_map.stride)
    receptive_field = int(time_map.receptive_field)
    offset = int(time_map.offset)
    if stride < 1 or receptive_field < 1:
        raise ValueError("encoder time map is missing stride/receptive_field")
    first = offset + int(encoder_index) * stride
    last = first + receptive_field - 1
    if last < first:
        last = first
    return int(first), int(last)


def build_sample_time_axis(
    *,
    audio_length: int,
    encoder_map: EncoderTimeMap | None,
    fbank: FbankTimeSpec | None,
    left_padding_ms: int,
    right_padding_ms: int,
    source_duration_sec: float,
    num_fbank_frames: int | None = None,
) -> SampleTimeAxis:
    """Nominal encoder-frame centers in source seconds, or unavailable."""

    import numpy as np

    t_count = int(audio_length)
    if t_count <= 0 or encoder_map is None or fbank is None:
        return SampleTimeAxis.unavailable(max(t_count, 0))
    try:
        if int(encoder_map.stride) < 1 or int(encoder_map.receptive_field) < 1:
            return SampleTimeAxis.unavailable(t_count)
        if num_fbank_frames is not None:
            predicted = encoder_map.output_frames(int(num_fbank_frames))
            if predicted != t_count:
                return SampleTimeAxis.unavailable(t_count)
        supports = np.zeros((t_count, 2), dtype=np.int64)
        for index in range(t_count):
            supports[index] = _fbank_support(encoder_map, index)
    except (TypeError, ValueError):
        return SampleTimeAxis.unavailable(t_count)

    left_sec = float(left_padding_ms) / 1000.0
    right_sec = float(right_padding_ms) / 1000.0
    duration = float(source_duration_sec)
    centers = np.empty(t_count, dtype=np.float64)
    for index in range(t_count):
        first = int(supports[index, 0])
        last = int(supports[index, 1])
        owned = [
            fbank_frame_center_sec(
                frame,
                frame_length_ms=fbank.frame_length_ms,
                frame_shift_ms=fbank.frame_shift_ms,
                snip_edges=fbank.snip_edges,
            )
            for frame in range(first, last + 1)
        ]
        model_center = float(sum(owned) / len(owned))
        centers[index] = model_center - left_sec
    is_left = centers < 0.0
    is_right = centers > duration
    is_padding = is_left | is_right
    return SampleTimeAxis(
        status="ok",
        method=encoder_map.method,
        centers_source_sec=centers,
        is_padding=is_padding,
        is_left_padding=is_left,
        is_right_padding=is_right,
        fbank_support=supports,
        encoder_frame_index=np.arange(t_count, dtype=np.int64),
        left_padding_sec=left_sec,
        right_padding_sec=right_sec,
        source_duration_sec=duration,
    )


def noise_region_masks(
    axis: SampleTimeAxis,
    noise_spans: Sequence[Sequence[float]] | None,
) -> tuple[Any, Any, list[dict[str, Any]]]:
    """Return (noise_union, outside_noise_annotation, per-interval records).

    Inclusion uses the nominal encoder-frame center. Overlapping spans are
    unioned so a frame is counted once. This is position correspondence, not
    receptive-field attribution.
    """

    import numpy as np

    if axis.status != "ok" or axis.centers_source_sec.size == 0:
        empty = np.zeros((0,), dtype=bool)
        return empty, empty, []
    centers = np.asarray(axis.centers_source_sec, dtype=np.float64)
    padding = np.asarray(axis.is_padding, dtype=bool)
    valid = ~padding
    if noise_spans is None:
        none = np.zeros(centers.shape, dtype=bool)
        return none, none, []

    noise = np.zeros(centers.shape, dtype=bool)
    intervals: list[dict[str, Any]] = []
    for span in noise_spans:
        if len(span) < 2:
            continue
        start = float(span[0])
        end = float(span[1])
        in_span = valid & _centers_in_span(centers, start, end)
        intervals.append(
            {
                "start": start,
                "end": end,
                "valid_frame_count": int(in_span.sum()),
            }
        )
        noise |= in_span
    outside = valid & ~noise
    return noise, outside, intervals


def _centers_in_span(centers, start: float, end: float):
    """Manifest spans are half-open ``[start, end)``."""

    return (centers >= float(start)) & (centers < float(end))


def region_stats(values, mask) -> dict[str, Any]:
    import numpy as np

    data = np.asarray(values, dtype=np.float64).reshape(-1)
    flags = np.asarray(mask, dtype=bool).reshape(-1)
    if data.size != flags.size:
        raise ValueError("values and mask must have the same length")
    selected = data[flags]
    if selected.size == 0:
        return {"mean": None, "min": None, "max": None, "query_count": 0}
    return {
        "mean": float(selected.mean()),
        "min": float(selected.min()),
        "max": float(selected.max()),
        "query_count": int(selected.size),
    }


def extra_region_metric_rows(
    *,
    run_id: str,
    sample_id: str,
    ablation: str,
    layer_ids: Sequence[int],
    head_ids: Sequence[int],
    audio_to_sink,
    time_axis: SampleTimeAxis,
    noise_spans: Sequence[Sequence[float]] | None,
    row_sum_error: float,
) -> list[dict[str, str]]:
    """CSV rows for ``noise`` / ``outside_noise_annotation`` when mapping works."""

    if time_axis.status != "ok" or noise_spans is None:
        return []
    tensor = _as_numpy(audio_to_sink)
    noise, outside, _intervals = noise_region_masks(time_axis, noise_spans)
    rows: list[dict[str, str]] = []
    for layer_index, layer_id in enumerate(layer_ids):
        for head_index, head_id in enumerate(head_ids):
            values = tensor[layer_index, head_index].reshape(-1)
            for region, mask in (("noise", noise), ("outside_noise_annotation", outside)):
                stats = region_stats(values, mask)
                rows.append(
                    {
                        "run_id": run_id,
                        "sample_id": sample_id,
                        "ablation": ablation,
                        "layer": str(int(layer_id)),
                        "head": str(int(head_id)),
                        "region": region,
                        "mean": _fmt_float(stats["mean"]),
                        "min": _fmt_float(stats["min"]),
                        "max": _fmt_float(stats["max"]),
                        "query_count": str(int(stats["query_count"])),
                        "key_count": "1",
                        "row_sum_error": _fmt_float(float(row_sum_error)),
                    }
                )
    return rows


def pair_time_grids_comparable(
    axis_a: SampleTimeAxis,
    axis_b: SampleTimeAxis,
    *,
    token_ids_a: Sequence[int],
    token_ids_b: Sequence[int],
    source_duration_a: float,
    source_duration_b: float,
    fbank_a: FbankTimeSpec | None,
    fbank_b: FbankTimeSpec | None,
    duration_atol: float = 1e-4,
    center_atol: float = 1e-5,
) -> tuple[bool, str]:
    """True when noisy−clean frame diffs need no interpolation."""

    if axis_a.status != "ok" or axis_b.status != "ok":
        return False, "time_axis_unavailable"
    if list(token_ids_a) != list(token_ids_b):
        return False, "token_sequence_mismatch"
    if abs(float(source_duration_a) - float(source_duration_b)) > duration_atol:
        return False, "source_duration_mismatch"
    if fbank_a is None or fbank_b is None or fbank_a.as_dict() != fbank_b.as_dict():
        return False, "frontend_mismatch"
    if int(axis_a.encoder_frame_index.size) != int(axis_b.encoder_frame_index.size):
        return False, "valid_frame_count_mismatch"
    import numpy as np

    if not np.allclose(
        axis_a.centers_source_sec,
        axis_b.centers_source_sec,
        atol=center_atol,
        rtol=0.0,
    ):
        return False, "nominal_time_grid_mismatch"
    if not np.array_equal(axis_a.is_padding, axis_b.is_padding):
        return False, "padding_mask_mismatch"
    return True, ""


def length_bin_label(length: float, bins: Sequence[float]) -> str:
    """Assign ``length`` to a half-open bin; the last edge starts the overflow bin."""

    edges = [float(item) for item in bins]
    if len(edges) < 2:
        return "all"
    value = float(length)
    for index in range(len(edges) - 1):
        if edges[index] <= value < edges[index + 1]:
            lo = _trim_bin_edge(edges[index])
            hi = _trim_bin_edge(edges[index + 1])
            return f"{lo}-{hi}"
    last = _trim_bin_edge(edges[-1])
    return f"{last}+"


def assemble_text_key_heatmap(text_to_sink, text_to_audio):
    """Concatenate sink as column 0 with audio keys. Shape ``(U, 1+T)``."""

    import numpy as np

    sink = np.asarray(text_to_sink, dtype=np.float32).reshape(-1)
    audio = np.asarray(text_to_audio, dtype=np.float32)
    if audio.ndim == 1:
        audio = audio.reshape(sink.size, -1)
    if audio.shape[0] != sink.size:
        raise ValueError(
            f"text_to_audio first dim {audio.shape[0]} != text length {sink.size}"
        )
    return np.concatenate([sink[:, None], audio], axis=1)


def render_sink_attention_report(
    output_dir: str | Path,
    *,
    encoder=None,
) -> dict[str, Any]:
    """Write ``report.html`` and ``figures/`` from a finished run directory."""

    output_dir = Path(output_dir)
    run_path = output_dir / "run.json"
    summary_path = output_dir / "summary.json"
    if not run_path.is_file():
        raise FileNotFoundError(f"run.json not found in {output_dir}")
    run = _load_json(run_path)
    summary = _load_json(summary_path) if summary_path.is_file() else {}
    records = _read_csv(output_dir / "records.csv")
    metrics = _read_csv(output_dir / "attention_metrics.csv")
    positions = _read_csv(output_dir / "position_scores.csv")
    pairs = _read_csv(output_dir / "pairs.csv")

    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    time_axis_status = str(
        run.get("time_axis_status")
        or (run.get("time_axis") or {}).get("status")
        or TIME_AXIS_UNAVAILABLE
    )
    time_axis_method = str(run.get("time_axis_method") or TIME_AXIS_UNAVAILABLE)
    encoder_map = encoder_time_map_from_json((run.get("time_axis") or {}).get("encoder"))
    if encoder_map is None and encoder is not None:
        encoder_map = inspect_encoder_time_map(encoder)
        if encoder_map is not None:
            time_axis_status = "ok"
            time_axis_method = encoder_map.method
    fbank = FbankTimeSpec.from_mapping((run.get("time_axis") or {}).get("fbank"))
    padding = run.get("audio_padding_ms") or (run.get("time_axis") or {}).get("padding_ms") or {}
    left_padding_ms = int(padding.get("left") or 0)
    right_padding_ms = int(padding.get("right") or 0)
    plot_dpi = int(((run.get("sink_diagnostics") or {}).get("plot_dpi")) or 160)
    length_bins = [float(v) for v in ((run.get("sink_diagnostics") or {}).get("length_bins") or [0, 100, 200, 400, 800])]
    group_field = str((run.get("sink_diagnostics") or {}).get("group_field") or "condition")
    samples_meta = dict(run.get("samples") or {})

    sample_ids = _unique_preserve([row["sample_id"] for row in records])
    selected_ids = [
        sample_id
        for sample_id in sample_ids
        if _sample_rows(records, sample_id)
        and _truthy(_sample_rows(records, sample_id)[0].get("report_selected"))
    ]
    if not selected_ids:
        selected_ids = [str(item) for item in run.get("report_selected") or []]

    skipped_plots: list[str] = []
    figure_index: list[dict[str, str]] = []
    plt = None
    try:
        plt = _import_matplotlib()
    except RuntimeError as exc:
        skipped_plots.append(str(exc))

    if plt is not None:
        figure_index.extend(
            _draw_group_figures(
                plt,
                figures_dir=figures_dir,
                records=records,
                metrics=metrics,
                group_field=group_field,
                length_bins=length_bins,
                dpi=plot_dpi,
            )
        )
        for sample_id in selected_ids:
            sample_rows = _sample_rows(records, sample_id)
            if not sample_rows:
                continue
            if sample_rows[0].get("status") != "ok":
                continue
            meta = dict(samples_meta.get(sample_id) or {})
            axis = _axis_for_sample(
                sample_id,
                records=sample_rows,
                meta=meta,
                encoder_map=encoder_map,
                fbank=fbank,
                left_padding_ms=left_padding_ms,
                right_padding_ms=right_padding_ms,
                fallback_status=time_axis_status,
            )
            sample_figs, sample_skips = _draw_sample_figures(
                plt,
                output_dir=output_dir,
                figures_dir=figures_dir,
                sample_id=sample_id,
                records=records,
                metrics=metrics,
                positions=positions,
                pairs=pairs,
                meta=meta,
                axis=axis,
                run=run,
                dpi=plot_dpi,
            )
            figure_index.extend(sample_figs)
            skipped_plots.extend(sample_skips)

    html = _build_html(
        run=run,
        summary=summary,
        records=records,
        metrics=metrics,
        pairs=pairs,
        sample_ids=sample_ids,
        selected_ids=selected_ids,
        figure_index=figure_index,
        skipped_plots=skipped_plots,
        time_axis_status=time_axis_status,
        time_axis_method=time_axis_method,
        group_field=group_field,
    )
    _atomic_write_text(output_dir / "report.html", html)

    result = {
        "status": "generated",
        "num_input": int(summary.get("num_input") or len(sample_ids)),
        "num_selected": len(selected_ids),
        "html": "report.html",
        "figures": figure_index,
        "skipped_plots": skipped_plots,
        "time_axis_status": time_axis_status,
        "time_axis_method": time_axis_method,
    }
    if summary_path.is_file():
        summary = dict(summary)
        summary["figures"] = figure_index
        summary["num_report_selected"] = len(selected_ids)
        _atomic_write_json(summary_path, summary)
    return result


def _axis_for_sample(
    sample_id: str,
    *,
    records: Sequence[Mapping[str, str]],
    meta: Mapping[str, Any],
    encoder_map: EncoderTimeMap | None,
    fbank: FbankTimeSpec | None,
    left_padding_ms: int,
    right_padding_ms: int,
    fallback_status: str,
) -> SampleTimeAxis:
    del sample_id
    audio_length = _optional_int(
        meta.get("audio_length") or records[0].get("audio_length")
    ) or 0
    stored_status = str(meta.get("time_axis_status") or "")
    # An empty centers list is a stored unavailable axis, not a missing key.
    # Do not rebuild from the run-level encoder map in that case.
    if stored_status == TIME_AXIS_UNAVAILABLE or "centers_source_sec" in meta:
        payload = dict(meta)
        if stored_status:
            payload["time_axis_status"] = stored_status
        elif fallback_status:
            payload.setdefault("time_axis_status", fallback_status)
        return SampleTimeAxis.from_dict(payload, audio_length=audio_length)
    duration = float(meta.get("source_duration_sec") or 0.0)
    return build_sample_time_axis(
        audio_length=audio_length,
        encoder_map=encoder_map,
        fbank=fbank,
        left_padding_ms=left_padding_ms,
        right_padding_ms=right_padding_ms,
        source_duration_sec=duration,
        num_fbank_frames=_optional_int(meta.get("num_fbank_frames")),
    )


def _draw_group_figures(
    plt,
    *,
    figures_dir: Path,
    records: Sequence[Mapping[str, str]],
    metrics: Sequence[Mapping[str, str]],
    group_field: str,
    length_bins: Sequence[float],
    dpi: int,
) -> list[dict[str, str]]:
    written: list[dict[str, str]] = []
    normal = [
        row
        for row in records
        if row.get("ablation") == "normal" and row.get("status") == "ok"
    ]
    s_audio = _per_sample_region_means(metrics, "audio")
    s_text = _per_sample_region_means(metrics, "text")
    by_group_audio: dict[str, list[float]] = {}
    by_group_text: dict[str, list[float]] = {}
    scatter_x: list[float] = []
    scatter_y: list[float] = []
    scatter_label: list[str] = []
    by_audio_bin: dict[str, list[float]] = {}
    by_text_len: dict[str, list[float]] = {}
    for row in normal:
        sample_id = row["sample_id"]
        if sample_id not in s_audio:
            continue
        group = row.get(group_field) or "unknown"
        by_group_audio.setdefault(group, []).append(s_audio[sample_id])
        if sample_id in s_text:
            by_group_text.setdefault(group, []).append(s_text[sample_id])
        score = _optional_float(row.get("qbyt_score"))
        if score is not None:
            scatter_x.append(score)
            scatter_y.append(s_audio[sample_id])
            scatter_label.append(row.get("label") or "unlabeled")
        audio_len = _optional_float(row.get("audio_length")) or 0.0
        text_len = _optional_float(row.get("text_length")) or 0.0
        by_audio_bin.setdefault(length_bin_label(audio_len, length_bins), []).append(
            s_audio[sample_id]
        )
        by_text_len.setdefault(str(int(text_len)), []).append(s_audio[sample_id])

    def _boxplot(data: dict[str, list[float]], filename: str, ylabel: str, title: str):
        usable = {key: vals for key, vals in data.items() if vals}
        if not usable:
            return
        path = figures_dir / filename
        fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
        labels = list(usable)
        ax.boxplot([usable[key] for key in labels])
        ax.set_xticklabels(labels)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=30)
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        written.append({"kind": "group", "path": _rel(path, figures_dir.parent)})

    _boxplot(
        by_group_audio,
        f"s_audio_by_{_safe_name(group_field)}.png",
        "per-sample mean S_audio",
        f"Observed per-sample mean audio-query sink weight by {group_field}",
    )
    _boxplot(
        by_group_text,
        f"s_text_by_{_safe_name(group_field)}.png",
        "per-sample mean S_text",
        f"Observed per-sample mean text-query sink weight by {group_field}",
    )
    _boxplot(
        by_audio_bin,
        "s_audio_by_audio_length_bin.png",
        "per-sample mean S_audio",
        "Observed per-sample mean S_audio by encoder-frame length bin",
    )
    _boxplot(
        by_text_len,
        "s_audio_by_text_length.png",
        "per-sample mean S_audio",
        "Observed per-sample mean S_audio by text length",
    )
    if scatter_x:
        path = figures_dir / "s_audio_vs_normal_score.png"
        fig, ax = plt.subplots(figsize=(6.5, 5), constrained_layout=True)
        colors = {"1": "#1f77b4", "0": "#d62728", "unlabeled": "#7f7f7f"}
        for label, color, marker in (
            ("1", colors["1"], "o"),
            ("0", colors["0"], "s"),
            ("unlabeled", colors["unlabeled"], "x"),
        ):
            xs = [x for x, lab in zip(scatter_x, scatter_label) if lab == label]
            ys = [y for y, lab in zip(scatter_y, scatter_label) if lab == label]
            if not xs:
                continue
            ax.scatter(xs, ys, c=color, marker=marker, label=f"label={label}", alpha=0.8)
        ax.set_xlabel("normal qbyt_score")
        ax.set_ylabel("per-sample mean S_audio")
        ax.set_title("Observed sink audio attention vs normal score")
        ax.legend()
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        written.append({"kind": "group", "path": _rel(path, figures_dir.parent)})
    return written


def _draw_sample_figures(
    plt,
    *,
    output_dir: Path,
    figures_dir: Path,
    sample_id: str,
    records: Sequence[Mapping[str, str]],
    metrics: Sequence[Mapping[str, str]],
    positions: Sequence[Mapping[str, str]],
    pairs: Sequence[Mapping[str, str]],
    meta: Mapping[str, Any],
    axis: SampleTimeAxis,
    run: Mapping[str, Any],
    dpi: int,
) -> tuple[list[dict[str, str]], list[str]]:
    written: list[dict[str, str]] = []
    skipped: list[str] = []
    meta = dict(meta)
    if not meta.get("readout_spec") and run.get("qbyt_readout"):
        meta["readout_spec"] = json.dumps(
            run.get("qbyt_readout"), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
    internal = str(meta.get("internal_sample_id") or _safe_name(sample_id))
    sample_dir = figures_dir / internal
    sample_dir.mkdir(parents=True, exist_ok=True)
    sample_rows = _sample_rows(records, sample_id)
    normal = next((row for row in sample_rows if row.get("ablation") == "normal"), sample_rows[0])
    traces = _load_sample_traces(output_dir, sample_id, internal, sample_rows)
    if not traces:
        skipped.append(f"{sample_id}: attention/spectrogram traces are absent")
    else:
        normal_trace = traces.get("normal") or next(iter(traces.values()))
        spec_path = _draw_spectrogram(
            plt,
            sample_dir / "spectrogram.png",
            trace=normal_trace,
            axis=axis,
            meta=meta,
            sample_row=normal,
            dpi=dpi,
        )
        if spec_path is None:
            skipped.append(f"{sample_id}: spectrogram skipped (no prepared_waveform)")
        else:
            written.append({"kind": "spectrogram", "sample_id": sample_id, "path": _rel(spec_path, output_dir)})
        heat = _draw_audio_to_sink(
            plt,
            sample_dir / "audio_to_sink.png",
            trace=normal_trace,
            axis=axis,
            dpi=dpi,
        )
        written.append({"kind": "audio_to_sink", "sample_id": sample_id, "path": _rel(heat, output_dir)})
        layer_ids = [int(v) for v in normal_trace["layer_ids"].tolist()]
        head_ids = [int(v) for v in normal_trace["head_ids"].tolist()]
        phonemes = _phonemes_for_sample(normal, normal_trace, meta)
        for layer_i, layer_id in enumerate(layer_ids):
            for head_i, head_id in enumerate(head_ids):
                path = sample_dir / f"text_to_keys_l{layer_id}_h{head_id}.png"
                _draw_text_to_keys(
                    plt,
                    path,
                    trace=normal_trace,
                    layer_index=layer_i,
                    head_index=head_i,
                    phonemes=phonemes,
                    axis=axis,
                    dpi=dpi,
                )
                written.append(
                    {
                        "kind": "text_to_keys",
                        "sample_id": sample_id,
                        "layer": str(layer_id),
                        "head": str(head_id),
                        "path": _rel(path, output_dir),
                    }
                )
        mean_path = sample_dir / "text_to_keys_mean.png"
        _draw_text_to_keys(
            plt,
            mean_path,
            trace=normal_trace,
            layer_index=None,
            head_index=None,
            phonemes=phonemes,
            axis=axis,
            dpi=dpi,
            mean=True,
        )
        written.append(
            {
                "kind": "text_to_keys_mean",
                "sample_id": sample_id,
                "path": _rel(mean_path, output_dir),
            }
        )

    pos_path = sample_dir / "position_logits.png"
    if _draw_position_logits(plt, pos_path, sample_id=sample_id, positions=positions, dpi=dpi):
        written.append({"kind": "position_logits", "sample_id": sample_id, "path": _rel(pos_path, output_dir)})
    score_path = sample_dir / "ablation_scores.png"
    if _draw_ablation_scores(plt, score_path, sample_rows=sample_rows, dpi=dpi):
        written.append({"kind": "ablation_scores", "sample_id": sample_id, "path": _rel(score_path, output_dir)})

    pair_note = _draw_pair_delta(
        plt,
        sample_dir=sample_dir,
        output_dir=output_dir,
        sample_id=sample_id,
        records=records,
        pairs=pairs,
        traces=traces,
        axis=axis,
        dpi=dpi,
        written=written,
        run=run,
    )
    if pair_note:
        skipped.append(pair_note)
    return written, skipped


def _load_sample_traces(
    output_dir: Path,
    sample_id: str,
    internal: str,
    sample_rows: Sequence[Mapping[str, str]],
) -> dict[str, Any]:
    import numpy as np

    traces: dict[str, Any] = {}
    for row in sample_rows:
        ablation = row.get("ablation") or "normal"
        path = None
        rel = row.get("trace_path") or ""
        if rel:
            candidate = output_dir / rel
            if candidate.is_file():
                path = candidate
        if path is None:
            name = f"{internal}__{ablation}.npz"
            for root in (output_dir / "traces", output_dir / ".partial" / "traces"):
                candidate = root / name
                if candidate.is_file():
                    path = candidate
                    break
        if path is None:
            continue
        with np.load(path, allow_pickle=False) as payload:
            traces[ablation] = {key: payload[key] for key in payload.files}
    return traces


def _draw_spectrogram(
    plt,
    path: Path,
    *,
    trace: Mapping[str, Any],
    axis: SampleTimeAxis,
    meta: Mapping[str, Any],
    sample_row: Mapping[str, str],
    dpi: int,
):
    import numpy as np

    waveform = trace.get("prepared_waveform")
    if waveform is None:
        return None
    wave = np.asarray(waveform, dtype=np.float32).reshape(-1)
    sample_rate = int(
        np.asarray(trace.get("prepared_sample_rate", meta.get("model_sample_rate") or 16000)).reshape(-1)[0]
    )
    fig, ax = plt.subplots(figsize=(10, 3.2), constrained_layout=True)
    duration = float(meta.get("source_duration_sec") or (wave.size / max(sample_rate, 1)))
    if np.any(np.abs(wave) > 0):
        nfft = max(16, int(round(sample_rate * 0.025)))
        noverlap = max(0, nfft - int(round(sample_rate * 0.010)))
        ax.specgram(wave, NFFT=nfft, Fs=sample_rate, noverlap=noverlap, cmap="magma")
    else:
        ax.set_xlim(0.0, max(duration, 1.0 / max(sample_rate, 1)))
        ax.set_ylim(0.0, sample_rate / 2.0)
        ax.text(
            0.5,
            0.5,
            "prepared waveform is silent; spans still marked",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=9,
            color="#555555",
        )
    ax.set_xlabel("source time (s), prepared waveform")
    ax.set_ylabel("frequency (Hz)")
    title = _sample_title(sample_row, meta)
    ax.set_title(title, fontsize=8)
    for start, end, color, label in _span_overlays(meta):
        ax.axvspan(start, end, color=color, alpha=0.18, label=label)
    left = float(axis.left_padding_sec or 0.0)
    right = float(axis.right_padding_sec or 0.0)
    if left > 0:
        ax.text(
            0.0,
            1.02,
            f"left padding {left:.3f}s (synthetic, not in this spectrogram)",
            transform=ax.get_xaxis_transform(),
            fontsize=8,
            color="#555555",
        )
    if right > 0:
        ax.text(
            duration,
            1.02,
            f"right padding {right:.3f}s",
            transform=ax.get_xaxis_transform(),
            fontsize=8,
            color="#555555",
            ha="right",
        )
    if axis.status != "ok":
        ax.set_xlabel("prepared-waveform time (s); attention uses encoder_frame_index")
    handles, labels = ax.get_legend_handles_labels()
    if labels:
        uniq = dict(zip(labels, handles))
        ax.legend(uniq.values(), uniq.keys(), loc="upper right", fontsize=8)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def _draw_audio_to_sink(plt, path: Path, *, trace, axis: SampleTimeAxis, dpi: int):
    import numpy as np

    data = np.asarray(trace["audio_to_sink"], dtype=np.float32)
    layer_ids = [int(v) for v in np.asarray(trace["layer_ids"]).tolist()]
    head_ids = [int(v) for v in np.asarray(trace["head_ids"]).tolist()]
    rows = []
    ylabels = []
    for layer_i, layer_id in enumerate(layer_ids):
        for head_i, head_id in enumerate(head_ids):
            rows.append(data[layer_i, head_i])
            ylabels.append(f"L{layer_id} H{head_id}")
    matrix = np.stack(rows, axis=0) if rows else np.zeros((1, 1), dtype=np.float32)
    fig, axes = plt.subplots(2, 1, figsize=(10, 6.5), constrained_layout=True)
    x_coords, xlabel = _x_coords(axis, matrix.shape[1])
    _imshow_attention(axes[0], matrix, x_coords=x_coords, ylabels=ylabels, xlabel=xlabel)
    axes[0].set_title("audio-query to sink [0, 1]")
    _shade_padding(axes[0], axis, x_coords)
    log = np.log10(np.clip(matrix, _LOG_FLOOR, 1.0))
    lo = float(log.min()) if log.size else -6.0
    hi = float(log.max()) if log.size else 0.0
    _imshow_attention(
        axes[1],
        log,
        x_coords=x_coords,
        ylabels=ylabels,
        xlabel=xlabel,
        vmin=lo,
        vmax=hi,
        cmap="magma",
    )
    axes[1].set_title(f"log10 auxiliary view, clim=[{lo:.2f}, {hi:.2f}] (does not replace the [0, 1] plot)")
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def _draw_text_to_keys(
    plt,
    path: Path,
    *,
    trace,
    layer_index: int | None,
    head_index: int | None,
    phonemes: Sequence[str],
    axis: SampleTimeAxis,
    dpi: int,
    mean: bool = False,
):
    import numpy as np

    sink = np.asarray(trace["text_to_sink"], dtype=np.float32)
    audio = np.asarray(trace["text_to_audio"], dtype=np.float32)
    if mean:
        sink_vec = sink.mean(axis=(0, 1))
        audio_mat = audio.mean(axis=(0, 1))
        title = "text-query to sink+audio, mean over layer/head (auxiliary)"
    else:
        sink_vec = sink[layer_index, head_index]
        audio_mat = audio[layer_index, head_index]
        title = f"text-query to sink+audio, layer {layer_index} head {head_index} [0, 1]"
        if layer_index is not None:
            layer_ids = [int(v) for v in np.asarray(trace["layer_ids"]).tolist()]
            head_ids = [int(v) for v in np.asarray(trace["head_ids"]).tolist()]
            title = (
                f"text-query to sink+audio, L{layer_ids[layer_index]} "
                f"H{head_ids[head_index]} [0, 1]"
            )
    matrix = assemble_text_key_heatmap(sink_vec, audio_mat)
    fig, ax = plt.subplots(figsize=(10, max(2.8, 0.35 * matrix.shape[0] + 1.8)), constrained_layout=True)
    im = ax.imshow(matrix, aspect="auto", origin="upper", vmin=0.0, vmax=1.0, cmap="viridis")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ylabels = list(phonemes[: matrix.shape[0]])
    while len(ylabels) < matrix.shape[0]:
        ylabels.append(f"p{len(ylabels)}")
    ax.set_yticks(range(matrix.shape[0]))
    ax.set_yticklabels(ylabels)
    xticks = [0]
    xticklabels = ["sink"]
    audio_len = matrix.shape[1] - 1
    if audio_len > 0:
        step = max(1, audio_len // 8)
        for index in range(step, audio_len, step):
            xticks.append(index + 1)
            if axis.status == "ok" and index < axis.centers_source_sec.size:
                xticklabels.append(f"{axis.centers_source_sec[index]:.3f}s")
            else:
                xticklabels.append(str(index))
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, rotation=30, ha="right")
    xlabel = "sink column, then source time (s)" if axis.status == "ok" else "sink column, then encoder_frame_index"
    ax.set_xlabel(xlabel)
    ax.set_ylabel("phoneme (text query)")
    ax.set_title(title)
    ax.axvline(0.5, color="white", linewidth=1.0, linestyle="--")
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def _draw_position_logits(plt, path: Path, *, sample_id: str, positions, dpi: int) -> bool:
    rows = [row for row in positions if row.get("sample_id") == sample_id]
    if not rows:
        return False
    by_ablation: dict[str, list[tuple[int, str, float, float]]] = {}
    for row in rows:
        pos = _optional_int(row.get("position"))
        logit = _optional_float(row.get("position_logit"))
        if pos is None or logit is None:
            continue
        delta = _optional_float(row.get("delta_position_logit")) or 0.0
        by_ablation.setdefault(row.get("ablation") or "normal", []).append(
            (pos, row.get("phoneme") or "", logit, delta)
        )
    if not by_ablation:
        return False
    has_delta = any(name != "normal" for name in by_ablation)
    nrows = 2 if has_delta else 1
    fig, axes = plt.subplots(nrows, 1, figsize=(8, 3.2 * nrows), constrained_layout=True)
    if nrows == 1:
        axes = [axes]
    normal = sorted(by_ablation.get("normal") or next(iter(by_ablation.values())))
    xs = [item[0] for item in normal]
    labels = [item[1] for item in normal]
    axes[0].plot(xs, [item[2] for item in normal], marker="o", label="normal")
    axes[0].set_xticks(xs)
    axes[0].set_xticklabels(labels, rotation=30, ha="right")
    axes[0].set_ylabel("position_logit")
    axes[0].set_title("EPS position logits")
    axes[0].legend()
    if has_delta:
        for name, items in by_ablation.items():
            if name == "normal":
                continue
            ordered = sorted(items)
            axes[1].plot(
                [item[0] for item in ordered],
                [item[3] for item in ordered],
                marker="o",
                label=name,
            )
        axes[1].axhline(0.0, color="#888888", linewidth=0.8)
        axes[1].set_xticks(xs)
        axes[1].set_xticklabels(labels, rotation=30, ha="right")
        axes[1].set_ylabel("delta_position_logit vs normal")
        axes[1].set_title("Ablation deltas (observed score change, not a causal proof)")
        axes[1].legend(fontsize=8)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return True


def _draw_ablation_scores(plt, path: Path, *, sample_rows, dpi: int) -> bool:
    rows = [row for row in sample_rows if row.get("status") == "ok"]
    xs = []
    scores = []
    for row in rows:
        value = _optional_float(row.get("qbyt_score"))
        if value is None:
            continue
        xs.append(row.get("ablation") or "")
        scores.append(value)
    if not xs:
        return False
    fig, ax = plt.subplots(figsize=(max(6, 0.7 * len(xs) + 2), 3.6), constrained_layout=True)
    ax.bar(range(len(xs)), scores, color="#4c78a8")
    ax.set_xticks(range(len(xs)))
    ax.set_xticklabels(xs, rotation=30, ha="right")
    ax.set_ylabel("qbyt_score")
    ax.set_title("normal / all / single-layer ablation scores")
    thresh = _optional_float(rows[0].get("threshold"))
    if thresh is not None:
        ax.axhline(thresh, color="#d62728", linestyle="--", label=f"threshold={thresh}")
        ax.legend()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return True


def _draw_pair_delta(
    plt,
    *,
    sample_dir: Path,
    output_dir: Path,
    sample_id: str,
    records,
    pairs,
    traces,
    axis: SampleTimeAxis,
    dpi: int,
    written: list[dict[str, str]],
    run: Mapping[str, Any],
) -> str | None:
    import numpy as np

    selected = {
        row["sample_id"]
        for row in records
        if _truthy(row.get("report_selected"))
    }
    partner = None
    comparable = False
    reason = ""
    for row in pairs:
        baseline = row.get("baseline_sample_id") or ""
        variant = row.get("variant_sample_id") or ""
        if sample_id not in {baseline, variant}:
            continue
        other = variant if sample_id == baseline else baseline
        if not other:
            continue
        if other not in selected:
            return (
                f"{sample_id}: pair figures omitted because partner {other} "
                "was not selected (quota cannot fit the pair group)"
            )
        partner = other
        comparable = _truthy(row.get("time_grid_comparable"))
        reason = row.get("pair_reason") or ""
        break
    if partner is None:
        return None
    if not comparable:
        return f"{sample_id}: pair scalar comparison only ({reason or 'time grid not comparable'}); no interpolated delta figure"
    if axis.status != "ok":
        return f"{sample_id}: pair attention delta skipped (time_axis unavailable)"
    other_rows = _sample_rows(records, partner)
    other_internal = _internal_id_for_sample(partner, run=run, sample_rows=other_rows)
    other_traces = _load_sample_traces(output_dir, partner, other_internal, other_rows)
    self_trace = traces.get("normal")
    other_trace = other_traces.get("normal")
    if self_trace is None or other_trace is None:
        return f"{sample_id}: pair attention delta skipped (missing traces)"
    a = np.asarray(self_trace["audio_to_sink"], dtype=np.float32)
    b = np.asarray(other_trace["audio_to_sink"], dtype=np.float32)
    if a.shape != b.shape:
        return f"{sample_id}: pair attention delta skipped (attention shape mismatch)"
    # noisy - clean: variant minus baseline when sample is the variant.
    sample_condition = _sample_rows(records, sample_id)[0].get("condition")
    if sample_condition == "clean":
        delta = b - a
        title = f"noisy-clean audio-to-sink delta ({partner} - {sample_id})"
    else:
        delta = a - b
        title = f"noisy-clean audio-to-sink delta ({sample_id} - {partner})"
    lim = float(max(np.abs(delta).max() if delta.size else 0.0, 1e-6))
    layer_ids = [int(v) for v in np.asarray(self_trace["layer_ids"]).tolist()]
    head_ids = [int(v) for v in np.asarray(self_trace["head_ids"]).tolist()]
    rows = []
    ylabels = []
    for layer_i, layer_id in enumerate(layer_ids):
        for head_i, head_id in enumerate(head_ids):
            rows.append(delta[layer_i, head_i])
            ylabels.append(f"L{layer_id} H{head_id}")
    matrix = np.stack(rows, axis=0)
    path = sample_dir / f"pair_delta_{_safe_name(partner)}.png"
    fig, ax = plt.subplots(figsize=(10, 4.5), constrained_layout=True)
    x_coords, xlabel = _x_coords(axis, matrix.shape[1])
    _imshow_attention(
        ax,
        matrix,
        x_coords=x_coords,
        ylabels=ylabels,
        xlabel=xlabel,
        vmin=-lim,
        vmax=lim,
        cmap="RdBu_r",
    )
    ax.set_title(f"{title}; symmetric clim=[{-lim:.3f}, {lim:.3f}]")
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    written.append({"kind": "pair_delta", "sample_id": sample_id, "path": _rel(path, output_dir)})
    return None


def _imshow_attention(
    ax,
    matrix,
    *,
    x_coords,
    ylabels: Sequence[str],
    xlabel: str,
    vmin: float = 0.0,
    vmax: float = 1.0,
    cmap: str = "viridis",
):
    import numpy as np

    data = np.asarray(matrix, dtype=np.float64)
    if data.ndim == 1:
        data = data[None, :]
    x = np.asarray(x_coords, dtype=np.float64)
    if x.size == 1:
        left, right = float(x[0]) - 0.5, float(x[0]) + 0.5
    else:
        step = float(x[1] - x[0]) if x.size > 1 else 1.0
        left = float(x[0] - step / 2.0)
        right = float(x[-1] + step / 2.0)
    im = ax.imshow(
        data,
        aspect="auto",
        origin="upper",
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        extent=[left, right, data.shape[0] - 0.5, -0.5],
    )
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_yticks(range(len(ylabels)))
    ax.set_yticklabels(ylabels)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("layer x head")


def _shade_padding(ax, axis: SampleTimeAxis, x_coords) -> None:
    if axis.status != "ok" or axis.is_padding.size == 0:
        return
    import numpy as np

    x = np.asarray(x_coords, dtype=np.float64)
    pad = np.asarray(axis.is_padding, dtype=bool)
    if pad.size != x.size:
        return
    step = float(x[1] - x[0]) if x.size > 1 else 0.01
    in_run = False
    start = 0.0
    for index, flag in enumerate(pad):
        if flag and not in_run:
            start = float(x[index] - step / 2.0)
            in_run = True
        elif not flag and in_run:
            ax.axvspan(start, float(x[index] - step / 2.0), color="#bbbbbb", alpha=0.35)
            in_run = False
    if in_run:
        ax.axvspan(start, float(x[-1] + step / 2.0), color="#bbbbbb", alpha=0.35)


def _x_coords(axis: SampleTimeAxis, width: int):
    import numpy as np

    if axis.status == "ok" and axis.centers_source_sec.size == width:
        return axis.centers_source_sec, "source time (s); gray = synthetic padding"
    return np.arange(width, dtype=np.float64), "encoder_frame_index"


def _span_overlays(meta: Mapping[str, Any]) -> list[tuple[float, float, str, str]]:
    overlays = []
    for start, end in meta.get("keyword_spans") or []:
        overlays.append((float(start), float(end), "#2ca02c", "keyword span"))
    for start, end in meta.get("noise_spans") or []:
        overlays.append((float(start), float(end), "#ff7f0e", "noise span"))
    return overlays


def _internal_id_for_sample(
    sample_id: str,
    *,
    run: Mapping[str, Any],
    sample_rows: Sequence[Mapping[str, str]],
) -> str:
    meta = (run.get("samples") or {}).get(sample_id) or {}
    internal = str(meta.get("internal_sample_id") or "").strip()
    if internal:
        return internal
    for row in sample_rows:
        rel = row.get("trace_path") or ""
        if rel:
            return Path(rel).name.split("__")[0]
    return sample_id


def _sample_title(row: Mapping[str, str], meta: Mapping[str, Any]) -> str:
    phonemes = row.get("keyword_phonemes") or " ".join(
        str(item) for item in (meta.get("phonemes") or [])
    )
    readout = str(meta.get("readout_spec") or "")
    line1 = (
        f"{row.get('sample_id', '')} | {row.get('keyword', '')} | {phonemes} | "
        f"condition={row.get('condition', '')} | "
        f"label={row.get('label', '') or 'unlabeled'}"
    )
    line2 = (
        f"score={row.get('qbyt_score', '')} threshold={row.get('threshold', '')}"
    )
    if readout:
        line2 = f"{line2} | {readout}"
    return f"{line1}\n{line2}"


def _phonemes_for_sample(row, trace, meta) -> list[str]:
    if meta.get("phonemes"):
        return [str(item) for item in meta["phonemes"]]
    raw = row.get("keyword_phonemes") or ""
    if raw:
        return raw.split()
    if "phonemes" in trace:
        return [str(item) for item in trace["phonemes"].tolist()]
    return []


def _is_synthetic_fixture(run: Mapping[str, Any], summary: Mapping[str, Any]) -> bool:
    for payload in (run, summary):
        if not isinstance(payload, Mapping):
            continue
        if _truthy(payload.get("synthetic_fixture")):
            return True
        sink = payload.get("sink_diagnostics")
        if isinstance(sink, Mapping) and _truthy(sink.get("synthetic_fixture")):
            return True
    return False


def _build_html(
    *,
    run: Mapping[str, Any],
    summary: Mapping[str, Any],
    records: Sequence[Mapping[str, str]],
    metrics: Sequence[Mapping[str, str]],
    pairs: Sequence[Mapping[str, str]],
    sample_ids: Sequence[str],
    selected_ids: Sequence[str],
    figure_index: Sequence[Mapping[str, str]],
    skipped_plots: Sequence[str],
    time_axis_status: str,
    time_axis_method: str,
    group_field: str,
) -> str:
    run_id = str(run.get("run_id") or summary.get("run_id") or "")
    synthetic = _is_synthetic_fixture(run, summary)
    title = html_escape(
        f"{SYNTHETIC_FIXTURE_BANNER} — QbyT sink diagnostics {run_id}"
        if synthetic
        else f"QbyT sink diagnostics {run_id}"
    )
    num_input = int(summary.get("num_input") or len(sample_ids))
    num_selected = len(selected_ids)
    readout = run.get("qbyt_readout") or {}
    readout_text = html_escape(
        json.dumps(readout, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    )
    group_imgs = [
        item
        for item in figure_index
        if item.get("kind") == "group"
    ]
    by_sample: dict[str, list[dict[str, str]]] = {}
    for item in figure_index:
        sid = item.get("sample_id")
        if sid:
            by_sample.setdefault(sid, []).append(dict(item))

    picker_options = "\n".join(
        f'<option value="{html_escape(sample_id, quote=True)}">{html_escape(sample_id)}</option>'
        for sample_id in selected_ids
    )
    group_img_html = "\n".join(
        f'<p><img src="{html_escape(item["path"], quote=True)}" alt="{html_escape(item["path"], quote=True)}"></p>'
        for item in group_imgs
    )
    skip_html = ""
    if skipped_plots:
        items = "".join(f"<li>{html_escape(item)}</li>" for item in skipped_plots)
        skip_html = f"<h2>Skipped plots</h2><ul>{items}</ul>"

    axis_note = (
        "Time mapping is available. Attention x-axes use nominal encoder-frame "
        "centers in source seconds (model-input time minus left padding). "
        "Padding columns are marked and do not change model key behavior."
        if time_axis_status == "ok"
        else (
            "time_axis_status=unavailable. Spectrogram uses prepared-waveform time; "
            "attention heatmaps use encoder_frame_index. Those axes are independent. "
            "Noise-span region stats and paired time diffs are omitted."
        )
    )
    sample_panels = []
    for sample_index, sample_id in enumerate(selected_ids):
        rows = _sample_rows(records, sample_id)
        if not rows:
            continue
        normal = next((row for row in rows if row.get("ablation") == "normal"), rows[0])
        figs = by_sample.get(sample_id, [])
        text_figs = [item for item in figs if item.get("kind") == "text_to_keys"]
        other_figs = [item for item in figs if item.get("kind") != "text_to_keys"]
        heading = html_escape(
            f"{sample_id} | {normal.get('keyword', '')} | {normal.get('keyword_phonemes', '')} | "
            f"condition={normal.get('condition', '')} label={normal.get('label', '') or 'unlabeled'} | "
            f"score={normal.get('qbyt_score', '')} threshold={normal.get('threshold', '')}"
        )
        img_html = "\n".join(
            f'<p><img src="{html_escape(item["path"], quote=True)}" alt="{html_escape(item.get("kind", "figure"), quote=True)}"></p>'
            for item in other_figs
        )
        layer_heads = sorted(
            {(item.get("layer", ""), item.get("head", "")) for item in text_figs}
        )
        options = "\n".join(
            f'<option value="{html_escape(layer, quote=True)}:{html_escape(head, quote=True)}">'
            f"L{html_escape(layer)} H{html_escape(head)}</option>"
            for layer, head in layer_heads
            if layer != "" and head != ""
        )
        heat_html = []
        for heat_index, item in enumerate(text_figs):
            key = f"{item.get('layer','')}:{item.get('head','')}"
            active_heat = " active" if heat_index == 0 else ""
            heat_html.append(
                f'<img class="text-heatmap{active_heat}" data-sample="{html_escape(sample_id, quote=True)}" '
                f'data-key="{html_escape(key, quote=True)}" '
                f'src="{html_escape(item["path"], quote=True)}" alt="text to sink and audio">'
            )
        panel_active = " active" if sample_index == 0 else ""
        sample_panels.append(
            f'<section class="sample-panel{panel_active}" data-sample="{html_escape(sample_id, quote=True)}">'
            f"<h3>{heading}</h3>"
            f"<p>readout spec: {readout_text}</p>"
            f"{img_html}"
            f"<p>Text-query heatmap (sink is its own first column): "
            f'<select class="head-picker" data-sample="{html_escape(sample_id, quote=True)}">{options}</select></p>'
            f"{''.join(heat_html)}"
            "</section>"
        )

    table_rows = []
    for row in records:
        table_rows.append(
            "<tr>"
            + "".join(f"<td>{html_escape(row.get(key, ''))}</td>" for key in _HTML_TABLE_COLS)
            + "</tr>"
        )
    header = "".join(
        f'<th data-col="{index}">{html_escape(name)}</th>'
        for index, name in enumerate(_HTML_TABLE_COLS)
    )
    pair_rows_html = []
    for row in pairs:
        pair_rows_html.append(
            "<tr>"
            + "".join(
                f"<td>{html_escape(row.get(key, ''))}</td>"
                for key in (
                    "baseline_sample_id",
                    "variant_sample_id",
                    "pair_status",
                    "time_grid_comparable",
                    "normal_score_delta",
                    "pair_reason",
                )
            )
            + "</tr>"
        )

    data_payload = {
        "selected": list(selected_ids),
        "sample_ids": list(sample_ids),
    }
    data_json = _json_for_script(data_payload)
    status = html_escape(str(summary.get("status", "")))
    body_panels = "\n".join(sample_panels) if sample_panels else "<p>No per-sample figures were selected.</p>"
    return "\n".join(
        [
            "<!DOCTYPE html>",
            '<html lang="en">',
            "<head>",
            '<meta charset="utf-8">',
            f"<title>{title}</title>",
            "<style>",
            "body{font-family:sans-serif;margin:1.5rem;color:#111;}",
            "table{border-collapse:collapse;margin:0.5rem 0 1.5rem;font-size:13px;}",
            "th,td{border:1px solid #ccc;padding:4px 8px;vertical-align:top;}",
            "th{cursor:pointer;background:#f4f4f4;}",
            "img{max-width:100%;height:auto;display:block;margin:0.4rem 0;}",
            ".sample-panel{display:none;border-top:1px solid #ddd;padding-top:1rem;}",
            ".sample-panel.active{display:block;}",
            ".text-heatmap{display:none;}",
            ".text-heatmap.active{display:block;}",
            ".synthetic-banner{background:#fff3cd;border:3px solid #b8860b;color:#4a3500;"
            "padding:0.9rem 1rem;font-weight:700;font-size:1.35rem;margin:0 0 0.6rem;}",
            ".synthetic-banner-note{background:#fff8e1;border:1px solid #e0c36a;"
            "padding:0.6rem 1rem;margin:0 0 1.2rem;}",
            "</style>",
            "</head>",
            "<body>",
            *(
                [
                    f'<div class="synthetic-banner" role="status">{html_escape(SYNTHETIC_FIXTURE_BANNER)}</div>',
                    "<p class=\"synthetic-banner-note\">Synthetic fixture. Not a trained-model "
                    "conclusion. Do not treat these scores as evidence that sink learned to "
                    "reject noise.</p>",
                ]
                if synthetic
                else []
            ),
            "<h1>QbyT sink attention diagnostics</h1>",
            f"<p>run_id: {html_escape(run_id)}</p>",
            f"<p>status: {status}</p>",
            "<p>"
            f"input samples: {num_input}, "
            f"success: {int(summary.get('num_success') or 0)}, "
            f"skipped: {int(summary.get('num_skipped') or 0)}, "
            f"unlabeled: {int(summary.get('num_unlabeled') or 0)}, "
            f"selected for per-sample figures: {num_selected}"
            "</p>",
            f"<p>time_axis_status={html_escape(time_axis_status)}; "
            f"time_axis_method={html_escape(time_axis_method)}</p>",
            f"<p>{html_escape(axis_note)}</p>",
            "<p>Automatic text reports observed weights and intervention score "
            "changes only. It does not claim that sink attention above any threshold "
            "is success, and it does not treat correlation as a causal proof.</p>",
            "<h2>Group figures</h2>",
            group_img_html or "<p>No group figures (empty or missing metrics).</p>",
            skip_html,
            "<h2>Per-sample figures</h2>",
            f"<p><label>sample <select id=\"sample-picker\">{picker_options}</select></label></p>",
            body_panels,
            "<h2>Records</h2>",
            "<p>Full scored rows. Figure selection does not drop records.</p>",
            f'<table id="records-table"><thead><tr>{header}</tr></thead><tbody>',
            *table_rows,
            "</tbody></table>",
            "<h2>Pairs</h2>",
            '<table id="pairs-table"><thead><tr>'
            "<th>baseline</th><th>variant</th><th>pair_status</th>"
            "<th>time_grid_comparable</th><th>normal_score_delta</th><th>pair_reason</th>"
            "</tr></thead><tbody>",
            *pair_rows_html,
            "</tbody></table>",
            f'<script type="application/json" id="report-data">{data_json}</script>',
            "<script>",
            _LOCAL_JS,
            "</script>",
            "</body>",
            "</html>",
            "",
        ]
    )


_HTML_TABLE_COLS = [
    "sample_id",
    "keyword",
    "keyword_phonemes",
    "condition",
    "label",
    "ablation",
    "qbyt_score",
    "detected",
    "status",
    "report_selected",
]

_LOCAL_JS = """
(function () {
  var dataEl = document.getElementById("report-data");
  var data = {};
  try { data = JSON.parse(dataEl.textContent || "{}"); } catch (e) { data = {}; }
  function showSample(id) {
    var panels = document.querySelectorAll(".sample-panel");
    for (var i = 0; i < panels.length; i++) {
      if (panels[i].getAttribute("data-sample") === id) panels[i].classList.add("active");
      else panels[i].classList.remove("active");
    }
    var pickers = document.querySelectorAll('.head-picker[data-sample="'+id+'"]');
    for (var j = 0; j < pickers.length; j++) {
      if (pickers[j].value) showHead(id, pickers[j].value);
    }
  }
  function showHead(sample, key) {
    var imgs = document.querySelectorAll('.text-heatmap[data-sample="'+sample+'"]');
    for (var i = 0; i < imgs.length; i++) {
      if (imgs[i].getAttribute("data-key") === key) imgs[i].classList.add("active");
      else imgs[i].classList.remove("active");
    }
  }
  var samplePicker = document.getElementById("sample-picker");
  if (samplePicker) {
    samplePicker.addEventListener("change", function () { showSample(samplePicker.value); });
    if (samplePicker.value) showSample(samplePicker.value);
  }
  var headPickers = document.querySelectorAll(".head-picker");
  for (var k = 0; k < headPickers.length; k++) {
    headPickers[k].addEventListener("change", function (ev) {
      showHead(ev.target.getAttribute("data-sample"), ev.target.value);
    });
    if (headPickers[k].value) showHead(headPickers[k].getAttribute("data-sample"), headPickers[k].value);
  }
  function sortTable(table, col) {
    var tbody = table.tBodies[0];
    if (!tbody) return;
    var rows = [];
    for (var i = 0; i < tbody.rows.length; i++) rows.push(tbody.rows[i]);
    var dir = table.getAttribute("data-sort-dir") === "asc" ? "desc" : "asc";
    table.setAttribute("data-sort-dir", dir);
    rows.sort(function (a, b) {
      var av = a.cells[col] ? a.cells[col].textContent : "";
      var bv = b.cells[col] ? b.cells[col].textContent : "";
      var an = parseFloat(av); var bn = parseFloat(bv);
      var cmp;
      if (!isNaN(an) && !isNaN(bn) && av !== "" && bv !== "") cmp = an - bn;
      else cmp = av.localeCompare(bv);
      return dir === "asc" ? cmp : -cmp;
    });
    for (var r = 0; r < rows.length; r++) tbody.appendChild(rows[r]);
  }
  var tables = document.querySelectorAll("table");
  for (var t = 0; t < tables.length; t++) {
    tables[t].addEventListener("click", function (ev) {
      var th = ev.target;
      if (!th || th.tagName !== "TH") return;
      var col = -1;
      var cells = th.parentNode.cells;
      for (var c = 0; c < cells.length; c++) if (cells[c] === th) col = c;
      if (col >= 0) sortTable(th.parentNode.parentNode.parentNode, col);
    });
  }
})();
""".strip()


def _import_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        return plt
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required for sink-attention figures. "
            "Install with: pip install matplotlib>=3.8"
        ) from exc


def _as_numpy(value):
    import numpy as np

    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _fmt_float(value: float | None) -> str:
    if value is None:
        return ""
    number = float(value)
    if not math.isfinite(number):
        return ""
    return repr(number)


def _json_floats(values) -> list[float]:
    import numpy as np

    array = np.asarray(values).reshape(-1)
    return [float(item) for item in array]


def _json_bools(values) -> list[bool]:
    import numpy as np

    array = np.asarray(values).reshape(-1)
    return [bool(item) for item in array]


def _json_ints(values) -> list[int]:
    import numpy as np

    array = np.asarray(values).reshape(-1)
    return [int(item) for item in array]


def _json_int_pairs(values) -> list[list[int]]:
    import numpy as np

    array = np.asarray(values)
    if array.size == 0:
        return []
    array = array.reshape(-1, 2)
    return [[int(a), int(b)] for a, b in array]


def _json_for_script(payload: Any) -> str:
    text = json.dumps(payload, ensure_ascii=False, allow_nan=False)
    return text.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _atomic_write_json(path: Path, payload: Any) -> None:
    _atomic_write_text(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
    )


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _unique_preserve(items: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered


def _sample_rows(records: Sequence[Mapping[str, str]], sample_id: str) -> list[Mapping[str, str]]:
    return [row for row in records if row.get("sample_id") == sample_id]


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value or "").strip().lower()
    return text in {"1", "true", "yes"}


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _safe_name(value: str) -> str:
    cleaned = _SAFE_NAME.sub("_", str(value)).strip("._")
    return cleaned[:80] or "sample"


def _rel(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _per_sample_region_means(
    metrics: Sequence[Mapping[str, str]], region: str, *, ablation: str = "normal"
) -> dict[str, float]:
    buckets: dict[str, list[float]] = {}
    for row in metrics:
        if row.get("ablation") != ablation or row.get("region") != region:
            continue
        mean = _optional_float(row.get("mean"))
        if mean is None:
            continue
        buckets.setdefault(row["sample_id"], []).append(mean)
    return {key: float(sum(vals) / len(vals)) for key, vals in buckets.items() if vals}


def _trim_bin_edge(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return repr(float(value))
