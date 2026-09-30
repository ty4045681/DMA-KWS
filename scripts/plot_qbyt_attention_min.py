#!/usr/bin/env python3
"""Minimal per-sample QbyT pooling attention figure from sink-diagnostics traces.

The script never reruns inference and never loads the model. It reads the
``text_to_sink`` / ``text_to_audio`` / ``audio_to_sink`` slices already stored
in the ``traces/*__<ablation>.npz`` files written by ``diagnose_qbyt_sink.py``
and renders one two-panel PNG per sample:

- top: phoneme (text query) x [sink column, audio keys] heatmap
- bottom: audio-query to sink attention, aligned with the heatmap audio columns

When ``run.json`` sits beside the traces directory the x-axis is a source-second
axis built from the stored nominal time axis. Without it the axis falls back to
encoder frame indices.

Examples:

    python3 scripts/plot_qbyt_attention_min.py /path/to/diagnose_run

    python3 scripts/plot_qbyt_attention_min.py /path/to/run/traces/a__normal.npz \
      --layer 0 --head 0 --output-dir /path/to/plots
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

from dma_kws.inference.qbyt_attention_report import (
    SampleTimeAxis,
    _import_matplotlib,
    _shade_padding,
    _span_overlays,
    _x_coords,
    assemble_text_key_heatmap,
)


def _resolve_trace_paths(source: Path, ablation: str) -> list[Path]:
    source = source.expanduser()
    if source.is_file():
        candidates = [source]
    elif source.is_dir():
        trace_dir = source / "traces" if (source / "traces").is_dir() else source
        candidates = sorted(trace_dir.glob("*.npz"))
    else:
        raise SystemExit(f"trace source not found: {source}")
    suffix = f"__{ablation}.npz"
    selected = [path for path in candidates if path.name.endswith(suffix)]
    if not selected:
        raise SystemExit(
            f"no *{suffix} traces found under {source} "
            f"(looked at {len(candidates)} npz file(s))"
        )
    return selected


def _resolve_run_json(source: Path) -> Path | None:
    source = source.expanduser()
    if source.is_file():
        candidates = [source.parent / "run.json", source.parent.parent / "run.json"]
    else:
        candidates = [source / "run.json", source.parent / "run.json"]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _load_json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"failed to read JSON from {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise SystemExit(f"expected a JSON object in {path}")
    return value


def _sample_meta_by_internal(run_payload: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    index: dict[str, Mapping[str, Any]] = {}
    for meta in (run_payload.get("samples") or {}).values():
        if not isinstance(meta, Mapping):
            continue
        internal = str(meta.get("internal_sample_id") or "").strip()
        if internal:
            index[internal] = meta
    return index


def _internal_sample_id(path: Path, ablation: str) -> str:
    name = path.name
    suffix = f"__{ablation}.npz"
    return name[: -len(suffix)] if name.endswith(suffix) else path.stem


def _load_trace(path: Path) -> dict[str, Any]:
    import numpy as np

    with np.load(path) as handle:
        trace = {name: handle[name] for name in handle.files}
    for key in ("layer_ids", "head_ids", "text_to_sink", "text_to_audio", "audio_to_sink"):
        if key not in trace:
            raise SystemExit(f"{path} is missing trace key {key!r}")
    if np.asarray(trace["text_to_audio"]).ndim != 4:
        raise SystemExit(
            f"{path} text_to_audio must be [layers, heads, text, audio], "
            f"got shape {tuple(np.asarray(trace['text_to_audio']).shape)}"
        )
    return trace


def _scalar(value: Any) -> float:
    import numpy as np

    return float(np.asarray(value).reshape(-1)[0])


def _length(value: Any) -> int:
    import numpy as np

    return int(np.asarray(value).reshape(-1)[0])


def aggregate_slice(
    trace: Mapping[str, Any],
    *,
    layer_id: int | None = None,
    head_id: int | None = None,
    aggregate: str = "mean",
) -> tuple[Any, Any, Any] | None:
    """Return (text->sink, text->audio, audio->sink) for one sample.

    Without ``layer_id``/``head_id`` the slices are reduced over the captured
    layer x head grid with ``mean`` or ``max``. With both, the single head is
    selected instead. Returns ``None`` when the trace captured no layers.
    """

    import numpy as np

    sink = np.asarray(trace["text_to_sink"], dtype=np.float32)
    audio = np.asarray(trace["text_to_audio"], dtype=np.float32)
    audio_to_sink = np.asarray(trace["audio_to_sink"], dtype=np.float32)
    if sink.size == 0 or audio.size == 0:
        return None
    if layer_id is not None or head_id is not None:
        if layer_id is None or head_id is None:
            raise ValueError("--layer and --head must be used together")
        layer_ids = [int(item) for item in np.asarray(trace["layer_ids"]).tolist()]
        head_ids = [int(item) for item in np.asarray(trace["head_ids"]).tolist()]
        if layer_id not in layer_ids:
            raise ValueError(f"layer {layer_id} is not in captured layers {layer_ids}")
        if head_id not in head_ids:
            raise ValueError(f"head {head_id} is not in captured heads {head_ids}")
        layer_index = layer_ids.index(layer_id)
        head_index = head_ids.index(head_id)
        return (
            sink[layer_index, head_index],
            audio[layer_index, head_index],
            audio_to_sink[layer_index, head_index],
        )
    reducer = np.mean if aggregate == "mean" else np.max
    return (
        reducer(sink, axis=(0, 1)),
        reducer(audio, axis=(0, 1)),
        reducer(audio_to_sink, axis=(0, 1)),
    )


def _phoneme_labels(trace: Mapping[str, Any], text_length: int) -> list[str]:
    raw = trace.get("phonemes")
    labels = [str(item) for item in raw.tolist()] if raw is not None else []
    labels = labels[:text_length]
    while len(labels) < text_length:
        labels.append(f"p{len(labels)}")
    return labels


def _heatmap_columns(axis: SampleTimeAxis, audio_length: int):
    import numpy as np

    coords, xlabel = _x_coords(axis, audio_length)
    coords = np.asarray(coords, dtype=np.float64)
    step = float(coords[1] - coords[0]) if coords.size > 1 else 1.0
    sink = float(coords[0]) - step if coords.size else 0.0
    return np.concatenate([[sink], coords]), step, xlabel


def _draw_time_overlays(
    ax, *, axis: SampleTimeAxis, meta: Mapping[str, Any], coords
) -> None:
    if not _uses_seconds(axis, len(coords)):
        return
    for start, end, color, _label in _span_overlays(meta):
        ax.axvspan(float(start), float(end), color=color, alpha=0.15)


def _uses_seconds(axis: SampleTimeAxis, width: int) -> bool:
    return axis.status == "ok" and axis.centers_source_sec.size == width


def _figure_name(
    internal_sample_id: str,
    *,
    layer_id: int | None,
    head_id: int | None,
    aggregate: str,
) -> str:
    if layer_id is not None:
        return f"{internal_sample_id}_l{layer_id}_h{head_id}.png"
    if aggregate != "mean":
        return f"{internal_sample_id}_{aggregate}.png"
    return f"{internal_sample_id}.png"


def _draw_sample_figure(
    plt,
    path: Path,
    *,
    internal_sample_id: str,
    trace: Mapping[str, Any],
    meta: Mapping[str, Any],
    axis: SampleTimeAxis,
    layer_id: int | None,
    head_id: int | None,
    aggregate: str,
    dpi: int,
) -> None:
    text_length = _length(trace["text_length"])
    audio_length = _length(trace["audio_length"])
    slices = aggregate_slice(
        trace, layer_id=layer_id, head_id=head_id, aggregate=aggregate
    )
    if slices is None:
        raise ValueError("trace captured no layers")
    sink_vec, audio_mat, audio_to_sink = slices
    matrix = assemble_text_key_heatmap(sink_vec, audio_mat)
    labels = _phoneme_labels(trace, text_length)
    columns, step, xlabel = _heatmap_columns(axis, audio_length)
    coords = columns[1:]

    fig, (ax_top, ax_bottom) = plt.subplots(
        2,
        1,
        sharex=True,
        figsize=(10, max(4.0, 0.35 * matrix.shape[0] + 3.4)),
        gridspec_kw={"height_ratios": [3, 1]},
        constrained_layout=True,
    )
    extent = [
        float(columns[0] - step / 2.0),
        float(coords[-1] + step / 2.0),
        matrix.shape[0] - 0.5,
        -0.5,
    ]
    im = ax_top.imshow(
        matrix,
        aspect="auto",
        origin="upper",
        vmin=0.0,
        vmax=1.0,
        cmap="viridis",
        extent=extent,
    )
    fig.colorbar(im, ax=ax_top, fraction=0.046, pad=0.04)
    ax_top.set_yticks(range(matrix.shape[0]))
    ax_top.set_yticklabels(labels)
    ax_top.set_ylabel("phoneme (text query)")
    ax_top.axvline(float(columns[0] + step / 2.0), color="white", linestyle="--", linewidth=1.0)
    score = _scalar(trace["qbyt_score"]) if "qbyt_score" in trace else float("nan")
    label = _aggregate_label(
        layer_id=layer_id, head_id=head_id, aggregate=aggregate
    )
    ax_top.set_title(
        f"{internal_sample_id} | qbyt_score={score:.4f} | "
        f"ablation={meta.get('ablation', 'normal')} | {label}"
    )
    _shade_padding(ax_top, axis, coords)
    _draw_time_overlays(ax_top, axis=axis, meta=meta, coords=coords)

    ax_bottom.fill_between(coords, 0.0, audio_to_sink, step="mid", color="#1f77b4", alpha=0.8)
    ax_bottom.set_ylim(bottom=0.0)
    ax_bottom.set_ylabel("audio -> sink")
    ax_bottom.set_xlabel(xlabel)
    _shade_padding(ax_bottom, axis, coords)
    _draw_time_overlays(ax_bottom, axis=axis, meta=meta, coords=coords)

    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _aggregate_label(
    *, layer_id: int | None, head_id: int | None, aggregate: str
) -> str:
    return f"l{layer_id}h{head_id}" if layer_id is not None else aggregate


def plot_min_attention(
    source: Path,
    *,
    ablation: str = "normal",
    aggregate: str = "mean",
    layer_id: int | None = None,
    head_id: int | None = None,
    output_dir: Path | None = None,
    dpi: int = 160,
    sample_filter: str | None = None,
    limit: int = 0,
) -> dict[str, Any]:
    """Render one minimal attention PNG per matching trace and return a summary."""

    if layer_id is None and head_id is not None:
        raise SystemExit("--head requires --layer (both select one layer x head cell)")
    traces = _resolve_trace_paths(source, ablation)
    if sample_filter:
        traces = [path for path in traces if sample_filter in path.name]
        if not traces:
            raise SystemExit(f"no traces match --sample {sample_filter!r}")
    if limit > 0:
        traces = traces[:limit]

    run_json = _resolve_run_json(source)
    run_payload = _load_json(run_json) if run_json is not None else {}
    meta_by_internal = _sample_meta_by_internal(run_payload)

    if output_dir is not None:
        destination = output_dir.expanduser().resolve()
    else:
        destination = (traces[0].parent.parent / "min_attention").resolve()
    destination.mkdir(parents=True, exist_ok=True)

    aggregate_label = _aggregate_label(
        layer_id=layer_id, head_id=head_id, aggregate=aggregate
    )
    if layer_id is not None:
        try:
            aggregate_slice(
                _load_trace(traces[0]),
                layer_id=layer_id,
                head_id=head_id,
                aggregate=aggregate,
            )
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
    try:
        plt = _import_matplotlib()
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc

    figures: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    fallback = 0
    for trace_path in traces:
        internal = _internal_sample_id(trace_path, ablation)
        meta = dict(meta_by_internal.get(internal) or {})
        try:
            trace = _load_trace(trace_path)
            if _length(trace["audio_length"]) < 1 or _length(trace["text_length"]) < 1:
                skipped.append({"sample_id": internal, "reason": "empty trace"})
                continue
            axis = SampleTimeAxis.from_dict(
                meta, audio_length=_length(trace["audio_length"])
            )
            if not _uses_seconds(axis, _length(trace["audio_length"])):
                fallback += 1
            meta.setdefault("ablation", ablation)
            name = _figure_name(
                internal, layer_id=layer_id, head_id=head_id, aggregate=aggregate
            )
            figure_path = destination / name
            _draw_sample_figure(
                plt,
                figure_path,
                internal_sample_id=internal,
                trace=trace,
                meta=meta,
                axis=axis,
                layer_id=layer_id,
                head_id=head_id,
                aggregate=aggregate,
                dpi=dpi,
            )
        except ValueError as exc:
            skipped.append({"sample_id": internal, "reason": str(exc)})
            continue
        figures.append(
            {
                "sample_id": internal,
                "path": str(figure_path),
                "aggregate": aggregate,
                "layer": layer_id,
                "head": head_id,
            }
        )

    return {
        "status": "generated",
        "num_plotted": len(figures),
        "num_skipped": len(skipped),
        "num_time_axis_fallback": fallback,
        "ablation": ablation,
        "aggregate": aggregate_label,
        "output_dir": str(destination),
        "figures": figures,
        "skipped": skipped,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Render a minimal phoneme x [sink, audio] attention heatmap per sample "
            "from diagnose_qbyt_sink.py traces. Does not rerun inference."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "traces",
        type=Path,
        help="diagnose run directory, its traces/ directory, or one __<ablation>.npz",
    )
    parser.add_argument(
        "--ablation",
        default="normal",
        help="ablation condition to plot (matches the npz filename suffix)",
    )
    parser.add_argument(
        "--aggregate",
        choices=("mean", "max"),
        default="mean",
        help="reduction over the captured layer x head grid",
    )
    parser.add_argument(
        "--layer",
        type=int,
        help="captured layer id to select instead of aggregating (needs --head)",
    )
    parser.add_argument(
        "--head",
        type=int,
        help="captured head id to select instead of aggregating (needs --layer)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="PNG directory; defaults to <run>/min_attention",
    )
    parser.add_argument(
        "--sample",
        help="only plot traces whose filename contains this substring",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="cap the number of plotted samples; 0 plots every match",
    )
    parser.add_argument("--dpi", type=int, default=160, help="PNG resolution")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.dpi <= 0:
        raise SystemExit("--dpi must be positive")
    if args.limit < 0:
        raise SystemExit("--limit must be >= 0")
    summary = plot_min_attention(
        args.traces,
        ablation=args.ablation,
        aggregate=args.aggregate,
        layer_id=args.layer,
        head_id=args.head,
        output_dir=args.output_dir,
        dpi=args.dpi,
        sample_filter=args.sample,
        limit=args.limit,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main(sys.argv[1:])
