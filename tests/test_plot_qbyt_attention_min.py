"""Minimal QbyT attention plot: NPZ aggregation, figure output, CLI contract."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.plot_qbyt_attention_min import (
    aggregate_slice,
    plot_min_attention,
)


def _write_trace(
    path: Path,
    *,
    sink: np.ndarray,
    audio: np.ndarray,
    audio_to_sink: np.ndarray,
    layer_ids: list[int] | None = None,
    head_ids: list[int] | None = None,
    phonemes: list[str] | None = None,
    text_length: int | None = None,
    audio_length: int | None = None,
    qbyt_score: float = 0.75,
) -> None:
    sink = np.asarray(sink, dtype=np.float32)
    audio = np.asarray(audio, dtype=np.float32)
    audio_to_sink = np.asarray(audio_to_sink, dtype=np.float32)
    n_layers, n_heads = sink.shape[:2]
    text_len = int(sink.shape[2]) if text_length is None else text_length
    audio_len = int(audio.shape[3]) if audio_length is None else audio_length
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        layer_ids=np.asarray(layer_ids if layer_ids is not None else range(n_layers)),
        head_ids=np.asarray(head_ids if head_ids is not None else range(n_heads)),
        text_to_sink=sink,
        text_to_audio=audio,
        audio_to_sink=audio_to_sink,
        phonemes=np.asarray(phonemes if phonemes is not None else ["AH0", "EY1", "V"], dtype=str),
        text_length=np.int64(text_len),
        audio_length=np.int64(audio_len),
        sink_index=np.int64(text_len),
        qbyt_score=np.float64(qbyt_score),
        position_logits=np.zeros(text_len, dtype=np.float32),
        ablation=np.array(path.stem.split("__")[-1]),
    )


def _synthetic_trace(path: Path, *, ablation: str = "normal") -> None:
    rng = np.random.default_rng(0)
    sink = rng.random((2, 2, 3), dtype=np.float32)
    audio = rng.random((2, 2, 3, 8), dtype=np.float32)
    audio_to_sink = rng.random((2, 2, 8), dtype=np.float32)
    _write_trace(
        path / f"sample_a__{ablation}.npz",
        sink=sink,
        audio=audio,
        audio_to_sink=audio_to_sink,
    )


def test_aggregate_mean_and_max_match_numpy(tmp_path):
    rng = np.random.default_rng(1)
    sink = rng.random((2, 3, 4), dtype=np.float32)
    audio = rng.random((2, 3, 4, 7), dtype=np.float32)
    audio_to_sink = rng.random((2, 3, 7), dtype=np.float32)
    trace = {
        "layer_ids": np.array([0, 1]),
        "head_ids": np.array([0, 1, 2]),
        "text_to_sink": sink,
        "text_to_audio": audio,
        "audio_to_sink": audio_to_sink,
    }

    mean_sink, mean_audio, mean_strip = aggregate_slice(trace, aggregate="mean")
    max_sink, max_audio, max_strip = aggregate_slice(trace, aggregate="max")

    assert np.allclose(mean_sink, sink.mean(axis=(0, 1)))
    assert np.allclose(mean_audio, audio.mean(axis=(0, 1)))
    assert np.allclose(mean_strip, audio_to_sink.mean(axis=(0, 1)))
    assert np.allclose(max_sink, sink.max(axis=(0, 1)))
    assert np.allclose(max_audio, audio.max(axis=(0, 1)))
    assert np.allclose(max_strip, audio_to_sink.max(axis=(0, 1)))


def test_aggregate_layer_head_selects_by_id_not_position():
    rng = np.random.default_rng(2)
    sink = rng.random((2, 2, 3), dtype=np.float32)
    audio = rng.random((2, 2, 3, 6), dtype=np.float32)
    audio_to_sink = rng.random((2, 2, 6), dtype=np.float32)
    trace = {
        "layer_ids": np.array([5, 9]),
        "head_ids": np.array([3, 7]),
        "text_to_sink": sink,
        "text_to_audio": audio,
        "audio_to_sink": audio_to_sink,
    }

    selected_sink, selected_audio, selected_strip = aggregate_slice(
        trace, layer_id=9, head_id=7, aggregate="mean"
    )

    assert np.allclose(selected_sink, sink[1, 1])
    assert np.allclose(selected_audio, audio[1, 1])
    assert np.allclose(selected_strip, audio_to_sink[1, 1])


def test_aggregate_requires_layer_and_head_together():
    trace = {
        "layer_ids": np.array([0]),
        "head_ids": np.array([0]),
        "text_to_sink": np.zeros((1, 1, 2), dtype=np.float32),
        "text_to_audio": np.zeros((1, 1, 2, 4), dtype=np.float32),
        "audio_to_sink": np.zeros((1, 1, 4), dtype=np.float32),
    }
    with pytest.raises(ValueError):
        aggregate_slice(trace, layer_id=0, aggregate="mean")


def test_plots_png_per_sample_from_run_directory(tmp_path):
    pytest.importorskip("matplotlib")
    run_dir = tmp_path / "run"
    _synthetic_trace(run_dir / "traces")
    (run_dir / "run.json").write_text(
        json.dumps(
            {
                "samples": {
                    "row-1": {
                        "internal_sample_id": "sample_a",
                        "time_axis_status": "ok",
                        "centers_source_sec": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    summary = plot_min_attention(run_dir, dpi=72)

    assert summary["status"] == "generated"
    assert summary["num_plotted"] == 1
    assert summary["num_skipped"] == 0
    assert summary["num_time_axis_fallback"] == 0
    figure = Path(summary["figures"][0]["path"])
    assert figure == run_dir / "min_attention" / "sample_a.png"
    assert figure.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_ablation_suffix_selects_only_matching_traces(tmp_path):
    pytest.importorskip("matplotlib")
    trace_dir = tmp_path / "traces"
    _synthetic_trace(trace_dir, ablation="normal")
    _synthetic_trace(trace_dir, ablation="block_sink_all")

    summary = plot_min_attention(tmp_path, dpi=72)

    assert summary["num_plotted"] == 1
    assert Path(summary["figures"][0]["path"]).name == "sample_a.png"


def test_layer_head_figure_name_and_selection(tmp_path):
    pytest.importorskip("matplotlib")
    trace_dir = tmp_path / "traces"
    _synthetic_trace(trace_dir)

    summary = plot_min_attention(trace_dir, layer_id=1, head_id=0, dpi=72)

    assert summary["aggregate"] == "l1h0"
    assert Path(summary["figures"][0]["path"]).name == "sample_a_l1_h0.png"


def test_unknown_layer_id_is_a_hard_error(tmp_path):
    pytest.importorskip("matplotlib")
    trace_dir = tmp_path / "traces"
    _synthetic_trace(trace_dir)

    with pytest.raises(SystemExit):
        plot_min_attention(trace_dir, layer_id=99, head_id=0, dpi=72)


def test_missing_time_axis_counts_fallback(tmp_path):
    pytest.importorskip("matplotlib")
    trace_dir = tmp_path / "traces"
    _synthetic_trace(trace_dir)

    summary = plot_min_attention(trace_dir, dpi=72)

    assert summary["num_plotted"] == 1
    assert summary["num_time_axis_fallback"] == 1


def test_stored_centers_length_mismatch_still_counts_fallback(tmp_path):
    pytest.importorskip("matplotlib")
    run_dir = tmp_path / "run"
    _synthetic_trace(run_dir / "traces")
    (run_dir / "run.json").write_text(
        json.dumps(
            {
                "samples": {
                    "row-1": {
                        "internal_sample_id": "sample_a",
                        "time_axis_status": "ok",
                        "centers_source_sec": [0.0, 0.1, 0.2],
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    summary = plot_min_attention(run_dir, dpi=72)

    assert summary["num_time_axis_fallback"] == 1


def test_max_aggregate_changes_figure_name(tmp_path):
    pytest.importorskip("matplotlib")
    trace_dir = tmp_path / "traces"
    _synthetic_trace(trace_dir)

    summary = plot_min_attention(trace_dir, aggregate="max", dpi=72)

    assert Path(summary["figures"][0]["path"]).name == "sample_a_max.png"


def test_missing_source_and_empty_selection_raise(tmp_path):
    with pytest.raises(SystemExit):
        plot_min_attention(tmp_path / "does_not_exist")

    trace_dir = tmp_path / "traces"
    _synthetic_trace(trace_dir, ablation="block_sink_all")
    with pytest.raises(SystemExit):
        plot_min_attention(trace_dir, ablation="normal")
