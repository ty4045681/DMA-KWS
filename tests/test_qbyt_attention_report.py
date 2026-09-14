"""T15: time-axis mapping, region stats, local HTML/PNG sink-attention report."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from dma_kws.inference.qbyt_attention_report import (
    FbankTimeSpec,
    SYNTHETIC_FIXTURE_BANNER,
    SampleTimeAxis,
    assemble_text_key_heatmap,
    build_sample_time_axis,
    fbank_frame_center_sec,
    inspect_encoder_time_map,
    length_bin_label,
    noise_region_masks,
    pair_time_grids_comparable,
    region_stats,
    render_sink_attention_report,
    _axis_for_sample,
    _sample_title,
)


pytest_plugins = ["test_diagnose_qbyt_sink"]

REPO_ROOT = Path(__file__).resolve().parents[1]
RECORDS_FIELDS = [
    "run_id",
    "sample_id",
    "manifest_record_number",
    "audio_path",
    "keyword",
    "keyword_phonemes",
    "query_id",
    "condition",
    "pair_id",
    "label",
    "ablation",
    "blocked_layers",
    "qbyt_raw_logit",
    "qbyt_score",
    "threshold",
    "detected",
    "delta_raw_logit",
    "delta_qbyt_score",
    "text_length",
    "audio_length",
    "status",
    "skip_reason",
    "trace_path",
    "report_selected",
]
METRICS_FIELDS = [
    "run_id",
    "sample_id",
    "ablation",
    "layer",
    "head",
    "region",
    "mean",
    "min",
    "max",
    "query_count",
    "key_count",
    "row_sum_error",
]
POSITION_FIELDS = [
    "run_id",
    "sample_id",
    "ablation",
    "position",
    "phoneme",
    "position_logit",
    "delta_position_logit",
]
PAIR_FIELDS = [
    "baseline_sample_id",
    "variant_sample_id",
    "match_key",
    "pair_status",
    "normal_score_delta",
    "time_grid_comparable",
    "pair_reason",
]


class _IdentityEncoder:
    def output_frames(self, num_input_frames: int) -> int:
        return int(num_input_frames)


class _WenetShapedEncoder:
    def __init__(self, subsampling_rate: int = 4, right_context: int = 6) -> None:
        self.embed = type(
            "Embed",
            (),
            {"subsampling_rate": subsampling_rate, "right_context": right_context},
        )()


class _BareEncoder:
    pass


def _fbank(*, snip_edges: bool = True) -> FbankTimeSpec:
    return FbankTimeSpec(
        frame_length_ms=25.0,
        frame_shift_ms=10.0,
        snip_edges=snip_edges,
        model_sample_rate=16000,
        backend="torchaudio_kaldi",
    )


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _npz_trace(
    path: Path,
    *,
    audio_length: int,
    text_length: int = 2,
    layers: tuple[int, ...] = (0, 1),
    heads: tuple[int, ...] = (0, 1),
    waveform: np.ndarray | None = None,
    sample_rate: int = 16000,
    phonemes: tuple[str, ...] = ("HH", "EY1"),
) -> None:
    n_l, n_h = len(layers), len(heads)
    audio_to_sink = np.full((n_l, n_h, audio_length), 0.25, dtype=np.float32)
    text_to_sink = np.full((n_l, n_h, text_length), 0.2, dtype=np.float32)
    text_to_audio = np.full((n_l, n_h, text_length, audio_length), 0.1, dtype=np.float32)
    payload = {
        "layer_ids": np.asarray(layers, dtype=np.int64),
        "head_ids": np.asarray(heads, dtype=np.int64),
        "audio_to_sink": audio_to_sink,
        "text_to_sink": text_to_sink,
        "text_to_audio": text_to_audio,
        "position_logits": np.linspace(-0.2, 0.3, text_length, dtype=np.float32),
        "delta_position_logits": np.zeros(text_length, dtype=np.float32),
        "raw_logit": np.asarray(0.4, dtype=np.float32),
        "qbyt_score": np.asarray(0.6, dtype=np.float32),
        "text_length": np.asarray(text_length, dtype=np.int64),
        "audio_length": np.asarray(audio_length, dtype=np.int64),
        "sink_index": np.asarray(text_length, dtype=np.int64),
        "row_sum_max_error": np.asarray(0.0, dtype=np.float32),
        "padding_mass_max": np.asarray(0.0, dtype=np.float32),
        "phonemes": np.asarray(phonemes, dtype=str),
        "blocked_layers": np.asarray([], dtype=np.int64),
        "ablation": np.asarray(["normal"], dtype=str),
    }
    if waveform is not None:
        payload["prepared_waveform"] = np.asarray(waveform, dtype=np.float32)
        payload["prepared_sample_rate"] = np.asarray(sample_rate, dtype=np.int64)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **payload)


def _base_run_payload(**overrides) -> dict:
    payload = {
        "schema_version": 1,
        "run_id": "run_test",
        "created_at": "2026-09-15T00:00:00+00:00",
        "qbyt_readout": {
            "version": 4,
            "mode": "eps_softmin",
            "temperature": 1.0,
            "sink_token": True,
            "text_position": "learned",
            "audio_position": "relative_bias",
        },
        "stream": "backend=conformer chunking=off (full context)",
        "audio_padding_ms": {"left": 0, "right": 0},
        "threshold": 0.5,
        "sink_diagnostics": {
            "max_report_samples": 40,
            "plot_dpi": 72,
            "length_bins": [0, 100, 200, 400, 800],
            "group_field": "condition",
            "save_traces": True,
        },
        "time_axis_method": "kaldi_fbank_center+identity_output_frames",
        "time_axis_status": "ok",
        "time_axis": {
            "status": "ok",
            "method": "kaldi_fbank_center+identity_output_frames",
            "fbank": {
                "frame_length_ms": 25.0,
                "frame_shift_ms": 10.0,
                "snip_edges": True,
                "backend": "torchaudio_kaldi",
            },
            "model_sample_rate": 16000,
            "padding_ms": {"left": 0, "right": 0},
            "encoder": {
                "method": "kaldi_fbank_center+identity_output_frames",
                "kind": "identity",
            },
        },
        "samples": {},
        "report_selected": [],
        "queries": [],
    }
    payload.update(overrides)
    return payload


def _record(
    *,
    sample_id: str,
    keyword: str = "hey eva",
    phonemes: str = "HH EY1",
    condition: str = "clean",
    label: str = "1",
    ablation: str = "normal",
    score: str = "0.6",
    detected: str = "true",
    text_length: str = "2",
    audio_length: str = "20",
    status: str = "ok",
    report_selected: str = "true",
    trace_path: str = "",
    pair_id: str = "",
    audio_path: str = "/tmp/clip.wav",
) -> dict[str, str]:
    return {
        "run_id": "run_test",
        "sample_id": sample_id,
        "manifest_record_number": "2",
        "audio_path": audio_path,
        "keyword": keyword,
        "keyword_phonemes": phonemes,
        "query_id": "[1,2]",
        "condition": condition,
        "pair_id": pair_id,
        "label": label,
        "ablation": ablation,
        "blocked_layers": "[]",
        "qbyt_raw_logit": "0.4",
        "qbyt_score": score,
        "threshold": "0.5",
        "detected": detected,
        "delta_raw_logit": "0.0",
        "delta_qbyt_score": "0.0",
        "text_length": text_length,
        "audio_length": audio_length,
        "status": status,
        "skip_reason": "",
        "trace_path": trace_path,
        "report_selected": report_selected,
    }


def test_fbank_centers_use_kaldi_window_not_duration_over_t():
    first = fbank_frame_center_sec(
        0, frame_length_ms=25.0, frame_shift_ms=10.0, snip_edges=True
    )
    tenth = fbank_frame_center_sec(
        9, frame_length_ms=25.0, frame_shift_ms=10.0, snip_edges=True
    )
    assert first == pytest.approx(0.0125)
    assert tenth == pytest.approx(0.1025)
    snip_false = fbank_frame_center_sec(
        0, frame_length_ms=25.0, frame_shift_ms=10.0, snip_edges=False
    )
    assert snip_false == pytest.approx(0.0)


def test_identity_encoder_time_map_subtracts_left_padding_and_marks_it():
    time_map = inspect_encoder_time_map(_IdentityEncoder())
    assert time_map is not None
    axis = build_sample_time_axis(
        audio_length=20,
        encoder_map=time_map,
        fbank=_fbank(),
        left_padding_ms=160,
        right_padding_ms=0,
        source_duration_sec=1.0,
    )
    assert axis.status == "ok"
    fake = np.arange(20, dtype=np.float64) * (1.0 / 20.0)
    assert not np.allclose(axis.centers_source_sec, fake)
    assert axis.centers_source_sec[0] == pytest.approx(0.0125 - 0.160)
    assert bool(axis.is_padding[0]) is True
    assert bool(axis.is_left_padding[0]) is True
    in_source = np.where(~axis.is_padding)[0]
    assert in_source.size
    assert axis.centers_source_sec[in_source[0]] >= 0.0


def test_wenet_subsampling_covers_frontend_offset_and_right_crop():
    time_map = inspect_encoder_time_map(_WenetShapedEncoder())
    assert time_map is not None
    assert time_map.kind == "wenet"
    axis = build_sample_time_axis(
        audio_length=3,
        encoder_map=time_map,
        fbank=_fbank(),
        left_padding_ms=0,
        right_padding_ms=0,
        source_duration_sec=0.16,
        num_fbank_frames=15,
    )
    assert axis.status == "ok"
    first, last = axis.fbank_support[0]
    assert int(first) == 0
    assert int(last) == 6
    # Right-edge encoder frame owns the leftover fbank frames, not a uniform tile.
    assert int(axis.fbank_support[-1, 1]) == 14
    assert axis.centers_source_sec[0] != pytest.approx(0.0)


def test_unverified_encoder_is_unavailable_and_keeps_frame_axis():
    assert inspect_encoder_time_map(_BareEncoder()) is None
    axis = build_sample_time_axis(
        audio_length=8,
        encoder_map=None,
        fbank=_fbank(),
        left_padding_ms=0,
        right_padding_ms=0,
        source_duration_sec=1.0,
    )
    assert axis.status == "unavailable"
    assert axis.method == "unavailable"
    assert axis.centers_source_sec.shape == (0,)
    assert axis.encoder_frame_index.tolist() == list(range(8))


def test_noise_span_union_and_empty_interval_is_missing_not_zero():
    time_map = inspect_encoder_time_map(_IdentityEncoder())
    axis = build_sample_time_axis(
        audio_length=20,
        encoder_map=time_map,
        fbank=_fbank(),
        left_padding_ms=0,
        right_padding_ms=0,
        source_duration_sec=0.3,
    )
    values = np.linspace(0.1, 0.9, 20)
    noise, outside, intervals = noise_region_masks(
        axis,
        noise_spans=((0.05, 0.12), (0.08, 0.15), (0.0, 0.001)),
    )
    stats_noise = region_stats(values, noise)
    stats_outside = region_stats(values, outside)
    assert stats_noise["query_count"] > 0
    assert stats_outside["query_count"] > 0
    # Overlapping [0.05,0.12) and [0.08,0.15) must not double-count; end is exclusive.
    assert int(noise.sum()) == int(
        ((axis.centers_source_sec >= 0.05) & (axis.centers_source_sec < 0.15) & ~axis.is_padding).sum()
    )
    empty = next(item for item in intervals if item["end"] == 0.001)
    assert empty["valid_frame_count"] == 0
    empty_stats = region_stats(values, np.zeros(20, dtype=bool))
    assert empty_stats["query_count"] == 0
    assert empty_stats["mean"] is None
    assert "clean_speech" not in json.dumps(intervals)


def test_pair_time_grid_rejects_length_mismatch_without_interpolation():
    time_map = inspect_encoder_time_map(_IdentityEncoder())
    a = build_sample_time_axis(
        audio_length=20,
        encoder_map=time_map,
        fbank=_fbank(),
        left_padding_ms=0,
        right_padding_ms=0,
        source_duration_sec=0.3,
    )
    b = build_sample_time_axis(
        audio_length=24,
        encoder_map=time_map,
        fbank=_fbank(),
        left_padding_ms=0,
        right_padding_ms=0,
        source_duration_sec=0.3,
    )
    ok, reason = pair_time_grids_comparable(
        a,
        b,
        token_ids_a=(1, 2),
        token_ids_b=(1, 2),
        source_duration_a=0.3,
        source_duration_b=0.3,
        fbank_a=_fbank(),
        fbank_b=_fbank(),
    )
    assert ok is False
    assert "interpolat" not in reason.lower()
    same, same_reason = pair_time_grids_comparable(
        a,
        a,
        token_ids_a=(1, 2),
        token_ids_b=(1, 2),
        source_duration_a=0.3,
        source_duration_b=0.3,
        fbank_a=_fbank(),
        fbank_b=_fbank(),
    )
    assert same is True
    assert same_reason == ""


def test_length_bins_last_edge_is_overflow():
    bins = [0.0, 100.0, 200.0, 400.0, 800.0]
    assert length_bin_label(0, bins) == "0-100"
    assert length_bin_label(99, bins) == "0-100"
    assert length_bin_label(100, bins) == "100-200"
    assert length_bin_label(800, bins) == "800+"
    assert length_bin_label(1200, bins) == "800+"


def test_text_heatmap_puts_sink_in_its_own_first_column():
    text_to_sink = np.array([0.9, 0.8, 0.7], dtype=np.float32)
    text_to_audio = np.arange(12, dtype=np.float32).reshape(3, 4) / 20.0
    matrix = assemble_text_key_heatmap(text_to_sink, text_to_audio)
    assert matrix.shape == (3, 5)
    assert np.allclose(matrix[:, 0], text_to_sink)
    assert np.allclose(matrix[:, 1:], text_to_audio)
    assert matrix.max() <= 1.0 + 1e-6


def _write_minimal_run(
    tmp_path: Path,
    *,
    keyword: str = "hey eva",
    selected: bool = True,
    save_traces: bool = True,
    audio_length: int = 20,
    n_samples: int = 1,
    max_report_samples: int = 40,
    time_axis_status: str = "ok",
) -> Path:
    out = tmp_path / "run"
    traces = out / "traces"
    traces.mkdir(parents=True, exist_ok=True)
    waveform = np.zeros((1, 3200), dtype=np.float32)
    records = []
    metrics = []
    positions = []
    samples_meta = {}
    selected_ids = []
    for index in range(n_samples):
        sample_id = f"sample_{index}"
        internal = f"sample_{index}_aa"
        trace_rel = f"traces/{internal}__normal.npz"
        if save_traces:
            _npz_trace(
                out / trace_rel,
                audio_length=audio_length,
                waveform=waveform,
            )
        report_flag = "true" if selected and index < max_report_samples else "false"
        if report_flag == "true":
            selected_ids.append(sample_id)
        records.append(
            _record(
                sample_id=sample_id,
                keyword=keyword,
                report_selected=report_flag,
                trace_path=trace_rel if save_traces else "",
                audio_length=str(audio_length),
                condition="clean" if index % 2 == 0 else "noisy",
                label="1" if index != 1 else "0",
            )
        )
        samples_meta[sample_id] = {
            "internal_sample_id": internal,
            "source_duration_sec": 0.2,
            "source_sample_rate": 16000,
            "model_sample_rate": 16000,
            "audio_length": audio_length,
            "text_length": 2,
            "keyword_spans": [[0.04, 0.12]],
            "noise_spans": [[0.0, 0.03]] if index % 2 else [],
            "time_axis_status": time_axis_status,
            "num_fbank_frames": audio_length,
            "query_id": "[1,2]",
            "token_ids": [1, 2],
            "phonemes": ["HH", "EY1"],
        }
        if time_axis_status != "ok":
            samples_meta[sample_id]["centers_source_sec"] = []
            samples_meta[sample_id]["status"] = "ok"
        for layer in (0, 1):
            for head in (0, 1):
                metrics.append(
                    {
                        "run_id": "run_test",
                        "sample_id": sample_id,
                        "ablation": "normal",
                        "layer": str(layer),
                        "head": str(head),
                        "region": "audio",
                        "mean": "0.25",
                        "min": "0.2",
                        "max": "0.3",
                        "query_count": str(audio_length),
                        "key_count": "1",
                        "row_sum_error": "0.0",
                    }
                )
                metrics.append(
                    {
                        "run_id": "run_test",
                        "sample_id": sample_id,
                        "ablation": "normal",
                        "layer": str(layer),
                        "head": str(head),
                        "region": "text",
                        "mean": "0.2",
                        "min": "0.1",
                        "max": "0.3",
                        "query_count": "2",
                        "key_count": "1",
                        "row_sum_error": "0.0",
                    }
                )
        positions.append(
            {
                "run_id": "run_test",
                "sample_id": sample_id,
                "ablation": "normal",
                "position": "0",
                "phoneme": "HH",
                "position_logit": "0.1",
                "delta_position_logit": "0.0",
            }
        )
    method = (
        "unavailable"
        if time_axis_status == "unavailable"
        else "kaldi_fbank_center+identity_output_frames"
    )
    run = _base_run_payload(
        time_axis_method=method,
        time_axis_status=time_axis_status,
        samples=samples_meta,
        report_selected=selected_ids,
        sink_diagnostics={
            "max_report_samples": max_report_samples,
            "plot_dpi": 72,
            "length_bins": [0, 100, 200, 400, 800],
            "group_field": "condition",
            "save_traces": save_traces,
        },
    )
    if time_axis_status == "unavailable":
        run["time_axis"]["status"] = "unavailable"
        run["time_axis"]["method"] = "unavailable"
        run["time_axis"]["encoder"] = {"kind": "unavailable", "method": "unavailable"}
    summary = {
        "status": "complete",
        "run_id": "run_test",
        "num_input": n_samples,
        "num_success": n_samples,
        "num_skipped": 0,
        "num_fail": 0,
        "num_unlabeled": 0,
        "max_parity_error": 0.0,
        "pairs": {
            "n_pair_ids": 0,
            "n_ok": 0,
            "n_missing_clean": 0,
            "n_multiple_clean": 0,
            "n_inconsistent": 0,
        },
        "group_metrics": {"clean": {"n": n_samples, "n_labeled": n_samples, "mean_qbyt_score": 0.6}},
        "output_index": {
            "records_csv": "records.csv",
            "report_html": "report.html",
            "figures_dir": "figures",
        },
    }
    _write_csv(out / "records.csv", RECORDS_FIELDS, records)
    _write_csv(out / "attention_metrics.csv", METRICS_FIELDS, metrics)
    _write_csv(out / "position_scores.csv", POSITION_FIELDS, positions)
    _write_csv(out / "pairs.csv", PAIR_FIELDS, [])
    (out / "run.json").write_text(json.dumps(run, indent=2) + "\n", encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return out


def test_html_escapes_csv_text_and_is_offline(tmp_path):
    keyword = '<script>alert("x")</script> & hey'
    out = _write_minimal_run(tmp_path, keyword=keyword)
    result = render_sink_attention_report(out)
    html_path = out / "report.html"
    assert html_path.is_file()
    html = html_path.read_text(encoding="utf-8")
    assert "<script>alert" not in html
    assert "&lt;script&gt;" in html
    assert "https://" not in html
    assert "http://" not in html
    assert "fetch(" not in html
    assert "cdn." not in html.lower()
    assert 'src="figures/' in html or "src='figures/" in html
    assert "absorbs noise" not in html.lower()
    assert "sink attention > 0.5" not in html
    assert SYNTHETIC_FIXTURE_BANNER not in html
    assert result["status"] in {"complete", "generated"}
    figures = list((out / "figures").rglob("*.png"))
    assert figures
    audio_heat = next(path for path in figures if "audio_to_sink" in path.name)
    spec = next(path for path in figures if "spectrogram" in path.name)
    assert audio_heat.stat().st_size > 0
    assert spec.stat().st_size > 0


def test_synthetic_fixture_flag_renders_visible_html_banner(tmp_path):
    out = _write_minimal_run(tmp_path)
    run_path = out / "run.json"
    summary_path = out / "summary.json"
    run = json.loads(run_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    run["synthetic_fixture"] = True
    summary["synthetic_fixture"] = True
    summary["banner"] = SYNTHETIC_FIXTURE_BANNER
    run_path.write_text(json.dumps(run, indent=2) + "\n", encoding="utf-8")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    render_sink_attention_report(out)
    html = (out / "report.html").read_text(encoding="utf-8")
    assert html.index(SYNTHETIC_FIXTURE_BANNER) < html.index("<h1>")
    assert 'class="synthetic-banner"' in html
    assert "Not a trained-model conclusion" in html
    assert SYNTHETIC_FIXTURE_BANNER in html


def test_unavailable_time_axis_separates_frame_axis_from_waveform_time(tmp_path):
    out = _write_minimal_run(tmp_path, time_axis_status="unavailable")
    render_sink_attention_report(out)
    html = (out / "report.html").read_text(encoding="utf-8")
    assert "encoder_frame_index" in html
    assert "unavailable" in html
    assert "independent" in html.lower() or "separate" in html.lower() or "frame axis" in html.lower()


def test_figure_cap_does_not_drop_records_csv_rows(tmp_path):
    out = _write_minimal_run(
        tmp_path, n_samples=3, max_report_samples=1, selected=True
    )
    render_sink_attention_report(out)
    with (out / "records.csv").open("r", encoding="utf-8", newline="") as handle:
        records = list(csv.DictReader(handle))
    sample_ids = {row["sample_id"] for row in records}
    assert sample_ids == {"sample_0", "sample_1", "sample_2"}
    selected = {row["sample_id"] for row in records if row["report_selected"] == "true"}
    assert selected == {"sample_0"}
    sample_dirs = [path for path in (out / "figures").iterdir() if path.is_dir()]
    assert len(sample_dirs) == 1


def test_empty_run_renders_without_zerodivision(tmp_path):
    out = tmp_path / "empty"
    out.mkdir()
    run = _base_run_payload(time_axis_method="unavailable", time_axis_status="unavailable")
    run["time_axis"]["status"] = "unavailable"
    run["time_axis"]["method"] = "unavailable"
    summary = {
        "status": "complete",
        "run_id": "run_test",
        "num_input": 0,
        "num_success": 0,
        "num_skipped": 0,
        "num_fail": 0,
        "num_unlabeled": 0,
        "max_parity_error": None,
        "pairs": {
            "n_pair_ids": 0,
            "n_ok": 0,
            "n_missing_clean": 0,
            "n_multiple_clean": 0,
            "n_inconsistent": 0,
        },
        "group_metrics": {},
        "output_index": {"report_html": "report.html", "figures_dir": "figures"},
    }
    _write_csv(out / "records.csv", RECORDS_FIELDS, [])
    _write_csv(out / "attention_metrics.csv", METRICS_FIELDS, [])
    _write_csv(out / "position_scores.csv", POSITION_FIELDS, [])
    _write_csv(out / "pairs.csv", PAIR_FIELDS, [])
    (out / "run.json").write_text(json.dumps(run, indent=2) + "\n", encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    result = render_sink_attention_report(out)
    html = (out / "report.html").read_text(encoding="utf-8")
    assert "run_test" in html
    assert result["num_input"] == 0
    assert (out / "figures").is_dir()


def test_offline_redraw_does_not_need_original_wav(tmp_path):
    out = _write_minimal_run(tmp_path)
    first = render_sink_attention_report(out)
    html1 = (out / "report.html").read_text(encoding="utf-8")
    for png in (out / "figures").rglob("*.png"):
        png.unlink()
    (out / "report.html").unlink()
    second = render_sink_attention_report(out)
    html2 = (out / "report.html").read_text(encoding="utf-8")
    assert html1
    assert html2
    assert list((out / "figures").rglob("*.png"))
    assert first["num_selected"] == second["num_selected"]


def test_offline_redraw_refuses_missing_trace_plots(tmp_path):
    out = _write_minimal_run(tmp_path, save_traces=False)
    result = render_sink_attention_report(out)
    html = (out / "report.html").read_text(encoding="utf-8")
    assert "trace" in html.lower()
    assert result["skipped_plots"]
    heatmaps = list((out / "figures").rglob("*audio_to_sink*.png"))
    assert heatmaps == []


def test_selection_is_deterministic_in_run_json(tmp_path, monkeypatch, install_runner):
    from test_diagnose_qbyt_sink import (
        _cfg,
        _run_diagnose,
        _sink_defaults,
        _write_csv as _diag_csv,
        _write_wav,
        _read_csv,
    )

    wavs = [_write_wav(tmp_path / f"c{i}.wav", 1.0) for i in range(4)]
    manifest = _diag_csv(
        tmp_path / "manifest.csv",
        "audio_path,keyword,label,sample_id,condition",
        f"{wavs[0].name},hey eva,1,id_a,clean",
        f"{wavs[1].name},hey eva,0,id_b,noisy",
        f"{wavs[2].name},hey eva,1,id_c,clean",
        f"{wavs[3].name},hey eva,0,id_d,room",
    )
    summary = _run_diagnose(
        monkeypatch,
        _cfg(
            tmp_path,
            manifest=manifest,
            sink=_sink_defaults(max_report_samples=2, ablations=[], plot_dpi=72),
        ),
    )
    assert summary["status"] == "complete"
    out = tmp_path / "out"
    run = json.loads((out / "run.json").read_text(encoding="utf-8"))
    records = _read_csv(out / "records.csv")
    selected_from_csv = sorted(
        {row["sample_id"] for row in records if row["report_selected"] == "true"}
    )
    assert run["report_selected"] == selected_from_csv
    assert len(run["report_selected"]) <= 2
    assert len({row["sample_id"] for row in records}) == 4
    html = (out / "report.html").read_text(encoding="utf-8")
    assert "4" in html
    assert str(len(run["report_selected"])) in html
    assert run["time_axis_method"] != "pending"
    assert run["time_axis_status"] in {"ok", "unavailable"}
    if run["time_axis_status"] == "ok":
        assert "identity" in run["time_axis_method"] or "encoder_output_frames" in run["time_axis_method"]
    figures = list((out / "figures").rglob("*.png"))
    assert figures
    assert (out / "report.html").is_file()


def test_cli_empty_and_skipped_still_write_html(tmp_path, monkeypatch, install_runner):
    from test_diagnose_qbyt_sink import _cfg, _run_diagnose, _write_csv as _diag_csv, _write_wav

    empty = _diag_csv(tmp_path / "empty.csv", "audio_path,keyword,label")
    empty_summary = _run_diagnose(
        monkeypatch, _cfg(tmp_path, manifest=empty, output_dir=str(tmp_path / "empty"))
    )
    assert empty_summary["status"] == "complete"
    html = (tmp_path / "empty" / "report.html").read_text(encoding="utf-8")
    assert "num_input" in html or "0" in html
    assert "https://" not in html

    short = _write_wav(tmp_path / "tiny.wav", 0.001)
    skipped_manifest = _diag_csv(
        tmp_path / "skipped.csv",
        "audio_path,keyword,label",
        f"{short.name},hey eva,1",
    )
    skipped = _run_diagnose(
        monkeypatch,
        _cfg(tmp_path, manifest=skipped_manifest, output_dir=str(tmp_path / "skipped")),
    )
    assert skipped["num_skipped"] >= 1
    skipped_html = (tmp_path / "skipped" / "report.html").read_text(encoding="utf-8")
    assert skipped_html
    assert "https://" not in skipped_html


def test_span_inclusion_is_half_open_at_end():
    n = 4
    centers = np.array([0.05, 0.10, 0.15, 0.20])
    axis = SampleTimeAxis(
        status="ok",
        method="test",
        centers_source_sec=centers,
        is_padding=np.zeros(n, dtype=bool),
        is_left_padding=np.zeros(n, dtype=bool),
        is_right_padding=np.zeros(n, dtype=bool),
        fbank_support=np.stack([np.arange(n), np.arange(n)], axis=1),
        encoder_frame_index=np.arange(n),
        source_duration_sec=0.3,
    )
    noise, _outside, intervals = noise_region_masks(axis, ((0.05, 0.15),))
    assert noise.tolist() == [True, True, False, False]
    assert intervals[0]["valid_frame_count"] == 2


def test_unavailable_stored_axis_is_not_rebuilt_from_run_encoder():
    live = SampleTimeAxis.unavailable(8)
    meta = live.as_dict()
    meta["audio_length"] = 8
    meta["time_axis_status"] = "unavailable"
    meta["status"] = "ok"
    meta["num_fbank_frames"] = 8
    axis = _axis_for_sample(
        "s1",
        records=[{"audio_length": "8"}],
        meta=meta,
        encoder_map=inspect_encoder_time_map(_IdentityEncoder()),
        fbank=_fbank(),
        left_padding_ms=0,
        right_padding_ms=0,
        fallback_status="ok",
    )
    assert axis.status == "unavailable"
    assert axis.encoder_frame_index.tolist() == list(range(8))

    empty_centers = {
        "centers_source_sec": [],
        "status": "ok",
        "audio_length": 8,
        "num_fbank_frames": 99,
    }
    rebuilt = _axis_for_sample(
        "s2",
        records=[{"audio_length": "8"}],
        meta=empty_centers,
        encoder_map=inspect_encoder_time_map(_IdentityEncoder()),
        fbank=_fbank(),
        left_padding_ms=0,
        right_padding_ms=0,
        fallback_status="ok",
    )
    assert rebuilt.status == "unavailable"


def test_sample_title_includes_phonemes_readout_and_threshold():
    title = _sample_title(
        {
            "sample_id": "id_a",
            "keyword": "hey eva",
            "keyword_phonemes": "HH EY1 IY1 V AH0",
            "condition": "clean",
            "label": "1",
            "qbyt_score": "0.6",
            "threshold": "0.5",
        },
        {"readout_spec": '{"mode":"eps_softmin","sink_token":true,"version":4}'},
    )
    assert "id_a" in title
    assert "hey eva" in title
    assert "HH EY1 IY1 V AH0" in title
    assert "clean" in title
    assert "label=1" in title
    assert "0.6" in title
    assert "threshold=0.5" in title
    assert "eps_softmin" in title


def test_pair_delta_uses_internal_sample_id_when_trace_path_empty(tmp_path):
    out = tmp_path / "pair_run"
    partial = out / ".partial" / "traces"
    partial.mkdir(parents=True)
    waveform = np.zeros((1, 3200), dtype=np.float32)
    audio_length = 20
    samples = [
        ("clean_001", "clean_xx", "clean"),
        ("noisy_001", "noisy_yy", "noisy"),
    ]
    records = []
    metrics = []
    samples_meta = {}
    for sample_id, internal, condition in samples:
        _npz_trace(
            partial / f"{internal}__normal.npz",
            audio_length=audio_length,
            waveform=waveform,
        )
        records.append(
            _record(
                sample_id=sample_id,
                condition=condition,
                pair_id="p001",
                report_selected="true",
                trace_path="",
                audio_length=str(audio_length),
                label="1",
            )
        )
        samples_meta[sample_id] = {
            "internal_sample_id": internal,
            "source_duration_sec": 0.2,
            "source_sample_rate": 16000,
            "model_sample_rate": 16000,
            "audio_length": audio_length,
            "text_length": 2,
            "keyword_spans": [],
            "noise_spans": [[0.0, 0.03]] if condition == "noisy" else [],
            "time_axis_status": "ok",
            "num_fbank_frames": audio_length,
            "query_id": "[1,2]",
            "token_ids": [1, 2],
            "phonemes": ["HH", "EY1"],
        }
        for layer in (0, 1):
            for head in (0, 1):
                metrics.append(
                    {
                        "run_id": "run_test",
                        "sample_id": sample_id,
                        "ablation": "normal",
                        "layer": str(layer),
                        "head": str(head),
                        "region": "audio",
                        "mean": "0.25",
                        "min": "0.2",
                        "max": "0.3",
                        "query_count": str(audio_length),
                        "key_count": "1",
                        "row_sum_error": "0.0",
                    }
                )
    run = _base_run_payload(
        samples=samples_meta,
        report_selected=["clean_001", "noisy_001"],
        sink_diagnostics={
            "max_report_samples": 40,
            "plot_dpi": 72,
            "length_bins": [0, 100, 200, 400, 800],
            "group_field": "condition",
            "save_traces": False,
        },
    )
    pairs = [
        {
            "baseline_sample_id": "clean_001",
            "variant_sample_id": "noisy_001",
            "match_key": '{"keyword":"hey eva"}',
            "pair_status": "ok",
            "normal_score_delta": "0.0",
            "time_grid_comparable": "true",
            "pair_reason": "",
        }
    ]
    summary = {
        "status": "complete",
        "run_id": "run_test",
        "num_input": 2,
        "num_success": 2,
        "num_skipped": 0,
        "num_fail": 0,
        "num_unlabeled": 0,
        "pairs": {"n_pair_ids": 1, "n_ok": 1},
        "group_metrics": {},
        "output_index": {"report_html": "report.html", "figures_dir": "figures"},
    }
    _write_csv(out / "records.csv", RECORDS_FIELDS, records)
    _write_csv(out / "attention_metrics.csv", METRICS_FIELDS, metrics)
    _write_csv(out / "position_scores.csv", POSITION_FIELDS, [])
    _write_csv(out / "pairs.csv", PAIR_FIELDS, pairs)
    (out / "run.json").write_text(json.dumps(run, indent=2) + "\n", encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    result = render_sink_attention_report(out)
    deltas = list((out / "figures").rglob("pair_delta_*.png"))
    assert deltas
    assert not any("missing traces" in item for item in result["skipped_plots"])
