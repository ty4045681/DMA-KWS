from __future__ import annotations

import hashlib
import json
import pickle
import struct
from pathlib import Path

import numpy as np
import pytest

from dma_kws.inference.audio_export import (
    SelectedWaveformExporter,
    WaveformExportConfig,
    parse_waveform_export_config,
    select_waveform_indices,
)


def _expected_random_indices(
    paths: list[str], *, count: int, seed: int
) -> tuple[int, ...]:
    ranked = sorted(
        range(len(paths)),
        key=lambda index: (
            hashlib.sha256(f"{seed}\0{index}\0{paths[index]}".encode()).digest(),
            index,
        ),
    )
    return tuple(sorted(ranked[:count]))


def test_random_selection_is_sha256_deterministic_and_pickle_friendly(tmp_path):
    paths = [f"/audio/clip_{index}.wav" for index in range(12)]
    exporter = SelectedWaveformExporter(
        output_dir=tmp_path,
        audio_paths=paths,
        mode="random",
        count=5,
        seed=19,
    )

    assert exporter.selected_indices == _expected_random_indices(
        paths,
        count=5,
        seed=19,
    )
    assert pickle.loads(pickle.dumps(exporter)).selected_indices == (
        exporter.selected_indices
    )
    assert select_waveform_indices(
        paths,
        WaveformExportConfig(mode="all", count=5, seed=19),
    ) == tuple(range(len(paths)))
    assert select_waveform_indices(
        paths,
        WaveformExportConfig(mode="disabled", count=5, seed=19),
    ) == ()


@pytest.mark.parametrize(
    ("raw", "error_type", "message"),
    [
        ({"mode": "sometimes"}, ValueError, "mode must be one of"),
        ({"mode": True}, ValueError, "mode must be one of"),
        ({"mode": "random", "count": 0}, ValueError, "count must be >= 1"),
        ({"count": True}, TypeError, "count must be an integer"),
        ({"seed": -1}, ValueError, "seed must be >= 0"),
        ({"seed": 1.5}, TypeError, "seed must be an integer"),
        ({"extra": 1}, ValueError, "unsupported keys"),
    ],
)
def test_waveform_export_config_is_strict(raw, error_type, message):
    with pytest.raises(error_type, match=message):
        parse_waveform_export_config({"audio_export": raw})


@pytest.mark.parametrize("mode", ["disabled", "all"])
def test_non_random_modes_accept_zero_count_used_by_batch_runner(mode):
    config = parse_waveform_export_config(
        {"audio_export": {"mode": mode, "count": 0, "seed": 2025}}
    )

    assert config == WaveformExportConfig(mode=mode, count=0, seed=2025)


def test_export_preserves_float_peaks_and_finalize_writes_manifest(tmp_path):
    torch = pytest.importorskip("torch")
    paths = ["/source/first.wav", "/source/second.wav"]
    exporter = SelectedWaveformExporter(
        output_dir=tmp_path / "exports",
        audio_paths=paths,
        mode="all",
        count=5,
        seed=7,
    )
    exporter.prepare()
    first = torch.tensor([[0.0, 1.5, -2.25, 0.5]], dtype=torch.float32)
    second = torch.tensor([[3.0, -0.25]], dtype=torch.float64)

    exporter(0, first, 16000)
    exporter(1, second, 8000)
    summary = exporter.finalize()

    first_path = Path(exporter.result_path(0))
    wav_payload = first_path.read_bytes()
    assert wav_payload[:4] == b"RIFF"
    assert wav_payload[8:12] == b"WAVE"
    assert struct.unpack_from("<H", wav_payload, 20)[0] == 3
    assert struct.unpack_from("<I", wav_payload, 24)[0] == 16000
    assert b"fact" in wav_payload
    data_offset = wav_payload.index(b"data", 12)
    data_size = struct.unpack_from("<I", wav_payload, data_offset + 4)[0]
    samples = np.frombuffer(
        wav_payload[data_offset + 8 : data_offset + 8 + data_size],
        dtype="<f4",
    )
    assert samples.tolist() == pytest.approx(first.squeeze(0).tolist())
    assert float(abs(samples).max()) == pytest.approx(2.25)
    assert not list(first_path.parent.glob(".*.wav"))

    assert summary == {
        "status": "generated",
        "mode": "all",
        "requested_count": "all",
        "seed": 7,
        "num_selected": 2,
        "num_exported": 2,
        "stage": "post_augmentation_pre_padding",
        "format": "WAV",
        "subtype": "FLOAT",
        "directory": str((tmp_path / "exports").resolve()),
        "manifest": str((tmp_path / "exports" / "index.jsonl").resolve()),
        "row_indices": [0, 1],
    }
    records = [
        json.loads(line)
        for line in exporter.index_path.read_text(encoding="utf-8").splitlines()
    ]
    assert records[0] == {
        "row_index": 0,
        "source_audio_path": paths[0],
        "exported_audio_path": str(first_path),
        "sample_rate": 16000,
        "num_samples": 4,
        "duration_sec": pytest.approx(4 / 16000),
        "peak_abs": pytest.approx(2.25),
    }
    assert records[1]["peak_abs"] == pytest.approx(3.0)


def test_prepare_removes_only_owned_row_wavs(tmp_path):
    output_dir = tmp_path / "exports"
    output_dir.mkdir()
    owned = output_dir / "row_00000000.wav"
    unrelated_wav = output_dir / "row_notes.wav"
    unrelated_json = output_dir / "summary.json"
    owned_index = output_dir / "index.jsonl"
    owned.write_bytes(b"old")
    owned_index.write_bytes(b"old")
    unrelated_wav.write_bytes(b"keep")
    unrelated_json.write_bytes(b"keep")
    exporter = SelectedWaveformExporter(
        output_dir=output_dir,
        audio_paths=["source.wav"],
        mode="all",
    )

    exporter.prepare()

    assert not owned.exists()
    assert not owned_index.exists()
    assert unrelated_wav.read_bytes() == b"keep"
    assert unrelated_json.read_bytes() == b"keep"


def test_disabled_prepare_removes_stale_owned_exports(tmp_path):
    output_dir = tmp_path / "exports"
    output_dir.mkdir()
    stale_wav = output_dir / "row_00000000.wav"
    stale_index = output_dir / "index.jsonl"
    stale_wav.write_bytes(b"old")
    stale_index.write_bytes(b"old")
    exporter = SelectedWaveformExporter(
        output_dir=output_dir,
        audio_paths=["source.wav"],
        mode="disabled",
        count=0,
    )

    exporter.prepare()

    assert not stale_wav.exists()
    assert not stale_index.exists()


def test_finalize_rejects_missing_selected_exports(tmp_path):
    exporter = SelectedWaveformExporter(
        output_dir=tmp_path,
        audio_paths=["source.wav"],
        mode="all",
    )
    exporter.prepare()

    with pytest.raises(FileNotFoundError, match="missing 1 selected"):
        exporter.finalize()
