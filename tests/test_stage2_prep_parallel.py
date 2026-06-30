from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dma_kws.stage2.pairs import scan_decoded_parquet_shard
from dma_kws.stage2.prep_console import resolve_num_workers
from dma_kws.stage2.prepare_paper import (
    _compute_fbank_jobs,
    _FbankJob,
    build_anchor_metadata,
    convert_aggregated_to_paper_parquet,
    resolve_fbank_rel_path,
    stream_fbank_from_decoded,
)
from scripts.prepare_stage2_paper import load_decoded_audio


def test_resolve_num_workers_auto_and_explicit():
    assert resolve_num_workers(0) >= 1
    assert resolve_num_workers(4) == 4
    with pytest.raises(ValueError):
        resolve_num_workers(-1)


def test_compute_fbank_jobs_parallel_matches_sequential(tmp_path):
    jobs = [
        _FbankJob(
            audio_path=f"clip-{index}.wav",
            fbank_path=tmp_path / f"{index}.npy",
            waveform=np.linspace(-0.1, 0.1, 800, dtype=np.float32),
            sample_rate=16000,
        )
        for index in range(6)
    ]

    def fake_compute(_audio_path, fbank_out_path, *, waveform, sample_rate, **_kwargs):
        del sample_rate
        fbank_out_path = Path(fbank_out_path)
        np.save(fbank_out_path, np.full((max(1, len(waveform) // 160), 80), 0.5, dtype=np.float32))
        return str(fbank_out_path)

    progress: list[str] = []

    def on_progress(stage: str, value: int) -> None:
        progress.append(f"{stage}:{value}")

    written = _compute_fbank_jobs(
        jobs,
        compute_fbank=fake_compute,
        num_workers=2,
        on_progress=on_progress,
    )
    assert written == 6
    assert all((tmp_path / f"{index}.npy").exists() for index in range(6))
    assert progress.count("fbank:1") == 6


def test_scan_decoded_parquet_shard_filters_needed_keys(tmp_path):
    df = pd.DataFrame(
        {
            "audio_rel": ["a.wav", "b.wav", "c.wav"],
            "audio": [
                np.zeros(4, dtype=np.float32),
                np.ones(4, dtype=np.float32),
                np.full(4, 0.5, dtype=np.float32),
            ],
            "sampling_rate": [16000, 16000, 16000],
        }
    )
    parquet_path = tmp_path / "shard.parquet"
    df.to_parquet(parquet_path, index=False)

    found = scan_decoded_parquet_shard(parquet_path, {"a.wav", "c.wav"}, read_parquet=pd.read_parquet)
    assert set(found) == {"a.wav", "c.wav"}


def test_load_decoded_audio_parallel_merges_shards(tmp_path):
    def write_shard(name: str, rows: list[tuple[str, np.ndarray]]) -> Path:
        df = pd.DataFrame(
            {
                "audio_rel": [rel for rel, _ in rows],
                "audio": [audio for _, audio in rows],
                "sampling_rate": [16000] * len(rows),
            }
        )
        path = tmp_path / name
        df.to_parquet(path, index=False)
        return path

    shard_a = write_shard(
        "a.parquet",
        [
            ("x.wav", np.linspace(0, 1, 8, dtype=np.float32)),
            ("y.wav", np.linspace(0, 1, 8, dtype=np.float32)),
        ],
    )
    shard_b = write_shard(
        "b.parquet",
        [
            ("z.wav", np.linspace(0, 1, 8, dtype=np.float32)),
        ],
    )

    loaded = load_decoded_audio(
        [shard_a, shard_b],
        {"x.wav", "y.wav", "z.wav"},
        num_workers=2,
        read_parquet=pd.read_parquet,
    )
    assert set(loaded) == {"x.wav", "y.wav", "z.wav"}
    assert loaded["x.wav"][1] == 16000


def test_convert_parallel_fbank_preserves_layout(tmp_path):
    from tests.test_prepare_stage2_paper import _fake_compute_fbank, _mock_audio, _synthetic_df

    processed = tmp_path / "processed" / "stage2_qbyt"
    paper_df, stats = convert_aggregated_to_paper_parquet(
        _synthetic_df(),
        clips_dir=processed / "clips",
        distances_dir=processed / "distances",
        fbank_dir=tmp_path / "features" / "fbank",
        audio_by_rel=_mock_audio(),
        compute_fbank=_fake_compute_fbank,
        num_workers=2,
    )
    assert len(paper_df) == 3
    assert stats["fbank_written"] == 4
    assert stats["fbank_skipped"] == 0
    assert stats["clips_total"] == 4


def test_streaming_fbank_matches_in_memory_path(tmp_path):
    from tests.test_prepare_stage2_paper import _fake_compute_fbank, _mock_audio, _synthetic_df

    df = _synthetic_df()
    audio = _mock_audio()

    # One decoded shard per clip, so streaming must release memory between shards.
    shard_paths = []
    for index, (audio_rel, (waveform, sr)) in enumerate(audio.items()):
        shard = pd.DataFrame(
            {"audio_rel": [audio_rel], "audio": [waveform], "sampling_rate": [sr]}
        )
        path = tmp_path / f"decoded-{index:04d}.parquet"
        shard.to_parquet(path, index=False)
        shard_paths.append(path)

    clips_dir = tmp_path / "clips"
    distances_dir = tmp_path / "distances"
    fbank_dir = tmp_path / "fbank"

    paper_df, fbank_targets, stats = build_anchor_metadata(
        df,
        clips_dir=clips_dir,
        distances_dir=distances_dir,
        fbank_dir=fbank_dir,
    )
    assert stats["anchors"] == 3
    assert stats["clips_total"] == 4
    assert len(fbank_targets) == 4

    written, missing = stream_fbank_from_decoded(
        shard_paths,
        fbank_targets,
        read_parquet=pd.read_parquet,
        compute_fbank=_fake_compute_fbank,
        num_workers=2,
    )
    assert written == 4
    assert missing == 0

    for _, row in paper_df.iterrows():
        for clip in np.load(row["clips_file"], allow_pickle=True):
            assert (fbank_dir / resolve_fbank_rel_path(clip["audio_path"])).exists()


def test_streaming_fbank_reports_missing_audio(tmp_path):
    from tests.test_prepare_stage2_paper import _fake_compute_fbank, _synthetic_df

    df = _synthetic_df()
    # Decoded shard only contains one of the referenced clips.
    shard = pd.DataFrame(
        {
            "audio_rel": ["hello world/a.wav"],
            "audio": [np.linspace(-0.1, 0.1, 1600, dtype=np.float32)],
            "sampling_rate": [16000],
        }
    )
    shard_path = tmp_path / "decoded-0000.parquet"
    shard.to_parquet(shard_path, index=False)

    _paper_df, fbank_targets, _stats = build_anchor_metadata(
        df,
        clips_dir=tmp_path / "clips",
        distances_dir=tmp_path / "distances",
        fbank_dir=tmp_path / "fbank",
    )

    written, missing = stream_fbank_from_decoded(
        [shard_path],
        fbank_targets,
        read_parquet=pd.read_parquet,
        compute_fbank=_fake_compute_fbank,
    )
    assert written == 1
    assert missing == len(fbank_targets) - 1
