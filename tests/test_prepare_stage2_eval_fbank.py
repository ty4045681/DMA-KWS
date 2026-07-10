from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

from dma_kws.stage2.prepare_eval_fbank import (
    collect_eval_wav_relpaths,
    prepare_eval_fbank,
    prepare_eval_fbank_from_csv,
    resolve_eval_fbank_path_from_wav,
)


def test_resolve_eval_fbank_path_from_wav_replaces_suffix():
    wav = Path("/data/eval/train-other-500/foo.wav")
    assert resolve_eval_fbank_path_from_wav(wav) == Path("/data/eval/train-other-500/foo.npy")


def test_resolve_eval_fbank_path_from_wav_mirrors_independent_root():
    wav = Path("/data/eval/train-other-500/foo.wav")

    path = resolve_eval_fbank_path_from_wav(
        wav,
        test_dir=Path("/data/eval"),
        fbank_dir=Path("/features/eval"),
    )

    assert path == Path("/features/eval/train-other-500/foo.npy")


def test_collect_eval_wav_relpaths_reads_anchor_and_comparison(tmp_path):
    eval_dir = tmp_path / "eval"
    csv_dir = eval_dir / "evaluation_set"
    csv_dir.mkdir(parents=True)

    csv_path = csv_dir / "libriphrase_diffspk_all_1word.csv"
    pd.DataFrame(
        {
            "anchor": ["a/foo.wav"],
            "comparison": ["b/bar.wav"],
            "anchor_text": ["hello"],
            "comparison_text": ["world"],
            "anchor_dur": [1.0],
            "comparison_dur": [1.0],
            "target": [0],
            "type": ["diffspk_easyneg"],
        }
    ).to_csv(csv_path, index=False)

    rel_paths = collect_eval_wav_relpaths(
        eval_dir,
        ["evaluation_set/libriphrase_diffspk_all_1word.csv"],
    )

    assert rel_paths == {"a/foo.wav", "b/bar.wav"}


def test_prepare_eval_fbank_passes_fbank_params_to_compute(tmp_path):
    wav_path = tmp_path / "sample.wav"
    wav_path.write_bytes(b"wav")

    fake_compute = MagicMock(return_value=str(wav_path.with_suffix(".npy")))

    prepare_eval_fbank(
        tmp_path,
        skip_existing=False,
        compute_fn=fake_compute,
        num_mel_bins=64,
        frame_length=20,
        frame_shift=8,
        dither=0.0,
        window_type="hamming",
        backend="lhotse_fbank",
        target_sample_rate=16000,
        snip_edges=False,
        low_freq=20.0,
        high_freq=-400.0,
    )

    fake_compute.assert_called_once_with(
        wav_path,
        wav_path.with_suffix(".npy"),
        num_mel_bins=64,
        frame_length=20,
        frame_shift=8,
        dither=0.0,
        window_type="hamming",
        backend="lhotse_fbank",
        target_sample_rate=16000,
        snip_edges=False,
        low_freq=20.0,
        high_freq=-400.0,
        extractor=None,
    )


def test_prepare_eval_fbank_writes_to_independent_root(tmp_path):
    test_dir = tmp_path / "eval"
    wav_path = test_dir / "clips" / "sample.wav"
    wav_path.parent.mkdir(parents=True)
    wav_path.write_bytes(b"wav")
    fbank_dir = tmp_path / "features" / "eval"
    fake_compute = MagicMock()

    written, skipped, failed = prepare_eval_fbank(
        test_dir,
        fbank_dir=fbank_dir,
        skip_existing=False,
        compute_fn=fake_compute,
    )

    assert (written, skipped, failed) == (1, 0, 0)
    assert fake_compute.call_args.args[0] == wav_path
    assert fake_compute.call_args.args[1] == fbank_dir / "clips" / "sample.npy"


def test_prepare_eval_fbank_writes_npy_next_to_wav(tmp_path):
    wav_path = tmp_path / "train-other-500" / "sample.wav"
    wav_path.parent.mkdir(parents=True)
    wav_path.write_bytes(b"wav")

    fake_compute = MagicMock(return_value=str(wav_path.with_suffix(".npy")))

    written, skipped, failed = prepare_eval_fbank(
        tmp_path,
        skip_existing=False,
        compute_fn=fake_compute,
    )

    assert written == 1
    assert skipped == 0
    assert failed == 0
    fake_compute.assert_called_once()
    assert fake_compute.call_args.args[0] == wav_path
    assert fake_compute.call_args.args[1] == wav_path.with_suffix(".npy")


def test_prepare_eval_fbank_skips_existing_npy(tmp_path):
    wav_path = tmp_path / "sample.wav"
    npy_path = wav_path.with_suffix(".npy")
    wav_path.write_bytes(b"wav")
    npy_path.write_bytes(b"npy")

    fake_compute = MagicMock()

    written, skipped, failed = prepare_eval_fbank(
        tmp_path,
        skip_existing=True,
        compute_fn=fake_compute,
    )

    assert written == 0
    assert skipped == 1
    assert failed == 0
    fake_compute.assert_not_called()


def test_prepare_eval_fbank_from_csv_uses_manifest_paths(tmp_path, monkeypatch):
    eval_dir = tmp_path / "eval"
    csv_dir = eval_dir / "evaluation_set"
    csv_dir.mkdir(parents=True)

    rel_wav = "clips/sample.wav"
    wav_path = eval_dir / rel_wav
    wav_path.parent.mkdir(parents=True)
    wav_path.write_bytes(b"wav")

    pd.DataFrame(
        {
            "anchor": [rel_wav],
            "comparison": [rel_wav],
            "anchor_text": ["hello"],
            "comparison_text": ["hello"],
            "anchor_dur": [1.0],
            "comparison_dur": [1.0],
            "target": [1],
            "type": ["diffspk_positive"],
        }
    ).to_csv(csv_dir / "libriphrase_diffspk_all_1word.csv", index=False)

    fake_compute = MagicMock(return_value=str(wav_path.with_suffix(".npy")))
    monkeypatch.setattr(
        "dma_kws.stage2.prepare_eval_fbank._DEFAULT_EVAL_CSV",
        ["evaluation_set/libriphrase_diffspk_all_1word.csv"],
    )

    written, skipped, failed = prepare_eval_fbank_from_csv(
        eval_dir,
        skip_existing=False,
        compute_fn=fake_compute,
    )

    assert written == 1
    assert skipped == 0
    assert failed == 0
    fake_compute.assert_called_once()
