from pathlib import Path
from functools import partial

import numpy as np
import pandas as pd
import pytest

from dma_kws.config import FbankConfig, fbank_kwargs
from dma_kws.stage2.dataset import LibriPhraseTrainDataset
from scripts.prepare_stage2_paper import collect_needed_audio_keys, find_decoded_parquets
from dma_kws.stage2.prepare_paper import (
    PARQUET_COLUMNS,
    build_clips_npy,
    build_distances_npy,
    compute_hard_negatives_from_phonemes,
    convert_aggregated_to_paper_parquet,
    resolve_fbank_rel_path,
    slug_from_ngram,
)


class _FakeTokenizer:
    def tokenize(self, text: str):
        tokens = text.split()
        ids = [hash(token) % 100 + 1 for token in tokens]
        return tokens, ids


def _synthetic_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ngram": ["hello world", "hello word", "goodbye"],
            "ngram_g2p": ["HH AH L OW W ER L D", "HH AH L OW W ER D", "G UH D B AY"],
            "clips": [
                [{"audio_path": "LP-100/hello world/a.wav"}, {"audio_path": "LP-100/hello world/b.wav"}],
                [{"audio_path": "LP-100/hello word/c.wav"}],
                [{"audio_path": "LP-100/goodbye/d.wav"}],
            ],
        }
    )


def _synthetic_gp1000_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ngram": ["hello world", "hello word"],
            "ngram_g2p": ["HH AH L OW W ER L D", "HH AH L OW W ER D"],
            "clips": [
                [
                    {"audio_path": "GP-1000/hello world/a.wav"},
                    {"audio_path": "GP-1000/hello world/b.wav"},
                ],
                [{"audio_path": "GP-1000/hello word/c.wav"}],
            ],
        }
    )


def _mock_gp1000_audio() -> dict[str, tuple[np.ndarray, int]]:
    return {
        "hello world/a.wav": (np.linspace(-0.1, 0.1, 1600, dtype=np.float32), 16000),
        "hello world/b.wav": (np.linspace(-0.2, 0.2, 1600, dtype=np.float32), 16000),
        "hello word/c.wav": (np.linspace(-0.15, 0.15, 1600, dtype=np.float32), 16000),
    }


def _synthetic_mixed_gp_lp460_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ngram": ["tomatoes"],
            "ngram_g2p": ["T AH M EY T OW Z"],
            "clips": [
                [
                    {"audio_path": "GP-1000/tomatoes/AUD0000001043_S0000775_000.wav"},
                    {"audio_path": "LP-460/tomatoes/1963-142776-0033_012.wav"},
                ],
            ],
        }
    )


def _mock_mixed_gp_lp460_audio() -> dict[str, tuple[np.ndarray, int]]:
    return {
        "tomatoes/AUD0000001043_S0000775_000.wav": (
            np.linspace(-0.1, 0.1, 1600, dtype=np.float32),
            16000,
        ),
        "tomatoes/1963-142776-0033_012.wav": (
            np.linspace(-0.2, 0.2, 1600, dtype=np.float32),
            16000,
        ),
    }


def _mock_audio() -> dict[str, tuple[np.ndarray, int]]:
    return {
        "hello world/a.wav": (np.linspace(-0.1, 0.1, 1600, dtype=np.float32), 16000),
        "hello world/b.wav": (np.linspace(-0.2, 0.2, 1600, dtype=np.float32), 16000),
        "hello word/c.wav": (np.linspace(-0.15, 0.15, 1600, dtype=np.float32), 16000),
        "goodbye/d.wav": (np.linspace(-0.05, 0.05, 1600, dtype=np.float32), 16000),
    }


def _fake_compute_fbank(_waveform_path, fbank_out_path, *, waveform, sample_rate, **_kwargs):
    del sample_rate
    fbank_out_path = Path(fbank_out_path)
    fbank_out_path.parent.mkdir(parents=True, exist_ok=True)
    frames = max(1, len(waveform) // 160)
    feat = np.full((frames, 80), 0.5, dtype=np.float32)
    np.save(fbank_out_path, feat)
    return str(fbank_out_path)


def test_slug_and_fbank_path_rewrite():
    assert slug_from_ngram("Hello World!") == "hello_world"
    assert resolve_fbank_rel_path("LP-100/hello/a.wav") == "LP-100-fbank/hello/a.npy"
    assert resolve_fbank_rel_path("LP-460/hello/a.wav") == "LP-460-fbank/hello/a.npy"
    assert resolve_fbank_rel_path("GP-1000/hello/a.wav") == "GP-1000-fbank/hello/a.npy"


def test_find_decoded_parquets_uses_dataset_glob(tmp_path):
    root = tmp_path / "gp1000"
    root.mkdir()
    gp_shard = root / "GP-1000-decoded-0000.parquet"
    gp_shard.write_bytes(b"")
    other = root / "aggregated_segments.parquet"
    other.write_bytes(b"")

    matches = find_decoded_parquets(root, decoded_glob="GP-1000-decoded-*.parquet")
    assert matches == [gp_shard]


def test_find_decoded_parquets_falls_back_when_glob_misses(tmp_path, capsys):
    root = tmp_path / "mixed"
    root.mkdir()
    fallback = root / "only.parquet"
    fallback.write_bytes(b"")

    matches = find_decoded_parquets(root, decoded_glob="GP-1000-decoded-*.parquet")
    assert matches == [fallback]
    captured = capsys.readouterr()
    assert "WARNING" in captured.out
    assert "GP-1000-decoded-*.parquet" in captured.out


def test_build_clips_and_distances_npy_roundtrip(tmp_path):
    clips = [{"audio_path": "LP-100/hello/a.wav"}, {"audio_path": "LP-100/hello/b.wav"}]
    clips_path = tmp_path / "clips-2-hello.npy"
    build_clips_npy(clips, clips_path)
    loaded = np.load(clips_path, allow_pickle=True)
    assert len(loaded) == 2
    assert loaded[0]["audio_path"] == "LP-100/hello/a.wav"

    build_distances_npy([], tmp_path / "dist-0-hello.npy")
    empty = np.load(tmp_path / "dist-0-hello.npy", allow_pickle=True)
    assert len(empty) == 0

    hard = [{"ngram": "hello word"}]
    build_distances_npy(hard, tmp_path / "dist-1-hello.npy")
    loaded_hard = np.load(tmp_path / "dist-1-hello.npy", allow_pickle=True)
    assert loaded_hard[0]["ngram"] == "hello word"


def test_compute_hard_negatives_prefers_small_edit_distance():
    candidates = [
        ("hello world", "HH AH L OW W ER L D"),
        ("hello word", "HH AH L OW W ER D"),
        ("goodbye", "G UH D B AY"),
    ]
    hard = compute_hard_negatives_from_phonemes(
        "hello world",
        "HH AH L OW W ER L D",
        candidates,
        top_k=1,
    )
    assert [item["ngram"] for item in hard] == ["hello word"]


def test_convert_aggregated_to_paper_parquet_writes_expected_layout(tmp_path):
    processed = tmp_path / "processed" / "stage2_qbyt"
    clips_dir = processed / "clips"
    distances_dir = processed / "distances"
    fbank_dir = tmp_path / "features" / "fbank"

    paper_df, stats = convert_aggregated_to_paper_parquet(
        _synthetic_df(),
        clips_dir=clips_dir,
        distances_dir=distances_dir,
        fbank_dir=fbank_dir,
        audio_by_rel=_mock_audio(),
        compute_fbank=_fake_compute_fbank,
    )

    assert list(paper_df.columns) == PARQUET_COLUMNS
    assert stats["anchors"] == 3
    assert stats["missing_audio"] == 0
    assert stats["fbank_written"] == 4

    hello_row = paper_df.loc[paper_df["ngram"] == "hello world"].iloc[0]
    assert hello_row["clips_file"].endswith("clips-2-hello_world.npy")
    assert hello_row["distances_file"].endswith("dist-2-hello_world.npy")

    clips = np.load(hello_row["clips_file"], allow_pickle=True)
    assert clips[0]["audio_path"] == "LP-100/hello world/a.wav"

    distances = np.load(hello_row["distances_file"], allow_pickle=True)
    assert distances[0]["ngram"] == "hello word"
    assert len(distances) == 2

    for clip in clips:
        fbank_path = fbank_dir / resolve_fbank_rel_path(clip["audio_path"])
        assert fbank_path.exists()
        feat = np.load(fbank_path)
        assert feat.shape[1] == 80


def test_convert_gp1000_writes_gp1000_fbank_layout(tmp_path):
    processed = tmp_path / "processed" / "stage2_qbyt"
    clips_dir = processed / "clips"
    distances_dir = processed / "distances"
    fbank_dir = tmp_path / "features" / "fbank"

    paper_df, stats = convert_aggregated_to_paper_parquet(
        _synthetic_gp1000_df(),
        clips_dir=clips_dir,
        distances_dir=distances_dir,
        fbank_dir=fbank_dir,
        audio_by_rel=_mock_gp1000_audio(),
        compute_fbank=_fake_compute_fbank,
    )

    assert stats["anchors"] == 2
    assert stats["missing_audio"] == 0
    assert stats["fbank_written"] == 3

    hello_row = paper_df.loc[paper_df["ngram"] == "hello world"].iloc[0]
    clips = np.load(hello_row["clips_file"], allow_pickle=True)
    assert clips[0]["audio_path"] == "GP-1000/hello world/a.wav"

    for clip in clips:
        fbank_path = fbank_dir / resolve_fbank_rel_path(clip["audio_path"])
        assert "GP-1000-fbank" in str(fbank_path)
        assert fbank_path.exists()


def test_convert_mixed_gp1000_lp460_prefix_writes_both_fbank_layouts(tmp_path):
    processed = tmp_path / "processed" / "stage2_qbyt"
    clips_dir = processed / "clips"
    distances_dir = processed / "distances"
    fbank_dir = tmp_path / "features" / "fbank"

    paper_df, stats = convert_aggregated_to_paper_parquet(
        _synthetic_mixed_gp_lp460_df(),
        clips_dir=clips_dir,
        distances_dir=distances_dir,
        fbank_dir=fbank_dir,
        audio_by_rel=_mock_mixed_gp_lp460_audio(),
        compute_fbank=_fake_compute_fbank,
    )

    assert stats["anchors"] == 1
    assert stats["missing_audio"] == 0
    assert stats["fbank_written"] == 2

    tomatoes_row = paper_df.loc[paper_df["ngram"] == "tomatoes"].iloc[0]
    clips = np.load(tomatoes_row["clips_file"], allow_pickle=True)
    assert clips[0]["audio_path"] == "GP-1000/tomatoes/AUD0000001043_S0000775_000.wav"
    assert clips[1]["audio_path"] == "LP-460/tomatoes/1963-142776-0033_012.wav"

    gp_fbank = fbank_dir / resolve_fbank_rel_path(clips[0]["audio_path"])
    lp_fbank = fbank_dir / resolve_fbank_rel_path(clips[1]["audio_path"])
    assert "GP-1000-fbank" in str(gp_fbank)
    assert "LP-460-fbank" in str(lp_fbank)
    assert gp_fbank.exists()
    assert lp_fbank.exists()


def test_collect_needed_audio_keys_strips_per_clip_prefix():
    needed = collect_needed_audio_keys(_synthetic_mixed_gp_lp460_df())
    assert needed == {
        "tomatoes/AUD0000001043_S0000775_000.wav",
        "tomatoes/1963-142776-0033_012.wav",
    }
    assert not any(key.startswith("GP-1000/") for key in needed)
    assert not any(key.startswith("LP-460/") for key in needed)


def test_convert_aggregated_forwards_bound_fbank_kwargs(tmp_path):
    seen_kwargs: list[dict] = []

    def capture(waveform_path, fbank_out_path, *, waveform, sample_rate, **kwargs):
        del waveform_path, sample_rate
        seen_kwargs.append(kwargs)
        return _fake_compute_fbank(
            "ignored",
            fbank_out_path,
            waveform=waveform,
            sample_rate=16000,
        )

    bound_compute = partial(
        capture,
        **fbank_kwargs(FbankConfig(dither=0.0, frame_shift=8, num_mel_bins=64)),
    )

    convert_aggregated_to_paper_parquet(
        _synthetic_df().head(1),
        clips_dir=tmp_path / "clips",
        distances_dir=tmp_path / "distances",
        fbank_dir=tmp_path / "fbank",
        audio_by_rel=_mock_audio(),
        compute_fbank=bound_compute,
    )

    assert len(seen_kwargs) == 2
    assert seen_kwargs[0] == {
        "num_mel_bins": 64,
        "frame_length": 25,
        "frame_shift": 8,
        "dither": 0.0,
        "window_type": "povey",
        "backend": "torchaudio_kaldi",
        "target_sample_rate": None,
        "snip_edges": True,
        "low_freq": 20.0,
        "high_freq": 0.0,
    }


def test_distances_npy_works_with_dataset_get_hard_negative(tmp_path):
    processed = tmp_path / "processed" / "stage2_qbyt"
    clips_dir = processed / "clips"
    distances_dir = processed / "distances"
    fbank_dir = tmp_path / "features" / "fbank"

    paper_df, _stats = convert_aggregated_to_paper_parquet(
        _synthetic_df(),
        clips_dir=clips_dir,
        distances_dir=distances_dir,
        fbank_dir=fbank_dir,
        audio_by_rel=_mock_audio(),
        compute_fbank=_fake_compute_fbank,
    )

    dataset = LibriPhraseTrainDataset(
        wav_dir=str(fbank_dir),
        tokenizer=_FakeTokenizer(),
        df=paper_df,
        sample_lens=1,
        seed=0,
    )

    hello_idx = dataset.anchor2idx["hello world"]
    distances_file = paper_df.loc[paper_df["ngram"] == "hello world", "distances_file"].iloc[0]
    distances = np.load(distances_file, allow_pickle=True)
    assert any(entry["ngram"] == "hello word" for entry in distances)

    hard_neg = {"ngram": "hello word"}

    negative_wav, negative_g2p, negative = dataset.get_hard_negative(hard_neg)
    assert negative == "hello word"
    assert negative_g2p == "HH AH L OW W ER D"
    assert "audio_path" in negative_wav
    assert negative_wav["audio_path"] == "LP-100/hello word/c.wav"

    # Ensure dataset can load the corresponding fbank for the hard negative clip.
    sample = dataset[hello_idx]
    assert sample["feat"].shape[1] == 80


def test_convert_respects_limit_anchors(tmp_path):
    processed = tmp_path / "processed" / "stage2_qbyt"
    paper_df, stats = convert_aggregated_to_paper_parquet(
        _synthetic_df(),
        clips_dir=processed / "clips",
        distances_dir=processed / "distances",
        fbank_dir=tmp_path / "features" / "fbank",
        audio_by_rel=_mock_audio(),
        limit_anchors=1,
        compute_fbank=_fake_compute_fbank,
    )
    assert len(paper_df) == 1
    assert stats["anchors"] == 1
