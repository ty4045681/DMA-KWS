from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dma_kws.stage2.distances import (
    build_distances_column,
    encode_phoneme_strings,
    phoneme_tokens_from_g2p,
    top_k_hard_negatives,
)
from dma_kws.stage2.prepare_paper import convert_aggregated_to_paper_parquet, parse_distances
from scripts.recompute_stage2_distances import (
    _default_output,
    resolve_input_parquet,
    resolve_output_parquet,
)


def _synthetic_g2p_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ngram": ["hello world", "hello word", "goodbye"],
            "ngram_lens": [2, 2, 1],
            "clips": [
                json.dumps([{"audio_path": "LP-100/hello world/a.wav"}]),
                json.dumps([{"audio_path": "LP-100/hello word/c.wav"}]),
                json.dumps([{"audio_path": "LP-100/goodbye/d.wav"}]),
            ],
            "ngram_g2p": ["HH AH L OW W ER L D", "HH AH L OW W ER D", "G UH D B AY"],
        }
    )


rapidfuzz = pytest.importorskip("rapidfuzz")


def test_default_output_renames_g2p_suffix():
    input_path = Path("/data/aggregated_segments_with_g2p.parquet")
    assert _default_output(input_path).name == "aggregated_segments_with_g2p_distance.parquet"


def test_resolve_input_parquet_uses_explicit_path():
    prep = {"input_parquet": "/tmp/custom.parquet"}
    assert resolve_input_parquet(prep, {}) == Path("/tmp/custom.parquet")


def test_resolve_input_parquet_falls_back_to_libriphrase_roots(tmp_path):
    root = tmp_path / "LibriPhrase-100"
    root.mkdir()
    parquet = root / "aggregated_segments_with_g2p.parquet"
    parquet.write_bytes(b"")

    prep = {"input_parquet": ""}
    paths = {"libriphrase460_root": "", "libriphrase100_root": str(root)}
    assert resolve_input_parquet(prep, paths) == parquet


def test_resolve_input_parquet_prefers_460_over_100(tmp_path):
    root460 = tmp_path / "LibriPhrase-460"
    root100 = tmp_path / "LibriPhrase-100"
    root460.mkdir()
    root100.mkdir()
    parquet460 = root460 / "aggregated_segments_with_g2p.parquet"
    parquet100 = root100 / "aggregated_segments_with_g2p.parquet"
    parquet460.write_bytes(b"")
    parquet100.write_bytes(b"")

    prep = {"input_parquet": ""}
    paths = {
        "libriphrase460_root": str(root460),
        "libriphrase100_root": str(root100),
    }
    assert resolve_input_parquet(prep, paths) == parquet460


def test_resolve_input_parquet_raises_without_default():
    with pytest.raises(SystemExit, match="prep.recompute_distances.input_parquet"):
        resolve_input_parquet({"input_parquet": ""}, {})


def test_resolve_output_parquet_defaults_next_to_input():
    input_path = Path("/data/aggregated_segments_with_g2p.parquet")
    assert resolve_output_parquet(input_path, "") == _default_output(input_path)


def test_resolve_output_parquet_honors_explicit_path():
    input_path = Path("/data/aggregated_segments_with_g2p.parquet")
    assert resolve_output_parquet(input_path, "/out/custom.parquet") == Path("/out/custom.parquet")


def test_encode_phoneme_strings_maps_distinct_phonemes():
    phones = [
        phoneme_tokens_from_g2p("HH AH L"),
        phoneme_tokens_from_g2p("HH AH"),
    ]
    encoded, vocab = encode_phoneme_strings(phones)
    assert len(encoded[0]) == 3
    assert len(encoded[1]) == 2
    assert len(vocab) == 3


def test_top_k_hard_negatives_ranks_closest_neighbor_first():
    df = _synthetic_g2p_df()
    phones = [phoneme_tokens_from_g2p(g2p) for g2p in df["ngram_g2p"]]
    ngrams = df["ngram"].tolist()

    neighbors = top_k_hard_negatives(phones, ngrams, top_k=2, block_size=2, workers=1)

    hello_neighbors = neighbors[0]
    assert hello_neighbors[0]["ngram"] == "hello word"
    assert hello_neighbors[0]["distance"] >= hello_neighbors[-1]["distance"]
    assert all(entry["ngram"] != "hello world" for entry in hello_neighbors)


def test_top_k_hard_negatives_excludes_self_and_zero_distance():
    phones = [
        ["AH", "L"],
        ["AH", "L"],
        ["B", "AY"],
    ]
    ngrams = ["alpha", "beta", "gamma"]
    neighbors = top_k_hard_negatives(phones, ngrams, top_k=5, block_size=2, workers=1)

    assert neighbors[0][0]["ngram"] == "gamma"
    assert all(entry["ngram"] != "alpha" for entry in neighbors[0])
    assert all(entry["ngram"] != "beta" for entry in neighbors[0])
    assert neighbors[1][0]["ngram"] == "gamma"
    assert all(entry["ngram"] != "beta" for entry in neighbors[1])


def test_build_distances_column_parse_distances_and_parquet_roundtrip(tmp_path):
    df = _synthetic_g2p_df()
    out_df = build_distances_column(df, top_k=2, block_size=2, workers=1)

    for raw, ngram in zip(out_df["distances"], out_df["ngram"], strict=True):
        parsed = parse_distances(raw)
        assert parsed
        assert all("ngram" in entry for entry in parsed)
        assert all(entry["ngram"] != ngram for entry in parsed)

    parquet_path = tmp_path / "with_distances.parquet"
    out_df.to_parquet(parquet_path, index=False)
    loaded = pd.read_parquet(parquet_path)
    assert "distances" in loaded.columns
    assert len(loaded.loc[0, "distances"]) == 2


def test_prepare_stage2_paper_consumes_precomputed_distances(tmp_path):
    df = build_distances_column(_synthetic_g2p_df(), top_k=2, block_size=2, workers=1)
    hello_distances = parse_distances(df.loc[df["ngram"] == "hello world", "distances"].iloc[0])
    expected_ngrams = [entry["ngram"] for entry in hello_distances]

    processed = tmp_path / "processed" / "stage2_qbyt"
    paper_df, stats = convert_aggregated_to_paper_parquet(
        df,
        clips_dir=processed / "clips",
        distances_dir=processed / "distances",
        fbank_dir=tmp_path / "features" / "fbank",
        audio_by_rel={},
        limit_anchors=1,
    )

    assert stats["anchors"] == 1
    distances_file = paper_df.iloc[0]["distances_file"]
    saved = np.load(distances_file, allow_pickle=True)
    saved_ngrams = [entry["ngram"] for entry in saved]
    assert saved_ngrams == expected_ngrams
