from pathlib import Path

import numpy as np
import pytest

from dma_kws.stage2.pairs import AnchorExample, PairRecord
from scripts import prepare_stage2_libriphrase as prep


class _FakeDF:
    def __init__(self, rows):
        self._rows = rows

    @property
    def columns(self):
        keys = []
        for row in self._rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        return keys

    def iterrows(self):
        for i, row in enumerate(self._rows):
            yield i, row


def test_materialize_pairs_writes_npy_and_rewrites_paths(tmp_path):
    audio_dir = tmp_path / "stage2_qbyt" / "audio"
    pairs = [
        PairRecord(
            anchor_text="hi",
            anchor_phonemes=["HH", "AY"],
            wav_path="LP-100/hi/1-2-3_000.wav",
            label=1,
            sample_rate=16000,
        ),
        PairRecord(
            anchor_text="hi",
            anchor_phonemes=["HH", "AY"],
            wav_path="LP-100/bye/4-5-6_000.wav",
            label=0,
            sample_rate=16000,
        ),
    ]
    rows = [
        {"audio_rel": "hi/1-2-3_000.wav", "audio": [0.1, 0.2, 0.3], "sampling_rate": 16000},
        {"audio_rel": "bye/4-5-6_000.wav", "audio": [0.4, 0.5], "sampling_rate": 16000},
    ]

    def fake_read(path):
        return _FakeDF(rows)

    rewritten, unmatched = prep.materialize_pairs(
        pairs,
        decoded_parquet_paths=[Path("0000.parquet")],
        audio_dir=audio_dir,
        read_parquet=fake_read,
    )

    assert unmatched == 0
    assert rewritten[0].wav_path == "audio/hi/1-2-3_000.wav.npy"
    saved = np.load(audio_dir / "hi" / "1-2-3_000.wav.npy")
    assert saved.dtype == np.float32
    np.testing.assert_allclose(saved, np.array([0.1, 0.2, 0.3], dtype=np.float32))


def test_materialize_pairs_counts_unmatched(tmp_path):
    audio_dir = tmp_path / "stage2_qbyt" / "audio"
    pairs = [
        PairRecord(
            anchor_text="hi",
            anchor_phonemes=["HH", "AY"],
            wav_path="LP-100/missing/9-9-9_000.wav",
            label=1,
            sample_rate=16000,
        )
    ]

    def fake_read(path):
        return _FakeDF([])

    rewritten, unmatched = prep.materialize_pairs(
        pairs,
        decoded_parquet_paths=[Path("0000.parquet")],
        audio_dir=audio_dir,
        read_parquet=fake_read,
    )

    assert unmatched == 1
    assert rewritten == []


def test_find_decoded_parquets_raises_when_none(tmp_path):
    with pytest.raises(SystemExit):
        prep.find_decoded_parquets(tmp_path)


def test_materialize_pairs_rejects_non_mono_audio(tmp_path):
    audio_dir = tmp_path / "stage2_qbyt" / "audio"
    pairs = [
        PairRecord(
            anchor_text="hi",
            anchor_phonemes=["HH", "AY"],
            wav_path="LP-100/hi/1-2-3_000.wav",
            label=1,
            sample_rate=16000,
        )
    ]
    rows = [
        {"audio_rel": "hi/1-2-3_000.wav", "audio": [[0.1, 0.2], [0.3, 0.4]], "sampling_rate": 16000},
    ]

    def fake_read(path):
        return _FakeDF(rows)

    with pytest.raises(SystemExit):
        prep.materialize_pairs(
            pairs,
            decoded_parquet_paths=[Path("0000.parquet")],
            audio_dir=audio_dir,
            read_parquet=fake_read,
        )


def _anchor(text):
    return AnchorExample(text=text, phonemes=["AH"], clips=[f"LP-100/{text}/0.wav"])


def test_split_anchors_disjoint_and_deterministic():
    anchors = [_anchor(f"phrase{i}") for i in range(10)]
    train, dev = prep.split_anchors(anchors, holdout_fraction=0.3, seed=2025)

    assert len(dev) == 3
    assert len(train) == 7
    train_texts = {a.text for a in train}
    dev_texts = {a.text for a in dev}
    assert train_texts.isdisjoint(dev_texts)
    assert train_texts | dev_texts == {a.text for a in anchors}
    # Deterministic for a fixed seed.
    train2, dev2 = prep.split_anchors(anchors, holdout_fraction=0.3, seed=2025)
    assert [a.text for a in dev2] == [a.text for a in dev]


def test_split_anchors_too_few_yields_empty_dev():
    anchors = [_anchor("only")]
    train, dev = prep.split_anchors(anchors, holdout_fraction=0.1, seed=2025)
    assert dev == []
    assert len(train) == 1


def test_split_anchors_zero_fraction_keeps_all():
    anchors = [_anchor(f"p{i}") for i in range(5)]
    train, dev = prep.split_anchors(anchors, holdout_fraction=0.0, seed=2025)
    assert dev == []
    assert len(train) == 5


class _FakeParquet:
    """Stand-in for pandas with the few attributes the script touches."""

    def __init__(self, frames_by_path, decoded_rows):
        self._frames_by_path = frames_by_path
        self._decoded_rows = decoded_rows

    def read_parquet(self, path):
        key = Path(path).name
        if key in self._frames_by_path:
            return _FakeDF(self._frames_by_path[key])
        return _FakeDF(self._decoded_rows)


def _anchor_rows(texts):
    rows = []
    for text in texts:
        rows.append(
            {
                "ngram": text,
                "ngram_g2p": "AH",
                "clips": [{"audio_path": f"LP-100/{text}/0.wav"}],
            }
        )
    return rows


def _decoded_rows(texts):
    return [
        {"audio_rel": f"{text}/0.wav", "audio": [0.1, 0.2], "sampling_rate": 16000}
        for text in texts
    ]


def _run_main(monkeypatch, tmp_path, *, extra_argv, frames_by_path, decoded_texts, config):
    import sys as _sys

    config_path = tmp_path / "config.yaml"
    monkeypatch.setattr(prep, "load_config", lambda _p: config)

    fake = _FakeParquet(frames_by_path, _decoded_rows(decoded_texts))
    monkeypatch.setitem(_sys.modules, "pandas", fake)
    # materialize_pairs reads decoded shards via the module-level default.
    monkeypatch.setattr(prep, "_read_parquet", fake.read_parquet)
    monkeypatch.setattr(prep, "find_decoded_parquets", lambda _root: [tmp_path / "decoded.parquet"])

    argv = ["prog", "--config", str(config_path), "--input-parquet", str(tmp_path / "train.parquet")]
    argv += extra_argv
    monkeypatch.setattr(_sys, "argv", argv)
    prep.main()


def _read_jsonl_texts(path):
    import json

    texts = []
    with path.open() as handle:
        for line in handle:
            texts.append(json.loads(line)["anchor_text"])
    return texts


def test_main_holdout_split_writes_disjoint_train_dev(monkeypatch, tmp_path):
    processed = tmp_path / "processed"
    config = {
        "paths": {
            "libriphrase100_root": str(tmp_path / "lp"),
            "processed_root": str(processed),
        },
        "stage2": {"dev": {"holdout_anchor_fraction": 0.5, "seed": 2025}},
    }
    texts = [f"phrase{i}" for i in range(4)]
    _run_main(
        monkeypatch,
        tmp_path,
        extra_argv=["--negatives-per-anchor", "0"],
        frames_by_path={"train.parquet": _anchor_rows(texts)},
        decoded_texts=texts,
        config=config,
    )

    out = processed / "stage2_qbyt"
    train_texts = set(_read_jsonl_texts(out / "train.jsonl"))
    dev_texts = set(_read_jsonl_texts(out / "dev.jsonl"))
    assert train_texts
    assert dev_texts
    assert train_texts.isdisjoint(dev_texts)
    assert train_texts | dev_texts == set(texts)


def test_main_eval_parquet_builds_dev_from_separate_source(monkeypatch, tmp_path):
    processed = tmp_path / "processed"
    config = {
        "paths": {
            "libriphrase100_root": str(tmp_path / "lp"),
            "processed_root": str(processed),
        },
        "stage2": {"dev": {"holdout_anchor_fraction": 0.1, "seed": 2025}},
    }
    train_texts = ["train_a", "train_b"]
    dev_texts = ["dev_x", "dev_y"]
    _run_main(
        monkeypatch,
        tmp_path,
        extra_argv=["--negatives-per-anchor", "0", "--eval-parquet", str(tmp_path / "eval.parquet")],
        frames_by_path={
            "train.parquet": _anchor_rows(train_texts),
            "eval.parquet": _anchor_rows(dev_texts),
        },
        decoded_texts=train_texts + dev_texts,
        config=config,
    )

    out = processed / "stage2_qbyt"
    assert set(_read_jsonl_texts(out / "train.jsonl")) == set(train_texts)
    assert set(_read_jsonl_texts(out / "dev.jsonl")) == set(dev_texts)


def test_main_too_few_anchors_writes_no_dev(monkeypatch, tmp_path):
    processed = tmp_path / "processed"
    config = {
        "paths": {
            "libriphrase100_root": str(tmp_path / "lp"),
            "processed_root": str(processed),
        },
        "stage2": {"dev": {"holdout_anchor_fraction": 0.1, "seed": 2025}},
    }
    texts = ["only_one"]
    _run_main(
        monkeypatch,
        tmp_path,
        extra_argv=["--negatives-per-anchor", "0"],
        frames_by_path={"train.parquet": _anchor_rows(texts)},
        decoded_texts=texts,
        config=config,
    )

    out = processed / "stage2_qbyt"
    assert (out / "train.jsonl").exists()
    assert not (out / "dev.jsonl").exists()
