from pathlib import Path

from dma_kws.stage2.pairs import (
    clip_to_audio_rel,
    decoded_glob_for_dataset,
    infer_dataset_id,
    iter_decoded_audio_rows,
    resolve_data_root,
)


def test_clip_to_audio_rel_strips_lp100_prefix():
    assert clip_to_audio_rel("LP-100/missus_rachel/103-1240-0000_000.wav") == "missus_rachel/103-1240-0000_000.wav"


def test_clip_to_audio_rel_without_prefix_is_identity():
    assert clip_to_audio_rel("missus_rachel/103-1240-0000_000.wav") == "missus_rachel/103-1240-0000_000.wav"


def test_clip_to_audio_rel_strips_gp1000_prefix():
    assert clip_to_audio_rel("GP-1000/hello world/a.wav") == "hello world/a.wav"


def test_clip_to_audio_rel_strips_lp460_prefix():
    assert clip_to_audio_rel("LP-460/hello world/a.wav") == "hello world/a.wav"


def test_infer_dataset_id_from_clips():
    assert infer_dataset_id(["GP-1000/foo/bar.wav"]) == "GP-1000"
    assert infer_dataset_id(["LP-460/foo/bar.wav"]) == "LP-460"
    assert infer_dataset_id(["LP-100/foo/bar.wav"]) == "LP-100"
    assert infer_dataset_id(["foo/bar.wav"]) is None


def test_decoded_glob_for_dataset():
    assert decoded_glob_for_dataset("GP-1000") == "GP-1000-decoded-*.parquet"
    assert decoded_glob_for_dataset(None) == "LP-100-decoded-*.parquet"


def test_resolve_data_root_from_config():
    paths = {
        "libriphrase100_root": "/data/LibriPhrase-100",
        "gigaphrase1000_root": "/data/GigaPhrase-1000",
    }
    assert resolve_data_root(paths, "GP-1000") == Path("/data/GigaPhrase-1000")
    assert resolve_data_root(paths, "LP-100") == Path("/data/LibriPhrase-100")
    assert resolve_data_root(paths, None) is None


class _FakeDF:
    def __init__(self, rows):
        self._rows = rows

    def iterrows(self):
        for i, row in enumerate(self._rows):
            yield i, row


def test_iter_decoded_audio_rows_filters_to_needed_keys():
    shard_rows = {
        Path("0000.parquet"): [
            {"audio_rel": "a/x.wav", "audio": [0.1, 0.2], "sampling_rate": 16000},
            {"audio_rel": "a/y.wav", "audio": [0.3], "sampling_rate": 16000},
        ],
        Path("0001.parquet"): [
            {"audio_rel": "b/z.wav", "audio": [0.4, 0.5], "sampling_rate": 16000},
        ],
    }

    def fake_read(path):
        return _FakeDF(shard_rows[path])

    needed = {"a/x.wav", "b/z.wav"}
    got = list(
        iter_decoded_audio_rows(
            [Path("0000.parquet"), Path("0001.parquet")],
            needed,
            read_parquet=fake_read,
        )
    )

    assert [(rel, sr) for rel, _, sr in got] == [("a/x.wav", 16000), ("b/z.wav", 16000)]
    assert got[0][1] == [0.1, 0.2]


def test_iter_decoded_audio_rows_stops_after_all_keys_found():
    first = _FakeDF([
        {"audio_rel": "a/x.wav", "audio": [0.1], "sampling_rate": 16000},
    ])

    def fake_read(path):
        if str(path) == "0001.parquet":
            raise AssertionError("should not read second shard once all keys found")
        return first

    got = list(
        iter_decoded_audio_rows(
            [Path("0000.parquet"), Path("0001.parquet")],
            {"a/x.wav"},
            read_parquet=fake_read,
        )
    )
    assert [rel for rel, _, _ in got] == ["a/x.wav"]
