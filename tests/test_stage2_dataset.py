import random
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from dma_kws.stage2.collate import train_collate_fn
from dma_kws.stage2.dataset import LibriPhraseTrainDataset, _resolve_fbank_path
from dma_kws.tokenizer import load_char_tokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
DICT_PATH = REPO_ROOT / "data" / "dict" / "lang_char.txt"


def _mock_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ngram": ["hello", "world"],
            "ngram_g2p": ["HH AH0 L OW1", "W ER1 L D"],
            "clips_file": ["clips-2-a.npy", "clips-2-b.npy"],
            "distances_file": ["dist-0-a.npy", "dist-2-b.npy"],
        }
    )


class _FakeTokenizer:
    def tokenize(self, text: str):
        tokens = text.split()
        ids = [hash(token) % 100 + 1 for token in tokens]
        return tokens, ids


@pytest.fixture
def mock_npy_loader(monkeypatch):
    clips = {
        "clips-2-a.npy": np.array(
            [{"audio_path": "LP-460/hello/a.wav"}, {"audio_path": "LP-460/hello/b.wav"}],
            dtype=object,
        ),
        "clips-2-b.npy": np.array(
            [{"audio_path": "LP-460/world/c.wav"}, {"audio_path": "LP-460/world/d.wav"}],
            dtype=object,
        ),
    }
    distances = {
        "dist-0-a.npy": np.array([], dtype=object),
        "dist-2-b.npy": np.array([{"ngram": "hello"}, {"ngram": "hello"}], dtype=object),
    }
    fbank = np.ones((5, 80), dtype=np.float32)

    def fake_load(path, allow_pickle=False):
        name = Path(path).name
        if name in clips:
            return clips[name]
        if name in distances:
            return distances[name]
        if name.endswith(".npy") and "fbank" in str(path):
            return fbank
        raise FileNotFoundError(path)

    monkeypatch.setattr("dma_kws.stage2.dataset.np.load", fake_load)
    return fbank


def test_resolve_fbank_path_replaces_prefix_and_extension():
    path = _resolve_fbank_path("/data/segments", "LP-460/hello/a.wav")
    assert path == "/data/segments/LP-460-fbank/hello/a.npy"

    path = _resolve_fbank_path("/data/segments", "GP-1000/world/b.wav")
    assert path == "/data/segments/GP-1000-fbank/world/b.npy"


def test_dataset_getitem_returns_expected_keys_and_seq_label_length(mock_npy_loader):
    dataset = LibriPhraseTrainDataset(
        wav_dir="/data/segments",
        tokenizer=_FakeTokenizer(),
        df=_mock_dataframe(),
        sample_lens=4,
        seed=0,
    )

    sample = dataset[0]

    assert set(sample.keys()) == {"anchor_seq", "query_seq", "feat", "label", "seq_label"}
    assert sample["anchor_seq"].dtype == torch.long
    assert sample["query_seq"].dtype == torch.long
    assert sample["feat"].shape == (5, 80)
    assert sample["label"].dtype == torch.long
    assert sample["seq_label"].dtype == torch.long
    assert sample["seq_label"].numel() == sample["anchor_seq"].numel()


def test_positive_pair_query_seq_matches_anchor(mock_npy_loader):
    """``query_seq`` describes the clip in ``feat``, so a positive pair repeats
    the anchor while a negative pair must not. Supervising the auxiliary CTC loss
    with the anchor instead would teach the trunk the wrong transcript on every
    negative."""
    dataset = LibriPhraseTrainDataset(
        wav_dir="/data/segments",
        tokenizer=_FakeTokenizer(),
        df=_mock_dataframe(),
        sample_lens=1,
        seed=1,
    )

    sample = dataset[0]

    assert sample["label"].item() == 1
    assert sample["query_seq"].tolist() == sample["anchor_seq"].tolist()


def test_dataset_positive_sample_has_matching_seq_label(mock_npy_loader):
    dataset = LibriPhraseTrainDataset(
        wav_dir="/data/segments",
        tokenizer=_FakeTokenizer(),
        df=_mock_dataframe(),
        sample_lens=1,
        seed=1,
    )

    sample = dataset[0]

    assert sample["label"].item() == 1
    assert torch.all(sample["seq_label"] == 1)


def test_dataset_rejects_unknown_seq_label_mode(mock_npy_loader):
    with pytest.raises(ValueError, match="Unsupported seq label mode"):
        LibriPhraseTrainDataset(
            wav_dir="/data/segments",
            tokenizer=_FakeTokenizer(),
            df=_mock_dataframe(),
            sample_lens=1,
            seq_label_mode="levenshtein",
        )


def test_dataset_rejects_stress_stripped_ngram_g2p(mock_npy_loader):
    """A parquet predating the stress-marked vocabulary tokenizes every vowel to
    ``<unk>``, which would then be both a QbyT phone row and a CTC target. The
    tokenizer cannot tell, so the dataset has to refuse it at load time."""
    df = _mock_dataframe()
    df.loc[0, "ngram_g2p"] = "HH AH L OW1"

    with pytest.raises(ValueError, match=r"outside the vocabulary: AH\b") as excinfo:
        LibriPhraseTrainDataset(
            wav_dir="/data/segments",
            tokenizer=_FakeTokenizer(),
            df=df,
            sample_lens=1,
        )

    message = str(excinfo.value)
    assert "in-memory dataframe" in message
    assert "'hello'" in message
    assert "force_g2p_recompute" in message


@pytest.mark.parametrize("ngram_g2p", ["", "   "])
def test_dataset_rejects_empty_ngram_g2p(mock_npy_loader, ngram_g2p):
    df = _mock_dataframe()
    df.loc[1, "ngram_g2p"] = ngram_g2p

    with pytest.raises(ValueError, match="'world' has an empty ngram_g2p"):
        LibriPhraseTrainDataset(
            wav_dir="/data/segments",
            tokenizer=_FakeTokenizer(),
            df=df,
            sample_lens=1,
        )


def test_dataset_names_the_parquet_when_rejecting_ngram_g2p(tmp_path):
    df = _mock_dataframe()
    df.loc[0, "ngram_g2p"] = "HH AH L OW"
    parquet_file = tmp_path / "legacy.parquet"
    df.to_parquet(parquet_file, index=False)

    with pytest.raises(ValueError, match="legacy.parquet") as excinfo:
        LibriPhraseTrainDataset(
            wav_dir="/data/segments",
            tokenizer=_FakeTokenizer(),
            parquet_file=parquet_file,
            sample_lens=1,
        )

    assert "AH, OW" in str(excinfo.value)


def test_containing_ngram_is_never_returned_as_a_negative(mock_npy_loader):
    df = pd.DataFrame(
        {
            "ngram": ["google", "hey google"],
            "ngram_g2p": ["G UW1 G AH0 L", "HH EY1 G UW1 G AH0 L"],
            "clips_file": ["clips-2-a.npy", "clips-2-b.npy"],
            "distances_file": ["dist-0-a.npy", "dist-2-b.npy"],
        }
    )
    dataset = LibriPhraseTrainDataset(
        wav_dir="/data/segments",
        tokenizer=_FakeTokenizer(),
        df=df,
        sample_lens=1,
        seed=0,
    )

    sample = dataset[0]

    # seed=0 takes the negative branch. The only candidate contains the full
    # anchor, so after bounded re-draws it must be relabeled as an occurrence.
    assert sample["label"].item() == 1
    assert sample["seq_label"].tolist() == [1] * sample["anchor_seq"].numel()


def test_dataset_collate_batch_shapes(mock_npy_loader):
    dataset = LibriPhraseTrainDataset(
        wav_dir="/data/segments",
        tokenizer=_FakeTokenizer(),
        df=_mock_dataframe(),
        sample_lens=4,
        seed=0,
    )
    batch = [dataset[i] for i in range(2)]
    collated = train_collate_fn(batch)

    assert collated["anchor"].shape[0] == 2
    assert collated["feat"].shape[0] == 2
    assert collated["feat_lengths"].shape == (2,)
    assert collated["label"].shape == (2,)
    assert collated["seq_label"].shape[0] == 2
    assert collated["seq_label_mask"].shape == collated["seq_label"].shape


def test_dataset_loads_tokenizer_from_dict_path(mock_npy_loader):
    dataset = LibriPhraseTrainDataset(
        wav_dir="/data/segments",
        dict_path=DICT_PATH,
        df=_mock_dataframe(),
        sample_lens=1,
        seed=0,
    )

    sample = dataset[0]
    anchor_ids = load_char_tokenizer(DICT_PATH)
    _, expected = anchor_ids.tokenize("HH AH0 L OW1")

    assert sample["anchor_seq"].tolist() == expected


def test_worker_init_fn_reseeds_dataset_rng_per_worker(monkeypatch):
    """Each DataLoader worker must draw a different random stream.

    The dataset is forked into every worker with its ``random.Random(seed)``
    already constructed, so without a ``worker_init_fn`` all workers replay the
    identical sequence of positive/negative decisions, hard-negative picks and
    clip choices. The bug is invisible in any loss curve: batches still look
    varied because each one comes from a single worker.
    """
    from dma_kws.stage2.dataset import stage2_worker_init_fn

    dataset = LibriPhraseTrainDataset(
        wav_dir="/data/segments",
        tokenizer=_FakeTokenizer(),
        df=_mock_dataframe(),
        sample_lens=4,
        seed=2025,
    )

    class _FakeWorkerInfo:
        def __init__(self, seed: int) -> None:
            self.dataset = dataset
            self.seed = seed

    drawn: list[list[float]] = []
    for worker_seed in (11, 22, 33):
        monkeypatch.setattr(
            "dma_kws.stage2.dataset.torch.utils.data.get_worker_info",
            lambda seed=worker_seed: _FakeWorkerInfo(seed),
        )
        stage2_worker_init_fn(0)
        drawn.append([dataset._rng.random() for _ in range(5)])

    assert drawn[0] != drawn[1]
    assert drawn[1] != drawn[2]


def test_worker_init_fn_reseeds_same_worker_differently_per_ddp_rank(monkeypatch):
    from dma_kws.stage2.dataset import stage2_worker_init_fn

    dataset = LibriPhraseTrainDataset(
        wav_dir="/data/segments",
        tokenizer=_FakeTokenizer(),
        df=_mock_dataframe(),
        sample_lens=4,
        seed=2025,
    )

    class _FakeWorkerInfo:
        def __init__(self) -> None:
            self.dataset = dataset
            self.seed = 123

    monkeypatch.setattr(
        "dma_kws.stage2.dataset.torch.utils.data.get_worker_info",
        lambda: _FakeWorkerInfo(),
    )
    draws = []
    for rank in ("0", "1"):
        monkeypatch.setenv("RANK", rank)
        stage2_worker_init_fn(0)
        draws.append([dataset._rng.random() for _ in range(5)])

    assert draws[0] != draws[1]


def test_worker_init_fn_is_a_noop_outside_workers(monkeypatch):
    """num_workers=0 runs in the main process, where the configured seed stands."""
    from dma_kws.stage2.dataset import stage2_worker_init_fn

    dataset = LibriPhraseTrainDataset(
        wav_dir="/data/segments",
        tokenizer=_FakeTokenizer(),
        df=_mock_dataframe(),
        sample_lens=4,
        seed=2025,
    )
    expected = random.Random(2025).random()

    monkeypatch.setattr(
        "dma_kws.stage2.dataset.torch.utils.data.get_worker_info", lambda: None
    )
    stage2_worker_init_fn(0)

    assert dataset._rng.random() == expected


def test_dataset_can_replace_a_speech_negative_with_pure_background(
    mock_npy_loader,
    monkeypatch,
):
    constructed = {}

    class _FakeBackgroundSampler:
        def __init__(self, **kwargs):
            constructed.update(kwargs)

        def extract(self, *, rng):
            assert isinstance(rng, random.Random)
            return torch.full((7, 80), 9.0)

    monkeypatch.setattr(
        "dma_kws.stage2.features.TrainingBackgroundSampler",
        _FakeBackgroundSampler,
    )
    dataset = LibriPhraseTrainDataset(
        wav_dir="/data/segments",
        tokenizer=_FakeTokenizer(),
        df=_mock_dataframe(),
        sample_lens=1,
        seed=0,
        background_negative={
            "enabled": True,
            "probability": 1.0,
            "audio_list_path": "/background/musan.list",
            "duration_seconds_min": 1.25,
            "duration_seconds_max": 2.5,
        },
        fbank_kwargs={"num_mel_bins": 80, "dither": 0.0},
    )

    sample = dataset[0]

    # seed=0 enters the negative half; probability=1 replaces that speech
    # negative while leaving the current anchor/query classification contract.
    assert sample["label"].item() == 0
    assert sample["query_seq"].numel() == 0
    assert sample["seq_label"].tolist() == [0] * sample["anchor_seq"].numel()
    assert sample["feat"].shape == (7, 80)
    assert torch.all(sample["feat"] == 9.0)
    assert constructed == {
        "audio_list_path": "/background/musan.list",
        "duration_seconds_min": 1.25,
        "duration_seconds_max": 2.5,
        "fbank_kwargs": {"num_mel_bins": 80, "dither": 0.0},
    }


def test_background_sampling_never_replaces_positive_half(
    mock_npy_loader,
    monkeypatch,
):
    class _FakeBackgroundSampler:
        def __init__(self, **_kwargs):
            pass

        def extract(self, *, rng):
            raise AssertionError("positive draws must not sample background")

    monkeypatch.setattr(
        "dma_kws.stage2.features.TrainingBackgroundSampler",
        _FakeBackgroundSampler,
    )
    dataset = LibriPhraseTrainDataset(
        wav_dir="/data/segments",
        tokenizer=_FakeTokenizer(),
        df=_mock_dataframe(),
        sample_lens=1,
        seed=1,
        background_negative={
            "enabled": True,
            "probability": 1.0,
            "audio_list_path": "/background/musan.list",
        },
    )

    sample = dataset[0]

    assert sample["label"].item() == 1
    assert sample["query_seq"].tolist() == sample["anchor_seq"].tolist()


@pytest.mark.parametrize("probability", [-0.01, 1.01, float("nan")])
def test_background_sampling_rejects_invalid_probability(monkeypatch, probability):
    class _MustNotBeConstructed:
        def __init__(self, **_kwargs):
            raise AssertionError("probability must be checked before source I/O")

    monkeypatch.setattr(
        "dma_kws.stage2.features.TrainingBackgroundSampler",
        _MustNotBeConstructed,
    )
    with pytest.raises(ValueError, match="probability must be between 0 and 1"):
        LibriPhraseTrainDataset(
            wav_dir="/data/segments",
            tokenizer=_FakeTokenizer(),
            df=_mock_dataframe(),
            background_negative={
                "enabled": True,
                "probability": probability,
                "audio_list_path": "/background/musan.list",
            },
        )


def test_background_sampling_rejects_unknown_configuration_field():
    with pytest.raises(ValueError, match="Unknown stage2.background_negative fields"):
        LibriPhraseTrainDataset(
            wav_dir="/data/segments",
            tokenizer=_FakeTokenizer(),
            df=_mock_dataframe(),
            background_negative={"enabled": False, "typo": True},
        )


def _three_anchor_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ngram": ["hello", "world", "google"],
            "ngram_g2p": ["HH AH0 L OW1", "W ER1 L D", "G UW1 G AH0 L"],
            "clips_file": ["clips-2-a.npy", "clips-2-b.npy", "clips-2-c.npy"],
            "distances_file": ["dist-0-a.npy", "dist-0-b.npy", "dist-0-c.npy"],
        }
    )


def _one_anchor_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ngram": ["hello"],
            "ngram_g2p": ["HH AH0 L OW1"],
            "clips_file": ["clips-2-a.npy"],
            "distances_file": ["dist-0-a.npy"],
        }
    )


def _patch_dataset_np_load(monkeypatch, extra=None):
    arrays = {
        "clips-2-a.npy": np.array(
            [{"audio_path": "LP-460/hello/a.wav"}, {"audio_path": "LP-460/hello/b.wav"}],
            dtype=object,
        ),
        "clips-2-b.npy": np.array(
            [{"audio_path": "LP-460/world/c.wav"}, {"audio_path": "LP-460/world/d.wav"}],
            dtype=object,
        ),
        "clips-2-c.npy": np.array(
            [{"audio_path": "LP-460/google/e.wav"}, {"audio_path": "LP-460/google/f.wav"}],
            dtype=object,
        ),
        "dist-0-a.npy": np.array([], dtype=object),
        "dist-0-b.npy": np.array([], dtype=object),
        "dist-0-c.npy": np.array([], dtype=object),
        "dist-2-b.npy": np.array([{"ngram": "hello"}, {"ngram": "hello"}], dtype=object),
    }
    if extra:
        arrays.update(extra)
    fbank = np.ones((5, 80), dtype=np.float32)
    calls: list[str] = []

    def fake_load(path, allow_pickle=False):
        path_s = str(path)
        calls.append(path_s)
        name = Path(path_s).name
        if name in arrays:
            return arrays[name]
        if name.endswith(".npy") and "fbank" in path_s:
            return fbank
        raise FileNotFoundError(path)

    monkeypatch.setattr("dma_kws.stage2.dataset.np.load", fake_load)
    return calls


def _train_dataset(df, *, seed=0, metadata_cache=None, sample_lens=1, **kwargs):
    ctor = dict(
        wav_dir="/data/segments",
        tokenizer=kwargs.pop("tokenizer", _FakeTokenizer()),
        df=df,
        sample_lens=sample_lens,
        seed=seed,
    )
    ctor.update(kwargs)
    if metadata_cache is not None:
        ctor["metadata_cache"] = metadata_cache
    return LibriPhraseTrainDataset(**ctor)


def test_get_negative_matches_randrange_skip_self_oracle(monkeypatch):
    _patch_dataset_np_load(monkeypatch)
    seed = 2025
    excluded = 1
    df = _three_anchor_dataframe()
    dataset = _train_dataset(df, seed=seed)
    oracle = random.Random(seed)
    n = 3
    observed = []
    expected = []
    for _ in range(32):
        _wav, _g2p, ngram = dataset.get_negative(excluded)
        observed.append(ngram)
        r = oracle.randrange(n - 1)
        idx = r + int(r >= excluded)
        expected.append(df["ngram"].iloc[idx])
        clip_count = int(Path(df["clips_file"].iloc[idx]).name.split("-")[-2])
        oracle.randint(0, clip_count - 1)
    assert observed == expected
    assert "world" not in observed


def test_get_negative_is_uniform_over_everyone_except_self(monkeypatch):
    _patch_dataset_np_load(monkeypatch)
    dataset = _train_dataset(_three_anchor_dataframe(), seed=7)
    counts = Counter()
    for _ in range(1000):
        _wav, _g2p, ngram = dataset.get_negative(0)
        counts[ngram] += 1
    assert "hello" not in counts
    assert counts["world"] + counts["google"] == 1000
    assert 400 <= counts["world"] <= 600
    assert 400 <= counts["google"] <= 600


def test_get_negative_with_two_anchors_always_selects_the_other(mock_npy_loader):
    dataset = _train_dataset(_mock_dataframe(), seed=0)
    for excluded, other in ((0, "world"), (1, "hello")):
        for _ in range(20):
            _wav, _g2p, ngram = dataset.get_negative(excluded)
            assert ngram == other


def test_single_anchor_dataset_allows_positives_but_rejects_regular_negatives(
    monkeypatch,
):
    _patch_dataset_np_load(monkeypatch)
    df = _one_anchor_dataframe()
    positive = _train_dataset(df, seed=1)
    sample = positive[0]
    assert sample["label"].item() == 1
    with pytest.raises(ValueError, match="fewer than 2 anchors"):
        positive.get_negative(0)
    negative_draw = _train_dataset(df, seed=0)
    with pytest.raises(ValueError, match="fewer than 2 anchors"):
        negative_draw[0]


def test_metadata_cache_bool_max_entries_is_rejected(mock_npy_loader):
    with pytest.raises(ValueError, match="max_entries"):
        _train_dataset(_mock_dataframe(), metadata_cache={"max_entries": True})


def test_metadata_cache_unknown_key_is_rejected(mock_npy_loader):
    with pytest.raises(ValueError, match="Unknown stage2.metadata_cache fields"):
        _train_dataset(_mock_dataframe(), metadata_cache={"max_entries": 1, "typo": 1})


def test_cache_on_and_off_draw_identical_clips_and_labels(monkeypatch):
    calls = _patch_dataset_np_load(monkeypatch)

    def trace(metadata_cache):
        calls.clear()
        dataset = _train_dataset(
            _mock_dataframe(),
            seed=11,
            sample_lens=8,
            metadata_cache=metadata_cache,
        )
        samples = []
        for index in range(8):
            sample = dataset[index]
            samples.append(
                (
                    sample["label"].item(),
                    tuple(sample["anchor_seq"].tolist()),
                    tuple(sample["query_seq"].tolist()),
                    tuple(sample["seq_label"].tolist()),
                )
            )
        fbank_paths = [path for path in calls if "fbank" in path]
        clip_choices = [
            dataset.get_random_clips("clips-2-a.npy")["audio_path"] for _ in range(6)
        ]
        return samples, fbank_paths, clip_choices

    off = trace({"max_entries": 0, "max_bytes": 1_000_000})
    on = trace({"max_entries": 8, "max_bytes": 1_000_000})
    assert on == off


def test_repeated_clips_file_does_not_reload_when_cache_is_on(monkeypatch):
    calls = _patch_dataset_np_load(monkeypatch)
    dataset = _train_dataset(
        _mock_dataframe(),
        metadata_cache={"max_entries": 8, "max_bytes": 1_000_000},
    )
    first = dataset.get_random_clips("clips-2-a.npy")
    n_loads = sum(Path(path).name == "clips-2-a.npy" for path in calls)
    second = dataset.get_random_clips("clips-2-a.npy")
    assert "audio_path" in first and "audio_path" in second
    assert sum(Path(path).name == "clips-2-a.npy" for path in calls) == n_loads


def test_omitted_metadata_cache_stays_on_uncached_path(monkeypatch):
    calls = _patch_dataset_np_load(monkeypatch)
    dataset = _train_dataset(_mock_dataframe())
    dataset.get_random_clips("clips-2-a.npy")
    dataset.get_random_clips("clips-2-a.npy")
    assert sum(Path(path).name == "clips-2-a.npy" for path in calls) == 2


def test_empty_metadata_cache_dict_enables_defaults(monkeypatch):
    calls = _patch_dataset_np_load(monkeypatch)
    dataset = _train_dataset(_mock_dataframe(), metadata_cache={})
    dataset.get_random_clips("clips-2-a.npy")
    dataset.get_random_clips("clips-2-a.npy")
    assert sum(Path(path).name == "clips-2-a.npy" for path in calls) == 1


def test_lru_eviction_reloads_file_and_still_returns_correct_clip(monkeypatch):
    calls = _patch_dataset_np_load(monkeypatch)
    dataset = _train_dataset(
        _mock_dataframe(),
        metadata_cache={"max_entries": 1, "max_bytes": 10_000_000},
    )
    first_a = dataset.get_random_clips("clips-2-a.npy")
    dataset.get_random_clips("clips-2-b.npy")
    n_a = sum(Path(path).name == "clips-2-a.npy" for path in calls)
    second_a = dataset.get_random_clips("clips-2-a.npy")
    assert first_a["audio_path"].startswith("LP-460/hello/")
    assert second_a["audio_path"].startswith("LP-460/hello/")
    assert sum(Path(path).name == "clips-2-a.npy" for path in calls) == n_a + 1


def test_oversized_payload_is_used_but_not_retained(monkeypatch):
    payload = np.array(
        [{"audio_path": "LP-460/hello/a.wav", "blob": "x" * 50_000}],
        dtype=object,
    )
    calls = _patch_dataset_np_load(monkeypatch, extra={"clips-1-big.npy": payload})
    df = pd.DataFrame(
        {
            "ngram": ["hello"],
            "ngram_g2p": ["HH AH0 L OW1"],
            "clips_file": ["clips-1-big.npy"],
            "distances_file": ["dist-0-a.npy"],
        }
    )
    dataset = _train_dataset(
        df,
        metadata_cache={"max_entries": 8, "max_bytes": 1024},
    )
    first = dataset.get_random_clips("clips-1-big.npy")
    second = dataset.get_random_clips("clips-1-big.npy")
    assert first["audio_path"] == "LP-460/hello/a.wav"
    assert second["audio_path"] == "LP-460/hello/a.wav"
    assert sum(Path(path).name == "clips-1-big.npy" for path in calls) == 2


def test_object_payload_counts_against_byte_budget_not_just_nbytes(monkeypatch):
    blob = "z" * 20_000
    payload = np.array(
        [{"audio_path": "LP-460/hello/a.wav", "blob": blob}],
        dtype=object,
    )
    assert payload.nbytes < 64
    calls = _patch_dataset_np_load(monkeypatch, extra={"clips-1-obj.npy": payload})
    df = pd.DataFrame(
        {
            "ngram": ["hello"],
            "ngram_g2p": ["HH AH0 L OW1"],
            "clips_file": ["clips-1-obj.npy"],
            "distances_file": ["dist-0-a.npy"],
        }
    )
    dataset = _train_dataset(
        df,
        metadata_cache={"max_entries": 8, "max_bytes": 512},
    )
    dataset.get_random_clips("clips-1-obj.npy")
    dataset.get_random_clips("clips-1-obj.npy")
    assert sum(Path(path).name == "clips-1-obj.npy" for path in calls) == 2


def test_containing_ngram_redraw_still_runs_with_metadata_cache(monkeypatch):
    _patch_dataset_np_load(monkeypatch)
    df = pd.DataFrame(
        {
            "ngram": ["google", "hey google"],
            "ngram_g2p": ["G UW1 G AH0 L", "HH EY1 G UW1 G AH0 L"],
            "clips_file": ["clips-2-a.npy", "clips-2-b.npy"],
            "distances_file": ["dist-0-a.npy", "dist-2-b.npy"],
        }
    )
    dataset = _train_dataset(
        df,
        seed=0,
        metadata_cache={"max_entries": 8, "max_bytes": 1_000_000},
    )
    sample = dataset[0]
    assert sample["label"].item() == 1
    assert sample["seq_label"].tolist() == [1] * sample["anchor_seq"].numel()


def test_hot_path_does_not_iloc_materialized_columns(mock_npy_loader):
    dataset = _train_dataset(_mock_dataframe(), seed=0, sample_lens=4)
    dataset.df.drop(columns=["ngram_g2p", "clips_file", "distances_file"], inplace=True)
    sample = dataset[0]
    _wav, g2p, ngram = dataset.get_negative(0)
    assert sample["anchor_seq"].numel() > 0
    assert ngram == "world"
    assert g2p == "W ER1 L D"
    hard = dataset.get_hard_negative({"ngram": "hello"})
    assert hard[2] == "hello"


def test_phoneme_token_cache_shares_lru_budget_with_npy(monkeypatch):
    calls = _patch_dataset_np_load(monkeypatch)

    class _CountingTokenizer(_FakeTokenizer):
        def __init__(self):
            self.calls: list[str] = []

        def tokenize(self, text: str):
            self.calls.append(text)
            return super().tokenize(text)

    tokenizer = _CountingTokenizer()
    dataset = _train_dataset(
        _mock_dataframe(),
        tokenizer=tokenizer,
        metadata_cache={"max_entries": 1, "max_bytes": 10_000_000},
    )
    dataset.get_random_clips("clips-2-a.npy")
    n_clip_loads = sum(Path(path).name == "clips-2-a.npy" for path in calls)
    dataset.get_random_clips("clips-2-a.npy")
    assert sum(Path(path).name == "clips-2-a.npy" for path in calls) == n_clip_loads
    tokens = dataset._tokenize_phoneme_string("HH AH0 L OW1")
    assert tokenizer.calls.count("HH AH0 L OW1") == 1
    again = dataset._tokenize_phoneme_string("HH AH0 L OW1")
    assert again == tokens
    assert tokenizer.calls.count("HH AH0 L OW1") == 1
    dataset.get_random_clips("clips-2-a.npy")
    assert sum(Path(path).name == "clips-2-a.npy" for path in calls) == n_clip_loads + 1


def test_cache_hit_still_consumes_randint(monkeypatch):
    _patch_dataset_np_load(monkeypatch)
    on = _train_dataset(
        _mock_dataframe(),
        seed=3,
        metadata_cache={"max_entries": 8, "max_bytes": 1_000_000},
    )
    off = _train_dataset(
        _mock_dataframe(),
        seed=3,
        metadata_cache={"max_entries": 0, "max_bytes": 1_000_000},
    )
    on_paths = [on.get_random_clips("clips-2-a.npy")["audio_path"] for _ in range(10)]
    off_paths = [off.get_random_clips("clips-2-a.npy")["audio_path"] for _ in range(10)]
    assert on_paths == off_paths
    assert len(set(on_paths)) > 1
