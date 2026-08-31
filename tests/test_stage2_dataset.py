import random
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
