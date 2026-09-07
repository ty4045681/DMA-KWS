import random
from collections import Counter

import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from dma_kws.stage2.dataset import LibriPhraseTrainDataset
from dma_kws.stage2.joint_dataset import JointAdaptationDataset, JointBatchSampler
from dma_kws.tokenizer import SEQ_LABEL_ORDERED_CONTIGUOUS_PREFIX


class _Keywords:
    def __init__(self, rows=None):
        self.df = pd.DataFrame(rows or [
            {"label": label, "speaker_id": speaker, "voice_id": speaker, "text": text}
            for label, text in [(1, "wake"), (0, "wait"), (0, "way")]
            for speaker in ["a", "b"]
        ])
        self._anchor_seq = [11, 12]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        label = int(self.df.iloc[index]["label"])
        return {
            "feat": torch.tensor([[float(index)]]),
            "anchor_seq": torch.tensor(self._anchor_seq),
            "query_seq": torch.tensor(self._anchor_seq if label else [99]),
            "label": torch.tensor(label),
            "seq_label": torch.tensor([label, label]),
        }


class _Replay:
    num_anchors = 100
    sample_lens = 3
    _background_probability = 0.4
    background_enabled = True

    def __len__(self):
        return self.sample_lens

    def sample_pair(self, index, kind, *, rng, anchor_seq=None):
        anchor = [100 + index] if anchor_seq is None else list(anchor_seq)
        positive = kind == "positive"
        query = anchor if positive else ([] if kind == "background" else [300])
        return {
            "feat": torch.tensor([[float(index), rng.random()]]),
            "anchor_seq": torch.tensor(anchor),
            "query_seq": torch.tensor(query, dtype=torch.long),
            "label": torch.tensor(int(positive)),
            "seq_label": torch.tensor([int(positive)] * len(anchor)),
        }


def _joint(**kwargs):
    return JointAdaptationDataset(_Keywords(), _Keywords(), _Replay(), **kwargs)


def test_joint_source_and_class_quotas_are_exact_over_full_cycle():
    dataset = _joint(sample_lens=2000)
    sampler = JointBatchSampler(dataset, batch_size=17)
    kinds, domains, labels = Counter(), Counter(), Counter()
    total = 0
    for batch in sampler:
        for ticket in batch:
            kinds[ticket[0]] += 1
            item = dataset[ticket]
            domains[item["domain_source"]] += 1
            labels[int(item["label"])] += 1
            assert item["source"] == int(item["domain_source"] in {1, 2})
        total += len(batch)
        # Fractional allocations carry over instead of rounding away the small
        # background strata in each 17-item batch.
        for kind, weight in dataset.weights.items():
            assert abs(kinds[kind] - total * weight) <= 1.0 + 1e-9
    for domain, weight in {0: 0.4, 1: 0.3, 2: 0.2, 3: 0.1}.items():
        assert abs(domains[domain] / total - weight) < 0.002
    assert abs(labels[0] - labels[1]) <= 2


def test_background_target_and_generic_queries_have_empty_transcripts():
    dataset = _joint()
    target = dataset[("background_target", 42)]
    generic = dataset[("background_generic", 42)]
    assert target["anchor_seq"].tolist() == [11, 12]
    assert generic["anchor_seq"].tolist() != [11, 12]
    for item in (target, generic):
        assert item["label"].item() == 0
        assert item["query_seq"].numel() == 0
        assert not item["seq_label"].any()


def test_replay_uses_full_anchor_pool_independently_of_epoch_length():
    dataset = _joint(sample_lens=1)
    indices = {int(dataset[("libri_positive", seed)]["feat"][0, 0]) for seed in range(100)}
    assert max(indices) >= 90
    assert len(indices) > 50


def test_real_speakers_and_tts_voices_and_negative_text_are_balanced():
    rows = [{"label": 1, "speaker_id": "a", "voice_id": "a", "text": "wake"}]
    rows += [{"label": 1, "speaker_id": "b", "voice_id": "b", "text": "wake"}] * 30
    rows += [{"label": 0, "speaker_id": "a", "voice_id": "a", "text": "wait"}]
    rows += [{"label": 0, "speaker_id": "a", "voice_id": "a", "text": "way"}] * 30
    rows += [{"label": 0, "speaker_id": "b", "voice_id": "b", "text": "wait"}]
    rows += [{"label": 0, "speaker_id": "b", "voice_id": "b", "text": "way"}] * 30
    keyword = _Keywords(rows)
    dataset = JointAdaptationDataset(keyword, keyword, _Replay())
    for kind in ["real_positive", "real_negative", "tts_positive", "tts_negative"]:
        speakers, combinations = Counter(), Counter()
        for seed in range(4000):
            item = dataset[(kind, seed)]
            row = keyword.df.iloc[int(item["feat"][0, 0])]
            speakers[row["speaker_id"]] += 1
            combinations[(row["voice_id"], row["text"])] += 1
        assert abs(speakers["a"] / 4000 - 0.5) < 0.025
        if kind == "tts_negative":
            assert all(abs(count / 4000 - 0.25) < 0.025 for count in combinations.values())


def test_sampler_restart_is_exact_and_next_epoch_changes_tickets():
    sampler = JointBatchSampler(_joint(sample_lens=640), batch_size=16, seed=7)
    full = list(sampler)
    sampler.set_epoch(0, start_batch=9)
    assert len(sampler) == 31
    assert list(sampler) == full[9:]
    sampler.set_epoch(1)
    second = list(sampler)
    assert second != full
    assert [Counter(k for k, _ in b) for b in second] == [Counter(k for k, _ in b) for b in full]


def test_short_epochs_preserve_long_term_quotas_and_exact_resume():
    dataset = _joint(sample_lens=34)
    sampler = JointBatchSampler(dataset, batch_size=17)
    counts = Counter()
    for epoch in range(200):
        sampler.set_epoch(epoch)
        batches = list(sampler)
        counts.update(kind for batch in batches for kind, _ in batch)
        sampler.set_epoch(epoch, start_batch=1)
        assert list(sampler) == batches[1:]
    assert counts == {kind: round(weight * 6800) for kind, weight in dataset.weights.items()}
    # Seeking across epochs is the same operation as continuing one large
    # epoch, including the ticket RNG stream and within-batch shuffle.
    reference = list(JointBatchSampler(_joint(sample_lens=6800), batch_size=17))
    sampler.set_epoch(199)
    assert list(sampler) == reference[-2:]


def test_launcher_rank_and_world_size_match_initialized_process_group(monkeypatch):
    monkeypatch.setenv("RANK", "1")
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    sampler = JointBatchSampler(_joint(sample_lens=640), batch_size=16)
    assert sampler.full_num_batches == 20
    before_init = list(sampler)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 1)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)
    assert sampler.full_num_batches == 20
    assert list(sampler) == before_init
    # Process-group state wins over stale launcher environment variables.
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "8")
    assert sampler.full_num_batches == 20
    assert list(sampler) == before_init


def test_ranks_receive_equal_quotas_and_disjoint_tickets(monkeypatch):
    sampler = JointBatchSampler(_joint(sample_lens=640), batch_size=16)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    rank_zero = list(sampler)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 1)
    rank_one = list(sampler)
    assert len(rank_zero) == len(rank_one) == 20
    assert [Counter(k for k, _ in b) for b in rank_zero] == [Counter(k for k, _ in b) for b in rank_one]
    assert {seed for b in rank_zero for _, seed in b}.isdisjoint(
        {seed for b in rank_one for _, seed in b}
    )


def _serialize_batch(batch):
    return [{key: value.tolist() if torch.is_tensor(value) else value for key, value in item.items()} for item in batch]


def test_tickets_produce_identical_samples_with_spawned_workers():
    dataset = _joint(sample_lens=64)
    direct = DataLoader(dataset, batch_sampler=JointBatchSampler(dataset, 16), collate_fn=_serialize_batch)
    workers = DataLoader(
        dataset,
        batch_sampler=JointBatchSampler(dataset, 16),
        collate_fn=_serialize_batch,
        num_workers=2,
        multiprocessing_context="spawn",
    )
    assert list(direct) == list(workers)


@pytest.mark.parametrize("kwargs", [
    {"mix_ratio": -0.1}, {"mix_ratio": float("nan")}, {"real_fraction": 1.1},
    {"background_keyword_fraction": float("inf")}, {"sample_lens": 0},
    {"sample_lens": True},
])
def test_reject_invalid_joint_config(kwargs):
    with pytest.raises(ValueError):
        _joint(**kwargs)


def test_reject_empty_missing_classes_and_inconsistent_anchors():
    real, tts, replay = _Keywords(), _Keywords(), _Replay()
    real.df = real.df.iloc[:0]
    with pytest.raises(ValueError, match="empty"):
        JointAdaptationDataset(real, tts, replay)
    real.df = tts.df[tts.df.label == 1]
    with pytest.raises(ValueError, match="positives and negatives"):
        JointAdaptationDataset(real, tts, replay)
    real = _Keywords()
    real._anchor_seq = [8]
    with pytest.raises(ValueError, match="same non-empty keyword"):
        JointAdaptationDataset(real, tts, replay)
    replay.num_anchors = 0
    with pytest.raises(ValueError, match="anchor pool is empty"):
        JointAdaptationDataset(tts, tts, replay)


def test_zero_weight_sources_are_never_requested():
    dataset = _joint(mix_ratio=1, real_fraction=1, sample_lens=40)
    counts = Counter(kind for batch in JointBatchSampler(dataset, 4) for kind, _ in batch)
    assert counts == {"real_positive": 20, "real_negative": 20}
    with pytest.raises(ValueError, match="disabled"):
        dataset[("background_target", 1)]


def test_too_short_epoch_and_invalid_resume_offset_are_rejected():
    with pytest.raises(ValueError, match="full batch per rank"):
        len(JointBatchSampler(_joint(sample_lens=4), 8))
    sampler = JointBatchSampler(_joint(sample_lens=16), 4)
    sampler.set_epoch(0, start_batch=5)
    with pytest.raises(ValueError, match="epoch-local"):
        list(sampler)


class _Tokenizer:
    def tokenize(self, text):
        tokens = text.split()
        return tokens, [{"HH": 1, "AH0": 2, "L": 3, "OW1": 4, "W": 5, "ER1": 6, "D": 7}[t] for t in tokens]


def _libri(monkeypatch, *, nested=False):
    dataset = LibriPhraseTrainDataset(
        wav_dir="/unused",
        tokenizer=_Tokenizer(),
        df=pd.DataFrame({
            "ngram": ["hello", "hello world" if nested else "world"],
            "ngram_g2p": ["HH AH0 L OW1", "HH AH0 L OW1 W ER1 L D" if nested else "W ER1 L D"],
            "clips_file": ["clips-1-a.npy", "clips-1-b.npy"],
            "distances_file": ["dist-0-a.npy", "dist-0-b.npy"],
        }),
        sample_lens=1,
        seed=0,
        seq_label_mode=SEQ_LABEL_ORDERED_CONTIGUOUS_PREFIX,
    )
    monkeypatch.setattr(dataset, "get_random_clips", lambda path: {"audio_path": path})
    monkeypatch.setattr(dataset, "get_random_distances", lambda path: None)
    monkeypatch.setattr(dataset, "_load_fbank", lambda path: torch.ones(2, 3))
    return dataset


def test_forced_libri_classes_preserve_legacy_rng_and_background_contract(monkeypatch):
    dataset = _libri(monkeypatch)
    state = dataset._rng.getstate()
    assert dataset.num_anchors == 2 and len(dataset) == 1
    assert dataset.sample_pair(0, "positive", rng=random.Random(0))["label"].item() == 1
    assert dataset.sample_pair(0, "negative", rng=random.Random(0))["label"].item() == 0
    assert dataset._rng.getstate() == state

    class Background:
        def extract(self, *, rng):
            # Both audio crop randomness and fbank dither must be local to the
            # ticket; perturbing the process RNG cannot affect the result.
            return torch.rand(2, 3) + rng.random()

    dataset._background_sampler = Background()
    assert dataset.background_enabled
    torch_state = torch.random.get_rng_state()
    first = dataset.sample_pair(0, "background", rng=random.Random(8), anchor_seq=[7, 8])
    assert torch.equal(torch.random.get_rng_state(), torch_state)
    torch.rand(4)
    second = dataset.sample_pair(0, "background", rng=random.Random(8), anchor_seq=[7, 8])
    assert torch.equal(first["feat"], second["feat"])
    assert first["anchor_seq"].tolist() == [7, 8]
    assert first["query_seq"].numel() == 0
    assert first["seq_label"].tolist() == [0, 0]
    assert first["label"].item() == 0
    assert dataset._rng.getstate() == state


def test_forced_negative_raises_for_containing_pool_but_legacy_relabels(monkeypatch):
    dataset = _libri(monkeypatch, nested=True)
    state = dataset._rng.getstate()
    with pytest.raises(ValueError, match="valid LibriPhrase speech negative"):
        dataset.sample_pair(0, "negative", rng=random.Random(0))
    assert dataset._rng.getstate() == state
    assert dataset[0]["label"].item() == 1


def test_forced_pair_rejects_disabled_background_and_wrong_override(monkeypatch):
    dataset = _libri(monkeypatch)
    with pytest.raises(ValueError, match="without a background sampler"):
        dataset.sample_pair(0, "background", rng=random.Random(0))
    with pytest.raises(ValueError, match="only allowed for background"):
        dataset.sample_pair(0, "positive", rng=random.Random(0), anchor_seq=[1])
    with pytest.raises(ValueError, match="Unknown LibriPhrase pair kind"):
        dataset.sample_pair(0, "other", rng=random.Random(0))
