import csv
from pathlib import Path

import numpy as np
import pytest
import torch

from dma_kws.stage2.adapt_dataset import (
    KeywordAdaptationDataset,
    MixedAdaptationDataset,
    load_adapt_manifest,
)
from dma_kws.stage2.adapt_paths import neg_slug_to_text, slugify, wav_to_fbank_mirror
from dma_kws.stage2.collate import train_collate_fn
from dma_kws.stage2.prepare_adapt import AdaptSample, split_train_eval, write_eval_manifest, write_train_manifest
from dma_kws.tokenizer import load_char_tokenizer


REPO_ROOT = Path(__file__).resolve().parents[1]
DICT_PATH = REPO_ROOT / "data" / "dict" / "lang_char.txt"


def test_slugify_and_neg_slug():
    assert slugify("hey eva") == "hey_eva"
    assert neg_slug_to_text("hey_ava") == "hey ava"


def test_wav_to_fbank_mirror():
    wav = Path("/tmp/data/adapt/hey_eva/raw/tts/positive/a.wav")
    fbank = wav_to_fbank_mirror(Path("/tmp/data/adapt/hey_eva/fbank"), wav)
    assert fbank.name == "a.npy"
    assert "tts/positive" in str(fbank)


def test_load_adapt_manifest(tmp_path: Path):
    manifest = tmp_path / "train.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["audio_path", "text", "label"])
        writer.writeheader()
        writer.writerow({"audio_path": "raw/tts/positive/a.wav", "text": "hey eva", "label": 1})
    rows = load_adapt_manifest(manifest)
    assert len(rows) == 1
    assert rows[0]["label"] == 1


def test_keyword_adaptation_dataset_seq_label(tmp_path: Path, monkeypatch):
    if not DICT_PATH.is_file():
        pytest.skip("dict not available")

    from dma_kws.stage2 import adapt_dataset
    from dma_kws.stage2.prepare_adapt import AdaptSample

    class _FakeG2p:
        def __call__(self, text: str):
            return [p.upper() for p in text.split()]

    monkeypatch.setattr(adapt_dataset, "text_to_phonemes", lambda _g2p, text: text.split())
    monkeypatch.setattr(adapt_dataset, "make_g2p", lambda: _FakeG2p())

    fbank_root = tmp_path / "fbank"
    pos_dir = fbank_root / "tts" / "positive"
    pos_dir.mkdir(parents=True)
    np.save(pos_dir / "pos.npy", np.zeros((4, 80), dtype=np.float32))

    manifest = tmp_path / "train.csv"
    write_train_manifest(
        manifest,
        [AdaptSample(audio_path="raw/tts/positive/pos.wav", text="hey eva", label=1, phase="tts")],
    )

    tokenizer = load_char_tokenizer(DICT_PATH)
    dataset = KeywordAdaptationDataset(
        manifest_path=manifest,
        keyword="hey eva",
        fbank_root=fbank_root,
        tokenizer=tokenizer,
        manifest_root=tmp_path,
    )
    pos = dataset[0]
    assert pos["label"].item() == 1
    assert pos["anchor_seq"].dtype == torch.long
    assert pos["seq_label"].numel() == pos["anchor_seq"].numel()
    batch = train_collate_fn([pos, pos])
    assert batch["anchor"].shape[0] == 2
    assert batch["seq_label_mask"].shape == batch["seq_label"].shape


def test_mixed_adaptation_sampling_ratio():
    if not DICT_PATH.is_file():
        pytest.skip("dict not available")

    class _TinyKeywordDataset:
        def __len__(self):
            return 10

        def __getitem__(self, index):
            return {
                "anchor_seq": torch.tensor([1, 2], dtype=torch.long),
                "feat": torch.zeros(3, 80),
                "label": torch.tensor(1, dtype=torch.long),
                "seq_label": torch.tensor([1, 0], dtype=torch.long),
            }

    class _TinyGenericDataset:
        def __len__(self):
            return 10

        def __getitem__(self, index):
            return {
                "anchor_seq": torch.tensor([3, 4], dtype=torch.long),
                "feat": torch.ones(3, 80),
                "label": torch.tensor(0, dtype=torch.long),
                "seq_label": torch.tensor([0, 1], dtype=torch.long),
            }

    mixed = MixedAdaptationDataset(
        keyword_dataset=_TinyKeywordDataset(),
        libri_dataset=_TinyGenericDataset(),
        sample_lens=1000,
        mix_ratio=0.5,
        seed=7,
    )
    keyword_hits = sum(1 for i in range(200) if mixed[i]["feat"].sum() == 0)
    assert 60 <= keyword_hits <= 140


def test_split_train_eval_and_eval_manifest(tmp_path: Path):
    samples = [
        AdaptSample(audio_path=f"a{i}.wav", text="hey eva", label=1, phase="tts")
        for i in range(10)
    ]
    train, eval_set = split_train_eval(samples, eval_fraction=0.2, seed=1)
    assert len(train) == 8
    assert len(eval_set) == 2
    eval_path = tmp_path / "eval.csv"
    write_eval_manifest(eval_path, eval_set, "hey eva")
    rows = list(csv.DictReader(eval_path.open(encoding="utf-8")))
    assert rows[0]["keyword"] == "hey eva"
