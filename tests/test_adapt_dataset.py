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
from dma_kws.stage2.prepare_adapt import (
    AdaptSample,
    prepare_keyword_adaptation,
    scan_raw_tree,
    split_train_eval,
    write_eval_manifest,
    write_train_manifest,
)
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

    eval_wav = Path("/tmp/data/adapt/hey_eva/raw/tts/eval/positive/b.wav")
    eval_fbank = wav_to_fbank_mirror(Path("/tmp/data/adapt/hey_eva/fbank"), eval_wav)
    assert eval_fbank.name == "b.npy"
    assert "tts/eval/positive" in str(eval_fbank)


def test_scan_raw_tree_assigns_explicit_splits(tmp_path: Path):
    wav_paths = [
        tmp_path / "raw" / "tts" / "positive" / "train_pos.wav",
        tmp_path / "raw" / "tts" / "negative" / "hey_ava" / "train_neg.wav",
        tmp_path / "raw" / "tts" / "eval" / "positive" / "eval_pos.wav",
        tmp_path / "raw" / "tts" / "eval" / "negative" / "hey_eve" / "eval_neg.wav",
    ]
    for wav_path in wav_paths:
        wav_path.parent.mkdir(parents=True, exist_ok=True)
        wav_path.touch()

    samples = {
        Path(sample.audio_path).name: sample
        for sample in scan_raw_tree(tmp_path, "hey eva")
    }

    assert samples["train_pos.wav"].split == "train"
    assert samples["train_pos.wav"].label == 1
    assert samples["train_pos.wav"].text == "hey eva"
    assert samples["train_neg.wav"].split == "train"
    assert samples["train_neg.wav"].label == 0
    assert samples["train_neg.wav"].text == "hey ava"
    assert samples["eval_pos.wav"].split == "eval"
    assert samples["eval_neg.wav"].split == "eval"
    assert samples["eval_neg.wav"].text == "hey eve"


def test_prepare_uses_explicit_eval_directory(tmp_path: Path, monkeypatch):
    train_wav = tmp_path / "raw" / "tts" / "positive" / "train.wav"
    eval_wav = tmp_path / "raw" / "tts" / "eval" / "negative" / "hey_ava" / "eval.wav"
    for wav_path in (train_wav, eval_wav):
        wav_path.parent.mkdir(parents=True, exist_ok=True)
        wav_path.touch()

    monkeypatch.setattr("dma_kws.stage2.prepare_adapt.make_g2p", lambda: object())
    monkeypatch.setattr("dma_kws.stage2.prepare_adapt.validate_g2p", lambda _text, _g2p: None)
    monkeypatch.setattr(
        "dma_kws.stage2.prepare_adapt.compute_and_save_fbank",
        lambda *_args, **_kwargs: True,
    )

    stats = prepare_keyword_adaptation(
        keyword="hey eva",
        data_root=tmp_path,
        fbank_params={},
        eval_fraction=1.0,
    )
    train_rows = list(csv.DictReader((tmp_path / "manifests" / "tts_train.csv").open()))
    eval_rows = list(csv.DictReader((tmp_path / "manifests" / "tts_eval.csv").open()))

    assert stats["phases"]["tts"]["train"] == 1
    assert stats["phases"]["tts"]["eval"] == 1
    assert train_rows == [
        {"audio_path": "raw/tts/positive/train.wav", "text": "hey eva", "label": "1"}
    ]
    assert eval_rows == [
        {
            "audio_path": "raw/tts/eval/negative/hey_ava/eval.wav",
            "text": "hey ava",
            "keyword": "hey eva",
            "label": "0",
        }
    ]


@pytest.mark.parametrize(
    ("relative_path", "error"),
    [
        ("raw/tts/positive/train.wav", "No evaluation wav files found for phase 'tts'"),
        ("raw/tts/eval/positive/eval.wav", "No training wav files found for phase 'tts'"),
    ],
)
def test_prepare_requires_explicit_train_and_eval_wavs(
    tmp_path: Path,
    relative_path: str,
    error: str,
):
    wav_path = tmp_path / relative_path
    wav_path.parent.mkdir(parents=True)
    wav_path.touch()

    with pytest.raises(ValueError, match=error):
        prepare_keyword_adaptation(keyword="hey eva", data_root=tmp_path, fbank_params={})


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
