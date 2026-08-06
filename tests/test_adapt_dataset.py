import csv
from pathlib import Path

import numpy as np
import pytest
import torch

from dma_kws.stage2.adapt_dataset import (
    KeywordAdaptationDataset,
    MixedAdaptationDataset,
    clips_eval_manifest_from_adapt,
    load_adapt_manifest,
)
from dma_kws.stage2.adapt_paths import directory_name_to_text, neg_slug_to_text, slugify, wav_to_fbank_mirror
from dma_kws.stage2.collate import train_collate_fn
from dma_kws.stage2.prepare_adapt import (
    AdaptSample,
    load_manifest_csv,
    prepare_keyword_adaptation,
    scan_external_source,
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


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("Hi_Eva", "hi eva"),
        ("HI EVA", "hi eva"),
        ("hi-eva", "hi eva"),
        ("hi__  eva", "hi eva"),
    ],
)
def test_directory_name_to_text(name: str, expected: str):
    assert directory_name_to_text(name) == expected


def test_wav_to_fbank_mirror():
    wav = Path("/tmp/data/adapt/hey_eva/raw/tts/positive/a.wav")
    fbank = wav_to_fbank_mirror(Path("/tmp/data/adapt/hey_eva/fbank"), wav)
    assert fbank.name == "a.npy"
    assert "tts/positive" in str(fbank)

    eval_wav = Path("/tmp/data/adapt/hey_eva/raw/tts/eval/positive/b.wav")
    eval_fbank = wav_to_fbank_mirror(Path("/tmp/data/adapt/hey_eva/fbank"), eval_wav)
    assert eval_fbank.name == "b.npy"
    assert "tts/eval/positive" in str(eval_fbank)

    external_wav = Path("/tmp/external/Hi_Eva/sample.wav")
    external_fbank = wav_to_fbank_mirror(Path("/tmp/fbank"), external_wav)
    assert external_fbank == Path("/tmp/fbank/external/tmp/external/Hi_Eva/sample.npy")


def test_scan_external_source_maps_negative_directories(tmp_path: Path):
    positive_dir = tmp_path / "positive"
    negative_root = tmp_path / "negative"
    for wav_path in (
        positive_dir / "pos.wav",
        negative_root / "Hi_Eva" / "neg_one.wav",
        negative_root / "HI EVA" / "nested" / "neg_two.WAV",
    ):
        wav_path.parent.mkdir(parents=True, exist_ok=True)
        wav_path.touch()

    samples = scan_external_source(
        phase="tts",
        keyword="hey eva",
        positive_dir=positive_dir,
        negative_root=negative_root,
    )

    assert [(sample.label, sample.text) for sample in samples] == [
        (1, "hey eva"),
        (0, "hi eva"),
        (0, "hi eva"),
    ]
    assert all(Path(sample.audio_path).is_absolute() for sample in samples)


def test_prepare_external_sources_splits_each_phase(tmp_path: Path, monkeypatch):
    sources: dict[str, dict[str, str]] = {}
    for phase in ("tts", "real"):
        positive_dir = tmp_path / "external" / phase / "positive"
        negative_root = tmp_path / "external" / phase / "negative"
        for wav_path in (
            positive_dir / "pos_one.wav",
            positive_dir / "pos_two.wav",
            negative_root / "Hi_Eva" / "neg_one.wav",
            negative_root / "Hi_Eva" / "neg_two.wav",
        ):
            wav_path.parent.mkdir(parents=True, exist_ok=True)
            wav_path.touch()
        sources[phase] = {"positive_dir": str(positive_dir), "negative_root": str(negative_root)}

    monkeypatch.setattr("dma_kws.stage2.prepare_adapt.make_g2p", lambda: object())
    monkeypatch.setattr("dma_kws.stage2.prepare_adapt.validate_g2p", lambda _text, _g2p: None)
    monkeypatch.setattr("dma_kws.stage2.prepare_adapt.compute_and_save_fbank", lambda *_args, **_kwargs: True)

    events: list[tuple[str, int]] = []
    stats = prepare_keyword_adaptation(
        keyword="hey eva",
        data_root=tmp_path / "prepared",
        fbank_params={},
        eval_fraction=0.5,
        eval_seed=1,
        sources=sources,
        on_progress=lambda stage, value: events.append((stage, value)),
    )

    assert stats["phases"]["tts"]["train"] == 2
    assert stats["phases"]["tts"]["eval"] == 2
    assert stats["phases"]["real"]["train"] == 2
    assert stats["phases"]["real"]["eval"] == 2
    for phase in ("tts", "real"):
        phase_stats = stats["phases"][phase]
        assert phase_stats["train_positive"] + phase_stats["train_negative"] == phase_stats["train"]
        assert phase_stats["eval_positive"] + phase_stats["eval_negative"] == phase_stats["eval"]
    rows = list(csv.DictReader((tmp_path / "prepared" / "manifests" / "tts_train.csv").open()))
    assert all(Path(row["audio_path"]).is_absolute() for row in rows)
    assert {row["text"] for row in rows} <= {"hey eva", "hi eva"}

    totals = dict(event for event in events if event[0].endswith("_total") or event[0] == "scan")
    assert totals["scan"] == 8
    assert totals["fbank_total"] == 8
    # G2P runs once per unique text, not once per sample
    assert totals["g2p_total"] == stats["unique_texts"] == 2
    assert sum(value for stage, value in events if stage == "fbank") == 8
    assert sum(value for stage, value in events if stage == "g2p") == 2


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


def test_clips_eval_manifest_preserves_mining_metadata(tmp_path: Path):
    source = tmp_path / "real_eval.csv"
    with source.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["audio_path", "text", "label", "speaker_id", "split", "source"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "audio_path": "/audio/a.wav",
                "text": "hey ava",
                "label": 0,
                "speaker_id": "speaker-1",
                "split": "train",
                "source": "speechocean762",
            }
        )

    output = clips_eval_manifest_from_adapt(source, "hey eva", tmp_path / "clips.csv")
    rows = list(csv.DictReader(output.open(encoding="utf-8")))
    assert rows == [
        {
            "audio_path": "/audio/a.wav",
            "keyword": "hey eva",
            "label": "0",
            "text": "hey ava",
            "speaker_id": "speaker-1",
            "split": "train",
            "source": "speechocean762",
        }
    ]


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
    neg_dir = fbank_root / "tts" / "negative"
    neg_dir.mkdir(parents=True)
    np.save(neg_dir / "contains.npy", np.zeros((4, 80), dtype=np.float32))

    manifest = tmp_path / "train.csv"
    write_train_manifest(
        manifest,
        [
            AdaptSample(
                audio_path="raw/tts/positive/pos.wav",
                text="hey eva",
                label=1,
                phase="tts",
            ),
            AdaptSample(
                audio_path="raw/tts/negative/contains.wav",
                text="please hey eva",
                label=0,
                phase="tts",
            ),
        ],
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
    with pytest.raises(ValueError, match="phrases containing the full keyword"):
        dataset[1]


def test_keyword_adaptation_dataset_uses_explicit_eva_pronunciation(tmp_path: Path):
    if not DICT_PATH.is_file():
        pytest.skip("dict not available")

    fbank_root = tmp_path / "fbank"
    for relative in ("real/positive/eva.npy", "real/near_negative/ava.npy"):
        path = fbank_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, np.zeros((4, 80), dtype=np.float32))

    manifest = tmp_path / "real_train.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "audio_path",
                "text",
                "label",
                "keyword_phonemes",
                "text_variant_phonemes",
            ],
        )
        writer.writeheader()
        writer.writerows(
            [
                {
                    "audio_path": "raw/real/positive/eva.wav",
                    "text": "hey eva",
                    "label": 1,
                    "keyword_phonemes": "HH EY1 IY1 V AH0",
                    "text_variant_phonemes": "HH EY1 IY1 V AH0",
                },
                {
                    "audio_path": "raw/real/near_negative/ava.wav",
                    "text": "hey ava",
                    "label": 0,
                    "keyword_phonemes": "HH EY1 IY1 V AH0",
                    "text_variant_phonemes": "HH EY1 EY1 V AH0",
                },
            ]
        )

    dataset = KeywordAdaptationDataset(
        manifest_path=manifest,
        keyword="hey eva",
        fbank_root=fbank_root,
        tokenizer=load_char_tokenizer(DICT_PATH),
        manifest_root=tmp_path,
        g2p=lambda _text: [],
    )

    assert dataset[0]["seq_label"][-1].item() == 1
    assert dataset[1]["seq_label"][-1].item() == 0


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


def test_manifest_explicit_split_preserves_metadata(tmp_path: Path, monkeypatch):
    audio_paths = [tmp_path / "train.wav", tmp_path / "eval.wav"]
    for audio_path in audio_paths:
        audio_path.touch()

    source = tmp_path / "source.csv"
    with source.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "audio_path",
                "text",
                "label",
                "phase",
                "split",
                "speaker_id",
                "device",
                "session",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "audio_path": audio_paths[0],
                "text": "hey eva",
                "label": 1,
                "phase": "real",
                "split": "train",
                "speaker_id": "speaker_train",
                "device": "phone",
                "session": "s1",
            }
        )
        writer.writerow(
            {
                "audio_path": audio_paths[1],
                "text": "hey ava",
                "label": 0,
                "phase": "real",
                "split": "eval",
                "speaker_id": "speaker_eval",
                "device": "pc",
                "session": "s2",
            }
        )

    loaded = load_manifest_csv(source)
    assert loaded[0].split == "train"
    assert loaded[0].metadata == {
        "speaker_id": "speaker_train",
        "device": "phone",
        "session": "s1",
    }

    monkeypatch.setattr("dma_kws.stage2.prepare_adapt.make_g2p", lambda: object())
    monkeypatch.setattr("dma_kws.stage2.prepare_adapt.validate_g2p", lambda *_args: None)
    monkeypatch.setattr(
        "dma_kws.stage2.prepare_adapt.compute_and_save_fbank",
        lambda *_args, **_kwargs: True,
    )
    output_root = tmp_path / "prepared"
    stats = prepare_keyword_adaptation(
        keyword="hey eva",
        data_root=output_root,
        fbank_params={},
        manifest_csv=source,
        # Explicit splits must win over this deliberately extreme fallback.
        eval_fraction=1.0,
    )

    assert stats["phases"]["real"]["train"] == 1
    assert stats["phases"]["real"]["eval"] == 1
    train_rows = list(
        csv.DictReader((output_root / "manifests" / "real_train.csv").open())
    )
    eval_rows = list(
        csv.DictReader((output_root / "manifests" / "real_eval.csv").open())
    )
    assert train_rows[0]["speaker_id"] == "speaker_train"
    assert train_rows[0]["device"] == "phone"
    assert train_rows[0]["session"] == "s1"
    assert eval_rows[0]["speaker_id"] == "speaker_eval"
    assert eval_rows[0]["device"] == "pc"


def test_manifest_explicit_split_rejects_speaker_leakage(tmp_path: Path):
    source = tmp_path / "source.csv"
    with source.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "audio_path",
                "text",
                "label",
                "phase",
                "split",
                "speaker_id",
            ],
        )
        writer.writeheader()
        for split in ("train", "eval"):
            writer.writerow(
                {
                    "audio_path": tmp_path / f"{split}.wav",
                    "text": "hey eva",
                    "label": 1,
                    "phase": "real",
                    "split": split,
                    "speaker_id": "same_speaker",
                }
            )

    with pytest.raises(ValueError, match="leaks speaker_id"):
        prepare_keyword_adaptation(
            keyword="hey eva",
            data_root=tmp_path / "prepared",
            fbank_params={},
            manifest_csv=source,
        )
