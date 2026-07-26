from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from dma_kws.stage1.prepare_fbank import (
    compute_fbank_for_wav,
    prepare_manifest_fbank,
    resolve_record_fbank_path,
    resolve_stage1_fbank_path,
)
from dma_kws.stage1.wenet_ctc import Stage1Dataset, encode_manifest_target
from dma_kws.tokenizer import load_char_tokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
DICT_PATH = REPO_ROOT / "data" / "dict" / "lang_char.txt"


def test_resolve_stage1_fbank_path_relative_to_audio_root():
    audio_root = Path("/data/LibriSpeech")
    wav = audio_root / "train-clean-100/103/1240/103-1240-0000.flac"
    fbank = resolve_stage1_fbank_path("/data/features/stage1_fbank", wav, audio_root=audio_root)
    assert fbank == Path(
        "/data/features/stage1_fbank/train-clean-100/103/1240/103-1240-0000.npy"
    )


def test_resolve_stage1_fbank_path_basename_fallback():
    wav = Path("/anywhere/103-1240-0000.flac")
    fbank = resolve_stage1_fbank_path("/data/features/stage1_fbank", wav)
    assert fbank == Path("/data/features/stage1_fbank/103-1240-0000.npy")


def test_prepare_manifest_fbank_writes_npy_and_updates_manifest(tmp_path, monkeypatch):
    wav_path = tmp_path / "audio" / "utt.wav"
    wav_path.parent.mkdir(parents=True)
    wav_path.write_bytes(b"wav")

    manifest = tmp_path / "train.jsonl"
    manifest.write_text(
        '{"utt_id":"utt","wav_path":"'
        + str(wav_path)
        + '","phonemes_g2p":"HH AH0 L OW1"}\n',
        encoding="utf-8",
    )

    saved = {}

    def fake_compute(wav, out, **_kwargs):
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.save(out, np.full((3, 80), 1.5, dtype=np.float32))
        saved["wav"] = str(wav)
        saved["out"] = str(out)
        return str(out)

    fbank_root = tmp_path / "fbank"
    out_path, written, skipped = prepare_manifest_fbank(
        manifest,
        fbank_root=fbank_root,
        audio_root=tmp_path / "audio",
        compute_fn=fake_compute,
    )

    assert written == 1
    assert skipped == 0
    assert out_path == manifest
    assert Path(saved["out"]).exists()

    lines = manifest.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    assert '"fbank_path"' in lines[0]

    out_path2, written2, skipped2 = prepare_manifest_fbank(
        manifest,
        fbank_root=fbank_root,
        audio_root=tmp_path / "audio",
        compute_fn=fake_compute,
    )
    assert written2 == 0
    assert skipped2 == 1
    assert out_path2 == manifest


def test_stage1_dataset_loads_precomputed_fbank(tmp_path):
    fbank_path = tmp_path / "utt.npy"
    np.save(fbank_path, np.full((4, 80), 2.0, dtype=np.float32))

    manifest = tmp_path / "dev.jsonl"
    manifest.write_text(
        '{"wav_path":"/ignored.wav","fbank_path":"'
        + str(fbank_path)
        + '","phonemes_g2p":"HH AH0 L OW1"}\n',
        encoding="utf-8",
    )

    tok = load_char_tokenizer(DICT_PATH)
    dataset = Stage1Dataset(
        manifest,
        tokenizer=tok,
        sample_rate=16000,
        num_mel_bins=80,
    )
    sample = dataset[0]
    assert sample["feat"].shape == (4, 80)
    assert torch.allclose(sample["feat"], torch.full((4, 80), 2.0))


def test_stage1_dataset_derives_fbank_from_root(tmp_path):
    audio_root = tmp_path / "LibriSpeech"
    wav_path = audio_root / "dev-clean/1272/128104/1272-128104-0000.flac"
    wav_path.parent.mkdir(parents=True)
    wav_path.write_bytes(b"flac")

    fbank_root = tmp_path / "features" / "stage1_fbank"
    fbank_path = fbank_root / "dev-clean/1272/128104/1272-128104-0000.npy"
    fbank_path.parent.mkdir(parents=True)
    np.save(fbank_path, np.full((5, 80), 3.0, dtype=np.float32))

    manifest = tmp_path / "dev.jsonl"
    manifest.write_text(
        '{"wav_path":"'
        + str(wav_path)
        + '","phonemes_g2p":"HH AH0 L OW1"}\n',
        encoding="utf-8",
    )

    tok = load_char_tokenizer(DICT_PATH)
    dataset = Stage1Dataset(
        manifest,
        tokenizer=tok,
        sample_rate=16000,
        num_mel_bins=80,
        fbank_root=fbank_root,
        audio_root=audio_root,
    )
    sample = dataset[0]
    assert sample["feat"].shape == (5, 80)


def test_stage1_dataset_rejects_legacy_stress_stripped_manifest(tmp_path):
    # Stress-stripped phonemes are absent from the vocabulary, so CharTokenizer
    # would map every vowel to <unk> and Stage I would train on consonants only.
    manifest = tmp_path / "train.jsonl"
    manifest.write_text(
        '{"wav_path":"/ignored.wav","phonemes_g2p":"HH AH L OW"}\n',
        encoding="utf-8",
    )

    tok = load_char_tokenizer(DICT_PATH)
    with pytest.raises(ValueError, match="outside the vocabulary"):
        Stage1Dataset(manifest, tokenizer=tok, sample_rate=16000, num_mel_bins=80)


def test_stage1_dataset_rejects_legacy_phonemes_list(tmp_path):
    manifest = tmp_path / "train.jsonl"
    manifest.write_text(
        '{"wav_path":"/ignored.wav","phonemes":["HH","AH","L","OW"]}\n',
        encoding="utf-8",
    )

    tok = load_char_tokenizer(DICT_PATH)
    with pytest.raises(ValueError, match="prepare_stage1_librispeech"):
        Stage1Dataset(manifest, tokenizer=tok, sample_rate=16000, num_mel_bins=80)


def test_resolve_record_fbank_path_prefers_explicit_field(tmp_path):
    record = {"wav_path": "/a.wav", "fbank_path": str(tmp_path / "custom.npy")}
    assert resolve_record_fbank_path(record) == tmp_path / "custom.npy"


def test_compute_fbank_for_wav_roundtrip(tmp_path, monkeypatch):
    wav_path = tmp_path / "utt.wav"
    wav_path.write_bytes(b"wav")
    out_path = tmp_path / "utt.npy"

    fake_feat = torch.randn(7, 80)

    monkeypatch.setattr(
        "dma_kws.stage1.prepare_fbank.load_audio",
        lambda _path, sample_rate: (torch.zeros(1, 1600), sample_rate),
    )
    monkeypatch.setattr(
        "dma_kws.stage1.prepare_fbank.extract_fbank",
        lambda *_args, **_kwargs: fake_feat,
    )

    result = compute_fbank_for_wav(wav_path, out_path)
    assert result == str(out_path)
    loaded = np.load(out_path)
    assert loaded.shape == (7, 80)
