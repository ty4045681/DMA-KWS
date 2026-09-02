from __future__ import annotations

import json

import pytest
from omegaconf import OmegaConf

import scripts.eval_phoneme_adapter_per as eval_phoneme_adapter_per
from dma_kws.inference.phoneme_per import (
    PhonemePerRunner,
    build_per_record,
    select_per_rows,
    summarize_per_results,
)


def _rows():
    return [
        {
            "audio_path": "/audio/positive.wav",
            "keyword": "hey snips",
            "label": 1,
            "text_variant": "hey snips",
        },
        {
            "audio_path": "/audio/negative.wav",
            "keyword": "hey snips",
            "label": 0,
            "text_variant": "turn on the lights",
        },
    ]


@pytest.mark.parametrize(
    "prep",
    [
        {"left_padding_ms": -1},
        {"right_padding_ms": -1},
    ],
)
def test_per_audio_padding_rejects_negative_values(prep):
    with pytest.raises(SystemExit, match="must be >= 0"):
        eval_phoneme_adapter_per._resolve_audio_padding_ms(prep)


def test_phoneme_per_runner_applies_waveform_padding(monkeypatch):
    torch = pytest.importorskip("torch")
    captured = {}

    monkeypatch.setattr(
        "dma_kws.inference.stage2_clip.load_audio",
        lambda _path, *, sample_rate: (torch.ones(1, 10), sample_rate),
    )

    def fake_waveform_to_fbank(waveform, *, sample_rate, **_kwargs):
        captured["num_samples"] = waveform.size(1)
        captured["sample_rate"] = sample_rate
        return torch.ones(2, 80)

    monkeypatch.setattr(
        "dma_kws.stage2.features.waveform_to_fbank",
        fake_waveform_to_fbank,
    )

    class PassthroughExtractor:
        @staticmethod
        def prepare_waveform(waveform, sample_rate):
            return waveform, sample_rate

    class FakeVerifier:
        fbank_extractor = PassthroughExtractor()
        fbank_kwargs = {
            "frame_length": 25,
            "frame_shift": 10,
            "snip_edges": True,
        }
        min_fbank_frames = 2

        @staticmethod
        def decode_phoneme_feats(feats):
            assert len(feats) == 1
            captured["decoded"] = True
            return [[2]]

    class FakeTokenizer:
        @staticmethod
        def ids2tokens(ids):
            assert ids in ([], [2])
            return ["HH"] if ids else []

    monkeypatch.setattr(
        "dma_kws.inference.phoneme_per.text_to_phonemes",
        lambda _g2p, _text: ["HH"],
    )
    monkeypatch.setattr(
        "dma_kws.inference.phoneme_per.tokenize_phoneme_string",
        lambda _tokenizer, _text: [2],
    )
    runner = PhonemePerRunner(
        verifier=FakeVerifier(),
        tokenizer=FakeTokenizer(),
        sample_rate=1000,
        g2p=object(),
    )

    default_results = runner.run_batch(
        _rows()[:1], reference_column="text_variant", num_workers=0
    )
    padded_results = runner.run_batch(
        _rows()[:1],
        reference_column="text_variant",
        num_workers=0,
        left_padding_ms=20,
        right_padding_ms=20,
    )

    assert default_results[0]["skipped"] is True
    assert padded_results[0]["skipped"] is False
    assert padded_results[0]["per"] == 0.0
    assert captured == {
        "num_samples": 50,
        "sample_rate": 1000,
        "decoded": True,
    }


@pytest.mark.parametrize(
    ("reference_column", "override_column"),
    [
        ("keyword", "keyword_phonemes"),
        ("text_variant", "text_variant_phonemes"),
    ],
)
def test_phoneme_per_runner_uses_per_row_reference_phoneme_overrides(
    monkeypatch,
    reference_column,
    override_column,
):
    torch = pytest.importorskip("torch")
    from dma_kws.inference import stage2_clip

    phone_ids = {
        "HH": 2,
        "EY1": 3,
        "IY1": 4,
        "V": 5,
        "AH0": 6,
        "EH1": 7,
    }
    ee_vah = ["HH", "EY1", "IY1", "V", "AH0"]
    ay_vah = ["HH", "EY1", "EY1", "V", "AH0"]
    default = ["HH", "EY1", "EH1", "V", "AH0"]
    expected_references = [ee_vah, ay_vah, default]
    expected_ids = [
        [phone_ids[phone] for phone in phonemes]
        for phonemes in expected_references
    ]
    g2p_calls = []

    class FakeDataset:
        def __init__(self, *, audio_paths, **_kwargs):
            self.audio_paths = list(audio_paths)

        def __len__(self):
            return len(self.audio_paths)

        def __getitem__(self, index):
            return index, torch.ones(2, 80), 1.0

    class FakeVerifier:
        fbank_extractor = object()
        fbank_kwargs = {}
        min_fbank_frames = 1

        @staticmethod
        def decode_phoneme_feats(feats):
            assert len(feats) == len(expected_ids)
            return expected_ids

    class FakeTokenizer:
        @staticmethod
        def tokenize(text):
            phonemes = text.split()
            return phonemes, [phone_ids[phone] for phone in phonemes]

        @staticmethod
        def ids2tokens(ids):
            id_phones = {value: key for key, value in phone_ids.items()}
            return [id_phones[token_id] for token_id in ids]

    def fake_text_to_phonemes(_g2p, text):
        g2p_calls.append(text)
        assert text == "hey eva"
        return default

    monkeypatch.setattr(stage2_clip, "ClipFeatureDataset", FakeDataset)
    monkeypatch.setattr(stage2_clip, "collate_clip_feature_batch", lambda batch: batch)
    monkeypatch.setattr(
        "dma_kws.inference.phoneme_per.text_to_phonemes",
        fake_text_to_phonemes,
    )
    runner = PhonemePerRunner(
        verifier=FakeVerifier(),
        tokenizer=FakeTokenizer(),
        sample_rate=16000,
        g2p=object(),
    )
    rows = [
        {
            "audio_path": "/audio/ee-vah.wav",
            "keyword": "hey eva",
            "label": 1,
            "text_variant": "hey eva",
            override_column: "HH EY1 IY1 V AH0",
        },
        {
            "audio_path": "/audio/ay-vah.wav",
            "keyword": "hey eva",
            "label": 1,
            "text_variant": "hey eva",
            override_column: ay_vah,
        },
        {
            "audio_path": "/audio/default.wav",
            "keyword": "hey eva",
            "label": 1,
            "text_variant": "hey eva",
            **({override_column: ""} if reference_column == "text_variant" else {}),
        },
    ]

    results = runner.run_batch(
        rows,
        reference_column=reference_column,
        batch_size=8,
        num_workers=0,
    )

    assert [result["reference_phonemes"] for result in results] == expected_references
    assert [result["per"] for result in results] == [0.0, 0.0, 0.0]
    assert g2p_calls == ["hey eva"]


@pytest.mark.parametrize(
    ("reference_column", "override_column"),
    [
        ("keyword", "keyword_phonemes"),
        ("text_variant", "text_variant_phonemes"),
    ],
)
def test_phoneme_per_runner_rejects_invalid_reference_phoneme_override(
    monkeypatch,
    reference_column,
    override_column,
):
    pytest.importorskip("torch")
    from dma_kws.inference import stage2_clip

    class FakeVerifier:
        fbank_extractor = object()
        fbank_kwargs = {}
        min_fbank_frames = 1

    class FakeTokenizer:
        @staticmethod
        def tokenize(_text):
            return ["HH"], [2]

    class UnexpectedDataset:
        def __init__(self, **_kwargs):
            raise AssertionError("audio loading must not start before override validation")

    monkeypatch.setattr(stage2_clip, "ClipFeatureDataset", UnexpectedDataset)
    monkeypatch.setattr(
        "dma_kws.inference.phoneme_per.text_to_phonemes",
        lambda _g2p, _text: ["HH"],
    )
    runner = PhonemePerRunner(
        verifier=FakeVerifier(),
        tokenizer=FakeTokenizer(),
        sample_rate=16000,
        g2p=object(),
    )
    row = {
        "audio_path": "/audio/invalid.wav",
        "keyword": "hey eva",
        "label": 1,
        "text_variant": "hey eva",
        override_column: "HH NOT_A_PHONE",
    }

    with pytest.raises(ValueError, match=f"Manifest row 1 {override_column}"):
        runner.run_batch([row], reference_column=reference_column, num_workers=0)


@pytest.mark.parametrize(
    "padding_overrides,expected_padding",
    [
        ({}, (0, 0)),
        ({"left_padding_ms": 0, "right_padding_ms": 240}, (0, 240)),
    ],
)
def test_per_eval_applies_and_records_padding(
    tmp_path,
    monkeypatch,
    padding_overrides,
    expected_padding,
):
    rows = _rows()[:1]
    captured = {}

    class FakeStreamPolicy:
        @staticmethod
        def describe():
            return {"mode": "test"}

    class FakeRunner:
        stream_policy = FakeStreamPolicy()

        def run_batch(self, batch_rows, **kwargs):
            captured["rows"] = batch_rows
            captured["kwargs"] = kwargs
            return [
                {
                    "audio_path": batch_rows[0]["audio_path"],
                    "keyword": batch_rows[0]["keyword"],
                    "reference_column": "text_variant",
                    "reference_text": batch_rows[0]["text_variant"],
                    "reference_phonemes": ["HH"],
                    "hypothesis_phonemes": ["HH"],
                    "edit_distance": 0,
                    "reference_length": 1,
                    "per": 0.0,
                    "skipped": False,
                }
            ]

    runner = FakeRunner()

    class FakeRunnerFactory:
        @staticmethod
        def from_config(_config, _prep, _device):
            return runner

    monkeypatch.setattr(
        eval_phoneme_adapter_per,
        "resolved_config",
        lambda _cfg: {
            "paths": {},
            "stage1": {},
            "stage2": {},
            "tokenizer": {},
        },
    )
    monkeypatch.setattr(
        eval_phoneme_adapter_per, "load_manifest", lambda _path: rows
    )
    monkeypatch.setattr(
        eval_phoneme_adapter_per,
        "resolve_accelerator",
        lambda _device: ("cpu", 1),
    )
    monkeypatch.setattr(
        eval_phoneme_adapter_per, "PhonemePerRunner", FakeRunnerFactory
    )
    cfg = OmegaConf.create(
        {
            "prep": {
                "manifest": "manifest.csv",
                "stage2_ckpt": "stage2.pt",
                "per_output_dir": str(tmp_path),
                "num_workers": 1,
                **padding_overrides,
            },
            "run": {"device": "cpu"},
        }
    )

    summary = eval_phoneme_adapter_per.run_eval(cfg)

    assert captured["rows"] == rows
    left_padding_ms, right_padding_ms = expected_padding
    expected_summary = {"left": left_padding_ms, "right": right_padding_ms}
    assert captured["kwargs"]["left_padding_ms"] == left_padding_ms
    assert captured["kwargs"]["right_padding_ms"] == right_padding_ms
    assert summary["audio_padding_ms"] == expected_summary
    saved_summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert saved_summary["audio_padding_ms"] == expected_summary


def test_keyword_reference_requires_positive_filter():
    with pytest.raises(ValueError, match="positive clips"):
        select_per_rows(_rows(), reference_column="keyword")

    with pytest.raises(ValueError, match="positive clips"):
        select_per_rows(_rows(), reference_column="keyword", label_filter=0)


def test_keyword_reference_selects_only_positive_rows():
    selected = select_per_rows(
        _rows(),
        reference_column="keyword",
        label_filter=1,
    )

    assert [row["audio_path"] for row in selected] == ["/audio/positive.wav"]


def test_text_variant_reference_supports_positive_and_negative_rows():
    selected = select_per_rows(_rows(), reference_column="text_variant")

    assert len(selected) == 2


def test_selected_row_must_have_reference_column():
    rows = [{"audio_path": "/audio/a.wav", "keyword": "hey snips", "label": 1}]

    with pytest.raises(ValueError, match="empty 'text_variant'"):
        select_per_rows(rows, reference_column="text_variant")


def test_csv_row_numbering_starts_after_header():
    rows = [
        {
            "audio_path": "/audio/a.wav",
            "keyword": "hey snips",
            "label": 1,
            "text_variant": "",
        }
    ]

    with pytest.raises(ValueError, match="Manifest row 2 has an empty 'text_variant'"):
        select_per_rows(
            rows,
            reference_column="text_variant",
            first_row_number=2,
        )


def test_direct_rows_with_invalid_label_report_row_number():
    rows = [
        {
            "audio_path": "/audio/a.wav",
            "keyword": "hey snips",
            "label": "pos",
            "text_variant": "hey snips",
        }
    ]

    with pytest.raises(ValueError, match="Manifest row 2 has invalid label 'pos'"):
        select_per_rows(
            rows,
            reference_column="text_variant",
            first_row_number=2,
        )


def test_build_per_record_invalid_label_reports_audio_context():
    row = {
        "audio_path": "/audio/a.wav",
        "keyword": "hey snips",
        "label": "pos",
        "text_variant": "hey snips",
    }

    with pytest.raises(
        ValueError,
        match="PER result for audio '/audio/a.wav' has invalid label 'pos'",
    ):
        build_per_record(
            row,
            reference_column="text_variant",
            reference_phonemes=["HH"],
            reference_ids=[2],
            hypothesis_phonemes=["HH"],
            hypothesis_ids=[2],
            skipped=False,
        )


def test_limit_is_applied_after_label_filter():
    rows = [
        {"audio_path": f"/{index}.wav", "keyword": "hey snips", "label": label}
        for index, label in enumerate([0, 1, 1])
    ]

    selected = select_per_rows(
        rows,
        reference_column="keyword",
        label_filter=1,
        limit=1,
    )

    assert selected[0]["audio_path"] == "/1.wav"


def test_per_summary_is_corpus_weighted_and_counts_skips():
    first = build_per_record(
        _rows()[0],
        reference_column="text_variant",
        reference_phonemes=["HH", "EY1"],
        reference_ids=[2, 3],
        hypothesis_phonemes=["HH"],
        hypothesis_ids=[2],
        skipped=False,
    )
    second = build_per_record(
        _rows()[1],
        reference_column="text_variant",
        reference_phonemes=["T", "ER1", "N"],
        reference_ids=[4, 5, 6],
        hypothesis_phonemes=[],
        hypothesis_ids=[],
        skipped=True,
    )

    summary = summarize_per_results([first, second])

    assert first["edit_distance"] == 1
    assert second["edit_distance"] == 3
    assert summary["total_edit_distance"] == 4
    assert summary["total_reference_phonemes"] == 5
    assert summary["per"] == pytest.approx(0.8)
    assert summary["num_skipped"] == 1
    assert summary["num_empty_hypotheses"] == 1
