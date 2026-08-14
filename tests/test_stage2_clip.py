from __future__ import annotations

import math

import pytest

from dma_kws.inference.stage2_clip import (
    ClipFeatureDataset,
    Stage2ClipRunner,
    collate_clip_feature_batch,
)
from dma_kws.tokenizer import load_char_tokenizer


def _fake_phonemes(text: str) -> list[str]:
    """Mimic g2p_en: upper-case ARPAbet with stress digits on vowels."""
    phones = {"hey eva": ["HH", "EY1", "IY1", "V", "AH0"]}
    return phones.get(text.lower(), text.upper().split())


def test_clip_feature_loading_api_is_public():
    assert ClipFeatureDataset.__name__ == "ClipFeatureDataset"
    batch = [(0, "feature", 1.0)]
    assert collate_clip_feature_batch(batch) is batch


def test_clip_feature_dataset_padding_precedes_min_frame_guard(monkeypatch):
    torch = pytest.importorskip("torch")
    source = torch.tensor([[0.25, -0.5, 0.75]])
    captured = {}

    monkeypatch.setattr(
        "dma_kws.inference.stage2_clip.load_audio",
        lambda _path, *, sample_rate: (source, sample_rate),
    )

    def fake_has_min_fbank_frames(num_samples, **_kwargs):
        captured["num_samples"] = num_samples
        return True

    monkeypatch.setattr(
        "dma_kws.inference.audio_utils.has_min_fbank_frames",
        fake_has_min_fbank_frames,
    )

    def fake_waveform_to_fbank(waveform, *, sample_rate, **_kwargs):
        captured["waveform"] = waveform.clone()
        captured["sample_rate"] = sample_rate
        return torch.ones(1, 80)

    monkeypatch.setattr(
        "dma_kws.stage2.features.waveform_to_fbank",
        fake_waveform_to_fbank,
    )

    class ResamplingExtractor:
        @staticmethod
        def prepare_waveform(waveform, sample_rate):
            return waveform.repeat_interleave(2, dim=1), sample_rate * 2

    dataset = ClipFeatureDataset(
        audio_paths=["clip.wav"],
        sample_rate=1000,
        fbank_extractor=ResamplingExtractor(),
        fbank_kwargs={
            "frame_length": 25,
            "frame_shift": 10,
            "snip_edges": True,
        },
        min_fbank_frames=1,
        left_padding_ms=2,
        right_padding_ms=3,
    )

    index, feat, end_sec = dataset[0]

    assert index == 0
    assert feat.shape == (1, 80)
    assert end_sec == pytest.approx(0.003)
    assert captured["num_samples"] == 16
    assert captured["sample_rate"] == 2000
    expected = torch.tensor(
        [[0.0] * 4 + [0.25, 0.25, -0.5, -0.5, 0.75, 0.75] + [0.0] * 6]
    )
    assert torch.equal(captured["waveform"], expected)
    assert torch.equal(source, torch.tensor([[0.25, -0.5, 0.75]]))


def test_clip_feature_dataset_transforms_prepared_audio_before_padding(monkeypatch):
    torch = pytest.importorskip("torch")
    source = torch.tensor([[1.0, 2.0]])
    captured = {}

    monkeypatch.setattr(
        "dma_kws.inference.stage2_clip.load_audio",
        lambda _path, *, sample_rate: (source, sample_rate),
    )
    monkeypatch.setattr(
        "dma_kws.inference.audio_utils.has_min_fbank_frames",
        lambda *_args, **_kwargs: True,
    )

    def fake_waveform_to_fbank(waveform, *, sample_rate, **_kwargs):
        captured["fbank_waveform"] = waveform.clone()
        captured["fbank_sample_rate"] = sample_rate
        return torch.ones(1, 80)

    monkeypatch.setattr(
        "dma_kws.stage2.features.waveform_to_fbank",
        fake_waveform_to_fbank,
    )

    class PreparingExtractor:
        @staticmethod
        def prepare_waveform(waveform, sample_rate):
            return waveform + 1.0, sample_rate * 2

    def transform(index, waveform, sample_rate):
        captured["transform"] = (index, waveform.clone(), sample_rate)
        return waveform * 10.0

    dataset = ClipFeatureDataset(
        audio_paths=["clip.wav"],
        sample_rate=1000,
        fbank_extractor=PreparingExtractor(),
        fbank_kwargs={
            "frame_length": 25,
            "frame_shift": 10,
            "snip_edges": True,
        },
        min_fbank_frames=1,
        left_padding_ms=1,
        right_padding_ms=1,
        waveform_transform=transform,
    )

    index, feat, end_sec = dataset[0]

    assert index == 0
    assert feat.shape == (1, 80)
    assert end_sec == pytest.approx(0.002)
    transform_index, transform_waveform, transform_sample_rate = captured["transform"]
    assert transform_index == 0
    assert transform_sample_rate == 2000
    assert torch.equal(transform_waveform, torch.tensor([[2.0, 3.0]]))
    assert captured["fbank_sample_rate"] == 2000
    assert torch.equal(
        captured["fbank_waveform"],
        torch.tensor([[0.0, 0.0, 20.0, 30.0, 0.0, 0.0]]),
    )


def test_clip_feature_dataset_padding_can_satisfy_encoder_length_guard(monkeypatch):
    torch = pytest.importorskip("torch")
    source = torch.ones(1, 10)
    captured = {}

    monkeypatch.setattr(
        "dma_kws.inference.stage2_clip.load_audio",
        lambda _path, *, sample_rate: (source, sample_rate),
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

    dataset = ClipFeatureDataset(
        audio_paths=["short.wav"],
        sample_rate=1000,
        fbank_extractor=PassthroughExtractor(),
        fbank_kwargs={
            "frame_length": 25,
            "frame_shift": 10,
            "snip_edges": True,
        },
        min_fbank_frames=2,
        left_padding_ms=20,
        right_padding_ms=20,
    )

    index, feat, end_sec = dataset[0]

    assert index == 0
    assert feat.shape == (2, 80)
    assert end_sec == pytest.approx(0.01)
    assert captured == {"num_samples": 50, "sample_rate": 1000}


@pytest.mark.parametrize("left_padding_ms,right_padding_ms", [(-1, 0), (0, -1)])
def test_clip_feature_dataset_rejects_negative_padding(
    left_padding_ms, right_padding_ms
):
    with pytest.raises(ValueError, match="must be >= 0"):
        ClipFeatureDataset(
            audio_paths=[],
            sample_rate=16000,
            fbank_extractor=None,
            fbank_kwargs={},
            min_fbank_frames=1,
            left_padding_ms=left_padding_ms,
            right_padding_ms=right_padding_ms,
        )


def _fake_g2p():
    return _fake_phonemes


class FakeWaveform:
    def __init__(self, num_samples: int) -> None:
        self._num_samples = num_samples

    def size(self, dim: int) -> int:
        if dim == 1:
            return self._num_samples
        raise ValueError(f"Unsupported dim: {dim}")


def _fake_audio_loader(path: str, *, sample_rate: int):
    del path
    waveform = FakeWaveform(sample_rate * 2)
    return waveform, sample_rate


class FakeVerifier:
    def __init__(self, scores: list[float] | None = None) -> None:
        self._scores = [0.9] if scores is None else scores
        self.calls: list[tuple] = []

    def verify_candidates(self, waveform, sample_rate: int, keyword_ids: list[int], candidates):
        self.calls.append((waveform, sample_rate, keyword_ids, candidates))
        output = []
        for candidate, score in zip(candidates, self._scores):
            output.append(
                {
                    "start_sec": candidate.start_sec,
                    "end_sec": candidate.end_sec,
                    "stage1_score": candidate.stage1_score,
                    "qbyt_score": score,
                }
            )
        return output


def _build_runner(
    *,
    threshold: float,
    verifier: FakeVerifier,
    monkeypatch=None,
) -> Stage2ClipRunner:
    if monkeypatch is not None:
        monkeypatch.setattr("dma_kws.inference.stage2_clip.make_g2p", _fake_g2p)
        monkeypatch.setattr(
            "dma_kws.inference.stage2_clip.text_to_phonemes",
            lambda _g2p, text: _fake_phonemes(text),
        )
    tokenizer = load_char_tokenizer("data/dict/lang_char.txt", split_with_space=" ")
    return Stage2ClipRunner(
        verifier=verifier,
        tokenizer=tokenizer,
        demo_cfg={"qbyt_threshold": threshold},
        sample_rate=16000,
    )


def test_clip_runner_scores_full_audio_span(monkeypatch):
    monkeypatch.setattr("dma_kws.inference.stage2_clip.load_audio", _fake_audio_loader)
    verifier = FakeVerifier(scores=[0.85])
    runner = _build_runner(threshold=0.5, verifier=verifier, monkeypatch=monkeypatch)

    result = runner.run("/tmp/clip.wav", "hello")

    assert result["audio"] == "/tmp/clip.wav"
    assert result["clip_span_sec"] == {"start_sec": 0.0, "end_sec": 2.0}
    candidates = verifier.calls[0][3]
    assert len(candidates) == 1
    assert candidates[0].start_sec == 0.0
    assert candidates[0].end_sec == 2.0
    assert candidates[0].stage1_score == 0.0


def test_clip_runner_detected_above_threshold(monkeypatch):
    monkeypatch.setattr("dma_kws.inference.stage2_clip.load_audio", _fake_audio_loader)
    runner = _build_runner(
        threshold=0.6,
        verifier=FakeVerifier(scores=[0.85]),
        monkeypatch=monkeypatch,
    )

    result = runner.run("/tmp/clip.wav", "hello")

    assert result["detected"] is True
    assert result["qbyt_score"] == 0.85
    assert result["skipped"] is False


def test_clip_runner_skipped_when_too_short(monkeypatch):
    monkeypatch.setattr("dma_kws.inference.stage2_clip.load_audio", _fake_audio_loader)
    runner = _build_runner(
        threshold=0.5,
        verifier=FakeVerifier(scores=[]),
        monkeypatch=monkeypatch,
    )

    result = runner.run("/tmp/clip.wav", "hello")

    assert result["detected"] is False
    assert result["qbyt_score"] == 0.0
    assert result["skipped"] is True


class FakeBatchVerifier:
    def __init__(self, scores: list[float]) -> None:
        from dma_kws.stage2.fbank import FbankExtractor

        self._scores = list(scores)
        self.batches: list[int] = []
        self.keyword_ids_batches: list[list[list[int]]] = []
        self.min_fbank_frames = 7
        self.fbank_extractor = FbankExtractor(dither=0.0)
        self.fbank_kwargs = {
            "num_mel_bins": 80,
            "frame_length": 25,
            "frame_shift": 10,
            "dither": 0.0,
            "window_type": "povey",
            "backend": "torchaudio_kaldi",
            "target_sample_rate": None,
            "snip_edges": True,
            "low_freq": 20.0,
            "high_freq": 0.0,
        }

    def score_clip_feats(self, feats, keyword_ids_batch):
        assert len(feats) == len(keyword_ids_batch)
        self.batches.append(len(feats))
        self.keyword_ids_batches.append(
            [list(keyword_ids) for keyword_ids in keyword_ids_batch]
        )
        scores = self._scores[: len(feats)]
        self._scores = self._scores[len(feats):]
        return scores

    def score_clip_feats_detailed(
        self,
        feats,
        keyword_ids_batch,
        *,
        include_eps_positions=False,
        include_seq_positions=False,
    ):
        scores = self.score_clip_feats(feats, keyword_ids_batch)
        records = []
        for score, keyword_ids in zip(scores, keyword_ids_batch):
            qbyt_logit = math.log(score / (1.0 - score))
            record = {
                "qbyt_score": score,
                "qbyt_logit": qbyt_logit,
                "completion_score": 0.25,
                "completion_logit": math.log(0.25 / 0.75),
            }
            if include_eps_positions:
                record["eps_position_logits"] = [qbyt_logit] * len(keyword_ids)
            if include_seq_positions:
                record["seq_position_logits"] = (
                    [0.5] * (len(keyword_ids) - 1) + [record["completion_logit"]]
                    if keyword_ids
                    else []
                )
            records.append(record)
        return records


def test_clip_runner_run_batch(monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchaudio")

    def loader(path: str, *, sample_rate: int):
        if path.endswith("short.wav"):
            return torch.zeros(1, 10), sample_rate
        return torch.zeros(1, sample_rate * 2), sample_rate

    monkeypatch.setattr("dma_kws.inference.stage2_clip.load_audio", loader)
    monkeypatch.setattr("dma_kws.inference.stage2_clip.make_g2p", _fake_g2p)
    g2p_calls: list[str] = []

    def fake_text_to_phonemes(_g2p, text):
        g2p_calls.append(text)
        return _fake_phonemes(text)

    monkeypatch.setattr(
        "dma_kws.inference.stage2_clip.text_to_phonemes", fake_text_to_phonemes
    )

    verifier = FakeBatchVerifier(scores=[0.9, 0.2])
    tokenizer = load_char_tokenizer("data/dict/lang_char.txt", split_with_space=" ")
    runner = Stage2ClipRunner(
        verifier=verifier,
        tokenizer=tokenizer,
        demo_cfg={"qbyt_threshold": 0.5},
        sample_rate=16000,
    )

    rows = [
        {"audio_path": "/tmp/a.wav", "keyword": "hello"},
        {
            "audio_path": "/tmp/short.wav",
            "keyword": "hello",
            "text_variant": "hey eva",
        },
        {"audio_path": "/tmp/b.wav", "keyword": "hello"},
    ]
    results = runner.run_batch(rows, batch_size=8, num_workers=0)

    assert [record["skipped"] for record in results] == [False, True, False]
    assert results[0]["qbyt_score"] == 0.9
    assert results[0]["detected"] is True
    assert results[1]["qbyt_score"] == 0.0
    assert results[1]["detected"] is False
    assert results[1]["text_variant_phonemes"] == [
        "HH",
        "EY1",
        "IY1",
        "V",
        "AH0",
    ]
    assert results[2]["qbyt_score"] == 0.2
    assert results[2]["detected"] is False
    assert g2p_calls == ["hello", "hey eva"]
    assert verifier.batches == [2]
    assert set(results[0]) == {
        "audio",
        "keyword",
        "keyword_phonemes",
        "clip_span_sec",
        "qbyt_score",
        "threshold",
        "detected",
        "skipped",
    }


def test_clip_runner_run_batch_uses_per_row_keyword_phoneme_overrides(monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchaudio")

    monkeypatch.setattr(
        "dma_kws.inference.stage2_clip.load_audio",
        lambda _path, *, sample_rate: (
            torch.zeros(1, sample_rate * 2),
            sample_rate,
        ),
    )
    monkeypatch.setattr("dma_kws.inference.stage2_clip.make_g2p", _fake_g2p)
    g2p_calls: list[str] = []
    ee_vah = ["HH", "EY1", "IY1", "V", "AH0"]
    ay_vah = ["HH", "EY1", "EY1", "V", "AH0"]
    hey_eve = ["HH", "EY1", "IY1", "V"]

    def fake_text_to_phonemes(_g2p, text):
        g2p_calls.append(text)
        assert text == "hey eva"
        return ay_vah

    monkeypatch.setattr(
        "dma_kws.inference.stage2_clip.text_to_phonemes", fake_text_to_phonemes
    )

    verifier = FakeBatchVerifier(scores=[0.9, 0.2, 0.8])
    tokenizer = load_char_tokenizer("data/dict/lang_char.txt", split_with_space=" ")
    runner = Stage2ClipRunner(
        verifier=verifier,
        tokenizer=tokenizer,
        demo_cfg={"qbyt_threshold": 0.5},
        sample_rate=16000,
    )
    rows = [
        {
            "audio_path": "/tmp/ee-vah.wav",
            "keyword": "hey eva",
            "keyword_phonemes": "HH EY1 IY1 V AH0",
            "text_variant": "hey eva",
            "text_variant_phonemes": "HH EY1 IY1 V",
        },
        {
            "audio_path": "/tmp/ay-vah.wav",
            "keyword": "hey eva",
            "keyword_phonemes": ay_vah,
        },
        {"audio_path": "/tmp/default.wav", "keyword": "hey eva"},
    ]

    results = runner.run_batch(rows, batch_size=8, num_workers=0)

    assert [result["keyword_phonemes"] for result in results] == [
        ee_vah,
        ay_vah,
        ay_vah,
    ]
    assert results[0]["text_variant_phonemes"] == hey_eve
    assert "text_variant_phonemes" not in results[1]
    assert "text_variant_phonemes" not in results[2]
    assert g2p_calls == ["hey eva"]
    assert verifier.keyword_ids_batches == [
        [
            [tokenizer.symbol_table[phone] for phone in ee_vah],
            [tokenizer.symbol_table[phone] for phone in ay_vah],
            [tokenizer.symbol_table[phone] for phone in ay_vah],
        ]
    ]


@pytest.mark.parametrize(
    "override",
    ["", None, [], "HH NOT_A_PHONE"],
)
def test_clip_runner_rejects_invalid_keyword_phoneme_override(monkeypatch, override):
    runner = _build_runner(
        threshold=0.5,
        verifier=FakeVerifier(scores=[0.7]),
        monkeypatch=monkeypatch,
    )

    with pytest.raises(ValueError, match="Manifest row 1 keyword_phonemes"):
        runner.run_batch(
            [
                {
                    "audio_path": "/tmp/clip.wav",
                    "keyword": "hey eva",
                    "keyword_phonemes": override,
                }
            ]
        )


def test_clip_runner_run_batch_can_include_score_details(monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchaudio")

    def loader(path: str, *, sample_rate: int):
        if path.endswith("short.wav"):
            return torch.zeros(1, 10), sample_rate
        return torch.zeros(1, sample_rate * 2), sample_rate

    monkeypatch.setattr("dma_kws.inference.stage2_clip.load_audio", loader)
    monkeypatch.setattr("dma_kws.inference.stage2_clip.make_g2p", _fake_g2p)
    monkeypatch.setattr(
        "dma_kws.inference.stage2_clip.text_to_phonemes",
        lambda _g2p, text: _fake_phonemes(text),
    )
    verifier = FakeBatchVerifier(scores=[0.9])
    tokenizer = load_char_tokenizer("data/dict/lang_char.txt", split_with_space=" ")
    runner = Stage2ClipRunner(
        verifier=verifier,
        tokenizer=tokenizer,
        demo_cfg={"qbyt_threshold": 0.5},
        sample_rate=16000,
    )

    results = runner.run_batch(
        [
            {"audio_path": "/tmp/a.wav", "keyword": "hello"},
            {"audio_path": "/tmp/short.wav", "keyword": "hello"},
        ],
        batch_size=8,
        num_workers=0,
        include_score_details=True,
        include_eps_positions=True,
        include_seq_positions=True,
    )

    assert results[0]["qbyt_logit"] == pytest.approx(math.log(9.0))
    assert results[0]["completion_score"] == pytest.approx(0.25)
    assert results[0]["completion_logit"] == pytest.approx(
        math.log(1.0 / 3.0)
    )
    assert results[0]["keyword_phonemes"] == ["HELLO"]
    assert results[0]["eps_position_logits"] == pytest.approx([math.log(9.0)])
    assert results[0]["seq_position_logits"] == pytest.approx(
        [math.log(1.0 / 3.0)]
    )
    assert results[1]["qbyt_logit"] is None
    assert results[1]["completion_score"] is None
    assert results[1]["completion_logit"] is None
    assert results[1]["eps_position_logits"] is None
    assert results[1]["seq_position_logits"] is None


def test_clip_runner_eps_positions_require_score_details(monkeypatch):
    runner = _build_runner(
        threshold=0.5,
        verifier=FakeVerifier(scores=[0.7]),
        monkeypatch=monkeypatch,
    )

    with pytest.raises(ValueError, match="requires include_score_details"):
        runner.run_batch([], include_eps_positions=True)
    with pytest.raises(ValueError, match="requires include_score_details"):
        runner.run_batch([], include_seq_positions=True)


def test_clip_runner_result_record_shape(monkeypatch):
    monkeypatch.setattr("dma_kws.inference.stage2_clip.load_audio", _fake_audio_loader)
    runner = _build_runner(
        threshold=0.5,
        verifier=FakeVerifier(scores=[0.7]),
        monkeypatch=monkeypatch,
    )

    result = runner.run("/tmp/clip.wav", "hello")

    assert set(result) == {
        "audio",
        "keyword",
        "keyword_phonemes",
        "clip_span_sec",
        "qbyt_score",
        "threshold",
        "detected",
        "skipped",
    }
