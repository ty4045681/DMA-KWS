from __future__ import annotations

import json
from pathlib import Path

import pytest

from dma_kws.inference.metrics import summarize_false_accept_rate
from dma_kws.inference.musan_fa import detect_subset
from dma_kws.inference.stage2_clip import Stage2ClipRunner
from dma_kws.tokenizer import load_char_tokenizer


def _fake_phonemes(text: str) -> list[str]:
    """Mimic g2p_en: upper-case ARPAbet with stress digits on vowels."""
    phones = {"hey eva": ["HH", "EY1", "IY1", "V", "AH0"]}
    return phones.get(text.lower(), text.upper().split())


def _fake_g2p():
    return _fake_phonemes


def _fake_text_to_phonemes(_g2p, text):
    return _fake_phonemes(text)


class FakeBatchVerifier:
    def __init__(self, scores: list[float]) -> None:
        from dma_kws.stage2.fbank import FbankExtractor

        self._scores = list(scores)
        self.batches: list[int] = []
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
        scores = self._scores[: len(feats)]
        self._scores = self._scores[len(feats) :]
        return scores


def _build_runner(threshold: float, verifier, monkeypatch) -> Stage2ClipRunner:
    monkeypatch.setattr("dma_kws.inference.stage2_clip.make_g2p", _fake_g2p)
    monkeypatch.setattr(
        "dma_kws.inference.stage2_clip.text_to_phonemes", _fake_text_to_phonemes
    )
    tokenizer = load_char_tokenizer("data/dict/lang_char.txt", split_with_space=" ")
    return Stage2ClipRunner(
        verifier=verifier,
        tokenizer=tokenizer,
        demo_cfg={"qbyt_threshold": threshold, "min_stage2_fbank_frames": 7},
        sample_rate=16000,
    )


def test_run_file_windows_counts_and_spans(monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchaudio")

    num_windows = 9
    scores = [0.1 * (i + 1) for i in range(num_windows)]
    verifier = FakeBatchVerifier(scores=scores)

    def loader(path: str, *, sample_rate: int):
        del path
        # 5 seconds of silence at 16 kHz
        return torch.zeros(1, sample_rate * 5), sample_rate

    monkeypatch.setattr("dma_kws.inference.stage2_clip.load_audio", loader)

    # Avoid real fbank extraction by returning identical tiny tensors.
    monkeypatch.setattr(
        "dma_kws.inference.stage2_clip.waveform_to_fbank",
        lambda *args, **kwargs: torch.zeros(1, 80),
    )
    monkeypatch.setattr(
        "dma_kws.inference.stage2_clip.has_min_fbank_frames", lambda *args, **kwargs: True
    )

    runner = _build_runner(threshold=0.5, verifier=verifier, monkeypatch=monkeypatch)
    results = runner.run_file_windows(
        "/tmp/musan/speech/x.wav",
        "hello",
        window_sec=1.0,
        hop_sec=0.5,
    )

    assert len(results) == num_windows
    assert verifier.batches == [num_windows]
    assert results[0]["clip_span_sec"] == {"start_sec": 0.0, "end_sec": 1.0}
    assert results[-1]["clip_span_sec"] == {"start_sec": 4.0, "end_sec": 5.0}
    assert all(result["audio"] == "/tmp/musan/speech/x.wav" for result in results)
    assert all(result["keyword"] == "hello" for result in results)
    assert results[4]["detected"] is True  # score 0.5 >= threshold 0.5
    assert results[3]["detected"] is False  # score 0.4 < threshold


def test_summarize_false_accept_rate_computes_fa_per_hour():
    results = [
        {"label": 0, "best_qbyt_score": 0.9, "detected": True},
        {"label": 0, "best_qbyt_score": 0.4, "detected": False},
        {"label": 0, "best_qbyt_score": 0.8, "detected": True},
        {"label": 0, "best_qbyt_score": 0.2, "detected": False},
    ]

    summary = summarize_false_accept_rate(results, threshold=0.5, total_hours=2.0)

    assert summary["num_samples"] == 4.0
    assert summary["fp"] == 2.0
    assert summary["tn"] == 2.0
    assert summary["total_hours"] == 2.0
    assert summary["fa_per_hour"] == pytest.approx(1.0)
    assert summary["fa_per_1000_hours"] == pytest.approx(1000.0)
    assert summary["fpr"] == pytest.approx(0.5)


def test_detect_subset(tmp_path):
    musan_root = tmp_path / "musan"
    speech_file = musan_root / "speech" / "x.wav"
    music_file = musan_root / "music" / "y.wav"
    noise_file = musan_root / "noise" / "z.wav"
    outside_file = tmp_path / "outside.wav"

    speech_file.parent.mkdir(parents=True)
    music_file.parent.mkdir(parents=True)
    noise_file.parent.mkdir(parents=True)
    speech_file.write_text("")
    music_file.write_text("")
    noise_file.write_text("")
    outside_file.write_text("")

    assert detect_subset(speech_file, musan_root) == "speech"
    assert detect_subset(music_file, musan_root) == "music"
    assert detect_subset(noise_file, musan_root) == "noise"
    assert detect_subset(outside_file, musan_root) == "other"


def test_aggregate_musan_fa(tmp_path):
    from scripts.aggregate_musan_fa import aggregate, write_tsv

    out_dir = tmp_path / "out"
    (out_dir / "ckpt1" / "hey_eva").mkdir(parents=True)
    (out_dir / "ckpt1" / "hey_android").mkdir(parents=True)

    (out_dir / "ckpt1" / "hey_eva" / "summary.json").write_text(
        json.dumps(
            {
                "keyword": "hey eva",
                "stage2_ckpt": "/path/to/stage2_step020000.pt",
                "total_files": 10,
                "total_hours": 1.0,
                "num_samples": 100,
                "metrics": {
                    "fp": 2,
                    "tn": 98,
                    "fa_per_hour": 2.0,
                    "fa_per_1000_hours": 2000.0,
                    "fpr": 0.02,
                },
                "subsets": {
                    "speech": {
                        "total_hours": 0.5,
                        "metrics": {"fa_per_hour": 4.0},
                    }
                },
            }
        )
    )
    (out_dir / "ckpt1" / "hey_android" / "summary.json").write_text(
        json.dumps(
            {
                "keyword": "hey android",
                "stage2_ckpt": "/path/to/stage2_step010000.pt",
                "total_files": 10,
                "total_hours": 1.0,
                "num_samples": 100,
                "metrics": {"fp": 1, "tn": 99, "fa_per_hour": 1.0, "fpr": 0.01},
                "subsets": {},
            }
        )
    )

    rows = aggregate(out_dir)
    assert len(rows) == 2
    assert rows[0]["keyword"] == "hey android"
    assert rows[0]["fa_per_hour"] == 1.0
    assert rows[1]["keyword"] == "hey eva"
    assert rows[1]["subset_speech_fa_per_hour"] == 4.0

    tsv_path = tmp_path / "summary.tsv"
    write_tsv(rows, tsv_path)
    assert tsv_path.exists()
    lines = tsv_path.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 3  # header + 2 rows
    assert "keyword" in lines[0]
    assert "subset_speech_fa_per_hour" in lines[0]
