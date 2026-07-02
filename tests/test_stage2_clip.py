from __future__ import annotations

from dma_kws.inference.stage2_clip import Stage2ClipRunner
from dma_kws.tokenizer import load_char_tokenizer


def _fake_g2p():
    return lambda text: text.upper().split()


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
            lambda _g2p, text: text.upper().split(),
        )
    tokenizer = load_char_tokenizer("data/dict/lang_char.txt", split_with_space=" ")
    return Stage2ClipRunner(
        verifier=verifier,
        tokenizer=tokenizer,
        demo_cfg={"qbyt_threshold": threshold, "min_stage2_fbank_frames": 7},
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
