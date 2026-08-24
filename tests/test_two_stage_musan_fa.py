from __future__ import annotations

import json
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from dma_kws.inference.musan_fa import (
    TWO_STAGE_WAKEUP_PROTOCOL,
    merge_musan_summaries,
    two_stage_wakeup_result_record,
)
from dma_kws.stage1.candidates import KeywordCandidate
import scripts.eval_two_stage_musan_fa as eval_two_stage_musan_fa


def _fake_phonemes(text: str) -> list[str]:
    phones = {"hey eva": ["HH", "EY1", "IY1", "V", "AH0"]}
    return phones.get(text.lower(), text.upper().split())


class FakeStreamPolicy:
    @staticmethod
    def describe():
        return {"mode": "test"}


class FakeLocator:
    def __init__(self, candidates_by_path: dict[str, list[KeywordCandidate]]) -> None:
        self._candidates_by_path = candidates_by_path
        self.calls: list[tuple[str, str]] = []

    def locate(self, audio_path: str, keyword: str, keyword_phonemes=None):
        phonemes = None if keyword_phonemes is None else tuple(keyword_phonemes)
        self.calls.append((audio_path, keyword, phonemes))
        return list(self._candidates_by_path.get(audio_path, []))


class FakeVerifier:
    def __init__(self, scores: list[float]) -> None:
        self._scores = list(scores)
        self.calls: list[tuple] = []
        self.stream_policy = FakeStreamPolicy()

    def verify_candidates(self, waveform, sample_rate: int, keyword_ids, candidates):
        del waveform, sample_rate, keyword_ids
        self.calls.append(list(candidates))
        output = []
        remaining = list(self._scores)
        for candidate in candidates:
            score = remaining.pop(0) if remaining else 0.0
            output.append(
                {
                    "start_sec": candidate.start_sec,
                    "end_sec": candidate.end_sec,
                    "stage1_score": candidate.stage1_score,
                    "qbyt_score": score,
                }
            )
        return output


def _candidate(start: float, end: float) -> KeywordCandidate:
    return KeywordCandidate(
        start_sec=start,
        end_sec=end,
        stage1_score=1.0,
        phonemes=[],
    )


def _install_pipeline(
    monkeypatch,
    *,
    locator: FakeLocator,
    verifier: FakeVerifier,
    threshold: float = 0.5,
):
    from dma_kws.inference.pipeline import TwoStageKWSPipeline
    from dma_kws.tokenizer import load_char_tokenizer

    def _fake_audio_loader(path: str, *, sample_rate: int):
        del path
        import torch

        return torch.zeros(1, sample_rate), sample_rate

    monkeypatch.setattr("dma_kws.inference.pipeline.load_audio", _fake_audio_loader)
    monkeypatch.setattr("dma_kws.inference.pipeline.make_g2p", lambda: _fake_phonemes)
    monkeypatch.setattr(
        "dma_kws.inference.pipeline.text_to_phonemes",
        lambda _g2p, text: _fake_phonemes(text),
    )

    tokenizer = load_char_tokenizer("data/dict/lang_char.txt", split_with_space=" ")
    pipeline = TwoStageKWSPipeline(
        locator=locator,
        verifier=verifier,
        tokenizer=tokenizer,
        demo_cfg={"qbyt_threshold": threshold},
        sample_rate=16000,
    )

    class FakeFactory:
        @staticmethod
        def from_config(_config, _prep, _device):
            return pipeline

    monkeypatch.setattr(eval_two_stage_musan_fa, "TwoStageKWSPipeline", FakeFactory)
    return pipeline


def _common_eval_patches(monkeypatch, audio_paths: list[Path], duration_sec: float):
    monkeypatch.setattr(
        eval_two_stage_musan_fa,
        "resolved_config",
        lambda _cfg: {
            "paths": {},
            "stage1": {},
            "stage2": {},
            "demo": {},
            "tokenizer": {},
            "locator": {"type": "sherpa_kws"},
        },
    )
    monkeypatch.setattr(
        eval_two_stage_musan_fa,
        "resolve_accelerator",
        lambda _device: ("cpu", 1),
    )
    monkeypatch.setattr(
        eval_two_stage_musan_fa,
        "build_score_provenance",
        lambda *_args, **_kwargs: {
            "qbyt_readout": {"mode": "eps_softmin", "temperature": 1.0},
            "sequence_objective": {
                "target_mode": "ordered_contiguous_prefix",
                "progress_weight": 0.5,
                "completion_weight": 0.5,
                "normalization": "sample",
            },
        },
    )
    monkeypatch.setattr(
        eval_two_stage_musan_fa,
        "iter_audio_files",
        lambda _root: audio_paths,
    )
    monkeypatch.setattr(
        eval_two_stage_musan_fa,
        "audio_duration_sec",
        lambda _path: duration_sec,
    )


def test_two_stage_wakeup_record_preserves_span_and_stage1_score():
    record = two_stage_wakeup_result_record(
        "/musan/noise/x.wav",
        "hey eva",
        "noise",
        12.0,
        ["HH", "EY1"],
        {
            "start_sec": 1.5,
            "end_sec": 2.25,
            "stage1_score": 0.8,
            "qbyt_score": 0.91,
        },
        candidate_index=2,
        threshold=0.5,
        qbyt_readout={"mode": "eps_softmin", "temperature": 0.5},
    )
    assert record["qbyt_score"] == pytest.approx(0.91)
    assert record["detected"] is True
    assert record["label"] == 0
    assert record["clip_span_sec"] == {"start_sec": 1.5, "end_sec": 2.25}
    assert record["stage1_score"] == pytest.approx(0.8)
    assert record["qbyt_readout_mode"] == "eps_softmin"
    assert record["manifest_meta"] == {
        "subset": "noise",
        "duration_sec": 12.0,
        "candidate_index": 2,
        "start_sec": 1.5,
        "end_sec": 2.25,
    }


def test_eval_two_stage_musan_fa_counts_multiple_wakeups_in_one_file(
    tmp_path, monkeypatch
):
    pytest.importorskip("torch")
    pytest.importorskip("matplotlib")
    musan_root = tmp_path / "musan"
    audio_path = musan_root / "noise" / "long.wav"
    audio_path.parent.mkdir(parents=True)
    audio_path.write_bytes(b"")
    output_dir = tmp_path / "out"

    locator = FakeLocator(
        {
            str(audio_path.resolve()): [
                _candidate(0.0, 1.0),
                _candidate(4.0, 5.0),
                _candidate(8.0, 9.0),
            ]
        }
    )
    verifier = FakeVerifier(scores=[0.9, 0.2, 0.8])
    _install_pipeline(monkeypatch, locator=locator, verifier=verifier)
    _common_eval_patches(monkeypatch, [audio_path], duration_sec=3600.0)

    cfg = OmegaConf.create(
        {
            "prep": {
                "keyword": "hey eva",
                "musan_root": str(musan_root),
                "stage2_ckpt": "stage2.pt",
                "output_dir": str(output_dir),
                "plot_dpi": 72,
            },
            "run": {"device": "cpu"},
        }
    )
    summary = eval_two_stage_musan_fa.run_eval(cfg)

    assert summary["eval_protocol"] == TWO_STAGE_WAKEUP_PROTOCOL
    assert summary["locator"] == "sherpa_kws"
    assert summary["total_files"] == 1
    assert summary["total_hours"] == pytest.approx(1.0)
    assert summary["num_samples"] == 3
    assert summary["num_stage1_candidates"] == 3
    assert summary["num_stage2_scored"] == 3
    assert summary["num_wakeups"] == 2
    assert summary["metrics"]["fp"] == pytest.approx(2.0)
    assert summary["metrics"]["fa_per_hour"] == pytest.approx(2.0)
    assert summary["subsets"]["noise"]["total_hours"] == pytest.approx(1.0)
    assert summary["subsets"]["noise"]["metrics"]["fp"] == pytest.approx(2.0)
    assert summary["keyword_phonemes"] == ["HH", "EY1", "IY1", "V", "AH0"]
    assert summary["keyword_phonemes_source"] == "g2p"

    rows = [
        json.loads(line)
        for line in (output_dir / "results.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    assert [row["qbyt_score"] for row in rows] == [0.9, 0.2, 0.8]
    assert [row["detected"] for row in rows] == [True, False, True]
    assert [row["manifest_meta"]["candidate_index"] for row in rows] == [0, 1, 2]


def test_eval_two_stage_musan_fa_zero_candidates_still_count_hours(
    tmp_path, monkeypatch
):
    pytest.importorskip("torch")
    musan_root = tmp_path / "musan"
    audio_path = musan_root / "speech" / "clean.wav"
    audio_path.parent.mkdir(parents=True)
    audio_path.write_bytes(b"")
    output_dir = tmp_path / "out"

    locator = FakeLocator({str(audio_path.resolve()): []})
    verifier = FakeVerifier(scores=[])
    pipeline = _install_pipeline(monkeypatch, locator=locator, verifier=verifier)
    _common_eval_patches(monkeypatch, [audio_path], duration_sec=1800.0)

    cfg = OmegaConf.create(
        {
            "prep": {
                "keyword": "hey eva",
                "keyword_phonemes": "HH EY1 EY1 V AH0",
                "musan_root": str(musan_root),
                "stage2_ckpt": "stage2.pt",
                "output_dir": str(output_dir),
                "plot_curves": False,
            },
            "run": {"device": "cpu"},
        }
    )
    summary = eval_two_stage_musan_fa.run_eval(cfg)

    assert summary["num_samples"] == 0
    assert summary["num_stage1_candidates"] == 0
    assert summary["num_wakeups"] == 0
    assert summary["metrics"]["fp"] == pytest.approx(0.0)
    assert summary["metrics"]["fa_per_hour"] == pytest.approx(0.0)
    assert summary["total_hours"] == pytest.approx(0.5)
    assert summary["subsets"]["speech"]["total_hours"] == pytest.approx(0.5)
    assert summary["subsets"]["speech"]["metrics"]["fp"] == pytest.approx(0.0)
    assert summary["keyword_phonemes"] == ["HH", "EY1", "EY1", "V", "AH0"]
    assert summary["keyword_phonemes_source"] == "prep.keyword_phonemes"
    assert (output_dir / "results.jsonl").read_text(encoding="utf-8") == ""
    assert pipeline._verifier.calls == []
    assert locator.calls[0][2] == ("HH", "EY1", "EY1", "V", "AH0")


def test_eval_two_stage_musan_fa_drops_unscored_short_candidates(
    tmp_path, monkeypatch
):
    pytest.importorskip("torch")
    musan_root = tmp_path / "musan"
    audio_path = musan_root / "music" / "clip.wav"
    audio_path.parent.mkdir(parents=True)
    audio_path.write_bytes(b"")
    output_dir = tmp_path / "out"

    candidates = [_candidate(0.0, 0.1), _candidate(1.0, 2.0)]
    locator = FakeLocator({str(audio_path.resolve()): candidates})

    class DroppingVerifier(FakeVerifier):
        def verify_candidates(self, waveform, sample_rate, keyword_ids, located):
            del waveform, sample_rate, keyword_ids
            kept = located[1:]
            return [
                {
                    "start_sec": kept[0].start_sec,
                    "end_sec": kept[0].end_sec,
                    "stage1_score": kept[0].stage1_score,
                    "qbyt_score": 0.4,
                }
            ]

    verifier = DroppingVerifier(scores=[0.4])
    _install_pipeline(monkeypatch, locator=locator, verifier=verifier)
    _common_eval_patches(monkeypatch, [audio_path], duration_sec=3600.0)

    cfg = OmegaConf.create(
        {
            "prep": {
                "keyword": "hey eva",
                "musan_root": str(musan_root),
                "stage2_ckpt": "stage2.pt",
                "output_dir": str(output_dir),
                "plot_curves": False,
            },
            "run": {"device": "cpu"},
        }
    )
    summary = eval_two_stage_musan_fa.run_eval(cfg)
    assert summary["num_stage1_candidates"] == 2
    assert summary["num_stage2_scored"] == 1
    assert summary["num_samples"] == 1
    assert summary["metrics"]["fp"] == pytest.approx(0.0)
    assert summary["metrics"]["tn"] == pytest.approx(1.0)
    assert summary["total_hours"] == pytest.approx(1.0)


def test_merge_rejects_mixed_eval_protocol():
    identity = {
        "keyword": "hey eva",
        "keyword_phonemes": ["HH"],
        "keyword_phonemes_source": "g2p",
        "stage2_ckpt": "stage2.pt",
        "window_sec": 3.0,
        "hop_sec": 3.0,
        "musan_root": "/musan",
        "stream": {"mode": "test"},
        "provenance": {"qbyt_readout": {"mode": "eps_mean"}},
        "amp": "off",
        "fbank_windows": "independent",
        "batch_size": 64,
        "num_shards": 2,
    }
    with pytest.raises(ValueError, match="eval_protocol"):
        merge_musan_summaries(
            [
                {**identity, "eval_protocol": TWO_STAGE_WAKEUP_PROTOCOL, "total_hours": 1.0},
                {**identity, "total_hours": 1.0},
            ],
            [[], []],
            output_dir="/tmp/merged",
            threshold=0.5,
        )


def test_merge_two_stage_wakeups_pools_hours_and_false_accepts():
    identity = {
        "eval_protocol": TWO_STAGE_WAKEUP_PROTOCOL,
        "locator": "sherpa_kws",
        "keyword": "hey eva",
        "keyword_phonemes": ["HH", "EY1"],
        "keyword_phonemes_source": "g2p",
        "stage2_ckpt": "stage2.pt",
        "musan_root": "/musan",
        "stream": {"mode": "test"},
        "provenance": {"qbyt_readout": {"mode": "eps_softmin", "temperature": 1.0}},
        "amp": "off",
        "num_shards": 2,
    }
    shard0 = [
        {
            "audio_path": "/musan/noise/a.wav",
            "label": 0,
            "qbyt_score": 0.9,
            "detected": True,
            "skipped": False,
            "manifest_meta": {"subset": "noise", "candidate_index": 1},
        },
        {
            "audio_path": "/musan/noise/a.wav",
            "label": 0,
            "qbyt_score": 0.2,
            "detected": False,
            "skipped": False,
            "manifest_meta": {"subset": "noise", "candidate_index": 0},
        },
    ]
    shard1 = [
        {
            "audio_path": "/musan/speech/b.wav",
            "label": 0,
            "qbyt_score": 0.8,
            "detected": True,
            "skipped": False,
            "manifest_meta": {"subset": "speech", "candidate_index": 0},
        }
    ]
    merged, results = merge_musan_summaries(
        [
            {
                **identity,
                "total_files": 1,
                "total_hours": 1.0,
                "num_stage1_candidates": 2,
                "num_stage2_scored": 2,
                "num_wakeups": 1,
                "subsets": {"noise": {"total_hours": 1.0}},
            },
            {
                **identity,
                "total_files": 1,
                "total_hours": 3.0,
                "num_stage1_candidates": 1,
                "num_stage2_scored": 1,
                "num_wakeups": 1,
                "subsets": {"speech": {"total_hours": 3.0}},
            },
        ],
        [shard0, shard1],
        output_dir="/tmp/merged",
        threshold=0.5,
    )
    assert merged["eval_protocol"] == TWO_STAGE_WAKEUP_PROTOCOL
    assert merged["locator"] == "sherpa_kws"
    assert merged["num_stage1_candidates"] == 3
    assert merged["num_stage2_scored"] == 3
    assert merged["num_wakeups"] == 2
    assert merged["metrics"]["fp"] == pytest.approx(2.0)
    assert merged["metrics"]["fa_per_hour"] == pytest.approx(0.5)
    assert [row["audio_path"] for row in results] == [
        "/musan/noise/a.wav",
        "/musan/noise/a.wav",
        "/musan/speech/b.wav",
    ]
    assert [row["manifest_meta"]["candidate_index"] for row in results[:2]] == [0, 1]
