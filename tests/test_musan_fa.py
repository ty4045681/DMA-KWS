from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from dma_kws.inference.metrics import summarize_false_accept_rate
from dma_kws.inference.musan_fa import detect_subset
from dma_kws.inference.stage2_clip import Stage2ClipRunner
from dma_kws.tokenizer import load_char_tokenizer
import scripts.eval_musan_fa as eval_musan_fa


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
        self._scores = self._scores[len(feats) :]
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
            completion_logit = math.log(1.0 / 3.0)
            record = {
                "qbyt_score": score,
                "qbyt_logit": qbyt_logit,
                "completion_score": 0.25,
                "completion_logit": completion_logit,
            }
            if include_eps_positions:
                record["eps_position_logits"] = [qbyt_logit] * len(keyword_ids)
            if include_seq_positions:
                record["seq_position_logits"] = (
                    [0.5] * (len(keyword_ids) - 1) + [completion_logit]
                )
            records.append(record)
        return records


def _build_runner(threshold: float, verifier, monkeypatch) -> Stage2ClipRunner:
    monkeypatch.setattr("dma_kws.inference.stage2_clip.make_g2p", _fake_g2p)
    monkeypatch.setattr(
        "dma_kws.inference.stage2_clip.text_to_phonemes", _fake_text_to_phonemes
    )
    tokenizer = load_char_tokenizer("data/dict/lang_char.txt", split_with_space=" ")
    return Stage2ClipRunner(
        verifier=verifier,
        tokenizer=tokenizer,
        demo_cfg={"qbyt_threshold": threshold},
        sample_rate=16000,
    )


def test_resolve_keyword_phonemes_uses_override_or_g2p(monkeypatch):
    runner = _build_runner(
        threshold=0.5,
        verifier=FakeBatchVerifier(scores=[]),
        monkeypatch=monkeypatch,
    )

    assert runner.resolve_keyword_phonemes("hey eva") == [
        "HH",
        "EY1",
        "IY1",
        "V",
        "AH0",
    ]
    assert runner.resolve_keyword_phonemes(
        "hey eva",
        "HH EY1 EY1 V AH0",
    ) == ["HH", "EY1", "EY1", "V", "AH0"]
    with pytest.raises(ValueError, match="unsupported phonemes"):
        runner.resolve_keyword_phonemes("hey eva", "HH NOT_A_PHONE")


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
        "dma_kws.stage2.features.waveform_to_fbank",
        lambda *args, **kwargs: torch.zeros(1, 80),
    )
    monkeypatch.setattr(
        "dma_kws.inference.audio_utils.has_min_fbank_frames",
        lambda *args, **kwargs: True,
    )

    runner = _build_runner(threshold=0.5, verifier=verifier, monkeypatch=monkeypatch)
    keyword_phonemes = ["HH", "EY1", "IY1", "V", "AH0"]
    results = runner.run_file_windows(
        "/tmp/musan/speech/x.wav",
        "hey eva",
        window_sec=1.0,
        hop_sec=0.5,
        keyword_phonemes=keyword_phonemes,
        include_score_details=True,
        include_eps_positions=True,
        include_seq_positions=True,
    )

    assert len(results) == num_windows
    assert verifier.batches == [num_windows]
    assert [result["window_index"] for result in results] == list(
        range(num_windows)
    )
    assert results[0]["clip_span_sec"] == {"start_sec": 0.0, "end_sec": 1.0}
    assert results[-1]["clip_span_sec"] == {"start_sec": 4.0, "end_sec": 5.0}
    assert all(result["audio"] == "/tmp/musan/speech/x.wav" for result in results)
    assert all(result["keyword"] == "hey eva" for result in results)
    assert all(
        result["keyword_phonemes"] == keyword_phonemes for result in results
    )
    tokenizer = runner._tokenizer
    expected_ids = [tokenizer.symbol_table[phone] for phone in keyword_phonemes]
    assert verifier.keyword_ids_batches == [[expected_ids] * num_windows]
    assert len(results[0]["eps_position_logits"]) == len(keyword_phonemes)
    assert len(results[0]["seq_position_logits"]) == len(keyword_phonemes)
    assert results[4]["detected"] is True  # score 0.5 >= threshold 0.5
    assert results[3]["detected"] is False  # score 0.4 < threshold


@pytest.mark.parametrize(
    ("configured_phonemes", "expected_phonemes", "expected_source"),
    [
        ("", ["HH", "EY1", "IY1", "V", "AH0"], "g2p"),
        (
            "HH EY1 EY1 V AH0",
            ["HH", "EY1", "EY1", "V", "AH0"],
            "prep.keyword_phonemes",
        ),
    ],
)
def test_eval_musan_writes_only_rich_json_outputs(
    tmp_path,
    monkeypatch,
    configured_phonemes,
    expected_phonemes,
    expected_source,
):
    pytest.importorskip("torch")
    musan_root = tmp_path / "musan"
    audio_path = musan_root / "speech" / "sample.wav"
    audio_path.parent.mkdir(parents=True)
    audio_path.write_bytes(b"")
    output_dir = tmp_path / "out"
    captured = {}

    class FakeStreamPolicy:
        @staticmethod
        def describe():
            return {"mode": "test"}

    class FakeRunner:
        _demo_cfg = {"qbyt_threshold": 0.5}
        stream_policy = FakeStreamPolicy()

        def resolve_keyword_phonemes(
            self,
            keyword,
            keyword_phonemes=None,
            *,
            field_name,
        ):
            captured["resolve"] = (keyword, keyword_phonemes, field_name)
            if keyword_phonemes is None:
                return ["HH", "EY1", "IY1", "V", "AH0"]
            return str(keyword_phonemes).split()

        def run_file_windows(self, audio, keyword, **kwargs):
            captured["run"] = (audio, keyword, kwargs)
            phones = list(kwargs["keyword_phonemes"] or expected_phonemes)
            qbyt_logit = math.log(3.0)
            completion_logit = 0.0
            return [
                {
                    "audio": audio,
                    "keyword": keyword,
                    "keyword_phonemes": phones,
                    "clip_span_sec": {"start_sec": 0.0, "end_sec": 3.0},
                    "window_index": 0,
                    "qbyt_score": 0.75,
                    "qbyt_logit": qbyt_logit,
                    "completion_score": 0.5,
                    "completion_logit": completion_logit,
                    "eps_position_logits": [qbyt_logit] * len(phones),
                    "seq_position_logits": [0.25] * (len(phones) - 1)
                    + [completion_logit],
                    "detected": True,
                    "threshold": 0.5,
                    "skipped": False,
                }
            ]

    runner = FakeRunner()

    class FakeRunnerFactory:
        @staticmethod
        def from_config(_config, _prep, _device):
            return runner

    provenance = {
        "qbyt_readout": {"mode": "eps_mean", "temperature": 1.0},
        "sequence_objective": {
            "target_mode": "ordered_contiguous_prefix",
            "progress_weight": 0.5,
            "completion_weight": 0.5,
            "normalization": "sample",
        },
    }
    monkeypatch.setattr(
        eval_musan_fa,
        "resolved_config",
        lambda _cfg: {
            "paths": {},
            "stage1": {},
            "stage2": {},
            "demo": {},
            "tokenizer": {},
        },
    )
    monkeypatch.setattr(
        eval_musan_fa,
        "resolve_accelerator",
        lambda _device: ("cpu", 1),
    )
    monkeypatch.setattr(eval_musan_fa, "Stage2ClipRunner", FakeRunnerFactory)
    monkeypatch.setattr(
        eval_musan_fa,
        "build_score_provenance",
        lambda *_args, **_kwargs: provenance,
    )
    monkeypatch.setattr(eval_musan_fa, "iter_audio_files", lambda _root: [audio_path])
    monkeypatch.setattr(eval_musan_fa, "audio_duration_sec", lambda _path: 3.0)

    cfg = OmegaConf.create(
        {
            "prep": {
                "keyword": "hey eva",
                "keyword_phonemes": configured_phonemes,
                "musan_root": str(musan_root),
                "stage2_ckpt": "stage2.pt",
                "window_sec": 3.0,
                "hop_sec": 1.0,
                "output_dir": str(output_dir),
            },
            "run": {"device": "cpu"},
        }
    )

    summary = eval_musan_fa.run_eval(cfg)

    assert {path.name for path in output_dir.iterdir()} == {
        "results.jsonl",
        "summary.json",
    }
    assert summary["keyword_phonemes"] == expected_phonemes
    assert summary["keyword_phonemes_source"] == expected_source
    assert summary["num_samples"] == 1
    assert "manifest" not in summary
    run_keyword_phonemes = expected_phonemes if configured_phonemes else None
    assert captured["run"][2] == {
        "window_sec": 3.0,
        "hop_sec": 1.0,
        "keyword_phonemes": run_keyword_phonemes,
        "include_score_details": True,
        "include_eps_positions": True,
        "include_seq_positions": True,
    }
    result = json.loads(
        (output_dir / "results.jsonl").read_text(encoding="utf-8").strip()
    )
    assert result["keyword_phonemes"] == expected_phonemes
    assert result["qbyt_logit"] == pytest.approx(math.log(3.0))
    assert result["completion_score"] == pytest.approx(0.5)
    assert len(result["eps_position_logits"]) == len(expected_phonemes)
    assert len(result["seq_position_scores"]) == len(expected_phonemes)
    assert result["expected_prefix_length"] is not None
    assert result["seq_position_targets"] is None
    assert result["manifest_meta"] == {
        "subset": "speech",
        "start_sec": 0.0,
        "end_sec": 3.0,
        "window_index": 0,
    }


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
