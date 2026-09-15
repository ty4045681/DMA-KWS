"""T10–T14: Hydra sink-attention diagnostics CLI and output contract."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import soundfile as sf
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from dma_kws.config import compose_config, config_to_dict, fbank_kwargs
from dma_kws.configs.schema import FbankConfig, StreamPolicy
from dma_kws.inference.score_calibration import PositiveAffineCalibrator
from dma_kws.inference.stage2_clip import Stage2ClipRunner
from dma_kws.inference.stage2_verifier import Stage2Verifier
from dma_kws.nn import min_input_frames_for_encoder
from dma_kws.stage2.fbank import FbankExtractor
from dma_kws.stage2.readout import QbyTScoreSpec
from dma_kws.stage2.readout_pooling import QbyTReadoutConfig
from dma_kws.tokenizer import load_char_tokenizer
from qbyt.pooling import QbyT


REPO_ROOT = Path(__file__).resolve().parents[1]
DICT_PATH = REPO_ROOT / "data" / "dict" / "lang_char.txt"
PARITY_ATOL = 1e-5
PARITY_RTOL = 1e-4
ENCODER_DIM = 48
EMBED_DIM = 64
SAMPLE_RATE = 16000

RECORDS_FIELDS = [
    "run_id",
    "sample_id",
    "manifest_record_number",
    "audio_path",
    "keyword",
    "keyword_phonemes",
    "query_id",
    "condition",
    "pair_id",
    "label",
    "ablation",
    "blocked_layers",
    "qbyt_raw_logit",
    "qbyt_score",
    "threshold",
    "detected",
    "delta_raw_logit",
    "delta_qbyt_score",
    "text_length",
    "audio_length",
    "status",
    "skip_reason",
    "trace_path",
    "report_selected",
]


def _write_wav(path: Path, duration_sec: float = 1.0, sample_rate: int = SAMPLE_RATE) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    n_samples = max(1, int(round(duration_sec * sample_rate)))
    sf.write(str(path), np.zeros(n_samples, dtype=np.float32), sample_rate)
    return path


def _write_csv(path: Path, header: str, *rows: str) -> Path:
    path.write_text(header + "\n" + "\n".join(rows) + "\n", encoding="utf-8")
    return path


class _ProjEncoder(nn.Module):
    def __init__(self, in_dim: int = 80, out_dim: int = ENCODER_DIM) -> None:
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)

    def output_frames(self, num_input_frames: int) -> int:
        return int(num_input_frames)

    def forward(self, feats, feat_lengths):
        speech = self.proj(feats)
        time = speech.size(1)
        arange = torch.arange(time, device=speech.device)
        mask = arange[None, None, :] < feat_lengths.to(device=speech.device)[:, None, None]
        return speech, mask


class _Stage2Model(nn.Module):
    def __init__(self, encoder: nn.Module, qbyt: nn.Module) -> None:
        super().__init__()
        self.encoder = encoder
        self.qbyt = qbyt
        self.adapter = None

    def encode_for_qbyt(self, feats, feat_lengths):
        from dma_kws.nn import run_encoder

        encoder_out, encoder_mask = run_encoder(
            self.encoder,
            feats,
            feat_lengths,
            policy=StreamPolicy(backend="conformer", enabled=False),
            mode="eval",
        )
        encoder_lens = encoder_mask.squeeze(1).sum(1)
        return encoder_out, encoder_lens

    def forward(self, feats, feat_lengths, anchors, anchor_lengths):
        speech, encoder_lens = self.encode_for_qbyt(feats, feat_lengths)
        logits, _extra = self.qbyt(
            speech,
            anchors,
            speech_lengths=encoder_lens,
            text_lengths=anchor_lengths,
        )
        return logits


class _StubG2P:
    def __call__(self, text: str) -> list[str]:
        table = {
            "hey eva": ["HH", "EY1", "IY1", "V", "AH0"],
            "ok google": ["OW1", "K", "G", "UW1", "G", "AH0", "L"],
        }
        return list(table.get(text, ["HH", "EY1"]))


def _pooling_score_spec() -> QbyTScoreSpec:
    return QbyTScoreSpec(
        version=4,
        value=QbyTReadoutConfig(
            mode="eps_softmin",
            temperature=1.0,
            sink_token=True,
            text_position="learned",
            audio_position="relative_bias",
        ),
    )


def _build_pooling_runner() -> Stage2ClipRunner:
    torch.manual_seed(0)
    tokenizer = load_char_tokenizer(str(DICT_PATH))
    fbank_cfg = FbankConfig(dither=0.0, window_type="povey")
    kwargs = fbank_kwargs(fbank_cfg)
    encoder = _ProjEncoder()
    qbyt = QbyT(
        encoder_output_size=ENCODER_DIM,
        num_embeds=len(tokenizer.symbol_table),
        embed_dim=EMBED_DIM,
        post_num_layers=2,
        readout_mode="eps_softmin",
        sink_token=True,
        text_position="learned",
        audio_position="relative_bias",
    ).eval()
    verifier = Stage2Verifier.__new__(Stage2Verifier)
    verifier._torch = torch
    verifier._demo_cfg = {"qbyt_threshold": 0.5, "min_stage2_encoder_frames": 1}
    verifier._device = torch.device("cpu")
    verifier._amp = None
    verifier._calibrator = PositiveAffineCalibrator(slope=1.0, bias=0.0)
    verifier._fbank_kwargs = kwargs
    verifier._fbank_extractor = FbankExtractor(**kwargs)
    verifier._stream_policy = StreamPolicy(backend="conformer", enabled=False)
    verifier._model = _Stage2Model(encoder, qbyt).eval()
    verifier._min_fbank_frames = min_input_frames_for_encoder(encoder, 1)
    verifier.qbyt_score = _pooling_score_spec()
    verifier.qbyt_alignment = None
    return Stage2ClipRunner(
        verifier=verifier,
        tokenizer=tokenizer,
        demo_cfg=verifier._demo_cfg,
        sample_rate=SAMPLE_RATE,
    )


@pytest.fixture(scope="module")
def pooling_runner() -> Stage2ClipRunner:
    with patch("dma_kws.inference.stage2_clip.make_g2p", return_value=_StubG2P()):
        return _build_pooling_runner()


@pytest.fixture
def install_runner(monkeypatch, pooling_runner):
    monkeypatch.setattr(
        Stage2ClipRunner,
        "from_config",
        classmethod(lambda cls, _config, _prep, _device: pooling_runner),
    )
    return pooling_runner


def _resolved_config() -> dict:
    return {
        "paths": {},
        "stage1": {"sample_rate": SAMPLE_RATE},
        "stage2": {
            "qbyt_readout_version": 4,
            "qbyt_readout": {
                "mode": "eps_softmin",
                "temperature": 1.0,
                "sink_token": True,
                "text_position": "learned",
                "audio_position": "relative_bias",
            },
            "encoder_output_dim": ENCODER_DIM,
            "qbyt_embed_dim": EMBED_DIM,
            "qbyt_layers": 2,
        },
        "demo": {"qbyt_threshold": 0.5, "min_stage2_encoder_frames": 1},
        "tokenizer": {"dict_path": str(DICT_PATH), "split_with_space": " "},
        "training": {"seed": 2025},
    }


def _sink_defaults(**overrides) -> dict:
    payload = {
        "mode": "clips",
        "capture_layers": "all",
        "capture_heads": "all",
        "save_traces": True,
        "save_full_attention": False,
        "ablations": ["block_sink_all"],
        "max_combined_tokens": 1024,
        "max_attention_bytes": 268435456,
        "parity_atol": 0.00001,
        "parity_rtol": 0.0001,
        "max_report_samples": 40,
        "plot_dpi": 160,
        "length_bins": [0, 100, 200, 400, 800],
        "group_field": "condition",
        "synthetic_fixture": False,
    }
    payload.update(overrides)
    return payload


def _cfg(tmp_path: Path, *, manifest: Path, sink=None, **prep) -> OmegaConf:
    ckpt = tmp_path / "stage2.pt"
    if not ckpt.exists():
        ckpt.write_bytes(b"stub-checkpoint")
    payload = {
        "manifest": str(manifest),
        "stage2_ckpt": str(ckpt),
        "output_dir": str(tmp_path / "out"),
        "batch_size": 1,
        "num_workers": 0,
        "amp": "off",
        "keyword_eval": {"mode": "per_row", "targets": [], "query_batch_size": 64},
        "sink_diagnostics": _sink_defaults() if sink is None else sink,
    }
    payload.update(prep)
    return OmegaConf.create({"prep": payload, "run": {"device": "cpu"}})


def _run_diagnose(monkeypatch, cfg):
    import scripts.diagnose_qbyt_sink as diagnose

    monkeypatch.setattr(diagnose, "resolved_config", lambda _cfg: _resolved_config())
    return diagnose.run_diagnose(cfg)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_padding_helper_is_shared_with_eval_stage2_clips():
    from dma_kws.inference.stage2_clip import resolve_clip_audio_padding_ms
    from scripts.eval_stage2_clips import _resolve_audio_padding_ms

    assert _resolve_audio_padding_ms is resolve_clip_audio_padding_ms
    assert resolve_clip_audio_padding_ms({}) == (0, 0)
    assert resolve_clip_audio_padding_ms({}, {"qbyt_readout_version": 6}) == (160, 160)
    assert resolve_clip_audio_padding_ms({}, {"qbyt_readout_version": 7}) == (0, 0)
    assert resolve_clip_audio_padding_ms(
        {"left_padding_ms": 0, "right_padding_ms": 240}
    ) == (0, 240)
    with pytest.raises(SystemExit, match="must be >= 0"):
        resolve_clip_audio_padding_ms({"left_padding_ms": -1})


def test_sink_diagnostics_defaults_do_not_change_other_prep_keys():
    config = config_to_dict(compose_config())
    prep = config["prep"]
    assert prep["batch_size"] == 0
    assert prep["num_workers"] == 0
    assert prep["keyword_eval"]["mode"] == "per_row"
    assert prep["audio_aug"]["transforms"]["volume_gain"]["enabled"] is False
    assert prep["musan_mix"]["noise"]["enabled"] is False
    sink = prep["sink_diagnostics"]
    assert sink["mode"] == "clips"
    assert sink["capture_layers"] == "all"
    assert sink["capture_heads"] == "all"
    assert sink["save_traces"] is True
    assert sink["save_full_attention"] is False
    assert list(sink["ablations"]) == ["block_sink_all", "block_sink_each_layer"]
    assert sink["max_combined_tokens"] == 1024
    assert sink["max_attention_bytes"] == 268435456
    assert sink["max_report_samples"] == 40
    assert sink["group_field"] == "condition"
    assert list(sink["length_bins"]) == [0, 100, 200, 400, 800]
    assert sink["synthetic_fixture"] is False


def test_t10_normal_rows_match_eval_stage2_clips_scoring_path(
    tmp_path, monkeypatch, install_runner
):
    runner = install_runner
    audio_a = _write_wav(tmp_path / "a.wav", 1.0)
    audio_b = _write_wav(tmp_path / "b.wav", 1.0)
    manifest = _write_csv(
        tmp_path / "manifest.csv",
        "audio_path,keyword,label",
        f"{audio_a.name},hey eva,1",
        f"{audio_b.name},hey eva,0",
        f"{audio_a.name},ok google,1",
    )
    from dma_kws.inference.manifest import load_manifest

    rows = load_manifest(manifest)
    eval_results = runner.run_batch(
        rows,
        batch_size=1,
        num_workers=0,
        left_padding_ms=0,
        right_padding_ms=0,
    )
    summary = _run_diagnose(monkeypatch, _cfg(tmp_path, manifest=manifest))
    assert summary["status"] == "complete"

    records = _read_csv(tmp_path / "out" / "records.csv")
    normal = [row for row in records if row["ablation"] == "normal"]
    assert [row["manifest_record_number"] for row in normal] == ["2", "3", "4"]
    assert [row["audio_path"] for row in normal] == [row["audio_path"] for row in rows]
    assert Path(normal[0]["audio_path"]).is_absolute()
    assert Path(normal[2]["audio_path"]) == Path(normal[0]["audio_path"])
    assert normal[0]["keyword"] != normal[2]["keyword"]
    assert [row["label"] for row in normal] == ["1", "0", "1"]

    for eval_row, diag_row in zip(eval_results, normal):
        assert diag_row["keyword_phonemes"].split() == eval_row["keyword_phonemes"]
        assert float(diag_row["qbyt_raw_logit"]) == pytest.approx(
            float(eval_row["qbyt_raw_logit"]), abs=PARITY_ATOL, rel=PARITY_RTOL
        )
        assert float(diag_row["qbyt_score"]) == pytest.approx(
            float(eval_row["qbyt_score"]), abs=PARITY_ATOL, rel=PARITY_RTOL
        )
        assert diag_row["delta_raw_logit"] == "0" or float(diag_row["delta_raw_logit"]) == 0.0
        assert float(diag_row["delta_qbyt_score"]) == pytest.approx(0.0)


def test_t11_empty_label_stays_unlabeled_and_illegal_fields_error(
    tmp_path, monkeypatch, install_runner
):
    wav = _write_wav(tmp_path / "clip.wav", 1.0)
    unlabeled = _write_csv(
        tmp_path / "unlabeled.csv",
        "audio_path,keyword,label",
        f"{wav.name},hey eva,",
    )
    summary = _run_diagnose(monkeypatch, _cfg(tmp_path, manifest=unlabeled))
    assert summary["status"] == "complete"
    assert summary["num_unlabeled"] == 1
    normal = [
        row
        for row in _read_csv(tmp_path / "out" / "records.csv")
        if row["ablation"] == "normal"
    ]
    assert normal[0]["label"] == ""

    bad_label = _write_csv(
        tmp_path / "bad_label.csv",
        "audio_path,keyword,label",
        f"{wav.name},hey eva,2",
    )
    with pytest.raises(SystemExit, match="label"):
        _run_diagnose(monkeypatch, _cfg(tmp_path, manifest=bad_label, output_dir=str(tmp_path / "bad_label")))

    empty_phones = _write_csv(
        tmp_path / "empty_phones.csv",
        "audio_path,keyword,keyword_phonemes",
        f"{wav.name},hey eva,",
    )
    with pytest.raises(SystemExit, match="keyword_phonemes"):
        _run_diagnose(
            monkeypatch,
            _cfg(tmp_path, manifest=empty_phones, output_dir=str(tmp_path / "empty_phones")),
        )

    nan_spans = _write_csv(
        tmp_path / "nan_spans.csv",
        "audio_path,keyword,keyword_spans",
        f'{wav.name},hey eva,"[[NaN, 1.0]]"',
    )
    with pytest.raises(SystemExit, match="keyword_spans"):
        _run_diagnose(
            monkeypatch,
            _cfg(tmp_path, manifest=nan_spans, output_dir=str(tmp_path / "nan_spans")),
        )

    dup = _write_csv(
        tmp_path / "dup.csv",
        "audio_path,keyword,sample_id",
        f"{wav.name},hey eva,same",
        f"{wav.name},hey eva,same",
    )
    with pytest.raises(SystemExit, match="sample_id"):
        _run_diagnose(monkeypatch, _cfg(tmp_path, manifest=dup, output_dir=str(tmp_path / "dup")))

    any_cfg = _cfg(tmp_path, manifest=unlabeled, output_dir=str(tmp_path / "any"))
    any_cfg.prep.keyword_eval.mode = "any"
    with pytest.raises(SystemExit, match="mode=any"):
        _run_diagnose(monkeypatch, any_cfg)


def test_t11_and_cli_guards_reject_aug_amp_windows_unknown_keys_and_missing_group(
    tmp_path, monkeypatch, install_runner
):
    wav = _write_wav(tmp_path / "clip.wav", 1.0)
    manifest = _write_csv(
        tmp_path / "manifest.csv",
        "audio_path,keyword,label",
        f"{wav.name},hey eva,1",
    )

    aug_cfg = _cfg(
        tmp_path,
        manifest=manifest,
        output_dir=str(tmp_path / "aug"),
        audio_aug={"transforms": {"volume_gain": {"enabled": True}}},
    )
    with pytest.raises(SystemExit, match="exported fixed audio"):
        _run_diagnose(monkeypatch, aug_cfg)

    musan_cfg = _cfg(
        tmp_path,
        manifest=manifest,
        output_dir=str(tmp_path / "musan"),
        musan_mix={"noise": {"enabled": True, "snr_db": 20.0}},
    )
    with pytest.raises(SystemExit, match="exported fixed audio"):
        _run_diagnose(monkeypatch, musan_cfg)

    amp_cfg = _cfg(tmp_path, manifest=manifest, output_dir=str(tmp_path / "amp"), amp="fp16")
    with pytest.raises(SystemExit, match="unsupported|FP32|amp"):
        _run_diagnose(monkeypatch, amp_cfg)

    windows_cfg = _cfg(
        tmp_path,
        manifest=manifest,
        output_dir=str(tmp_path / "windows"),
        sink=_sink_defaults(mode="windows"),
    )
    with pytest.raises(SystemExit, match="stage D not implemented"):
        _run_diagnose(monkeypatch, windows_cfg)

    unknown_cfg = _cfg(
        tmp_path,
        manifest=manifest,
        output_dir=str(tmp_path / "unknown"),
        sink=_sink_defaults(not_a_real_key=1),
    )
    with pytest.raises(SystemExit, match="unknown"):
        _run_diagnose(monkeypatch, unknown_cfg)

    missing_group = _cfg(
        tmp_path,
        manifest=manifest,
        output_dir=str(tmp_path / "group"),
        sink=_sink_defaults(group_field="variant"),
    )
    with pytest.raises(SystemExit, match="group_field"):
        _run_diagnose(monkeypatch, missing_group)


def test_resolve_sink_diagnostics_rejects_bool_and_float_limits():
    import scripts.diagnose_qbyt_sink as diagnose

    for key in (
        "max_combined_tokens",
        "max_attention_bytes",
        "max_report_samples",
        "plot_dpi",
    ):
        with pytest.raises(SystemExit, match="integer"):
            diagnose.resolve_sink_diagnostics({"sink_diagnostics": {key: True}})
        with pytest.raises(SystemExit, match="integer"):
            diagnose.resolve_sink_diagnostics({"sink_diagnostics": {key: 1.5}})


def test_batch_size_zero_becomes_one_not_eval_default(tmp_path, monkeypatch, install_runner):
    import scripts.diagnose_qbyt_sink as diagnose

    assert diagnose.resolve_diagnostic_batch_size({"batch_size": 0}) == 1
    assert diagnose.resolve_diagnostic_batch_size({"batch_size": -3}) == 1
    assert diagnose.resolve_diagnostic_num_workers({"num_workers": 0}) == 0
    wav = _write_wav(tmp_path / "clip.wav", 1.0)
    manifest = _write_csv(
        tmp_path / "manifest.csv",
        "audio_path,keyword,label",
        f"{wav.name},hey eva,1",
    )
    cfg = _cfg(tmp_path, manifest=manifest, batch_size=0, num_workers=0)
    summary = _run_diagnose(monkeypatch, cfg)
    assert summary["status"] == "complete"
    assert summary["batch_size"] == 1
    assert summary["num_workers"] == 0
    assert summary["max_parity_error"] is not None
    assert summary["max_parity_error"] >= 0.0
    assert math.isfinite(summary["max_parity_error"])


def test_num_workers_forced_zero_when_waveform_observer_installed(
    tmp_path, monkeypatch, install_runner
):
    wav = _write_wav(tmp_path / "clip.wav", 1.0)
    manifest = _write_csv(
        tmp_path / "manifest.csv",
        "audio_path,keyword,label",
        f"{wav.name},hey eva,1",
    )
    summary = _run_diagnose(
        monkeypatch, _cfg(tmp_path, manifest=manifest, num_workers=2)
    )
    assert summary["status"] == "complete"
    assert summary["num_workers"] == 0
    traces = list((tmp_path / "out" / "traces").glob("*__normal.npz"))
    assert traces
    with np.load(traces[0], allow_pickle=False) as payload:
        assert "prepared_waveform" in payload


def test_t12_pairs_and_span_duration_after_decode(tmp_path, monkeypatch, install_runner):
    clean = _write_wav(tmp_path / "clean.wav", 2.0)
    noisy = _write_wav(tmp_path / "noisy.wav", 2.0)
    extra = _write_wav(tmp_path / "extra.wav", 2.0)
    manifest = _write_csv(
        tmp_path / "pairs.csv",
        "audio_path,keyword,label,sample_id,condition,pair_id,keyword_spans,noise_spans",
        f'{clean.name},hey eva,1,clean_001,clean,p001,"[[0.8,1.6]]",[]',
        f'{noisy.name},hey eva,1,noisy_001,noisy,p001,"[[0.8,1.6]]","[[0.5,2.0]]"',
        f'{extra.name},hey eva,0,noise_only,noise_only,p_missing,"[[0.0,1.0]]","[[0.0,2.0]]"',
    )
    summary = _run_diagnose(monkeypatch, _cfg(tmp_path, manifest=manifest))
    assert summary["status"] == "complete"
    pairs = _read_csv(tmp_path / "out" / "pairs.csv")
    statuses = {row["pair_status"] for row in pairs}
    assert "ok" in statuses
    assert "missing_clean_baseline" in statuses
    ok_row = next(row for row in pairs if row["pair_status"] == "ok")
    assert ok_row["baseline_sample_id"] == "clean_001"
    assert ok_row["variant_sample_id"] == "noisy_001"
    assert ok_row["time_grid_comparable"] == "true"
    metrics = _read_csv(tmp_path / "out" / "attention_metrics.csv")
    regions = {row["region"] for row in metrics}
    assert "noise" in regions
    assert "outside_noise_annotation" in regions
    for row in metrics:
        if row["region"] in {"noise", "outside_noise_annotation"} and row["query_count"] == "0":
            assert row["mean"] == ""
    assert ok_row["normal_score_delta"] != ""
    missing = next(row for row in pairs if row["pair_status"] == "missing_clean_baseline")
    assert missing["pair_reason"]
    assert "interpolat" not in (missing.get("pair_reason") or "").lower()

    run = json.loads((tmp_path / "out" / "run.json").read_text(encoding="utf-8"))
    assert run["audio_padding_ms"] == {"left": 0, "right": 0}

    overflow = _write_csv(
        tmp_path / "overflow.csv",
        "audio_path,keyword,keyword_spans",
        f'{clean.name},hey eva,"[[0.0, 3.0]]"',
    )
    with pytest.raises(SystemExit, match="duration"):
        _run_diagnose(
            monkeypatch,
            _cfg(tmp_path, manifest=overflow, output_dir=str(tmp_path / "overflow")),
        )


def test_t13_empty_all_skipped_zero_frames_and_over_budget(
    tmp_path, monkeypatch, install_runner
):
    empty = _write_csv(tmp_path / "empty.csv", "audio_path,keyword,label")
    empty_summary = _run_diagnose(
        monkeypatch, _cfg(tmp_path, manifest=empty, output_dir=str(tmp_path / "empty"))
    )
    assert empty_summary["status"] == "complete"
    assert empty_summary["num_input"] == 0
    assert empty_summary["num_success"] == 0
    assert (tmp_path / "empty" / "report.html").is_file()
    assert (tmp_path / "empty" / "records.csv").is_file()
    assert _read_csv(tmp_path / "empty" / "records.csv") == []

    short = _write_wav(tmp_path / "tiny.wav", 0.001)
    skipped_manifest = _write_csv(
        tmp_path / "skipped.csv",
        "audio_path,keyword,label",
        f"{short.name},hey eva,1",
    )
    skipped_summary = _run_diagnose(
        monkeypatch,
        _cfg(tmp_path, manifest=skipped_manifest, output_dir=str(tmp_path / "skipped")),
    )
    assert skipped_summary["status"] == "complete"
    assert skipped_summary["num_skipped"] >= 1
    skipped_rows = _read_csv(tmp_path / "skipped" / "records.csv")
    assert skipped_rows
    for row in skipped_rows:
        assert row["status"] == "skipped"
        assert row["skip_reason"]
        assert row["qbyt_raw_logit"] == ""
        assert row["qbyt_score"] == ""
        assert row["detected"] == ""
        assert row["delta_raw_logit"] == ""
        assert row["delta_qbyt_score"] == ""

    long_wav = _write_wav(tmp_path / "long.wav", 1.0)
    budget_manifest = _write_csv(
        tmp_path / "budget.csv",
        "audio_path,keyword,label",
        f"{long_wav.name},hey eva,1",
    )
    budget_summary = _run_diagnose(
        monkeypatch,
        _cfg(
            tmp_path,
            manifest=budget_manifest,
            output_dir=str(tmp_path / "budget"),
            sink=_sink_defaults(max_combined_tokens=4, ablations=[]),
        ),
    )
    assert budget_summary["status"] == "complete"
    budget_rows = _read_csv(tmp_path / "budget" / "records.csv")
    assert budget_rows
    assert all(row["status"] == "skipped" for row in budget_rows)
    assert all(row["skip_reason"] == "resource_limit" for row in budget_rows)
    assert all(row["qbyt_score"] == "" for row in budget_rows)


def test_t14_csv_quoting_npz_json_finite_and_refuse_existing_output(
    tmp_path, monkeypatch, install_runner
):
    wav = _write_wav(tmp_path / "clip, name.wav", 1.0)
    manifest = _write_csv(
        tmp_path / "quoted.csv",
        "audio_path,keyword,label,sample_id,condition",
        f'"{wav.name}","hey, eva",1,"id,1","clean,room"',
    )
    cfg = _cfg(tmp_path, manifest=manifest)
    summary = _run_diagnose(monkeypatch, cfg)
    assert summary["status"] == "complete"
    out = tmp_path / "out"
    records = _read_csv(out / "records.csv")
    assert records[0]["keyword"] == "hey, eva"
    assert records[0]["sample_id"] == "id,1"
    assert records[0]["condition"] == "clean,room"
    assert set(records[0]) >= set(RECORDS_FIELDS)

    trace_rows = [row for row in records if row["trace_path"]]
    assert trace_rows
    for row in trace_rows:
        npz_path = out / row["trace_path"]
        with np.load(npz_path, allow_pickle=False) as payload:
            assert "audio_to_sink" in payload
            assert "text_to_sink" in payload
            assert payload["audio_to_sink"].dtype != object

    run = json.loads((out / "run.json").read_text(encoding="utf-8"))
    dumped = json.dumps(run, allow_nan=False)
    assert dumped
    assert run["schema_version"]
    assert "git_commit" in run
    assert run["manifest_sha256"]
    assert run["checkpoint_sha256"]
    assert run["tokenizer"]["sha256"]
    assert run["qbyt_readout"]["version"] == 4
    assert run["qbyt_readout"]["sink_token"] is True
    assert run["expanded_ablations"]
    assert run["time_axis_method"] != "pending"
    assert run["time_axis_status"] in {"ok", "unavailable"}
    assert run["time_axis"]["fbank"]["frame_shift_ms"] == 10.0
    sample_meta = next(iter(run["samples"].values()))
    assert sample_meta.get("num_fbank_frames")
    summary_json = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    json.dumps(summary_json, allow_nan=False)
    assert summary_json["status"] == "complete"
    html = (out / "report.html").read_text(encoding="utf-8")
    assert "hey, eva" not in html or "&" in html or "hey, eva" in html
    assert "<script src=\"https://" not in html
    assert "合成 fixture" not in html
    assert run["synthetic_fixture"] is False
    assert summary_json["synthetic_fixture"] is False
    assert "banner" not in run
    assert "banner" not in summary_json

    with pytest.raises(SystemExit, match="output"):
        _run_diagnose(monkeypatch, cfg)


def test_synthetic_fixture_stamps_json_and_html_banner(
    tmp_path, monkeypatch, install_runner
):
    from dma_kws.inference.qbyt_attention_report import SYNTHETIC_FIXTURE_BANNER

    wav = _write_wav(tmp_path / "clip.wav", 1.0)
    manifest = _write_csv(
        tmp_path / "manifest.csv",
        "audio_path,keyword,label",
        f"{wav.name},hey eva,1",
    )
    summary = _run_diagnose(
        monkeypatch,
        _cfg(
            tmp_path,
            manifest=manifest,
            sink=_sink_defaults(synthetic_fixture=True, ablations=[]),
        ),
    )
    assert summary["status"] == "complete"
    assert summary["synthetic_fixture"] is True
    assert summary["banner"] == SYNTHETIC_FIXTURE_BANNER
    out = tmp_path / "out"
    run = json.loads((out / "run.json").read_text(encoding="utf-8"))
    summary_json = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    html = (out / "report.html").read_text(encoding="utf-8")
    assert run["synthetic_fixture"] is True
    assert run["banner"] == SYNTHETIC_FIXTURE_BANNER
    assert run["sink_diagnostics"]["synthetic_fixture"] is True
    assert summary_json["synthetic_fixture"] is True
    assert summary_json["banner"] == SYNTHETIC_FIXTURE_BANNER
    assert SYNTHETIC_FIXTURE_BANNER in html
    assert 'class="synthetic-banner"' in html
    assert "Not a trained-model conclusion" in html
    assert "sink learned to reject noise" in html


def test_missing_audio_fails_the_run(tmp_path, monkeypatch, install_runner):
    manifest = _write_csv(
        tmp_path / "missing.csv",
        "audio_path,keyword,label",
        "no_such_file.wav,hey eva,1",
    )
    with pytest.raises(SystemExit, match="audio|not found|No such"):
        _run_diagnose(monkeypatch, _cfg(tmp_path, manifest=manifest))


def test_group_value_reads_known_columns_not_only_extra_fields(tmp_path):
    import scripts.diagnose_qbyt_sink as diagnose
    from dma_kws.inference.manifest import load_manifest
    from dma_kws.inference.qbyt_attention_manifest import (
        validate_attention_manifest_rows,
    )

    manifest = _write_csv(
        tmp_path / "groups.csv",
        "audio_path,keyword,label,sample_id,condition,pair_id,pronunciation_id,note",
        "a.wav,hey eva,1,clean_001,clean,p001,prA,keep-extra",
        "b.wav,ok google,0,noisy_001,noisy,p001,prB,keep-extra",
        "c.wav,hey eva,1,solo_001,clean,,,keep-extra",
    )
    rows = validate_attention_manifest_rows(load_manifest(manifest))
    assert "keyword" not in rows[0].extra_fields
    assert "pair_id" not in rows[0].extra_fields
    assert rows[0].extra_fields["note"] == "keep-extra"
    assert diagnose._group_value(rows[0], "keyword") == "hey eva"
    assert diagnose._group_value(rows[1], "keyword") == "ok google"
    assert diagnose._group_value(rows[0], "pair_id") == "p001"
    assert diagnose._group_value(rows[0], "pronunciation_id") == "prA"
    assert diagnose._group_value(rows[0], "sample_id") == "clean_001"
    assert diagnose._group_value(rows[0], "condition") == "clean"
    assert diagnose._group_value(rows[0], "label") == "1"
    assert diagnose._group_value(rows[2], "pair_id") == "unknown"
    assert diagnose._group_value(rows[0], "note") == "keep-extra"


def test_group_field_keyword_and_pair_id_appear_in_summary(
    tmp_path, monkeypatch, install_runner
):
    wav_a = _write_wav(tmp_path / "a.wav", 1.0)
    wav_b = _write_wav(tmp_path / "b.wav", 1.0)
    manifest = _write_csv(
        tmp_path / "manifest.csv",
        "audio_path,keyword,label,sample_id,condition,pair_id",
        f"{wav_a.name},hey eva,1,clean_001,clean,p001",
        f"{wav_b.name},ok google,0,other_001,noisy,p002",
    )
    keyword_summary = _run_diagnose(
        monkeypatch,
        _cfg(
            tmp_path,
            manifest=manifest,
            output_dir=str(tmp_path / "by_keyword"),
            sink=_sink_defaults(group_field="keyword", ablations=[]),
        ),
    )
    assert set(keyword_summary["group_metrics"]) == {"hey eva", "ok google"}
    assert all(entry["n"] == 1 for entry in keyword_summary["group_metrics"].values())
    assert not (tmp_path / "by_keyword" / ".partial").exists()

    pair_summary = _run_diagnose(
        monkeypatch,
        _cfg(
            tmp_path,
            manifest=manifest,
            output_dir=str(tmp_path / "by_pair"),
            sink=_sink_defaults(group_field="pair_id", ablations=[]),
        ),
    )
    assert set(pair_summary["group_metrics"]) == {"p001", "p002"}


def test_extra_group_field_is_copied_onto_records_and_group_pngs(
    tmp_path, monkeypatch, install_runner
):
    wav_a = _write_wav(tmp_path / "a.wav", 1.0)
    wav_b = _write_wav(tmp_path / "b.wav", 1.0)
    manifest = _write_csv(
        tmp_path / "manifest.csv",
        "audio_path,keyword,label,sample_id,variant",
        f"{wav_a.name},hey eva,1,clean_001,room",
        f"{wav_b.name},ok google,0,other_001,street",
    )
    summary = _run_diagnose(
        monkeypatch,
        _cfg(
            tmp_path,
            manifest=manifest,
            sink=_sink_defaults(group_field="variant", ablations=[]),
        ),
    )
    assert summary["status"] == "complete"
    assert set(summary["group_metrics"]) == {"room", "street"}
    records = _read_csv(tmp_path / "out" / "records.csv")
    assert "variant" in records[0]
    variants = {
        row["sample_id"]: row["variant"]
        for row in records
        if row["ablation"] == "normal"
    }
    assert variants == {"clean_001": "room", "other_001": "street"}
    assert (tmp_path / "out" / "figures" / "s_audio_by_variant.png").is_file()


def test_equivalent_phoneme_formats_pair_after_enrollment(
    tmp_path, monkeypatch, install_runner
):
    clean = _write_wav(tmp_path / "clean.wav", 1.0)
    noisy = _write_wav(tmp_path / "noisy.wav", 1.0)
    manifest = tmp_path / "equiv.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "audio_path",
                "keyword",
                "keyword_phonemes",
                "condition",
                "pair_id",
                "label",
            ]
        )
        writer.writerow(
            [clean.name, "hey eva", "HH EY1 IY1 V AH0", "clean", "p1", "1"]
        )
        writer.writerow(
            [
                noisy.name,
                "hey eva",
                '["HH", "EY1", "IY1", "V", "AH0"]',
                "noisy",
                "p1",
                "1",
            ]
        )
    summary = _run_diagnose(monkeypatch, _cfg(tmp_path, manifest=manifest))
    assert summary["status"] == "complete"
    assert summary["pairs"]["n_ok"] == 1
    assert summary["pairs"]["n_inconsistent"] == 0
    pairs = _read_csv(tmp_path / "out" / "pairs.csv")
    assert {row["pair_status"] for row in pairs} == {"ok"}
    ok = next(row for row in pairs if row["pair_status"] == "ok")
    assert ok["baseline_sample_id"]
    assert ok["variant_sample_id"]


def test_report_render_failure_marks_summary_failed_and_keeps_records(
    tmp_path, monkeypatch, install_runner
):
    import scripts.diagnose_qbyt_sink as diagnose

    wav = _write_wav(tmp_path / "a.wav", 1.0)
    manifest = _write_csv(
        tmp_path / "manifest.csv",
        "audio_path,keyword,label",
        f"{wav.name},hey eva,1",
    )

    def _boom(*_args, **_kwargs):
        raise RuntimeError("injected render failure")

    monkeypatch.setattr(diagnose, "render_sink_attention_report", _boom)
    with pytest.raises((SystemExit, RuntimeError), match="injected render failure"):
        _run_diagnose(monkeypatch, _cfg(tmp_path, manifest=manifest))
    out = tmp_path / "out"
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "failed"
    assert "injected render failure" in str(summary.get("error", ""))
    assert (out / "records.csv").is_file()
    assert (out / "run.json").is_file()
    assert not (out / "report.html").exists()


def test_materialize_scored_sample_drops_capture_tensors():
    import scripts.diagnose_qbyt_sink as diagnose
    from dma_kws.inference.qbyt_attention_diagnostics import (
        SampleAttentionDiagnostics,
        SampleAttentionTrace,
        SinkAblationResult,
        SinkAblationSpec,
    )
    from dma_kws.inference.qbyt_attention_manifest import AttentionManifestRow

    text_len, audio_len = 2, 3
    trace = SampleAttentionTrace(
        layer_ids=(0,),
        head_ids=(0, 1),
        audio_to_sink=torch.rand(1, 2, audio_len),
        text_to_sink=torch.rand(1, 2, text_len),
        text_to_audio=torch.rand(1, 2, text_len, audio_len),
        position_logits=torch.tensor([0.2, -0.1]),
        raw_logit=0.5,
        text_length=text_len,
        audio_length=audio_len,
        sink_index=text_len,
        row_sum_max_error=0.0,
        padding_mass_max=0.0,
    )
    spec = SinkAblationSpec(name="normal")
    result = SinkAblationResult(
        spec=spec,
        trace=trace,
        raw_logit=0.5,
        qbyt_score=0.6,
        threshold=0.5,
        detected=True,
        delta_raw_logit=0.0,
        delta_qbyt_score=0.0,
        delta_position_logits=torch.zeros(text_len),
    )
    sample = SampleAttentionDiagnostics(
        normal_raw_logit=0.5,
        normal_qbyt_score=0.6,
        threshold=0.5,
        normal=result,
        ablations=(),
    )
    row = AttentionManifestRow(
        audio_path="/tmp/a.wav",
        keyword="hey eva",
        label=1,
        keyword_phonemes=None,
        phoneme_override=None,
        sample_id="s1",
        internal_sample_id="s1_abc",
        condition="clean",
        pair_id="p001",
        keyword_spans=None,
        noise_spans=None,
        pronunciation_id="prA",
        pair_status="ok",
        pair_reason=None,
        record_number=2,
        extra_fields={},
    )
    prepared = diagnose.PreparedSample(
        row=row,
        raw_row={"audio_path": row.audio_path, "keyword": row.keyword},
        phonemes=["HH", "EY1"],
        token_ids=[1, 2],
        query_id="[1,2]",
        source_duration_sec=1.0,
    )
    outcome = diagnose._materialize_scored_sample(
        run_id="run",
        prepared=prepared,
        sample=sample,
        trace_paths={"normal": "traces/s1__normal.npz"},
    )
    assert not hasattr(outcome, "diagnostics") or outcome.__dict__.get("diagnostics") is None
    assert outcome.conditions == [] if hasattr(outcome, "conditions") else True
    assert "diagnostics" not in outcome.__dataclass_fields__
    assert "conditions" not in outcome.__dataclass_fields__
    assert outcome.normal_qbyt_score == pytest.approx(0.6)
    assert outcome.normal_detected is True
    assert outcome.record_rows[0]["ablation"] == "normal"
    assert outcome.metric_rows
    assert outcome.position_rows
    assert all("audio_to_sink" not in row for row in outcome.record_rows)
