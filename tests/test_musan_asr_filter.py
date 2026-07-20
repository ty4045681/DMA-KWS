from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from dma_kws.inference.musan_asr_filter import (
    AudioChunk,
    _patch_wenet_config,
    best_keyword_match,
    build_filter_records,
    copy_filtered_tree,
    create_audio_chunks,
    filter_musan,
    normalize_match_text,
    resolve_model_files,
    run_wenet_recognize,
    validate_filter_options,
    validate_keywords,
    write_filter_report,
)
from scripts.filter_musan_by_wenet_asr import load_keywords, resolve_device


def _make_model_dir(root: Path, *, cmvn_name: str = "global_cmvn") -> Path:
    root.mkdir()
    for name in ("model.pt", "train.yaml", cmvn_name, "unigram5000.model", "units.txt"):
        (root / name).write_text("dataset_conf: {}\n" if name == "train.yaml" else "x")
    return root


def test_resolve_model_files_discovers_expected_layout(tmp_path):
    model_dir = _make_model_dir(tmp_path / "model", cmvn_name="global_cvmn")

    files = resolve_model_files(model_dir)

    assert files.checkpoint.name == "model.pt"
    assert files.cmvn.name == "global_cvmn"
    assert files.bpe_model.name == "unigram5000.model"
    assert files.units.name == "units.txt"


def test_resolve_model_files_requires_checkpoint_override_when_ambiguous(tmp_path):
    model_dir = _make_model_dir(tmp_path / "model")
    (model_dir / "second.pt").write_text("x")

    with pytest.raises(ValueError, match="pass --checkpoint"):
        resolve_model_files(model_dir)


def test_resolve_model_files_allows_individual_overrides(tmp_path):
    paths = {}
    for name in ("checkpoint.pt", "config.yaml", "cmvn", "bpe.model", "symbols.txt"):
        paths[name] = tmp_path / name
        paths[name].write_text("x")

    files = resolve_model_files(
        checkpoint=paths["checkpoint.pt"],
        config=paths["config.yaml"],
        cmvn=paths["cmvn"],
        bpe_model=paths["bpe.model"],
        units=paths["symbols.txt"],
    )

    assert files.checkpoint == paths["checkpoint.pt"].resolve()
    assert files.units == paths["symbols.txt"].resolve()


def test_patch_wenet_config_updates_existing_layout_only(tmp_path):
    model_dir = _make_model_dir(tmp_path / "model")
    files = resolve_model_files(model_dir)
    files.config.write_text(
        "cmvn_conf:\n  cmvn_file: old\n"
        "tokenizer_conf:\n  symbol_table_path: old\n  bpe_path: old\n"
        "dict: old\n"
    )
    output = tmp_path / "resolved.yaml"

    _patch_wenet_config(files, output)

    import yaml

    config = yaml.safe_load(output.read_text())
    assert config["cmvn_conf"]["cmvn_file"] == str(files.cmvn)
    assert config["tokenizer_conf"]["symbol_table_path"] == str(files.units)
    assert config["tokenizer_conf"]["bpe_path"] == str(files.bpe_model)
    assert config["dict"] == str(files.units)
    assert "symbol_table" not in config
    assert "bpe_model" not in config


def test_normalize_and_fuzzy_match_multiple_keywords():
    assert normalize_match_text(" Hey, E_V_A! ") == "heyeva"
    keyword, score = best_keyword_match("hey ever", ["hey android", "hey eva"])

    assert keyword == "hey eva"
    assert score >= 85
    assert validate_keywords(["Hey Eva", " hey eva "]) == ["Hey Eva"]


def test_validate_filter_options_rejects_invalid_values():
    defaults = {
        "threshold": 85,
        "window_sec": 30,
        "overlap_sec": 1,
        "batch_size": 8,
        "beam_size": 10,
        "mode": "attention_rescoring",
    }
    for name, value in (
        ("threshold", 101),
        ("window_sec", 0),
        ("overlap_sec", 30),
        ("batch_size", 0),
        ("beam_size", -1),
        ("mode", ""),
    ):
        options = {**defaults, name: value}
        with pytest.raises(ValueError):
            validate_filter_options(**options)


def test_create_audio_chunks_uses_window_and_overlap(tmp_path):
    sf = pytest.importorskip("soundfile")
    wav = tmp_path / "long.wav"
    sample_rate = 100
    sf.write(wav, np.zeros((6500,), dtype=np.float32), sample_rate)

    chunks = create_audio_chunks([wav], tmp_path / "chunks", window_sec=30, overlap_sec=1)

    assert [(chunk.start_sec, chunk.end_sec) for chunk in chunks] == [
        (0.0, 30.0),
        (29.0, 59.0),
        (58.0, 65.0),
    ]
    assert all(chunk.path.is_file() for chunk in chunks)


def test_build_records_removes_whole_file_when_any_window_matches(tmp_path):
    musan = tmp_path / "musan"
    first = musan / "speech" / "first.wav"
    second = musan / "noise" / "second.wav"
    first.parent.mkdir(parents=True)
    second.parent.mkdir(parents=True)
    first.write_bytes(b"")
    second.write_bytes(b"")
    chunks = [
        AudioChunk("a0", first, tmp_path / "a0.flac", 0.0, 30.0),
        AudioChunk("a1", first, tmp_path / "a1.flac", 29.0, 40.0),
        AudioChunk("b0", second, tmp_path / "b0.flac", 0.0, 10.0),
    ]

    records = build_filter_records(
        [first, second],
        chunks,
        {"a0": "ordinary speech", "a1": "hey ever", "b0": "background noise"},
        ["hey eva", "hey android"],
        threshold=85,
        musan_root=musan,
    )

    assert records[0]["matched"] is True
    assert records[0]["matched_keyword"] == "hey eva"
    assert records[1]["matched"] is False
    assert records[1]["subset"] == "noise"


def test_copy_filtered_tree_preserves_layout_and_non_audio(tmp_path):
    source = tmp_path / "musan"
    (source / "speech").mkdir(parents=True)
    (source / "noise" / "empty").mkdir(parents=True)
    (source / "speech" / "remove.wav").write_bytes(b"remove")
    (source / "speech" / "keep.wav").write_bytes(b"keep")
    (source / "README.txt").write_text("metadata")
    destination = tmp_path / "filtered"
    records = [
        {"relative_path": "speech/remove.wav", "matched": True},
        {"relative_path": "speech/keep.wav", "matched": False},
    ]

    copy_filtered_tree(source, destination, records)

    assert not (destination / "speech" / "remove.wav").exists()
    assert (destination / "speech" / "keep.wav").read_bytes() == b"keep"
    assert (destination / "README.txt").read_text() == "metadata"
    assert (destination / "noise" / "empty").is_dir()


def test_copy_filtered_tree_rejects_existing_or_nested_output(tmp_path):
    source = tmp_path / "musan"
    source.mkdir()
    existing = tmp_path / "existing"
    existing.mkdir()

    with pytest.raises(FileExistsError):
        copy_filtered_tree(source, existing, [])
    with pytest.raises(ValueError, match="inside"):
        copy_filtered_tree(source, source / "filtered", [])


def test_write_filter_report_records_summary(tmp_path):
    report_dir = tmp_path / "report"
    records = [
        {"relative_path": "speech/a.wav", "subset": "speech", "matched": True, "matched_keyword": "hey eva"},
        {"relative_path": "noise/b.wav", "subset": "noise", "matched": False, "matched_keyword": None},
    ]

    summary = write_filter_report(report_dir, records, summary_metadata={"threshold": 85})

    assert summary["removed_audio_files"] == 1
    assert summary["kept_audio_files"] == 1
    assert summary["keyword_hits"] == {"hey eva": 1}
    assert summary["subsets"]["speech"]["removed"] == 1
    rows = [json.loads(line) for line in (report_dir / "results.jsonl").read_text().splitlines()]
    assert rows == records


def test_run_wenet_recognize_supports_legacy_cli_paths(tmp_path, monkeypatch):
    model_dir = _make_model_dir(tmp_path / "model")
    files = resolve_model_files(model_dir)
    chunk_path = tmp_path / "chunk.flac"
    chunk_path.write_bytes(b"audio")
    source = tmp_path / "source.wav"
    chunk = AudioChunk("chunk0", source, chunk_path, 0.0, 1.0)
    captured = {}

    monkeypatch.setattr(
        "dma_kws.inference.musan_asr_filter._recognize_help",
        lambda _env: "--mode --result_file --dict --bpe_model --cmvn_file --gpu",
    )

    def fake_run(command, **kwargs):
        del kwargs
        captured["command"] = command
        result_path = Path(command[command.index("--result_file") + 1])
        result_path.write_text("chunk0 hey eva\n")

    monkeypatch.setattr("dma_kws.inference.musan_asr_filter.subprocess.run", fake_run)

    transcripts = run_wenet_recognize([chunk], files, tmp_path / "workspace", device="cuda:2")

    command = captured["command"]
    assert transcripts == {"chunk0": "hey eva"}
    assert command[command.index("--gpu") + 1] == "2"
    assert command[command.index("--dict") + 1] == str(files.units)
    assert command[command.index("--bpe_model") + 1] == str(files.bpe_model)
    assert command[command.index("--cmvn_file") + 1] == str(files.cmvn)


def test_run_wenet_recognize_supports_current_cli_paths(tmp_path, monkeypatch):
    model_dir = _make_model_dir(tmp_path / "model")
    files = resolve_model_files(model_dir)
    chunk_path = tmp_path / "chunk.flac"
    chunk_path.write_bytes(b"audio")
    chunk = AudioChunk("chunk0", tmp_path / "source.wav", chunk_path, 0.0, 1.0)

    monkeypatch.setattr(
        "dma_kws.inference.musan_asr_filter._recognize_help",
        lambda _env: "--modes --result_dir --device --gpu",
    )

    def fake_run(command, **kwargs):
        del kwargs
        result_dir = Path(command[command.index("--result_dir") + 1])
        transcript = result_dir / "attention_rescoring" / "text"
        transcript.parent.mkdir(parents=True)
        transcript.write_text("chunk0 hey android\n")

    monkeypatch.setattr("dma_kws.inference.musan_asr_filter.subprocess.run", fake_run)

    transcripts = run_wenet_recognize([chunk], files, tmp_path / "workspace")

    assert transcripts == {"chunk0": "hey android"}


def test_filter_musan_end_to_end_with_mock_decoder(tmp_path, monkeypatch):
    sf = pytest.importorskip("soundfile")
    source = tmp_path / "musan"
    (source / "speech").mkdir(parents=True)
    (source / "noise").mkdir(parents=True)
    sf.write(source / "speech" / "remove.wav", np.zeros(1600, dtype=np.float32), 16000)
    sf.write(source / "noise" / "keep.wav", np.zeros(1600, dtype=np.float32), 16000)
    (source / "LICENSE").write_text("license")
    model_dir = _make_model_dir(tmp_path / "model")
    files = resolve_model_files(model_dir)

    def fake_recognize(chunks, *_args, **_kwargs):
        return {
            chunk.key: "hey eva" if chunk.source.name == "remove.wav" else "rain and wind"
            for chunk in chunks
        }

    monkeypatch.setattr("dma_kws.inference.musan_asr_filter.run_wenet_recognize", fake_recognize)
    output = tmp_path / "filtered"
    report = tmp_path / "report"

    summary = filter_musan(
        musan_root=source,
        output_root=output,
        report_dir=report,
        files=files,
        keywords=["hey eva", "hey android"],
    )

    assert summary["removed_audio_files"] == 1
    assert not (output / "speech" / "remove.wav").exists()
    assert (output / "noise" / "keep.wav").is_file()
    assert (output / "LICENSE").read_text() == "license"


def test_cli_keyword_file_and_device_validation(tmp_path):
    keyword_file = tmp_path / "keywords.txt"
    keyword_file.write_text("# wake words\nhey eva\n\nhey android\n")

    assert load_keywords(["hi galaxy"], str(keyword_file)) == [
        "hi galaxy",
        "hey eva",
        "hey android",
    ]
    assert resolve_device("cpu") == "cpu"
    with pytest.raises(ValueError, match="device must"):
        resolve_device("mps")
    with pytest.raises(ValueError, match="device must"):
        resolve_device("cuda:abc")
