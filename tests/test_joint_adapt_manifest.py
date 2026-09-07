import csv
import json
from pathlib import Path

import pytest

from dma_kws.stage2.joint_manifest import (
    split_joint_samples,
    validate_background_eval_split,
    validate_joint_manifests,
)
from dma_kws.stage2.prepare_adapt import AdaptSample, prepare_keyword_adaptation


def _rows():
    return {
        f"{phase}_{split}": [
            dict(audio_path=f"raw/{phase}/{split}/{label}.wav", text="hey eva" if label else "hey ava",
                 label=label, speaker_id=f"speaker_{split}", voice_id="shared_voice", split=split,
                 session_id="session_1", recording_id=f"{split}_{label}")
            for label in (0, 1)
        ]
        for phase in ("real", "tts") for split in ("train", "eval")
    }


def _write(root, rows):
    directory = root / "manifests"
    directory.mkdir(exist_ok=True, parents=True)
    for name, values in rows.items():
        fields = sorted({key for row in values for key in row}) or ["audio_path", "text", "label"]
        with (directory / f"{name}.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(values)


def test_joint_validates_four_manifests_without_loading_raw_audio(tmp_path):
    rows = _rows()
    # IDs are scoped by phase, so a real train speaker can share a synthetic
    # eval name; local session_1 is also allowed for different speakers.
    for row in rows["tts_eval"]:
        row["speaker_id"] = "speaker_train"
        row["session_id"] = "session_eval"
    _write(tmp_path, rows)
    paths = validate_joint_manifests(tmp_path)
    assert set(paths) == set(rows)
    assert paths["real_train"] == tmp_path / "manifests/real_train.csv"


def test_joint_requires_every_source_manifest(tmp_path):
    _write(tmp_path, _rows())
    (tmp_path / "manifests/tts_train.csv").unlink()
    with pytest.raises(FileNotFoundError, match="tts_train"):
        validate_joint_manifests(tmp_path)


@pytest.mark.parametrize("field,value,match", [
    ("audio_path", "", "audio_path"), ("text", "NaN", "text"),
    ("label", "-1", "label"), ("label", "0.5", "label"), ("label", "nan", "label"),
    ("split", "eval", "expected 'train'"), ("phase", "tts", "expected 'real'"),
])
def test_joint_rejects_invalid_rows(tmp_path, field, value, match):
    rows = _rows()
    rows["real_train"][0][field] = value
    _write(tmp_path, rows)
    with pytest.raises(ValueError, match=match):
        validate_joint_manifests(tmp_path)


@pytest.mark.parametrize("name,values", [("tts_train", []), ("real_eval", [_rows()["real_eval"][1]])])
def test_joint_requires_nonempty_balanced_sources(tmp_path, name, values):
    rows = _rows()
    rows[name] = values
    _write(tmp_path, rows)
    with pytest.raises(ValueError, match=name):
        validate_joint_manifests(tmp_path)


def test_joint_requires_text_column(tmp_path):
    rows = _rows()
    for row in rows["tts_train"]:
        del row["text"]
    _write(tmp_path, rows)
    with pytest.raises(ValueError, match="missing columns.*text"):
        validate_joint_manifests(tmp_path)


def test_joint_rejects_cross_phase_pronunciation_conflict(tmp_path):
    rows = _rows()
    rows["real_train"][0]["keyword_phonemes"] = "HH EY1 IY1 V AH0"
    rows["tts_eval"][0]["keyword_phonemes"] = "HH EY1 EH1 V AH0"
    _write(tmp_path, rows)
    with pytest.raises(ValueError, match="keyword_phonemes"):
        validate_joint_manifests(tmp_path)
    rows["tts_eval"][0]["keyword_phonemes"] = " HH  EY1 IY1 V AH0 "
    _write(tmp_path, rows)
    validate_joint_manifests(tmp_path)


def test_joint_normalizes_audio_paths_against_data_root(tmp_path):
    rows = _rows()
    rows["tts_eval"][0]["audio_path"] = str(tmp_path / "raw/real/train/../train/0.wav")
    _write(tmp_path, rows)
    with pytest.raises(ValueError, match="audio/source path"):
        validate_joint_manifests(tmp_path)


def test_joint_resolves_symlink_leakage(tmp_path):
    rows = _rows()
    source = tmp_path / rows["real_train"][0]["audio_path"]
    source.parent.mkdir(parents=True)
    source.touch()
    linked = tmp_path / "linked.wav"
    linked.symlink_to(source)
    rows["tts_eval"][0]["audio_path"] = str(linked)
    _write(tmp_path, rows)
    with pytest.raises(ValueError, match="audio/source path"):
        validate_joint_manifests(tmp_path)


@pytest.mark.parametrize("field", ["source_id", "recording_id", "original_recording_id"])
def test_joint_rejects_tts_original_recording_leakage(tmp_path, field):
    rows = _rows()
    rows["tts_train"][0][field] = "original"
    rows["tts_eval"][0][field] = "original"
    _write(tmp_path, rows)
    with pytest.raises(ValueError, match=field):
        validate_joint_manifests(tmp_path)


def test_joint_rejects_cross_phase_derivative_leakage(tmp_path):
    rows = _rows()
    rows["tts_eval"][0]["source_audio_path"] = rows["real_train"][0]["audio_path"]
    _write(tmp_path, rows)
    with pytest.raises(ValueError, match="audio/source path"):
        validate_joint_manifests(tmp_path)


def test_joint_rejects_real_speaker_leakage(tmp_path):
    rows = _rows()
    rows["real_eval"][0]["speaker_id"] = " SPEAKER_TRAIN "
    rows["real_eval"][0].pop("session_id")
    _write(tmp_path, rows)
    with pytest.raises(ValueError, match="speaker_id"):
        validate_joint_manifests(tmp_path)


def _samples():
    return [
        AdaptSample(audio_path=f"raw/{phase}/{speaker}/{label}/{variant}.wav",
                    text="hey eva" if label else "hey ava", label=label, phase=phase,
                    metadata={"speaker_id": f"speaker_{speaker}", "voice_id": "same_voice",
                              "source_id": f"source_{speaker}_{label}", "session_id": "1"})
        for phase in ("real", "tts") for speaker in range(4)
        for label in (0, 1) for variant in range(2)
    ]


def test_joint_split_preserves_all_samples_speakers_and_derivatives(tmp_path):
    samples = _samples()
    splits = split_joint_samples(samples, data_root=tmp_path, eval_fraction=0.25, seed=7)
    repeat = split_joint_samples(samples, data_root=tmp_path, eval_fraction=0.25, seed=7)
    assert splits == repeat
    assert sum(len(part) for pair in splits.values() for part in pair) == len(samples)
    for phase, (train, evaluation) in splits.items():
        assert {row.label for row in train} == {row.label for row in evaluation} == {0, 1}
        assert {row.metadata["source_id"] for row in train}.isdisjoint(
            row.metadata["source_id"] for row in evaluation
        )
        if phase == "real":
            assert {row.metadata["speaker_id"] for row in train}.isdisjoint(
                row.metadata["speaker_id"] for row in evaluation
            )


def test_joint_split_does_not_use_voice_as_recording_group(tmp_path):
    samples = _samples()
    for sample in samples:
        if sample.phase == "tts":
            sample.metadata.pop("speaker_id")
            sample.metadata.pop("session_id")
    splits = split_joint_samples(samples, data_root=tmp_path, eval_fraction=0.25, seed=2)
    assert all(splits["tts"])
    assert {sample.metadata["voice_id"] for sample in splits["tts"][0]} == {"same_voice"}


def test_joint_split_groups_cross_phase_original_paths(tmp_path):
    samples = _samples()
    for sample in samples:
        if sample.phase == "tts":
            sample.metadata["original_audio_path"] = sample.audio_path.replace("/tts/", "/real/")
    splits = split_joint_samples(samples, data_root=tmp_path, eval_fraction=0.25, seed=3)
    real_eval = {sample.audio_path for sample in splits["real"][1]}
    assert {sample.metadata["original_audio_path"] for sample in splits["tts"][1]} == real_eval


def test_joint_split_preserves_explicit_user_assignment(tmp_path):
    samples = _samples()
    for sample in samples:
        sample.split = "eval" if sample.metadata["speaker_id"] == "speaker_0" else "train"
    splits = split_joint_samples(samples, data_root=tmp_path, eval_fraction=0.9, seed=4)
    for train, evaluation in splits.values():
        assert all(sample.split == "train" for sample in train)
        assert all(sample.split == "eval" for sample in evaluation)


def test_joint_split_rejects_insufficient_independent_speakers(tmp_path):
    samples = _samples()
    for sample in samples:
        if sample.phase == "real":
            sample.metadata["speaker_id"] = "only_speaker"
    with pytest.raises(ValueError, match="independent recording groups.*explicit splits"):
        split_joint_samples(samples, data_root=tmp_path, eval_fraction=0.2, seed=1)


def test_joint_preparation_checks_splits_before_extracting_features(tmp_path, monkeypatch):
    from dma_kws.stage2 import prepare_adapt

    samples = _samples()
    monkeypatch.setattr(prepare_adapt, "scan_raw_tree", lambda *_: samples)
    monkeypatch.setattr(prepare_adapt, "make_g2p", lambda: object())
    monkeypatch.setattr(prepare_adapt, "validate_g2p", lambda *_: None)
    monkeypatch.setattr(prepare_adapt, "compute_and_save_fbank", lambda *_args, **_kwargs: True)
    for sample in samples:
        path = tmp_path / sample.audio_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    stats = prepare_keyword_adaptation(keyword="hey eva", data_root=tmp_path,
                                       fbank_params={}, joint=True)
    validate_joint_manifests(tmp_path)
    assert stats["fbank_written"] == len(samples)


def test_joint_preparation_rejects_missing_configured_csv_without_fallback(tmp_path, monkeypatch):
    from dma_kws.stage2 import prepare_adapt

    def unexpected_scan(*_args, **_kwargs):
        raise AssertionError("An explicit missing source CSV must not fall back to raw data")

    monkeypatch.setattr(prepare_adapt, "scan_raw_tree", unexpected_scan)
    monkeypatch.setattr(prepare_adapt, "scan_external_sources", unexpected_scan)
    with pytest.raises(FileNotFoundError, match="source manifest.*missing.csv"):
        prepare_keyword_adaptation(keyword="hey eva", data_root=tmp_path, fbank_params={},
                                   manifest_csv=tmp_path / "missing.csv", joint=True)


def test_joint_preparation_explains_missing_source_phase(tmp_path):
    source = tmp_path / "source.csv"
    with source.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["audio_path", "text", "label"])
        writer.writeheader()
        writer.writerow(dict(audio_path="raw/real/positive.wav", text="hey eva", label=1))
    with pytest.raises(ValueError, match="source CSV must declare phase=real or phase=tts"):
        prepare_keyword_adaptation(keyword="hey eva", data_root=tmp_path, fbank_params={},
                                   manifest_csv=source, joint=True)


def _audio_list(tmp_path, name, paths):
    path = tmp_path / name
    path.write_text("\n".join(str(value) for value in paths) + "\n")
    return path


def test_joint_online_musan_split_normalizes_paths(tmp_path):
    train = tmp_path / "train.wav"
    evaluation = tmp_path / "eval.wav"
    train.touch()
    evaluation.touch()
    train_list = _audio_list(tmp_path, "train.txt", ["./train.wav"])
    eval_list = _audio_list(tmp_path, "eval.txt", [evaluation])
    config = dict(enabled=True, mode="online", audio_list_path=str(train_list))
    validate_background_eval_split(config, eval_list)
    _audio_list(tmp_path, "eval.txt", ["sub/../train.wav"])
    with pytest.raises(ValueError, match="MUSAN train/eval source leakage"):
        validate_background_eval_split(config, eval_list)


def _cache(tmp_path):
    root = tmp_path / "musan"
    cache = tmp_path / "cache"
    cache.mkdir()
    manifest = dict(format_version=1, split_role="train", split=dict(musan_root=str(root)),
                    recordings=dict(path="recordings.jsonl"))
    (cache / "manifest.json").write_text(json.dumps(manifest))
    (cache / "recordings.jsonl").write_text(json.dumps(
        dict(source_id="noise/train.wav", relative_path="noise/train.wav", list_entry="../musan/noise/train.wav")
    ) + "\n")
    return root, dict(enabled=True, mode="fbank_cache", cache_manifest=str(cache / "manifest.json"))


def test_joint_cached_musan_split_allows_missing_original_training_audio(tmp_path):
    root, config = _cache(tmp_path)
    evaluation = root / "noise/eval.wav"
    evaluation.parent.mkdir(parents=True)
    evaluation.touch()
    eval_list = _audio_list(tmp_path, "eval.txt", [evaluation])
    validate_background_eval_split(config, eval_list)
    assert not (root / "noise/train.wav").exists()
    train = root / "noise/train.wav"
    train.touch()
    _audio_list(tmp_path, "eval.txt", [train])
    with pytest.raises(ValueError, match="MUSAN train/eval source leakage"):
        validate_background_eval_split(config, eval_list)


def test_joint_cached_musan_requires_source_provenance(tmp_path):
    _root, config = _cache(tmp_path)
    evaluation = tmp_path / "eval.wav"
    evaluation.touch()
    eval_list = _audio_list(tmp_path, "eval.txt", [evaluation])
    (tmp_path / "cache/recordings.jsonl").write_text("{}\n")
    with pytest.raises(ValueError, match="missing source path"):
        validate_background_eval_split(config, eval_list)


def test_joint_cached_musan_detects_relocated_recording(tmp_path):
    _root, config = _cache(tmp_path)
    evaluation = tmp_path / "another_host/musan/noise/train.wav"
    evaluation.parent.mkdir(parents=True)
    evaluation.touch()
    eval_list = _audio_list(tmp_path, "eval.txt", [evaluation])
    with pytest.raises(ValueError, match="MUSAN train/eval source leakage"):
        validate_background_eval_split(config, eval_list)


def test_joint_musan_check_is_optional_without_eval_list():
    validate_background_eval_split(dict(enabled=True, mode="fbank_cache"), "")
    validate_background_eval_split(dict(enabled=False), "/nonexistent/eval.txt")
