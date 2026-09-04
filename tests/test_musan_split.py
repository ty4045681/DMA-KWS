from __future__ import annotations

import hashlib
import json
from pathlib import Path
import wave

import pytest

from dma_kws.data_prep.musan_split import (
    MusanRecording,
    build_musan_split,
    discover_musan_recordings,
    split_musan_recordings,
)
from dma_kws.inference.manifest import load_audio_file_list
from dma_kws.inference.musan_fa import musan_catalog_sha256


def _write_wav(path: Path, duration_seconds: float, sample_rate: int = 8000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = round(duration_seconds * sample_rate)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(b"\0\0" * frames)


def _build_complete_musan(root: Path) -> dict[str, list[Path]]:
    layout = {
        "music/fma": [0.4, 0.3, 0.2, 0.1],
        "noise/free-sound": [0.7, 0.1, 0.1, 0.1],
        "speech/us-gov": [0.1, 0.2, 0.3, 0.4],
    }
    paths: dict[str, list[Path]] = {}
    for stratum, durations in layout.items():
        category, source = stratum.split("/")
        stratum_paths: list[Path] = []
        for index, duration in enumerate(durations):
            path = root / category / source / f"{category}-{source}-{index:04d}.wav"
            _write_wav(path, duration)
            stratum_paths.append(path.resolve())
        paths[stratum] = stratum_paths

    music_paths = paths["music/fma"]
    annotations = root / "music" / "fma" / "ANNOTATIONS"
    annotations.write_text(
        "\n".join(
            [
                f"{music_paths[0].name} genre N artist-a",
                f"{music_paths[1].name} genre N artist-a",
                f"{music_paths[2].name} genre N artist-b",
                f"{music_paths[3].name} genre N artist-c",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return paths


def _recording(name: str, duration_seconds: int) -> MusanRecording:
    relative_path = f"noise/free-sound/{name}.wav"
    return MusanRecording(
        path=Path("/virtual") / relative_path,
        relative_path=relative_path,
        category="noise",
        source="free-sound",
        duration_microseconds=duration_seconds * 1_000_000,
        frames=duration_seconds * 8000,
        sample_rate=8000,
        group_id=relative_path,
        group_kind="recording",
    )


def test_duration_balancing_finds_three_short_groups_for_eval():
    recordings = tuple(
        _recording(name, duration)
        for name, duration in zip(("long", "a", "b", "c"), (70, 10, 10, 10))
    )

    split = split_musan_recordings(recordings, train_ratio=0.70, seed=9)

    assert sum(item.duration_microseconds for item in split.train) == 70_000_000
    assert {item.relative_path for item in split.eval} == {
        "noise/free-sound/a.wav",
        "noise/free-sound/b.wav",
        "noise/free-sound/c.wav",
    }


def test_split_is_order_independent_group_safe_and_seeded():
    recordings = tuple(_recording(f"equal-{index}", 10) for index in range(8))

    first = split_musan_recordings(recordings, train_ratio=0.5, seed=3)
    repeated = split_musan_recordings(tuple(reversed(recordings)), train_ratio=0.5, seed=3)
    assert [item.relative_path for item in first.train] == [
        item.relative_path for item in repeated.train
    ]

    partitions = {
        tuple(
            item.relative_path
            for item in split_musan_recordings(
                recordings, train_ratio=0.5, seed=seed
            ).train
        )
        for seed in range(8)
    }
    assert len(partitions) > 1


def test_build_musan_split_writes_disjoint_lists_and_matching_catalog_hash(tmp_path):
    musan_root = tmp_path / "musan"
    paths = _build_complete_musan(musan_root)
    output_dir = tmp_path / "split"

    summary = build_musan_split(
        musan_root, output_dir, train_ratio=0.60, seed=20260831
    )
    train = load_audio_file_list(output_dir / "train_background.list")
    evaluation = load_audio_file_list(output_dir / "eval_musan.list")

    assert set(train).isdisjoint(evaluation)
    assert set(train) | set(evaluation) == {
        path for stratum_paths in paths.values() for path in stratum_paths
    }
    assert (output_dir / "split.json").is_file()
    assert summary["splits"]["eval"]["catalog_sha256"] == musan_catalog_sha256(
        evaluation, musan_root
    )
    assert summary["splits"]["train"]["catalog_sha256"] == musan_catalog_sha256(
        train, musan_root
    )

    membership = {path: "train" for path in train} | {
        path: "eval" for path in evaluation
    }
    assert membership[paths["music/fma"][0]] == membership[paths["music/fma"][1]]
    for stratum_paths in paths.values():
        assert {membership[path] for path in stratum_paths} == {"train", "eval"}

    expected_eval_hash = hashlib.sha256(
        "".join(
            f"{path.relative_to(musan_root).as_posix()}\n" for path in sorted(evaluation)
        ).encode("utf-8")
    ).hexdigest()
    assert summary["splits"]["eval"]["catalog_sha256"] == expected_eval_hash
    assert all(
        values["absolute_error_microseconds"] <= values["largest_group_microseconds"]
        for values in summary["balance_by_stratum"].values()
    )

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        build_musan_split(musan_root, output_dir)


def test_single_artist_music_source_falls_back_to_recording_groups(tmp_path):
    musan_root = tmp_path / "musan"
    paths = _build_complete_musan(musan_root)
    annotations = musan_root / "music" / "fma" / "ANNOTATIONS"
    annotations.write_text(
        "".join(
            f"{path.name} genre N only-artist\n" for path in paths["music/fma"]
        ),
        encoding="utf-8",
    )

    recordings = discover_musan_recordings(musan_root)
    music = [record for record in recordings if record.category == "music"]
    assert {record.group_kind for record in music} == {"recording"}

    split = split_musan_recordings(recordings, train_ratio=0.6, seed=17)
    train_music = {record.path for record in split.train if record.category == "music"}
    eval_music = {record.path for record in split.eval if record.category == "music"}
    assert train_music
    assert eval_music
    assert train_music.isdisjoint(eval_music)


def test_artist_grouping_decision_uses_only_audio_present_in_root(tmp_path):
    musan_root = tmp_path / "musan"
    paths = _build_complete_musan(musan_root)
    paths["music/fma"][2].unlink()
    paths["music/fma"][3].unlink()

    recordings = discover_musan_recordings(musan_root)
    music = [record for record in recordings if record.category == "music"]
    assert len(music) == 2
    assert {record.group_kind for record in music} == {"recording"}
    split = split_musan_recordings(recordings, train_ratio=0.6, seed=17)
    assert any(record.category == "music" for record in split.train)
    assert any(record.category == "music" for record in split.eval)


@pytest.mark.parametrize("train_ratio", [0.0, 1.0, float("nan"), float("inf")])
def test_split_rejects_invalid_ratio(train_ratio):
    with pytest.raises(ValueError, match="train_ratio"):
        split_musan_recordings(
            (_recording("a", 1), _recording("b", 1)),
            train_ratio=train_ratio,
        )


def test_split_rejects_singleton_stratum_and_negative_seed():
    with pytest.raises(ValueError, match="at least 2"):
        split_musan_recordings((_recording("only", 1),))
    with pytest.raises(ValueError, match="non-negative integer"):
        split_musan_recordings(
            (_recording("a", 1), _recording("b", 1)), seed=-1
        )


def test_discovery_requires_complete_musan_and_rejects_broken_audio(tmp_path):
    incomplete = tmp_path / "incomplete"
    _write_wav(incomplete / "music" / "fma" / "music.wav", 0.1)
    with pytest.raises(NotADirectoryError, match="noise"):
        discover_musan_recordings(incomplete)

    complete = tmp_path / "complete"
    for category in ("music", "noise", "speech"):
        path = complete / category / "source" / f"{category}.wav"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"not an audio file")
    with pytest.raises(ValueError, match="Could not read audio metadata"):
        discover_musan_recordings(complete)


@pytest.mark.parametrize(
    "train_categories",
    [("music", "noise"), ("noise", "music")],
)
def test_train_categories_keeps_speech_eval_only(tmp_path, train_categories):
    musan_root = tmp_path / "musan"
    paths = _build_complete_musan(musan_root)
    recordings = discover_musan_recordings(musan_root)

    split = split_musan_recordings(
        recordings, train_ratio=0.60, seed=20260831, train_categories=train_categories
    )
    train_paths = {record.path for record in split.train}
    eval_paths = {record.path for record in split.eval}
    speech_paths = set(paths["speech/us-gov"])
    music_paths = set(paths["music/fma"])
    noise_paths = set(paths["noise/free-sound"])
    catalog_paths = {
        path for stratum_paths in paths.values() for path in stratum_paths
    }

    assert train_paths.isdisjoint(eval_paths)
    assert train_paths | eval_paths == catalog_paths
    assert not any(path in train_paths for path in speech_paths)
    assert speech_paths <= eval_paths
    assert train_paths & music_paths and eval_paths & music_paths
    assert train_paths & noise_paths and eval_paths & noise_paths

    output_dir = tmp_path / "split"
    summary = build_musan_split(
        musan_root,
        output_dir,
        train_ratio=0.60,
        seed=20260831,
        train_categories=train_categories,
    )
    payload = json.loads((output_dir / "split.json").read_text(encoding="utf-8"))
    assert payload["schema_version"] == 2
    assert payload["policy"]["train_categories"] == ["music", "noise"]
    assert summary["policy"]["train_categories"] == ["music", "noise"]
    for stratum, values in payload["balance_by_stratum"].items():
        if stratum.startswith("speech/"):
            assert values["target_train_microseconds"] == 0
            assert values["actual_train_microseconds"] == 0


def test_train_categories_rejects_unknown_and_empty():
    recordings = (_recording("a", 1), _recording("b", 1))
    with pytest.raises(ValueError, match="train_categories"):
        split_musan_recordings(recordings, train_categories=("speechh",))
    with pytest.raises(ValueError, match="train_categories"):
        split_musan_recordings(recordings, train_categories=())


def test_default_train_categories_still_splits_speech(tmp_path):
    musan_root = tmp_path / "musan"
    paths = _build_complete_musan(musan_root)
    recordings = discover_musan_recordings(musan_root)

    split = split_musan_recordings(recordings, train_ratio=0.60, seed=20260831)
    membership = {record.path: "train" for record in split.train} | {
        record.path: "eval" for record in split.eval
    }
    for stratum_paths in paths.values():
        assert {membership[path] for path in stratum_paths} == {"train", "eval"}


def test_eval_only_category_allows_singleton_stratum():
    speech = MusanRecording(
        path=Path("/virtual/speech/us-gov/only.wav"),
        relative_path="speech/us-gov/only.wav",
        category="speech",
        source="us-gov",
        duration_microseconds=1_000_000,
        frames=8000,
        sample_rate=8000,
        group_id="speech/us-gov/only.wav",
        group_kind="recording",
    )
    music = tuple(
        MusanRecording(
            path=Path(f"/virtual/music/fma/music-{index}.wav"),
            relative_path=f"music/fma/music-{index}.wav",
            category="music",
            source="fma",
            duration_microseconds=1_000_000,
            frames=8000,
            sample_rate=8000,
            group_id=f"music/fma/music-{index}.wav",
            group_kind="recording",
        )
        for index in range(2)
    )

    split = split_musan_recordings(
        (speech, *music),
        train_ratio=0.5,
        seed=3,
        train_categories=("music", "noise"),
    )
    assert {record.relative_path for record in split.train} <= {
        record.relative_path for record in music
    }
    assert speech.relative_path in {record.relative_path for record in split.eval}
    assert not any(record.category == "speech" for record in split.train)
