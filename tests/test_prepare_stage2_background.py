from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random
import shutil

import numpy as np
import pytest
import soundfile as sf
import torch

from dma_kws.config import compose_config, config_to_dict, fbank_kwargs, get_fbank_config
from dma_kws.stage2.background_sampling import (
    BackgroundCropSpec,
    draw_crop_spec,
    materialize_crop,
    probe_source_info,
)
from dma_kws.stage2.fbank import FbankExtractor


DEFAULT_EXPERIMENT = "icefall_zipformer_stage2_eps_softmin_v41"
SEED = 2025
K = 2


def _extraction_overrides() -> list[str]:
    try:
        import lhotse  # noqa: F401
    except ImportError:
        return ["fbank.backend=torchaudio_kaldi"]
    return []


def _catalog_sha256(relative_paths: list[str]) -> str:
    payload = "".join(f"{path}\n" for path in sorted(relative_paths)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _write_wav(path: Path, array: np.ndarray, sample_rate: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.asarray(array, dtype=np.float32), sample_rate, subtype="FLOAT")
    return path


def _write_list(path: Path, entries: list[str]) -> Path:
    path.write_text("".join(f"{entry}\n" for entry in entries), encoding="utf-8")
    return path


def _write_split_json(
    split_dir: Path,
    *,
    musan_root: Path,
    train_rel: list[str],
    eval_rel: list[str],
    train_categories: list[str] = ("music", "noise"),
) -> Path:
    summary = {
        "schema_version": 2,
        "dataset": "MUSAN",
        "musan_root": str(musan_root.resolve()),
        "policy": {"train_categories": list(train_categories)},
        "splits": {
            "train": {
                "list": "train_background.list",
                "recordings": len(train_rel),
                "catalog_sha256": _catalog_sha256(train_rel),
            },
            "eval": {
                "list": "eval_musan.list",
                "recordings": len(eval_rel),
                "catalog_sha256": _catalog_sha256(eval_rel),
            },
        },
    }
    path = split_dir / "split.json"
    path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def _make_synthetic_split(tmp_path: Path) -> dict[str, Path | str]:
    musan_root = tmp_path / "musan"
    split_dir = tmp_path / "split"
    split_dir.mkdir(parents=True)

    short = _write_wav(
        musan_root / "noise" / "free-sound" / "short.wav",
        np.linspace(-0.4, 0.4, 80, dtype=np.float32),
        8000,
    )
    long_left = np.linspace(-0.2, 0.2, 64_000, dtype=np.float32)
    long_right = np.linspace(0.35, -0.15, 64_000, dtype=np.float32)
    long = _write_wav(
        musan_root / "music" / "fma" / "long.wav",
        np.stack([long_left, long_right], axis=1),
        16000,
    )
    speech = _write_wav(
        musan_root / "speech" / "us-gov" / "eval.wav",
        np.linspace(-0.1, 0.1, 1600, dtype=np.float32),
        16000,
    )

    train_rel = [
        "music/fma/long.wav",
        "noise/free-sound/short.wav",
    ]
    eval_rel = ["speech/us-gov/eval.wav"]
    _write_list(
        split_dir / "train_background.list",
        [str(long.resolve()), str(short.resolve())],
    )
    _write_list(split_dir / "eval_musan.list", [str(speech.resolve())])
    _write_split_json(split_dir, musan_root=musan_root, train_rel=train_rel, eval_rel=eval_rel)
    return {
        "musan_root": musan_root,
        "split_dir": split_dir,
        "short": short,
        "long": long,
        "speech": speech,
        "short_id": "noise/free-sound/short.wav",
        "long_id": "music/fma/long.wav",
    }


def _prepare_kwargs(split_dir: Path, output_dir: Path, **overrides):
    kwargs = {
        "split_dir": split_dir,
        "output_dir": output_dir,
        "experiment": DEFAULT_EXPERIMENT,
        "overrides": _extraction_overrides(),
        "crops_per_recording": K,
        "seed": SEED,
        "workers": 1,
        "shard_size_mib": 1,
    }
    kwargs.update(overrides)
    return kwargs


def _oracle_crop_rng(seed: int, source_id: str, crop_ordinal: int) -> random.Random:
    payload = f"{int(seed)}\0{source_id}\0{int(crop_ordinal)}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def _load_manifest(output_dir: Path) -> dict:
    return json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))


def _load_recordings(output_dir: Path) -> list[dict]:
    lines = (output_dir / "recordings.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def test_crop_rng_matches_documented_sha256_payload():
    from dma_kws.data_prep.stage2_background import crop_rng

    source_id = "noise/free-sound/short.wav"
    first = crop_rng(SEED, source_id, 0)
    second = crop_rng(SEED, source_id, 0)
    oracle = _oracle_crop_rng(SEED, source_id, 0)
    assert first.random() == second.random() == oracle.random()
    other = _oracle_crop_rng(SEED, source_id, 1)
    assert crop_rng(SEED, source_id, 1).random() == other.random()
    assert crop_rng(SEED, source_id, 0).random() != other.random()


def _drive_pool_window(*, n_jobs: int, workers: int, wait_completed) -> tuple[list[int], int]:
    from dma_kws.data_prep.stage2_background import _iter_ordered_pool_results

    outstanding: list[int] = []
    max_outstanding = 0

    def submit(index: int) -> int:
        nonlocal max_outstanding
        outstanding.append(index)
        max_outstanding = max(max_outstanding, len(outstanding))
        return index

    def collect(handle: int, index: int) -> int:
        assert handle == index
        return index

    yielded: list[int] = []
    for value in _iter_ordered_pool_results(
        n_jobs,
        workers=workers,
        submit=submit,
        collect=collect,
        wait_completed=wait_completed,
    ):
        yielded.append(value)
        outstanding.remove(value)
    assert outstanding == []
    return yielded, max_outstanding


def test_ordered_pool_window_cannot_run_ahead_of_yield_by_more_than_workers():
    def complete_highest_first(in_flight):
        handle = max(in_flight, key=lambda key: in_flight[key])
        return (handle,)

    def complete_all(in_flight):
        return tuple(in_flight)

    n_jobs = 8
    workers = 2
    high_first, high_max = _drive_pool_window(
        n_jobs=n_jobs, workers=workers, wait_completed=complete_highest_first
    )
    all_at_once, all_max = _drive_pool_window(
        n_jobs=n_jobs, workers=workers, wait_completed=complete_all
    )
    assert high_first == all_at_once == list(range(n_jobs))
    assert high_max <= workers
    assert all_max <= workers

    wider, wider_max = _drive_pool_window(
        n_jobs=9, workers=3, wait_completed=complete_highest_first
    )
    assert wider == list(range(9))
    assert wider_max <= 3


def test_serial_and_workers2_are_bit_identical_and_match_online_extract(tmp_path):
    from dma_kws.data_prep.stage2_background import prepare_stage2_background

    layout = _make_synthetic_split(tmp_path)
    serial_dir = tmp_path / "cache-serial"
    parallel_dir = tmp_path / "cache-parallel"
    kwargs = _prepare_kwargs(layout["split_dir"], serial_dir)
    prepare_stage2_background(**kwargs)
    prepare_stage2_background(**{**kwargs, "output_dir": parallel_dir, "workers": 2})

    serial_manifest = _load_manifest(serial_dir)
    parallel_manifest = _load_manifest(parallel_dir)
    assert serial_manifest["cache_id"] == parallel_manifest["cache_id"]
    assert serial_manifest["format_version"] == 1
    assert serial_manifest["split_role"] == "train"
    assert serial_manifest["K"] == K
    assert serial_manifest["seed"] == SEED
    assert serial_manifest["dtype"] == "float32"
    assert serial_manifest["num_sources"] == 2
    assert serial_manifest["num_crops"] == 4
    assert (serial_dir / "crops.npy").read_bytes() == (parallel_dir / "crops.npy").read_bytes()

    serial_shards = sorted(serial_dir.glob("features-*.npy"))
    parallel_shards = sorted(parallel_dir.glob("features-*.npy"))
    assert [path.name for path in serial_shards] == [path.name for path in parallel_shards]
    for left, right in zip(serial_shards, parallel_shards):
        assert left.read_bytes() == right.read_bytes()

    recordings = _load_recordings(serial_dir)
    assert [row["source_id"] for row in recordings] == [
        layout["long_id"],
        layout["short_id"],
    ]
    assert all(row["crop_count"] == K for row in recordings)
    assert recordings[0]["crop_start"] == 0
    assert recordings[1]["crop_start"] == K

    crops = np.load(serial_dir / "crops.npy", allow_pickle=False)
    assert len(crops) == 4
    assert [str(row["source_id"]) for row in crops] == [
        layout["long_id"],
        layout["long_id"],
        layout["short_id"],
        layout["short_id"],
    ]
    assert [int(row["ordinal"]) for row in crops] == [0, 1, 0, 1]
    assert [int(row["crop_id"]) for row in crops] == [0, 1, 2, 3]

    composed = config_to_dict(
        compose_config(DEFAULT_EXPERIMENT, _extraction_overrides())
    )
    extractor = FbankExtractor(**fbank_kwargs(get_fbank_config(composed)))
    duration_min = float(composed["stage2"]["background_negative"]["duration_seconds_min"])
    duration_max = float(composed["stage2"]["background_negative"]["duration_seconds_max"])
    sources = {
        layout["long_id"]: probe_source_info(layout["long"], source_id=layout["long_id"]),
        layout["short_id"]: probe_source_info(layout["short"], source_id=layout["short_id"]),
    }
    for row in crops:
        source_id = str(row["source_id"])
        source = sources[source_id]
        oracle_rng = _oracle_crop_rng(SEED, source_id, int(row["ordinal"]))
        duration = oracle_rng.uniform(duration_min, duration_max)
        spec = draw_crop_spec(source, duration, rng=oracle_rng)
        assert spec == BackgroundCropSpec(
            source_id=source_id,
            duration_seconds=float(row["duration_seconds"]),
            read_start_frame=int(row["read_start_frame"]),
            read_num_frames=int(row["read_num_frames"]),
            target_num_samples=int(row["target_num_samples"]),
            final_offset=int(row["final_offset"]),
        )
        waveform, sample_rate = materialize_crop(source, spec)
        expected = extractor.extract(waveform, sample_rate)
        shard = np.load(
            serial_dir / f"features-{int(row['shard_index']):05d}.npy",
            allow_pickle=False,
        )
        stored = shard[
            int(row["frame_offset"]) : int(row["frame_offset"]) + int(row["num_frames"])
        ]
        assert stored.shape == tuple(expected.shape)
        torch.testing.assert_close(
            torch.from_numpy(np.ascontiguousarray(stored)),
            expected.cpu(),
            rtol=1e-5,
            atol=1e-5,
        )


def test_corrupt_recording_fails_with_source_path_and_no_output(tmp_path):
    from dma_kws.data_prep.stage2_background import prepare_stage2_background

    layout = _make_synthetic_split(tmp_path)
    layout["short"].write_bytes(b"not a wav file")
    output_dir = tmp_path / "cache"
    with pytest.raises(Exception, match="short.wav"):
        prepare_stage2_background(**_prepare_kwargs(layout["split_dir"], output_dir))
    assert not output_dir.exists()


def test_existing_output_dir_is_refused_and_untouched(tmp_path):
    from dma_kws.data_prep.stage2_background import prepare_stage2_background

    layout = _make_synthetic_split(tmp_path)
    output_dir = tmp_path / "cache"
    output_dir.mkdir()
    marker = output_dir / "keep-me.txt"
    marker.write_text("original", encoding="utf-8")
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        prepare_stage2_background(**_prepare_kwargs(layout["split_dir"], output_dir))
    assert marker.read_text(encoding="utf-8") == "original"
    assert list(output_dir.iterdir()) == [marker]


def test_failed_build_leaves_no_complete_cache(tmp_path, monkeypatch):
    from dma_kws.data_prep import stage2_background as module

    layout = _make_synthetic_split(tmp_path)
    output_dir = tmp_path / "cache"
    original = module._extract_source_crops

    def flaky(job, extractor=None):
        if int(job.source_index) >= 1:
            raise RuntimeError(f"injected failure for {job.path}")
        return original(job, extractor=extractor)

    monkeypatch.setattr(module, "_extract_source_crops", flaky)
    with pytest.raises(Exception, match="injected failure"):
        module.prepare_stage2_background(
            **_prepare_kwargs(layout["split_dir"], output_dir, workers=1)
        )
    assert not output_dir.exists()
    leftovers = [
        path
        for path in output_dir.parent.iterdir()
        if path.name.startswith(f".{output_dir.name}.")
    ]
    assert leftovers == []


def test_verify_only_accepts_good_cache_and_detects_bit_rot(tmp_path):
    from dma_kws.data_prep.stage2_background import (
        prepare_stage2_background,
        verify_stage2_background_cache,
    )

    layout = _make_synthetic_split(tmp_path)
    output_dir = tmp_path / "cache"
    prepare_stage2_background(**_prepare_kwargs(layout["split_dir"], output_dir))
    verify_stage2_background_cache(output_dir)

    shard = next(output_dir.glob("features-*.npy"))
    original_shard = shard.read_bytes()
    corrupted = bytearray(original_shard)
    corrupted[-1] ^= 0xFF
    shard.write_bytes(corrupted)
    with pytest.raises(Exception, match="features-"):
        verify_stage2_background_cache(output_dir)
    shard.write_bytes(original_shard)

    crops_path = output_dir / "crops.npy"
    original_crops = crops_path.read_bytes()
    corrupted_crops = bytearray(original_crops)
    corrupted_crops[-1] ^= 0xFF
    crops_path.write_bytes(corrupted_crops)
    with pytest.raises(Exception, match="crops.npy"):
        verify_stage2_background_cache(output_dir)
    crops_path.write_bytes(original_crops)
    verify_stage2_background_cache(output_dir)


def test_nonzero_dither_fails_before_writing(tmp_path):
    from dma_kws.data_prep.stage2_background import prepare_stage2_background

    layout = _make_synthetic_split(tmp_path)
    output_dir = tmp_path / "cache"
    with pytest.raises(ValueError, match="dither"):
        prepare_stage2_background(
            **_prepare_kwargs(
                layout["split_dir"],
                output_dir,
                overrides=_extraction_overrides() + ["fbank.dither=0.1"],
            )
        )
    assert not output_dir.exists()


def test_train_list_overlapping_eval_fails(tmp_path):
    from dma_kws.data_prep.stage2_background import prepare_stage2_background

    layout = _make_synthetic_split(tmp_path)
    split_dir = layout["split_dir"]
    long_path = str(layout["long"].resolve())
    _write_list(split_dir / "eval_musan.list", [long_path])
    _write_split_json(
        split_dir,
        musan_root=layout["musan_root"],
        train_rel=["music/fma/long.wav", "noise/free-sound/short.wav"],
        eval_rel=["music/fma/long.wav"],
    )
    output_dir = tmp_path / "cache"
    with pytest.raises(ValueError, match="overlap"):
        prepare_stage2_background(**_prepare_kwargs(split_dir, output_dir))
    assert not output_dir.exists()


def test_speech_in_train_categories_or_list_fails(tmp_path):
    from dma_kws.data_prep.stage2_background import prepare_stage2_background

    layout = _make_synthetic_split(tmp_path)
    split_dir = layout["split_dir"]
    output_dir = tmp_path / "cache-categories"
    _write_split_json(
        split_dir,
        musan_root=layout["musan_root"],
        train_rel=["music/fma/long.wav", "noise/free-sound/short.wav"],
        eval_rel=["speech/us-gov/eval.wav"],
        train_categories=["music", "noise", "speech"],
    )
    with pytest.raises(ValueError, match="speech"):
        prepare_stage2_background(**_prepare_kwargs(split_dir, output_dir))
    assert not output_dir.exists()

    layout = _make_synthetic_split(tmp_path / "speech-list")
    split_dir = layout["split_dir"]
    speech_path = str(layout["speech"].resolve())
    _write_list(
        split_dir / "train_background.list",
        [str(layout["long"].resolve()), speech_path],
    )
    _write_list(split_dir / "eval_musan.list", [str(layout["short"].resolve())])
    _write_split_json(
        split_dir,
        musan_root=layout["musan_root"],
        train_rel=["music/fma/long.wav", "speech/us-gov/eval.wav"],
        eval_rel=["noise/free-sound/short.wav"],
    )
    output_dir = tmp_path / "cache-speech-list"
    with pytest.raises(ValueError, match="speech"):
        prepare_stage2_background(**_prepare_kwargs(split_dir, output_dir))
    assert not output_dir.exists()


def test_moving_cache_directory_still_verifies(tmp_path):
    from dma_kws.data_prep.stage2_background import (
        prepare_stage2_background,
        verify_stage2_background_cache,
    )

    layout = _make_synthetic_split(tmp_path)
    output_dir = tmp_path / "cache"
    prepare_stage2_background(**_prepare_kwargs(layout["split_dir"], output_dir))
    moved = tmp_path / "relocated" / "cache"
    shutil.move(str(output_dir), str(moved))
    verify_stage2_background_cache(moved)
    assert not output_dir.exists()


def test_bool_rejected_for_integer_flags(tmp_path):
    from dma_kws.data_prep.stage2_background import (
        build_parser,
        prepare_stage2_background,
    )

    layout = _make_synthetic_split(tmp_path)
    output_dir = tmp_path / "cache"
    with pytest.raises(ValueError, match="int"):
        prepare_stage2_background(
            **_prepare_kwargs(layout["split_dir"], output_dir, workers=True)
        )
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--split-dir",
                str(layout["split_dir"]),
                "--output-dir",
                str(output_dir),
                "--workers",
                "True",
            ]
        )


def _make_generic_catalog(
    tmp_path: Path,
    *,
    dataset_id: str = "dns",
    train_names: tuple[str, ...] = ("clip-a", "clip-b"),
    val_names: tuple[str, ...] = ("clip-val",),
    ineligible_train: tuple[str, ...] = ("speech-skip",),
    long_id_chars: int = 0,
):
    from dma_kws.data_prep.background_manifest import (
        CATALOG_JSON_NAME,
        background_record_from_mapping,
        build_catalog,
        write_catalog,
        write_recordings_jsonl,
    )

    root = tmp_path / dataset_id / "audio"
    catalog_dir = tmp_path / dataset_id / "catalog"
    catalog_dir.mkdir(parents=True)
    records = []
    wavs = {}

    def _add(name: str, split: str, eligible: bool, marker: float, recording_id: str | None = None):
        wav = _write_wav(
            root / f"{name}.wav",
            np.linspace(-0.25 * marker, 0.25 * marker, 32_000, dtype=np.float32),
            16000,
        )
        rec_id = recording_id or f"{dataset_id}:{name}"
        record = background_record_from_mapping(
            {
                "schema_version": 1,
                "dataset_id": dataset_id,
                "recording_id": rec_id,
                "relative_path": f"{name}.wav",
                "audio_path": str(wav.resolve()),
                "group_id": f"{dataset_id}:group:{name}",
                "origin_ids": [f"{dataset_id}:origin:{name}"],
                "split": split,
                "categories": ["noise"] if eligible else ["speech"],
                "background_eligible": eligible,
                "eligibility_basis": "curated_allowlist_v1" if eligible else "excluded",
                "duration_seconds": 2.0,
                "sample_rate": 16000,
                "channels": 1,
                "audio_sha256": hashlib.sha256(wav.read_bytes()).hexdigest(),
                "license_id": "cc0",
                "provenance_complete": True,
            }
        )
        records.append(record)
        wavs[name] = wav
        return record

    for index, name in enumerate(train_names):
        recording_id = None
        if long_id_chars and index == 0:
            recording_id = f"{dataset_id}:" + ("L" * long_id_chars)
        _add(name, "train", True, marker=float(index + 1), recording_id=recording_id)
    for index, name in enumerate(val_names):
        _add(name, "val", True, marker=float(10 + index))
    for index, name in enumerate(ineligible_train):
        _add(name, "train", False, marker=float(20 + index))

    manifest_path = catalog_dir / "recordings.jsonl"
    write_recordings_jsonl(manifest_path, records)
    catalog = build_catalog(
        dataset_id=dataset_id,
        root=str(root.resolve()),
        records=records,
        recordings_path=manifest_path,
        filter_policy_version="curated_allowlist_v1",
        filter_policy_hash="b" * 64,
        split_seed=SEED,
        split_rules={"policy": "group_random"},
    )
    catalog_path = catalog_dir / CATALOG_JSON_NAME
    write_catalog(catalog_path, catalog)
    train_ids = [
        record.recording_id
        for record in records
        if record.split == "train" and record.background_eligible
    ]
    return {
        "dataset_id": dataset_id,
        "root": root,
        "catalog_dir": catalog_dir,
        "manifest": manifest_path,
        "catalog_path": catalog_path,
        "records": records,
        "wavs": wavs,
        "catalog": catalog,
        "train_ids": train_ids,
    }


def _prepare_v2_kwargs(manifest: Path, output_dir: Path, **overrides):
    kwargs = {
        "source_manifest": manifest,
        "output_dir": output_dir,
        "experiment": DEFAULT_EXPERIMENT,
        "overrides": _extraction_overrides(),
        "crops_per_recording": K,
        "seed": SEED,
        "workers": 1,
        "shard_size_mib": 1,
    }
    kwargs.update(overrides)
    return kwargs


def test_cli_split_dir_and_source_manifest_are_mutually_exclusive(tmp_path):
    from dma_kws.data_prep.stage2_background import build_parser

    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--split-dir",
                str(tmp_path / "split"),
                "--source-manifest",
                str(tmp_path / "recordings.jsonl"),
                "--output-dir",
                str(tmp_path / "cache"),
            ]
        )


def test_cli_verify_only_requires_only_output_dir(tmp_path):
    from dma_kws.data_prep.stage2_background import build_parser

    parser = build_parser()
    args = parser.parse_args(
        ["--verify-only", "--output-dir", str(tmp_path / "cache")]
    )
    assert args.verify_only is True
    assert Path(args.output_dir) == tmp_path / "cache"
    assert getattr(args, "split_dir", None) is None
    assert getattr(args, "source_manifest", None) is None


def test_cli_build_requires_split_or_source_manifest(tmp_path):
    from dma_kws.data_prep.stage2_background import main

    with pytest.raises((SystemExit, ValueError), match="split-dir|source-manifest"):
        main(["--output-dir", str(tmp_path / "cache")])


def test_prepare_v2_roundtrip_matches_online_crop_fbank(tmp_path):
    from dma_kws.data_prep.background_manifest import eligible_train_records
    from dma_kws.data_prep.stage2_background import (
        prepare_stage2_background,
        verify_stage2_background_cache,
    )

    layout = _make_generic_catalog(tmp_path)
    output_dir = tmp_path / "cache-v2"
    summary = prepare_stage2_background(**_prepare_v2_kwargs(layout["manifest"], output_dir))
    assert summary["format_version"] == 2
    assert summary["num_sources"] == len(layout["train_ids"])
    assert summary["num_crops"] == len(layout["train_ids"]) * K
    assert summary["K"] == K
    verified = verify_stage2_background_cache(output_dir)
    assert verified["ok"] is True
    assert verified["format_version"] == 2

    manifest = _load_manifest(output_dir)
    assert manifest["format_version"] == 2
    assert manifest["split_role"] == "train"
    assert manifest["dataset_id"] == layout["dataset_id"]
    assert manifest["semantic_catalog_hash"] == layout["catalog"].semantic_catalog_hash
    assert "train_catalog_hash" in manifest
    assert manifest["crop_policy_version"]
    assert "workers" not in manifest or "workers" in (manifest.get("build") or {})
    assert manifest.get("build", {}).get("workers") == 1
    cached_ids = [row["recording_id"] for row in _load_recordings(output_dir)]
    assert cached_ids == sorted(layout["train_ids"])
    val_ids = [
        record.recording_id for record in layout["records"] if record.split == "val"
    ]
    ineligible = [
        record.recording_id
        for record in layout["records"]
        if record.split == "train" and not record.background_eligible
    ]
    assert not set(val_ids) & set(cached_ids)
    assert not set(ineligible) & set(cached_ids)
    assert "source_id" not in np.load(output_dir / "crops.npy", allow_pickle=False).dtype.names

    composed = config_to_dict(
        compose_config(DEFAULT_EXPERIMENT, _extraction_overrides())
    )
    extractor = FbankExtractor(**fbank_kwargs(get_fbank_config(composed)))
    duration_min = float(composed["stage2"]["background_negative"]["duration_seconds_min"])
    duration_max = float(composed["stage2"]["background_negative"]["duration_seconds_max"])
    crops = np.load(output_dir / "crops.npy", allow_pickle=False)
    recordings = _load_recordings(output_dir)
    by_index = {int(row["recording_index"]): row for row in recordings}
    for row in crops:
        record = by_index[int(row["recording_index"])]
        recording_id = str(record["recording_id"])
        wav = next(
            item.audio_path
            for item in eligible_train_records(layout["records"])
            if item.recording_id == recording_id
        )
        source = probe_source_info(wav, source_id=recording_id)
        oracle_rng = _oracle_crop_rng(SEED, recording_id, int(row["ordinal"]))
        duration = oracle_rng.uniform(duration_min, duration_max)
        spec = draw_crop_spec(source, duration, rng=oracle_rng)
        assert spec == BackgroundCropSpec(
            source_id=recording_id,
            duration_seconds=float(row["duration_seconds"]),
            read_start_frame=int(row["read_start_frame"]),
            read_num_frames=int(row["read_num_frames"]),
            target_num_samples=int(row["target_num_samples"]),
            final_offset=int(row["final_offset"]),
        )
        waveform, sample_rate = materialize_crop(source, spec)
        expected = extractor.extract(waveform, sample_rate)
        shard = np.load(
            output_dir / f"features-{int(row['shard_index']):05d}.npy",
            allow_pickle=False,
        )
        stored = shard[
            int(row["frame_offset"]) : int(row["frame_offset"]) + int(row["num_frames"])
        ]
        torch.testing.assert_close(
            torch.from_numpy(np.ascontiguousarray(stored)),
            expected.cpu(),
            rtol=1e-5,
            atol=1e-5,
        )


def test_prepare_v2_preserves_long_recording_ids(tmp_path):
    from dma_kws.data_prep.stage2_background import prepare_stage2_background

    layout = _make_generic_catalog(tmp_path, long_id_chars=1500)
    output_dir = tmp_path / "cache-long-id"
    prepare_stage2_background(**_prepare_v2_kwargs(layout["manifest"], output_dir))
    stored = _load_recordings(output_dir)
    expected = layout["train_ids"][0]
    assert len(expected) > 1024
    assert stored[0]["recording_id"] == expected
    assert stored[0]["recording_id"] == layout["train_ids"][0]


def test_prepare_v2_nonzero_dither_fails_before_writing(tmp_path):
    from dma_kws.data_prep.stage2_background import prepare_stage2_background

    layout = _make_generic_catalog(tmp_path)
    output_dir = tmp_path / "cache"
    with pytest.raises(ValueError, match="dither"):
        prepare_stage2_background(
            **_prepare_v2_kwargs(
                layout["manifest"],
                output_dir,
                overrides=_extraction_overrides() + ["fbank.dither=0.1"],
            )
        )
    assert not output_dir.exists()


def test_prepare_v2_refuses_existing_output_and_failed_build_is_atomic(tmp_path, monkeypatch):
    from dma_kws.data_prep import stage2_background as module

    layout = _make_generic_catalog(tmp_path)
    output_dir = tmp_path / "cache"
    output_dir.mkdir()
    marker = output_dir / "keep-me.txt"
    marker.write_text("original", encoding="utf-8")
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        module.prepare_stage2_background(
            **_prepare_v2_kwargs(layout["manifest"], output_dir)
        )
    assert marker.read_text(encoding="utf-8") == "original"

    fresh = tmp_path / "fresh-cache"
    original = module._extract_source_crops

    def flaky(job, extractor=None):
        if int(job.source_index) >= 1:
            raise RuntimeError(f"injected failure for {job.path}")
        return original(job, extractor=extractor)

    monkeypatch.setattr(module, "_extract_source_crops", flaky)
    with pytest.raises(Exception, match="injected failure"):
        module.prepare_stage2_background(**_prepare_v2_kwargs(layout["manifest"], fresh))
    assert not fresh.exists()
    leftovers = [
        path
        for path in fresh.parent.iterdir()
        if path.name.startswith(f".{fresh.name}.")
    ]
    assert leftovers == []


def test_verify_only_does_not_open_source_audio_and_unknown_version_fails(
    tmp_path, monkeypatch
):
    from dma_kws.data_prep.stage2_background import (
        _cache_id_for,
        main,
        prepare_stage2_background,
        verify_stage2_background_cache,
    )

    layout = _make_generic_catalog(tmp_path)
    output_dir = tmp_path / "cache"
    prepare_stage2_background(**_prepare_v2_kwargs(layout["manifest"], output_dir))

    def boom(*_args, **_kwargs):
        raise AssertionError("verify-only must not open source audio")

    monkeypatch.setattr("dma_kws.stage2.background_sampling.probe_source_info", boom)
    monkeypatch.setattr("dma_kws.stage2.background_sampling.materialize_crop", boom)
    monkeypatch.setattr("soundfile.SoundFile", boom)
    verify_stage2_background_cache(output_dir)
    summary = main(["--verify-only", "--output-dir", str(output_dir)])
    assert summary["ok"] is True
    assert summary["format_version"] == 2

    manifest = _load_manifest(output_dir)
    manifest["format_version"] = 99
    manifest["cache_id"] = _cache_id_for(manifest)
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="format_version"):
        verify_stage2_background_cache(output_dir)


def test_prepare_v2_rejects_audio_that_no_longer_matches_catalog_hash(tmp_path):
    from dma_kws.data_prep.stage2_background import prepare_stage2_background

    layout = _make_generic_catalog(tmp_path)
    shutil.copyfile(layout["wavs"]["clip-val"], layout["wavs"]["clip-a"])
    with pytest.raises(ValueError, match=r"audio_sha256|content_sha256"):
        prepare_stage2_background(
            **_prepare_v2_kwargs(layout["manifest"], tmp_path / "cache-mismatch")
        )
    assert not (tmp_path / "cache-mismatch").exists()


def test_verify_v2_rejects_cache_when_content_hash_disagrees_with_snapshot(
    tmp_path,
):
    from dma_kws.data_prep.stage2_background import (
        _cache_id_for,
        prepare_stage2_background,
        verify_stage2_background_cache,
    )

    layout = _make_generic_catalog(tmp_path)
    output_dir = tmp_path / "cache-hash"
    prepare_stage2_background(**_prepare_v2_kwargs(layout["manifest"], output_dir))
    rows = _load_recordings(output_dir)
    rows[0]["content_sha256"] = "c" * 64
    recordings_path = output_dir / "recordings.jsonl"
    recordings_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    payload = recordings_path.read_bytes()
    manifest = _load_manifest(output_dir)
    manifest["recordings"] = {
        **manifest["recordings"],
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size": len(payload),
    }
    manifest["cache_id"] = _cache_id_for(manifest)
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=r"content_sha256|audio_sha256"):
        verify_stage2_background_cache(output_dir)
