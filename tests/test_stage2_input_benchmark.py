from __future__ import annotations

import ast
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import torch

from dma_kws.config import compose_config, config_to_dict
from dma_kws.pathing import PROJECT_ROOT
from dma_kws.stage2.dataset import stage2_worker_init_fn
from dma_kws.tokenizer import load_char_tokenizer
from tests.test_prepare_stage2_background import (
    DEFAULT_EXPERIMENT,
    _extraction_overrides,
)
from tests.test_stage2_background_cache import _build_cache


TIMING_JSON_KEYS = (
    "mode",
    "status",
    "git_commit",
    "config",
    "cache_id",
    "pytorch_version",
    "cuda_version",
    "device",
    "cpu_cores",
    "workers",
    "prefetch",
    "batch_size",
    "precision",
    "accumulate_grad_batches",
    "warmup_batches_requested",
    "warmup_batches_actual",
    "measure_batches_requested",
    "measure_batches_actual",
    "optimizer_steps_requested",
    "optimizer_steps_actual",
    "wait_p50_ms",
    "wait_p95_ms",
    "first_next_seconds",
    "warmup_seconds",
    "measured_wall_seconds",
    "samples_per_second",
    "effective_audio_seconds_per_second",
    "frame_length",
    "padding_ratio",
    "background_count",
    "positive_count",
    "negative_count",
    "process_rss_bytes",
    "cache_hits",
    "cache_misses",
    "open_shard_high_water",
    "gpu_util",
)

DICT_PATH = PROJECT_ROOT / "data" / "dict" / "lang_char.txt"


def _assert_timing_contract(payload: dict) -> None:
    missing = [key for key in TIMING_JSON_KEYS if key not in payload]
    assert missing == [], missing
    assert payload["status"] == "ok"
    assert "speedup" not in payload
    assert payload["gpu_util"] != 0
    assert payload["gpu_util"] == "unavailable" or isinstance(
        payload["gpu_util"], (int, float)
    )
    for name in (
        "wait_p50_ms",
        "wait_p95_ms",
        "first_next_seconds",
        "warmup_seconds",
        "measured_wall_seconds",
        "samples_per_second",
        "effective_audio_seconds_per_second",
        "padding_ratio",
    ):
        value = payload[name]
        assert isinstance(value, (int, float)), name
        assert value == value, name
    frame = payload["frame_length"]
    assert isinstance(frame, dict)
    for name in ("min", "max", "mean", "p50", "p95", "count"):
        assert name in frame
        assert isinstance(frame[name], (int, float))
    assert isinstance(payload["config"], dict)
    assert "stage2" in payload["config"]


def _tiny_libriphrase_tree(tmp_path: Path) -> tuple[Path, Path]:
    wav_dir = tmp_path / "fbank"
    (wav_dir / "LP-460-fbank" / "hello").mkdir(parents=True)
    (wav_dir / "LP-460-fbank" / "world").mkdir(parents=True)
    np.save(
        wav_dir / "LP-460-fbank" / "hello" / "a.npy",
        np.ones((5, 80), dtype=np.float32),
    )
    np.save(
        wav_dir / "LP-460-fbank" / "hello" / "b.npy",
        np.ones((7, 80), dtype=np.float32),
    )
    np.save(
        wav_dir / "LP-460-fbank" / "world" / "c.npy",
        np.ones((6, 80), dtype=np.float32),
    )
    np.save(
        wav_dir / "LP-460-fbank" / "world" / "d.npy",
        np.ones((4, 80), dtype=np.float32),
    )

    meta = tmp_path / "meta"
    meta.mkdir()
    np.save(
        meta / "clips-2-a.npy",
        np.array(
            [
                {"audio_path": "LP-460/hello/a.wav"},
                {"audio_path": "LP-460/hello/b.wav"},
            ],
            dtype=object,
        ),
    )
    np.save(
        meta / "clips-2-b.npy",
        np.array(
            [
                {"audio_path": "LP-460/world/c.wav"},
                {"audio_path": "LP-460/world/d.wav"},
            ],
            dtype=object,
        ),
    )
    np.save(meta / "dist-0-a.npy", np.array([], dtype=object))
    np.save(
        meta / "dist-2-b.npy",
        np.array([{"ngram": "hello"}, {"ngram": "hello"}], dtype=object),
    )

    parquet = tmp_path / "train.parquet"
    pd.DataFrame(
        {
            "ngram": ["hello", "world"],
            "ngram_g2p": ["HH AH0 L OW1", "W ER1 L D"],
            "clips_file": [str(meta / "clips-2-a.npy"), str(meta / "clips-2-b.npy")],
            "distances_file": [str(meta / "dist-0-a.npy"), str(meta / "dist-2-b.npy")],
        }
    ).to_parquet(parquet)
    return parquet, wav_dir


def _tiny_loader_config(tmp_path: Path, extra_overrides: list[str] | None = None) -> dict:
    parquet, wav_dir = _tiny_libriphrase_tree(tmp_path)
    overrides = [
        f"stage2.parquet_file={parquet}",
        f"stage2.wav_dir={wav_dir}",
        "stage2.num_workers=0",
        "stage2.batch_size_per_gpu=2",
        "stage2.sample_lens=8",
        "stage2.dataloader.pin_memory=false",
        "stage2.background_negative.enabled=false",
        *_extraction_overrides(),
        *(extra_overrides or ()),
    ]
    return config_to_dict(compose_config(DEFAULT_EXPERIMENT, overrides))


class _StubTrainDataset(torch.utils.data.Dataset):
    def __init__(self, lengths: list[int], labels: list[int], query_empty: list[bool]):
        self.lengths = list(lengths)
        self.labels = list(labels)
        self.query_empty = list(query_empty)
        self.sample_lens = len(self.lengths)

    def __len__(self) -> int:
        return self.sample_lens

    def __getitem__(self, index: int) -> dict:
        length = self.lengths[index]
        empty = self.query_empty[index]
        query = torch.tensor([], dtype=torch.long) if empty else torch.tensor([1], dtype=torch.long)
        return {
            "anchor_seq": torch.tensor([1, 2], dtype=torch.long),
            "query_seq": query,
            "feat": torch.ones(length, 80, dtype=torch.float32),
            "label": torch.tensor(self.labels[index], dtype=torch.long),
            "seq_label": torch.tensor([1, 0], dtype=torch.long),
        }


def test_padding_ratio_formula_locked():
    from dma_kws.stage2.input_benchmark import padding_ratio

    lengths = torch.tensor([4, 6])
    assert padding_ratio(lengths) == pytest.approx(1.0 - 10.0 / 12.0)
    equal = torch.tensor([8, 8, 8])
    assert padding_ratio(equal) == pytest.approx(0.0)
    assert padding_ratio(torch.tensor([5])) == pytest.approx(0.0)


def test_train_py_calls_shared_builders():
    source = (PROJECT_ROOT / "dma_kws/stage2/train.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    training = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "run_stage2_training"
    )
    call_names = []
    for node in ast.walk(training):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name):
            call_names.append(func.id)
        elif isinstance(func, ast.Attribute):
            call_names.append(func.attr)
    assert "build_stage2_train_dataset" in call_names
    assert "build_stage2_train_dataloader" in call_names
    assert "LibriPhraseTrainDataset" not in call_names


def test_build_stage2_train_dataset_forwards_background_and_metadata(tmp_path, monkeypatch):
    from dma_kws.stage2.train_data import build_stage2_train_dataset

    captured: dict = {}

    class _FakeDataset:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr("dma_kws.stage2.train_data.LibriPhraseTrainDataset", _FakeDataset)
    parquet = tmp_path / "train.parquet"
    parquet.write_bytes(b"parquet")
    wav_dir = tmp_path / "fbank"
    wav_dir.mkdir()
    config = {
        "paths": {"processed_root": str(tmp_path), "feature_root": str(tmp_path)},
        "stage1": {"num_workers": 0, "input_dim": 80},
        "stage2": {
            "parquet_file": str(parquet),
            "wav_dir": str(wav_dir),
            "negative_ratio": 1,
            "hard_negative_ratio": 1,
            "sample_lens": 8,
            "background_negative": {"enabled": False, "probability": 0.25},
            "metadata_cache": {"max_entries": 8, "max_bytes": 1024},
            "noise_augmentation": {},
            "sequence_loss": {
                "target_mode": "ordered_contiguous_prefix",
                "progress_weight": 0.3,
                "normalization": "sample",
            },
        },
        "training": {"seed": 11},
        "fbank": {"num_mel_bins": 80, "dither": 0.0},
    }
    with patch("dma_kws.stage2.train_data.process_rank", return_value=2):
        build_stage2_train_dataset(config, tokenizer=object())

    assert captured["background_negative"] == {"enabled": False, "probability": 0.25}
    assert captured["metadata_cache"] == {"max_entries": 8, "max_bytes": 1024}
    assert captured["seed"] == 11 + 1_000_003 * 2
    assert captured["noise_augmentation"] == {}
    assert "fbank_kwargs" in captured


def test_build_stage2_train_dataloader_keeps_train_semantics():
    from dma_kws.stage2.train_data import build_stage2_train_dataloader

    config = {
        "stage1": {"num_workers": 0},
        "stage2": {
            "batch_size_per_gpu": 2,
            "num_workers": 0,
            "dataloader": {
                "pin_memory": False,
                "persistent_workers": True,
                "prefetch_factor": 8,
            },
        },
    }
    dataset = _StubTrainDataset([4, 6, 5, 8], [1, 0, 1, 0], [False, True, False, False])
    loader = build_stage2_train_dataloader(config, dataset)
    assert loader.batch_size == 2
    assert loader.drop_last is True
    assert isinstance(loader.sampler, torch.utils.data.RandomSampler)
    assert loader.worker_init_fn is stage2_worker_init_fn
    assert loader.pin_memory is False
    assert not hasattr(loader, "prefetch_factor") or loader.num_workers == 0


def test_background_mode_json_contract_and_exact_crop_match(tmp_path):
    from dma_kws.stage2.input_benchmark import run_input_benchmark

    built = _build_cache(tmp_path)
    config = built["composed"]
    config["stage2"]["batch_size_per_gpu"] = 2
    output = tmp_path / "background.json"
    payload = run_input_benchmark(
        mode="background",
        config=config,
        output=output,
        warmup_batches=1,
        measure_batches=2,
        device="cpu",
        manifest=built["manifest_path"],
        split_dir=built["layout"]["split_dir"],
    )
    written = json.loads(output.read_text(encoding="utf-8"))
    assert written["mode"] == payload["mode"] == "background"
    _assert_timing_contract(payload)
    assert payload["cache_id"]
    times = payload["component_times"]
    for name in (
        "read_seconds",
        "crop_seconds",
        "resample_seconds",
        "fbank_seconds",
        "cache_read_seconds",
    ):
        assert isinstance(times[name], (int, float)), name
        assert times[name] >= 0.0
    assert isinstance(payload["cache_hits"], int)
    assert isinstance(payload["cache_misses"], int)
    assert isinstance(payload["open_shard_high_water"], int)
    assert payload["open_shard_high_water"] >= 1
    assert payload["measure_batches_actual"] == 2
    assert payload["batch_size"] == 2


def test_loader_mode_json_contract_on_tiny_parquet(tmp_path):
    from dma_kws.stage2.input_benchmark import run_input_benchmark

    config = _tiny_loader_config(tmp_path)
    tokenizer = load_char_tokenizer(DICT_PATH)
    output = tmp_path / "loader.json"
    payload = run_input_benchmark(
        mode="loader",
        config=config,
        output=output,
        warmup_batches=1,
        measure_batches=2,
        device="cpu",
        tokenizer=tokenizer,
    )
    _assert_timing_contract(payload)
    assert payload["mode"] == "loader"
    assert payload["first_next_seconds"] >= 0.0
    assert payload["warmup_batches_actual"] == 1
    assert payload["measure_batches_actual"] == 2
    assert payload["workers"] == 0
    counts = (
        payload["positive_count"]
        + payload["negative_count"]
        + payload["background_count"]
    )
    assert counts == payload["batch_size"] * payload["measure_batches_actual"]
    assert 0.0 <= payload["padding_ratio"] < 1.0
    assert payload["cache_hits"] in (0, "unavailable")
    assert payload["cache_misses"] in (0, "unavailable")


def test_loader_mode_padding_and_composition_with_stub_dataset(tmp_path):
    from dma_kws.stage2.input_benchmark import run_input_benchmark

    config = _tiny_loader_config(tmp_path)
    config["stage2"]["batch_size_per_gpu"] = 2
    config["stage2"]["num_workers"] = 0
    # Exactly one drop_last batch of [4, 6], so shuffle cannot regroup equal lengths.
    dataset = _StubTrainDataset(
        lengths=[4, 6],
        labels=[1, 0],
        query_empty=[False, True],
    )
    output = tmp_path / "loader-stub.json"
    payload = run_input_benchmark(
        mode="loader",
        config=config,
        output=output,
        warmup_batches=1,
        measure_batches=2,
        device="cpu",
        dataset=dataset,
    )
    _assert_timing_contract(payload)
    expected = 1.0 - 10.0 / 12.0
    assert payload["padding_ratio"] == pytest.approx(expected)
    assert payload["frame_length"]["min"] == 4
    assert payload["frame_length"]["max"] == 6


def test_train_mode_missing_checkpoint_fails_without_toy_throughput(tmp_path):
    from dma_kws.stage2.input_benchmark import (
        TrainModeUnavailable,
        run_input_benchmark,
    )

    config = _tiny_loader_config(tmp_path)
    config["stage2"]["init_checkpoint"] = ""
    output = tmp_path / "train.json"
    with pytest.raises(TrainModeUnavailable, match="init_checkpoint"):
        run_input_benchmark(
            mode="train",
            config=config,
            output=output,
            warmup_batches=1,
            measure_batches=2,
            device="cpu",
        )
    assert not output.exists()


def test_train_mode_cuda_without_cuda_fails_cleanly(tmp_path, monkeypatch):
    from dma_kws.stage2.input_benchmark import (
        TrainModeUnavailable,
        run_input_benchmark,
    )

    config = _tiny_loader_config(tmp_path)
    config["stage2"]["init_checkpoint"] = str(tmp_path / "missing.pt")
    output = tmp_path / "train-cuda.json"
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    with pytest.raises(TrainModeUnavailable, match="CUDA"):
        run_input_benchmark(
            mode="train",
            config=config,
            output=output,
            warmup_batches=1,
            measure_batches=2,
            device="cuda",
        )
    assert not output.exists()


def test_profile_keeps_timing_schema_and_separates_trace(tmp_path):
    from dma_kws.stage2.input_benchmark import run_input_benchmark

    config = _tiny_loader_config(tmp_path)
    dataset = _StubTrainDataset([4, 6], [1, 0], [False, True])
    output = tmp_path / "profile-timing.json"
    payload = run_input_benchmark(
        mode="loader",
        config=config,
        output=output,
        warmup_batches=1,
        measure_batches=1,
        device="cpu",
        dataset=dataset,
        profile=True,
    )
    _assert_timing_contract(payload)
    assert "self_cpu_time_total" not in payload
    assert "cpu_time_total" not in payload
    profile = payload.get("profile")
    assert isinstance(profile, dict)
    trace_path = Path(profile["trace_path"])
    assert trace_path.is_file()
    assert trace_path != output
    timing_keys = set(json.loads(output.read_text(encoding="utf-8")))
    assert "self_cpu_time_total" not in timing_keys


def test_cli_requires_output_and_background_paths(tmp_path):
    from dma_kws.stage2.input_benchmark import build_parser

    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--mode", "loader"])
    args = parser.parse_args(
        [
            "--mode",
            "background",
            "--manifest",
            str(tmp_path / "manifest.json"),
            "--split-dir",
            str(tmp_path / "split"),
            "--output",
            str(tmp_path / "out.json"),
            "--warmup-batches",
            "1",
            "--measure-batches",
            "2",
            "--override",
            "stage2.num_workers=0",
        ]
    )
    assert args.mode == "background"
    assert args.experiment == DEFAULT_EXPERIMENT
    assert args.override == ["stage2.num_workers=0"]
