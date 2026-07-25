#!/usr/bin/env python3
"""Prepare paper-format Stage II QbyT training data from LibriPhrase parquet."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from pathlib import Path
from typing import Any, Callable

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

from dma_kws.config import fbank_kwargs, get_fbank_config, require_sections
from dma_kws.hydra_app import CONFIG_DIR, resolved_config
from dma_kws.stage2.pairs import (
    clip_to_audio_rel,
    decoded_glob_for_dataset,
    infer_dataset_id,
    iter_decoded_audio_rows,
    resolve_data_root,
    scan_decoded_parquet_shard,
)
from dma_kws.stage2.fbank import FbankExtractor
from dma_kws.stage2.prep_console import Stage2PrepReporter, resolve_num_workers
from dma_kws.stage2.prepare_paper import (
    OUTPUT_PARQUET_NAME,
    build_anchor_metadata,
    compute_fbank_for_clip,
    convert_aggregated_to_paper_parquet,
    parse_clips,
    stream_fbank_from_decoded,
)


def find_default_parquet(root: Path) -> Path:
    matches = sorted(root.rglob("*.parquet"))
    if not matches:
        raise SystemExit(f"No parquet files found under {root}; pass prep.input_parquet explicitly")
    return matches[0]


def find_decoded_parquets(root: Path, *, decoded_glob: str | None = None) -> list[Path]:
    if root.is_file():
        return [root]
    glob_pattern = decoded_glob or "LP-100-decoded-*.parquet"
    matches = sorted(root.rglob(glob_pattern))
    if not matches and decoded_glob:
        print(
            f"WARNING: No shards matching {glob_pattern!r} under {root}; "
            "falling back to *.parquet (may include unrelated files)"
        )
        matches = sorted(root.rglob("*.parquet"))
    elif not matches:
        matches = sorted(root.rglob("*.parquet"))
    if not matches:
        raise SystemExit(f"No decoded parquet shards found under {root}")
    return matches


def _read_parquet(path: Path):
    import pandas as pd

    return pd.read_parquet(path)


def iter_clip_audio_paths(df, *, limit_anchors: int = 0):
    count = 0
    for _, row in df.iterrows():
        clips = parse_clips(row.get("clips"))
        if not clips:
            continue
        for clip in clips:
            yield clip["audio_path"]
        count += 1
        if limit_anchors and count >= limit_anchors:
            break


def collect_needed_audio_keys(
    df,
    *,
    limit_anchors: int = 0,
) -> set[str]:
    needed: set[str] = set()
    count = 0
    for _, row in df.iterrows():
        clips = parse_clips(row.get("clips"))
        if not clips:
            continue
        for clip in clips:
            needed.add(clip_to_audio_rel(clip["audio_path"]))
        count += 1
        if limit_anchors and count >= limit_anchors:
            break
    return needed


def resolve_decoded_root(
    prep: dict[str, Any],
    config: dict,
    paths: dict,
    dataset_id: str | None,
) -> Path:
    if prep.get("decoded_parquet_root"):
        return Path(prep["decoded_parquet_root"])
    if prep.get("dataset_root"):
        return Path(prep["dataset_root"])
    stage2_prep = (config.get("stage2") or {}).get("prep") or {}
    if stage2_prep.get("data_root"):
        return Path(stage2_prep["data_root"])
    if dataset_id:
        root = resolve_data_root(paths, dataset_id)
        if root is not None:
            return root
    return Path(paths["libriphrase100_root"])


def resolve_output_subdir(prep: dict[str, Any], config: dict) -> str:
    if prep.get("output_subdir"):
        return str(prep["output_subdir"])
    stage2_prep = (config.get("stage2") or {}).get("prep") or {}
    return stage2_prep.get("output_subdir", "stage2_qbyt")


def load_decoded_audio(
    decoded_parquet_paths: list[Path],
    needed_keys: set[str],
    *,
    num_workers: int = 1,
    read_parquet: Callable[[Path], Any] = _read_parquet,
    on_shard_done: Callable[[Path, int], None] | None = None,
) -> dict[str, tuple[np.ndarray, int]]:
    if num_workers <= 1 or len(decoded_parquet_paths) <= 1:
        audio_by_rel: dict[str, tuple[np.ndarray, int]] = {}
        for parquet_path in decoded_parquet_paths:
            shard_hits = 0
            for audio_rel, audio, sample_rate in iter_decoded_audio_rows(
                [parquet_path],
                needed_keys - set(audio_by_rel),
                read_parquet=read_parquet,
            ):
                array = np.asarray(audio, dtype=np.float32)
                if array.ndim != 1:
                    raise SystemExit(
                        f"Expected mono 1-D audio for {audio_rel}, got shape {array.shape}"
                    )
                audio_by_rel[audio_rel] = (array, int(sample_rate))
                shard_hits += 1
            if on_shard_done is not None:
                on_shard_done(parquet_path, shard_hits)
            if len(audio_by_rel) >= len(needed_keys):
                break
        return audio_by_rel

    audio_by_rel = {}
    worker_count = min(num_workers, len(decoded_parquet_paths))
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = {
            executor.submit(
                scan_decoded_parquet_shard,
                parquet_path,
                needed_keys,
                read_parquet=read_parquet,
            ): parquet_path
            for parquet_path in decoded_parquet_paths
        }
        for future in as_completed(futures):
            parquet_path = futures[future]
            shard = future.result()
            shard_hits = 0
            for audio_rel, (audio, sample_rate) in shard.items():
                if audio_rel in audio_by_rel:
                    continue
                array = np.asarray(audio, dtype=np.float32)
                if array.ndim != 1:
                    raise SystemExit(
                        f"Expected mono 1-D audio for {audio_rel}, got shape {array.shape}"
                    )
                audio_by_rel[audio_rel] = (array, int(sample_rate))
                shard_hits += 1
            if on_shard_done is not None:
                on_shard_done(parquet_path, shard_hits)
    return audio_by_rel


@hydra.main(version_base=None, config_path=str(CONFIG_DIR), config_name="config")
def main(cfg: DictConfig) -> None:
    try:
        import pandas as pd
    except ImportError as exc:
        raise SystemExit("Missing dependency pandas/pyarrow. Install with: pip install pandas pyarrow") from exc

    config = resolved_config(cfg)
    require_sections(config, ["paths"])
    prep = OmegaConf.to_container(cfg.prep, resolve=True)
    if not isinstance(prep, dict):
        prep = {}

    reporter = Stage2PrepReporter(use_rich=bool(prep.get("use_rich", True)))
    num_workers = resolve_num_workers(int(prep.get("num_workers", 0)))

    paths = config["paths"]
    limit_anchors = int(prep.get("limit_anchors", 0))

    input_path = (
        Path(prep["input_parquet"])
        if prep.get("input_parquet")
        else find_default_parquet(Path(paths["libriphrase100_root"]))
    )
    processed_root = Path(paths["processed_root"])
    feature_root = Path(paths.get("feature_root", processed_root.parent / "features"))
    stage2 = config.get("stage2") or {}
    fbank_cfg = get_fbank_config(config)

    output_subdir = resolve_output_subdir(prep, config)
    output_dir = processed_root / output_subdir
    clips_dir = output_dir / "clips"
    distances_dir = output_dir / "distances"
    fbank_dir = Path(stage2["wav_dir"]) if stage2.get("wav_dir") else feature_root / "fbank"
    output_parquet = output_dir / OUTPUT_PARQUET_NAME

    reporter.section("Plan")
    reporter.print_plan(
        [
            ("input_parquet", str(input_path)),
            ("decoded_root", "(resolved after dataset detection)"),
            ("output_parquet", str(output_parquet)),
            ("clips_dir", str(clips_dir)),
            ("distances_dir", str(distances_dir)),
            ("fbank_dir", str(fbank_dir)),
            ("fbank_backend", fbank_cfg.backend),
            ("target_sample_rate", str(fbank_cfg.target_sample_rate or "source")),
            ("limit_anchors", str(limit_anchors or "all")),
            ("num_workers", str(num_workers)),
        ]
    )

    reporter.section("Load aggregated parquet")
    df = pd.read_parquet(input_path)
    reporter.info(f"Loaded {len(df)} rows from {input_path}")

    dataset_id = infer_dataset_id(iter_clip_audio_paths(df, limit_anchors=limit_anchors))
    dataset_label = dataset_id or "unknown (LP-100 defaults)"
    reporter.info(f"Detected dataset: {dataset_label}")

    needed_keys = collect_needed_audio_keys(df, limit_anchors=limit_anchors)
    reporter.info(f"Collected {len(needed_keys)} unique clip audio keys")

    decoded_root = resolve_decoded_root(prep, config, paths, dataset_id)
    decoded_glob = decoded_glob_for_dataset(dataset_id) if dataset_id else "LP-100-decoded-*.parquet"
    decoded_parquet_paths = find_decoded_parquets(decoded_root, decoded_glob=decoded_glob)
    reporter.info(f"Decoded root: {decoded_root}")
    reporter.info(f"Decoded glob: {decoded_glob}")
    reporter.info(f"Found {len(decoded_parquet_paths)} decoded parquet shards")

    fbank_params = fbank_kwargs(fbank_cfg)
    fbank_extractor = FbankExtractor(**fbank_params)
    compute_fbank_fn = partial(
        compute_fbank_for_clip,
        extractor=fbank_extractor,
        **fbank_params,
    )

    reporter.section("Build anchor metadata")
    with reporter.track("Build anchor metadata", total=None) as anchor_bar:

        def on_anchor(stage: str, value: int) -> None:
            if stage == "anchor_total" and anchor_bar is not None:
                anchor_bar.set_total(value)
            elif stage == "anchor" and anchor_bar is not None:
                anchor_bar.update(value)

        paper_df, fbank_targets, stats = build_anchor_metadata(
            df,
            clips_dir=clips_dir,
            distances_dir=distances_dir,
            fbank_dir=fbank_dir,
            limit_anchors=limit_anchors,
            force_g2p_recompute=bool(prep.get("force_g2p_recompute", False)),
            on_progress=on_anchor,
        )
    reporter.info(
        f"Prepared {stats['anchors']} anchors; {len(fbank_targets)} fbank files to compute "
        f"({stats['fbank_skipped']} already present)"
    )
    if stats["g2p_recomputed"]:
        reporter.warn(
            "Recomputed ngram_g2p with g2p_en: the input parquet had no stress markers "
            "(or recompute was forced), so its phonemes could not match the vocabulary."
        )

    reporter.section("Write output")
    output_dir.mkdir(parents=True, exist_ok=True)
    paper_df.to_parquet(output_parquet, index=False)
    del paper_df

    reporter.section("Compute fbank (streaming decoded shards)")
    with reporter.tasks() as tasks:
        shard_task = tasks.add("Scan decoded parquet shards", total=len(decoded_parquet_paths))
        fbank_task = tasks.add("Compute fbank features", total=len(fbank_targets))

        def on_fbank(stage: str, value: int) -> None:
            if stage == "fbank_total":
                tasks.set_total(fbank_task, value)
            elif stage == "fbank":
                tasks.advance(fbank_task, value)

        fbank_written, missing = stream_fbank_from_decoded(
            decoded_parquet_paths,
            fbank_targets,
            read_parquet=_read_parquet,
            compute_fbank=compute_fbank_fn,
            num_workers=num_workers,
            on_progress=on_fbank,
            on_shard_done=lambda _path, _hits: tasks.advance(shard_task, 1),
        )
    stats["fbank_written"] = fbank_written
    stats["missing_audio"] = missing

    reporter.print_stats(
        [
            ("anchors", str(stats["anchors"])),
            ("g2p_recomputed", str(bool(stats["g2p_recomputed"]))),
            ("clips_total", str(stats["clips_total"])),
            ("fbank_written", str(stats["fbank_written"])),
            ("fbank_skipped", str(stats["fbank_skipped"])),
            ("missing_in_decoded", str(missing)),
            ("output_parquet", str(output_parquet)),
        ]
    )

    if missing:
        reporter.warn(
            f"{missing} referenced clips were not found in decoded parquet "
            "(their fbank files were skipped)"
        )

    reporter.done("Stage II paper prep complete.")


if __name__ == "__main__":
    main()
