#!/usr/bin/env python3
"""Evaluate Stage-II-only DMA-KWS false accepts on the MUSAN corpus.

Slides a fixed-length window over each MUSAN audio file and scores every window
with the Stage-II QbyT verifier for a single keyword. Reports overall and
per-subset (music/noise/speech) FA/hour.

Output schema matches ``scripts/eval_stage2_clips.py``:
  - ``results.jsonl`` has one JSON object per scored window.
  - ``summary.json`` contains aggregate metrics.
  - ``fa_per_hour_curve.png`` and ``fa_per_hour_curve.csv`` record the exact
    threshold/FA-hour sweep when ``prep.plot_curves`` is enabled and valid
    scored windows are available.

Set ``prep.keyword_phonemes`` to a space-separated ARPAbet sequence (or a
Hydra list) to override keyword G2P. A missing or blank value retains automatic
G2P. These are the only JSON files written to ``prep.output_dir``; the plot and
its CSV are the only optional side artifacts.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterator, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Mapping

import hydra
from omegaconf import DictConfig, OmegaConf

from dma_kws.config import require_sections
from dma_kws.hydra_app import CONFIG_DIR, resolved_config
from dma_kws.inference.detection_plots import (
    DEFAULT_PLOT_DPI,
    write_false_accept_rate_plot as _write_false_accept_rate_plot,
)
from dma_kws.inference.keyword_set import (
    WINDOW_EVAL_PROTOCOL,
    KeywordEvalConfigError,
    KeywordSetScoreError,
    keyword_eval_mode,
    keyword_eval_provenance_block,
    keyword_set_summary_fields,
)
from dma_kws.inference.manifest import iter_audio_files, load_audio_file_list
from dma_kws.pathing import resolve_dict_path
from dma_kws.inference.metrics import summarize_false_accept_rate
from dma_kws.inference.musan_fa import (
    audio_duration_sec,
    compute_file_metrics,
    detect_subset,
    metrics_record,
    musan_catalog_sha256,
    musan_keyword_set_result_record,
    musan_result_record,
    select_shard,
    subset_summary,
)
from dma_kws.inference.stage2_clip import PreparedFileWindows, Stage2ClipRunner
from dma_kws.inference.stage2_reporting import build_score_provenance
from dma_kws.inference.stage2_verifier import resolve_inference_amp
from dma_kws.training.device import resolve_accelerator


def _positive_int(value: object, *, default: int, name: str) -> int:
    parsed = int(value or 0)
    if parsed < 0:
        raise SystemExit(f"{name} must be >= 0")
    return default if parsed == 0 else parsed


def _iter_prepared_windows(
    runner: Stage2ClipRunner,
    source_rows: Sequence[Mapping[str, Any]],
    *,
    keyword: str,
    window_sec: float,
    hop_sec: float,
    keyword_phonemes: list[str] | None,
    fbank_windows: str,
    num_workers: int,
) -> Iterator[tuple[Mapping[str, Any], PreparedFileWindows]]:
    """Yield ``(source_row, prepared windows)``, prefetching when requested."""

    def prepare(row: Mapping[str, Any]) -> PreparedFileWindows:
        return runner.prepare_file_windows(
            row["audio_path"],
            keyword,
            window_sec=window_sec,
            hop_sec=hop_sec,
            keyword_phonemes=keyword_phonemes,
            fbank_windows=fbank_windows,
        )

    if num_workers <= 1:
        for row in source_rows:
            yield row, prepare(row)
        return

    pending: list[tuple[Mapping[str, Any], Future[PreparedFileWindows]]] = []
    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        for row in source_rows:
            pending.append((row, pool.submit(prepare, row)))
            if len(pending) >= num_workers:
                ready_row, future = pending.pop(0)
                yield ready_row, future.result()
        for ready_row, future in pending:
            yield ready_row, future.result()


def _iter_window_feats(
    runner: Stage2ClipRunner,
    source_rows: Sequence[Mapping[str, Any]],
    *,
    window_sec: float,
    hop_sec: float,
    fbank_windows: str,
    num_workers: int,
) -> Iterator[tuple[Mapping[str, Any], list[Any], list[tuple[int, float, float]]]]:
    def prepare(row: Mapping[str, Any]):
        return runner.extract_file_window_features(
            row["audio_path"],
            window_sec=window_sec,
            hop_sec=hop_sec,
            fbank_windows=fbank_windows,
        )

    if num_workers <= 1:
        for row in source_rows:
            feats, spans = prepare(row)
            yield row, feats, spans
        return

    pending: list[tuple[Mapping[str, Any], Future]] = []
    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        for row in source_rows:
            pending.append((row, pool.submit(prepare, row)))
            if len(pending) >= num_workers:
                ready_row, future = pending.pop(0)
                feats, spans = future.result()
                yield ready_row, feats, spans
        for ready_row, future in pending:
            feats, spans = future.result()
            yield ready_row, feats, spans


def run_eval(cfg: DictConfig) -> dict:
    try:
        import torch
    except ImportError as exc:
        raise SystemExit(
            "Missing torch/torchaudio. Install CUDA PyTorch on the remote training machine first."
        ) from exc

    torch.multiprocessing.set_sharing_strategy("file_system")

    config = resolved_config(cfg)
    require_sections(config, ["paths", "stage1", "stage2", "demo", "tokenizer"])
    prep = OmegaConf.to_container(cfg.prep, resolve=True)
    if not isinstance(prep, dict):
        prep = {}
    run_cfg = cfg.run

    eval_mode = keyword_eval_mode(prep)
    keyword = str(prep.get("keyword", "")).strip()
    if eval_mode != "any" and not keyword:
        raise SystemExit("prep.keyword is required")
    if eval_mode == "any" and keyword:
        raise SystemExit(
            "prep.keyword_eval.mode=any cannot be combined with non-empty prep.keyword"
        )
    musan_root = str(prep.get("musan_root", ""))
    if not musan_root:
        raise SystemExit("prep.musan_root is required")
    musan_root_path = Path(musan_root)
    if not musan_root_path.is_dir():
        raise SystemExit(f"MUSAN root not found: {musan_root}")
    stage2_ckpt = str(prep.get("stage2_ckpt", ""))
    if not stage2_ckpt:
        raise SystemExit("prep.stage2_ckpt is required")
    stage2_calibration = str(prep.get("stage2_calibration", "")).strip()

    window_sec = float(prep.get("window_sec", 0.0) or 3.0)
    hop_sec = float(prep.get("hop_sec", 0.0) or 3.0)
    if window_sec <= 0 or hop_sec <= 0:
        raise SystemExit("window_sec and hop_sec must be positive")
    batch_size = _positive_int(prep.get("batch_size"), default=64, name="prep.batch_size")
    num_workers = int(prep.get("num_workers", 0) or 0)
    if num_workers < 0:
        raise SystemExit("prep.num_workers must be >= 0")
    try:
        amp = resolve_inference_amp(prep.get("amp"))
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    fbank_windows = str(prep.get("fbank_windows") or "independent").strip().lower()
    if fbank_windows not in {"independent", "file"}:
        raise SystemExit(
            "prep.fbank_windows must be 'independent' or 'file', "
            f"got {prep.get('fbank_windows')!r}"
        )
    num_shards = int(prep.get("num_shards", 1) or 1)
    shard_index = int(prep.get("shard_index", 0) or 0)
    if num_shards < 1:
        raise SystemExit("prep.num_shards must be >= 1")
    if shard_index < 0 or shard_index >= num_shards:
        raise SystemExit(
            f"prep.shard_index must be in [0, {num_shards}), got {shard_index}"
        )

    musan_audio_list = str(prep.get("musan_audio_list_path", "") or "").strip()
    try:
        audio_files = (
            load_audio_file_list(musan_audio_list)
            if musan_audio_list
            else iter_audio_files(musan_root_path)
        )
        catalog_sha256 = musan_catalog_sha256(audio_files, musan_root_path)
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    if not audio_files:
        source = musan_audio_list or musan_root
        raise SystemExit(f"No audio files found in MUSAN evaluation source: {source}")

    output_dir_override = str(prep.get("output_dir", ""))
    output_dir = Path(output_dir_override or "outputs/eval_musan_fa")
    output_dir.mkdir(parents=True, exist_ok=True)

    accelerator, _ = resolve_accelerator(str(run_cfg.device))
    device = torch.device(accelerator if accelerator == "cpu" else "cuda")
    runner = Stage2ClipRunner.from_config(config, prep, device)
    threshold = float(runner._demo_cfg.get("qbyt_threshold", 0.5))
    keyword_set = None
    if eval_mode == "any":
        try:
            keyword_set = runner.enroll_keyword_set(
                prep,
                tokenizer_dict_path=resolve_dict_path(config),
            )
            from dma_kws.inference.keyword_set import require_qbyt_threshold

            threshold = require_qbyt_threshold(
                runner._demo_cfg.get("qbyt_threshold", 0.5),
                field="demo.qbyt_threshold",
            )
        except KeywordEvalConfigError as exc:
            raise SystemExit(str(exc)) from exc
        if keyword_set is None:
            raise SystemExit("prep.keyword_eval.mode=any produced no keyword set")

    keyword_phonemes: list[str] = []
    keyword_phonemes_source = None
    override_phonemes = None
    if eval_mode != "any":
        raw_keyword_phonemes = prep.get("keyword_phonemes")
        use_keyword_phoneme_override = raw_keyword_phonemes is not None and not (
            isinstance(raw_keyword_phonemes, str)
            and not raw_keyword_phonemes.strip()
        )
        try:
            keyword_phonemes = runner.resolve_keyword_phonemes(
                keyword,
                raw_keyword_phonemes if use_keyword_phoneme_override else None,
                field_name="prep.keyword_phonemes",
            )
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        keyword_phonemes_source = (
            "prep.keyword_phonemes" if use_keyword_phoneme_override else "g2p"
        )
        override_phonemes = (
            keyword_phonemes if use_keyword_phoneme_override else None
        )

    stream_description = runner.stream_policy.describe()
    provenance = build_score_provenance(
        config,
        checkpoint_path=stage2_ckpt,
        calibration_path=stage2_calibration or None,
        stream=stream_description,
        # Sliding windows are scored as-is; unlike clip evaluation, this path
        # does not add zero-valued waveform context around each window.
        left_padding_ms=0,
        right_padding_ms=0,
        keyword_eval=keyword_eval_provenance_block(keyword_set, mode=eval_mode),
    )
    catalog = [
        {
            "audio_path": str(audio_path.resolve()),
            "subset": detect_subset(audio_path, musan_root_path),
            "duration_sec": audio_duration_sec(audio_path),
        }
        for audio_path in audio_files
    ]
    try:
        source_rows = select_shard(
            catalog,
            num_shards=num_shards,
            shard_index=shard_index,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    results_path = output_dir / "results.jsonl"
    all_results: list[dict[str, Any]] = []
    subset_results: dict[str, list[dict[str, Any]]] = defaultdict(list)
    subset_hours: dict[str, float] = defaultdict(float)
    total_hours = 0.0
    source_files: list[dict[str, Any]] = []

    with results_path.open("w", encoding="utf-8") as results_handle:
        if eval_mode == "any":
            window_iter = _iter_window_feats(
                runner,
                source_rows,
                window_sec=window_sec,
                hop_sec=hop_sec,
                fbank_windows=fbank_windows,
                num_workers=num_workers,
            )
            for source_row, feats, spans in window_iter:
                audio_path = source_row["audio_path"]
                subset = source_row["subset"]
                duration = float(source_row["duration_sec"])
                total_hours += duration / 3600.0
                subset_hours[subset] += duration / 3600.0
                try:
                    window_results = runner.score_window_features_multi(
                        audio_path,
                        feats,
                        spans,
                        keyword_set,
                        batch_size=batch_size,
                    )
                except (KeywordEvalConfigError, KeywordSetScoreError, RuntimeError) as exc:
                    raise SystemExit(str(exc)) from exc
                source_files.append(
                    {
                        "path": audio_path,
                        "subset": subset,
                        "duration": duration,
                        "scored_window_count": len(window_results),
                    }
                )
                for window_result in window_results:
                    record = musan_keyword_set_result_record(
                        audio_path,
                        subset,
                        window_result,
                        window_index=int(window_result["window_index"]),
                    )
                    all_results.append(record)
                    subset_results[subset].append(record)
                    results_handle.write(
                        json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n"
                    )
        else:
            for source_row, prepared in _iter_prepared_windows(
                runner,
                source_rows,
                keyword=keyword,
                window_sec=window_sec,
                hop_sec=hop_sec,
                keyword_phonemes=override_phonemes,
                fbank_windows=fbank_windows,
                num_workers=num_workers,
            ):
                audio_path = source_row["audio_path"]
                subset = source_row["subset"]
                duration = float(source_row["duration_sec"])
                total_hours += duration / 3600.0
                subset_hours[subset] += duration / 3600.0
                window_results = runner.score_prepared_windows(
                    prepared,
                    batch_size=batch_size,
                )
                source_files.append(
                    {
                        "path": audio_path,
                        "subset": subset,
                        "duration": duration,
                        "scored_window_count": len(window_results),
                    }
                )
                for window_result in window_results:
                    record = musan_result_record(
                        audio_path,
                        keyword,
                        subset,
                        window_result,
                        window_index=int(window_result["window_index"]),
                    )
                    all_results.append(record)
                    subset_results[subset].append(record)
                    results_handle.write(
                        json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n"
                    )

    summary: dict[str, Any] = {
        "num_samples": len(all_results),
        "num_skipped": sum(
            bool(record.get("skipped", False)) for record in all_results
        ),
        "output_dir": str(output_dir.resolve()),
        "musan_root": str(musan_root_path.resolve()),
        "musan_audio_list_path": (
            str(Path(musan_audio_list).expanduser().resolve())
            if musan_audio_list
            else None
        ),
        "musan_catalog_sha256": catalog_sha256,
        "keyword": keyword,
        "keyword_phonemes": keyword_phonemes,
        "keyword_phonemes_source": keyword_phonemes_source,
        "stage2_ckpt": stage2_ckpt,
        "stage2_calibration": stage2_calibration or None,
        "window_sec": window_sec,
        "hop_sec": hop_sec,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "amp": amp or "off",
        "fbank_windows": fbank_windows,
        "num_shards": num_shards,
        "shard_index": shard_index,
        "total_files": len(source_rows),
        "total_hours": total_hours,
        "stream": stream_description,
        "provenance": provenance,
        "keyword_eval_mode": eval_mode,
        "fa_count_unit": "window",
        "source_files": source_files,
        "file_metrics": compute_file_metrics(
            all_results,
            source_files,
            threshold=threshold,
        ),
    }
    if eval_mode == "any" and keyword_set is not None:
        summary["eval_protocol"] = WINDOW_EVAL_PROTOCOL
        summary.update(keyword_set_summary_fields(keyword_set))
        summary["score_semantics"] = "max_over_keywords_and_pronunciations"
    scored_windows = [
        record for record in all_results if not bool(record.get("skipped", False))
    ]
    if not scored_windows:
        summary["metrics_status"] = "no_scored_windows"

    overall_metrics = summarize_false_accept_rate(
        [metrics_record(record) for record in all_results],
        threshold=threshold,
        total_hours=total_hours,
    )
    if overall_metrics:
        summary["metrics"] = overall_metrics

    subsets: dict[str, dict[str, Any]] = {}
    for subset in sorted(subset_results):
        subsets[subset] = subset_summary(
            subset_results[subset],
            threshold=threshold,
            total_hours=subset_hours[subset],
        )
    if subsets:
        summary["subsets"] = subsets

    if bool(prep.get("plot_curves", True)):
        summary["plots"] = _write_false_accept_rate_plot(
            all_results,
            output_dir=output_dir,
            threshold=threshold,
            total_hours=total_hours,
            dpi=int(prep.get("plot_dpi", DEFAULT_PLOT_DPI)),
        )

    summary_path = output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, allow_nan=False)

    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))
    return summary


@hydra.main(version_base=None, config_path=str(CONFIG_DIR), config_name="config")
def main(cfg: DictConfig) -> None:
    run_eval(cfg)


if __name__ == "__main__":
    main()
