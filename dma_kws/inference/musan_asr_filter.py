"""Filter MUSAN audio that a WeNet ASR model transcribes as wake words."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unicodedata
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import yaml

from dma_kws.inference.manifest import iter_audio_files


@dataclass(frozen=True)
class WenetModelFiles:
    checkpoint: Path
    config: Path
    cmvn: Path
    bpe_model: Path
    units: Path


@dataclass(frozen=True)
class AudioChunk:
    key: str
    source: Path
    path: Path
    start_sec: float
    end_sec: float


def _resolve_file(value: str | Path | None) -> Path | None:
    if value is None or not str(value).strip():
        return None
    return Path(value).expanduser().resolve()


def _discover_one(model_dir: Path, names: Sequence[str], label: str) -> Path:
    matches = [model_dir / name for name in names if (model_dir / name).is_file()]
    if not matches:
        raise FileNotFoundError(f"Could not find {label} under {model_dir}: {', '.join(names)}")
    return matches[0]


def resolve_model_files(
    model_dir: str | Path | None = None,
    *,
    checkpoint: str | Path | None = None,
    config: str | Path | None = None,
    cmvn: str | Path | None = None,
    bpe_model: str | Path | None = None,
    units: str | Path | None = None,
) -> WenetModelFiles:
    root = _resolve_file(model_dir)
    if root is not None and not root.is_dir():
        raise NotADirectoryError(f"WeNet model directory not found: {root}")

    checkpoint_path = _resolve_file(checkpoint)
    if checkpoint_path is None:
        if root is None:
            raise ValueError("model_dir or checkpoint is required")
        pts = sorted(root.glob("*.pt"))
        preferred = [path for path in pts if path.name in {"final.pt", "avg.pt"}]
        if len(pts) != 1 and len(preferred) != 1:
            names = ", ".join(path.name for path in pts) or "none"
            raise ValueError(f"Expected one checkpoint under {root}, found: {names}; pass --checkpoint")
        checkpoint_path = preferred[0] if preferred else pts[0]

    def supplied_or_discovered(value: str | Path | None, names: Sequence[str], label: str) -> Path:
        path = _resolve_file(value)
        if path is not None:
            return path
        if root is None:
            raise ValueError(f"model_dir or {label} path is required")
        return _discover_one(root, names, label)

    files = WenetModelFiles(
        checkpoint=checkpoint_path,
        config=supplied_or_discovered(config, ("train.yaml",), "config"),
        cmvn=supplied_or_discovered(cmvn, ("global_cmvn", "global_cvmn"), "CMVN"),
        bpe_model=supplied_or_discovered(
            bpe_model, ("unigram5000.model", "bpe.model"), "BPE model"
        ),
        units=supplied_or_discovered(units, ("units.txt", "words.txt"), "unit table"),
    )
    for field, path in asdict(files).items():
        if not path.is_file():
            raise FileNotFoundError(f"WeNet {field} file not found: {path}")
    return files


def normalize_match_text(value: str) -> str:
    text = unicodedata.normalize("NFKC", value).casefold()
    return "".join(char for char in text if char.isalnum())


def validate_keywords(keywords: Iterable[str]) -> list[str]:
    values: list[str] = []
    seen: set[str] = set()
    for value in keywords:
        keyword = value.strip()
        normalized = normalize_match_text(keyword)
        if not normalized:
            raise ValueError("keywords must contain non-empty letters or numbers")
        if normalized not in seen:
            seen.add(normalized)
            values.append(keyword)
    if not values:
        raise ValueError("At least one keyword is required")
    return values


def validate_filter_options(
    *,
    threshold: float,
    window_sec: float,
    overlap_sec: float,
    batch_size: int,
    beam_size: int,
    mode: str,
) -> None:
    if not 0 <= threshold <= 100:
        raise ValueError("threshold must be between 0 and 100")
    if window_sec <= 0:
        raise ValueError("window_sec must be positive")
    if overlap_sec < 0 or overlap_sec >= window_sec:
        raise ValueError("overlap_sec must be non-negative and smaller than window_sec")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if beam_size <= 0:
        raise ValueError("beam_size must be positive")
    if not mode.strip():
        raise ValueError("mode must not be empty")


def best_keyword_match(transcript: str, keywords: Sequence[str]) -> tuple[str | None, float]:
    from dma_kws.inference.musan_asr_rescore import bounded_keyword_match

    keyword, _, score = bounded_keyword_match(transcript, keywords)
    return keyword, score


def _patch_wenet_config(files: WenetModelFiles, output_path: Path) -> None:
    with files.config.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"WeNet config must be a mapping: {files.config}")

    cmvn_conf = config.get("cmvn_conf")
    if isinstance(cmvn_conf, dict):
        cmvn_conf["cmvn_file"] = str(files.cmvn)

    tokenizer_conf = config.get("tokenizer_conf")
    if isinstance(tokenizer_conf, dict):
        tokenizer_conf["symbol_table_path"] = str(files.units)
        tokenizer_conf["bpe_path"] = str(files.bpe_model)
        if "bpe_model" in tokenizer_conf:
            tokenizer_conf["bpe_model"] = str(files.bpe_model)
    replacements = {
        "cmvn_file": files.cmvn,
        "dict": files.units,
        "symbol_table": files.units,
        "bpe_model": files.bpe_model,
    }
    for key, value in replacements.items():
        if key in config:
            config[key] = str(value)

    with output_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, allow_unicode=True, sort_keys=False)


def create_audio_chunks(
    audio_files: Sequence[Path],
    output_dir: Path,
    *,
    window_sec: float,
    overlap_sec: float,
) -> list[AudioChunk]:
    if window_sec <= 0:
        raise ValueError("window_sec must be positive")
    if overlap_sec < 0 or overlap_sec >= window_sec:
        raise ValueError("overlap_sec must be non-negative and smaller than window_sec")
    try:
        import soundfile as sf
    except ImportError as exc:
        raise ImportError("soundfile is required for MUSAN ASR filtering") from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    chunks: list[AudioChunk] = []
    for file_index, source in enumerate(audio_files):
        info = sf.info(str(source))
        window_frames = max(1, round(window_sec * info.samplerate))
        step_frames = max(1, round((window_sec - overlap_sec) * info.samplerate))
        start = 0
        chunk_index = 0
        while start < info.frames:
            end = min(start + window_frames, info.frames)
            waveform, sample_rate = sf.read(
                str(source), start=start, stop=end, dtype="float32", always_2d=True
            )
            key = f"audio_{file_index:08d}_chunk_{chunk_index:05d}"
            chunk_path = output_dir / f"{key}.flac"
            sf.write(str(chunk_path), waveform, sample_rate, format="FLAC")
            chunks.append(
                AudioChunk(
                    key=key,
                    source=source,
                    path=chunk_path,
                    start_sec=start / info.samplerate,
                    end_sec=end / info.samplerate,
                )
            )
            if end == info.frames:
                break
            start += step_frames
            chunk_index += 1
    return chunks


def _wenet_environment(wenet_root: str | Path | None) -> dict[str, str]:
    env = os.environ.copy()
    root_value = wenet_root or env.get("WENET_ROOT")
    if root_value:
        root = Path(root_value).expanduser().resolve()
        if not (root / "wenet" / "bin" / "recognize.py").is_file():
            raise FileNotFoundError(f"Invalid WeNet root (missing wenet/bin/recognize.py): {root}")
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = str(root) + (os.pathsep + existing if existing else "")
    return env


def _recognize_help(env: Mapping[str, str]) -> str:
    result = subprocess.run(
        [sys.executable, "-m", "wenet.bin.recognize", "--help"],
        env=dict(env),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "Could not run wenet.bin.recognize; install WeNet or pass --wenet-root.\n"
            + result.stdout.strip()
        )
    return result.stdout


def _write_wenet_manifest(chunks: Sequence[AudioChunk], path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(
                json.dumps({"key": chunk.key, "wav": str(chunk.path), "txt": ""}, ensure_ascii=False)
                + "\n"
            )


def _parse_transcript_file(path: Path) -> dict[str, str]:
    transcripts: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\n")
            if not line:
                continue
            key, separator, text = line.partition(" ")
            transcripts[key] = text.strip() if separator else ""
    return transcripts


def run_wenet_recognize(
    chunks: Sequence[AudioChunk],
    files: WenetModelFiles,
    workspace: Path,
    *,
    wenet_root: str | Path | None = None,
    device: str = "cpu",
    batch_size: int = 8,
    beam_size: int = 10,
    mode: str = "attention_rescoring",
) -> dict[str, str]:
    if not chunks:
        return {}
    workspace.mkdir(parents=True, exist_ok=True)
    env = _wenet_environment(wenet_root)
    help_text = _recognize_help(env)
    config_path = workspace / "train.resolved.yaml"
    manifest_path = workspace / "chunks.list"
    _patch_wenet_config(files, config_path)
    _write_wenet_manifest(chunks, manifest_path)

    command = [
        sys.executable,
        "-m",
        "wenet.bin.recognize",
        "--config",
        str(config_path),
        "--test_data",
        str(manifest_path),
        "--data_type",
        "raw",
        "--checkpoint",
        str(files.checkpoint),
        "--batch_size",
        str(batch_size),
        "--beam_size",
        str(beam_size),
    ]
    if "--device" in help_text:
        command.extend(("--device", device.split(":", 1)[0]))
    if "--gpu" in help_text:
        gpu = device.split(":", 1)[1] if device.startswith("cuda:") else "0" if device == "cuda" else "-1"
        command.extend(("--gpu", gpu))
    if "--dict" in help_text:
        command.extend(("--dict", str(files.units)))
    if "--symbol_table" in help_text:
        command.extend(("--symbol_table", str(files.units)))
    if "--bpe_model" in help_text:
        command.extend(("--bpe_model", str(files.bpe_model)))
    if "--cmvn_file" in help_text:
        command.extend(("--cmvn_file", str(files.cmvn)))

    if "--modes" in help_text:
        command.extend(("--modes", mode))
    elif "--mode" in help_text:
        command.extend(("--mode", mode))
    else:
        raise RuntimeError("Unsupported wenet.bin.recognize CLI: missing --modes/--mode")

    result_dir: Path | None = None
    if "--result_dir" in help_text:
        result_dir = workspace / "decode"
        transcript_path = result_dir / mode / "text"
        command.extend(("--result_dir", str(result_dir)))
    elif "--result_file" in help_text:
        transcript_path = workspace / "decode.text"
        command.extend(("--result_file", str(transcript_path)))
    else:
        raise RuntimeError("Unsupported wenet.bin.recognize CLI: missing --result_dir/--result_file")

    subprocess.run(command, env=env, check=True)
    if not transcript_path.is_file() and result_dir is not None:
        candidates = list(result_dir.rglob("text"))
        if len(candidates) == 1:
            transcript_path = candidates[0]
    if not transcript_path.is_file():
        raise FileNotFoundError(f"Could not locate WeNet transcript output: {transcript_path}")
    transcripts = _parse_transcript_file(transcript_path)
    missing = [chunk.key for chunk in chunks if chunk.key not in transcripts]
    if missing:
        raise RuntimeError(f"WeNet returned no transcript for {len(missing)} chunks; first: {missing[0]}")
    return transcripts


def build_filter_records(
    audio_files: Sequence[Path],
    chunks: Sequence[AudioChunk],
    transcripts: Mapping[str, str],
    keywords: Sequence[str],
    *,
    threshold: float,
    musan_root: Path,
) -> list[dict]:
    if not 0 <= threshold <= 100:
        raise ValueError("threshold must be between 0 and 100")
    grouped: dict[Path, list[AudioChunk]] = defaultdict(list)
    for chunk in chunks:
        grouped[chunk.source].append(chunk)

    records: list[dict] = []
    for audio_path in audio_files:
        windows: list[dict] = []
        best_keyword: str | None = None
        best_score = 0.0
        for chunk in grouped[audio_path]:
            transcript = transcripts[chunk.key]
            keyword, score = best_keyword_match(transcript, keywords)
            windows.append(
                {
                    "start_sec": chunk.start_sec,
                    "end_sec": chunk.end_sec,
                    "transcript": transcript,
                    "best_keyword": keyword,
                    "match_score": score,
                }
            )
            if score > best_score:
                best_keyword, best_score = keyword, score
        matched = best_keyword is not None and best_score >= threshold
        relative_path = audio_path.resolve().relative_to(musan_root.resolve())
        records.append(
            {
                "audio_path": str(audio_path.resolve()),
                "relative_path": str(relative_path),
                "subset": relative_path.parts[0] if len(relative_path.parts) > 1 else "other",
                "matched": matched,
                "matched_keyword": best_keyword if matched else None,
                "best_match_score": best_score,
                "windows": windows,
            }
        )
    return records


def copy_filtered_tree(musan_root: Path, output_root: Path, records: Sequence[Mapping]) -> None:
    source = musan_root.resolve()
    destination = output_root.resolve()
    if destination.exists():
        raise FileExistsError(f"Output path already exists: {destination}")
    if destination == source or source in destination.parents:
        raise ValueError("output_root must not be the MUSAN root or a directory inside it")
    destination.parent.mkdir(parents=True, exist_ok=True)
    excluded = {Path(str(record["relative_path"])) for record in records if record["matched"]}

    def ignore(directory: str, names: list[str]) -> set[str]:
        relative_dir = Path(directory).resolve().relative_to(source)
        return {name for name in names if relative_dir / name in excluded}

    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    staged = temporary / destination.name
    try:
        shutil.copytree(source, staged, symlinks=True, copy_function=shutil.copy2, ignore=ignore)
        staged.replace(destination)
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


def write_filter_report(
    report_dir: Path,
    records: Sequence[Mapping],
    *,
    summary_metadata: Mapping,
) -> dict:
    if report_dir.exists():
        raise FileExistsError(f"Report path already exists: {report_dir}")
    report_dir.mkdir(parents=True)
    results_path = report_dir / "results.jsonl"
    with results_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    matched = [record for record in records if record["matched"]]
    subsets: dict[str, Counter] = defaultdict(Counter)
    for record in records:
        subsets[str(record["subset"])]["total"] += 1
        subsets[str(record["subset"])]["removed" if record["matched"] else "kept"] += 1
    summary = {
        **dict(summary_metadata),
        "total_audio_files": len(records),
        "kept_audio_files": len(records) - len(matched),
        "removed_audio_files": len(matched),
        "keyword_hits": dict(Counter(str(record["matched_keyword"]) for record in matched)),
        "subsets": {name: dict(counts) for name, counts in sorted(subsets.items())},
        "results": str(results_path.resolve()),
    }
    with (report_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    return summary


def filter_musan(
    *,
    musan_root: str | Path,
    output_root: str | Path,
    files: WenetModelFiles,
    keywords: Sequence[str],
    report_dir: str | Path | None = None,
    wenet_root: str | Path | None = None,
    threshold: float = 85.0,
    window_sec: float = 30.0,
    overlap_sec: float = 1.0,
    device: str = "cpu",
    batch_size: int = 8,
    beam_size: int = 10,
    mode: str = "attention_rescoring",
) -> dict:
    source = Path(musan_root).expanduser().resolve()
    destination = Path(output_root).expanduser().resolve()
    report = (
        Path(report_dir).expanduser().resolve()
        if report_dir is not None
        else destination.parent / f"{destination.name}_asr_filter_report"
    )
    if not source.is_dir():
        raise NotADirectoryError(f"MUSAN root not found: {source}")
    if destination.exists():
        raise FileExistsError(f"Output path already exists: {destination}")
    if destination == source or source in destination.parents:
        raise ValueError("output_root must not be the MUSAN root or a directory inside it")
    if report.exists():
        raise FileExistsError(f"Report path already exists: {report}")
    if report == source or source in report.parents:
        raise ValueError("report_dir must not be the MUSAN root or a directory inside it")
    if report == destination or destination in report.parents:
        raise ValueError("report_dir must be separate from output_root")
    keyword_values = validate_keywords(keywords)
    validate_filter_options(
        threshold=threshold,
        window_sec=window_sec,
        overlap_sec=overlap_sec,
        batch_size=batch_size,
        beam_size=beam_size,
        mode=mode,
    )
    audio_files = iter_audio_files(source)
    if not audio_files:
        raise ValueError(f"No audio files found under MUSAN root: {source}")

    with tempfile.TemporaryDirectory(prefix="musan_wenet_filter_") as temporary:
        workspace = Path(temporary)
        chunks = create_audio_chunks(
            audio_files, workspace / "chunks", window_sec=window_sec, overlap_sec=overlap_sec
        )
        transcripts = run_wenet_recognize(
            chunks,
            files,
            workspace,
            wenet_root=wenet_root,
            device=device,
            batch_size=batch_size,
            beam_size=beam_size,
            mode=mode,
        )
        records = build_filter_records(
            audio_files,
            chunks,
            transcripts,
            keyword_values,
            threshold=threshold,
            musan_root=source,
        )

    copy_filtered_tree(source, destination, records)
    return write_filter_report(
        report,
        records,
        summary_metadata={
            "musan_root": str(source),
            "output_root": str(destination),
            "report_dir": str(report),
            "keywords": keyword_values,
            "threshold": threshold,
            "window_sec": window_sec,
            "overlap_sec": overlap_sec,
            "device": device,
            "batch_size": batch_size,
            "beam_size": beam_size,
            "mode": mode,
            "model_files": {name: str(path) for name, path in asdict(files).items()},
        },
    )
