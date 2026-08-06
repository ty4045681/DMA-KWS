"""Prepare the Chinese-accent English corpora used by the DMA-KWS pipeline.

The local dataset tree and the path written to manifests are deliberately
separate.  This lets a tree prepared on macOS be copied to a fixed Linux path
without rewriting any manifest rows.
"""

from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import csv
import hashlib
import json
import logging
import math
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import tempfile
from typing import Any, Iterable, Mapping, Sequence
import wave
import zipfile

from dma_kws.g2p import HEY_EVA_PHONEMES, text_to_phonemes
from dma_kws.phonemes import normalize_english_text
from dma_kws.tokenizer import unsupported_phones


logger = logging.getLogger(__name__)


DEFAULT_REMOTE_DATASET_ROOT = PurePosixPath(
    "/home/q00931063/DMA-KWS/data/dma-kws/chinese_accent_english_datasets"
)
DEFAULT_SPLIT_SEED = 20260806

SPEECHOCEAN_LICENSE = "CC BY 4.0"
L2_ARCTIC_LICENSE = "CC BY-NC 4.0"
EDACC_LICENSE = "CC BY-SA 4.0"
REAL_RECORDINGS_LICENSE = "User-collected non-commercial research data"

L2_MANDARIN_SPLITS: Mapping[str, str] = {
    "BWC": "train",
    "LXC": "train",
    "NCC": "dev",
    "TXHC": "test",
}
L2_MANDARIN_GENDERS: Mapping[str, str] = {
    "BWC": "male",
    "LXC": "female",
    "NCC": "female",
    "TXHC": "male",
}
EDACC_MANDARIN_SPLITS: Mapping[str, str] = {
    "EDACC-C16-A": "dev",
    "EDACC-C16-B": "dev",
    "EDACC-C19-A": "test",
    "EDACC-C42-A": "test",
    "EDACC-C04-B": "control",
}

_EDACC_REJECT_MARKERS = (
    "IGNORE_TIME_SEGMENT_IN_SCORING",
    "<OVERLAP>",
    "<DTMF>",
    "<NO-SPEECH>",
)
_EDACC_IGNORABLE_TAG_RE = re.compile(r"<LAUGH>", flags=re.IGNORECASE)
_REAL_SPEAKER_SUFFIX_RE = re.compile(r"\d+$")
HEY_EVA_PHONEMES_G2P = " ".join(HEY_EVA_PHONEMES)

TRAIN_MIX_RATIOS: Mapping[str, float] = {
    "librispeech": 0.60,
    "speechocean762": 0.25,
    "l2_arctic_mandarin": 0.10,
    "hey_eva_real_pc": 0.05,
}
DEV_SELECT_RATIOS: Mapping[str, float] = {
    "librispeech": 0.40,
    "speechocean762": 0.30,
    "l2_arctic_mandarin": 0.20,
    "hey_eva_real_pc": 0.10,
}


@dataclass(frozen=True)
class ManifestPathMapper:
    """Map a local dataset path to the corresponding remote absolute path."""

    local_root: Path
    manifest_root: PurePosixPath | str = DEFAULT_REMOTE_DATASET_ROOT

    def __post_init__(self) -> None:
        local_root = Path(self.local_root).expanduser().resolve()
        manifest_root = PurePosixPath(str(self.manifest_root))
        if not local_root.is_absolute():
            raise ValueError(f"local_root must be absolute: {self.local_root}")
        if not manifest_root.is_absolute():
            raise ValueError(f"manifest_root must be an absolute POSIX path: {manifest_root}")
        object.__setattr__(self, "local_root", local_root)
        object.__setattr__(self, "manifest_root", manifest_root)

    def relative_path(self, local_path: Path | str) -> PurePosixPath:
        resolved = Path(local_path).expanduser().resolve()
        try:
            relative = resolved.relative_to(self.local_root)
        except ValueError as exc:
            raise ValueError(
                f"Audio path {resolved} is outside local dataset root {self.local_root}; "
                "it cannot be mapped safely after copying the dataset tree"
            ) from exc
        return PurePosixPath(*relative.parts)

    def manifest_path(self, local_path: Path | str) -> str:
        """Return the absolute Linux path to write into a task manifest."""
        return str(self.manifest_root / self.relative_path(local_path))


def _stable_key(value: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode("utf-8")).hexdigest()


def _apportion_dev_counts(group_sizes: Mapping[tuple[str, str], int], total: int) -> dict:
    """Allocate exactly ``total`` dev speakers across demographic strata."""
    population = sum(group_sizes.values())
    if total < 0 or total > population:
        raise ValueError(f"dev speaker count {total} is outside 0..{population}")
    if population == 0:
        return {}

    raw = {group: total * size / population for group, size in group_sizes.items()}
    allocated = {group: min(size, int(raw[group])) for group, size in group_sizes.items()}
    remaining = total - sum(allocated.values())
    order = sorted(
        group_sizes,
        key=lambda group: (raw[group] - int(raw[group]), group_sizes[group], group),
        reverse=True,
    )
    while remaining:
        progressed = False
        for group in order:
            if allocated[group] < group_sizes[group]:
                allocated[group] += 1
                remaining -= 1
                progressed = True
                if remaining == 0:
                    break
        if not progressed:  # Defensive: the initial range check should make this impossible.
            raise RuntimeError("Could not apportion SpeechOcean dev speakers")
    return allocated


def stratified_speechocean_dev_speakers(
    speakers: Iterable[str],
    *,
    ages: Mapping[str, str],
    genders: Mapping[str, str],
    dev_speaker_count: int = 25,
    seed: int = DEFAULT_SPLIT_SEED,
) -> set[str]:
    """Choose a deterministic age/gender-stratified speaker-level dev set."""
    strata: dict[tuple[str, str], list[str]] = defaultdict(list)
    for speaker in sorted(set(speakers)):
        raw_age = str(ages.get(speaker, "unknown")).strip()
        try:
            age_group = "child" if int(raw_age) < 18 else "adult"
        except ValueError:
            age_group = "unknown"
        gender = str(genders.get(speaker, "unknown")).strip().lower() or "unknown"
        strata[(age_group, gender)].append(speaker)

    allocation = _apportion_dev_counts(
        {group: len(group_speakers) for group, group_speakers in strata.items()},
        dev_speaker_count,
    )
    selected: set[str] = set()
    for group, group_speakers in strata.items():
        ordered = sorted(group_speakers, key=lambda speaker: _stable_key(speaker, seed))
        selected.update(ordered[: allocation[group]])
    return selected


def _read_kaldi_map(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            parts = stripped.split(maxsplit=1)
            if len(parts) != 2:
                raise ValueError(f"{path}:{line_number}: expected '<key> <value>'")
            key, value = parts
            if key in values:
                raise ValueError(f"{path}:{line_number}: duplicate key {key!r}")
            values[key] = value
    return values


def _ctc_fields(g2p, text: str, *, context: str) -> dict[str, Any]:
    normalized = normalize_english_text(text)
    if not normalized:
        raise ValueError(f"{context}: transcript is empty after normalization")
    phones = text_to_phonemes(g2p, normalized)
    if not phones:
        raise ValueError(f"{context}: G2P produced no phonemes")
    unsupported = unsupported_phones(phones)
    if unsupported:
        raise ValueError(f"{context}: unsupported phonemes: {', '.join(unsupported)}")
    return {
        "text": text.strip(),
        "normalized_text": normalized,
        "phonemes": phones,
        "phonemes_g2p": " ".join(phones),
    }


def _common_record(
    *,
    mapper: ManifestPathMapper,
    local_audio_path: Path,
    g2p,
    text: str,
    utt_id: str,
    source: str,
    source_utt_id: str,
    speaker_id: str,
    split: str,
    license_name: str,
    context: str,
    metadata: Mapping[str, Any] | None = None,
    verify_audio: bool = True,
) -> dict[str, Any]:
    if verify_audio and not local_audio_path.is_file():
        raise FileNotFoundError(f"{context}: audio file not found: {local_audio_path}")
    relative_path = mapper.relative_path(local_audio_path)
    record: dict[str, Any] = {
        "utt_id": utt_id,
        "source": source,
        "source_utt_id": source_utt_id,
        "speaker_id": speaker_id,
        "split": split,
        "wav_path": mapper.manifest_path(local_audio_path),
        "dataset_relative_path": str(relative_path),
        "license": license_name,
    }
    record.update(_ctc_fields(g2p, text, context=context))
    if metadata:
        record.update(metadata)
    return record


def prepare_speechocean_records(
    root: Path,
    *,
    mapper: ManifestPathMapper,
    g2p,
    dev_speaker_count: int = 25,
    seed: int = DEFAULT_SPLIT_SEED,
    verify_audio: bool = True,
) -> list[dict[str, Any]]:
    """Parse only the utterances referenced by official SpeechOcean manifests."""
    root = Path(root).resolve()
    train_utt2spk = _read_kaldi_map(root / "train" / "utt2spk")
    ages = _read_kaldi_map(root / "train" / "spk2age")
    genders = _read_kaldi_map(root / "train" / "spk2gender")
    dev_speakers = stratified_speechocean_dev_speakers(
        train_utt2spk.values(),
        ages=ages,
        genders=genders,
        dev_speaker_count=dev_speaker_count,
        seed=seed,
    )

    records: list[dict[str, Any]] = []
    for official_split in ("train", "test"):
        split_dir = root / official_split
        wavs = _read_kaldi_map(split_dir / "wav.scp")
        texts = _read_kaldi_map(split_dir / "text")
        utt2spk = _read_kaldi_map(split_dir / "utt2spk")
        split_ages = _read_kaldi_map(split_dir / "spk2age")
        split_genders = _read_kaldi_map(split_dir / "spk2gender")
        key_sets = {"wav.scp": set(wavs), "text": set(texts), "utt2spk": set(utt2spk)}
        if len({frozenset(keys) for keys in key_sets.values()}) != 1:
            counts = ", ".join(f"{name}={len(keys)}" for name, keys in key_sets.items())
            raise ValueError(f"{split_dir}: utterance keys do not match ({counts})")

        for source_utt_id in sorted(wavs):
            wav_spec = wavs[source_utt_id]
            if wav_spec.endswith("|"):
                raise ValueError(
                    f"{split_dir / 'wav.scp'}: command-style wav.scp entries are unsupported"
                )
            speaker = utt2spk[source_utt_id]
            task_split = (
                "test"
                if official_split == "test"
                else ("dev" if speaker in dev_speakers else "train")
            )
            local_audio_path = root / wav_spec
            records.append(
                _common_record(
                    mapper=mapper,
                    local_audio_path=local_audio_path,
                    g2p=g2p,
                    text=texts[source_utt_id],
                    utt_id=f"speechocean762-{source_utt_id}",
                    source="speechocean762",
                    source_utt_id=source_utt_id,
                    speaker_id=f"speechocean762-{speaker}",
                    split=task_split,
                    license_name=SPEECHOCEAN_LICENSE,
                    context=f"SpeechOcean utterance {source_utt_id}",
                    metadata={
                        "source_split": official_split,
                        "accent": "Mandarin L1 English",
                        "age": int(split_ages[speaker]),
                        "gender": split_genders[speaker],
                    },
                    verify_audio=verify_audio,
                )
            )
    return records


def prepare_l2_arctic_records(
    root: Path,
    *,
    mapper: ManifestPathMapper,
    g2p,
    audio_root: Path | None = None,
    verify_audio: bool = True,
) -> list[dict[str, Any]]:
    """Build CTC records from L2-ARCTIC orthographic transcripts.

    TextGrid annotations are intentionally not read: the adapter target is the
    canonical transcript converted by the same project G2P used at inference.
    """
    root = Path(root).resolve()
    audio_root = Path(audio_root or root).resolve()
    records: list[dict[str, Any]] = []
    for speaker, split in L2_MANDARIN_SPLITS.items():
        transcript_dir = root / speaker / "transcript"
        if not transcript_dir.is_dir():
            raise FileNotFoundError(f"L2-ARCTIC transcript directory not found: {transcript_dir}")
        for transcript_path in sorted(transcript_dir.glob("*.txt")):
            source_utt_id = transcript_path.stem
            if audio_root == root:
                local_audio_path = root / speaker / "wav" / f"{source_utt_id}.wav"
            else:
                local_audio_path = audio_root / speaker / f"{source_utt_id}.wav"
            text = transcript_path.read_text(encoding="utf-8").strip()
            records.append(
                _common_record(
                    mapper=mapper,
                    local_audio_path=local_audio_path,
                    g2p=g2p,
                    text=text,
                    utt_id=f"l2_arctic-{speaker}-{source_utt_id}",
                    source="l2_arctic_mandarin",
                    source_utt_id=source_utt_id,
                    speaker_id=f"l2_arctic-{speaker}",
                    split=split,
                    license_name=L2_ARCTIC_LICENSE,
                    context=f"L2-ARCTIC {speaker}/{source_utt_id}",
                    metadata={
                        "source_split": "scripted",
                        "accent": "Mandarin L1 English",
                        "gender": L2_MANDARIN_GENDERS[speaker],
                    },
                    verify_audio=verify_audio,
                )
            )
    return records


def _validate_pcm16_mono_16k(path: Path) -> None:
    try:
        with wave.open(str(path), "rb") as wav:
            properties = (wav.getframerate(), wav.getnchannels(), wav.getsampwidth())
    except (EOFError, wave.Error) as exc:
        raise ValueError(f"Invalid derived WAV: {path}") from exc
    if properties != (16000, 1, 2):
        raise ValueError(
            f"Derived WAV must be 16 kHz mono PCM16, got "
            f"sample_rate={properties[0]}, channels={properties[1]}, "
            f"sample_width={properties[2]}: {path}"
        )


def _normalize_one_l2_wav(
    source_path: Path,
    destination_path: Path,
    *,
    ffmpeg_bin: str,
) -> str:
    if destination_path.is_file():
        _validate_pcm16_mono_16k(destination_path)
        return "skipped"
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = destination_path.with_name(f".{destination_path.name}.tmp.wav")
    if temporary_path.exists():
        temporary_path.unlink()
    command = [
        ffmpeg_bin,
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(source_path),
        "-ac",
        "1",
        "-ar",
        "16000",
        "-sample_fmt",
        "s16",
        "-c:a",
        "pcm_s16le",
        str(temporary_path),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode:
        if temporary_path.exists():
            temporary_path.unlink()
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"ffmpeg failed for {source_path}: {detail}")
    _validate_pcm16_mono_16k(temporary_path)
    temporary_path.replace(destination_path)
    return "written"


def normalize_l2_arctic_audio(
    raw_root: Path,
    output_root: Path,
    *,
    ffmpeg_bin: str = "ffmpeg",
    num_workers: int = 4,
) -> dict[str, int]:
    """Materialize the four Mandarin speakers as 16 kHz mono PCM16 WAVs."""
    raw_root = Path(raw_root).resolve()
    output_root = Path(output_root).resolve()
    if num_workers < 1:
        raise ValueError(f"num_workers must be >= 1, got {num_workers}")
    executable = shutil.which(ffmpeg_bin)
    if executable is None:
        raise FileNotFoundError(
            f"ffmpeg executable not found: {ffmpeg_bin!r}; install ffmpeg or pass --ffmpeg-bin"
        )

    tasks: list[tuple[Path, Path]] = []
    for speaker in L2_MANDARIN_SPLITS:
        transcript_dir = raw_root / speaker / "transcript"
        if not transcript_dir.is_dir():
            raise FileNotFoundError(f"L2-ARCTIC transcript directory not found: {transcript_dir}")
        for transcript_path in sorted(transcript_dir.glob("*.txt")):
            source_path = raw_root / speaker / "wav" / f"{transcript_path.stem}.wav"
            if not source_path.is_file():
                raise FileNotFoundError(f"L2-ARCTIC audio file not found: {source_path}")
            tasks.append((source_path, output_root / speaker / f"{transcript_path.stem}.wav"))

    results: dict[str, int] = defaultdict(int)

    def convert(task: tuple[Path, Path]) -> str:
        return _normalize_one_l2_wav(task[0], task[1], ffmpeg_bin=executable)

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        for status in executor.map(convert, tasks):
            results[status] += 1
    return {"total": len(tasks), "written": results["written"], "skipped": results["skipped"]}


def _clean_edacc_text(text: str) -> tuple[str | None, str | None]:
    upper = text.upper()
    for marker in _EDACC_REJECT_MARKERS:
        if marker in upper:
            return None, marker
    cleaned = _EDACC_IGNORABLE_TAG_RE.sub(" ", text)
    cleaned = " ".join(cleaned.split())
    if not normalize_english_text(cleaned):
        return None, "empty_after_cleaning"
    return cleaned, None


def normalize_edacc_source_manifest_paths(
    root: Path,
    *,
    mapper: ManifestPathMapper,
) -> dict[str, int]:
    """Replace stale extraction-host paths in the portable EdAcc source indexes."""

    root = Path(root).resolve()
    counts: dict[str, int] = {}
    jsonl_path = root / "manifest.jsonl"
    if jsonl_path.is_file():
        rows = _read_jsonl_records(jsonl_path)
        for row_number, row in enumerate(rows, start=1):
            relative = str(row.get("relative_audio_path", "")).strip()
            if not relative:
                raise ValueError(f"{jsonl_path}:{row_number}: missing relative_audio_path")
            row["audio_path"] = mapper.manifest_path(root / relative)
        temporary = jsonl_path.with_suffix(".jsonl.tmp")
        _write_jsonl(temporary, rows)
        temporary.replace(jsonl_path)
        counts[jsonl_path.name] = len(rows)

    csv_path = root / "manifest.csv"
    if csv_path.is_file():
        with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = list(reader.fieldnames or [])
            rows = [dict(row) for row in reader]
        if "audio_path" not in fieldnames or "relative_audio_path" not in fieldnames:
            raise ValueError(f"{csv_path}: requires audio_path and relative_audio_path columns")
        for row_number, row in enumerate(rows, start=2):
            relative = str(row.get("relative_audio_path", "")).strip()
            if not relative:
                raise ValueError(f"{csv_path}:{row_number}: missing relative_audio_path")
            row["audio_path"] = mapper.manifest_path(root / relative)
        temporary = csv_path.with_suffix(".csv.tmp")
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(csv_path)
        counts[csv_path.name] = len(rows)
    return counts


def prepare_edacc_records(
    root: Path,
    *,
    mapper: ManifestPathMapper,
    g2p,
    verify_audio: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Prepare fixed speaker-disjoint EdAcc evaluation and quarantine records."""
    root = Path(root).resolve()
    manifest_path = root / "manifest.jsonl"
    records: list[dict[str, Any]] = []
    quarantine: list[dict[str, Any]] = []
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            source = json.loads(line)
            speaker = str(source["speaker_id"])
            split = EDACC_MANDARIN_SPLITS.get(speaker)
            if split is None:
                raise ValueError(
                    f"{manifest_path}:{line_number}: unexpected Mandarin speaker {speaker!r}"
                )
            relative_audio = source.get("relative_audio_path")
            if not relative_audio:
                raise ValueError(
                    f"{manifest_path}:{line_number}: missing relative_audio_path; "
                    "the stale local absolute audio_path is deliberately not used"
                )
            local_audio_path = root / str(relative_audio)
            source_utt_id = Path(str(relative_audio)).stem
            cleaned_text, reject_reason = _clean_edacc_text(str(source.get("text", "")))
            if reject_reason:
                if verify_audio and not local_audio_path.is_file():
                    raise FileNotFoundError(
                        f"EdAcc row {line_number}: audio file not found: {local_audio_path}"
                    )
                quarantine.append(
                    {
                        "utt_id": f"edacc-{speaker}-{source_utt_id}",
                        "source": "edacc_mandarin",
                        "source_utt_id": source_utt_id,
                        "speaker_id": speaker,
                        "split": split,
                        "wav_path": mapper.manifest_path(local_audio_path),
                        "dataset_relative_path": str(mapper.relative_path(local_audio_path)),
                        "text": str(source.get("text", "")),
                        "reason": reject_reason,
                        "license": EDACC_LICENSE,
                    }
                )
                continue
            assert cleaned_text is not None
            records.append(
                _common_record(
                    mapper=mapper,
                    local_audio_path=local_audio_path,
                    g2p=g2p,
                    text=cleaned_text,
                    utt_id=f"edacc-{speaker}-{source_utt_id}",
                    source="edacc_mandarin",
                    source_utt_id=source_utt_id,
                    speaker_id=speaker,
                    split=split,
                    license_name=EDACC_LICENSE,
                    context=f"EdAcc row {line_number} ({speaker}/{source_utt_id})",
                    metadata={
                        "source_split": source.get("source_split"),
                        "accent": source.get("accent", "Chinese"),
                        "raw_accent": source.get("raw_accent"),
                        "l1": source.get("l1", "Mandarin"),
                        "gender": source.get("gender"),
                        "duration_seconds": source.get("duration_seconds"),
                        "sample_rate": source.get("sample_rate"),
                    },
                    verify_audio=verify_audio,
                )
            )
    return records, quarantine


def normalize_real_speaker_name(name: str) -> str:
    """Merge aliases whose only difference is a trailing numeric suffix."""
    return _REAL_SPEAKER_SUFFIX_RE.sub("", str(name).strip()).strip().casefold()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _copy_file_idempotently(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.is_file() and source.stat().st_size == destination.stat().st_size:
            if _sha256_file(source) == _sha256_file(destination):
                return
        raise FileExistsError(
            f"Refusing to overwrite a different anonymized recording: {destination}"
        )
    shutil.copy2(source, destination)


def _real_recording_source_index(recordings_root: Path) -> dict[str, Path]:
    by_group: dict[str, Path] = {}
    for wav_path in sorted(recordings_root.rglob("*.wav")):
        group_id = wav_path.parent.name
        if group_id in by_group:
            raise ValueError(
                f"Multiple WAV files use recording group id {group_id!r}: "
                f"{by_group[group_id]} and {wav_path}"
            )
        by_group[group_id] = wav_path
    return by_group


def _real_speaker_splits(
    grouped_rows: Mapping[str, list[dict[str, str]]],
    *,
    seed: int,
) -> tuple[dict[str, str], set[str]]:
    """Apply the approved alias/train and 8/2/2 gender-stratified split."""
    split_by_speaker: dict[str, str] = {}
    quarantined: set[str] = set()
    complete: list[str] = []
    alias_groups: list[str] = []

    for speaker, rows in grouped_rows.items():
        labels = [row["类别"].strip() for row in rows]
        raw_names = {row["姓名"].strip() for row in rows}
        positive_count = labels.count("positive")
        negative_count = labels.count("near_negative")
        if len(raw_names) > 1:
            if len(rows) != 40 or positive_count != 20 or negative_count != 20:
                quarantined.add(speaker)
            else:
                alias_groups.append(speaker)
            continue
        if len(rows) == 20 and positive_count == 10 and negative_count == 10:
            complete.append(speaker)
        else:
            quarantined.add(speaker)

    if len(alias_groups) != 1:
        raise ValueError(
            f"Expected one complete 40-row alias group, found {len(alias_groups)}"
        )
    if len(complete) != 12:
        raise ValueError(f"Expected 12 complete 20-row speakers, found {len(complete)}")
    if sum(len(grouped_rows[speaker]) for speaker in quarantined) != 1:
        raise ValueError(
            "Expected exactly one row from incomplete speakers, found "
            f"{sum(len(grouped_rows[speaker]) for speaker in quarantined)}"
        )

    split_by_speaker[alias_groups[0]] = "train"
    by_gender: dict[str, list[str]] = defaultdict(list)
    for speaker in complete:
        genders = {row["性别"].strip().casefold() for row in grouped_rows[speaker]}
        if len(genders) != 1:
            raise ValueError(f"Real speaker {speaker!r} has inconsistent gender values: {genders}")
        by_gender[next(iter(genders))].append(speaker)
    if {gender: len(speakers) for gender, speakers in by_gender.items()} != {
        "female": 4,
        "male": 8,
    }:
        raise ValueError(
            "Expected 4 female and 8 male complete real speakers, got "
            f"{dict((gender, len(speakers)) for gender, speakers in by_gender.items())}"
        )

    # Hold out one female and one male for each evaluation split.  All remaining
    # complete speakers train, giving 8/2/2; the 40-row alias group is train-only.
    for gender, speakers in sorted(by_gender.items()):
        ordered = sorted(speakers, key=lambda speaker: _stable_key(speaker, seed))
        split_by_speaker[ordered[0]] = "test"
        split_by_speaker[ordered[1]] = "dev"
        for speaker in ordered[2:]:
            split_by_speaker[speaker] = "train"
    return split_by_speaker, quarantined


def prepare_real_recording_records(
    recordings_root: Path,
    *,
    destination_root: Path,
    mapper: ManifestPathMapper,
    g2p,
    seed: int = DEFAULT_SPLIT_SEED,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Anonymize/copy the 281 PC recordings and apply the fixed speaker split."""
    recordings_root = Path(recordings_root).expanduser().resolve()
    destination_root = Path(destination_root).expanduser().resolve()
    csv_path = recordings_root / "recording_info.csv"
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    required = {"姓名", "性别", "类别", "唤醒词", "录音组ID"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"{csv_path} must contain columns: {sorted(required)}")
    if len(rows) != 281:
        raise ValueError(f"Expected 281 real recording rows, found {len(rows)}")

    grouped_rows: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        normalized_name = normalize_real_speaker_name(row["姓名"])
        if not normalized_name:
            raise ValueError("Real recording row has an empty normalized speaker name")
        grouped_rows[normalized_name].append(row)
    split_by_speaker, quarantined_speakers = _real_speaker_splits(grouped_rows, seed=seed)
    source_index = _real_recording_source_index(recordings_root)
    if len(source_index) != len(rows):
        raise ValueError(
            f"Expected one source WAV per CSV row, found wavs={len(source_index)}, rows={len(rows)}"
        )

    # Hash ordering makes IDs deterministic without encoding the source name in
    # the copied tree or any output manifest.
    ordered_speakers = sorted(grouped_rows, key=lambda speaker: _stable_key(speaker, seed))
    aliases = {
        speaker: f"real_pc_spk_{index:03d}"
        for index, speaker in enumerate(ordered_speakers, start=1)
    }
    records: list[dict[str, Any]] = []
    quarantine: list[dict[str, Any]] = []
    for speaker in ordered_speakers:
        alias = aliases[speaker]
        split = "quarantine" if speaker in quarantined_speakers else split_by_speaker[speaker]
        speaker_rows = sorted(
            grouped_rows[speaker],
            key=lambda row: (row["录音组ID"], row["类别"], row["唤醒词"]),
        )
        for utterance_index, row in enumerate(speaker_rows, start=1):
            group_id = row["录音组ID"].strip()
            source_audio_path = source_index.get(group_id)
            if source_audio_path is None:
                raise FileNotFoundError(f"No WAV found for real recording group {group_id!r}")
            raw_label = row["类别"].strip()
            if raw_label not in {"positive", "near_negative"}:
                raise ValueError(f"Unexpected real recording category: {raw_label!r}")
            label = 1 if raw_label == "positive" else 0
            label_dir = "positive" if label else "near_negative"
            anonymous_utt = f"{alias}_{utterance_index:03d}"
            destination_path = (
                destination_root / split / alias / label_dir / f"{anonymous_utt}.wav"
            )
            _copy_file_idempotently(source_audio_path, destination_path)
            record = _common_record(
                mapper=mapper,
                local_audio_path=destination_path,
                g2p=g2p,
                text=row["唤醒词"],
                utt_id=f"hey_eva_real_pc-{anonymous_utt}",
                source="hey_eva_real_pc",
                source_utt_id=anonymous_utt,
                speaker_id=alias,
                split=split,
                license_name=REAL_RECORDINGS_LICENSE,
                context=f"Anonymized real recording {anonymous_utt}",
                metadata={
                    "accent": "Mandarin L1 English",
                    "device": "pc",
                    "gender": row["性别"].strip().casefold(),
                    "age_group": row.get("年龄段", "").strip(),
                    "environment": row.get("录音环境", "").strip(),
                    "label": label,
                    "negative_type": "near_phrase" if label == 0 else "",
                },
            )
            if split == "quarantine":
                record["reason"] = "incomplete_speaker_set"
                quarantine.append(record)
            else:
                records.append(record)

    split_counts = defaultdict(int)
    for record in records:
        split_counts[str(record["split"])] += 1
    expected = {"train": 200, "dev": 40, "test": 40}
    if dict(split_counts) != expected or len(quarantine) != 1:
        raise ValueError(
            f"Unexpected real recording split counts: {dict(split_counts)}, "
            f"quarantine={len(quarantine)}; expected {expected}, quarantine=1"
        )
    return records, quarantine


def _write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> int:
    materialized = list(records)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in materialized:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    return len(materialized)


def _write_csv(path: Path, records: Sequence[Mapping[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "audio_path",
        "text",
        "keyword",
        "keyword_phonemes",
        "text_variant_phonemes",
        "label",
        "phase",
        "split",
        "speaker_id",
        "device",
        "source",
        "negative_type",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({field: record.get(field, "") for field in fieldnames})
    return len(records)


def write_real_lora_views(view_root: Path, records: Sequence[dict[str, Any]]) -> dict[str, int]:
    """Write LoRA CSV views while keeping the blind speaker split isolated."""
    view_root = Path(view_root).resolve()
    mapping = {"train": "real_train.csv", "dev": "real_eval.csv", "test": "real_blind_test.csv"}
    counts: dict[str, int] = {}
    rows_by_source_split: dict[str, list[dict[str, Any]]] = {}
    for source_split, filename in mapping.items():
        output_split = "train" if source_split == "train" else (
            "eval" if source_split == "dev" else "blind_test"
        )
        rows = [
            {
                "audio_path": record["wav_path"],
                "text": record["text"],
                "keyword": "hey eva",
                "keyword_phonemes": HEY_EVA_PHONEMES_G2P,
                "text_variant_phonemes": record["phonemes_g2p"],
                "label": record["label"],
                "phase": "real",
                "split": output_split,
                "speaker_id": record["speaker_id"],
                "device": record["device"],
                "source": record["source"],
                "negative_type": record.get("negative_type", ""),
            }
            for record in records
            if record["split"] == source_split
        ]
        rows_by_source_split[source_split] = rows
        relative = f"views/hey_eva_adapt/manifests/{filename}"
        counts[relative] = _write_csv(view_root / "manifests" / filename, rows)
    # This is the explicit-split source manifest accepted by
    # prepare_keyword_adaptation; blind-test rows are intentionally absent.
    source_rows = rows_by_source_split["train"] + rows_by_source_split["dev"]
    counts["views/hey_eva_adapt/manifests/real_source.csv"] = _write_csv(
        view_root / "manifests" / "real_source.csv", source_rows
    )
    return counts


def hard_negative_candidate_records(
    records: Sequence[dict[str, Any]],
    *,
    include_real: bool = True,
) -> list[dict[str, Any]]:
    """Return a broad train-only pool for later model-score hard-negative mining."""
    allowed_sources = {"speechocean762", "l2_arctic_mandarin"}
    candidates: list[dict[str, Any]] = []
    for record in records:
        if record["split"] != "train":
            continue
        is_generic = record["source"] in allowed_sources
        is_real_negative = (
            include_real
            and record["source"] == "hey_eva_real_pc"
            and int(record.get("label", 1)) == 0
        )
        if not (is_generic or is_real_negative):
            continue
        # Never write an actual keyword occurrence into a pool whose ground
        # truth is fixed to zero, even if a future source revision adds one.
        if normalize_english_text(str(record["text"])) == "hey eva":
            continue
        candidate = {
            "audio_path": record["wav_path"],
            "keyword": "hey eva",
            "keyword_phonemes": HEY_EVA_PHONEMES_G2P,
            "label": 0,
            "text_variant": record["text"],
            "source": record["source"],
            "speaker_id": record["speaker_id"],
            "split": "train",
            "utt_id": record["utt_id"],
        }
        variant_phonemes = record.get("phonemes_g2p")
        if variant_phonemes is None and isinstance(record.get("phonemes"), list):
            variant_phonemes = " ".join(str(phone) for phone in record["phonemes"])
        if variant_phonemes:
            candidate["text_variant_phonemes"] = str(variant_phonemes)
        candidates.append(candidate)
    return candidates


def _read_jsonl_records(path: Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _validate_librispeech_manifest(records: Sequence[dict[str, Any]], path: Path) -> None:
    if not records:
        raise ValueError(f"LibriSpeech manifest is empty: {path}")
    for index, record in enumerate(records):
        wav_path = PurePosixPath(str(record.get("wav_path", "")))
        if not wav_path.is_absolute():
            raise ValueError(
                f"{path}: row {index} LibriSpeech wav_path must already be an absolute "
                f"Linux path, got {wav_path}"
            )
        if wav_path.parts[:2] in (("/", "Users"), ("/", "private")):
            raise ValueError(
                f"{path}: row {index} uses a local macOS path rather than the remote Linux path: "
                f"{wav_path}"
            )
        phones = record.get("phonemes_g2p", record.get("phonemes"))
        if phones is None:
            raise ValueError(f"{path}: row {index} lacks phonemes_g2p/phonemes")
        phone_tokens = str(phones).split() if isinstance(phones, str) else list(phones)
        invalid = unsupported_phones(phone_tokens)
        if invalid:
            raise ValueError(
                f"{path}: row {index} has unsupported phonemes: {', '.join(invalid)}"
            )


def _validate_librispeech_split_disjointness(
    train_records: Sequence[dict[str, Any]],
    dev_records: Sequence[dict[str, Any]],
) -> None:
    for field in ("wav_path", "utt_id"):
        train_values = {
            str(record[field]).strip()
            for record in train_records
            if str(record.get(field, "")).strip()
        }
        dev_values = {
            str(record[field]).strip()
            for record in dev_records
            if str(record.get(field, "")).strip()
        }
        overlap = train_values & dev_values
        if overlap:
            raise ValueError(
                f"LibriSpeech train/dev manifests overlap by {field}: "
                f"{len(overlap)} duplicated values"
            )

    def speaker_id(record: Mapping[str, Any]) -> str | None:
        explicit = str(record.get("speaker_id", "")).strip()
        if explicit:
            return explicit
        # Canonical LibriSpeech ids are speaker-chapter-utterance.  Restrict the
        # inference to numeric ids so unrelated custom names are not conflated.
        match = re.match(r"^(\d+)-\d+-", str(record.get("utt_id", "")))
        return match.group(1) if match else None

    unresolved = [
        str(record.get("utt_id", record.get("wav_path", "<unknown>")))
        for record in (*train_records, *dev_records)
        if speaker_id(record) is None
    ]
    if unresolved:
        preview = ", ".join(unresolved[:5])
        raise ValueError(
            "LibriSpeech manifest rows must provide speaker_id or use canonical "
            f"numeric speaker-chapter-utterance utt_id; unresolved: {preview}"
        )

    train_speakers = {speaker_id(record) for record in train_records}
    dev_speakers = {speaker_id(record) for record in dev_records}
    speaker_overlap = train_speakers & dev_speakers
    if speaker_overlap:
        raise ValueError(
            "LibriSpeech train/dev manifests overlap by speaker_id: "
            f"{len(speaker_overlap)} duplicated speakers"
        )


def _ratio_counts(ratios: Mapping[str, float], total: int) -> dict[str, int]:
    if total <= 0:
        raise ValueError(f"Mixture size must be positive, got {total}")
    if abs(sum(ratios.values()) - 1.0) > 1e-8:
        raise ValueError(f"Mixture ratios must sum to 1.0: {ratios}")
    raw = {source: total * ratio for source, ratio in ratios.items()}
    counts = {source: int(value) for source, value in raw.items()}
    remaining = total - sum(counts.values())
    for source in sorted(ratios, key=lambda item: (raw[item] - counts[item], item), reverse=True):
        if remaining == 0:
            break
        counts[source] += 1
        remaining -= 1
    return counts


def _deterministic_sample(
    records: Sequence[dict[str, Any]],
    count: int,
    *,
    seed: int,
    source: str,
    partition: str = "",
) -> list[dict[str, Any]]:
    if count and not records:
        raise ValueError(f"Cannot sample {count} rows from empty source {source!r}")
    if count > len(records):
        logger.warning(
            "Oversampling source %r partition=%s: requested=%d available=%d "
            "mean_reuse=%.2fx max_reuse=%dx",
            source,
            partition or "unspecified",
            count,
            len(records),
            count / len(records),
            math.ceil(count / len(records)),
        )
    sampled: list[dict[str, Any]] = []
    cycle = 0
    while len(sampled) < count:
        ordered = sorted(
            records,
            key=lambda record: _stable_key(
                f"{source}:{cycle}:{record.get('utt_id', record.get('wav_path', ''))}", seed
            ),
        )
        take = min(count - len(sampled), len(ordered))
        sampled.extend(dict(record) for record in ordered[:take])
        cycle += 1
    for index, record in enumerate(sampled):
        record.setdefault("source", source)
        record["mixture_instance_id"] = f"{source}-{index:06d}"
    return sampled


def _write_mixture_recipe(
    path: Path,
    *,
    train_size: int,
    dev_size: int,
    status: str = "recipe_only_until_librispeech_manifests_are_supplied",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "status": status,
                "train_mix": {"size": train_size, "ratios": dict(TRAIN_MIX_RATIOS)},
                "dev_select": {"size": dev_size, "ratios": dict(DEV_SELECT_RATIOS)},
                "requirements": {
                    "librispeech_train_manifest": "absolute Linux wav_path required",
                    "librispeech_dev_manifest": "absolute Linux wav_path required",
                    "command": "scripts/prepare_chinese_accent_english.py --mixture-only "
                    "--source-root <REMOTE_DATASET_ROOT> --librispeech-train-manifest <TRAIN> "
                    "--librispeech-dev-manifest <DEV>",
                },
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def materialize_adapter_mixtures(
    *,
    manifest_dir: Path,
    librispeech_train_manifest: Path,
    librispeech_dev_manifest: Path,
    train_size: int = 50000,
    dev_size: int = 5000,
    seed: int = DEFAULT_SPLIT_SEED,
) -> dict[str, int]:
    """Materialize exact-ratio Stage-I manifests after LibriSpeech is available."""
    manifest_dir = Path(manifest_dir).resolve()
    adapter_dir = manifest_dir / "adapter"
    libri_train = _read_jsonl_records(librispeech_train_manifest)
    libri_dev = _read_jsonl_records(librispeech_dev_manifest)
    _validate_librispeech_manifest(libri_train, Path(librispeech_train_manifest))
    _validate_librispeech_manifest(libri_dev, Path(librispeech_dev_manifest))
    _validate_librispeech_split_disjointness(libri_train, libri_dev)
    for record in libri_train + libri_dev:
        record["source"] = "librispeech"

    source_paths = {
        "speechocean762": (
            adapter_dir / "speechocean762_train.jsonl",
            adapter_dir / "speechocean762_dev.jsonl",
        ),
        "l2_arctic_mandarin": (
            adapter_dir / "l2_arctic_mandarin_train.jsonl",
            adapter_dir / "l2_arctic_mandarin_dev.jsonl",
        ),
        "hey_eva_real_pc": (
            adapter_dir / "real_train.jsonl",
            adapter_dir / "real_dev.jsonl",
        ),
    }
    train_sources = {"librispeech": libri_train}
    dev_sources = {"librispeech": libri_dev}
    for source, (train_path, dev_path) in source_paths.items():
        if not train_path.is_file() or not dev_path.is_file():
            raise FileNotFoundError(
                f"Cannot build final mixtures before source manifests exist: "
                f"{train_path}, {dev_path}"
            )
        train_sources[source] = _read_jsonl_records(train_path)
        dev_sources[source] = _read_jsonl_records(dev_path)

    train_counts = _ratio_counts(TRAIN_MIX_RATIOS, train_size)
    dev_counts = _ratio_counts(DEV_SELECT_RATIOS, dev_size)
    train_mix: list[dict[str, Any]] = []
    dev_select: list[dict[str, Any]] = []
    for source, count in train_counts.items():
        train_mix.extend(
            _deterministic_sample(
                train_sources[source],
                count,
                seed=seed,
                source=source,
                partition="train",
            )
        )
    for source, count in dev_counts.items():
        dev_select.extend(
            _deterministic_sample(
                dev_sources[source],
                count,
                seed=seed + 1,
                source=source,
                partition="dev",
            )
        )
    train_mix.sort(key=lambda record: _stable_key(record["mixture_instance_id"], seed + 2))
    dev_select.sort(key=lambda record: _stable_key(record["mixture_instance_id"], seed + 3))
    counts = {
        "adapter/train_mix.jsonl": _write_jsonl(adapter_dir / "train_mix.jsonl", train_mix),
        "adapter/dev_select.jsonl": _write_jsonl(adapter_dir / "dev_select.jsonl", dev_select),
    }
    _write_mixture_recipe(
        adapter_dir / "mixture_recipe.json",
        train_size=train_size,
        dev_size=dev_size,
        status="materialized",
    )
    return counts


def _records_for(
    records: Sequence[dict[str, Any]], *, source: str, split: str
) -> list[dict[str, Any]]:
    return [record for record in records if record["source"] == source and record["split"] == split]


def _validate_records(records: Sequence[dict[str, Any]], manifest_root: PurePosixPath) -> dict:
    utt_ids: set[str] = set()
    audio_paths: set[str] = set()
    speaker_splits: dict[tuple[str, str], set[str]] = defaultdict(set)
    for record in records:
        utt_id = str(record["utt_id"])
        if utt_id in utt_ids:
            raise ValueError(f"Duplicate utterance id: {utt_id}")
        utt_ids.add(utt_id)
        wav_path = str(record["wav_path"])
        if not PurePosixPath(wav_path).is_absolute():
            raise ValueError(f"Manifest audio path is not absolute: {wav_path}")
        try:
            PurePosixPath(wav_path).relative_to(manifest_root)
        except ValueError as exc:
            raise ValueError(
                f"Manifest audio path {wav_path} does not use remote root {manifest_root}"
            ) from exc
        if wav_path in audio_paths:
            raise ValueError(f"Duplicate audio path: {wav_path}")
        audio_paths.add(wav_path)
        speaker_splits[(str(record["source"]), str(record["speaker_id"]))].add(
            str(record["split"])
        )

    overlaps = {
        f"{source}:{speaker}": sorted(splits)
        for (source, speaker), splits in speaker_splits.items()
        if len(splits) != 1
    }
    if overlaps:
        raise ValueError(f"Speakers occur in multiple splits: {overlaps}")
    return {
        "records": len(records),
        "unique_utterances": len(utt_ids),
        "unique_audio_paths": len(audio_paths),
        "speaker_split_overlap": 0,
    }


def _write_split_assignments(path: Path, records: Sequence[dict[str, Any]]) -> None:
    assignments = sorted(
        {
            (str(record["source"]), str(record["speaker_id"]), str(record["split"]))
            for record in records
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["source", "speaker_id", "split"])
        writer.writerows(assignments)


@dataclass(frozen=True)
class BuildOutputs:
    manifest_dir: Path
    reports_dir: Path
    counts: Mapping[str, int]


def build_chinese_accent_manifests(
    *,
    source_root: Path,
    manifest_root: PurePosixPath | str = DEFAULT_REMOTE_DATASET_ROOT,
    g2p,
    output_dir: Path | None = None,
    reports_dir: Path | None = None,
    speechocean_root: Path | None = None,
    l2_arctic_root: Path | None = None,
    edacc_root: Path | None = None,
    real_recordings_root: Path | None = None,
    real_destination_root: Path | None = None,
    l2_derived_root: Path | None = None,
    normalize_l2_audio: bool = True,
    ffmpeg_bin: str = "ffmpeg",
    audio_workers: int = 4,
    librispeech_train_manifest: Path | None = None,
    librispeech_dev_manifest: Path | None = None,
    train_mix_size: int = 50000,
    dev_select_size: int = 5000,
    seed: int = DEFAULT_SPLIT_SEED,
    speechocean_dev_speakers: int = 25,
    verify_audio: bool = True,
) -> BuildOutputs:
    """Build source-specific adapter/evaluation manifests and audit reports."""
    source_root = Path(source_root).expanduser().resolve()
    output_dir = Path(output_dir or source_root / "manifests").resolve()
    reports_dir = Path(reports_dir or source_root / "reports").resolve()
    speechocean_root = Path(
        speechocean_root or source_root / "raw" / "speechocean762"
    ).resolve()
    l2_arctic_root = Path(l2_arctic_root or source_root / "raw" / "l2_arctic_v5").resolve()
    edacc_root = Path(edacc_root or source_root / "raw" / "edacc_mandarin").resolve()
    real_destination_root = Path(
        real_destination_root or source_root / "raw" / "hey_eva_real" / "pc"
    ).resolve()
    l2_derived_root = Path(
        l2_derived_root or source_root / "derived" / "audio_16k" / "l2_arctic"
    ).resolve()
    mapper = ManifestPathMapper(source_root, manifest_root)
    edacc_source_paths = normalize_edacc_source_manifest_paths(
        edacc_root,
        mapper=mapper,
    )

    l2_normalization = {"total": 0, "written": 0, "skipped": 0}
    if normalize_l2_audio:
        l2_normalization = normalize_l2_arctic_audio(
            l2_arctic_root,
            l2_derived_root,
            ffmpeg_bin=ffmpeg_bin,
            num_workers=audio_workers,
        )

    speech_records = prepare_speechocean_records(
        speechocean_root,
        mapper=mapper,
        g2p=g2p,
        dev_speaker_count=speechocean_dev_speakers,
        seed=seed,
        verify_audio=verify_audio,
    )
    l2_records = prepare_l2_arctic_records(
        l2_arctic_root,
        mapper=mapper,
        g2p=g2p,
        audio_root=l2_derived_root if normalize_l2_audio else None,
        verify_audio=verify_audio,
    )
    edacc_records, quarantine_records = prepare_edacc_records(
        edacc_root,
        mapper=mapper,
        g2p=g2p,
        verify_audio=verify_audio,
    )
    real_records: list[dict[str, Any]] = []
    real_quarantine: list[dict[str, Any]] = []
    if real_recordings_root is not None:
        real_records, real_quarantine = prepare_real_recording_records(
            real_recordings_root,
            destination_root=real_destination_root,
            mapper=mapper,
            g2p=g2p,
            seed=seed,
        )
    all_records = speech_records + l2_records + edacc_records + real_records
    split_audit = _validate_records(all_records, mapper.manifest_root)

    manifest_groups: dict[str, list[dict[str, Any]]] = {}
    for source in ("speechocean762", "l2_arctic_mandarin"):
        for split in ("train", "dev", "test"):
            manifest_groups[f"adapter/{source}_{split}.jsonl"] = _records_for(
                all_records, source=source, split=split
            )
    for split in ("train", "dev", "test"):
        manifest_groups[f"adapter/accent_{split}.jsonl"] = [
            record
            for record in all_records
            if record["source"] in {"speechocean762", "l2_arctic_mandarin"}
            and record["split"] == split
        ]
    for split in ("dev", "test", "control"):
        manifest_groups[f"eval/edacc_{split}.jsonl"] = _records_for(
            all_records, source="edacc_mandarin", split=split
        )
    if real_records:
        for split in ("train", "dev", "test"):
            manifest_groups[f"adapter/real_{split}.jsonl"] = _records_for(
                all_records, source="hey_eva_real_pc", split=split
            )
    manifest_groups["lora/hard_negative_candidates.jsonl"] = hard_negative_candidate_records(
        all_records,
        include_real=bool(real_records),
    )
    manifest_groups["catalog.jsonl"] = all_records
    manifest_groups["quarantine/edacc.jsonl"] = quarantine_records
    if real_quarantine:
        manifest_groups["quarantine/real_pc.jsonl"] = real_quarantine

    counts: dict[str, int] = {}
    for relative_path, records in manifest_groups.items():
        counts[relative_path] = _write_jsonl(output_dir / relative_path, records)
    _write_split_assignments(output_dir / "speaker_assignments.csv", all_records)
    _write_mixture_recipe(
        output_dir / "adapter" / "mixture_recipe.json",
        train_size=train_mix_size,
        dev_size=dev_select_size,
    )
    if real_records:
        counts.update(write_real_lora_views(source_root / "views" / "hey_eva_adapt", real_records))
    if (librispeech_train_manifest is None) != (librispeech_dev_manifest is None):
        raise ValueError(
            "librispeech_train_manifest and librispeech_dev_manifest must be supplied together"
        )
    if librispeech_train_manifest is not None and librispeech_dev_manifest is not None:
        counts.update(
            materialize_adapter_mixtures(
                manifest_dir=output_dir,
                librispeech_train_manifest=librispeech_train_manifest,
                librispeech_dev_manifest=librispeech_dev_manifest,
                train_size=train_mix_size,
                dev_size=dev_select_size,
                seed=seed,
            )
        )

    reports_dir.mkdir(parents=True, exist_ok=True)
    inventory = {
        "source_root": str(source_root),
        "manifest_root": str(mapper.manifest_root),
        "split_seed": seed,
        "counts": counts,
        "l2_audio_normalization": l2_normalization,
        "edacc_source_manifest_paths": edacc_source_paths,
        "sources": {
            "speechocean762": str(speechocean_root),
            "l2_arctic_mandarin": str(l2_arctic_root),
            "edacc_mandarin": str(edacc_root),
        },
    }
    (reports_dir / "inventory.json").write_text(
        json.dumps(inventory, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (reports_dir / "split_audit.json").write_text(
        json.dumps(split_audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    phone_counts: dict[str, int] = defaultdict(int)
    for record in all_records:
        for phone in record["phonemes"]:
            phone_counts[str(phone)] += 1
    (reports_dir / "phoneme_audit.json").write_text(
        json.dumps(
            {
                "unsupported_phones": [],
                "phone_counts": dict(sorted(phone_counts.items())),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return BuildOutputs(manifest_dir=output_dir, reports_dir=reports_dir, counts=counts)


def _safe_extract_members(archive: zipfile.ZipFile, destination: Path) -> None:
    destination = destination.resolve()
    for member in archive.infolist():
        member_path = PurePosixPath(member.filename)
        if member_path.is_absolute() or ".." in member_path.parts:
            raise ValueError(f"Unsafe path in L2-ARCTIC archive: {member.filename!r}")
        target = (destination / Path(*member_path.parts)).resolve()
        try:
            target.relative_to(destination)
        except ValueError as exc:
            raise ValueError(f"Unsafe path in L2-ARCTIC archive: {member.filename!r}") from exc
        if member.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        with archive.open(member) as reader, target.open("wb") as writer:
            shutil.copyfileobj(reader, writer)


def extract_l2_arctic_mandarin(archive_path: Path, destination: Path) -> None:
    """Extract only the four Mandarin speakers plus corpus metadata.

    L2-ARCTIC stores one zip per speaker inside the release zip, so each nested
    archive is streamed through a temporary file instead of held in memory.
    Existing speaker directories are rejected to avoid merging partial trees.
    """
    archive_path = Path(archive_path).expanduser().resolve()
    destination = Path(destination).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as outer:
        for metadata_name in ("LICENSE", "README.md", "README.pdf", "PROMPTS"):
            if metadata_name in outer.namelist():
                target = destination / metadata_name
                if not target.exists():
                    with outer.open(metadata_name) as reader, target.open("wb") as writer:
                        shutil.copyfileobj(reader, writer)
        for speaker in L2_MANDARIN_SPLITS:
            speaker_dir = destination / speaker
            if speaker_dir.exists():
                raise FileExistsError(
                    f"Refusing to merge L2-ARCTIC speaker into existing directory: {speaker_dir}"
                )
            nested_name = f"{speaker}.zip"
            with tempfile.NamedTemporaryFile(suffix=f"-{nested_name}") as temporary:
                with outer.open(nested_name) as reader:
                    shutil.copyfileobj(reader, temporary)
                temporary.flush()
                with zipfile.ZipFile(temporary.name) as nested:
                    _safe_extract_members(nested, destination)


def count_records_by_source_split(records: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    """Small reporting helper kept public for tests and follow-up tooling."""
    counts: dict[str, int] = defaultdict(int)
    for record in records:
        counts[f"{record['source']}:{record['split']}"] += 1
    return dict(sorted(counts.items()))
