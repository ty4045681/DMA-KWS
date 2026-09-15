"""MUSAN, DNS, and FSD50K adapters that convert native metadata to catalog records."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
import csv
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Protocol
import wave

from dma_kws.data_prep.background_manifest import (
    BACKGROUND_MANIFEST_SCHEMA_VERSION,
    BackgroundRecord,
    sha256_file,
)
from dma_kws.data_prep.musan_split import (
    MUSAN_AUDIO_EXTENSIONS,
    discover_musan_recordings,
)


ALLOWED_ADAPTERS = ("musan", "dns", "fsd50k")
FSD50K_DEV_AUDIO = "FSD50K.dev_audio"
FSD50K_EVAL_AUDIO = "FSD50K.eval_audio"
FSD50K_GROUND_TRUTH = "FSD50K.ground_truth"
FSD50K_METADATA = "FSD50K.metadata"
FSD50K_DEV_CLIPS_JSON = "dev_clips_info_FSD50K.json"
FSD50K_EVAL_CLIPS_JSON = "eval_clips_info_FSD50K.json"
FSD50K_CLIP_REQUIRED = ("uploader", "license")
MUSAN_TRAIN_LIST = "train_background.list"
MUSAN_EVAL_LIST = "eval_musan.list"


@dataclass(frozen=True)
class SourceImportConfig:
    adapter: str
    id: str
    root: str
    split_dir: str = ""
    metadata: str = ""
    eligible_ids_file: str = ""
    split_policy: str = "group_random"
    category_allow: tuple[str, ...] = ()
    category_exclude: tuple[str, ...] = ()


class SourceAdapter(Protocol):
    def discover(self, config: SourceImportConfig) -> Iterable[BackgroundRecord]:
        ...


def source_field_error(
    source_id: str,
    field: str,
    *,
    reason: str,
    expected: str,
    actual: object,
    file: str | Path | None = None,
) -> ValueError:
    location = f"source id={source_id!r} field={field}"
    if file is not None:
        location += f" file={file}"
    return ValueError(
        f"{location}: {reason}; expected {expected}, got {actual!r}"
    )


def _valid_sha256(value: str) -> bool:
    if len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def get_source_adapter(name: str) -> SourceAdapter:
    try:
        adapter_cls = SOURCE_ADAPTERS[name]
    except KeyError as exc:
        allowed = ", ".join(SOURCE_ADAPTERS)
        raise ValueError(
            f"unknown background source adapter {name!r}; allowed: {allowed}"
        ) from exc
    return adapter_cls()


def probe_audio(path: str | Path) -> tuple[int, int, int, float]:
    """Return frames, sample_rate, channels, duration_seconds."""
    audio_path = Path(path)
    try:
        import soundfile as sf
    except ModuleNotFoundError:
        sf = None
    if sf is not None:
        try:
            info = sf.info(str(audio_path))
        except (RuntimeError, TypeError) as exc:
            raise ValueError(f"Could not read audio metadata: {audio_path}") from exc
        frames = int(info.frames)
        sample_rate = int(info.samplerate)
        channels = int(info.channels)
    elif audio_path.suffix.casefold() != ".wav":
        raise RuntimeError(
            f"Reading {audio_path.suffix} metadata requires the soundfile dependency"
        )
    else:
        try:
            with wave.open(str(audio_path), "rb") as handle:
                frames = int(handle.getnframes())
                sample_rate = int(handle.getframerate())
                channels = int(handle.getnchannels())
        except (EOFError, OSError, wave.Error) as exc:
            raise ValueError(f"Could not read audio metadata: {audio_path}") from exc
    if frames <= 0 or sample_rate <= 0 or channels <= 0:
        raise ValueError(
            f"Audio must have positive frames, sample rate, and channels: {audio_path} "
            f"(frames={frames}, sample_rate={sample_rate}, channels={channels})"
        )
    return frames, sample_rate, channels, frames / float(sample_rate)


def _require_source_root(root: str | Path, *, label: str) -> Path:
    path = Path(root).expanduser().resolve()
    if not path.is_dir():
        raise NotADirectoryError(f"{label} not found: {path}")
    return path


def _iter_audio_files(root: Path, *, extensions: Sequence[str] = MUSAN_AUDIO_EXTENSIONS):
    allowed = {item.casefold() for item in extensions}
    files = sorted(
        (
            path
            for path in root.rglob("*")
            if path.is_file() and path.suffix.casefold() in allowed
        ),
        key=lambda item: item.as_posix(),
    )
    for path in files:
        resolved = path.resolve()
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise ValueError(
                f"audio resolves outside {root}: {path} -> {resolved}"
            ) from exc
        yield resolved


def _load_path_list(path: Path) -> list[Path]:
    if not path.is_file():
        raise FileNotFoundError(f"split list not found: {path}")
    files: list[Path] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        entry = line.strip()
        if not entry or entry.startswith("#"):
            continue
        item = Path(entry).expanduser()
        if not item.is_absolute():
            item = path.parent / item
        files.append(item.resolve())
    return files


def _read_csv_rows(
    path: Path, required: Sequence[str], *, source_id: str
) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise source_field_error(
                source_id,
                required[0] if required else "columns",
                reason=f"missing required columns: {', '.join(required)}",
                expected=", ".join(required),
                actual="",
                file=path,
            )
        fieldnames = [name.lstrip("\ufeff").strip() for name in reader.fieldnames]
        missing = [name for name in required if name not in fieldnames]
        if missing:
            raise source_field_error(
                source_id,
                missing[0],
                reason=f"missing required columns: {', '.join(missing)}",
                expected=", ".join(required),
                actual=", ".join(fieldnames),
                file=path,
            )
        rows: list[dict[str, str]] = []
        for raw in reader:
            row = {
                (key or "").lstrip("\ufeff").strip(): (value or "").strip()
                for key, value in raw.items()
            }
            rows.append(row)
        return rows


def _load_fsd50k_clip_info(path: Path, *, source_id: str) -> dict[str, dict[str, str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise source_field_error(
            source_id,
            "metadata",
            reason="clip info JSON must be a dict keyed by fname",
            expected="a JSON object mapping fname to clip metadata",
            actual=type(payload).__name__,
            file=path,
        )
    clips: dict[str, dict[str, str]] = {}
    for fname, info in payload.items():
        if not isinstance(info, Mapping):
            raise source_field_error(
                source_id,
                "metadata",
                reason=f"clip info for fname {fname!r} must be a mapping",
                expected="keys uploader and license",
                actual=type(info).__name__,
                file=path,
            )
        missing = [name for name in FSD50K_CLIP_REQUIRED if name not in info]
        if missing:
            raise source_field_error(
                source_id,
                missing[0],
                reason=f"missing required columns: {', '.join(missing)}",
                expected=", ".join(FSD50K_CLIP_REQUIRED),
                actual=", ".join(str(key) for key in info),
                file=path,
            )
        clips[str(fname)] = {
            "license": str(info.get("license") or "").strip(),
            "uploader": str(info.get("uploader") or "").strip(),
        }
    return clips


def _first_existing(candidates: Sequence[Path], *, label: str) -> Path:
    for path in candidates:
        if path.is_file():
            return path
    looked = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"{label} not found; looked in: {looked}")


def _split_labels(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.split(",") if part.strip())


def _audio_basename(fname: str) -> str:
    name = fname.strip()
    if not name:
        raise ValueError("fname must not be empty")
    if Path(name).suffix:
        return Path(name).name
    return f"{name}.wav"


class MusanSourceAdapter:
    def discover(self, config: SourceImportConfig) -> Iterable[BackgroundRecord]:
        root = _require_source_root(config.root, label="MUSAN root")
        native = discover_musan_recordings(root)
        membership: dict[Path, str] | None = None
        if config.split_dir:
            split_dir = Path(config.split_dir).expanduser().resolve()
            train_paths = set(_load_path_list(split_dir / MUSAN_TRAIN_LIST))
            eval_paths = set(_load_path_list(split_dir / MUSAN_EVAL_LIST))
            overlap = train_paths & eval_paths
            if overlap:
                raise ValueError(
                    f"old MUSAN train/eval lists overlap: {sorted(str(path) for path in overlap)[:3]}"
                )
            membership = {path: "train" for path in train_paths}
            membership.update({path: "test" for path in eval_paths})

        records: list[BackgroundRecord] = []
        for item in native:
            if membership is None:
                split = "train"
            else:
                try:
                    split = membership[item.path.resolve()]
                except KeyError as exc:
                    raise source_field_error(
                        config.id,
                        "split_dir",
                        reason="MUSAN recording not present in split_dir lists",
                        expected="a member of train_background.list or eval_musan.list",
                        actual=item.relative_path,
                        file=split_dir,
                    ) from exc
            _frames, _sample_rate, channels, _duration = probe_audio(item.path)
            digest = sha256_file(item.path)
            if not _valid_sha256(digest):
                raise source_field_error(
                    config.id,
                    "audio_sha256",
                    reason="empty or invalid audio_sha256 from adapter",
                    expected="64 hex characters",
                    actual=digest,
                    file=item.path,
                )
            native_id = item.relative_path
            records.append(
                BackgroundRecord(
                    schema_version=BACKGROUND_MANIFEST_SCHEMA_VERSION,
                    dataset_id=config.id,
                    recording_id=f"{config.id}:{native_id}",
                    relative_path=item.relative_path,
                    audio_path=str(item.path),
                    group_id=f"{config.id}:{item.group_id}",
                    origin_ids=(f"{config.id}:{native_id}",),
                    split=split,
                    categories=(item.category,),
                    background_eligible=False,
                    eligibility_basis="",
                    duration_seconds=item.duration_seconds,
                    sample_rate=item.sample_rate,
                    channels=channels,
                    audio_sha256=digest,
                    license_id="",
                    provenance_complete=True,
                )
            )
        return records


class DnsSourceAdapter:
    def discover(self, config: SourceImportConfig) -> Iterable[BackgroundRecord]:
        root = _require_source_root(config.root, label="DNS noise root")
        sidecar: dict[str, dict[str, str]] = {}
        if config.metadata:
            meta_path = Path(config.metadata).expanduser()
            if not meta_path.is_file():
                raise source_field_error(
                    config.id,
                    "metadata",
                    reason="configured DNS metadata sidecar is missing or not a file",
                    expected="an existing sidecar CSV file",
                    actual=str(meta_path),
                    file=meta_path,
                )
            for row in _read_csv_rows(
                meta_path, required=("relative_path",), source_id=config.id
            ):
                sidecar[row["relative_path"]] = row

        records: list[BackgroundRecord] = []
        for path in _iter_audio_files(root):
            relative_path = path.relative_to(root).as_posix()
            _frames, sample_rate, channels, duration_seconds = probe_audio(path)
            digest = sha256_file(path)
            if not _valid_sha256(digest):
                raise source_field_error(
                    config.id,
                    "audio_sha256",
                    reason="empty or invalid audio_sha256 from adapter",
                    expected="64 hex characters",
                    actual=digest,
                    file=path,
                )
            extra = sidecar.get(relative_path, {})
            group_token = extra.get("group_id") or relative_path
            origin_raw = extra.get("origin_ids") or extra.get("origin_id") or ""
            origin_ids = _split_labels(origin_raw) if origin_raw else ()
            records.append(
                BackgroundRecord(
                    schema_version=BACKGROUND_MANIFEST_SCHEMA_VERSION,
                    dataset_id=config.id,
                    recording_id=f"{config.id}:{relative_path}",
                    relative_path=relative_path,
                    audio_path=str(path),
                    group_id=f"{config.id}:{group_token}",
                    origin_ids=origin_ids,
                    split="train",
                    categories=(),
                    background_eligible=False,
                    eligibility_basis="",
                    duration_seconds=duration_seconds,
                    sample_rate=sample_rate,
                    channels=channels,
                    audio_sha256=digest,
                    license_id=extra.get("license") or extra.get("license_id") or "",
                    provenance_complete=bool(origin_ids),
                )
            )
        if not records:
            raise source_field_error(
                config.id,
                "root",
                reason="no supported audio found under DNS noise root",
                expected="audio files under the configured noise root",
                actual=str(root),
                file=root,
            )
        return records


class Fsd50kSourceAdapter:
    def discover(self, config: SourceImportConfig) -> Iterable[BackgroundRecord]:
        root = _require_source_root(config.root, label="FSD50K root")
        metadata_root = (
            Path(config.metadata).expanduser() if config.metadata else None
        )
        ground_truth_candidates = []
        clips_candidates = []
        if metadata_root is not None:
            ground_truth_candidates.extend(
                [
                    metadata_root,
                    metadata_root / FSD50K_GROUND_TRUTH,
                ]
            )
            clips_candidates.extend(
                [
                    metadata_root,
                    metadata_root / FSD50K_METADATA,
                ]
            )
        ground_truth_candidates.extend([root / FSD50K_GROUND_TRUTH, root])
        clips_candidates.extend([root / FSD50K_METADATA, root])

        dev_csv = _first_existing(
            [directory / "dev.csv" for directory in ground_truth_candidates],
            label="FSD50K.ground_truth/dev.csv",
        )
        eval_csv = _first_existing(
            [directory / "eval.csv" for directory in ground_truth_candidates],
            label="FSD50K.ground_truth/eval.csv",
        )
        dev_clips = _first_existing(
            [directory / FSD50K_DEV_CLIPS_JSON for directory in clips_candidates],
            label="FSD50K.metadata/dev_clips_info_FSD50K.json",
        )
        eval_clips = _first_existing(
            [directory / FSD50K_EVAL_CLIPS_JSON for directory in clips_candidates],
            label="FSD50K.metadata/eval_clips_info_FSD50K.json",
        )

        clip_meta = _load_fsd50k_clip_info(dev_clips, source_id=config.id)
        clip_meta.update(_load_fsd50k_clip_info(eval_clips, source_id=config.id))

        records: list[BackgroundRecord] = []
        records.extend(
            self._records_from_ground_truth(
                config,
                root=root,
                csv_path=dev_csv,
                required=("fname", "labels", "split"),
                audio_dir=FSD50K_DEV_AUDIO,
                clip_meta=clip_meta,
                force_split=None,
            )
        )
        records.extend(
            self._records_from_ground_truth(
                config,
                root=root,
                csv_path=eval_csv,
                required=("fname", "labels"),
                audio_dir=FSD50K_EVAL_AUDIO,
                clip_meta=clip_meta,
                force_split="test",
            )
        )
        return records

    def _records_from_ground_truth(
        self,
        config: SourceImportConfig,
        *,
        root: Path,
        csv_path: Path,
        required: Sequence[str],
        audio_dir: str,
        clip_meta: dict[str, dict[str, str]],
        force_split: str | None,
    ) -> list[BackgroundRecord]:
        records: list[BackgroundRecord] = []
        for row in _read_csv_rows(csv_path, required=required, source_id=config.id):
            fname = row["fname"]
            if force_split is not None:
                split = force_split
            else:
                split = row["split"].strip()
                if split == "eval":
                    split = "test"
            if split not in {"train", "val", "test"}:
                raise source_field_error(
                    config.id,
                    "split",
                    reason=f"fname {fname!r} has invalid split",
                    expected="train, val, or test",
                    actual=split,
                    file=csv_path,
                )
            meta = clip_meta.get(str(fname))
            if meta is None:
                raise source_field_error(
                    config.id,
                    "metadata",
                    reason=f"clip info missing for fname {fname!r}",
                    expected="an entry in dev_clips_info_FSD50K.json or eval_clips_info_FSD50K.json",
                    actual=fname,
                )
            uploader = meta.get("uploader", "")
            license_id = meta.get("license", "")
            wav_name = _audio_basename(fname)
            relative_path = f"{audio_dir}/{wav_name}"
            audio_path = (root / relative_path).resolve()
            if not audio_path.is_file():
                raise FileNotFoundError(f"FSD50K audio not found: {audio_path}")
            _frames, sample_rate, channels, duration_seconds = probe_audio(audio_path)
            digest = sha256_file(audio_path)
            if not _valid_sha256(digest):
                raise source_field_error(
                    config.id,
                    "audio_sha256",
                    reason="empty or invalid audio_sha256 from adapter",
                    expected="64 hex characters",
                    actual=digest,
                    file=audio_path,
                )
            native_id = Path(fname).stem if Path(fname).suffix else fname
            group_token = f"uploader:{uploader}" if uploader else native_id
            origin_ids = (f"freesound:{native_id}",)
            records.append(
                BackgroundRecord(
                    schema_version=BACKGROUND_MANIFEST_SCHEMA_VERSION,
                    dataset_id=config.id,
                    recording_id=f"{config.id}:{native_id}",
                    relative_path=relative_path,
                    audio_path=str(audio_path),
                    group_id=f"{config.id}:{group_token}",
                    origin_ids=origin_ids,
                    split=split,
                    categories=_split_labels(row["labels"]),
                    background_eligible=False,
                    eligibility_basis="",
                    duration_seconds=duration_seconds,
                    sample_rate=sample_rate,
                    channels=channels,
                    audio_sha256=digest,
                    license_id=license_id,
                    provenance_complete=True,
                )
            )
        return records


SOURCE_ADAPTERS: dict[str, type] = {
    "musan": MusanSourceAdapter,
    "dns": DnsSourceAdapter,
    "fsd50k": Fsd50kSourceAdapter,
}


__all__ = [
    "ALLOWED_ADAPTERS",
    "DnsSourceAdapter",
    "Fsd50kSourceAdapter",
    "MusanSourceAdapter",
    "SOURCE_ADAPTERS",
    "SourceAdapter",
    "SourceImportConfig",
    "get_source_adapter",
    "probe_audio",
    "source_field_error",
]
