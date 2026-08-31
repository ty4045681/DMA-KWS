"""Build leakage-safe train/evaluation allowlists from a complete MUSAN tree."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Iterable, Mapping, Sequence
import wave


MUSAN_CATEGORIES = ("music", "noise", "speech")
MUSAN_AUDIO_EXTENSIONS = (".wav", ".flac", ".mp3", ".m4a")
DEFAULT_MUSAN_SPLIT_SEED = 20260831
DEFAULT_MUSAN_TRAIN_RATIO = 0.60
MUSAN_SPLIT_SCHEMA_VERSION = 1

_UNKNOWN_ARTISTS = {"", "-", "n/a", "na", "none", "unknown"}


@dataclass(frozen=True)
class MusanRecording:
    """One canonical audio recording and the leakage group it belongs to."""

    path: Path
    relative_path: str
    category: str
    source: str
    duration_microseconds: int
    frames: int
    sample_rate: int
    group_id: str
    group_kind: str

    @property
    def stratum(self) -> tuple[str, str]:
        return self.category, self.source

    @property
    def duration_seconds(self) -> float:
        return self.duration_microseconds / 1_000_000.0


@dataclass(frozen=True)
class MusanSplit:
    """In-memory result of a deterministic MUSAN split."""

    train: tuple[MusanRecording, ...]
    eval: tuple[MusanRecording, ...]


@dataclass(frozen=True)
class _RecordingGroup:
    group_id: str
    recordings: tuple[MusanRecording, ...]
    duration_microseconds: int


def _stable_digest(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}\0{value}".encode("utf-8")).hexdigest()


def _path_catalog_sha256(recordings: Iterable[MusanRecording]) -> str:
    """Hash the sorted root-relative paths using the evaluator's convention."""
    digest = hashlib.sha256()
    for relative_path in sorted(record.relative_path for record in recordings):
        digest.update(relative_path.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _metadata_catalog_sha256(recordings: Iterable[MusanRecording]) -> str:
    """Hash path plus decoded shape to make silent catalog changes auditable."""
    digest = hashlib.sha256()
    for record in sorted(recordings, key=lambda item: item.relative_path):
        row = (
            f"{record.relative_path}\0{record.frames}\0{record.sample_rate}\0"
            f"{record.group_id}\n"
        )
        digest.update(row.encode("utf-8"))
    return digest.hexdigest()


def _read_music_artists(source_dir: Path) -> dict[str, str]:
    """Read MUSAN's optional ``music/<source>/ANNOTATIONS`` artist field."""
    annotation_path = source_dir / "ANNOTATIONS"
    if not annotation_path.is_file():
        return {}

    artists: dict[str, str] = {}
    with annotation_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            fields = stripped.split()
            if len(fields) < 4:
                continue
            recording_id = Path(fields[0]).stem
            artist = " ".join(fields[3:]).strip()
            if artist.casefold() in _UNKNOWN_ARTISTS:
                continue
            previous = artists.get(recording_id)
            if previous is not None and previous != artist:
                raise ValueError(
                    f"{annotation_path}:{line_number}: conflicting artist for "
                    f"{recording_id!r}"
                )
            artists[recording_id] = artist
    return artists


def _audio_info(path: Path) -> tuple[int, int, int]:
    try:
        import soundfile as sf
    except ModuleNotFoundError:
        sf = None

    if sf is not None:
        try:
            info = sf.info(str(path))
        except (RuntimeError, TypeError) as exc:
            raise ValueError(f"Could not read audio metadata: {path}") from exc
        frames = int(info.frames)
        sample_rate = int(info.samplerate)
    elif path.suffix.casefold() != ".wav":
        raise RuntimeError(
            f"Reading {path.suffix} metadata requires the soundfile dependency"
        )
    else:
        try:
            with wave.open(str(path), "rb") as handle:
                frames = int(handle.getnframes())
                sample_rate = int(handle.getframerate())
        except (EOFError, OSError, wave.Error) as exc:
            raise ValueError(f"Could not read audio metadata: {path}") from exc
    if frames <= 0 or sample_rate <= 0:
        raise ValueError(
            f"Audio must have positive frames and sample rate: {path} "
            f"(frames={frames}, sample_rate={sample_rate})"
        )
    duration_microseconds = (frames * 1_000_000 + sample_rate // 2) // sample_rate
    if duration_microseconds <= 0:
        raise ValueError(f"Audio duration rounds to zero microseconds: {path}")
    return frames, sample_rate, duration_microseconds


def discover_musan_recordings(
    musan_root: str | Path,
    *,
    audio_extensions: Sequence[str] = MUSAN_AUDIO_EXTENSIONS,
) -> tuple[MusanRecording, ...]:
    """Discover and validate all audio under MUSAN's music/noise/speech roots."""
    root = Path(musan_root).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"MUSAN root not found: {root}")

    extensions = {extension.casefold() for extension in audio_extensions}
    if not extensions:
        raise ValueError("audio_extensions must not be empty")

    recordings: list[MusanRecording] = []
    seen_paths: dict[Path, Path] = {}
    seen_file_ids: dict[tuple[int, int], Path] = {}
    for category in MUSAN_CATEGORIES:
        category_dir = root / category
        if not category_dir.is_dir():
            raise NotADirectoryError(
                f"Complete MUSAN tree must contain {category!r}: {category_dir}"
            )

        files = sorted(
            (
                path
                for path in category_dir.rglob("*")
                if path.is_file() and path.suffix.casefold() in extensions
            ),
            key=lambda path: path.as_posix(),
        )
        if not files:
            raise ValueError(f"No supported audio found under {category_dir}")

        source_stems: dict[str, set[str]] = defaultdict(set)
        for path in files:
            relative_discovered = path.relative_to(category_dir)
            discovered_source = (
                relative_discovered.parts[0]
                if len(relative_discovered.parts) > 1
                else "_root"
            )
            source_stems[discovered_source].add(path.stem)

        artist_cache: dict[str, Mapping[str, str]] = {}
        for discovered_path in files:
            canonical_path = discovered_path.resolve()
            try:
                relative = canonical_path.relative_to(root)
            except ValueError as exc:
                raise ValueError(
                    f"MUSAN audio resolves outside the dataset root: "
                    f"{discovered_path} -> {canonical_path}"
                ) from exc
            previous = seen_paths.get(canonical_path)
            if previous is not None:
                raise ValueError(
                    f"Duplicate canonical MUSAN audio: {previous} and {discovered_path}"
                )
            seen_paths[canonical_path] = discovered_path
            stat_result = canonical_path.stat()
            file_id = (stat_result.st_dev, stat_result.st_ino)
            if stat_result.st_ino:
                previous_identity = seen_file_ids.get(file_id)
                if previous_identity is not None:
                    raise ValueError(
                        f"MUSAN audio aliases the same physical file: "
                        f"{previous_identity} and {discovered_path}"
                    )
                seen_file_ids[file_id] = discovered_path

            relative_to_category = relative.relative_to(category)
            source = (
                relative_to_category.parts[0]
                if len(relative_to_category.parts) > 1
                else "_root"
            )
            group_kind = "recording"
            group_token = relative.as_posix()
            if category == "music" and source != "_root":
                if source not in artist_cache:
                    source_artists = _read_music_artists(category_dir / source)
                    distinct_artists = {
                        artist.casefold()
                        for recording_id, artist in source_artists.items()
                        if recording_id in source_stems[source]
                    }
                    # Official music/rfm contains many recordings but only one
                    # artist. Keeping that artist atomic would make a stratified
                    # train/eval split impossible, so fall back to recording-level
                    # atoms for any single-artist source.
                    artist_cache[source] = (
                        source_artists if len(distinct_artists) >= 2 else {}
                    )
                artist = artist_cache[source].get(canonical_path.stem)
                if artist is not None:
                    group_kind = "artist"
                    group_token = artist.casefold()

            frames, sample_rate, duration_microseconds = _audio_info(canonical_path)
            recordings.append(
                MusanRecording(
                    path=canonical_path,
                    relative_path=relative.as_posix(),
                    category=category,
                    source=source,
                    duration_microseconds=duration_microseconds,
                    frames=frames,
                    sample_rate=sample_rate,
                    group_id=f"{category}/{source}/{group_kind}:{group_token}",
                    group_kind=group_kind,
                )
            )

    return tuple(sorted(recordings, key=lambda item: item.relative_path))


def _group_recordings(records: Sequence[MusanRecording]) -> tuple[_RecordingGroup, ...]:
    grouped: dict[str, list[MusanRecording]] = defaultdict(list)
    for record in records:
        grouped[record.group_id].append(record)
    return tuple(
        _RecordingGroup(
            group_id=group_id,
            recordings=tuple(sorted(group_records, key=lambda item: item.relative_path)),
            duration_microseconds=sum(
                record.duration_microseconds for record in group_records
            ),
        )
        for group_id, group_records in sorted(grouped.items())
    )


def _partition_digest(seed: int, groups: Sequence[_RecordingGroup]) -> str:
    return _stable_digest(seed, "\n".join(sorted(group.group_id for group in groups)))


def _ensure_nonempty_partition(
    train_groups: Sequence[_RecordingGroup],
    eval_groups: Sequence[_RecordingGroup],
    *,
    target_train_microseconds: int,
) -> tuple[list[_RecordingGroup], list[_RecordingGroup]]:
    train = list(train_groups)
    evaluation = list(eval_groups)
    if not train:
        selected = min(
            evaluation,
            key=lambda group: (
                abs(group.duration_microseconds - target_train_microseconds),
                group.group_id,
            ),
        )
        evaluation.remove(selected)
        train.append(selected)
    if not evaluation:
        total_train = sum(group.duration_microseconds for group in train)
        selected = min(
            train,
            key=lambda group: (
                abs(
                    total_train
                    - group.duration_microseconds
                    - target_train_microseconds
                ),
                group.group_id,
            ),
        )
        train.remove(selected)
        evaluation.append(selected)
    return train, evaluation


def _repair_partition(
    train_groups: Sequence[_RecordingGroup],
    eval_groups: Sequence[_RecordingGroup],
    *,
    target_train_microseconds: int,
    seed: int,
) -> tuple[list[_RecordingGroup], list[_RecordingGroup]]:
    """Apply deterministic one-group moves and swaps while error improves."""
    train, evaluation = _ensure_nonempty_partition(
        train_groups,
        eval_groups,
        target_train_microseconds=target_train_microseconds,
    )
    train_duration = sum(group.duration_microseconds for group in train)

    for _ in range(4 * (len(train) + len(evaluation))):
        current_error = abs(train_duration - target_train_microseconds)
        if current_error == 0:
            break
        candidates: list[
            tuple[int, str, str, _RecordingGroup, _RecordingGroup | None]
        ] = []
        if len(train) > 1:
            for group in train:
                new_duration = train_duration - group.duration_microseconds
                candidates.append(
                    (
                        abs(new_duration - target_train_microseconds),
                        _stable_digest(seed, f"move-train-to-eval\0{group.group_id}"),
                        "train_to_eval",
                        group,
                        None,
                    )
                )
        if len(evaluation) > 1:
            for group in evaluation:
                new_duration = train_duration + group.duration_microseconds
                candidates.append(
                    (
                        abs(new_duration - target_train_microseconds),
                        _stable_digest(seed, f"move-eval-to-train\0{group.group_id}"),
                        "eval_to_train",
                        group,
                        None,
                    )
                )
        for train_group in train:
            for eval_group in evaluation:
                new_duration = (
                    train_duration
                    - train_group.duration_microseconds
                    + eval_group.duration_microseconds
                )
                candidates.append(
                    (
                        abs(new_duration - target_train_microseconds),
                        _stable_digest(
                            seed,
                            f"swap\0{train_group.group_id}\0{eval_group.group_id}",
                        ),
                        "swap",
                        train_group,
                        eval_group,
                    )
                )
        best = min(candidates, default=None, key=lambda item: (item[0], item[1]))
        if best is None or best[0] >= current_error:
            break

        _, _, action, first_group, second_group = best
        if action == "train_to_eval":
            train.remove(first_group)
            evaluation.append(first_group)
            train_duration -= first_group.duration_microseconds
        elif action == "eval_to_train":
            evaluation.remove(first_group)
            train.append(first_group)
            train_duration += first_group.duration_microseconds
        else:
            assert second_group is not None
            train.remove(first_group)
            evaluation.remove(second_group)
            train.append(second_group)
            evaluation.append(first_group)
            train_duration += (
                second_group.duration_microseconds - first_group.duration_microseconds
            )
    return train, evaluation


def _initial_partitions(
    groups: Sequence[_RecordingGroup],
    *,
    target_train_microseconds: int,
    seed: int,
    stratum_name: str,
) -> list[tuple[list[_RecordingGroup], list[_RecordingGroup]]]:
    total_duration = sum(group.duration_microseconds for group in groups)
    target_eval_microseconds = total_duration - target_train_microseconds

    largest_first = sorted(
        groups,
        key=lambda group: (
            -group.duration_microseconds,
            _stable_digest(seed, f"{stratum_name}\0lpt\0{group.group_id}"),
        ),
    )
    train: list[_RecordingGroup] = []
    evaluation: list[_RecordingGroup] = []
    train_remaining = target_train_microseconds
    eval_remaining = target_eval_microseconds
    for group in largest_first:
        if train_remaining > eval_remaining:
            side = "train"
        elif eval_remaining > train_remaining:
            side = "eval"
        else:
            side = (
                "train"
                if int(
                    _stable_digest(
                        seed, f"{stratum_name}\0tie\0{group.group_id}"
                    )[:2],
                    16,
                )
                % 2
                == 0
                else "eval"
            )
        if side == "train":
            train.append(group)
            train_remaining -= group.duration_microseconds
        else:
            evaluation.append(group)
            eval_remaining -= group.duration_microseconds
    candidates = [(train, evaluation)]

    for restart in range(8):
        ordered = sorted(
            groups,
            key=lambda group: (
                _stable_digest(
                    seed,
                    f"{stratum_name}\0restart-{restart}\0{group.group_id}",
                ),
                group.group_id,
            ),
        )
        prefix_duration = 0
        prefixes: list[tuple[int, int]] = []
        for split_index, group in enumerate(ordered[:-1], start=1):
            prefix_duration += group.duration_microseconds
            prefixes.append(
                (abs(prefix_duration - target_train_microseconds), split_index)
            )
        _, split_index = min(prefixes, key=lambda item: (item[0], item[1]))
        candidates.append((ordered[:split_index], ordered[split_index:]))
    return candidates


def _split_stratum(
    records: Sequence[MusanRecording],
    *,
    train_ratio: float,
    seed: int,
) -> tuple[tuple[MusanRecording, ...], tuple[MusanRecording, ...]]:
    groups = _group_recordings(records)
    category, source = records[0].stratum
    if len(groups) < 2:
        raise ValueError(
            f"MUSAN stratum {category}/{source} has only {len(groups)} independent "
            "group; at least 2 are required to keep both splits represented"
        )

    total_duration = sum(group.duration_microseconds for group in groups)
    target_duration = round(total_duration * train_ratio)
    initial_candidates = _initial_partitions(
        groups,
        target_train_microseconds=target_duration,
        seed=seed,
        stratum_name=f"{category}/{source}",
    )
    train_groups, eval_groups = min(
        initial_candidates,
        key=lambda partition: (
            abs(
                sum(group.duration_microseconds for group in partition[0])
                - target_duration
            ),
            _partition_digest(seed, partition[0]),
        ),
    )
    train_groups, eval_groups = _repair_partition(
        train_groups,
        eval_groups,
        target_train_microseconds=target_duration,
        seed=seed,
    )
    train = tuple(record for group in train_groups for record in group.recordings)
    evaluation = tuple(record for group in eval_groups for record in group.recordings)
    return train, evaluation


def split_musan_recordings(
    recordings: Sequence[MusanRecording],
    *,
    train_ratio: float = DEFAULT_MUSAN_TRAIN_RATIO,
    seed: int = DEFAULT_MUSAN_SPLIT_SEED,
) -> MusanSplit:
    """Split by category/source, balancing duration without splitting a group."""
    if not 0.0 < train_ratio < 1.0:
        raise ValueError(f"train_ratio must be in (0, 1), got {train_ratio}")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError(f"seed must be a non-negative integer, got {seed!r}")
    if not recordings:
        raise ValueError("MUSAN recording catalog is empty")

    paths = [record.path for record in recordings]
    if len(set(paths)) != len(paths):
        raise ValueError("MUSAN recording catalog contains duplicate canonical paths")
    group_strata: dict[str, tuple[str, str]] = {}
    for record in recordings:
        if record.duration_microseconds <= 0:
            raise ValueError(
                f"MUSAN recording duration must be positive: {record.relative_path}"
            )
        previous_stratum = group_strata.setdefault(record.group_id, record.stratum)
        if previous_stratum != record.stratum:
            raise ValueError(
                f"MUSAN leakage group {record.group_id!r} crosses strata: "
                f"{previous_stratum} and {record.stratum}"
            )

    strata: dict[tuple[str, str], list[MusanRecording]] = defaultdict(list)
    for record in recordings:
        strata[record.stratum].append(record)

    train: list[MusanRecording] = []
    evaluation: list[MusanRecording] = []
    for stratum in sorted(strata):
        stratum_train, stratum_eval = _split_stratum(
            strata[stratum], train_ratio=train_ratio, seed=seed
        )
        train.extend(stratum_train)
        evaluation.extend(stratum_eval)

    train = sorted(train, key=lambda item: item.relative_path)
    evaluation = sorted(evaluation, key=lambda item: item.relative_path)
    all_paths = {record.path for record in recordings}
    train_paths = {record.path for record in train}
    eval_paths = {record.path for record in evaluation}
    if train_paths & eval_paths:
        raise AssertionError("Internal error: train and eval audio paths overlap")
    if train_paths | eval_paths != all_paths:
        raise AssertionError("Internal error: split does not cover the MUSAN catalog")

    train_groups = {record.group_id for record in train}
    eval_groups = {record.group_id for record in evaluation}
    if train_groups & eval_groups:
        raise AssertionError("Internal error: a leakage group crosses train and eval")
    return MusanSplit(train=tuple(train), eval=tuple(evaluation))


def _duration_summary(recordings: Sequence[MusanRecording]) -> dict[str, Any]:
    duration_microseconds = sum(
        record.duration_microseconds for record in recordings
    )
    duration_seconds = duration_microseconds / 1_000_000.0
    groups_by_kind: dict[str, set[str]] = defaultdict(set)
    for record in recordings:
        groups_by_kind[record.group_kind].add(record.group_id)
    return {
        "recordings": len(recordings),
        "groups": len({record.group_id for record in recordings}),
        "groups_by_kind": {
            kind: len(group_ids) for kind, group_ids in sorted(groups_by_kind.items())
        },
        "duration_microseconds": duration_microseconds,
        "duration_seconds": duration_seconds,
        "duration_hours": duration_seconds / 3600.0,
        "catalog_sha256": _path_catalog_sha256(recordings),
        "metadata_sha256": _metadata_catalog_sha256(recordings),
    }


def _split_summary(recordings: Sequence[MusanRecording]) -> dict[str, Any]:
    summary = _duration_summary(recordings)
    by_category: dict[str, Any] = {}
    by_stratum: dict[str, Any] = {}
    for category in MUSAN_CATEGORIES:
        category_records = [record for record in recordings if record.category == category]
        by_category[category] = _duration_summary(category_records)
    for stratum in sorted({record.stratum for record in recordings}):
        category, source = stratum
        stratum_records = [record for record in recordings if record.stratum == stratum]
        by_stratum[f"{category}/{source}"] = _duration_summary(stratum_records)
    summary["by_category"] = by_category
    summary["by_stratum"] = by_stratum
    return summary


def _balance_summary(
    catalog: Sequence[MusanRecording],
    train: Sequence[MusanRecording],
    *,
    train_ratio: float,
) -> dict[str, Any]:
    train_paths = {record.path for record in train}
    balance: dict[str, Any] = {}
    for stratum in sorted({record.stratum for record in catalog}):
        category, source = stratum
        stratum_records = [record for record in catalog if record.stratum == stratum]
        stratum_train = [record for record in stratum_records if record.path in train_paths]
        total_microseconds = sum(
            record.duration_microseconds for record in stratum_records
        )
        train_microseconds = sum(
            record.duration_microseconds for record in stratum_train
        )
        target_microseconds = round(total_microseconds * train_ratio)
        group_durations = [
            group.duration_microseconds for group in _group_recordings(stratum_records)
        ]
        balance[f"{category}/{source}"] = {
            "target_train_microseconds": target_microseconds,
            "actual_train_microseconds": train_microseconds,
            "absolute_error_microseconds": abs(
                train_microseconds - target_microseconds
            ),
            "actual_train_ratio": train_microseconds / total_microseconds,
            "largest_group_microseconds": max(group_durations),
        }
    return balance


def _render_audio_list(recordings: Sequence[MusanRecording]) -> str:
    return "".join(f"{record.path}\n" for record in recordings)


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", text=True
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def build_musan_split(
    musan_root: str | Path,
    output_dir: str | Path,
    *,
    train_ratio: float = DEFAULT_MUSAN_TRAIN_RATIO,
    seed: int = DEFAULT_MUSAN_SPLIT_SEED,
) -> dict[str, Any]:
    """Create two disjoint allowlists and an auditable ``split.json``."""
    root = Path(musan_root).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve()
    artifact_names = {
        "train": "train_background.list",
        "eval": "eval_musan.list",
        "summary": "split.json",
    }
    if destination.exists():
        raise FileExistsError(
            f"Refusing to overwrite an existing MUSAN split directory: {destination}"
        )

    recordings = discover_musan_recordings(root)
    split = split_musan_recordings(recordings, train_ratio=train_ratio, seed=seed)
    summary: dict[str, Any] = {
        "schema_version": MUSAN_SPLIT_SCHEMA_VERSION,
        "dataset": "MUSAN",
        "musan_root": str(root),
        "seed": seed,
        "train_ratio_target": train_ratio,
        "policy": {
            "stratification": "category/source",
            "balance_unit": "decoded_audio_duration",
            "assignment": "duration_balanced_multi_start_with_local_repair",
            "grouping": {
                "music": (
                    "artist_when_ANNOTATIONS_has_at_least_two_distinct_artists_"
                    "else_recording"
                ),
                "noise": "recording",
                "speech": "recording",
            },
        },
        "catalog": _split_summary(recordings),
        "balance_by_stratum": _balance_summary(
            recordings, split.train, train_ratio=train_ratio
        ),
        "splits": {
            "train": {"list": artifact_names["train"], **_split_summary(split.train)},
            "eval": {"list": artifact_names["eval"], **_split_summary(split.eval)},
        },
    }
    total_duration = summary["catalog"]["duration_seconds"]
    summary["train_ratio_actual"] = (
        summary["splits"]["train"]["duration_seconds"] / total_duration
    )

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(dir=destination.parent, prefix=f".{destination.name}.")
    )
    try:
        _atomic_write_text(
            staging / artifact_names["train"], _render_audio_list(split.train)
        )
        _atomic_write_text(
            staging / artifact_names["eval"], _render_audio_list(split.eval)
        )
        _atomic_write_text(
            staging / artifact_names["summary"],
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        )
        os.replace(staging, destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return summary


__all__ = [
    "DEFAULT_MUSAN_SPLIT_SEED",
    "DEFAULT_MUSAN_TRAIN_RATIO",
    "MUSAN_CATEGORIES",
    "MUSAN_SPLIT_SCHEMA_VERSION",
    "MusanRecording",
    "MusanSplit",
    "build_musan_split",
    "discover_musan_recordings",
    "split_musan_recordings",
]
