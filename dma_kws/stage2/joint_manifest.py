"""Manifest checks and group-preserving splits for joint LoRA adaptation.

Audio paths and original/source audio paths are relative to ``data_root``.
``speaker_id`` is a person identifier within the real phase; synthetic voices
may occur in both splits. ``source_id`` / ``recording_id`` identify an original
recording within a phase, while path identifiers work across phases. Session
identifiers are local to (phase, speaker_id or voice_id), never global IDs.
Absent metadata cannot establish speaker or derivative provenance: callers
should provide it, or supply independently collected train/eval directories.
"""

from __future__ import annotations

import csv
import json
import random
from collections import Counter
from collections.abc import Mapping, Sequence
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from dma_kws.inference.manifest import load_audio_file_list
from dma_kws.stage2.adapt_paths import phase_manifest

_NAMES = ("real_train", "real_eval", "tts_train", "tts_eval")
_PATH_FIELDS = ("audio_path", "original_audio_path", "source_audio_path")
_ID_FIELDS = ("source_id", "recording_id", "original_recording_id")
_MISSING = {"", "nan", "none", "null", "<na>", "n/a", "na"}


def _value(value: Any) -> str:
    text = str(value).strip() if value is not None else ""
    return "" if text.casefold() in _MISSING else text


def _canonical(value: str | Path, base: Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def _identity_keys(row: Mapping[str, Any], phase: str, root: Path) -> set[tuple]:
    keys: set[tuple] = set()
    for field in _PATH_FIELDS:
        if value := _value(row.get(field)):
            keys.add(("audio/source path", str(_canonical(value, root))))
    for field in _ID_FIELDS:
        if value := _value(row.get(field)):
            keys.add((field, phase, value.casefold()))
    speaker = _value(row.get("speaker_id")).casefold()
    if phase == "real" and speaker:
        keys.add(("speaker_id", phase, speaker))
    owner = speaker or _value(row.get("voice_id")).casefold()
    if session := _value(row.get("session_id")):
        keys.add(("session_id", phase, owner, session.casefold()))
    return keys


def _label(row: Mapping[str, Any], location: str) -> int:
    try:
        label = Decimal(_value(row.get("label")))
    except InvalidOperation:
        label = Decimal("NaN")
    if not label.is_finite() or label not in (0, 1):
        raise ValueError(f"{location}: label must be 0 or 1, got {row.get('label')!r}")
    return int(label)


def validate_joint_rows(
    data_root: Path,
    rows_by_name: Mapping[str, Sequence[Mapping[str, Any]]],
) -> None:
    """Validate all four sources before preparation or model construction."""
    roots = Path(data_root).expanduser().resolve()
    assigned: dict[tuple, tuple[str, str]] = {}
    pronunciations: set[str] = set()
    for name in _NAMES:
        rows = rows_by_name.get(name, ())
        if not rows:
            raise ValueError(f"Joint adaptation requires non-empty {name}.csv")
        phase, split = name.split("_")
        labels: set[int] = set()
        for index, row in enumerate(rows, start=2):
            location = f"{name}.csv:{index}"
            for field in ("audio_path", "text"):
                if not _value(row.get(field)):
                    raise ValueError(f"{location}: {field} must not be empty or NaN")
            labels.add(_label(row, location))
            declared = _value(row.get("split")).casefold()
            if declared and declared != split:
                raise ValueError(f"{location}: split={declared!r}, expected {split!r}")
            declared_phase = _value(row.get("phase")).casefold()
            if declared_phase and declared_phase != phase:
                raise ValueError(f"{location}: phase={declared_phase!r}, expected {phase!r}")
            if value := _value(row.get("keyword_phonemes")):
                pronunciations.add(" ".join(value.split()))
            for key in _identity_keys(row, phase, roots):
                previous = assigned.get(key)
                if previous is not None and previous[0] != split:
                    raise ValueError(
                        f"Joint adaptation train/eval leakage via {key[0]} {key[1:]!r}: "
                        f"{previous[1]} and {location}. Supply speaker/source-disjoint "
                        "explicit splits or independently collected directories."
                    )
                assigned[key] = (split, location)
        if labels != {0, 1}:
            raise ValueError(f"{name}.csv must contain both positive and negative labels")
    if len(pronunciations) > 1:
        raise ValueError(
            "Joint manifests have inconsistent keyword_phonemes overrides: "
            f"{sorted(pronunciations)}"
        )


def validate_joint_manifests(data_root: Path) -> dict[str, Path]:
    """Read and audit the four prepared CSVs; cached features need no WAV files."""
    data_root = Path(data_root)
    paths = {
        name: phase_manifest(data_root, name.split("_")[0], split=name.split("_")[1])
        for name in _NAMES
    }
    rows_by_name = {}
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Joint adaptation source manifest not found: {path}")
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            missing = {"audio_path", "text", "label"} - set(reader.fieldnames or ())
            if missing:
                raise ValueError(f"{path} missing columns: {sorted(missing)}")
            rows_by_name[name] = list(reader)
    validate_joint_rows(data_root, rows_by_name)
    return paths


def split_joint_samples(
    samples: Sequence[Any], *, data_root: Path, eval_fraction: float, seed: int,
) -> dict[str, tuple[list[Any], list[Any]]]:
    """Keep linked speakers/recordings together, preserving declared splits.

    Samples follow ``prepare_adapt.AdaptSample``. The grouping is global so an
    original recording reused by another phase cannot cross train/eval.
    """
    if not 0 < eval_fraction < 1:
        raise ValueError("Joint eval_fraction must be strictly between 0 and 1")
    rows = [dict(sample.metadata, audio_path=sample.audio_path, text=sample.text,
                 label=sample.label, phase=sample.phase, split=sample.split)
            for sample in samples]
    if any(row["phase"] not in ("real", "tts") for row in rows):
        raise ValueError(
            "Joint source CSV must declare phase=real or phase=tts for every row; "
            "missing phase values cannot identify the source."
        )
    for phase in ("real", "tts"):
        phase_rows = [row for row in rows if row["phase"] == phase]
        if not phase_rows:
            raise ValueError(f"Joint preparation requires both real and tts sources; missing {phase}")
        declared = [_value(row.get("split")).casefold() for row in phase_rows]
        if any(declared) and not all(declared):
            raise ValueError(f"Joint {phase} mixes explicit and missing split values")
        if set(declared) - {"", "train", "eval"}:
            raise ValueError(f"Joint {phase} split must be train or eval")
    parent = list(range(len(rows)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    seen: dict[tuple, int] = {}
    for index, row in enumerate(rows):
        for key in _identity_keys(row, row["phase"], Path(data_root)):
            if key in seen:
                parent[find(index)] = find(seen[key])
            else:
                seen[key] = index
    groups: dict[int, list[int]] = {}
    for index in range(len(rows)):
        groups.setdefault(find(index), []).append(index)
    counts = {group: Counter((rows[i]["phase"], _label(rows[i], f"sample {i}"))
                             for i in indices)
              for group, indices in groups.items()}
    totals = sum(counts.values(), Counter())
    categories = {(phase, label) for phase in ("real", "tts") for label in (0, 1)}
    if any(totals[category] < 2 for category in categories):
        raise ValueError("Joint preparation needs train/eval positive and negative samples in each phase")
    fixed: dict[int, str] = {}
    for group, indices in groups.items():
        declared = {_value(rows[i]["split"]).casefold() for i in indices} - {""}
        if len(declared) > 1:
            raise ValueError("Joint explicit train/eval leakage in a speaker/source group")
        if declared:
            fixed[group] = declared.pop()

    initial_eval = {group for group, split in fixed.items() if split == "eval"}
    available = [group for group in groups if group not in fixed]
    rng = random.Random(seed)
    selected = None
    for _attempt in range(64):
        eval_groups = set(initial_eval)
        eval_counts = sum((counts[group] for group in eval_groups), Counter())
        candidates = list(available)
        rng.shuffle(candidates)
        while missing := {key for key in categories if eval_counts[key] == 0}:
            eligible = [group for group in candidates
                        if any(counts[group][key] for key in missing)
                        and all(eval_counts[key] + counts[group][key] < totals[key]
                                for key in categories)]
            if not eligible:
                break
            group = max(eligible, key=lambda g: sum(counts[g][key] > 0 for key in missing))
            eval_groups.add(group)
            eval_counts.update(counts[group])
            candidates.remove(group)
        if all(0 < eval_counts[key] < totals[key] for key in categories):
            selected = eval_groups
            # Approach each phase's requested size without discarding a class.
            for group in candidates:
                before = sum(abs(eval_counts[key] - eval_fraction * totals[key]) for key in categories)
                after = sum(abs(eval_counts[key] + counts[group][key] - eval_fraction * totals[key])
                            for key in categories)
                if after < before and all(eval_counts[key] + counts[group][key] < totals[key]
                                          for key in categories):
                    selected.add(group)
                    eval_counts.update(counts[group])
            break
    if selected is None:
        raise ValueError(
            "Joint preparation cannot form speaker/source-disjoint train/eval with both labels "
            "in each phase. Add independent recording groups or supply valid explicit splits."
        )
    result = {phase: ([], []) for phase in ("real", "tts")}
    named_rows: dict[str, list[dict]] = {name: [] for name in _NAMES}
    for index, sample in enumerate(samples):
        split = "eval" if find(index) in selected else "train"
        result[sample.phase][split == "eval"].append(sample)
        named_rows[f"{sample.phase}_{split}"].append(dict(rows[index], split=split))
    validate_joint_rows(data_root, named_rows)
    return result


def validate_background_eval_split(
    background_cfg: Mapping[str, Any], eval_audio_list: str | Path | None,
) -> None:
    """Reject MUSAN train/eval source overlap without opening audio or shards."""
    if not _value(eval_audio_list) or not background_cfg.get("enabled", False):
        return
    evaluation = set(load_audio_file_list(eval_audio_list))
    if not evaluation:
        return
    mode = str(background_cfg.get("mode", "online")).strip().casefold()
    relative_sources: set[tuple[str, ...]] = set()
    if mode == "online":
        training = set(load_audio_file_list(background_cfg.get("audio_list_path", "")))
    elif mode == "fbank_cache":
        manifest_path = Path(str(background_cfg.get("cache_manifest", ""))).expanduser()
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Background cache manifest not found: {manifest_path}")
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        if not isinstance(manifest, dict) or manifest.get("split_role") != "train":
            raise ValueError(f"Background cache must declare split_role=train: {manifest_path}")
        root = _value((manifest.get("split") or {}).get("musan_root"))
        spec = manifest.get("recordings")
        if not root or not isinstance(spec, dict) or not _value(spec.get("path")):
            raise ValueError(f"Background cache needs split.musan_root and recordings.path: {manifest_path}")
        recordings_path = manifest_path.parent / spec["path"]
        training = set()
        with recordings_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                relative = _value(record.get("relative_path")) or _value(record.get("source_id"))
                if not relative:
                    raise ValueError(f"{recordings_path}:{line_number} missing source path")
                training.add(_canonical(relative, _canonical(root, manifest_path.parent)))
                relative_path = Path(relative)
                if not relative_path.is_absolute() and ".." not in relative_path.parts:
                    relative_sources.add(relative_path.parts)
                entry = _value(record.get("list_entry"))
                if entry and Path(entry).expanduser().is_absolute():
                    training.add(_canonical(entry, manifest_path.parent))
        if not training:
            raise ValueError(f"Background cache recordings list is empty: {recordings_path}")
    else:
        raise ValueError(f"Unsupported background negative mode: {mode!r}")
    overlap = training & evaluation
    # Cache creation and evaluation may run on hosts with different MUSAN mount
    # points. Relative MUSAN recording IDs still identify the same source.
    source_depths = {len(parts) for parts in relative_sources}
    for path in evaluation:
        if any(path.parts[-depth:] in relative_sources for depth in source_depths):
            overlap.add(path)
    if overlap:
        raise ValueError(f"MUSAN train/eval source leakage: {sorted(overlap)[:5]}")
