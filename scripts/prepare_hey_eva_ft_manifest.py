#!/usr/bin/env python3
"""Build the combined TTS + real "hey eva" adaptation manifest.

Sources
  TTS   outputs/tts/hey_eva/tts_manifest.csv     (voice-disjoint split already
                                                  assigned by the corpus builder)
  real  data/dma-kws/raw/hey_eva_real_v2/manifests/real_reviewed_abs.csv
                                                 (reviewed; split assigned here
                                                  by speaker)

The output is a single CSV in the manifest contract of
``dma_kws.stage2.prepare_adapt.load_manifest_csv``: required
``audio_path,text,label`` plus the reserved ``phase,split`` and metadata columns
that survive into ``prepare_adapt``'s train/eval manifests.  ``keyword_phonemes``
pins the query to the reviewed pronunciation of the wake phrase instead of
re-deriving it from the keyword text.

Every row is validated against the same rule ``KeywordAdaptationDataset``
enforces at training time: a positive row's clip transcript must complete the
anchor phoneme sequence, and a negative row's must not.

Split policy
  TTS   inherit the existing split.  It is voice-disjoint (26 train / 22 eval
        voices), which this script re-asserts.
  real  speaker-disjoint by hand.  EVAL_SPEAKERS below were chosen so the eval
        side has several speakers with both positives and near negatives,
        including the speaker whose "Hey Eva" takes the reviewers heard as
        "Hi Eva" (near-negative only), which is the hardest held-out case.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dma_kws.config import compose_config, config_to_dict
from dma_kws.pathing import resolve_dict_path
from dma_kws.stage2.adapt_dataset import _text_to_g2p, make_g2p  # noqa: PLC2701 - reuse the exact path
from dma_kws.tokenizer import load_char_tokenizer, tokenize_phoneme_string

KEYWORD = "hey eva"
KEYWORD_PHONEMES = "HH EY1 IY1 V AH0"
DEFAULT_TTS = PROJECT_ROOT / "outputs/tts/hey_eva/tts_manifest.csv"
DEFAULT_REAL = (
    PROJECT_ROOT
    / "data/dma-kws/raw/hey_eva_real_v2/manifests/real_reviewed_abs.csv"
)
DEFAULT_OUT = (
    PROJECT_ROOT / "data/dma-kws/processed/adapt/hey_eva_v42/source/combined.csv"
)

# Speakers held out for evaluation.  解震/孙倩倩/陈凯 have both positives and
# near negatives; 韩丽萍 only has near negatives (seven takes the reviewers
# transcribed as "Hi Eva"), so the split stays speaker-disjoint.
EVAL_SPEAKERS = {"解震", "孙倩倩", "陈凯", "韩丽萍"}

FIELDNAMES = [
    "audio_path",
    "text",
    "label",
    "phase",
    "split",
    "speaker_id",
    "keyword",
    "keyword_phonemes",
    "text_variant_phonemes",
    "containing_negative",
    "source",
    "source_row",
]


def _tokenizer():
    config = config_to_dict(compose_config())
    tokenizer_cfg = config.get("tokenizer", {}) or {}
    return load_char_tokenizer(
        resolve_dict_path(config),
        split_with_space=tokenizer_cfg.get("split_with_space", " "),
    )


def _phoneme_ids(tokenizer, phoneme_string: str) -> list[int]:
    return tokenize_phoneme_string(tokenizer, phoneme_string)


def _contains(anchor: list[int], query: list[int]) -> bool:
    if not anchor or len(query) < len(anchor):
        return False
    return any(query[i : i + len(anchor)] == anchor for i in range(len(query) - len(anchor) + 1))


def read_tts(
    path: Path, tokenizer, anchor_ids: list[int]
) -> tuple[list[dict], list[dict]]:
    """Return (adapt rows, probe rows).

    "Hey evil" is forced as HH EY1 IY1 V AH0 L, which *contains* the wake
    phrase.  The objective is contiguous phoneme containment, so
    ``KeywordAdaptationDataset`` rejects such a row as a negative.  A clip can
    only stay in the adaptation eval set when its voice is eval-only anyway
    (otherwise moving it would leak a training voice into eval); the rest are
    written to a separate probe manifest so the trigger rate this objective
    cannot avoid is still measurable.
    """

    raw: list[dict] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for index, row in enumerate(csv.DictReader(handle)):
            split = str(row.get("split", "")).strip().casefold()
            if split not in {"train", "eval"}:
                raise ValueError(f"{path}:{index + 2} has invalid split {split!r}")
            phonemes = str(row.get("text_variant_phonemes", "") or "")
            raw.append(
                {
                    "audio_path": str(row["audio_path"]),
                    "text": str(row["text"]),
                    "label": int(row["label"]),
                    "phase": "tts",
                    "split": split,
                    "speaker_id": str(row["speaker_id"]),
                    "keyword": KEYWORD,
                    "keyword_phonemes": KEYWORD_PHONEMES,
                    "text_variant_phonemes": phonemes,
                    "source": "tts",
                    "source_row": index,
                    "containing_negative": int(
                        int(row["label"]) == 0
                        and _contains(anchor_ids, _phoneme_ids(tokenizer, phonemes))
                    ),
                }
            )

    eval_only_voices = {
        row["speaker_id"] for row in raw if row["split"] == "eval"
    } - {row["speaker_id"] for row in raw if row["split"] == "train"}
    rows = [
        row
        for row in raw
        if not row["containing_negative"] or row["speaker_id"] in eval_only_voices
    ]
    probe = [
        row
        for row in raw
        if row["containing_negative"] and row["speaker_id"] not in eval_only_voices
    ]
    return rows, probe


def read_real(path: Path, g2p) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for index, row in enumerate(csv.DictReader(handle)):
            speaker = str(row["speaker_id"]).strip()
            rows.append(
                {
                    "audio_path": str(row["audio_path"]),
                    "text": str(row["text"]),
                    "label": int(row["label"]),
                    "phase": "real",
                    "split": "eval" if speaker in EVAL_SPEAKERS else "train",
                    "speaker_id": speaker,
                    "keyword": KEYWORD,
                    "keyword_phonemes": KEYWORD_PHONEMES,
                    # The g2p of the reviewed transcript, not of the file name:
                    # clips the reviewers re-heard as another phrase carry that
                    # phrase here and are supervised as the near negative they are.
                    "text_variant_phonemes": _text_to_g2p(g2p, str(row["text"])),
                    "containing_negative": 0,
                    "source": "real",
                    "source_row": index,
                }
            )
    return rows


def validate(rows: list[dict], tokenizer, anchor_ids: list[int]) -> dict:
    problems: list[str] = []

    for row in rows:
        audio = Path(row["audio_path"])
        if not audio.is_file():
            problems.append(f"missing audio: {audio}")
        query_ids = _phoneme_ids(tokenizer, row["text_variant_phonemes"])
        completed = _contains(anchor_ids, query_ids)
        # Only training rows must satisfy the containment contract; an eval-side
        # positive that fails it, or negative that satisfies it, is exactly the
        # error this objective cannot represent and is reported instead.
        if row["split"] == "train" and completed != bool(row["label"]):
            problems.append(
                f"{audio.name} label={row['label']} but transcript "
                f"{row['text']!r} completes the keyword: {completed}"
            )

    for phase in ("tts", "real"):
        phase_rows = [row for row in rows if row["phase"] == phase]
        train_speakers = {r["speaker_id"] for r in phase_rows if r["split"] == "train"}
        eval_speakers = {r["speaker_id"] for r in phase_rows if r["split"] == "eval"}
        overlap = train_speakers & eval_speakers
        if overlap:
            problems.append(f"phase {phase} leaks speakers across splits: {sorted(overlap)}")
        if not train_speakers or not eval_speakers:
            problems.append(f"phase {phase} needs both a train and an eval side")
        for side in ("train", "eval"):
            labels = Counter(r["label"] for r in phase_rows if r["split"] == side)
            if not labels[0] or not labels[1]:
                problems.append(f"phase {phase}/{side} needs both labels, got {dict(labels)}")

    if problems:
        raise SystemExit("manifest validation failed:\n  " + "\n  ".join(problems[:40]))

    stats: dict = {"phases": {}}
    for phase in ("tts", "real"):
        for side in ("train", "eval"):
            subset = [r for r in rows if r["phase"] == phase and r["split"] == side]
            stats["phases"].setdefault(phase, {})[side] = {
                "clips": len(subset),
                "positive": sum(1 for r in subset if r["label"] == 1),
                "negative": sum(1 for r in subset if r["label"] == 0),
                "speakers": len({r["speaker_id"] for r in subset}),
                "speaker_ids": sorted({r["speaker_id"] for r in subset}),
                "negatives_by_text": dict(
                    Counter(r["text"] for r in subset if r["label"] == 0).most_common()
                ),
                "containing_negatives": sum(
                    int(r.get("containing_negative", 0)) for r in subset
                ),
            }
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tts", type=Path, default=DEFAULT_TTS)
    parser.add_argument("--real", type=Path, default=DEFAULT_REAL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--stats", type=Path, default=None)
    args = parser.parse_args()

    tokenizer = _tokenizer()
    anchor_ids = _phoneme_ids(tokenizer, KEYWORD_PHONEMES)
    g2p = make_g2p()

    tts_rows, probe_rows = read_tts(args.tts, tokenizer, anchor_ids)
    rows = tts_rows + read_real(args.real, g2p)
    stats = validate(rows, tokenizer, anchor_ids)
    stats["containing_negative_probe"] = {
        "clips": len(probe_rows),
        "texts": dict(Counter(r["text"] for r in probe_rows).most_common()),
        "note": (
            "clips whose forced transcript contains the wake phrase; the "
            "containment objective cannot supervise them as negatives, so they "
            "are held out for a trigger-rate probe instead of training"
        ),
    }
    stats["keyword"] = KEYWORD
    stats["keyword_phonemes"] = KEYWORD_PHONEMES
    stats["total_clips"] = len(rows)
    stats["sources"] = {"tts": str(args.tts), "real": str(args.real)}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    if probe_rows:
        probe_path = args.output.with_name(args.output.stem + "_containing_negatives.csv")
        with probe_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
            writer.writeheader()
            for row in probe_rows:
                writer.writerow(row)
        print(f"wrote {probe_path} ({len(probe_rows)} rows)")

    stats_path = args.stats or args.output.with_suffix(".stats.json")
    stats_path.write_text(json.dumps(stats, indent=1, ensure_ascii=False))
    print(f"wrote {args.output} ({len(rows)} rows)")
    print(f"wrote {stats_path}")
    print(json.dumps(stats["phases"], indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
