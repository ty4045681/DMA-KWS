#!/usr/bin/env python3
"""Check every generated clip against the phones it was asked to say.

Two signals run per clip:

1. duration and peak amplitude, which catch silent or truncated output;
2. a phoneme CTC decode from facebook/wav2vec2-lv-60-espeak-cv-ft.

The decisive check is the vowel before the first /v/. That vowel separates
"Eva" from "Ava" in this corpus, and text ASR cannot see it: Whisper reads a
forced /eɪvə/ as "Eva" as readily as /iːvə/. A decode with no /v/ at all fails,
which is what catches a dictionary rule that drops a whole word.

    python3 scripts/verify_hey_eva_tts.py --run-root outputs/tts/hey_eva
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence  # noqa: F401

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.generate_hey_eva_tts import (
    audio_health,
    load_plan,
    phone_base,
    word_tokens,
)

DEFAULT_MODEL = "facebook/wav2vec2-lv-60-espeak-cv-ft"
# The CTC model insists on 16 kHz. ElevenLabs already returns 16 kHz, Piper
# returns 22.05 kHz, and Kokoro returns 24 kHz.
TARGET_SAMPLE_RATE = 16000
VERIFICATION_FIELDS = (
    "key",
    "provider",
    "voice_id",
    "voice_name",
    "text",
    "expected_phonemes",
    "expected_vowel",
    "decoded",
    "decoded_vowel",
    "duration",
    "peak",
    "transcript",
    "verdict",
    "reason",
)
# The CTC model merges EY and AY into aɪ or eː, and it renders EH as ɛ or e.
# So the decode side classifies into coarse groups, and each expected ARPAbet
# family lists the groups that satisfy it. The one distinction that stays sharp
# is i-like against everything else, which is exactly what separates Eva from
# Ava.
ACCEPTED_GROUPS = {
    "IY": {"IY"},
    "EY": {"OPEN"},
    "AY": {"OPEN"},
    "EH": {"EH", "OPEN"},
    "AH": {"AH", "OPEN"},
    "ER": {"AH", "OPEN", "EH"},
    "AA": {"OPEN", "AH"},
    "AE": {"OPEN", "EH"},
    "AO": {"OPEN"},
    "OW": {"OPEN"},
    "UH": {"AH", "OPEN"},
    # The CTC model renders /uː/ as u, ʉ, oː, or ɔ depending on the voice, so
    # UW and OPEN overlap. The human review covers the fine Google/Goggle call.
    "UW": {"UW", "OPEN"},
}
# ARPAbet vowel bases that this corpus can produce. Anything else decodes to
# "unknown" and lands in review instead of passing silently.
ARPABET_FAMILY = {
    "IY": "IY",
    "IH": "IY",
    "EY": "EY",
    "AY": "AY",
    "EH": "EH",
    "AH": "AH",
    "AX": "AH",
    "AA": "AA",
    "AE": "AE",
    "AO": "AO",
    "OW": "OW",
    "UH": "UH",
    "UW": "UW",
    "ER": "ER",
}
VOWEL_CHARS = set("aeiouyæɛɪʊʌəɜɝɔɑɒɐᵻɨøœɚʉɯɤɘɵɞɶ")
LENGTH_MARKS = set("ːˑ")
# Two-symbol nuclei that must stay together. A run like "aɪeː" holds two
# nuclei, and the one that matters is the last.
NUCLEI = frozenset(
    {"aɪ", "eɪ", "ɔɪ", "aʊ", "oʊ", "əʊ", "ɪə", "eə", "ʊə", "ɛː", "iː", "uː",
     "ɑː", "ɔː", "ɜː", "aː", "eː", "oː"}
)


def probe_position(
    phones: Sequence[str], keyword_phones: Sequence[str] = ()
) -> tuple[int, str, str]:
    """Return (index, family, token) for the vowel that decides the phrase.

    Anchor on the keyword: the probe is the first place the phrase differs from
    it, moved forward to the next vowel. A phrase that matches the keyword
    (a positive) probes its last stressed vowel instead. That keeps the check on
    the phoneme that actually distinguishes a near miss.
    """

    difference: int | None = None
    if keyword_phones:
        for index in range(max(len(phones), len(keyword_phones))):
            left = phones[index] if index < len(phones) else None
            right = keyword_phones[index] if index < len(keyword_phones) else None
            if left != right:
                difference = index
                break
    if difference is not None:
        for index in range(difference, len(phones)):
            family = ARPABET_FAMILY.get(phone_base(phones[index]))
            if family:
                return index, family, phones[index]
    for index in range(len(phones) - 1, -1, -1):
        phone = phones[index]
        family = ARPABET_FAMILY.get(phone_base(phone))
        if family and any(char in "12" for char in phone):
            return index, family, phone
    for index in range(len(phones) - 1, -1, -1):
        family = ARPABET_FAMILY.get(phone_base(phones[index]))
        if family:
            return index, family, phones[index]
    return -1, "unknown", ""


def probe_vowel(
    phones: Sequence[str], keyword_phones: Sequence[str] = ()
) -> tuple[str, str]:
    """Return (family, token) for the vowel that decides the phrase."""

    _index, family, token = probe_position(phones, keyword_phones)
    return family, token


def vowel_ordinal(phones: Sequence[str], position: int) -> int:
    """1-based position of the probe vowel among the vowels of the phrase."""

    return sum(
        1 for phone in phones[: position + 1] if phone_base(phone) in ARPABET_FAMILY
    )


def vowel_before(decoded: str, marker: str = "v") -> str:
    """Return the vowel run that ends just before the first marker character."""

    cut = decoded.find(marker)
    head = decoded if cut < 0 else decoded[:cut]
    run: list[str] = []
    for char in reversed(head):
        if char in VOWEL_CHARS or char in LENGTH_MARKS:
            run.append(char)
        elif run:
            break
    return "".join(reversed(run))


def last_nucleus(run: str) -> str:
    """Return the final vowel nucleus of a vowel run.

    A decode without a word gap can glue two nuclei together, as in "aɪeː" for
    "Hey Ava". Splitting off the last nucleus keeps the check on the vowel that
    distinguishes the words.
    """

    if not run:
        return ""
    if run[-1] in LENGTH_MARKS and len(run) >= 2 and run[-2] in VOWEL_CHARS:
        return run[-2:]
    if len(run) >= 2 and run[-2:] in NUCLEI:
        return run[-2:]
    return run[-1]


def split_nuclei(run: str) -> list[str]:
    """Split a vowel run into nuclei. "aɪeː" becomes ["aɪ", "eː"]."""

    nuclei: list[str] = []
    current = ""
    for char in run:
        if not current:
            current = char
            continue
        candidate = current + char
        if char in LENGTH_MARKS or candidate in NUCLEI:
            current = candidate
            continue
        nuclei.append(current)
        current = char
    if current:
        nuclei.append(current)
    return nuclei


def vowel_nuclei(decoded: str) -> list[str]:
    """Every vowel nucleus of a decoded string, in order."""

    nuclei: list[str] = []
    run = ""
    for char in decoded:
        if char in VOWEL_CHARS or char in LENGTH_MARKS:
            run += char
        elif run:
            nuclei.extend(split_nuclei(run))
            run = ""
    if run:
        nuclei.extend(split_nuclei(run))
    return nuclei


def classify_vowel(symbol: str) -> str:
    """Map a decoded vowel to one coarse group: IY, EH, OPEN, UW, or AH."""

    if not symbol:
        return "unknown"
    if any(char in symbol for char in "iɪɨᵻ") and not any(
        char in symbol for char in "aeɛ"
    ):
        return "IY"
    if "ɛ" in symbol:
        return "EH"
    if any(char in symbol for char in "aeoɔæɑɒ"):
        return "OPEN"
    if any(char in symbol for char in "uʉɯøœɶ"):
        return "UW"
    if any(char in symbol for char in "əʌɐɜʊɚ"):
        return "AH"
    return "unknown"


def judge(
    expected_phonemes: str,
    decoded: str,
    keyword_phonemes: str = "",
    markers: Sequence[str] = (),
    strict: bool = True,
    strict_probe: bool = True,
) -> tuple[str, str, str, str]:
    """Return (verdict, reason, expected_vowel, decoded_vowel).

    strict=False for phrases shorter than the keyword, such as a truncated
    "Hey": those are supposed to be missing the second word.

    strict_probe=False for negatives. A near miss only has to avoid the
    keyword, so a vowel the decoder reports differently is a note, not a
    failure. Positives keep the strict check: a mispronounced wake word would
    teach the model the wrong target.
    """

    phones = expected_phonemes.split()
    keyword = keyword_phonemes.split() if keyword_phonemes else []
    position, expected_family, expected_token = probe_position(phones, keyword)
    if not decoded.strip():
        return "fail", "no phonemes decoded", expected_token, ""
    if strict and markers and not any(marker in decoded for marker in markers):
        return (
            "fail",
            "the second word's consonant is missing from the decode",
            expected_token,
            "",
        )
    expected_count = sum(1 for phone in phones if phone_base(phone) in ARPABET_FAMILY)
    found_all = vowel_nuclei(decoded)
    if strict and len(found_all) < expected_count:
        return (
            "fail",
            f"decoded {len(found_all)} of {expected_count} vowels, so a word is missing",
            expected_token,
            "",
        )
    if not found_all:
        return (
            "review",
            "the decode carries no vowel nuclei",
            expected_token,
            "",
        )
    ordinal = vowel_ordinal(phones, position)
    if ordinal < 1:
        return (
            "review",
            "the probe vowel falls outside the decoded vowels",
            expected_token,
            "",
        )
    # Align by position ratio, not by raw index: the decoder drops or inserts
    # nuclei often enough that an off-by-one would land on the wrong vowel.
    ratio = ordinal / max(expected_count, 1)
    index = min(len(found_all) - 1, max(0, round(ratio * len(found_all)) - 1))
    found = found_all[index]
    group = classify_vowel(found)
    if group == "unknown":
        return (
            "review",
            f"probe vowel decoded as {found!r}, outside the known classes",
            expected_token,
            found,
        )
    accepted = ACCEPTED_GROUPS.get(expected_family)
    if accepted is None:
        return (
            "review",
            f"no acceptance rule for {expected_token} ({expected_family})",
            expected_token,
            found,
        )
    if group in accepted:
        return "pass", f"{found!r} matches {expected_token}", expected_token, found
    return (
        "fail" if strict_probe else "review",
        f"probe vowel is {found!r} ({group}), expected {expected_token} "
        f"({expected_family})",
        expected_token,
        found,
    )


# IPA characters the CTC model uses for each ARPAbet consonant.
CONSONANT_MARKERS = {
    "B": ("b",),
    "CH": ("ʧ", "tʃ"),
    "D": ("d",),
    "DH": ("ð",),
    "F": ("f",),
    "G": ("ɡ", "g"),
    "HH": ("h",),
    "JH": ("ʤ", "dʒ"),
    "K": ("k",),
    "L": ("l",),
    "M": ("m",),
    "N": ("n",),
    "NG": ("ŋ",),
    "P": ("p",),
    "R": ("ɹ", "r"),
    "S": ("s",),
    "SH": ("ʃ",),
    "T": ("t",),
    "TH": ("θ",),
    "V": ("v",),
    "W": ("w",),
    "Y": ("j",),
    "Z": ("z",),
}


def keyword_markers(plan) -> tuple[str, ...]:
    """IPA characters for the first consonant of the keyword's second word.

    A decode that lost the second word will not contain them.
    """

    tokens = word_tokens(plan.keyword)
    if len(tokens) < 2:
        return ()
    for phone in plan.words[tokens[1]]:
        markers = CONSONANT_MARKERS.get(phone_base(phone))
        if markers:
            return markers
    return ()


def transcript_words(transcript: str) -> list[str]:
    """Words of at least two letters, for the whisper gate."""

    return [word for word in re.findall(r"[A-Za-z']+", transcript) if len(word) >= 2]


def whisper_gate(transcript: str, expected_words: int) -> tuple[bool, str]:
    """Phrases of two or more words need two or more words in the transcript."""

    if expected_words < 2:
        return True, ""
    words = transcript_words(transcript)
    if len(words) >= 2:
        return True, ""
    return False, f"transcript {transcript.strip()!r} has fewer than two words"


def read_keys(path: Path) -> set[str]:
    """Keys listed in a JSONL file, such as rejected.jsonl."""

    keys: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            key = str(json.loads(line).get("key") or "")
            if key:
                keys.add(key)
    return keys


def read_report(path: Path) -> dict[str, dict[str, Any]]:
    """Existing verification.csv rows, keyed by clip."""

    if not path.is_file():
        return {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return {row["key"]: dict(row) for row in csv.DictReader(handle)}


def merge_results(
    previous: Mapping[str, Mapping[str, Any]],
    results: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Replace the rows that were just verified and keep the rest."""

    updated = {str(row["key"]): dict(row) for row in results}
    merged = [updated.pop(str(row["key"]), dict(row)) for row in previous.values()]
    merged.extend(updated.values())
    return merged


def read_index(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise SystemExit(f"index not found: {path}")
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def expected_phonemes_for(row: Mapping[str, Any], plan) -> str:
    phones = str(row.get("phonemes") or "")
    if phones:
        return phones
    text = str(row.get("text") or "")
    parts: list[str] = []
    for token in word_tokens(text):
        parts.extend(plan.words[token])
    return " ".join(parts)


def resample_to_target(audio, rate: int, target: int = TARGET_SAMPLE_RATE):
    """Resample mono audio for the CTC model. A no-op at the target rate."""

    if int(rate) == target:
        return audio
    from math import gcd

    from scipy.signal import resample_poly

    divisor = gcd(int(rate), target)
    return resample_poly(audio, target // divisor, int(rate) // divisor)


def resolve_device(requested: str) -> str:
    """Pick the torch device. "auto" prefers CUDA when torch can use it."""

    if requested != "auto":
        return requested
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def load_decoder(model_name: str, device: str = "cpu"):
    """Load the phoneme CTC model and return a decode function."""

    import soundfile as sf
    import torch
    from huggingface_hub import hf_hub_download
    from transformers import AutoFeatureExtractor, Wav2Vec2ForCTC

    extractor = AutoFeatureExtractor.from_pretrained(model_name)
    vocabulary = json.load(open(hf_hub_download(model_name, "vocab.json")))
    inverse = {index: token for token, index in vocabulary.items()}
    model = Wav2Vec2ForCTC.from_pretrained(model_name).eval().to(device)

    def decode(path: Path) -> str:
        audio, rate = sf.read(str(path))
        audio = resample_to_target(audio, rate)
        inputs = extractor(audio, sampling_rate=TARGET_SAMPLE_RATE, return_tensors="pt")
        with torch.no_grad():
            logits = model(inputs.input_values.to(device)).logits
        ids = logits.argmax(dim=-1)[0].tolist()
        tokens: list[str] = []
        previous = None
        for index in ids:
            if index == previous:
                continue
            token = inverse.get(index, "")
            if token and token not in ("<pad>", "<s>", "</s>"):
                tokens.append(" " if token in ("|", " ") else token)
            previous = index
        return "".join(tokens).strip()

    return decode


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            print(json.dumps(row, ensure_ascii=False, sort_keys=True), file=handle)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=PROJECT_ROOT / "outputs/tts/hey_eva")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/tts/hey_eva.yaml")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--only-keys",
        type=Path,
        help="verify only the keys listed in this JSONL file",
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="merge this pass into the existing verification.csv",
    )
    parser.add_argument("--skip-phonemes", action="store_true", help="only run the cheap gate")
    parser.add_argument(
        "--whisper",
        action="store_true",
        help="also require two transcript words for two-word phrases",
    )
    parser.add_argument("--whisper-model", default="base.en")
    parser.add_argument(
        "--device",
        default="auto",
        help="auto, cpu, or cuda for both models (default: auto)",
    )
    parser.add_argument(
        "--whisper-provider",
        action="append",
        help="run the whisper gate only for these providers (repeatable)",
    )
    parser.add_argument(
        "--prune-missing",
        action="store_true",
        help="drop index rows whose audio file is gone",
    )
    parser.add_argument("--min-duration", type=float, default=0.25)
    parser.add_argument("--min-peak", type=float, default=0.02)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_root = Path(args.run_root).expanduser().resolve()
    plan = load_plan(args.config)
    rows = read_index(run_root / "index.jsonl")
    if args.limit:
        rows = rows[: args.limit]

    present = [row for row in rows if (run_root / str(row["audio_path"])).is_file()]
    if args.only_keys is not None:
        wanted = read_keys(args.only_keys)
        present = [row for row in present if str(row["key"]) in wanted]
        print(f"verifying only the {len(present)} keys from {args.only_keys}", flush=True)
    missing = len(rows) - len(present) if args.only_keys is None else 0
    if missing and args.prune_missing:
        index_path = run_root / "index.jsonl"
        write_jsonl(index_path, present)
    if missing:
        print(f"skipping {missing} index rows whose audio is gone", flush=True)

    markers = keyword_markers(plan)
    keyword_tokens = word_tokens(plan.keyword)
    report = run_root / "verification.csv"
    previous = read_report(report) if args.merge else {}

    def flush(partial: Sequence[Mapping[str, Any]]) -> None:
        """Write the report so far, so a crash does not lose the whole pass."""

        rows = merge_results(previous, partial) if args.merge else list(partial)
        with report.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(VERIFICATION_FIELDS))
            writer.writeheader()
            writer.writerows(rows)

    device = resolve_device(args.device)
    print(f"device: {device}", flush=True)
    decode = None if args.skip_phonemes else load_decoder(args.model, device)
    transcriber = None
    if args.whisper:
        import whisper

        speech_model = whisper.load_model(args.whisper_model, device=device)

        def transcriber(path: Path) -> str:
            return speech_model.transcribe(
                str(path), language="en", temperature=0
            )["text"]

    results: list[dict[str, Any]] = []
    for position, row in enumerate(present, start=1):
        path = run_root / str(row["audio_path"])
        payload = path.read_bytes()
        duration, peak = audio_health(payload)
        expected = expected_phonemes_for(row, plan)
        recorded = str(row.get("sha256") or "")
        actual = hashlib.sha256(payload).hexdigest()
        if recorded and recorded != actual:
            verdict, reason, expected_vowel, decoded, found = (
                "fail",
                "the file does not match the hash recorded in index.jsonl",
                "",
                "",
                "",
            )
        elif duration < args.min_duration or peak < args.min_peak:
            verdict, reason, expected_vowel, decoded, found = (
                "fail",
                f"silent or too short (duration={duration:.3f}s peak={peak:.3f})",
                "",
                "",
                "",
            )
        else:
            decoded = decode(path) if decode else ""
            phrase_tokens = word_tokens(str(row["text"]))
            # The missing-word guard only makes sense when the phrase carries
            # the keyword's second word. "Hey Eric" is not supposed to have the
            # /v/ of "eva".
            row_markers = (
                markers
                if len(keyword_tokens) >= 2
                and len(phrase_tokens) >= len(keyword_tokens)
                and phrase_tokens[1] == keyword_tokens[1]
                else ()
            )
            verdict, reason, expected_vowel, found = judge(
                expected,
                decoded,
                plan.keyword_phonemes,
                row_markers,
                strict=len(phrase_tokens) >= len(keyword_tokens),
                strict_probe=int(row["label"]) == 1,
            )
        transcript = ""
        gate_providers = {name.casefold() for name in args.whisper_provider or ()}
        if transcriber is not None and (
            not gate_providers or str(row["provider"]).casefold() in gate_providers
        ):
            transcript = transcriber(path).strip()
            if verdict != "fail":
                passed, why = whisper_gate(
                    transcript, len(word_tokens(str(row["text"])))
                )
                if not passed:
                    verdict, reason = "fail", why
        results.append(
            {
                "key": row["key"],
                "provider": row["provider"],
                "voice_id": row["voice_id"],
                "voice_name": row["voice_name"],
                "text": row["text"],
                "expected_phonemes": expected,
                "expected_vowel": expected_vowel,
                "decoded": decoded,
                "decoded_vowel": found,
                "duration": round(duration, 3),
                "peak": round(peak, 3),
                "transcript": transcript,
                "verdict": verdict,
                "reason": reason,
            }
        )
        print(
            f'[{position}/{len(present)}] {row["voice_name"]:8s} {row["text"]:10s} '
            f"{verdict:6s} {decoded!r} {reason}",
            flush=True,
        )
        if position % 200 == 0:
            flush(results)

    flush(results)
    if args.merge:
        results = merge_results(previous, results)

    failed = [row for row in results if row["verdict"] == "fail"]
    review = [row for row in results if row["verdict"] == "review"]
    write_jsonl(run_root / "rejected.jsonl", failed)
    write_jsonl(run_root / "needs_review.jsonl", review)

    summary = {
        "clips": len(results),
        "pass": len(results) - len(failed) - len(review),
        "review": len(review),
        "fail": len(failed),
        "missing": missing,
        "report": str(report),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
