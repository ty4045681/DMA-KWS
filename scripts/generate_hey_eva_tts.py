#!/usr/bin/env python3
"""Generate Hey Eva wake-word clips from the providers named in a YAML plan.

One word table fixes the ARPAbet phones for every word the corpus uses, so each
phrase resolves to one phone string. The script writes the clips and the
source.csv that scripts/prepare_tts_lora_manifest.py reads.

Examples:
    python3 scripts/generate_hey_eva_tts.py --config configs/tts/hey_eva.yaml --dry-run
    ELEVENLABS_API_KEY=... python3 scripts/generate_hey_eva_tts.py --config configs/tts/hey_eva.yaml --limit 8
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import re
import shlex
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

import numpy as np
import soundfile as sf
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dma_kws.g2p import HEY_EVA_PHONEMES

KEYWORD_PHONEMES = " ".join(HEY_EVA_PHONEMES)
DEFAULT_KEYWORD = "hey eva"
KEYWORD_TOKENS = tuple(DEFAULT_KEYWORD.split())
SOURCE_FIELDS = (
    "audio_path",
    "keyword",
    "label",
    "voice_id",
    "voice_name",
    "text_variant",
    "sha256",
    "keyword_phonemes",
    "text_variant_phonemes",
    "speed",
    "volume",
)
# Values accepted by POST /v1/text-to-speech/{voice_id}?output_format=...
# See https://elevenlabs.io/docs/api-reference/text-to-speech/convert
OUTPUT_FORMATS = frozenset(
    {
        "alaw_8000",
        "mp3_22050_32",
        "mp3_24000_48",
        "mp3_44100_128",
        "mp3_44100_192",
        "mp3_44100_32",
        "mp3_44100_64",
        "mp3_44100_96",
        "opus_48000_128",
        "opus_48000_192",
        "opus_48000_32",
        "opus_48000_64",
        "opus_48000_96",
        "pcm_16000",
        "pcm_22050",
        "pcm_24000",
        "pcm_32000",
        "pcm_44100",
        "pcm_48000",
        "pcm_8000",
        "ulaw_8000",
        "wav_16000",
        "wav_22050",
        "wav_24000",
        "wav_32000",
        "wav_44100",
        "wav_48000",
        "wav_8000",
    }
)
WORD_RE = re.compile(r"[A-Za-z]+")


class ProviderError(RuntimeError):
    """A provider rejected a request or returned an unusable payload."""


class Provider(Protocol):
    """What run_jobs needs from a speech engine."""

    alias: str

    @property
    def directory(self) -> str: ...

    def prepare(
        self,
        plan: Plan,
        jobs: Sequence[Job],
        root: Path,
        log: Callable[[str], None],
    ) -> None: ...

    def synthesize(self, job: Job, destination: Path) -> int: ...


class Transport(Protocol):
    def __call__(
        self,
        method: str,
        url: str,
        headers: Mapping[str, str],
        body: bytes | None,
        timeout: float = ...,
    ) -> tuple[int, bytes]: ...


def urllib_transport(
    method: str,
    url: str,
    headers: Mapping[str, str],
    body: bytes | None,
    timeout: float = 60.0,
) -> tuple[int, bytes]:
    """Send one HTTP request and return (status, body). Never raises on 4xx."""

    request = urllib.request.Request(
        url, data=body, method=method, headers=dict(headers)
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return int(response.status), response.read()
    except urllib.error.HTTPError as exc:
        return int(exc.code), exc.read()


def audio_health(payload: bytes) -> tuple[float, float]:
    """Return (duration_seconds, peak_amplitude) for a WAV payload."""

    audio, rate = sf.read(io.BytesIO(payload))
    if not rate or audio.size == 0:
        return 0.0, 0.0
    return audio.size / float(rate), float(np.max(np.abs(audio)))


@dataclass(frozen=True)
class Voice:
    voice_id: str
    name: str
    accent: str = ""
    gender: str = ""


@dataclass(frozen=True)
class Phrase:
    text: str
    label: int
    phonemes: str
    characters: int
    # True when every word of the phrase carries a phoneme rule. Such a phrase
    # would arrive at the provider as phonemes only, which eleven_v3 rejects,
    # so it uses the alias dictionary instead.
    use_alias: bool = False


@dataclass(frozen=True)
class Job:
    provider: str
    voice: Voice
    phrase: Phrase
    speed: float = 1.0
    volume: float = 1.0

    @property
    def key(self) -> str:
        parts = [
            self.provider,
            self.voice.voice_id,
            self.phrase.text,
            self.phrase.phonemes,
        ]
        # Defaults keep the original key, so an existing index stays valid when
        # an axis is added later.
        if self.speed != 1.0:
            parts.append(f"speed={self.speed}")
        if self.volume != 1.0:
            parts.append(f"volume={self.volume}")
        return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:16]

    @property
    def filename(self) -> str:
        slug = re.sub(r"[^a-z0-9]+", "_", self.phrase.text.casefold()).strip("_")
        return f"{self.key}_{slug}.wav"


@dataclass(frozen=True)
class Plan:
    source: Path
    run_root: Path
    keyword: str
    keyword_phonemes: str
    words: Mapping[str, tuple[str, ...]]
    ipa_words: Mapping[str, str]
    say_as_words: Mapping[str, str]
    phoneme_words: tuple[str, ...]
    phrases: tuple[Phrase, ...]
    providers: Mapping[str, Mapping[str, Any]]
    budget: Mapping[str, Any]

    def positives(self) -> tuple[Phrase, ...]:
        return tuple(phrase for phrase in self.phrases if phrase.label == 1)

    def negatives(self) -> tuple[Phrase, ...]:
        return tuple(phrase for phrase in self.phrases if phrase.label == 0)


def word_tokens(text: str) -> list[str]:
    """Split text into lowercase words and drop punctuation."""

    return [token.casefold() for token in WORD_RE.findall(text)]


def phrase_phonemes(text: str, words: Mapping[str, Sequence[str]]) -> str:
    """Return the space-joined phones for one phrase, or raise on a missing word."""

    tokens = word_tokens(text)
    if not tokens:
        raise ValueError(f"phrase {text!r} has no letters")
    phones: list[str] = []
    for token in tokens:
        if token not in words:
            raise ValueError(
                f"phrase {text!r} uses {token!r}, which is missing from the word table"
            )
        phones.extend(str(phone) for phone in words[token])
    return " ".join(phones)


def load_plan(path: str | Path) -> Plan:
    """Read the YAML plan and check it against the project keyword phones."""

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"plan not found: {source}")
    document = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError(f"{source} must contain a YAML mapping")

    raw_words = document.get("words")
    if not isinstance(raw_words, dict) or not raw_words:
        raise ValueError(f"{source}: 'words' must be a non-empty mapping")
    words: dict[str, tuple[str, ...]] = {}
    ipa_words: dict[str, str] = {}
    say_as_words: dict[str, str] = {}
    for word, entry in raw_words.items():
        key = str(word).casefold()
        if not WORD_RE.fullmatch(key):
            raise ValueError(f"{source}: word {word!r} must contain letters only")
        if isinstance(entry, list):
            phones, ipa, say_as = entry, "", ""
        elif isinstance(entry, dict):
            phones = entry.get("phones")
            ipa = str(entry.get("ipa") or "")
            say_as = str(entry.get("say_as") or "")
        else:
            raise ValueError(
                f"{source}: word {word!r} must be a phone list or a mapping with 'phones'"
            )
        if not isinstance(phones, list) or not phones:
            raise ValueError(f"{source}: phones for {word!r} must be a non-empty list")
        words[key] = tuple(str(phone) for phone in phones)
        ipa_words[key] = ipa
        say_as_words[key] = say_as

    raw_phoneme_words = document.get("phoneme_words")
    if not isinstance(raw_phoneme_words, list) or not raw_phoneme_words:
        raise ValueError(f"{source}: 'phoneme_words' must be a non-empty list")
    phoneme_words: list[str] = []
    for word in raw_phoneme_words:
        key = str(word).casefold()
        if key not in words:
            raise ValueError(f"{source}: phoneme_words entry {word!r} is not in 'words'")
        phoneme_words.append(key)

    keyword = " ".join(str(document.get("keyword", DEFAULT_KEYWORD)).split()).casefold()
    keyword_tokens = tuple(word_tokens(keyword))
    if not keyword_tokens:
        raise ValueError(f"{source}: 'keyword' must contain letters")
    missing_keyword_words = [token for token in keyword_tokens if token not in words]
    if missing_keyword_words:
        raise ValueError(
            f"{source}: keyword words are missing from 'words': {missing_keyword_words}"
        )
    keyword_phonemes = phrase_phonemes(keyword, words)

    raw_phrases = document.get("phrases")
    if not isinstance(raw_phrases, list) or not raw_phrases:
        raise ValueError(f"{source}: 'phrases' must be a non-empty list")
    phrases: list[Phrase] = []
    for entry in raw_phrases:
        if not isinstance(entry, dict):
            raise ValueError(f"{source}: every phrase must be a mapping")
        text = str(entry.get("text", "")).strip()
        label = int(entry.get("label", -1))
        if not text:
            raise ValueError(f"{source}: every phrase needs 'text'")
        if label not in (0, 1):
            raise ValueError(f"{source}: phrase {text!r} needs label 0 or 1")
        tokens = word_tokens(text)
        use_alias = all(token in set(phoneme_words) for token in tokens)
        if use_alias:
            missing_alias = [
                token for token in tokens if not say_as_words.get(token)
            ]
            if missing_alias:
                raise ValueError(
                    f"{source}: phrase {text!r} would be rewritten end to end by the "
                    "phoneme rules, which eleven_v3 rejects as empty text. Give these "
                    f"words a 'say_as' spelling so the phrase can use the alias "
                    f"dictionary: {missing_alias}"
                )
        if label == 1 and tuple(word_tokens(text)) != keyword_tokens:
            raise ValueError(
                f"{source}: positive phrase {text!r} must contain {keyword!r} exactly once"
            )
        phones = phrase_phonemes(text, words)
        if label == 1 and phones != keyword_phonemes:
            raise ValueError(
                f"{source}: positive phrase {text!r} resolves to {phones!r}, "
                f"but the configured keyword {keyword!r} is {keyword_phonemes!r}"
            )
        phrases.append(
            Phrase(
                text=text,
                label=label,
                phonemes=phones,
                characters=len(text),
                use_alias=use_alias,
            )
        )

    raw_providers = document.get("providers")
    if not isinstance(raw_providers, dict) or not raw_providers:
        raise ValueError(f"{source}: 'providers' must be a non-empty mapping")

    alphabet = ""
    for provider_config in raw_providers.values():
        alphabet = str((provider_config or {}).get("dictionary_alphabet") or alphabet)
    if alphabet == "ipa":
        missing_ipa = [word for word in phoneme_words if not ipa_words.get(word)]
        if missing_ipa:
            raise ValueError(
                f"{source}: dictionary_alphabet is ipa but these words carry no ipa: "
                f"{missing_ipa}"
            )

    run_root = Path(str(document.get("run_root", "outputs/tts/hey_eva")))
    if not run_root.is_absolute():
        run_root = PROJECT_ROOT / run_root
    budget = document.get("budget") or {}
    if not isinstance(budget, dict):
        raise ValueError(f"{source}: 'budget' must be a mapping")
    return Plan(
        source=source,
        run_root=run_root,
        keyword=keyword,
        keyword_phonemes=keyword_phonemes,
        words=words,
        ipa_words=ipa_words,
        say_as_words=say_as_words,
        phoneme_words=tuple(phoneme_words),
        phrases=tuple(phrases),
        providers=raw_providers,
        budget=budget,
    )


def phrase_ipa(plan: Plan, text: str) -> str:
    """Join the per-word IPA spellings, for engines that take IPA directly."""

    return " ".join(plan.ipa_words[token] for token in word_tokens(text))


def phone_base(phone: str) -> str:
    """Strip the ARPAbet stress digit from one phone."""

    return "".join(char for char in phone.upper() if char.isalpha())


# ARPAbet to the phone set misaki feeds Kokoro. Misaki uses its own symbols for
# the diphthongs: A = eɪ, I = aɪ, O = oʊ, W = aʊ, Y = ɔɪ.
MISAKI_PHONES = {
    "AA": "ɑ",
    "AE": "æ",
    "AH": "ə",
    "AO": "ɔ",
    "AW": "W",
    "AY": "I",
    "B": "b",
    "CH": "ʧ",
    "D": "d",
    "DH": "ð",
    "EH": "ɛ",
    "ER": "ɜɹ",
    "EY": "A",
    "F": "f",
    "G": "ɡ",
    "HH": "h",
    "IH": "ɪ",
    "IY": "i",
    "JH": "ʤ",
    "K": "k",
    "L": "l",
    "M": "m",
    "N": "n",
    "NG": "ŋ",
    "OW": "O",
    "OY": "Y",
    "P": "p",
    "R": "ɹ",
    "S": "s",
    "SH": "ʃ",
    "T": "t",
    "TH": "θ",
    "UH": "ʊ",
    "UW": "u",
    "V": "v",
    "W": "w",
    "Y": "j",
    "Z": "z",
}
STRESS_MARKS = {"1": "ˈ", "2": "ˌ"}


def misaki_phone(phone: str) -> str:
    """Render one ARPAbet phone the way misaki writes it."""

    base = phone_base(phone)
    symbol = MISAKI_PHONES.get(base)
    if symbol is None:
        raise ValueError(f"no misaki symbol for ARPAbet phone {phone!r}")
    stress = STRESS_MARKS.get("".join(char for char in phone if char.isdigit()), "")
    return stress + symbol


def phrase_misaki(plan: Plan, text: str) -> str:
    """Render a phrase for Kokoro, one misaki phone string per word."""

    words: list[str] = []
    for token in word_tokens(text):
        words.append("".join(misaki_phone(phone) for phone in plan.words[token]))
    return " ".join(words)


def _normalized(value: str) -> str:
    return " ".join(value.split()).casefold()


def build_jobs(
    plan: Plan,
    *,
    providers: Sequence[str] | None = None,
    voice_ids: Sequence[str] | None = None,
    texts: Sequence[str] | None = None,
    speeds: Sequence[float] | None = None,
) -> list[Job]:
    """Cross every selected voice with every selected phrase."""

    wanted_providers = {name.casefold() for name in providers or ()}
    wanted_voices = {voice.casefold() for voice in voice_ids or ()}
    wanted_texts = {_normalized(text) for text in texts or ()}
    wanted_speeds = {float(value) for value in speeds or ()}
    jobs: list[Job] = []
    for provider_name, provider_config in plan.providers.items():
        if wanted_providers and provider_name.casefold() not in wanted_providers:
            continue
        raw_speeds = (provider_config or {}).get("speeds") or [1.0]
        speeds = [float(value) for value in raw_speeds]
        if any(speed <= 0 for speed in speeds):
            raise ValueError(f"provider {provider_name!r} has a speed that is not positive")
        raw_volumes = (provider_config or {}).get("volumes") or [1.0]
        volumes = [float(value) for value in raw_volumes]
        if any(volume <= 0 for volume in volumes):
            raise ValueError(f"provider {provider_name!r} has a volume that is not positive")
        raw_voices = (provider_config or {}).get("voices") or []
        for entry in raw_voices:
            if not isinstance(entry, dict) or not entry.get("voice_id"):
                raise ValueError(f"provider {provider_name!r} has an invalid voice entry")
            voice = Voice(
                voice_id=str(entry["voice_id"]),
                name=str(entry.get("name") or entry["voice_id"]),
                accent=str(entry.get("accent") or ""),
                gender=str(entry.get("gender") or ""),
            )
            if wanted_voices and voice.voice_id.casefold() not in wanted_voices:
                continue
            for phrase in plan.phrases:
                if wanted_texts and _normalized(phrase.text) not in wanted_texts:
                    continue
                for speed in speeds:
                    if wanted_speeds and speed not in wanted_speeds:
                        continue
                    for volume in volumes:
                        jobs.append(
                            Job(
                                provider=provider_name,
                                voice=voice,
                                phrase=phrase,
                                speed=speed,
                                volume=volume,
                            )
                        )
    return jobs


def rules_fingerprint(rules: Sequence[Mapping[str, str]]) -> str:
    """Stable hash of a rule set, so a stale dictionary is never reused."""

    payload = json.dumps(list(rules), ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def dictionary_rules(plan: Plan, kind: str = "phoneme") -> list[dict[str, str]]:
    """Build the rules for one dictionary.

    kind="phoneme" writes a pronunciation rule per forced word, in IPA or
    ARPAbet. kind="alias" writes respellings for the words used by phrases that
    the phoneme rules would otherwise rewrite end to end.
    """

    alphabet = "cmu_arpabet"
    for provider_config in plan.providers.values():
        alphabet = str((provider_config or {}).get("dictionary_alphabet") or alphabet)
    if kind == "alias":
        used = {
            token
            for phrase in plan.phrases
            if phrase.use_alias
            for token in word_tokens(phrase.text)
        }
        chosen = sorted(word for word in plan.phoneme_words if word in used)
    elif kind == "phoneme":
        chosen = sorted(plan.phoneme_words)
    else:
        raise ValueError(f"unknown dictionary kind {kind!r}")

    rules: list[dict[str, str]] = []
    seen: set[str] = set()
    for word in chosen:
        if kind == "alias":
            payload = plan.say_as_words.get(word, "")
            if not payload:
                raise ValueError(
                    f"word {word!r} has no say_as spelling for the alias dictionary"
                )
        elif alphabet == "ipa":
            payload = plan.ipa_words.get(word, "")
            if not payload:
                raise ValueError(f"word {word!r} has no ipa for alphabet {alphabet!r}")
        else:
            payload = " ".join(plan.words[word])
        for spelling in (word, word.capitalize()):
            if spelling in seen:
                continue
            seen.add(spelling)
            if kind == "alias":
                rules.append(
                    {"string_to_replace": spelling, "type": "alias", "alias": payload}
                )
            else:
                rules.append(
                    {
                        "string_to_replace": spelling,
                        "type": "phoneme",
                        "phoneme": payload,
                        "alphabet": alphabet,
                    }
                )
    return rules


@dataclass
class ElevenLabsProvider:
    """ElevenLabs adapter.

    Only eleven_v3, eleven_v4, and eleven_flash_v2 read pronunciation
    dictionary phoneme rules. Every other model drops them without an error.
    """

    api_key: str
    model_id: str = "eleven_flash_v2"
    output_format: str = "wav_24000"
    alias: str = "elevenlabs"
    transport: Transport = urllib_transport
    sleep: Callable[[float], None] = time.sleep
    base_url: str = "https://api.elevenlabs.io"
    max_attempts: int = 4
    timeout: float = 90.0
    # A provider may answer 200 with an empty WAV. Retry, then fail the job.
    min_duration: float = 0.25
    min_peak: float = 0.02
    voice_settings: Mapping[str, float] | None = None
    _locators: dict[str, dict[str, str]] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        if self.output_format not in OUTPUT_FORMATS:
            raise ValueError(f"unknown output_format {self.output_format!r}")
        if not self.output_format.startswith("wav_"):
            raise ValueError("output_format must be a wav_* value so clips land as WAV")

    @property
    def directory(self) -> str:
        return f"{self.alias}_output"

    @property
    def sample_rate(self) -> int:
        return int(self.output_format.split("_")[1])

    def _request(
        self, method: str, path: str, payload: Mapping[str, Any] | None = None
    ) -> bytes:
        url = f"{self.base_url}{path}"
        body = json.dumps(payload).encode("utf-8") if payload is not None else None
        headers: dict[str, str] = {
            "xi-api-key": self.api_key,
            "accept": "application/json",
        }
        if body is not None:
            headers["Content-Type"] = "application/json"
        delay = 1.0
        status = 0
        last = ""
        for attempt in range(1, self.max_attempts + 1):
            status, raw = self.transport(method, url, headers, body, timeout=self.timeout)
            if status == 200:
                return raw
            last = raw[:200].decode("utf-8", "replace")
            if status == 429 or status >= 500:
                if attempt == self.max_attempts:
                    break
                self.sleep(delay)
                delay *= 2
                continue
            break
        raise ProviderError(f"{method} {path} failed with {status}: {last!r}")

    def list_voices(self) -> list[dict[str, Any]]:
        data = json.loads(self._request("GET", "/v1/voices"))
        voices = data.get("voices")
        if not isinstance(voices, list):
            raise ProviderError("/v1/voices returned no voice list")
        return voices

    def create_dictionary(
        self, name: str, rules: Sequence[Mapping[str, str]]
    ) -> dict[str, str]:
        raw = self._request(
            "POST",
            "/v1/pronunciation-dictionaries/add-from-rules",
            {"name": name, "rules": list(rules)},
        )
        data = json.loads(raw)
        dictionary_id = str(data.get("id") or "")
        version_id = str(data.get("version_id") or "")
        if not dictionary_id:
            raise ProviderError("dictionary response has no id")
        if not version_id:
            detail = json.loads(
                self._request("GET", f"/v1/pronunciation-dictionaries/{dictionary_id}")
            )
            version_id = str(
                detail.get("latest_version_id") or detail.get("version_id") or ""
            )
        if not version_id:
            raise ProviderError("dictionary response has no version_id")
        return {
            "pronunciation_dictionary_id": dictionary_id,
            "version_id": version_id,
            "name": name,
            "rules": str(len(rules)),
        }

    def prepare(
        self,
        plan: Plan,
        jobs: Sequence[Job],
        root: Path,
        log: Callable[[str], None],
    ) -> None:
        """Create or reuse the pronunciation dictionaries this run needs."""

        provider_config = plan.providers[self.alias] or {}
        dictionary_path = root / "dictionary.json"
        dictionary_name = str(provider_config.get("dictionary_name") or "hey-eva-corpus")
        cached: dict[str, Any] = {}
        if dictionary_path.is_file():
            stored = json.loads(dictionary_path.read_text(encoding="utf-8"))
            cached = (
                stored
                if ("phoneme" in stored or "alias" in stored)
                else {"phoneme": stored}
            )

        def ensure(kind: str) -> dict[str, str]:
            rules = dictionary_rules(plan, kind)
            fingerprint = rules_fingerprint(rules)
            entry = cached.get(kind)
            if entry and entry.get("rules_sha") == fingerprint:
                log(
                    f"dictionary[{kind}]: reusing "
                    f"{entry['pronunciation_dictionary_id']} ({entry.get('rules')} rules)"
                )
                return entry
            suffix = "" if kind == "phoneme" else "-alias"
            entry = self.create_dictionary(f"{dictionary_name}{suffix}", rules)
            entry["rules_sha"] = fingerprint
            cached[kind] = entry
            write_json(dictionary_path, cached)
            log(
                f"dictionary[{kind}]: created {entry['pronunciation_dictionary_id']} "
                f"with {len(rules)} rules"
            )
            return entry

        self._locators = {"phoneme": ensure("phoneme")}
        if any(job.phrase.use_alias for job in jobs):
            self._locators["alias"] = ensure("alias")

    def synthesize(self, job: Job, destination: Path) -> int:
        locator = self._locators.get("alias" if job.phrase.use_alias else "phoneme")
        payload: dict[str, Any] = {"text": job.phrase.text, "model_id": self.model_id}
        settings = dict(self.voice_settings or {})
        if job.speed != 1.0:
            settings["speed"] = job.speed
        if settings:
            payload["voice_settings"] = settings
        if locator:
            payload["pronunciation_dictionary_locators"] = [
                {
                    "pronunciation_dictionary_id": locator["pronunciation_dictionary_id"],
                    "version_id": locator["version_id"],
                }
            ]
        query = urllib.parse.urlencode({"output_format": self.output_format})
        voice = urllib.parse.quote(job.voice.voice_id, safe="")
        problem = "no attempt was made"
        for attempt in range(1, self.max_attempts + 1):
            raw = self._request("POST", f"/v1/text-to-speech/{voice}?{query}", payload)
            if not raw.startswith(b"RIFF"):
                problem = "response is not a RIFF/WAVE payload"
            else:
                duration, peak = audio_health(raw)
                if duration >= self.min_duration and peak >= self.min_peak:
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_bytes(raw)
                    return len(raw)
                problem = f"empty audio (duration={duration:.3f}s peak={peak:.3f})"
            if attempt < self.max_attempts:
                self.sleep(float(attempt))
        raise ProviderError(
            f"POST /v1/text-to-speech/{voice} gave no usable audio in "
            f"{self.max_attempts} attempts: {problem}"
        )


@dataclass
class PiperProvider:
    """Piper adapter.

    Speed arrives as --length-scale, which is the inverse of speed, and
    loudness as --volume. Piper has no pitch parameter. The text goes in on
    stdin, wrapped in double brackets so espeak-ng reads it as IPA.
    """

    alias: str = "piper"
    # The interpreter that has piper installed. The module flag is added by the
    # adapter, so this is just the executable (or a wrapper).
    python: Sequence[str] = ("python3",)
    data_dir: Path = Path("outputs/tts/piper_models")
    runner: Callable[[Sequence[str], bytes], None] | None = None
    _plan: Plan | None = field(init=False, default=None)

    @property
    def directory(self) -> str:
        return f"{self.alias}_output"

    def prepare(
        self,
        plan: Plan,
        jobs: Sequence[Job],
        root: Path,
        log: Callable[[str], None],
    ) -> None:
        self._plan = plan
        self.data_dir.mkdir(parents=True, exist_ok=True)
        for name in sorted({job.voice.voice_id for job in jobs}):
            if (self.data_dir / f"{name}.onnx").is_file():
                continue
            log(f"piper: downloading voice {name}")
            self._run_subprocess(
                [
                    *self.python,
                    "-m",
                    "piper.download_voices",
                    name,
                    "--data-dir",
                    str(self.data_dir),
                ],
                b"",
            )

    def synthesize(self, job: Job, destination: Path) -> int:
        plan = self._plan
        if plan is None:
            raise ProviderError("PiperProvider.prepare was not called")
        command = [
            *self.python,
            "-m",
            "piper",
            "-m",
            job.voice.voice_id,
            "--data-dir",
            str(self.data_dir),
            "-f",
            str(destination),
        ]
        if job.speed != 1.0:
            command += ["--length-scale", f"{1.0 / job.speed:.6f}"]
        if job.volume != 1.0:
            command += ["--volume", f"{job.volume:.3f}"]
        payload = f"[[{phrase_ipa(plan, job.phrase.text)}]]".encode("utf-8")
        destination.parent.mkdir(parents=True, exist_ok=True)
        self._run_subprocess(command, payload)
        if not destination.is_file():
            raise ProviderError("piper produced no output file")
        return destination.stat().st_size

    def _run_subprocess(self, command: Sequence[str], payload: bytes) -> None:
        if self.runner is not None:
            self.runner(command, payload)
            return
        result = subprocess.run(list(command), input=payload, capture_output=True)
        if result.returncode != 0:
            detail = result.stderr.decode("utf-8", "replace").strip()[:200]
            raise ProviderError(f"piper exited {result.returncode}: {detail}")


@dataclass
class KokoroProvider:
    """Kokoro adapter.

    Speed goes to generate_from_tokens(speed=...). Kokoro exposes no loudness
    and no pitch, so job.volume is ignored here.
    """

    alias: str = "kokoro"
    lang_code: str = "a"
    sample_rate: int = 24000
    pipeline_factory: Callable[[], Any] | None = None
    _pipeline: Any = field(init=False, default=None)
    _plan: Plan | None = field(init=False, default=None)

    @property
    def directory(self) -> str:
        return f"{self.alias}_output"

    def prepare(
        self,
        plan: Plan,
        jobs: Sequence[Job],
        root: Path,
        log: Callable[[str], None],
    ) -> None:
        self._plan = plan
        if self._pipeline is None:
            if self.pipeline_factory is not None:
                self._pipeline = self.pipeline_factory()
            else:
                from kokoro import KPipeline

                self._pipeline = KPipeline(lang_code=self.lang_code)
            log(f"kokoro: pipeline ready (lang {self.lang_code})")

    def synthesize(self, job: Job, destination: Path) -> int:
        plan = self._plan
        if plan is None or self._pipeline is None:
            raise ProviderError("KokoroProvider.prepare was not called")
        tokens = phrase_misaki(plan, job.phrase.text)
        chunks: list[Any] = []
        for result in self._pipeline.generate_from_tokens(
            tokens=tokens, voice=job.voice.voice_id, speed=job.speed
        ):
            audio = getattr(result, "audio", None)
            if audio is not None:
                chunks.append(np.asarray(audio))
        if not chunks:
            raise ProviderError("kokoro produced no audio")
        destination.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(destination), np.concatenate(chunks), self.sample_rate)
        return destination.stat().st_size


def build_provider(name: str, plan: Plan) -> Provider:
    """Construct the adapter for one provider name from the plan."""

    provider_config = plan.providers.get(name)
    if provider_config is None:
        raise ValueError(f"unknown provider {name!r} in {plan.source}")
    key = name.casefold()
    alias = str(provider_config.get("alias") or key)
    if key == "elevenlabs":
        env_name = str(provider_config.get("api_key_env") or "ELEVENLABS_API_KEY")
        api_key = os.environ.get(env_name, "").strip()
        if not api_key:
            raise SystemExit(f"{env_name} is not set; export it or use --dry-run")
        raw_settings = provider_config.get("voice_settings") or {}
        if not isinstance(raw_settings, dict):
            raise ValueError(f"provider {name!r} has a non-mapping voice_settings")
        return ElevenLabsProvider(
            api_key=api_key,
            model_id=str(provider_config.get("model_id") or "eleven_v3"),
            output_format=str(provider_config.get("output_format") or "wav_16000"),
            alias=alias,
            voice_settings={
                str(setting): float(value) for setting, value in raw_settings.items()
            },
        )
    if key == "piper":
        python = provider_config.get("python") or ["python3"]
        if isinstance(python, str):
            python = shlex.split(python)
        data_dir = Path(str(provider_config.get("data_dir") or "outputs/tts/piper_models"))
        if not data_dir.is_absolute():
            data_dir = PROJECT_ROOT / data_dir
        return PiperProvider(
            alias=alias,
            python=[str(part) for part in python],
            data_dir=data_dir,
        )
    if key == "kokoro":
        return KokoroProvider(
            alias=alias,
            lang_code=str(provider_config.get("lang_code") or "a"),
        )
    raise ValueError(f"provider {name!r} has no adapter yet")


def load_excluded(root: Path) -> set[str]:
    """Keys the verifier failed, plus keys a reviewer flagged bad.

    The verdict log is append-only, so the last entry for a key decides.
    """

    keys: set[str] = set()
    rejected = root / "rejected.jsonl"
    if rejected.is_file():
        for line in rejected.read_text(encoding="utf-8").splitlines():
            if line.strip():
                key = str(json.loads(line).get("key") or "")
                if key:
                    keys.add(key)
    verdicts: dict[str, str] = {}
    reviews = root / "reviews.jsonl"
    if reviews.is_file():
        for line in reviews.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                key = str(row.get("key") or "")
                if key:
                    verdicts[key] = str(row.get("verdict") or "")
    keys.update(key for key, verdict in verdicts.items() if verdict == "bad")
    return keys


def write_index(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Replace the index file with exactly these rows."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            print(json.dumps(row, ensure_ascii=False, sort_keys=True), file=handle)


def prune_keys(root: Path, keys: set[str]) -> int:
    """Delete the audio and index rows for the given keys.

    A rejected clip keeps its row and its file, so a rerun would skip it. Drop
    both to let the next run synthesize that job again.
    """

    index_path = root / "index.jsonl"
    keep: list[dict[str, Any]] = []
    removed = 0
    for row in load_index(index_path):
        if str(row.get("key")) in keys:
            path = root / str(row.get("audio_path") or "")
            if path.is_file():
                path.unlink()
            removed += 1
            continue
        keep.append(row)
    if removed:
        write_index(index_path, keep)
    return removed


def load_index(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def append_index(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        print(json.dumps(row, ensure_ascii=False, sort_keys=True), file=handle)


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), file=handle)


def write_source_csv(rows: Sequence[Mapping[str, Any]], destination: Path) -> None:
    """Write the CSV that scripts/prepare_tts_lora_manifest.py reads."""

    ordered = sorted(rows, key=lambda row: str(row["audio_path"]))
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(SOURCE_FIELDS))
        writer.writeheader()
        for row in ordered:
            writer.writerow({field: row[field] for field in SOURCE_FIELDS})


def run_jobs(
    jobs: Sequence[Job],
    provider: Provider,
    *,
    plan: Plan,
    resume: bool = True,
    run_root: Path | None = None,
    exclude: set[str] | None = None,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Synthesize the jobs and keep index.jsonl and source.csv in step."""

    root = Path(run_root or plan.run_root)
    root.mkdir(parents=True, exist_ok=True)
    excluded = set(exclude or ())
    index_path = root / "index.jsonl"
    index = load_index(index_path)
    done = {
        str(row["key"]): row
        for row in index
        if Path(root / str(row["audio_path"])).is_file()
    }

    provider.prepare(plan, jobs, root, log)

    completed = skipped = 0
    failures: list[dict[str, str]] = []
    characters = 0
    total = len(jobs)
    for position, job in enumerate(jobs, start=1):
        if resume and job.key in done:
            skipped += 1
            continue
        relative = f"{provider.directory}/{job.voice.voice_id}/{job.filename}"
        destination = root / relative
        try:
            size = provider.synthesize(job, destination)
        except ProviderError as exc:
            failures.append({"key": job.key, "text": job.phrase.text, "error": str(exc)})
            log(f"[{position}/{total}] FAIL {job.voice.name} {job.phrase.text!r}: {exc}")
            continue
        row = {
            "key": job.key,
            "provider": provider.alias,
            "voice_id": job.voice.voice_id,
            "voice_name": job.voice.name,
            "label": job.phrase.label,
            "text": job.phrase.text,
            "phonemes": job.phrase.phonemes,
            "keyword_phonemes": plan.keyword_phonemes,
            "forced": "alias" if job.phrase.use_alias else "phoneme",
            "speed": job.speed,
            "volume": job.volume,
            "audio_path": relative,
            "bytes": size,
            "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
            "characters": job.phrase.characters,
        }
        append_index(index_path, row)
        index.append(row)
        done[job.key] = row
        completed += 1
        characters += job.phrase.characters
        log(
            f"[{position}/{total}] {job.voice.name} {job.phrase.text!r} "
            f"-> {relative} ({size} bytes)"
        )

    source_rows = [
        {
            "audio_path": row["audio_path"],
            "keyword": plan.keyword,
            "label": row["label"],
            "voice_id": row["voice_id"],
            "voice_name": row["voice_name"],
            "text_variant": row["text"],
            "sha256": row["sha256"],
            "keyword_phonemes": row["keyword_phonemes"],
            "text_variant_phonemes": row["phonemes"],
            "speed": row.get("speed", 1.0),
            "volume": row.get("volume", 1.0),
        }
        for row in done.values()
        if str(row["key"]) not in excluded
    ]
    source_path = root / "source.csv"
    write_source_csv(source_rows, source_path)
    return {
        "jobs": total,
        "completed": completed,
        "skipped": skipped,
        "failed": len(failures),
        "characters": characters,
        "excluded": sum(1 for row in done.values() if str(row["key"]) in excluded),
        "source_csv": str(source_path),
        "index": str(index_path),
        "failures": failures,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    default_config = PROJECT_ROOT / "configs/tts/hey_eva.yaml"
    parser.add_argument("--config", type=Path, default=default_config)
    parser.add_argument("--dry-run", action="store_true", help="print the plan and stop")
    parser.add_argument("--limit", type=int, default=0, help="stop after this many jobs")
    parser.add_argument("--provider", action="append", help="restrict to a provider")
    parser.add_argument("--voice-id", action="append", help="restrict to a voice id")
    parser.add_argument("--text", action="append", help="restrict to a phrase text")
    parser.add_argument("--speed", action="append", type=float, help="restrict to a speed")
    parser.add_argument("--run-root", type=Path, help="override the plan run_root")
    parser.add_argument("--no-resume", action="store_true", help="redo indexed jobs")
    parser.add_argument(
        "--exclude-rejected",
        action="store_true",
        help="keep verifier failures and human-flagged clips out of source.csv",
    )
    parser.add_argument(
        "--retry-rejected",
        action="store_true",
        help="delete rejected clips so this run synthesizes them again",
    )
    parser.add_argument(
        "--prepare-dictionary",
        action="store_true",
        help="create the pronunciation dictionary and stop (costs no credits)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    plan = load_plan(args.config)
    jobs = build_jobs(
        plan,
        providers=args.provider,
        voice_ids=args.voice_id,
        texts=args.text,
        speeds=args.speed,
    )
    run_root = Path(args.run_root).expanduser().resolve() if args.run_root else plan.run_root

    max_characters = int(plan.budget.get("max_characters", 0) or 0)
    max_jobs = int(plan.budget.get("max_jobs", 0) or 0)

    if args.limit:
        jobs = jobs[: args.limit]
    characters = sum(job.phrase.characters for job in jobs)
    positives = sum(job.phrase.label == 1 for job in jobs)

    print(f"plan: {plan.source}")
    print(f"run_root: {run_root}")
    print(f"keyword: {plan.keyword} ({plan.keyword_phonemes})")
    print(
        f"phrases: {len(plan.phrases)} "
        f"({len(plan.positives())} positive, {len(plan.negatives())} negative)"
    )
    for name, provider_config in plan.providers.items():
        config = provider_config or {}
        voices = len(config.get("voices") or [])
        speeds = config.get("speeds") or [1.0]
        volumes = config.get("volumes") or [1.0]
        detail = config.get("model_id") or config.get("lang_code") or ""
        print(
            f"provider {name}: {voices} voices, {len(speeds)} speeds, "
            f"{len(volumes)} volumes {detail}"
        )
    print(f"jobs: {len(jobs)} ({positives} positive, {len(jobs) - positives} negative)")
    print(f"characters: {characters} (budget {max_characters or 'unset'})")
    if max_jobs and len(jobs) > max_jobs:
        raise SystemExit(f"job count {len(jobs)} exceeds budget.max_jobs {max_jobs}")
    if max_characters and characters > max_characters:
        raise SystemExit(
            f"character count {characters} exceeds budget.max_characters {max_characters}"
        )

    if args.dry_run:
        return 0
    if not jobs:
        raise SystemExit("no jobs selected")

    provider_names = args.provider or list(plan.providers)
    if args.prepare_dictionary:
        first = provider_names[0]
        dictionary_provider = build_provider(first, plan)
        if not isinstance(dictionary_provider, ElevenLabsProvider):
            raise SystemExit(f"provider {first!r} has no pronunciation dictionary")
        provider_config = plan.providers[first] or {}
        locator = dictionary_provider.create_dictionary(
            str(provider_config.get("dictionary_name") or "hey-eva-corpus"),
            dictionary_rules(plan),
        )
        write_json(run_root / "dictionary.json", locator)
        print(json.dumps(locator, ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    dropped = load_excluded(run_root)
    if args.retry_rejected:
        # Only touch keys this run will actually synthesize. Pruning a key that
        # the current selection does not cover would delete it for good.
        selected = {job.key for job in jobs}
        removed = prune_keys(run_root, dropped & selected)
        if removed:
            print(f"retry: dropped {removed} rejected clips for regeneration")
    exclude = dropped if args.exclude_rejected else set()
    summaries: list[dict[str, Any]] = []
    for provider_name in provider_names:
        provider_jobs = [job for job in jobs if job.provider == provider_name]
        if not provider_jobs:
            continue
        provider = build_provider(provider_name, plan)
        summaries.append(
            run_jobs(
                provider_jobs,
                provider,
                plan=plan,
                resume=not args.no_resume,
                run_root=run_root,
                exclude=exclude,
            )
        )
    if not summaries:
        raise SystemExit("no jobs selected")
    payload = summaries if len(summaries) > 1 else summaries[0]
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 1 if any(summary["failed"] for summary in summaries) else 0


if __name__ == "__main__":
    raise SystemExit(main())
