import csv
import io
import json
import wave
from pathlib import Path

import numpy as np
import pytest

from dataclasses import replace

from scripts.generate_hey_eva_tts import (
    KEYWORD_PHONEMES,
    ElevenLabsProvider,
    Job,
    ProviderError,
    build_jobs,
    dictionary_rules,
    load_excluded,
    prune_keys,
    rules_fingerprint,
    load_plan,
    main,
    run_jobs,
    word_tokens,
)
from scripts.prepare_tts_lora_manifest import build_tts_lora_manifest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_PLAN = PROJECT_ROOT / "configs/tts/hey_eva.yaml"


def _wav_bytes(seconds: float = 0.5, frequency: float = 220.0, amplitude: float = 0.3) -> bytes:
    """A real tone, so the adapter's silence gate accepts it."""

    rate = 16000
    frames = int(rate * seconds)
    timeline = np.linspace(0.0, seconds, frames, endpoint=False)
    samples = (amplitude * np.sin(2.0 * np.pi * frequency * timeline) * 32767.0).astype("<i2")
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(samples.tobytes())
    return buffer.getvalue()


def _silent_wav_bytes(seconds: float = 0.2) -> bytes:
    return _wav_bytes(seconds=seconds, amplitude=0.0)


def _write_plan(tmp_path: Path, *, voices: int = 2, include_unknown_word: bool = False) -> Path:
    voice_lines = "\n".join(
        f"      - {{voice_id: voice-{index}, name: Voice {index}}}"
        for index in range(voices)
    )
    extra = ""
    if include_unknown_word:
        extra = '  - {text: "Hey Zebra", label: 0}\n'
    plan = (
        f"run_root: {tmp_path / 'run'}\n"
        "keyword: hey eva\n"
        "words:\n"
        "  hey: [HH, EY1]\n"
        "  eva: [IY1, V, AH0]\n"
        "  ava: [EY1, V, AH0]\n"
        "phoneme_words: [eva, ava]\n"
        "phrases:\n"
        '  - {text: "Hey Eva", label: 1}\n'
        '  - {text: "Hey Ava", label: 0}\n'
        f"{extra}"
        "providers:\n"
        "  elevenlabs:\n"
        "    model_id: eleven_flash_v2\n"
        "    output_format: wav_24000\n"
        "    dictionary_alphabet: cmu_arpabet\n"
        "    voices:\n"
        f"{voice_lines}\n"
        "budget:\n"
        "  max_characters: 1000\n"
        "  max_jobs: 100\n"
    )
    path = tmp_path / "plan.yaml"
    path.write_text(plan, encoding="utf-8")
    return path


class FakeTransport:
    """Records requests and replays scripted responses."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def __call__(self, method, url, headers, body, timeout=60.0):
        self.calls.append(
            {
                "method": method,
                "url": url,
                "headers": dict(headers),
                "json": json.loads(body) if body else None,
            }
        )
        if not self.responses:
            raise AssertionError(f"unexpected request: {method} {url}")
        status, payload = self.responses.pop(0)
        if isinstance(payload, str):
            payload = payload.encode("utf-8")
        return status, payload


def _provider(transport, sleeps=None) -> ElevenLabsProvider:
    return ElevenLabsProvider(
        api_key="test-key",
        transport=transport,
        sleep=(sleeps.append if sleeps is not None else (lambda _seconds: None)),
    )


def test_project_plan_positives_match_keyword_phones():
    plan = load_plan(PROJECT_PLAN)
    assert len(plan.positives()) == 5
    for phrase in plan.positives():
        assert phrase.phonemes == KEYWORD_PHONEMES


def test_positive_phrase_with_repeated_keyword_is_rejected(tmp_path):
    plan_text = _write_plan(tmp_path).read_text(encoding="utf-8")
    plan_text = plan_text.replace(
        '- {text: "Hey Eva", label: 1}',
        '- {text: "Hey Eva, hey Eva", label: 1}',
    )
    path = tmp_path / "repeated.yaml"
    path.write_text(plan_text, encoding="utf-8")
    with pytest.raises(ValueError, match="exactly once"):
        load_plan(path)


def test_project_plan_words_cover_every_phrase():
    plan = load_plan(PROJECT_PLAN)
    assert len(plan.phrases) == len(plan.positives()) + len(plan.negatives())
    assert len(plan.providers["elevenlabs"]["voices"]) == 22


def test_unknown_word_is_rejected(tmp_path):
    path = _write_plan(tmp_path, include_unknown_word=True)
    with pytest.raises(ValueError, match="zebra"):
        load_plan(path)


def test_keyword_words_must_be_in_the_word_table(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text(
        "run_root: /tmp/ignored\n"
        "keyword: hey google\n"
        "words:\n"
        "  hey: [HH, EY1]\n"
        "phoneme_words: [hey]\n"
        "phrases:\n"
        '  - {text: "Hey Google", label: 1}\n'
        "providers:\n"
        "  elevenlabs:\n"
        "    voices:\n"
        "      - {voice_id: a}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="keyword words are missing"):
        load_plan(path)


def test_build_jobs_crosses_voices_with_phrases(tmp_path):
    plan = load_plan(_write_plan(tmp_path, voices=3))
    jobs = build_jobs(plan)
    assert len(jobs) == 6
    assert len({job.key for job in jobs}) == 6
    assert all(job.filename.endswith(".wav") for job in jobs)
    assert {job.voice.voice_id for job in jobs} == {"voice-0", "voice-1", "voice-2"}


def test_build_jobs_filters_by_voice_and_provider(tmp_path):
    plan = load_plan(_write_plan(tmp_path, voices=3))
    jobs = build_jobs(plan, providers=["elevenlabs"], voice_ids=["voice-1"])
    assert [job.voice.voice_id for job in jobs] == ["voice-1", "voice-1"]
    assert build_jobs(plan, providers=["kokoro"]) == []


def test_build_jobs_filters_by_speed(tmp_path):
    plan_path = tmp_path / "speeds.yaml"
    plan_path.write_text(
        f"run_root: {tmp_path / 'run'}\n"
        "keyword: hey eva\n"
        "words:\n"
        "  hey: [HH, EY1]\n"
        "  eva: [IY1, V, AH0]\n"
        "phoneme_words: [eva]\n"
        "phrases:\n"
        '  - {text: "Hey Eva", label: 1}\n'
        "providers:\n"
        "  elevenlabs:\n"
        "    dictionary_alphabet: cmu_arpabet\n"
        "    speeds: [0.9, 1.0, 1.1]\n"
        "    voices:\n"
        "      - {voice_id: v1}\n",
        encoding="utf-8",
    )
    plan = load_plan(plan_path)
    assert [job.speed for job in build_jobs(plan, speeds=[1.0])] == [1.0]
    assert build_jobs(plan, speeds=[1.25]) == []


def test_build_jobs_filters_by_text(tmp_path):
    plan = load_plan(_write_plan(tmp_path, voices=2))
    jobs = build_jobs(plan, texts=["  hey ava "])
    assert len(jobs) == 2
    assert {job.phrase.text for job in jobs} == {"Hey Ava"}
    assert build_jobs(plan, texts=["Hey Zebra"]) == []


def test_build_jobs_crosses_speeds_and_keeps_default_keys(tmp_path):
    plan_path = tmp_path / "speeds.yaml"
    plan_path.write_text(
        f"run_root: {tmp_path / 'run'}\n"
        "keyword: hey eva\n"
        "words:\n"
        "  hey: [HH, EY1]\n"
        "  eva: [IY1, V, AH0]\n"
        "phoneme_words: [eva]\n"
        "phrases:\n"
        '  - {text: "Hey Eva", label: 1}\n'
        "providers:\n"
        "  elevenlabs:\n"
        "    dictionary_alphabet: cmu_arpabet\n"
        "    speeds: [0.9, 1.0, 1.1]\n"
        "    voices:\n"
        "      - {voice_id: v1}\n",
        encoding="utf-8",
    )
    plan = load_plan(plan_path)
    jobs = build_jobs(plan)
    assert [job.speed for job in jobs] == [0.9, 1.0, 1.1]
    assert len({job.key for job in jobs}) == 3

    reference = Job(provider="elevenlabs", voice=jobs[0].voice, phrase=jobs[0].phrase)
    default = next(job for job in jobs if job.speed == 1.0)
    assert default.key == reference.key
    assert jobs[0].key != reference.key


def test_provider_sends_voice_settings_and_speed(tmp_path):
    plan = load_plan(_write_plan(tmp_path, voices=1))
    _cache_dictionary(plan.run_root, plan)
    jobs = build_jobs(plan)

    transport = FakeTransport([(200, _wav_bytes())])
    run_jobs(jobs[:1], _provider(transport), plan=plan, log=lambda _m: None)
    assert "voice_settings" not in transport.calls[-1]["json"]

    slow = replace(jobs[0], speed=0.9)
    transport = FakeTransport([(200, _wav_bytes(frequency=300.0))])
    provider = ElevenLabsProvider(
        api_key="k",
        transport=transport,
        sleep=lambda _seconds: None,
        voice_settings={"stability": 0.4, "similarity_boost": 0.6},
    )
    run_jobs([slow], provider, plan=plan, log=lambda _m: None)
    payload = transport.calls[-1]["json"]
    assert payload["voice_settings"] == {
        "stability": 0.4,
        "similarity_boost": 0.6,
        "speed": 0.9,
    }
    rows = [
        json.loads(line)
        for line in (plan.run_root / "index.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert rows[-1]["speed"] == 0.9
    with Path(plan.run_root / "source.csv").open(encoding="utf-8", newline="") as handle:
        assert "speed" in next(csv.reader(handle))


def test_dry_run_needs_no_key_and_writes_nothing(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    plan_path = _write_plan(tmp_path)
    exit_code = main(["--config", str(plan_path), "--dry-run"])
    output = capsys.readouterr().out
    assert exit_code == 0
    assert "jobs: 4" in output
    assert "characters: 28" in output
    assert not (tmp_path / "run").exists()


def test_dictionary_rules_cover_both_cases(tmp_path):
    plan = load_plan(_write_plan(tmp_path))
    rules = dictionary_rules(plan)
    replacements = {rule["string_to_replace"] for rule in rules}
    assert {"eva", "Eva", "ava", "Ava"} <= replacements
    assert replacements.isdisjoint({"hey", "Hey"})
    for rule in rules:
        assert rule["type"] == "phoneme"
        assert rule["alphabet"] == "cmu_arpabet"
    eva = next(rule for rule in rules if rule["string_to_replace"] == "Eva")
    assert eva["phoneme"] == "IY1 V AH0"


def test_provider_creates_dictionary_then_sends_locator(tmp_path):
    plan = load_plan(_write_plan(tmp_path, voices=1))
    jobs = build_jobs(plan)
    payload = _wav_bytes()
    transport = FakeTransport(
        [(200, json.dumps({"id": "dict-1", "version_id": "ver-1"})), (200, payload)]
    )
    provider = _provider(transport)
    summary = run_jobs(jobs[:1], provider, plan=plan, log=lambda _message: None)

    assert summary["completed"] == 1
    assert summary["failed"] == 0
    assert transport.calls[0]["url"].endswith("/v1/pronunciation-dictionaries/add-from-rules")
    assert transport.calls[0]["json"]["rules"][0]["alphabet"] == "cmu_arpabet"
    assert transport.calls[1]["json"]["pronunciation_dictionary_locators"] == [
        {"pronunciation_dictionary_id": "dict-1", "version_id": "ver-1"}
    ]
    assert transport.calls[1]["json"]["model_id"] == "eleven_flash_v2"
    assert "output_format=wav_24000" in transport.calls[1]["url"]
    written = plan.run_root / summary["source_csv"].split("run/")[-1]
    assert written.is_file()
    audio = next((plan.run_root / "elevenlabs_output").rglob("*.wav"))
    assert audio.read_bytes() == payload


def test_provider_retries_on_rate_limit(tmp_path):
    plan = load_plan(_write_plan(tmp_path, voices=1))
    jobs = build_jobs(plan)
    _cache_dictionary(plan.run_root, plan)
    sleeps: list[float] = []
    transport = FakeTransport([(429, b"slow down"), (200, _wav_bytes())])
    summary = run_jobs(
        jobs[:1], _provider(transport, sleeps), plan=plan, log=lambda _message: None
    )
    assert summary["completed"] == 1
    assert sleeps == [1.0]
    assert len(transport.calls) == 2


def test_provider_rejects_non_wav_payload(tmp_path):
    plan = load_plan(_write_plan(tmp_path, voices=1))
    _cache_dictionary(plan.run_root, plan)
    transport = FakeTransport([(200, b"ID3\x03\x00mp3 payload")] * 4)
    summary = run_jobs(
        build_jobs(plan)[:1], _provider(transport), plan=plan, log=lambda _message: None
    )
    assert summary["failed"] == 1
    assert summary["completed"] == 0
    assert "RIFF" in summary["failures"][0]["error"]


def _cache_dictionary(root: Path, plan) -> None:
    """Pre-seed the phoneme dictionary so no creation call is needed."""

    rules = dictionary_rules(plan, "phoneme")
    root.mkdir(parents=True, exist_ok=True)
    (root / "dictionary.json").write_text(
        json.dumps(
            {
                "phoneme": {
                    "pronunciation_dictionary_id": "dict-1",
                    "version_id": "ver-1",
                    "rules": str(len(rules)),
                    "rules_sha": rules_fingerprint(rules),
                }
            }
        ),
        encoding="utf-8",
    )


def test_provider_retries_when_the_audio_is_silent(tmp_path):
    plan = load_plan(_write_plan(tmp_path, voices=1))
    _cache_dictionary(plan.run_root, plan)
    sleeps: list[float] = []
    transport = FakeTransport([(200, _silent_wav_bytes()), (200, _wav_bytes())])
    summary = run_jobs(
        build_jobs(plan)[:1], _provider(transport, sleeps), plan=plan, log=lambda _m: None
    )
    assert summary["completed"] == 1
    assert summary["failed"] == 0
    assert sleeps == [1.0]
    assert len(transport.calls) == 2


def test_provider_fails_when_every_attempt_is_silent(tmp_path):
    plan = load_plan(_write_plan(tmp_path, voices=1))
    _cache_dictionary(plan.run_root, plan)
    transport = FakeTransport([(200, _silent_wav_bytes()), (200, _silent_wav_bytes())])
    provider = ElevenLabsProvider(
        api_key="k", transport=transport, sleep=lambda _seconds: None, max_attempts=2
    )
    summary = run_jobs(build_jobs(plan)[:1], provider, plan=plan, log=lambda _m: None)
    assert summary["failed"] == 1
    assert summary["completed"] == 0
    assert "empty audio" in summary["failures"][0]["error"]


def test_project_plan_sends_no_phoneme_only_phrase():
    plan = load_plan(PROJECT_PLAN)
    assert "hey" not in plan.phoneme_words
    rules = dictionary_rules(plan)
    assert {rule["string_to_replace"] for rule in rules}.isdisjoint({"hey", "Hey"})
    covered = set(plan.phoneme_words)
    for phrase in plan.phrases:
        if all(token in covered for token in word_tokens(phrase.text)):
            assert phrase.use_alias, phrase.text
            for token in word_tokens(phrase.text):
                assert plan.say_as_words.get(token), (phrase.text, token)


def test_alias_phrase_gets_the_alias_dictionary(tmp_path):
    plan_path = tmp_path / "alias.yaml"
    plan_path.write_text(
        f"run_root: {tmp_path / 'run'}\n"
        "keyword: hey eva\n"
        "words:\n"
        '  hey: {phones: [HH, EY1], ipa: "ˈheɪ"}\n'
        '  eva: {phones: [IY1, V, AH0], ipa: "ˈiːvə", say_as: "Eevah"}\n'
        "phoneme_words: [eva]\n"
        "phrases:\n"
        '  - {text: "Eva", label: 0}\n'
        "providers:\n"
        "  elevenlabs:\n"
        "    model_id: eleven_v3\n"
        "    output_format: wav_16000\n"
        "    dictionary_alphabet: ipa\n"
        "    voices:\n"
        "      - {voice_id: v1}\n"
        "      - {voice_id: v2}\n",
        encoding="utf-8",
    )
    plan = load_plan(plan_path)
    jobs = build_jobs(plan)
    assert len(jobs) == 2
    assert all(job.phrase.use_alias for job in jobs)

    transport = FakeTransport(
        [
            (200, json.dumps({"id": "dict-phoneme", "version_id": "v1"})),
            (200, json.dumps({"id": "dict-alias", "version_id": "v2"})),
            (200, _wav_bytes(frequency=220.0)),
            (200, _wav_bytes(frequency=240.0)),
        ]
    )
    summary = run_jobs(jobs, _provider(transport), plan=plan, log=lambda _m: None)
    assert summary["completed"] == 2

    names = [json.loads(call["json"] and json.dumps(call["json"]))["name"] for call in transport.calls[:2]]
    assert names == ["hey-eva-corpus", "hey-eva-corpus-alias"]
    alias_rules = transport.calls[1]["json"]["rules"]
    assert [rule["type"] for rule in alias_rules] == ["alias", "alias"]
    assert alias_rules[0]["alias"] == "Eevah"
    for call in transport.calls[2:]:
        assert call["json"]["pronunciation_dictionary_locators"] == [
            {"pronunciation_dictionary_id": "dict-alias", "version_id": "v2"}
        ]
    index = [
        json.loads(line)
        for line in (plan.run_root / "index.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert {row["forced"] for row in index} == {"alias"}


def test_a_second_keyword_loads_from_its_own_plan(tmp_path):
    path = tmp_path / "google.yaml"
    path.write_text(
        f"run_root: {tmp_path / 'run'}\n"
        "keyword: hey google\n"
        "words:\n"
        '  hey: {phones: [HH, EY1], ipa: "hˈeɪ"}\n'
        '  google: {phones: [G, UW1, G, AH0, L], ipa: "ɡˈuːɡəl", say_as: "Googal"}\n'
        "phoneme_words: [google]\n"
        "phrases:\n"
        '  - {text: "Hey Google", label: 1}\n'
        '  - {text: "Google", label: 0}\n'
        "providers:\n"
        "  elevenlabs:\n"
        "    dictionary_alphabet: ipa\n"
        "    voices:\n"
        "      - {voice_id: v1}\n",
        encoding="utf-8",
    )
    plan = load_plan(path)
    assert plan.keyword_phonemes == "HH EY1 G UW1 G AH0 L"
    assert plan.positives()[0].phonemes == plan.keyword_phonemes
    assert plan.phrases[1].use_alias


def test_project_plan_words_carry_ipa_for_every_rule():
    plan = load_plan(PROJECT_PLAN)
    assert plan.providers["elevenlabs"]["dictionary_alphabet"] == "ipa"
    rules = dictionary_rules(plan)
    assert rules, "the project plan must produce rules"
    for rule in rules:
        assert rule["alphabet"] == "ipa"
        assert rule["phoneme"], rule
    eva = next(rule for rule in rules if rule["string_to_replace"] == "Eva")
    assert eva["phoneme"] == "ˈiːvə"


def test_ipa_alphabet_without_ipa_spelling_is_rejected(tmp_path):
    plan_text = _write_plan(tmp_path).read_text(encoding="utf-8").replace(
        "dictionary_alphabet: cmu_arpabet", "dictionary_alphabet: ipa"
    )
    path = tmp_path / "no_ipa.yaml"
    path.write_text(plan_text, encoding="utf-8")
    with pytest.raises(ValueError, match="carry no ipa"):
        load_plan(path)


def test_fully_rewritten_phrase_is_rejected(tmp_path):
    path = tmp_path / "covered.yaml"
    path.write_text(
        "run_root: /tmp/ignored\n"
        "keyword: hey eva\n"
        "words:\n"
        "  hey: [HH, EY1]\n"
        "  eva: [IY1, V, AH0]\n"
        "phoneme_words: [hey, eva]\n"
        "phrases:\n"
        '  - {text: "Hey Eva", label: 1}\n'
        "providers:\n"
        "  elevenlabs:\n"
        "    voices:\n"
        "      - {voice_id: a}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="rewritten end to end"):
        load_plan(path)


def test_prune_keys_removes_audio_and_rows(tmp_path):
    plan = load_plan(_write_plan(tmp_path, voices=1))
    _cache_dictionary(plan.run_root, plan)
    transport = FakeTransport([(200, _wav_bytes())])
    summary = run_jobs(
        build_jobs(plan)[:1], _provider(transport), plan=plan, log=lambda _m: None
    )
    row = json.loads(
        (plan.run_root / "index.jsonl").read_text(encoding="utf-8").strip().splitlines()[0]
    )
    audio = plan.run_root / row["audio_path"]
    assert audio.is_file()
    assert prune_keys(plan.run_root, {row["key"]}) == 1
    assert not audio.is_file()
    assert (plan.run_root / "index.jsonl").read_text(encoding="utf-8").strip() == ""
    assert summary["completed"] == 1


def test_load_excluded_takes_rejects_and_the_latest_human_verdict(tmp_path):
    root = tmp_path / "run"
    root.mkdir()
    (root / "rejected.jsonl").write_text(
        json.dumps({"key": "k-verifier", "verdict": "fail"}) + "\n", encoding="utf-8"
    )
    (root / "reviews.jsonl").write_text(
        json.dumps({"key": "k-changed-mind", "verdict": "bad"}) + "\n"
        + json.dumps({"key": "k-changed-mind", "verdict": "ok"}) + "\n"
        + json.dumps({"key": "k-bad", "verdict": "bad"}) + "\n",
        encoding="utf-8",
    )
    assert load_excluded(root) == {"k-verifier", "k-bad"}


def test_run_jobs_keeps_excluded_keys_out_of_the_source_csv(tmp_path):
    plan = load_plan(_write_plan(tmp_path, voices=2))
    _cache_dictionary(plan.run_root, plan)
    jobs = build_jobs(plan)
    transport = FakeTransport(
        [
            (200, _wav_bytes(frequency=220.0)),
            (200, _wav_bytes(frequency=240.0)),
            (200, _wav_bytes(frequency=260.0)),
            (200, _wav_bytes(frequency=280.0)),
        ]
    )
    summary = run_jobs(
        jobs,
        _provider(transport),
        plan=plan,
        exclude={jobs[0].key},
        log=lambda _message: None,
    )
    assert summary["completed"] == 4
    assert summary["excluded"] == 1
    with Path(summary["source_csv"]).open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 3
    assert jobs[0].filename not in {row["audio_path"] for row in rows}


def test_provider_rejects_unknown_output_format():
    with pytest.raises(ValueError, match="output_format"):
        ElevenLabsProvider(api_key="k", output_format="mp3_44100_128")


def test_source_csv_feeds_prepare_tts_lora_manifest(tmp_path):
    plan = load_plan(_write_plan(tmp_path, voices=2))
    jobs = build_jobs(plan)
    transport = FakeTransport(
        [
            (200, json.dumps({"id": "dict-1", "version_id": "ver-1"})),
            (200, _wav_bytes(frequency=220.0)),
            (200, _wav_bytes(frequency=240.0)),
            (200, _wav_bytes(frequency=260.0)),
            (200, _wav_bytes(frequency=280.0)),
        ]
    )
    summary = run_jobs(jobs, _provider(transport), plan=plan, log=lambda _message: None)
    assert summary["completed"] == 4

    source = Path(summary["source_csv"])
    with source.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 4
    assert {row["text_variant_phonemes"] for row in rows} == {
        "HH EY1 IY1 V AH0",
        "HH EY1 EY1 V AH0",
    }
    assert all(row["keyword_phonemes"] == KEYWORD_PHONEMES for row in rows)

    manifest = tmp_path / "tts_manifest.csv"
    result = build_tts_lora_manifest(
        source,
        manifest,
        audio_root=plan.run_root,
        keyword="hey eva",
        eval_fraction=0.5,
        seed=2025,
        verify_audio=True,
    )
    assert result["output_rows"] == 4
    assert result["unique_speakers"] == 2
    assert result["speaker_overlap"] == []
    assert set(result["splits"]) == {"train", "eval"}
    with manifest.open(encoding="utf-8-sig", newline="") as handle:
        written = list(csv.DictReader(handle))
    assert {row["tts_provider"] for row in written} == {"elevenlabs"}
