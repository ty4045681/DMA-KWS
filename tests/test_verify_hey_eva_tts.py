import numpy as np

from scripts.verify_hey_eva_tts import (
    classify_vowel,
    merge_results,
    judge,
    last_nucleus,
    probe_vowel,
    resample_to_target,
    transcript_words,
    vowel_before,
    whisper_gate,
)


def test_whisper_gate_needs_two_words_for_a_two_word_phrase():
    assert whisper_gate("Hey Eva!", 2) == (True, "")
    passed, why = whisper_gate("He-e-e-e-e-e-e-e", 2)
    assert not passed and "fewer than two words" in why
    passed, _why = whisper_gate("Heiva!", 2)
    assert not passed
    assert whisper_gate("either.", 1) == (True, "")


def test_merge_results_replaces_only_verified_keys():
    previous = {"a": {"key": "a", "verdict": "fail"}, "b": {"key": "b", "verdict": "pass"}}
    merged = merge_results(previous, [{"key": "a", "verdict": "pass"}, {"key": "c", "verdict": "fail"}])
    by_key = {row["key"]: row["verdict"] for row in merged}
    assert by_key == {"a": "pass", "b": "pass", "c": "fail"}


def test_transcript_words_drops_one_letter_tokens():
    assert transcript_words("Hey Eva!") == ["Hey", "Eva"]
    assert transcript_words("He-e-e-e-e") == ["He"]
    assert transcript_words("") == []


def test_resample_to_target_keeps_the_duration():
    rate, seconds = 24000, 0.5
    timeline = np.linspace(0.0, seconds, int(rate * seconds), endpoint=False)
    tone = np.sin(2.0 * np.pi * 200.0 * timeline)
    converted = resample_to_target(tone, rate)
    assert converted.size == 8000
    unchanged = resample_to_target(tone, 16000)
    assert unchanged is tone


def test_probe_vowel_uses_the_keyword_difference():
    keyword = "HH EY1 IY1 V AH0".split()
    assert probe_vowel("HH EY1 EY1 V AH0".split(), keyword) == ("EY", "EY1")
    assert probe_vowel("HH EY1 AY1 V AH0 N".split(), keyword) == ("AY", "AY1")
    assert probe_vowel("HH EY1 IY1 V".split(), keyword) == ("IY", "IY1")
    google = "HH EY1 G UW1 G AH0 L".split()
    assert probe_vowel("HH EY1 G AA1 G AH0 L".split(), google) == ("AA", "AA1")
    assert probe_vowel("HH EY1 G UW1 G AH0 L".split(), google) == ("UW", "UW1")
    assert probe_vowel("G UW1 G AH0 L".split(), google) == ("UW", "UW1")


def test_probe_vowel_without_a_keyword_uses_the_last_stressed_vowel():
    assert probe_vowel("HH EY1 IY1 V AH0".split()) == ("IY", "IY1")
    assert probe_vowel("HH EY1 EY1 V AH0".split()) == ("EY", "EY1")
    assert probe_vowel("HH EY1 EH1 V AH0 N".split()) == ("EH", "EH1")
    assert probe_vowel("HH EY1 IY1 V".split()) == ("IY", "IY1")


def test_probe_vowel_without_a_keyword_falls_back_to_a_stressed_vowel():
    assert probe_vowel("HH EY1".split()) == ("EY", "EY1")
    # "Hey either" has no vowel after the first difference from "Hey Eva", so
    # the last stressed vowel carries the check instead of the final schwa-r.
    assert probe_vowel("HH EY1 IY1 DH ER0".split()) == ("IY", "IY1")


def test_vowel_before_stops_at_the_marker():
    assert vowel_before("haɪʔiːva") == "iː"
    assert vowel_before("haɪʔaɪva") == "aɪ"
    assert vowel_before("haɪeːva") == "aɪeː"
    assert vowel_before("heːʔiva") == "i"


def test_last_nucleus_splits_glued_vowels():
    assert last_nucleus("aɪeː") == "eː"
    assert last_nucleus("aɪiː") == "iː"
    assert last_nucleus("aɪ") == "aɪ"
    assert last_nucleus("i") == "i"
    assert last_nucleus("") == ""


def test_glued_vowels_still_classify_correctly():
    # no word gap in the decode: "Hey Eva" and "Hey Ava" must not be confused
    assert judge("HH EY1 IY1 V AH0", "haɪiːva")[0] == "pass"
    assert judge("HH EY1 EY1 V AH0", "haɪeɪva")[0] == "pass"
    assert judge("HH EY1 EY1 V AH0", "haɪiːva")[0] == "fail"


def test_classify_vowel_maps_the_families():
    assert classify_vowel("iː") == "IY"
    assert classify_vowel("i") == "IY"
    assert classify_vowel("ɪ") == "IY"
    assert classify_vowel("aɪ") == "OPEN"
    assert classify_vowel("eː") == "OPEN"
    assert classify_vowel("ɛ") == "EH"
    assert classify_vowel("ə") == "AH"
    assert classify_vowel("uː") == "UW"
    assert classify_vowel("ʉ") == "UW"
    assert classify_vowel("ø") == "UW"
    assert classify_vowel("ɑː") == "OPEN"
    assert classify_vowel("") == "unknown"


def test_ay_and_ey_both_accept_an_open_vowel():
    assert judge("HH EY1 AY1 V AH0 N", "haɪʔaɪvən")[0] == "pass"
    assert judge("HH EY1 EY1 V AH0", "haɪʔaɪva")[0] == "pass"
    assert judge("HH EY1 AY1 V AH0 N", "haɪʔiːvən")[0] == "fail"
    assert judge("HH EY1 AY1 V AH0 N", "hˈeɪ")[0] == "fail"


def test_eh_accepts_a_front_mid_vowel_but_not_iy():
    assert judge("HH EY1 EH1 V AH0 N", "haɪʔɛvən")[0] == "pass"
    assert judge("HH EY1 EH1 V AH0 N", "haɪʔevən")[0] == "pass"
    assert judge("HH EY1 EH1 V AH0 N", "haɪʔiːvən")[0] == "fail"


def test_eric_checks_the_last_vowel_without_a_v():
    assert judge("HH EY1 EH1 R IH0 K", "heɪˈɛɹɪk")[0] == "pass"
    assert judge("HH EY1 EH1 R IH0 K", "heɪˈɛɹɪ")[0] == "pass"


def test_judge_accepts_the_measured_kokoro_and_v3_decodes():
    for decoded in ("haɪʔiːva", "heːʔiva", "haɪʔiva"):
        verdict, reason, expected, found = judge("HH EY1 IY1 V AH0", decoded)
        assert verdict == "pass", (decoded, reason)
        assert expected == "IY1"
    for decoded in ("haɪʔaɪva", "haɪeːva"):
        verdict, reason, _expected, _found = judge("HH EY1 EY1 V AH0", decoded)
        assert verdict == "pass", (decoded, reason)


def test_judge_fails_when_ava_is_read_as_eva():
    verdict, reason, _expected, found = judge("HH EY1 EY1 V AH0", "haɪʔiːva")
    assert verdict == "fail"
    assert "IY" in reason and found == "iː"


def test_judge_fails_when_the_second_word_is_missing():
    verdict, reason, _expected, _found = judge("HH EY1 IY1 V AH0", "hˈeɪ")
    assert verdict == "fail"
    assert "a word is missing" in reason


def test_judge_fails_on_empty_decode():
    verdict, reason, _expected, _found = judge("HH EY1 IY1 V AH0", "")
    assert verdict == "fail"
    assert "no phonemes decoded" in reason


def test_negatives_report_a_probe_mismatch_as_review():
    strict, _reason, _expected, _found = judge("HH EY1 EH1 R IH0 K", "haɪʔiːɹɪk")
    assert strict == "fail"
    relaxed, reason, _expected, _found = judge(
        "HH EY1 EH1 R IH0 K", "haɪʔiːɹɪk", strict_probe=False
    )
    assert relaxed == "review"
    assert "probe vowel" in reason


def test_judge_survives_a_decode_without_vowels():
    # A full-length phrase with no vowels at all is a fail.
    assert judge("HH EY1 IY1 V AH0", "t5k5")[0] == "fail"
    # A truncated phrase has no vowel-count floor, so it lands in review.
    verdict, reason, _expected, _found = judge("HH EY1 IY1 V AH0", "t5k5", strict=False)
    assert verdict == "review"
    assert "no vowel nuclei" in reason


def test_judge_returns_review_for_an_unknown_vowel():
    verdict, reason, _expected, _found = judge("HH EY1 IY1 V AH0", "hˈeɪ ˈyːvə")
    assert verdict == "review"
    assert "outside the known classes" in reason
