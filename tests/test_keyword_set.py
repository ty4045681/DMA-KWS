from __future__ import annotations

import copy

import pytest
from omegaconf import OmegaConf

from dma_kws.config import compose_config, config_to_dict
from dma_kws.inference.keyword_set import (
    KeywordEvalConfigError,
    KeywordSetScoreError,
    SCORE_SEMANTICS,
    aggregate_query_scores,
    canonical_json_sha256,
    file_sha256,
    keyword_eval_mode,
    keyword_eval_provenance_block,
    normalize_keyword_text,
    reject_unsupported_any_mode,
    require_positive_int,
    require_qbyt_threshold,
    resolve_keyword_set,
    semantic_keyword_eval,
)
from dma_kws.tokenizer import load_char_tokenizer


DICT_PATH = "data/dict/lang_char.txt"
HEY_EVA_A = "HH EY1 IY1 V AH0"
HEY_EVA_B = "HH EY1 EY1 V AH0"
OK_LAMP = "OW1 K EY1 L AE1 M P"


class FakeG2P:
    def __init__(self, mapping: dict[str, list[str]]) -> None:
        self.mapping = mapping

    def __call__(self, text: str) -> list[str]:
        if text not in self.mapping:
            raise AssertionError(f"unexpected G2P text {text!r}")
        return list(self.mapping[text])


def _tokenizer():
    return load_char_tokenizer(DICT_PATH, split_with_space=" ")


def _g2p():
    return FakeG2P(
        {
            "hey eva": HEY_EVA_A.split(),
            "ok lamp": OK_LAMP.split(),
            "hello": ["HH", "AH0", "L", "OW1"],
        }
    )


def _enroll(prep, *, g2p=None, tokenizer=None):
    return resolve_keyword_set(
        prep,
        tokenizer or _tokenizer(),
        g2p=g2p if g2p is not None else _g2p(),
        tokenizer_dict_path=DICT_PATH,
    )


def _any_prep(targets, **kwargs):
    section = {
        "mode": "any",
        "targets": targets,
        "query_batch_size": kwargs.pop("query_batch_size", 64),
    }
    prep = {"keyword": "", "keyword_phonemes": "", "keywords": [], "keyword_eval": section}
    prep.update(kwargs)
    return prep


def test_normalize_keyword_text_trims_collapses_and_casefolds():
    assert normalize_keyword_text("  Hey   EVA  ") == "hey eva"
    assert normalize_keyword_text("hey, eva") == "hey, eva"


def test_normalize_keyword_text_rejects_empty_and_non_string():
    with pytest.raises(KeywordEvalConfigError, match="non-empty string"):
        normalize_keyword_text("   ")
    with pytest.raises(KeywordEvalConfigError, match="non-empty string"):
        normalize_keyword_text(None)


def test_require_positive_int_rejects_bool_and_non_positive():
    assert require_positive_int(64, field="query_batch_size") == 64
    with pytest.raises(KeywordEvalConfigError, match="bool"):
        require_positive_int(True, field="query_batch_size")
    with pytest.raises(KeywordEvalConfigError):
        require_positive_int(0, field="query_batch_size")


def test_require_qbyt_threshold_rejects_bool_and_out_of_range():
    assert require_qbyt_threshold(0.5) == 0.5
    assert require_qbyt_threshold(0) == 0.0
    assert require_qbyt_threshold(1) == 1.0
    with pytest.raises(KeywordEvalConfigError, match="bool"):
        require_qbyt_threshold(True)
    with pytest.raises(KeywordEvalConfigError):
        require_qbyt_threshold(1.5)
    with pytest.raises(KeywordEvalConfigError):
        require_qbyt_threshold(float("nan"))


def test_per_row_default_returns_none_and_rejects_targets():
    assert _enroll({"keyword_eval": {"mode": "per_row", "targets": []}}) is None
    assert keyword_eval_mode({}) == "per_row"
    with pytest.raises(KeywordEvalConfigError, match="must be empty"):
        _enroll(
            {
                "keyword_eval": {
                    "mode": "per_row",
                    "targets": [{"text": "hey eva", "pronunciations": [HEY_EVA_A]}],
                }
            }
        )


def test_any_mode_requires_non_empty_targets():
    with pytest.raises(KeywordEvalConfigError, match="non-empty targets"):
        _enroll(_any_prep([]))


def test_explicit_pronunciations_do_not_append_g2p():
    keyword_set = _enroll(
        _any_prep(
            [{"text": "hey eva", "pronunciations": [HEY_EVA_B]}],
        )
    )
    assert keyword_set is not None
    assert len(keyword_set.queries) == 1
    assert list(keyword_set.keywords[0].pronunciations[0].phonemes) == HEY_EVA_B.split()
    assert keyword_set.keywords[0].pronunciations[0].source == "explicit"


def test_omitted_and_null_pronunciations_use_one_g2p_sequence():
    omitted = _enroll(_any_prep([{"text": "hey eva"}]))
    null = _enroll(_any_prep([{"text": "hey eva", "pronunciations": None}]))
    assert omitted is not None and null is not None
    assert omitted.keyword_set_id == null.keyword_set_id
    assert list(omitted.keywords[0].pronunciations[0].phonemes) == HEY_EVA_A.split()
    assert omitted.keywords[0].pronunciations[0].source == "g2p"


def test_explicit_empty_list_errors_with_target_index():
    with pytest.raises(KeywordEvalConfigError, match=r"targets\[0\].*empty list"):
        _enroll(_any_prep([{"text": "hey eva", "pronunciations": []}]))


def test_empty_string_pronunciation_errors_with_indexes():
    with pytest.raises(
        KeywordEvalConfigError,
        match=r"targets\[0\].*pronunciations\[1\]",
    ):
        _enroll(
            _any_prep(
                [{"text": "hey eva", "pronunciations": [HEY_EVA_A, ""]}]
            )
        )


def test_illegal_arpabet_errors_without_unk_fallback():
    with pytest.raises(KeywordEvalConfigError, match="unsupported phonemes"):
        _enroll(
            _any_prep(
                [{"text": "hey eva", "pronunciations": ["HH EY1 NOTAPHONE"]}]
            )
        )


def test_unpublished_phonemes_alias_points_at_pronunciations():
    with pytest.raises(KeywordEvalConfigError, match="pronunciations"):
        _enroll(
            _any_prep(
                [{"text": "hey eva", "phonemes": [HEY_EVA_A]}]
            )
        )


def test_unknown_target_key_errors():
    with pytest.raises(KeywordEvalConfigError, match="unsupported keys"):
        _enroll(_any_prep([{"text": "hey eva", "phones": [HEY_EVA_A]}]))


def test_any_mode_rejects_legacy_keyword_fields():
    with pytest.raises(KeywordEvalConfigError, match="prep.keyword"):
        _enroll(_any_prep([{"text": "hey eva"}], keyword="hey eva"))
    with pytest.raises(KeywordEvalConfigError, match="keyword_phonemes"):
        _enroll(_any_prep([{"text": "hey eva"}], keyword_phonemes=HEY_EVA_A))
    with pytest.raises(KeywordEvalConfigError, match="prep.keywords"):
        _enroll(_any_prep([{"text": "hey eva"}], keywords=["hey eva"]))


def test_same_text_merges_pronunciation_union_and_keeps_stress_distinct():
    keyword_set = _enroll(
        _any_prep(
            [
                {"text": "Hey EVA", "pronunciations": [HEY_EVA_A]},
                {"text": "hey eva", "pronunciations": [HEY_EVA_B, HEY_EVA_A]},
            ]
        )
    )
    assert keyword_set is not None
    assert keyword_set.texts == ("hey eva",)
    phonemes = [tuple(item.phonemes) for item in keyword_set.keywords[0].pronunciations]
    assert set(phonemes) == {tuple(HEY_EVA_A.split()), tuple(HEY_EVA_B.split())}
    assert any("merged 2 target entries" in item for item in keyword_set.merge_diagnostics)
    stress_a = _enroll(_any_prep([{"text": "x", "pronunciations": ["AH0"]}]))
    stress_b = _enroll(_any_prep([{"text": "x", "pronunciations": ["AH1"]}]))
    assert stress_a is not None and stress_b is not None
    assert stress_a.queries[0].token_ids != stress_b.queries[0].token_ids


def test_cross_keyword_shared_query_keeps_both_identities():
    shared = "HH EY1"
    keyword_set = _enroll(
        _any_prep(
            [
                {"text": "alpha", "pronunciations": [shared]},
                {"text": "beta", "pronunciations": [shared]},
            ]
        )
    )
    assert keyword_set is not None
    assert keyword_set.num_queries == 1
    assert keyword_set.texts == ("alpha", "beta")
    assert keyword_set.keywords[0].keyword_id != keyword_set.keywords[1].keyword_id
    assert (
        keyword_set.keywords[0].pronunciations[0].query_index
        == keyword_set.keywords[1].pronunciations[0].query_index
    )


def test_keyword_set_id_ignores_order_display_source_and_batch_size():
    first = _enroll(
        _any_prep(
            [
                {"text": "ok lamp", "pronunciations": [OK_LAMP]},
                {"text": "hey eva", "pronunciations": [HEY_EVA_B, HEY_EVA_A]},
            ],
            query_batch_size=8,
        )
    )
    second = _enroll(
        _any_prep(
            [
                {"text": "HEY EVA", "pronunciations": [HEY_EVA_A, HEY_EVA_B, HEY_EVA_A]},
                {"text": "ok lamp"},
            ],
            query_batch_size=64,
        )
    )
    assert first is not None and second is not None
    assert first.keyword_set_id == second.keyword_set_id
    assert first.query_batch_size != second.query_batch_size


def test_adding_a_pronunciation_changes_keyword_set_id():
    one = _enroll(_any_prep([{"text": "hey eva", "pronunciations": [HEY_EVA_A]}]))
    two = _enroll(
        _any_prep(
            [{"text": "hey eva", "pronunciations": [HEY_EVA_A, HEY_EVA_B]}]
        )
    )
    assert one is not None and two is not None
    assert one.keyword_set_id != two.keyword_set_id


def test_ids_are_sha256_not_python_hash():
    keyword_set = _enroll(
        _any_prep([{"text": "hey eva", "pronunciations": [HEY_EVA_A]}])
    )
    assert keyword_set is not None
    assert len(keyword_set.keyword_set_id) == 64
    assert len(keyword_set.keywords[0].keyword_id) == 64
    assert len(keyword_set.keywords[0].pronunciations[0].pronunciation_id) == 64
    assert keyword_set.keyword_set_id == canonical_json_sha256(
        {
            "keyword_set_schema_version": 1,
            "text_normalization_version": 1,
            "lexicon": {
                "hey eva": [list(keyword_set.queries[0].token_ids)],
            },
            "tokenizer_dict_sha256": file_sha256(DICT_PATH),
        }
    )


def test_aggregation_two_pronunciations_takes_max_and_one_hit():
    keyword_set = _enroll(
        _any_prep(
            [{"text": "hey eva", "pronunciations": [HEY_EVA_A, HEY_EVA_B]}]
        )
    )
    assert keyword_set is not None
    ordered = sorted(
        keyword_set.keywords[0].pronunciations,
        key=lambda item: item.token_ids,
    )
    scores = [(0.0, 0.0)] * keyword_set.num_queries
    scores[ordered[0].query_index] = (0.1, 0.2)
    scores[ordered[1].query_index] = (2.0, 0.9)
    result = aggregate_query_scores(keyword_set, scores, threshold=0.5)
    assert result.qbyt_score == pytest.approx(0.9)
    assert result.detected is True
    assert result.matched_keywords == ("hey eva",)
    assert result.best_pronunciation_id == ordered[1].pronunciation_id
    assert result.qbyt_raw_logit == pytest.approx(2.0)


def test_aggregation_two_keywords_one_audio_hit():
    keyword_set = _enroll(
        _any_prep(
            [
                {"text": "hey eva", "pronunciations": [HEY_EVA_A, HEY_EVA_B]},
                {"text": "ok lamp", "pronunciations": [OK_LAMP]},
            ]
        )
    )
    assert keyword_set is not None
    scores = [(0.0, 0.0)] * keyword_set.num_queries
    by_text = {keyword.text: keyword for keyword in keyword_set.keywords}
    for pronunciation in by_text["hey eva"].pronunciations:
        if pronunciation.phonemes == tuple(HEY_EVA_A.split()):
            scores[pronunciation.query_index] = (0.1, 0.2)
        else:
            scores[pronunciation.query_index] = (0.2, 0.3)
    scores[by_text["ok lamp"].pronunciations[0].query_index] = (1.5, 0.8)
    result = aggregate_query_scores(keyword_set, scores, threshold=0.5)
    assert result.detected is True
    assert result.matched_keywords == ("ok lamp",)
    assert result.best_keyword == "ok lamp"


def test_aggregation_all_below_threshold_still_reports_best():
    keyword_set = _enroll(
        _any_prep(
            [
                {"text": "hey eva", "pronunciations": [HEY_EVA_A]},
                {"text": "ok lamp", "pronunciations": [OK_LAMP]},
            ]
        )
    )
    assert keyword_set is not None
    scores = [(0.0, 0.1)] * keyword_set.num_queries
    lamp = [k for k in keyword_set.keywords if k.text == "ok lamp"][0]
    scores[lamp.pronunciations[0].query_index] = (0.4, 0.4)
    result = aggregate_query_scores(keyword_set, scores, threshold=0.5)
    assert result.detected is False
    assert result.matched_keywords == ()
    assert result.best_keyword == "ok lamp"
    assert result.qbyt_score == pytest.approx(0.4)


def test_aggregation_score_equals_threshold_detects():
    keyword_set = _enroll(_any_prep([{"text": "hey eva", "pronunciations": [HEY_EVA_A]}]))
    assert keyword_set is not None
    result = aggregate_query_scores(keyword_set, [(0.0, 0.5)], threshold=0.5)
    assert result.detected is True


def test_aggregation_tie_breaks_by_text_then_tokens_not_input_order():
    keyword_set_ab = _enroll(
        _any_prep(
            [
                {"text": "zeta", "pronunciations": [HEY_EVA_A]},
                {"text": "alpha", "pronunciations": [HEY_EVA_A]},
            ]
        )
    )
    keyword_set_ba = _enroll(
        _any_prep(
            [
                {"text": "alpha", "pronunciations": [HEY_EVA_A]},
                {"text": "zeta", "pronunciations": [HEY_EVA_A]},
            ]
        )
    )
    assert keyword_set_ab is not None and keyword_set_ba is not None
    scores = [(1.0, 0.9)]
    first = aggregate_query_scores(keyword_set_ab, scores, threshold=0.5)
    second = aggregate_query_scores(keyword_set_ba, scores, threshold=0.5)
    assert first.best_keyword == second.best_keyword == "alpha"
    assert set(first.matched_keywords) == {"alpha", "zeta"}

    two = _enroll(
        _any_prep(
            [{"text": "hey eva", "pronunciations": [HEY_EVA_B, HEY_EVA_A]}]
        )
    )
    assert two is not None
    scores = [(1.0, 0.8)] * two.num_queries
    result = aggregate_query_scores(two, scores, threshold=0.5)
    expected = min(two.keywords[0].pronunciations, key=lambda item: item.token_ids)
    assert result.best_pronunciation_id == expected.pronunciation_id


def test_aggregation_rejects_non_finite_and_pair_count_mismatch():
    keyword_set = _enroll(
        _any_prep(
            [{"text": "hey eva", "pronunciations": [HEY_EVA_A, HEY_EVA_B]}]
        )
    )
    assert keyword_set is not None
    with pytest.raises(KeywordSetScoreError, match="not finite"):
        aggregate_query_scores(
            keyword_set,
            [(0.1, 0.2), (float("nan"), 0.3)],
            threshold=0.5,
            audio_id="clip.wav",
        )
    with pytest.raises(KeywordSetScoreError, match="expected=2"):
        aggregate_query_scores(keyword_set, [(0.1, 0.2)], threshold=0.5)


def test_skipped_aggregation_never_detects_at_threshold_zero():
    keyword_set = _enroll(_any_prep([{"text": "hey eva", "pronunciations": [HEY_EVA_A]}]))
    assert keyword_set is not None
    result = aggregate_query_scores(
        keyword_set,
        [(0.0, 0.0)],
        threshold=0.0,
        scored=False,
    )
    assert result.detected is False
    assert result.qbyt_score == 0.0
    assert result.qbyt_raw_logit is None
    assert result.best_keyword is None
    assert result.matched_keywords == ()


def test_hydra_compose_multi_wakeup_retains_prep_and_v41_readout():
    cfg = compose_config(
        "icefall_zipformer_stage2_eps_softmin_v41",
        overrides=["+keyword_eval=multi_wakeup"],
    )
    prep = OmegaConf.to_container(cfg.prep, resolve=True)
    assert isinstance(prep, dict)
    section = prep["keyword_eval"]
    assert section["mode"] == "any"
    assert section["query_batch_size"] == 64
    texts = [target["text"] for target in section["targets"]]
    assert texts == ["hey eva", "ok lamp"]
    hey = section["targets"][0]
    assert hey["pronunciations"] == [HEY_EVA_A, HEY_EVA_B]
    assert "pronunciations" not in section["targets"][1]
    assert prep["manifest"] == ""
    assert prep["keyword"] == ""
    config = config_to_dict(cfg)
    readout = config["stage2"]["qbyt_readout"]
    assert config["stage2"]["qbyt_readout_version"] == 4
    assert readout["mode"] == "eps_softmin"
    assert readout["sink_token"] is True
    assert readout["audio_position"] == "relative_bias"

    again = compose_config(
        "icefall_zipformer_stage2_eps_softmin_v41",
        overrides=["+keyword_eval=multi_wakeup"],
    )
    again_prep = OmegaConf.to_container(again.prep, resolve=True)
    assert again_prep["keyword_eval"]["mode"] == "any"
    assert [t["text"] for t in again_prep["keyword_eval"]["targets"]] == texts


def test_hydra_compose_hey_eva_variants_and_eval_condition():
    cfg = compose_config(
        "icefall_zipformer_stage2_eps_softmin_v41",
        overrides=["+keyword_eval=hey_eva_variants", "+eval_condition=clean"],
    )
    prep = OmegaConf.to_container(cfg.prep, resolve=True)
    assert prep["keyword_eval"]["mode"] == "any"
    assert len(prep["keyword_eval"]["targets"]) == 1
    assert prep["keyword_eval"]["targets"][0]["text"] == "hey eva"
    assert prep["musan_mix"]["noise"]["enabled"] is False


def test_enroll_overlay_targets_with_g2p_fallback_for_ok_lamp():
    cfg = compose_config(
        "icefall_zipformer_stage2_eps_softmin_v41",
        overrides=["+keyword_eval=multi_wakeup"],
    )
    prep = OmegaConf.to_container(cfg.prep, resolve=True)
    keyword_set = _enroll(prep)
    assert keyword_set is not None
    assert keyword_set.texts == ("hey eva", "ok lamp")
    hey = [item for item in keyword_set.keywords if item.text == "hey eva"][0]
    assert len(hey.pronunciations) == 2
    lamp = [item for item in keyword_set.keywords if item.text == "ok lamp"][0]
    assert lamp.pronunciations[0].source == "g2p"
    assert list(lamp.pronunciations[0].phonemes) == OK_LAMP.split()


def test_reject_unsupported_any_mode():
    reject_unsupported_any_mode({"keyword_eval": {"mode": "per_row"}}, entry="x")
    with pytest.raises(KeywordEvalConfigError, match="does not support"):
        reject_unsupported_any_mode(
            {"keyword_eval": {"mode": "any", "targets": [{"text": "a"}]}},
            entry="scripts/eval_two_stage_kws.py",
        )


def test_provenance_block_per_row_and_any():
    assert keyword_eval_provenance_block(None, mode="per_row") == {"mode": "per_row"}
    keyword_set = _enroll(
        _any_prep([{"text": "hey eva", "pronunciations": [HEY_EVA_B, HEY_EVA_A]}])
    )
    block = keyword_eval_provenance_block(keyword_set, mode="any")
    assert block["mode"] == "any"
    assert block["aggregation"] == SCORE_SEMANTICS
    assert block["keyword_set_id"] == keyword_set.keyword_set_id
    reversed_set = _enroll(
        _any_prep([{"text": "hey eva", "pronunciations": [HEY_EVA_A, HEY_EVA_B]}])
    )
    assert semantic_keyword_eval(block, schema_version=4) == semantic_keyword_eval(
        keyword_eval_provenance_block(reversed_set, mode="any"),
        schema_version=4,
    )
    assert semantic_keyword_eval(None, schema_version=3) == {"mode": "per_row"}
